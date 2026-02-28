"""
Sample OpenClaw Agent — Audio-Enabled

Agent implementation using OpenClaw's OpenAI-compatible HTTP gateway
(/v1/chat/completions) with **server-side** STT and TTS.

Audio pipeline
--------------
  Client mic → PCM16 → STT service (moshi-server)
    → transcribed text → OpenClaw SSE
    → sentence buffer → TTS service (Pocket TTS)
    → PCM16 → on_audio_delta → Avatar blendshapes

When STT / TTS services are unavailable the agent degrades gracefully
to text-only mode (client must handle speech and synthesis).
"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import orjson

from core.logger import get_logger
from core.settings import get_settings
from services.knowledge_service import KnowledgeService

from ..base_agent import BaseAgent, ConversationState
from .openclaw_settings import get_openclaw_settings

logger = get_logger(__name__)

# Regex: split after sentence-ending punctuation that is followed by whitespace,
# or at newlines.  The lookbehind keeps the punctuation with the left segment.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
# Clause-level split (comma, semicolon, colon, dash) — used for faster
# TTS first-chunk when no sentence boundary has appeared yet.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:\u2014])\s+")
_UNSUPPORTED_TTS_CHARS = re.compile(
    r"["
    r"\U0001F1E6-\U0001F1FF"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA70-\U0001FAFF"
    r"\u2600-\u26FF"
    r"\u2700-\u27BF"
    r"]+"
)


class SampleOpenClawAgent(BaseAgent):
    """
    OpenClaw agent with optional server-side STT and TTS.

    Full-duplex audio loop when both services are available::

        mic audio → moshi-server ASR → text → OpenClaw LLM
                                                     ↓
        avatar ← on_audio_delta ← Pocket TTS ← sentences

    Falls back to text-only when services are missing.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._oc = get_openclaw_settings()
        self._http_client: httpx.AsyncClient | None = None
        self._connected = False
        self._state = ConversationState()

        # ── Event callbacks ─────────────────────────────────────────
        self._on_audio_delta: Callable[[bytes], Awaitable[None]] | None = None
        self._on_transcript_delta: Callable[[str, str, str | None, str | None], Awaitable[None]] | None = None
        self._on_response_start: Callable[[str], Awaitable[None]] | None = None
        self._on_response_end: Callable[[str, str | None], Awaitable[None]] | None = None
        self._on_user_transcript: Callable[[str, str], Awaitable[None]] | None = None
        self._on_interrupted: Callable[[], Awaitable[None]] | None = None
        self._on_error: Callable[[Any], Awaitable[None]] | None = None
        self._on_cancel_sync: Callable[[], None] | None = None

        # ── Conversation history ────────────────────────────────────
        self._messages: list[dict[str, str]] = []
        self._system_prompt: str = ""

        # ── Response / interruption state ───────────────────────────
        self._response_cancelled = False
        self._active_stream_task: asyncio.Task[None] | None = None
        self._tts_cancelled = asyncio.Event()

        # ── STT / TTS services (initialised in connect()) ──────────
        self._stt: Any = None  # STTService | None
        self._tts: Any = None  # TTSService | None
        self._stt_available = False
        self._tts_available = False

        # ── Audio-input pipeline ────────────────────────────────────
        self._audio_input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._audio_worker_task: asyncio.Task[None] | None = None
        self._user_transcript_buffer: str = ""
        self._pause_flushing = False
        self._bargein_speech_frames: int = 0

        # ── Per-turn latency metrics ────────────────────────────────
        self._metric_input_source: str = "unknown"
        self._metric_input_finalized_at: float | None = None
        self._metric_stt_first_audio_sent_at: float | None = None
        self._metric_stt_first_word_at: float | None = None
        self._metric_stt_finalized_at: float | None = None
        self._metric_oc_request_start_at: float | None = None
        self._metric_oc_first_token_at: float | None = None
        self._metric_oc_stream_done_at: float | None = None
        self._metric_tts_first_sentence_at: float | None = None
        self._metric_tts_first_audio_chunk_at: float | None = None
        self._metric_tts_done_at: float | None = None
        self._metric_tts_chunks_emitted: int = 0
        self._metric_tts_bytes_emitted: int = 0

    def _append_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = max(4, int(self._oc.history_max_messages))
        if len(self._messages) <= max_messages:
            return

        first_is_system = bool(self._messages) and self._messages[0].get("role") == "system"
        if first_is_system:
            tail_count = max_messages - 1
            self._messages = [self._messages[0], *self._messages[-tail_count:]]
            return

        self._messages = self._messages[-max_messages:]

    def _reset_turn_metrics(self, input_source: str) -> None:
        self._metric_input_source = input_source
        self._metric_input_finalized_at = None
        self._metric_stt_first_audio_sent_at = None
        self._metric_stt_first_word_at = None
        self._metric_stt_finalized_at = None
        self._metric_oc_request_start_at = None
        self._metric_oc_first_token_at = None
        self._metric_oc_stream_done_at = None
        self._metric_tts_first_sentence_at = None
        self._metric_tts_first_audio_chunk_at = None
        self._metric_tts_done_at = None
        self._metric_tts_chunks_emitted = 0
        self._metric_tts_bytes_emitted = 0

    @staticmethod
    def _ms_value(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000, 1)

    @classmethod
    def _ms(cls, start: float | None, end: float | None) -> str:
        value = cls._ms_value(start, end)
        if value is None:
            return "n/a"
        return f"{value:.1f}"

    def _log_turn_metrics(self, turn_metrics: dict[str, Any] | None = None) -> None:
        if turn_metrics is None:
            turn_metrics = {
                "source": self._metric_input_source,
                "input_finalized_at": self._metric_input_finalized_at,
                "stt_first_audio_sent_at": self._metric_stt_first_audio_sent_at,
                "stt_first_word_at": self._metric_stt_first_word_at,
                "stt_finalized_at": self._metric_stt_finalized_at,
                "oc_request_start_at": self._metric_oc_request_start_at,
                "oc_first_token_at": self._metric_oc_first_token_at,
                "oc_stream_done_at": self._metric_oc_stream_done_at,
                "tts_first_sentence_at": self._metric_tts_first_sentence_at,
                "tts_first_audio_chunk_at": self._metric_tts_first_audio_chunk_at,
                "tts_done_at": self._metric_tts_done_at,
                "tts_chunks": self._metric_tts_chunks_emitted,
                "tts_bytes": self._metric_tts_bytes_emitted,
            }

        metrics_record = {
            "ts": int(time.time() * 1000),
            "source": turn_metrics.get("source"),
            "stt_first_word_ms": self._ms_value(
                turn_metrics.get("stt_first_audio_sent_at"), turn_metrics.get("stt_first_word_at")
            ),
            "stt_finalize_ms": self._ms_value(
                turn_metrics.get("stt_first_word_at"), turn_metrics.get("stt_finalized_at")
            ),
            "stt_total_ms": self._ms_value(
                turn_metrics.get("stt_first_audio_sent_at"), turn_metrics.get("stt_finalized_at")
            ),
            "oc_first_token_ms": self._ms_value(
                turn_metrics.get("oc_request_start_at"), turn_metrics.get("oc_first_token_at")
            ),
            "oc_stream_ms": self._ms_value(
                turn_metrics.get("oc_request_start_at"), turn_metrics.get("oc_stream_done_at")
            ),
            "tts_first_audio_ms": self._ms_value(
                turn_metrics.get("tts_first_sentence_at"), turn_metrics.get("tts_first_audio_chunk_at")
            ),
            "tts_total_ms": self._ms_value(turn_metrics.get("tts_first_sentence_at"), turn_metrics.get("tts_done_at")),
            "tts_chunks": turn_metrics.get("tts_chunks", 0),
            "tts_bytes": turn_metrics.get("tts_bytes", 0),
            "e2e_input_final_to_first_audio_ms": self._ms_value(
                turn_metrics.get("input_finalized_at"), turn_metrics.get("tts_first_audio_chunk_at")
            ),
        }

        logger.info(
            f"TURN_METRIC source={turn_metrics.get('source')} "
            f"stt_first_word_ms={self._ms(turn_metrics.get('stt_first_audio_sent_at'), turn_metrics.get('stt_first_word_at'))} "
            f"stt_finalize_ms={self._ms(turn_metrics.get('stt_first_word_at'), turn_metrics.get('stt_finalized_at'))} "
            f"stt_total_ms={self._ms(turn_metrics.get('stt_first_audio_sent_at'), turn_metrics.get('stt_finalized_at'))}"
        )
        logger.info(
            f"TURN_METRIC oc_first_token_ms={self._ms(turn_metrics.get('oc_request_start_at'), turn_metrics.get('oc_first_token_at'))} "
            f"oc_stream_ms={self._ms(turn_metrics.get('oc_request_start_at'), turn_metrics.get('oc_stream_done_at'))}"
        )
        logger.info(
            f"TURN_METRIC tts_first_audio_ms={self._ms(turn_metrics.get('tts_first_sentence_at'), turn_metrics.get('tts_first_audio_chunk_at'))} "
            f"tts_total_ms={self._ms(turn_metrics.get('tts_first_sentence_at'), turn_metrics.get('tts_done_at'))} "
            f"tts_chunks={turn_metrics.get('tts_chunks', 0)} tts_bytes={turn_metrics.get('tts_bytes', 0)}"
        )
        logger.info(
            f"TURN_METRIC e2e_input_final_to_first_audio_ms={self._ms(turn_metrics.get('input_finalized_at'), turn_metrics.get('tts_first_audio_chunk_at'))}"
        )

        try:
            metrics_path = Path(__file__).resolve().parents[3] / "logs" / "pipeline_metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(metrics_record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug(f"Failed to write metrics file: {exc}")

    # ================================================================
    # Properties
    # ================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def transcript_speed(self) -> float:
        return self._oc.transcript_speed

    @property
    def input_sample_rate(self) -> int:
        return 24000

    @property
    def output_sample_rate(self) -> int:
        return 24000

    # ================================================================
    # Event handlers
    # ================================================================

    def set_event_handlers(
        self,
        on_audio_delta: Callable[[bytes], Awaitable[None]] | None = None,
        on_transcript_delta: (Callable[[str, str, str | None, str | None], Awaitable[None]] | None) = None,
        on_response_start: Callable[[str], Awaitable[None]] | None = None,
        on_response_end: (Callable[[str, str | None], Awaitable[None]] | None) = None,
        on_user_transcript: (Callable[[str, str], Awaitable[None]] | None) = None,
        on_interrupted: Callable[[], Awaitable[None]] | None = None,
        on_error: Callable[[Any], Awaitable[None]] | None = None,
        on_cancel_sync: Callable[[], None] | None = None,
    ) -> None:
        self._on_audio_delta = on_audio_delta
        self._on_transcript_delta = on_transcript_delta
        self._on_response_start = on_response_start
        self._on_response_end = on_response_end
        self._on_user_transcript = on_user_transcript
        self._on_interrupted = on_interrupted
        self._on_error = on_error
        self._on_cancel_sync = on_cancel_sync

    # ================================================================
    # Lifecycle
    # ================================================================

    async def connect(self) -> None:  # noqa: C901 (setup complexity)
        if self._connected:
            return

        oc = self._oc
        logger.info("Connecting OpenClaw agent")

        # 1. Validate token
        if not oc.auth_token:
            raise ValueError(
                "AUTH_TOKEN is required — set it in .env or match gateway.auth.token in openclaw.json"
            )

        # 2. HTTP client
        headers: dict[str, str] = {
            "Authorization": f"Bearer {oc.auth_token}",
            "Content-Type": "application/json",
        }
        if oc.agent_id:
            headers["x-openclaw-agent-id"] = oc.agent_id
        if oc.session_key:
            headers["x-openclaw-session-key"] = oc.session_key

        self._http_client = httpx.AsyncClient(
            base_url=oc.base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=oc.connect_timeout,
                read=oc.read_timeout,
                write=10.0,
                pool=5.0,
            ),
            transport=httpx.AsyncHTTPTransport(retries=max(0, oc.max_retries)),
        )

        # 3. Knowledge / system prompt
        knowledge = await KnowledgeService.load_knowledge_base(self._settings.knowledge_base_source)
        self._system_prompt = KnowledgeService.format_instructions(self._settings.assistant_instructions, knowledge)
        thinking_mode = (oc.thinking_mode or "default").strip().lower()
        if thinking_mode in {"off", "minimal"}:
            guidance = (
                "\n\nResponse style: prioritize low-latency concise answers. "
                "Avoid long deliberation and keep reasoning brief."
            )
            self._system_prompt = f"{self._system_prompt}{guidance}"

        self._messages = [{"role": "system", "content": self._system_prompt}]

        # 4. Connectivity probe (non-fatal)
        try:
            probe = await self._http_client.get("/", timeout=3.0)
            logger.info(f"OpenClaw reachable at {oc.base_url} (status={probe.status_code})")
        except Exception as exc:
            logger.warning(f"OpenClaw probe: {exc}")

        # 5. STT service
        if oc.stt_enabled:
            try:
                from services.stt_service import STTService

                self._stt = STTService(
                    on_word=self._on_stt_word,
                    on_error=self._on_error,
                    stt_model=oc.stt_model,
                    vad_start_threshold=oc.stt_vad_start_threshold,
                    vad_end_threshold=oc.stt_vad_end_threshold,
                    vad_min_silence_ms=oc.stt_vad_min_silence_ms,
                    initial_prompt=oc.stt_initial_prompt,
                )
                await self._stt.connect()
                self._stt_available = True
                logger.info("STT service connected (backend=onnx)")
            except Exception as exc:
                logger.warning(f"STT unavailable — text-input only: {exc}")
                self._stt = None
                self._stt_available = False

        # 6. TTS service
        if oc.tts_enabled:
            try:
                from services.tts_service import TTSService

                self._tts = TTSService(
                    model_dir=oc.tts_onnx_model_dir,
                    voice_path=oc.tts_voice_path,
                    voice_name=oc.tts_voice_name,
                    noise_scale=oc.tts_noise_scale,
                    noise_w_scale=oc.tts_noise_w_scale,
                    length_scale=oc.tts_length_scale,
                )
                await self._tts.load()
                self._tts_available = True
                logger.info("TTS service loaded (backend=onnx)")
            except Exception as exc:
                logger.warning(f"TTS unavailable — text-only output: {exc}")
                self._tts = None
                self._tts_available = False

        # 7. Audio-input worker (needs STT)
        if self._stt_available:
            self._audio_worker_task = asyncio.create_task(self._audio_input_worker())

        self._connected = True
        logger.info(
            f"OpenClaw agent ready (stt={self._stt_available}, tts={self._tts_available}, model={oc.agent_model})"
        )

    async def disconnect(self) -> None:
        # Cancel active tasks
        for task in (self._active_stream_task, self._audio_worker_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._stt:
            await self._stt.disconnect()

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._connected = False
        self._messages = []
        self._state = ConversationState()
        logger.info("OpenClaw agent disconnected")

    # ================================================================
    # Text input
    # ================================================================

    def send_text_message(self, text: str) -> None:
        """
        Send a text message to OpenClaw.

        Used for direct text input (typed messages, or client-side STT).
        """
        if not self._connected or not self._http_client:
            logger.warning("Not connected — dropping message")
            return

        if self._state.is_responding:
            self.cancel_response()

        self._reset_turn_metrics("text")
        self._metric_input_finalized_at = time.perf_counter()

        self._append_message("user", text)

        if self._on_user_transcript:
            asyncio.create_task(self._on_user_transcript(text, "user"))

        self._response_cancelled = False
        self._tts_cancelled.clear()
        self._active_stream_task = asyncio.create_task(self._stream_and_speak())

    # ================================================================
    # Audio input
    # ================================================================

    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Buffer incoming PCM16 audio for the STT service.

        No-op when STT is unavailable (text-only mode).
        """
        if not self._stt_available:
            return
        try:
            self._audio_input_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.warning(f"Audio queue full, dropping {len(audio_bytes)} bytes")

    # ================================================================
    # Interruption
    # ================================================================

    def cancel_response(self) -> None:
        if not self._state.is_responding:
            return

        logger.debug("Cancelling OpenClaw response")
        self._response_cancelled = True
        self._tts_cancelled.set()
        self._state.is_responding = False
        self._state.audio_done = True

        if self._active_stream_task and not self._active_stream_task.done():
            self._active_stream_task.cancel()

        # Notify chat_session synchronously so _handle_response_end()
        # aborts immediately (stops audio playback).
        if self._on_cancel_sync:
            self._on_cancel_sync()

        # Clear STT state immediately so stale _has_speech doesn't
        # trigger an instant re-barge-in on the next response.
        if self._stt:
            self._stt.reset_turn(reset_vad_state=True)
        self._user_transcript_buffer = ""
        self._bargein_speech_frames = 0

    # ================================================================
    # Audio-input worker  (STT → pause → respond)
    # ================================================================

    async def _audio_input_worker(self) -> None:
        """
        Background loop: drain audio queue → STT → detect pause → respond.
        """
        assert self._stt is not None
        chunks_sent = 0
        idle_ticks = 0  # consecutive idle timeouts
        idle_timeout_sec = 0.15
        IDLE_PAUSE_TICKS = 3  # 3 × 0.15s = 0.45s of silence → treat as pause
        try:
            while self._connected:
                try:
                    audio_bytes = await asyncio.wait_for(self._audio_input_queue.get(), timeout=idle_timeout_sec)
                except asyncio.TimeoutError:
                    idle_ticks += 1
                    # Check for pause during idle: if user spoke then
                    # audio stopped, treat sustained silence as a pause.
                    has_speech = self._stt.has_speech
                    transcript = self._user_transcript_buffer.strip()
                    if has_speech and transcript and not self._pause_flushing and not self._state.is_responding:
                        logger.debug(
                            f"Audio worker: idle tick {idle_ticks}/{IDLE_PAUSE_TICKS} "
                            f"(speech={has_speech}, buf={transcript!r})"
                        )
                        if idle_ticks >= IDLE_PAUSE_TICKS:
                            logger.info(
                                f"Idle-pause detected after {idle_ticks} ticks, flushing transcript: {transcript!r}"
                            )
                            await self._handle_user_pause()
                            idle_ticks = 0
                    continue

                # Got a chunk — reset idle counter
                idle_ticks = 0
                chunks_sent += 1
                if chunks_sent % 50 == 1:
                    logger.debug(
                        f"Audio worker: chunk #{chunks_sent} ({len(audio_bytes)} bytes, queue_size={self._audio_input_queue.qsize()})"
                    )

                # Barge-in: keep feeding audio to VAD so the user
                # can interrupt mid-response (like OpenAI/Gemini).
                # Require 4 consecutive speech frames (~128ms) to avoid
                # single-frame noise triggers.
                if self._state.is_responding:
                    await self._stt.send_audio(audio_bytes)
                    if self._stt.has_speech:
                        self._bargein_speech_frames += 1
                        if self._bargein_speech_frames >= 4:
                            logger.info(
                                f"Barge-in: sustained speech ({self._bargein_speech_frames} frames), cancelling"
                            )
                            self.cancel_response()
                    else:
                        self._bargein_speech_frames = 0
                    continue

                if self._metric_stt_first_audio_sent_at is None:
                    self._metric_input_source = "stt"
                    self._metric_stt_first_audio_sent_at = time.perf_counter()

                await self._stt.send_audio(audio_bytes)

                # Check for end-of-turn pause (EMA-based)
                if not self._pause_flushing and self._stt.check_pause():
                    pause_score = self._stt.pause_score
                    logger.info(f"Pause detected (score={pause_score:.3f}), flushing...")
                    await self._handle_user_pause()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Audio-input worker error: {exc}", exc_info=True)

    async def _handle_user_pause(self) -> None:
        """User stopped speaking — flush STT, collect transcript, respond."""
        self._pause_flushing = True
        try:
            if self._metric_stt_first_audio_sent_at is None:
                self._reset_turn_metrics("stt")

            # Push silence to flush any buffered audio through the pipeline
            await self._stt.flush_silence()

            # Brief wait for remaining word events to arrive
            await asyncio.sleep(0.05)

            transcript = self._user_transcript_buffer.strip()
            self._user_transcript_buffer = ""
            self._stt.reset_turn()

            if not transcript:
                return

            self._metric_stt_finalized_at = time.perf_counter()
            self._metric_input_finalized_at = self._metric_stt_finalized_at

            logger.info(f"User said: {transcript!r}")

            self._append_message("user", transcript)

            if self._on_user_transcript:
                await self._on_user_transcript(transcript, "user")

            # Ensure any cancelled task from a barge-in has finished
            if self._active_stream_task and not self._active_stream_task.done():
                try:
                    await asyncio.wait_for(self._active_stream_task, timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

            # Generate response
            self._response_cancelled = False
            self._tts_cancelled.clear()
            self._active_stream_task = asyncio.create_task(self._stream_and_speak())
        finally:
            self._pause_flushing = False

    async def _on_stt_word(self, text: str, _start_time: float) -> None:
        """Callback from STT — accumulate transcribed words."""
        self._metric_input_source = "stt"
        if self._metric_stt_first_audio_sent_at is None:
            self._reset_turn_metrics("stt")

        if self._metric_stt_first_word_at is None:
            self._metric_stt_first_word_at = time.perf_counter()

        # Add space between words (moshi sends individual words)
        if self._user_transcript_buffer and not self._user_transcript_buffer.endswith(" "):
            self._user_transcript_buffer += " "
        self._user_transcript_buffer += text
        logger.debug(f"STT word: {text!r}  (buf: {self._user_transcript_buffer!r})")

    # ================================================================
    # SSE streaming + TTS
    # ================================================================

    async def _stream_and_speak(self) -> None:
        """
        Send conversation to OpenClaw SSE endpoint, buffer tokens into
        sentences, and synthesize each sentence via TTS.

        If TTS is unavailable only transcript deltas are fired.
        """
        if not self._http_client:
            return

        oc = self._oc
        session_id = f"session_{int(time.time() * 1000)}"
        item_id = f"openclaw_{int(time.time() * 1000)}"

        payload: dict[str, Any] = {
            "model": oc.agent_model,
            "messages": list(self._messages),
            "stream": True,
        }
        if oc.user_id:
            payload["user"] = oc.user_id

        # ── Signal response start ───────────────────────────────────
        self._state.session_id = session_id
        self._state.is_responding = True
        self._state.transcript_buffer = ""
        self._state.audio_done = False
        logger.info(f"Starting response stream for session {session_id}")

        if self._on_response_start:
            await self._on_response_start(session_id)

        full_response = ""
        sentence_buffer = ""

        # TTS sentence queue (None = sentinel for "done")
        tts_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        tts_task: asyncio.Task[None] | None = None

        turn_metrics: dict[str, Any] = {
            "source": self._metric_input_source,
            "input_finalized_at": self._metric_input_finalized_at,
            "stt_first_audio_sent_at": self._metric_stt_first_audio_sent_at,
            "stt_first_word_at": self._metric_stt_first_word_at,
            "stt_finalized_at": self._metric_stt_finalized_at,
            "oc_request_start_at": None,
            "oc_first_token_at": None,
            "oc_stream_done_at": None,
            "tts_first_sentence_at": None,
            "tts_first_audio_chunk_at": None,
            "tts_done_at": None,
            "tts_chunks": 0,
            "tts_bytes": 0,
        }

        if self._tts_available:
            tts_task = asyncio.create_task(self._tts_worker(tts_queue, turn_metrics, item_id))

        try:
            turn_metrics["oc_request_start_at"] = time.perf_counter()
            async with self._http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = f"OpenClaw {response.status_code}: {body.decode('utf-8', errors='replace')}"
                    logger.error(error_msg)
                    if self._on_error:
                        await self._on_error({"error": error_msg})
                    return

                # ── Parse SSE stream ────────────────────────────────
                async for line in response.aiter_lines():
                    if self._response_cancelled:
                        break

                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = orjson.loads(data)
                    except Exception:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content")
                    if not content or self._response_cancelled:
                        continue

                    if turn_metrics["oc_first_token_at"] is None:
                        turn_metrics["oc_first_token_at"] = time.perf_counter()

                    full_response += content
                    self._state.transcript_buffer = full_response

                    # Stream transcript deltas to the client as tokens arrive
                    # (same as OpenAI/Gemini — word-level for real-time subtitles).
                    if self._on_transcript_delta and not self._response_cancelled:
                        await self._on_transcript_delta(content, "assistant", item_id, None)

                    # Sentence buffering → TTS
                    if self._tts_available:
                        sentence_buffer += content
                        sentences, sentence_buffer = self._extract_sentences(sentence_buffer)
                        for sent in sentences:
                            sanitized = self._sanitize_for_tts(sent)
                            if not sanitized:
                                continue
                            if turn_metrics["tts_first_sentence_at"] is None:
                                turn_metrics["tts_first_sentence_at"] = time.perf_counter()
                            await tts_queue.put((sent, sanitized))

            # ── SSE done — flush remaining sentence to TTS ──────────
            if self._tts_available:
                if not self._response_cancelled:
                    remaining = sentence_buffer.strip()
                    if remaining:
                        sanitized = self._sanitize_for_tts(remaining)
                        if sanitized:
                            if turn_metrics["tts_first_sentence_at"] is None:
                                turn_metrics["tts_first_sentence_at"] = time.perf_counter()
                            await tts_queue.put((remaining, sanitized))
                    await tts_queue.put(None)  # signal TTS worker to stop
                    # Wait for ALL TTS audio to finish before signalling end
                    if tts_task:
                        await tts_task
                else:
                    # Barge-in: cancel TTS immediately — don't block on
                    # the Piper thread finishing the current sentence.
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                        try:
                            await tts_task
                        except (asyncio.CancelledError, Exception):
                            pass

        except asyncio.CancelledError:
            logger.debug("SSE stream cancelled")
        except httpx.ConnectError as exc:
            error_msg = f"Cannot reach OpenClaw at {oc.base_url}: {exc}. Is OpenClaw running?"
            logger.error(error_msg)
            if self._on_error:
                await self._on_error({"error": error_msg})
        except httpx.ReadTimeout:
            logger.warning("OpenClaw SSE timed out")
            if self._on_error:
                await self._on_error({"error": "OpenClaw response timed out"})
        except Exception as exc:
            logger.error(f"OpenClaw stream error: {exc}", exc_info=True)
            if self._on_error:
                await self._on_error({"error": str(exc)})
        finally:
            turn_metrics["oc_stream_done_at"] = time.perf_counter()

            # Ensure TTS task terminates even on error
            if tts_task and not tts_task.done():
                if self._response_cancelled:
                    # Barge-in: cancel TTS immediately — don't block on
                    # the Piper thread finishing the current sentence.
                    tts_task.cancel()
                else:
                    try:
                        await tts_queue.put(None)
                    except Exception:
                        pass
                try:
                    await tts_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Add assistant message to conversation history
            if full_response and not self._response_cancelled:
                self._append_message("assistant", full_response)

            # Keep is_responding=True during audio playback so the
            # audio worker enters the barge-in path while the client
            # is still hearing audio.  cancel_response() sets it False
            # immediately if barge-in fires.
            if not self._response_cancelled and self._on_response_end:
                await self._on_response_end(full_response, item_id)
                # Barge-in may have fired while waiting for audio drain
                if self._response_cancelled and self._on_interrupted:
                    await self._on_interrupted()
            elif self._response_cancelled and self._on_interrupted:
                await self._on_interrupted()

            self._state.is_responding = False
            self._state.audio_done = True

            self._log_turn_metrics(turn_metrics)

            # Prepare STT timing markers for the next user turn.
            # (Without this reset, STT metrics can accumulate across turns.)
            self._metric_stt_first_audio_sent_at = None
            self._metric_stt_first_word_at = None
            self._metric_stt_finalized_at = None
            self._metric_input_finalized_at = None
            self._metric_input_source = "stt"

            self._active_stream_task = None

    # ================================================================
    # TTS worker
    # ================================================================

    async def _tts_worker(
        self,
        queue: asyncio.Queue[tuple[str, str] | None],
        turn_metrics: dict[str, Any],
        item_id: str,
    ) -> None:
        """Synthesize queued sentences via Piper TTS one at a time.

        Each sentence is synthesised individually with a 200ms breath pause
        between them for natural pacing.  The TTS service yields ~100ms
        delivery chunks so audio streams progressively to the frontend.
        """
        assert self._tts is not None
        sentence_count = 0
        # 200ms silence at output sample rate — natural breath pause between sentences
        pause_samples = int(self._tts.sample_rate * 0.20)
        silence_pad = bytes(pause_samples * 2)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if self._response_cancelled:
                    break

                _original, sanitized = item

                # Breath pause between sentences (not before the first)
                if sentence_count > 0 and self._on_audio_delta and not self._response_cancelled:
                    await self._on_audio_delta(silence_pad)

                logger.debug(f"TTS synthesizing: {sanitized!r}")
                async for chunk in self._tts.synthesize(sanitized, cancelled=self._tts_cancelled):
                    if self._response_cancelled:
                        break
                    if turn_metrics["tts_first_audio_chunk_at"] is None:
                        turn_metrics["tts_first_audio_chunk_at"] = time.perf_counter()
                    turn_metrics["tts_chunks"] += 1
                    turn_metrics["tts_bytes"] += len(chunk)
                    if self._on_audio_delta:
                        await self._on_audio_delta(chunk)

                sentence_count += 1

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"TTS worker error: {exc}", exc_info=True)
        finally:
            turn_metrics["tts_done_at"] = time.perf_counter()

    # ================================================================
    # Sentence extraction
    # ================================================================

    def _sanitize_for_tts(self, text: str) -> str:
        cleaned = _UNSUPPORTED_TTS_CHARS.sub("", text)
        return " ".join(cleaned.split())

    def _extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        """
        Split *buffer* at sentence boundaries (punctuation + whitespace).

        Returns ``(complete_sentences, remaining_buffer)``.
        Falls back to clause boundaries for faster first-chunk TTS,
        then forces a split when buffer exceeds ``tts_sentence_max_chars``.
        """
        parts = _SENTENCE_BOUNDARY.split(buffer)

        if len(parts) <= 1:
            # No sentence boundary — try clause boundary for faster first chunk
            if len(buffer) >= 60:
                clause_parts = _CLAUSE_BOUNDARY.split(buffer)
                if len(clause_parts) > 1:
                    sentences = [p.strip() for p in clause_parts[:-1] if p.strip()]
                    return sentences, clause_parts[-1]
            if len(buffer) > self._oc.tts_sentence_max_chars:
                idx = buffer.rfind(" ", 0, self._oc.tts_sentence_max_chars)
                if idx > 0:
                    return [buffer[:idx].strip()], buffer[idx + 1 :]
            return [], buffer

        sentences = [p.strip() for p in parts[:-1] if p.strip()]
        remainder = parts[-1]
        return sentences, remainder
