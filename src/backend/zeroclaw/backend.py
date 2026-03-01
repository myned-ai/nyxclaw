"""
Sample ZeroClaw Agent — Audio-Enabled

Implements ZeroClaw gateway chat via WebSocket (`/ws/chat`) while reusing
the same local STT/TTS flow used by the OpenClaw sample agent.
"""

import asyncio
import json
import random
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import websockets

from core.logger import get_logger
from core.settings import get_settings
from ..base_agent import BaseAgent, ConversationState
from .settings import get_zeroclaw_settings

logger = get_logger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
# Clause-level split (comma, semicolon, colon, dash) — used for faster
# TTS first-chunk when no sentence boundary has appeared yet.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:\u2014])\s+")
_TOOL_FILLERS = [
    "On it.",
    "One sec.",
    "Working on it.",
    "Let me handle that.",
    "Give me a moment.",
    "Hang on.",
    "Let me take care of that.",
]

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


class ZeroClawBackend(BaseAgent):
    """
    ZeroClaw agent with local STT/TTS.

    Transport difference vs OpenClaw:
    - OpenClaw sample uses HTTP SSE (`/v1/chat/completions`)
    - ZeroClaw uses WebSocket (`/ws/chat`)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._zc = get_zeroclaw_settings()
        self._connected = False
        self._state = ConversationState()

        self._on_audio_delta: Callable[[bytes], Awaitable[None]] | None = None
        self._on_transcript_delta: Callable[[str, str, str | None, str | None], Awaitable[None]] | None = None
        self._on_response_start: Callable[[str], Awaitable[None]] | None = None
        self._on_response_end: Callable[[str, str | None], Awaitable[None]] | None = None
        self._on_user_transcript: Callable[[str, str], Awaitable[None]] | None = None
        self._on_interrupted: Callable[[], Awaitable[None]] | None = None
        self._on_error: Callable[[Any], Awaitable[None]] | None = None
        self._on_cancel_sync: Callable[[], None] | None = None

        self._messages: list[dict[str, str]] = []
        self._system_prompt: str = ""

        self._response_cancelled = False
        self._active_stream_task: asyncio.Task[None] | None = None
        self._tts_cancelled = asyncio.Event()
        self._cancel_event = asyncio.Event()

        self._stt: Any = None
        self._tts: Any = None
        self._stt_available = False
        self._tts_available = False

        self._audio_input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._audio_worker_task: asyncio.Task[None] | None = None
        self._user_transcript_buffer: str = ""
        self._pause_flushing = False
        self._bargein_speech_frames: int = 0

        # Persistent WebSocket to ZeroClaw — kept open across messages so
        # the server-side WsSession (browser, conversation history, tools)
        # survives between turns.
        self._zc_ws: websockets.WebSocketClientProtocol | None = None
        self._zc_ws_clean: bool = True

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
        max_messages = max(4, int(self._zc.history_max_messages))
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
            "oc_first_token_to_first_sentence_ms": self._ms_value(
                turn_metrics.get("oc_first_token_at"), turn_metrics.get("tts_first_sentence_at")
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
            f"oc_first_token_to_first_sentence_ms={self._ms(turn_metrics.get('oc_first_token_at'), turn_metrics.get('tts_first_sentence_at'))} "
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

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def transcript_speed(self) -> float:
        return self._zc.transcript_speed

    @property
    def input_sample_rate(self) -> int:
        return 24000

    @property
    def output_sample_rate(self) -> int:
        return 24000

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

    def append_audio(self, audio_bytes: bytes) -> None:
        if not self._stt_available:
            return
        try:
            self._audio_input_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.warning(f"Audio queue full, dropping {len(audio_bytes)} bytes")

    async def _audio_input_worker(self) -> None:
        assert self._stt is not None
        chunks_sent = 0
        idle_ticks = 0
        idle_timeout_sec = 0.15
        idle_pause_ticks = 3
        try:
            while self._connected:
                try:
                    audio_bytes = await asyncio.wait_for(self._audio_input_queue.get(), timeout=idle_timeout_sec)
                except asyncio.TimeoutError:
                    idle_ticks += 1
                    has_speech = self._stt.has_speech
                    transcript = self._user_transcript_buffer.strip()
                    if has_speech and transcript and not self._pause_flushing and not self._state.is_responding:
                        logger.debug(
                            f"Audio worker: idle tick {idle_ticks}/{idle_pause_ticks} "
                            f"(speech={has_speech}, buf={transcript!r})"
                        )
                        if idle_ticks >= idle_pause_ticks:
                            logger.info(
                                f"Idle-pause detected after {idle_ticks} ticks, flushing transcript: {transcript!r}"
                            )
                            await self._handle_user_pause()
                            idle_ticks = 0
                    continue

                idle_ticks = 0
                chunks_sent += 1
                if chunks_sent % 50 == 1:
                    logger.debug(
                        f"Audio worker: chunk #{chunks_sent} ({len(audio_bytes)} bytes, "
                        f"queue_size={self._audio_input_queue.qsize()})"
                    )

                if self._state.is_responding:
                    # Barge-in: keep feeding audio to VAD so the user
                    # can interrupt mid-response (like OpenAI/Gemini).
                    # Require 4 consecutive speech frames (~128ms) to avoid
                    # single-frame noise triggers.
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

                if not self._pause_flushing and self._stt.check_pause():
                    pause_score = self._stt.pause_score
                    logger.info(f"Pause detected (score={pause_score:.3f}), flushing...")
                    await self._handle_user_pause()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Audio-input worker error: {exc}", exc_info=True)

    async def _handle_user_pause(self) -> None:
        self._pause_flushing = True
        try:
            if self._metric_stt_first_audio_sent_at is None:
                self._reset_turn_metrics("stt")

            await self._stt.flush_silence()
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

            # Wait for the in-flight stream task to drain the WS gracefully.
            # The task exits on its own once _cancel_event is set + drain finishes.
            if self._active_stream_task and not self._active_stream_task.done():
                try:
                    await asyncio.wait_for(self._active_stream_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Stream drain timed out, force-cancelling")
                    self._active_stream_task.cancel()
                    try:
                        await self._active_stream_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._zc_ws_clean = False
                except (asyncio.CancelledError, Exception):
                    pass

            self._response_cancelled = False
            self._tts_cancelled.clear()
            self._cancel_event.clear()
            self._active_stream_task = asyncio.create_task(self._stream_and_speak())
        finally:
            self._pause_flushing = False

    async def _on_stt_word(self, text: str, _start_time: float) -> None:
        self._metric_input_source = "stt"
        if self._metric_stt_first_audio_sent_at is None:
            self._reset_turn_metrics("stt")

        if self._metric_stt_first_word_at is None:
            self._metric_stt_first_word_at = time.perf_counter()

        if self._user_transcript_buffer and not self._user_transcript_buffer.endswith(" "):
            self._user_transcript_buffer += " "
        self._user_transcript_buffer += text
        logger.debug(f"STT word: {text!r}  (buf: {self._user_transcript_buffer!r})")

    async def _tts_worker(
        self,
        queue: asyncio.Queue[tuple[str, str] | None],
        turn_metrics: dict[str, Any],
        item_id: str,
    ) -> None:
        """Synthesize queued sentences via streaming Piper TTS."""
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

                original, sanitized = item

                # Breath pause between sentences (not before the first)
                if sentence_count > 0 and self._on_audio_delta and not self._response_cancelled:
                    await self._on_audio_delta(silence_pad)

                # Emit transcript delta in sync with audio (after silence pad,
                # before this sentence's audio starts).
                if self._on_transcript_delta and not self._response_cancelled:
                    await self._on_transcript_delta(original, "assistant", item_id, None)

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

    def _sanitize_for_tts(self, text: str) -> str:
        cleaned = _UNSUPPORTED_TTS_CHARS.sub("", text)
        return " ".join(cleaned.split())

    def _extract_sentences(self, buffer: str, first_chunk: bool = False) -> tuple[list[str], str]:
        parts = _SENTENCE_BOUNDARY.split(buffer)

        if len(parts) <= 1:
            # No sentence boundary — try clause boundary for faster first chunk.
            # Use a lower threshold for the very first chunk to reduce latency.
            clause_threshold = 40 if first_chunk else 60
            if len(buffer) >= clause_threshold:
                clause_parts = _CLAUSE_BOUNDARY.split(buffer)
                if len(clause_parts) > 1:
                    sentences = [p.strip() for p in clause_parts[:-1] if p.strip()]
                    return sentences, clause_parts[-1]
            if len(buffer) > self._zc.tts_sentence_max_chars:
                idx = buffer.rfind(" ", 0, self._zc.tts_sentence_max_chars)
                if idx > 0:
                    return [buffer[:idx].strip()], buffer[idx + 1 :]
            return [], buffer

        sentences = [p.strip() for p in parts[:-1] if p.strip()]
        remainder = parts[-1]
        return sentences, remainder

    def _build_ws_chat_url(self) -> str:
        """Build ZeroClaw chat websocket URL including optional token."""
        zc = self._zc
        base = zc.base_url.rstrip("/")
        parsed = urlparse(base)

        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError(f"Unsupported ZeroClaw base URL scheme: {parsed.scheme}")

        ws_scheme = "wss" if scheme in {"https", "wss"} else "ws"
        path = f"{parsed.path.rstrip('/')}/ws/chat" if parsed.path else "/ws/chat"
        query = parsed.query

        if zc.auth_token:
            token_query = urlencode({"token": zc.auth_token})
            query = f"{query}&{token_query}" if query else token_query

        return urlunparse((ws_scheme, parsed.netloc, path, "", query, ""))

    async def connect(self) -> None:
        if self._connected:
            return

        zc = self._zc
        logger.info("Connecting ZeroClaw agent")

        self._system_prompt = self._settings.assistant_instructions

        thinking_mode = (zc.thinking_mode or "default").strip().lower()
        if thinking_mode in {"off", "minimal"}:
            guidance = (
                "\n\nResponse style: prioritize low-latency concise answers. "
                "Avoid long deliberation and keep reasoning brief."
            )
            self._system_prompt = f"{self._system_prompt}{guidance}"

        self._messages = [{"role": "system", "content": self._system_prompt}]

        try:
            ws_url = self._build_ws_chat_url()
            self._zc_ws = await websockets.connect(
                ws_url,
                open_timeout=zc.connect_timeout,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
            )
            self._zc_ws_clean = True
            logger.info(f"ZeroClaw WebSocket connected at {zc.base_url}")
        except Exception as exc:
            logger.warning(f"ZeroClaw connect failed (will retry per message): {exc}")
            self._zc_ws = None

        if zc.stt_enabled:
            try:
                from services.stt_service import STTService

                self._stt = STTService(
                    on_word=self._on_stt_word,
                    on_error=self._on_error,
                    stt_model=zc.stt_model,
                    vad_start_threshold=zc.stt_vad_start_threshold,
                    vad_end_threshold=zc.stt_vad_end_threshold,
                    vad_min_silence_ms=zc.stt_vad_min_silence_ms,
                    initial_prompt=zc.stt_initial_prompt,
                )
                await self._stt.connect()
                self._stt_available = True
                logger.info("STT service connected (backend=onnx)")
            except Exception as exc:
                logger.warning(f"STT unavailable — text-input only: {exc}")
                self._stt = None
                self._stt_available = False

        if zc.tts_enabled:
            try:
                from services.tts_service import TTSService

                self._tts = TTSService(
                    model_dir=zc.tts_onnx_model_dir,
                    voice_path=zc.tts_voice_path,
                    voice_name=zc.tts_voice_name,
                    noise_scale=zc.tts_noise_scale,
                    noise_w_scale=zc.tts_noise_w_scale,
                    length_scale=zc.tts_length_scale,
                )
                await self._tts.load()
                self._tts_available = True
                logger.info("TTS service loaded (backend=onnx)")
            except Exception as exc:
                logger.warning(f"TTS unavailable — text-only output: {exc}")
                self._tts = None
                self._tts_available = False

        if self._stt_available:
            self._audio_worker_task = asyncio.create_task(self._audio_input_worker())

        self._connected = True
        logger.info(
            f"ZeroClaw agent ready (stt={self._stt_available}, tts={self._tts_available}, model={zc.agent_model})"
        )

    async def _ensure_zc_ws(self) -> websockets.WebSocketClientProtocol:
        """Ensure persistent WebSocket to ZeroClaw is open, reconnecting if needed."""
        if self._zc_ws is not None and self._zc_ws.close_code is None and self._zc_ws_clean:
            return self._zc_ws

        # Close stale connection if it exists
        if self._zc_ws is not None:
            try:
                await self._zc_ws.close()
            except Exception:
                pass
            self._zc_ws = None

        ws_url = self._build_ws_chat_url()
        self._zc_ws = await websockets.connect(
            ws_url,
            open_timeout=self._zc.connect_timeout,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=20,
        )
        self._zc_ws_clean = True
        logger.info("ZeroClaw WebSocket reconnected")
        return self._zc_ws

    async def _drain_ws_until_done(self, ws: websockets.WebSocketClientProtocol, timeout: float = 3.0) -> None:
        """Read and discard remaining ZeroClaw messages until 'done' or timeout.

        Called after a barge-in cancellation so the persistent WebSocket is in a
        clean state (no stale chunks) before the next turn's message is sent.
        """
        deadline = time.monotonic() + timeout
        drained = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug(f"WS drain: timed out after discarding {drained} msgs")
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 0.5))
                drained += 1
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "done":
                            logger.debug(f"WS drain: received 'done' after {drained} msgs")
                            return
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                logger.debug(f"WS drain: idle after {drained} msgs")
                break
            except Exception as exc:
                logger.debug(f"WS drain error after {drained} msgs: {exc}")
                break

    async def _stream_and_speak(self) -> None:
        """
        Send the last user turn to ZeroClaw WebSocket chat endpoint and
        synthesize streamed text with local TTS.
        """
        zc = self._zc
        session_id = f"session_{int(time.time() * 1000)}"
        item_id = f"zeroclaw_{int(time.time() * 1000)}"

        user_content = ""
        for message in reversed(self._messages):
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break

        if not user_content:
            logger.warning("No user message found for ZeroClaw stream")
            return

        self._state.session_id = session_id
        self._state.is_responding = True
        self._state.transcript_buffer = ""
        self._state.audio_done = False

        if self._on_response_start:
            await self._on_response_start(session_id)

        full_response = ""
        sentence_buffer = ""
        spoke_filler = False
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
            websocket = await self._ensure_zc_ws()

            await websocket.send(json.dumps({"type": "message", "content": user_content}))

            while True:
                if self._response_cancelled:
                    break

                # Wait for either a WS message or a barge-in cancel signal.
                # Using asyncio.wait instead of plain recv() so we can break
                # immediately on cancel without task.cancel() (which would
                # dirty the WS and force a reconnect).
                recv_fut = asyncio.create_task(websocket.recv())
                cancel_fut = asyncio.create_task(self._cancel_event.wait())
                try:
                    done_futs, _ = await asyncio.wait(
                        {recv_fut, cancel_fut},
                        timeout=zc.read_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for fut in (recv_fut, cancel_fut):
                        if not fut.done():
                            fut.cancel()
                            try:
                                await fut
                            except (asyncio.CancelledError, Exception):
                                pass

                if not done_futs:
                    raise asyncio.TimeoutError("ZeroClaw read timeout")

                if cancel_fut in done_futs:
                    break

                raw_message = recv_fut.result()

                if isinstance(raw_message, bytes):
                    continue

                try:
                    message = json.loads(raw_message)
                except Exception:
                    continue

                message_type = message.get("type")
                if message_type == "error":
                    error_msg = message.get("message", "Unknown ZeroClaw error")
                    if self._on_error:
                        await self._on_error({"error": error_msg})
                    break

                if message_type == "tool_call":
                    if not spoke_filler and not full_response and not self._response_cancelled:
                        spoke_filler = True
                        tool_name = message.get("name", "unknown")
                        filler = random.choice(_TOOL_FILLERS)
                        logger.info(f"Tool call: {tool_name} — filler: {filler!r}")
                        if self._tts_available:
                            sanitized = self._sanitize_for_tts(filler)
                            if sanitized:
                                if turn_metrics["tts_first_sentence_at"] is None:
                                    turn_metrics["tts_first_sentence_at"] = time.perf_counter()
                                await tts_queue.put((filler, sanitized))
                        elif self._on_transcript_delta:
                            await self._on_transcript_delta(filler, "assistant", item_id, None)
                    continue

                if message_type == "chunk":
                    content = str(message.get("content", ""))
                elif message_type == "done":
                    done_text = str(message.get("full_response", ""))
                    if done_text and not full_response:
                        content = done_text
                    else:
                        content = ""
                else:
                    continue

                if content and not self._response_cancelled:
                    if turn_metrics["oc_first_token_at"] is None:
                        turn_metrics["oc_first_token_at"] = time.perf_counter()

                    full_response += content
                    self._state.transcript_buffer = full_response

                    # When TTS is active, transcript deltas are emitted by the
                    # TTS worker in sync with audio.  Only emit from LLM stream
                    # when TTS is unavailable (text-only mode).
                    if not self._tts_available and self._on_transcript_delta and not self._response_cancelled:
                        await self._on_transcript_delta(content, "assistant", item_id, None)

                    if self._tts_available:
                        sentence_buffer += content
                        is_first = turn_metrics["tts_first_sentence_at"] is None
                        sentences, sentence_buffer = self._extract_sentences(sentence_buffer, first_chunk=is_first)
                        for sent in sentences:
                            sanitized = self._sanitize_for_tts(sent)
                            if not sanitized:
                                continue
                            if turn_metrics["tts_first_sentence_at"] is None:
                                turn_metrics["tts_first_sentence_at"] = time.perf_counter()
                            await tts_queue.put((sent, sanitized))

                if message_type == "done":
                    break

            # After barge-in, tell ZeroClaw to abort the LLM call and
            # discard the interrupted turn from server-side history, then
            # drain remaining messages so the WS is clean for the next turn.
            if self._response_cancelled and websocket.close_code is None:
                try:
                    await websocket.send(json.dumps({"type": "cancel"}))
                    logger.debug("Sent cancel message to ZeroClaw")
                except Exception as exc:
                    logger.debug(f"Failed to send cancel to ZeroClaw: {exc}")
                    self._zc_ws_clean = False
                try:
                    # Short drain — server already got cancel, 0.5s is enough.
                    # If it doesn't send "done" in time, mark dirty & reconnect.
                    await self._drain_ws_until_done(websocket, timeout=0.5)
                except Exception as exc:
                    logger.debug(f"WS drain error, marking dirty: {exc}")
                    self._zc_ws_clean = False

            if self._tts_available:
                if not self._response_cancelled:
                    remaining = sentence_buffer.strip()
                    if remaining:
                        sanitized = self._sanitize_for_tts(remaining)
                        if sanitized:
                            if turn_metrics["tts_first_sentence_at"] is None:
                                turn_metrics["tts_first_sentence_at"] = time.perf_counter()
                            await tts_queue.put((remaining, sanitized))
                    await tts_queue.put(None)
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
            # Only reached via disconnect() force-cancel, not barge-in
            self._zc_ws_clean = False
            logger.debug("ZeroClaw stream force-cancelled (disconnect)")
        except asyncio.TimeoutError:
            self._zc_ws_clean = False
            logger.warning("ZeroClaw websocket read timed out")
            if self._on_error:
                await self._on_error({"error": "ZeroClaw response timed out"})
        except websockets.exceptions.ConnectionClosed as exc:
            self._zc_ws = None
            if not self._response_cancelled and self._on_error:
                await self._on_error({"error": f"ZeroClaw websocket closed: {exc}"})
        except Exception as exc:
            self._zc_ws_clean = False
            logger.error(f"ZeroClaw stream error: {exc}", exc_info=True)
            if self._on_error:
                await self._on_error({"error": str(exc)})
        finally:
            turn_metrics["oc_stream_done_at"] = time.perf_counter()

            if tts_task and not tts_task.done():
                if self._response_cancelled:
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

            self._metric_stt_first_audio_sent_at = None
            self._metric_stt_first_word_at = None
            self._metric_stt_finalized_at = None
            self._metric_input_finalized_at = None
            self._metric_input_source = "stt"
            self._active_stream_task = None

    async def disconnect(self) -> None:
        for task in (self._active_stream_task, self._audio_worker_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._zc_ws:
            try:
                await self._zc_ws.close()
            except Exception:
                pass
            self._zc_ws = None

        if self._stt:
            await self._stt.disconnect()

        self._connected = False
        self._messages = []
        self._state = ConversationState()
        logger.info("ZeroClaw agent disconnected")

    def send_text_message(self, text: str) -> None:
        if not self._connected:
            logger.warning("Not connected — dropping message")
            return

        if self._state.is_responding:
            self.cancel_response()

        self._reset_turn_metrics("text")
        self._metric_input_finalized_at = time.perf_counter()

        self._append_message("user", text)

        if self._on_user_transcript:
            asyncio.create_task(self._on_user_transcript(text, "user"))

        # Schedule via async helper so we wait for the old stream to drain
        asyncio.create_task(self._start_new_stream())

    async def _start_new_stream(self) -> None:
        """Wait for any in-flight stream to finish draining, then start a new one."""
        if self._active_stream_task and not self._active_stream_task.done():
            try:
                await asyncio.wait_for(self._active_stream_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._active_stream_task.cancel()
                try:
                    await self._active_stream_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._zc_ws_clean = False
            except (asyncio.CancelledError, Exception):
                pass

        self._response_cancelled = False
        self._tts_cancelled.clear()
        self._cancel_event.clear()
        self._active_stream_task = asyncio.create_task(self._stream_and_speak())

    def cancel_response(self) -> None:
        if not self._state.is_responding:
            return

        logger.debug("Cancelling ZeroClaw response")
        self._response_cancelled = True
        self._tts_cancelled.set()
        self._cancel_event.set()
        self._state.is_responding = False
        self._state.audio_done = True
        # Don't cancel the task — let it exit gracefully and drain
        # the WS so the persistent connection survives barge-in.

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
