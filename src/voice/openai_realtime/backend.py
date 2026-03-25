"""
OpenAI Voice Backend

Uses OpenAI Realtime API for server-side VAD + STT, sends transcript to
the configured LLM backend, and synthesizes speech via OpenAI TTS API.

LLM routing by agent_type:
  - openclaw → HTTP SSE at /v1/chat/completions
  - zeroclaw → WebSocket at /ws/avatar

Pipeline: Client audio → OpenAI Realtime (VAD+STT) → transcript
          → OpenClaw SSE / ZeroClaw WS (LLM) → sentence buffer
          → OpenAI TTS API → PCM16 audio → wav2arkit → Client
"""

import asyncio
import base64
import json
import random
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
import orjson
import websockets

from backend.base_agent import TOOL_FILLERS, BaseAgent, ConversationState
from core.logger import get_logger
from core.settings import get_settings

from .settings import get_openai_realtime_settings

logger = get_logger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
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


class OpenAIRealtimeBackend(BaseAgent):
    """
    OpenAI voice layer over an OpenClaw/ZeroClaw LLM backend.

    - VAD + STT: OpenAI Realtime API (server-side, no local models)
    - LLM: OpenClaw via HTTP SSE, or ZeroClaw via WebSocket
    - TTS: OpenAI TTS API (streaming PCM16 @ 24kHz)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._rt = get_openai_realtime_settings()
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
        self._cancel_event = asyncio.Event()

        # ── OpenAI clients ──────────────────────────────────────────
        self._openai: Any = None  # AsyncOpenAI
        self._realtime_conn: Any = None  # AsyncRealtimeConnection
        self._realtime_task: asyncio.Task[None] | None = None
        self._session_ready = asyncio.Event()

        # ── HTTP client for OpenClaw SSE ────────────────────────────
        self._http_client: httpx.AsyncClient | None = None

        # ── WebSocket for ZeroClaw ──────────────────────────────────
        self._zc_ws: websockets.WebSocketClientProtocol | None = None
        self._zc_ws_clean: bool = True

    # ================================================================
    # History helpers
    # ================================================================

    def _append_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = max(4, int(self._rt.history_max_messages))
        if len(self._messages) <= max_messages:
            return
        first_is_system = bool(self._messages) and self._messages[0].get("role") == "system"
        if first_is_system:
            tail_count = max_messages - 1
            self._messages = [self._messages[0], *self._messages[-tail_count:]]
            return
        self._messages = self._messages[-max_messages:]

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
        return self._rt.transcript_speed

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
        on_tool_call: (Callable[[str, dict, str | None], Awaitable[None]] | None) = None,
    ) -> None:
        self._on_audio_delta = on_audio_delta
        self._on_transcript_delta = on_transcript_delta
        self._on_response_start = on_response_start
        self._on_response_end = on_response_end
        self._on_user_transcript = on_user_transcript
        self._on_interrupted = on_interrupted
        self._on_error = on_error
        self._on_cancel_sync = on_cancel_sync
        self._on_tool_call = on_tool_call

    # ================================================================
    # Lifecycle
    # ================================================================

    async def connect(self) -> None:
        if self._connected:
            return

        rt = self._rt
        if not rt.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for openai voice mode")
        if not rt.auth_token:
            raise ValueError("AUTH_TOKEN is required — set it in .env for LLM backend access")

        agent_type = self._settings.agent_type
        logger.info(f"Connecting OpenAI voice backend (llm={agent_type})")

        # 1. OpenAI client (shared for Realtime + TTS)
        from openai import AsyncOpenAI

        self._openai = AsyncOpenAI(api_key=rt.openai_api_key)

        # 2. LLM client — transport depends on agent_type
        if agent_type == "zeroclaw":
            try:
                ws_url = self._build_ws_chat_url()
                self._zc_ws = await websockets.connect(
                    ws_url,
                    open_timeout=rt.connect_timeout,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                )
                self._zc_ws_clean = True
                logger.info(f"ZeroClaw WebSocket connected at {rt.base_url}")
            except Exception as exc:
                logger.warning(f"ZeroClaw connect failed (will retry per message): {exc}")
                self._zc_ws = None
        else:
            # OpenClaw — HTTP SSE
            headers: dict[str, str] = {
                "Authorization": f"Bearer {rt.auth_token}",
                "Content-Type": "application/json",
            }
            if rt.agent_id:
                headers["x-openclaw-agent-id"] = rt.agent_id
            if rt.session_key:
                headers["x-openclaw-session-key"] = rt.session_key

            self._http_client = httpx.AsyncClient(
                base_url=rt.base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=rt.connect_timeout,
                    read=rt.read_timeout,
                    write=10.0,
                    pool=5.0,
                ),
                transport=httpx.AsyncHTTPTransport(retries=max(0, rt.max_retries)),
            )

            # Non-fatal connectivity probe
            try:
                probe = await self._http_client.get("/", timeout=3.0)
                logger.info(f"LLM backend reachable at {rt.base_url} (status={probe.status_code})")
            except Exception as exc:
                logger.warning(f"LLM backend probe: {exc}")

        # 3. System prompt
        self._system_prompt = self._settings.assistant_instructions
        thinking_mode = (rt.thinking_mode or "default").strip().lower()
        if thinking_mode in {"off", "minimal"}:
            guidance = (
                "\n\nResponse style: prioritize low-latency concise answers. "
                "Avoid long deliberation and keep reasoning brief."
            )
            self._system_prompt = f"{self._system_prompt}{guidance}"
        self._messages = [{"role": "system", "content": self._system_prompt}]

        # 4. Start Realtime session for VAD + STT
        self._realtime_task = asyncio.create_task(self._run_realtime_session())
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            if self._realtime_task and not self._realtime_task.done():
                self._realtime_task.cancel()
            raise ConnectionError("OpenAI Realtime session timed out")

        self._connected = True
        logger.info(
            f"OpenAI voice backend ready "
            f"(stt={rt.openai_realtime_model}, "
            f"tts={rt.openai_tts_model}/{rt.openai_voice}, "
            f"llm={agent_type}:{rt.agent_model})"
        )

    # ================================================================
    # ZeroClaw WebSocket helpers
    # ================================================================

    def _build_ws_chat_url(self) -> str:
        """Build ZeroClaw chat WebSocket URL including optional token."""
        rt = self._rt
        base = rt.base_url.rstrip("/")
        parsed = urlparse(base)

        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError(f"Unsupported ZeroClaw base URL scheme: {parsed.scheme}")

        ws_scheme = "wss" if scheme in {"https", "wss"} else "ws"
        path = f"{parsed.path.rstrip('/')}/ws/avatar" if parsed.path else "/ws/avatar"
        query = parsed.query

        if rt.auth_token:
            token_query = urlencode({"token": rt.auth_token})
            query = f"{query}&{token_query}" if query else token_query

        return urlunparse((ws_scheme, parsed.netloc, path, "", query, ""))

    async def _ensure_zc_ws(self) -> websockets.WebSocketClientProtocol:
        """Ensure persistent WebSocket to ZeroClaw is open, reconnecting if needed."""
        if self._zc_ws is not None and self._zc_ws.close_code is None and self._zc_ws_clean:
            return self._zc_ws

        if self._zc_ws is not None:
            try:
                await self._zc_ws.close()
            except Exception:
                pass
            self._zc_ws = None

        ws_url = self._build_ws_chat_url()
        self._zc_ws = await websockets.connect(
            ws_url,
            open_timeout=self._rt.connect_timeout,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=20,
        )
        self._zc_ws_clean = True
        logger.info("ZeroClaw WebSocket reconnected")
        return self._zc_ws

    async def _drain_ws_until_done(self, ws: websockets.WebSocketClientProtocol, timeout: float = 3.0) -> None:
        """Read and discard remaining ZeroClaw messages until 'done' or timeout."""
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

    # ================================================================
    # Realtime session (VAD + STT only)
    # ================================================================

    async def _run_realtime_session(self) -> None:
        """Background task: OpenAI Realtime WebSocket for VAD + STT."""
        try:
            async with self._openai.realtime.connect(
                model=self._rt.openai_realtime_model,
            ) as conn:
                self._realtime_conn = conn

                # Configure for STT-only: no audio output, no auto-response
                # GA Realtime API requires "type" and nests audio config
                await conn.session.update(
                    session={
                        "type": "realtime",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": 24000,
                                },
                                "transcription": {
                                    "model": self._rt.openai_transcription_model,
                                    "language": self._rt.openai_transcription_language,
                                    **({"prompt": self._rt.openai_transcription_prompt} if self._rt.openai_transcription_prompt else {}),
                                },
                                "turn_detection": {
                                    "type": self._rt.openai_vad_type,
                                    "create_response": False,
                                },
                            },
                        },
                    }
                )

                self._session_ready.set()
                logger.info("OpenAI Realtime session configured (VAD+STT only)")

                async for event in conn:
                    await self._handle_realtime_event(event)

        except asyncio.CancelledError:
            logger.debug("OpenAI Realtime session cancelled")
        except Exception as exc:
            logger.error(f"OpenAI Realtime session error: {exc}", exc_info=True)
            if self._on_error:
                await self._on_error({"error": str(exc)})
        finally:
            self._realtime_conn = None
            self._connected = False
            self._session_ready.clear()

    async def _handle_realtime_event(self, event: Any) -> None:
        """Route Realtime events — only transcription and barge-in."""
        event_type = event.type

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript.strip() if event.transcript else ""
            if transcript and not self._is_prompt_hallucination(transcript):
                logger.info(f"User said: {transcript!r}")
                await self._on_transcript_ready(transcript)
            elif transcript:
                logger.debug(f"Filtered prompt hallucination: {transcript!r}")

        elif event_type == "input_audio_buffer.speech_started":
            if self._state.is_responding:
                logger.info("Barge-in: user speech detected during response")
                self.cancel_response()

        elif event_type == "error":
            error_msg = str(event.error) if hasattr(event, "error") else str(event)
            logger.error(f"OpenAI Realtime error: {error_msg}")
            if self._on_error:
                await self._on_error({"error": error_msg})

    def _is_prompt_hallucination(self, transcript: str) -> bool:
        """Check if transcript is just hallucinated prompt words (no real speech).

        gpt-4o-transcribe can echo vocabulary hint words on silence/breathing.
        Filter transcripts that contain ONLY prompt words with no other content.
        """
        prompt = self._rt.openai_transcription_prompt
        if not prompt:
            return False

        # Build set of prompt words (case-insensitive)
        prompt_words = {w.strip().lower() for w in prompt.replace(",", " ").split() if w.strip()}
        if not prompt_words:
            return False

        # Strip punctuation from transcript and check if all words are prompt words
        transcript_words = {
            w.strip(".,!?;:\"'()-").lower()
            for w in transcript.split()
            if w.strip(".,!?;:\"'()-")
        }
        return len(transcript_words) > 0 and transcript_words.issubset(prompt_words)

    async def _on_transcript_ready(self, transcript: str) -> None:
        """User finished speaking — send transcript to LLM."""
        if self._state.is_responding:
            self.cancel_response()

        # Wait for previous stream task to finish
        if self._active_stream_task and not self._active_stream_task.done():
            try:
                await asyncio.wait_for(self._active_stream_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        self._append_message("user", transcript)

        if self._on_user_transcript:
            await self._on_user_transcript(transcript, "user")

        self._response_cancelled = False
        self._tts_cancelled.clear()
        self._cancel_event.clear()
        self._active_stream_task = asyncio.create_task(self._stream_and_speak())

    # ================================================================
    # Text input
    # ================================================================

    def send_text_message(self, text: str) -> None:
        if not self._connected:
            logger.warning("Not connected — dropping text message")
            return

        if self._state.is_responding:
            self.cancel_response()

        self._append_message("user", text)

        if self._on_user_transcript:
            asyncio.create_task(self._on_user_transcript(text, "user"))

        self._response_cancelled = False
        self._tts_cancelled.clear()
        self._cancel_event.clear()
        self._active_stream_task = asyncio.create_task(self._stream_and_speak())

    # ================================================================
    # Audio input
    # ================================================================

    def append_audio(self, audio_bytes: bytes) -> None:
        if not self._connected or not self._realtime_conn:
            return
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        asyncio.create_task(self._append_audio(encoded))

    async def _append_audio(self, encoded: str) -> None:
        try:
            if self._realtime_conn:
                await self._realtime_conn.input_audio_buffer.append(audio=encoded)
        except Exception as exc:
            logger.debug(f"Audio append failed: {exc}")

    # ================================================================
    # Interruption
    # ================================================================

    def cancel_response(self) -> None:
        if not self._state.is_responding:
            return

        logger.debug("Cancelling response")
        self._response_cancelled = True
        self._tts_cancelled.set()
        self._cancel_event.set()
        self._state.is_responding = False
        self._state.audio_done = True

        # For OpenClaw SSE, force-cancel the stream task.
        # For ZeroClaw WS, the cancel_event breaks the recv loop cleanly.
        if self._settings.agent_type != "zeroclaw":
            if self._active_stream_task and not self._active_stream_task.done():
                self._active_stream_task.cancel()

        if self._on_cancel_sync:
            self._on_cancel_sync()

    # ================================================================
    # LLM token iterators
    # ================================================================

    async def _iter_tokens_sse(self) -> AsyncGenerator[str, None]:
        """Yield text tokens from OpenClaw HTTP SSE (/v1/chat/completions)."""
        if not self._http_client:
            return

        rt = self._rt
        payload: dict[str, Any] = {
            "model": rt.agent_model,
            "messages": list(self._messages),
            "stream": True,
        }
        if rt.user_id:
            payload["user"] = rt.user_id

        async with self._http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                error_msg = f"LLM {response.status_code}: {body.decode('utf-8', errors='replace')}"
                logger.error(error_msg)
                if self._on_error:
                    await self._on_error({"error": error_msg})
                return

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
                if content and not self._response_cancelled:
                    yield content

    async def _iter_tokens_avatar_sse(
        self,
        tts_queue: asyncio.Queue[tuple[str, str, bool] | None] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield speech text from OpenClaw avatar SSE endpoint.

        Uses /v1/chat/completions/avatar which returns custom SSE event types:
        speech_chunk, rich_content, tool_call, tool_result, done.

        Speech chunks are already sentence-split by the server, so we yield
        them directly for TTS without buffering.

        When *tts_queue* is provided, filler phrases for tool calls are pushed
        directly onto the TTS queue so the avatar speaks while tools execute.
        """
        if not self._http_client:
            return

        rt = self._rt
        payload: dict[str, Any] = {
            "model": rt.agent_model,
            "messages": list(self._messages),
            "stream": True,
        }
        if rt.user_id:
            payload["user"] = rt.user_id

        endpoint = rt.avatar_endpoint
        current_event: str | None = None
        spoke_filler = False
        has_content = False

        async with self._http_client.stream("POST", endpoint, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                error_msg = f"LLM {response.status_code}: {body.decode('utf-8', errors='replace')}"
                logger.error(error_msg)
                if self._on_error:
                    await self._on_error({"error": error_msg})
                return

            async for line in response.aiter_lines():
                if self._response_cancelled:
                    break

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue

                if not line.startswith("data: "):
                    if line == "":
                        current_event = None
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = orjson.loads(data_str)
                except Exception:
                    current_event = None
                    continue

                if current_event == "speech_chunk":
                    content = data.get("content", "")
                    if content and not self._response_cancelled:
                        has_content = True
                        yield content

                elif current_event == "rich_content":
                    rc_content = data.get("content", "")
                    if rc_content and self._on_tool_call:
                        logger.info(f"Rich content received ({len(rc_content)} chars)")
                        await self._on_tool_call("rich_content", {"content": rc_content}, self._state.item_id)

                elif current_event == "tool_call":
                    tool_name = data.get("name", "unknown")

                    # Speak a filler phrase on the first tool call so the avatar
                    # isn't silent during execution.  Only one filler per turn.
                    if not spoke_filler and not self._response_cancelled:
                        spoke_filler = True
                        filler = random.choice(TOOL_FILLERS)
                        logger.info(f"Tool call: {tool_name} — filler: {filler!r}")
                        if tts_queue is not None:
                            sanitized = self._sanitize_for_tts(filler)
                            if sanitized:
                                await tts_queue.put((filler, sanitized, True))
                        elif not self._response_cancelled:
                            yield filler
                    else:
                        logger.info(f"Tool call: {tool_name}")

                    if self._on_tool_call:
                        await self._on_tool_call(
                            "tool_call",
                            {"name": tool_name, "tool_call_id": data.get("tool_call_id")},
                            self._state.item_id,
                        )

                elif current_event == "tool_result":
                    tool_name = data.get("name", "unknown")
                    success = data.get("success", True)
                    duration_ms = data.get("duration_ms")
                    logger.info(f"Tool result: {tool_name} success={success} ({duration_ms}ms)")
                    if self._on_tool_call:
                        await self._on_tool_call(
                            "tool_result",
                            {"name": tool_name, "success": success, "duration_ms": duration_ms},
                            self._state.item_id,
                        )

                elif current_event == "done":
                    # Server sends authoritative speech text — yield if not
                    # already covered by speech_chunk events
                    pass

                current_event = None

    async def _iter_tokens_ws(
        self,
        tts_queue: asyncio.Queue[tuple[str, str, bool] | None] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text tokens from ZeroClaw WebSocket (/ws/avatar).

        When *tts_queue* is provided, filler phrases for non-rich tool calls
        are pushed directly onto the TTS queue (with ``is_filler=True``) so
        they play immediately without waiting for the next text chunk.
        """
        user_content = ""
        for message in reversed(self._messages):
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break

        if not user_content:
            logger.warning("No user message found for ZeroClaw stream")
            return

        websocket = await self._ensure_zc_ws()
        await websocket.send(json.dumps({"type": "message", "content": user_content}))

        has_content = False
        spoke_filler = False

        while True:
            if self._response_cancelled:
                break

            # Wait for either a WS message or a cancel signal.
            # This avoids force-cancelling recv() which would dirty the WS.
            recv_fut = asyncio.create_task(websocket.recv())
            cancel_fut = asyncio.create_task(self._cancel_event.wait())
            try:
                done_futs, _ = await asyncio.wait(
                    {recv_fut, cancel_fut},
                    timeout=self._rt.read_timeout,
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

            msg_type = message.get("type")

            if msg_type == "error":
                error_msg = message.get("message", "Unknown ZeroClaw error")
                if self._on_error:
                    await self._on_error({"error": error_msg})
                break

            if msg_type == "rich_content":
                # Avatar channel: rich content — forward to client, not TTS
                if self._on_tool_call:
                    rc_content = str(message.get("content", ""))
                    if rc_content:
                        logger.info(f"Rich content received ({len(rc_content)} chars)")
                        await self._on_tool_call("rich_content", {"content": rc_content}, self._state.item_id)
                continue

            if msg_type == "tool_call":
                tool_name = message.get("name", "unknown")
                tool_args = message.get("args", {})

                # Speak a filler phrase on the first tool call so the avatar
                # isn't silent during execution.  Only one filler per turn.
                if not spoke_filler and not self._response_cancelled:
                    spoke_filler = True
                    filler = random.choice(TOOL_FILLERS)
                    logger.info(f"Tool call: {tool_name} — filler: {filler!r}")
                    if tts_queue is not None:
                        sanitized = self._sanitize_for_tts(filler)
                        if sanitized:
                            await tts_queue.put((filler, sanitized, True))
                else:
                    logger.info(f"Tool call: {tool_name}")

                continue

            if msg_type == "speech_chunk":
                content = str(message.get("content", ""))
                if content:
                    has_content = True
                    yield content
            elif msg_type == "done":
                done_text = str(message.get("full_response", ""))
                if done_text and not has_content:
                    yield done_text
                break

    # ================================================================
    # Streaming + TTS
    # ================================================================

    async def _stream_and_speak(self) -> None:
        """
        Send conversation to LLM, buffer tokens into sentences,
        and synthesize each via OpenAI TTS API.

        Routes to HTTP SSE (OpenClaw) or WebSocket (ZeroClaw) based on agent_type.
        """
        rt = self._rt
        agent_type = self._settings.agent_type
        session_id = f"session_{int(time.time() * 1000)}"
        item_id = f"openai_rt_{int(time.time() * 1000)}"

        # ── Signal response start ───────────────────────────────────
        self._state.session_id = session_id
        self._state.item_id = item_id
        self._state.is_responding = True
        self._state.transcript_buffer = ""
        self._state.audio_done = False

        if self._on_response_start:
            await self._on_response_start(session_id)

        full_response = ""
        sentence_buffer = ""

        # TTS sentence queue (None = sentinel for "done")
        # Tuple: (original_text, sanitized_text, is_filler)
        tts_queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(tts_queue, item_id))

        try:
            # Choose LLM streaming method based on agent type
            use_avatar = agent_type == "openclaw" and self._rt.use_avatar_endpoint
            if agent_type == "zeroclaw":
                token_stream = self._iter_tokens_ws(tts_queue=tts_queue)
            elif use_avatar:
                token_stream = self._iter_tokens_avatar_sse(tts_queue=tts_queue)
            else:
                token_stream = self._iter_tokens_sse()

            async for content in token_stream:
                if self._response_cancelled:
                    break

                # Avatar SSE / ZeroClaw WS yield pre-split sentences — add space separator
                # Standard SSE yields raw tokens that already include whitespace
                is_sentence_level = use_avatar or agent_type == "zeroclaw"
                if is_sentence_level:
                    full_response += (" " if full_response else "") + content
                else:
                    full_response += content
                self._state.transcript_buffer = full_response

                if use_avatar:
                    sanitized = self._sanitize_for_tts(content)
                    if sanitized:
                        await tts_queue.put((content, sanitized, False))
                else:
                    sentence_buffer += content
                    sentences, sentence_buffer = self._extract_sentences(sentence_buffer)
                    for sent in sentences:
                        sanitized = self._sanitize_for_tts(sent)
                        if sanitized:
                            await tts_queue.put((sent, sanitized, False))

            # ── Stream done — flush remaining sentence to TTS ──────
            if not self._response_cancelled:
                if not use_avatar:
                    remaining = sentence_buffer.strip()
                    if remaining:
                        sanitized = self._sanitize_for_tts(remaining)
                        if sanitized:
                            await tts_queue.put((remaining, sanitized, False))
                await tts_queue.put(None)
                if tts_task:
                    await tts_task
            else:
                if not tts_task.done():
                    tts_task.cancel()
                    try:
                        await tts_task
                    except (asyncio.CancelledError, Exception):
                        pass

        except asyncio.CancelledError:
            logger.debug("LLM stream cancelled")
        except httpx.ConnectError as exc:
            error_msg = f"Cannot reach LLM at {rt.base_url}: {exc}. Is the backend running?"
            logger.error(error_msg)
            if self._on_error:
                await self._on_error({"error": error_msg})
        except httpx.ReadTimeout:
            logger.warning("LLM SSE timed out")
            if self._on_error:
                await self._on_error({"error": "LLM response timed out"})
        except websockets.exceptions.ConnectionClosed as exc:
            self._zc_ws = None
            if not self._response_cancelled and self._on_error:
                await self._on_error({"error": f"ZeroClaw websocket closed: {exc}"})
        except asyncio.TimeoutError:
            if agent_type == "zeroclaw":
                self._zc_ws_clean = False
            logger.warning("LLM stream timed out")
            if self._on_error:
                await self._on_error({"error": "LLM response timed out"})
        except Exception as exc:
            if agent_type == "zeroclaw":
                self._zc_ws_clean = False
            logger.error(f"LLM stream error: {exc}", exc_info=True)
            if self._on_error:
                await self._on_error({"error": str(exc)})
        finally:
            # ZeroClaw post-barge-in cleanup: send cancel + drain
            if agent_type == "zeroclaw" and self._response_cancelled:
                if self._zc_ws and self._zc_ws.close_code is None:
                    try:
                        await self._zc_ws.send(json.dumps({"type": "cancel"}))
                        logger.debug("Sent cancel message to ZeroClaw")
                    except Exception as exc:
                        logger.debug(f"Failed to send cancel to ZeroClaw: {exc}")
                        self._zc_ws_clean = False
                    try:
                        await self._drain_ws_until_done(self._zc_ws, timeout=0.5)
                    except Exception as exc:
                        logger.debug(f"WS drain error, marking dirty: {exc}")
                        self._zc_ws_clean = False

            # Ensure TTS task terminates even on error
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

            if not self._response_cancelled and self._on_response_end:
                await self._on_response_end(full_response, item_id)
                if self._response_cancelled and self._on_interrupted:
                    await self._on_interrupted()
            elif self._response_cancelled and self._on_interrupted:
                await self._on_interrupted()

            self._state.is_responding = False
            self._state.audio_done = True
            self._active_stream_task = None

    # ================================================================
    # TTS worker (OpenAI TTS API)
    # ================================================================

    async def _tts_worker(
        self,
        queue: asyncio.Queue[tuple[str, str, bool] | None],
        item_id: str,
    ) -> None:
        """Synthesize queued sentences via OpenAI TTS API."""
        sentence_count = 0
        # 200ms silence at 24kHz PCM16 mono — breath pause between sentences
        pause_samples = int(24000 * 0.20)
        silence_pad = bytes(pause_samples * 2)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if self._response_cancelled:
                    break

                original, sanitized, is_filler = item

                # Skip filler if real content already arrived — the tool
                # returned fast and there's no silence to fill.
                if is_filler and not queue.empty():
                    logger.debug("Filler skipped — real content already queued")
                    continue

                # Breath pause between sentences (not before the first)
                if sentence_count > 0 and self._on_audio_delta and not self._response_cancelled:
                    await self._on_audio_delta(silence_pad)

                # Skip filler — it's spoken but not part of the real transcript,
                # and its offsets would corrupt the virtual cursor for real content.
                if not is_filler and self._on_transcript_delta and not self._response_cancelled:
                    await self._on_transcript_delta(original, "assistant", item_id, None)

                logger.debug(f"TTS synthesizing: {sanitized!r}")

                try:
                    # Re-chunk OpenAI's variable-size HTTP streaming chunks
                    # into fixed ~100ms (4800 bytes) pieces to match Piper TTS
                    # delivery cadence.  Without this, OpenAI sends large
                    # bursty chunks (8-16KB) that cause wav2arkit frame bursts
                    # and blendshape/audio desync.
                    delivery_bytes = 24000 // 10 * 2  # 4800 bytes = 100ms @ 24kHz PCM16
                    rechunk_buf = bytearray()

                    async with self._openai.audio.speech.with_streaming_response.create(
                        model=self._rt.openai_tts_model,
                        voice=self._rt.openai_voice,
                        input=sanitized,
                        response_format="pcm",
                        speed=self._rt.openai_tts_speed,
                    ) as response:
                        async for chunk in response.iter_bytes():
                            if self._response_cancelled:
                                break
                            rechunk_buf.extend(chunk)
                            while len(rechunk_buf) >= delivery_bytes:
                                if self._response_cancelled:
                                    break
                                if self._on_audio_delta:
                                    await self._on_audio_delta(bytes(rechunk_buf[:delivery_bytes]))
                                del rechunk_buf[:delivery_bytes]

                    # Flush any remaining audio in the re-chunk buffer
                    if rechunk_buf and not self._response_cancelled and self._on_audio_delta:
                        await self._on_audio_delta(bytes(rechunk_buf))
                except Exception as exc:
                    logger.error(f"OpenAI TTS error: {exc}")
                    if self._response_cancelled:
                        break
                    continue

                sentence_count += 1

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"TTS worker error: {exc}", exc_info=True)

    # ================================================================
    # Sentence extraction
    # ================================================================

    def _sanitize_for_tts(self, text: str) -> str:
        cleaned = _UNSUPPORTED_TTS_CHARS.sub("", text)
        return " ".join(cleaned.split())

    def _extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        parts = _SENTENCE_BOUNDARY.split(buffer)
        if len(parts) <= 1:
            if len(buffer) >= 60:
                clause_parts = _CLAUSE_BOUNDARY.split(buffer)
                if len(clause_parts) > 1:
                    sentences = [p.strip() for p in clause_parts[:-1] if p.strip()]
                    return sentences, clause_parts[-1]
            if len(buffer) > self._rt.tts_sentence_max_chars:
                idx = buffer.rfind(" ", 0, self._rt.tts_sentence_max_chars)
                if idx > 0:
                    return [buffer[:idx].strip()], buffer[idx + 1 :]
            return [], buffer

        sentences = [p.strip() for p in parts[:-1] if p.strip()]
        remainder = parts[-1]
        return sentences, remainder

    # ================================================================
    # Disconnect
    # ================================================================

    async def disconnect(self) -> None:
        for task in (self._active_stream_task, self._realtime_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._zc_ws:
            try:
                await self._zc_ws.close()
            except Exception:
                pass
            self._zc_ws = None

        self._openai = None
        self._realtime_conn = None
        self._connected = False
        self._messages = []
        self._state = ConversationState()
        self._session_ready.clear()
        self._cancel_event.clear()
        logger.info("OpenAI voice backend disconnected")
