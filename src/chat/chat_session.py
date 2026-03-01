import asyncio
import base64
import concurrent.futures
import time
from pathlib import Path
from typing import Any

import orjson
from fastapi import WebSocket

from core.logger import get_logger
from core.settings import Settings
from services import Wav2ArkitService, create_agent_instance

logger = get_logger(__name__)


class ChatSession:
    """
    Represents a single active conversation session.
    Holds all state specific to one user connection.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        settings: Settings,
        wav2arkit_service: Wav2ArkitService,
    ):
        logger.info(f"Initializing ChatSession for client: {session_id}")

        self.websocket = websocket
        self.session_id = session_id
        self.settings = settings
        self.wav2arkit_service = wav2arkit_service

        # Unique Agent Instance per Session
        self.agent = create_agent_instance()

        # Determine negotiated input sample rate
        self.input_sample_rate = self.agent.input_sample_rate
        logger.info(f"Session {self.session_id}: Negotiated input sample rate: {self.input_sample_rate}")

        # Determine negotiated output sample rate
        self.output_sample_rate = self.agent.output_sample_rate
        logger.info(f"Session {self.session_id}: Negotiated output sample rate: {self.output_sample_rate}")

        # Client State
        self.is_streaming_audio = False
        self.user_id = ""

        # Audio Debugging
        self.debug_file_path = None
        if self.settings.debug_audio_capture:
            timestamp = int(time.time())
            self.debug_file_path = Path(f"debug_input_{self.session_id[:8]}_{timestamp}.pcm")
            logger.info(f"Session {self.session_id}: Debug audio capture ENABLED -> {self.debug_file_path}")
        self.is_active = True

        # Audio and frame processing state
        self.audio_buffer: bytearray = bytearray()
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=120)
        self.audio_chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=15)

        # Tracking state
        self.current_turn_id: str | None = None
        self.current_turn_session_id: str | None = None  # Turn-level session ID from agent
        self.first_audio_received: bool = False
        self.speech_ended: bool = False
        self.is_interrupted: bool = False
        self._cancel_playback: bool = False
        self.speech_start_time: float = 0.0
        self.actual_audio_start_time: float = 0.0
        self.total_frames_emitted: int = 0
        self.total_audio_received: float = 0.0
        self.blendshape_frame_idx: int = 0
        self._last_user_transcript_finalized_at: float = 0.0
        self._last_response_start_at: float = 0.0
        self._last_first_audio_at: float = 0.0
        self._last_input_source: str = "unknown"
        self._interrupt_sent: bool = False

        # Track accumulated text for accurate interruption cutting
        self.current_turn_text: str = ""

        # Where we are in the text stream (time-wise)
        self.virtual_cursor_text_ms = 0.0

        # Calibration constant for transcript timing (configurable via settings)
        self.chars_per_second = self.agent.transcript_speed
        logger.info(
            f"Session {self.session_id}: Using agent-specific transcript speed: {self.chars_per_second} chars/sec"
        )

        # Background tasks
        self.frame_emit_task: asyncio.Task | None = None
        self.inference_task: asyncio.Task | None = None

        # Thread safety and re-entry prevention during interruptions
        self._interruption_lock = asyncio.Lock()

        # Dedicated thread pool for heavier inference tasks
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._setup_agent_handlers()

    def _setup_agent_handlers(self) -> None:
        """Setup event handlers specific to this session's agent."""
        self.agent.set_event_handlers(
            on_audio_delta=self._handle_audio_delta,
            on_response_start=self._handle_response_start,
            on_response_end=self._handle_response_end,
            on_user_transcript=self._handle_user_transcript,
            on_transcript_delta=self._handle_transcript_delta,
            on_interrupted=self._handle_interrupted,
            on_cancel_sync=self._on_agent_cancel_sync,
        )

    def _on_agent_cancel_sync(self) -> None:
        """Sync callback from agent.cancel_response().

        Sends the interrupt message to the frontend immediately (fire-and-forget)
        and sets flags so workers stop emitting frames.  The full cleanup
        (_execute_interruption_sequence) still runs later via _handle_interrupted.
        """
        self._cancel_playback = True
        self.audio_buffer.clear()

        if not self._interrupt_sent:
            self._interrupt_sent = True
            current_offset_ms = 0
            if self.settings.blendshape_fps > 0:
                current_offset_ms = int(
                    (self.total_frames_emitted / self.settings.blendshape_fps) * 1000
                )
            asyncio.create_task(self._send_barge_in_interrupt(current_offset_ms))

    async def _send_barge_in_interrupt(self, offset_ms: int) -> None:
        """Fire-and-forget: tell the client to stop playback NOW."""
        try:
            await self.send_json(
                {
                    "type": "interrupt",
                    "timestamp": int(time.time() * 1000),
                    "turnId": self.current_turn_id,
                    "offsetMs": offset_ms,
                }
            )
            await self.send_json({"type": "avatar_state", "state": "Listening"})
            logger.info(
                f"Session {self.session_id}: Barge-in interrupt sent immediately (offset={offset_ms}ms)"
            )
        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to send barge-in interrupt: {e}")

    async def start(self) -> None:
        """Initialize connection to agent."""

        # [PROTOCOL] Configure client with negotiated sample rate
        await self.send_json({"type": "config", "audio": {"inputSampleRate": self.input_sample_rate}})

        if not self.agent.is_connected:
            await self.agent.connect()

    async def stop(self) -> None:
        """Cleanup resources."""
        self.is_active = False
        self.is_interrupted = True

        await self._cancel_and_wait_task(self.inference_task)
        await self._cancel_and_wait_task(self.frame_emit_task)

        # Shutdown executor
        self._inference_executor.shutdown(wait=False)

        # Disconnect Agent
        await self.agent.disconnect()
        logger.info(f"Session {self.session_id} stopped.")

    async def _cancel_and_wait_task(self, task: asyncio.Task | None) -> None:
        """Helper to safely cancel and await a task."""
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error during task cancellation: {e}")

    async def send_json(self, message: dict[str, Any]) -> None:
        """Send JSON to this specific client."""
        if not self.is_active:
            return

        try:
            message_str = orjson.dumps(message).decode("utf-8")
            await self.websocket.send_text(message_str)
        except Exception as e:
            # Silence expected errors on disconnect
            if "Unexpected ASGI message" in str(e) or "websocket.close" in str(e):
                logger.debug(f"Socket closed while sending to {self.session_id}: {e}")
            else:
                logger.error(f"Error sending to client {self.session_id}: {e}")

    async def _drain_queue(self, queue: asyncio.Queue) -> None:
        """Helper to flush an asyncio queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _handle_response_start(self, session_id: str) -> None:
        """Handle AI response start."""
        now = time.perf_counter()
        self._last_response_start_at = now
        logger.info(
            f"Session {self.session_id}: Response start received, creating turn for session {session_id} "
            f"(inference_task={'alive' if self.inference_task and not self.inference_task.done() else 'gone'}, "
            f"speech_ended={self.speech_ended})"
        )
        if self._last_user_transcript_finalized_at > 0:
            user_to_model_ms = (now - self._last_user_transcript_finalized_at) * 1000
            logger.info(
                f"Session {self.session_id}: Latency user_final->model_start = {user_to_model_ms:.1f}ms "
                f"(source={self._last_input_source})"
            )
        self.current_turn_session_id = session_id  # Store turn-level session ID
        self.current_turn_id = f"turn_{int(time.time() * 1000)}_{session_id[:8]}"
        self.speech_start_time = time.time()
        self.actual_audio_start_time = 0
        self.total_frames_emitted = 0
        self.total_audio_received = 0
        self.blendshape_frame_idx = 0
        self.speech_ended = False
        self.is_interrupted = False
        self._cancel_playback = False
        self._interrupt_sent = False
        self.first_audio_received = False
        self.current_turn_text = ""
        self.virtual_cursor_text_ms = 0.0

        # Clear queues
        self.audio_buffer.clear()
        await self._drain_queue(self.frame_queue)
        await self._drain_queue(self.audio_chunk_queue)

        if self.wav2arkit_service.is_available:
            self.wav2arkit_service.reset_context()

        # Send start event BEFORE starting tasks to ensure client is ready to receive frames
        logger.debug(
            f"Session {self.session_id}: Sending audio_start with turnId={self.current_turn_id}, sessionId={self.current_turn_session_id}"
        )
        await self.send_json({"type": "avatar_state", "state": "Responding"})
        await self.send_json(
            {
                "type": "audio_start",
                "sessionId": self.current_turn_session_id,
                "turnId": self.current_turn_id,
                "sampleRate": self.output_sample_rate,
                "format": "audio/pcm16",
                "timestamp": int(time.time() * 1000),
            }
        )

        # Start background tasks
        if self.frame_emit_task is None or self.frame_emit_task.done():
            self.frame_emit_task = asyncio.create_task(self._emit_frames())
        if self.inference_task is None or self.inference_task.done():
            self.inference_task = asyncio.create_task(self._inference_worker())

    async def _handle_audio_delta(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from agent."""
        if self.is_interrupted or self._cancel_playback:
            logger.debug(f"Session {self.session_id}: audio_delta DROPPED (interrupted/cancelled)")
            return

        # Auto-(re)start worker tasks if they died or were never created.
        # This handles the voice-STT path where TTS audio may arrive
        # slightly after the initial inference_task completed or before
        # _handle_response_start had a chance to create it.
        if self.inference_task is None or self.inference_task.done():
            logger.info(f"Session {self.session_id}: inference_task gone — restarting workers")
            self.speech_ended = False
            self.inference_task = asyncio.create_task(self._inference_worker())
        if self.frame_emit_task is None or self.frame_emit_task.done():
            self.frame_emit_task = asyncio.create_task(self._emit_frames())

        if not self.first_audio_received:
            self.first_audio_received = True
            self.actual_audio_start_time = time.time()
            first_audio_now = time.perf_counter()
            self._last_first_audio_at = first_audio_now

            if self._last_response_start_at > 0:
                model_to_first_audio_ms = (first_audio_now - self._last_response_start_at) * 1000
                logger.info(
                    f"Session {self.session_id}: Latency model_start->first_audio = {model_to_first_audio_ms:.1f}ms"
                )

            if self._last_user_transcript_finalized_at > 0:
                user_to_first_audio_ms = (first_audio_now - self._last_user_transcript_finalized_at) * 1000
                logger.info(
                    f"Session {self.session_id}: Latency user_final->first_audio (E2E) = {user_to_first_audio_ms:.1f}ms "
                    f"(source={self._last_input_source})"
                )

        if not self.wav2arkit_service.is_available:
            logger.debug(f"Session {self.session_id}: Sending audio_chunk to client ({len(audio_bytes)} bytes PCM16)")
            await self.send_json(
                {
                    "type": "audio_chunk",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "sessionId": self.current_turn_session_id or self.session_id,
                    "timestamp": int(time.time() * 1000),
                }
            )
            return

        self.audio_buffer.extend(audio_bytes)

        chunk_samples = len(audio_bytes) // 2
        chunk_duration = chunk_samples / self.output_sample_rate
        self.total_audio_received += chunk_duration

        # Extract ALL complete 500ms chunks from the buffer (not just one).
        # TTS sentences produce large chunks (800-3400ms each); extracting
        # only one chunk per call left most audio trapped in the buffer
        # until _handle_response_end flushed it as one massive bulk chunk,
        # causing wav2arkit to stall and frames to burst.
        chunk_bytes_size = int(self.settings.audio_chunk_duration * self.output_sample_rate * 2)
        while len(self.audio_buffer) >= chunk_bytes_size:
            chunk_bytes = bytes(self.audio_buffer[:chunk_bytes_size])
            self.audio_buffer = bytearray(self.audio_buffer[chunk_bytes_size:])

            if self.is_interrupted or self._cancel_playback:
                return

            await self.audio_chunk_queue.put(chunk_bytes)

    async def _handle_response_end(self, transcript: str, item_id: str | None = None) -> None:
        """Handle AI response end."""
        if self.is_interrupted or self._cancel_playback:
            return

        # Wait for audio to stabilize
        last_audio = self.total_audio_received
        stable_count = 0
        max_wait_iterations = 60
        iterations = 0
        while stable_count < 15 and iterations < max_wait_iterations:
            await asyncio.sleep(0.05)
            iterations += 1
            if self.is_interrupted or self._cancel_playback:
                return
            if self.total_audio_received == last_audio:
                stable_count += 1
            else:
                stable_count = 0
                last_audio = self.total_audio_received

        if self.is_interrupted or self._cancel_playback:
            return

        # Flush remaining audio
        buffer_samples = len(self.audio_buffer) // 2
        if buffer_samples > 0 and self.wav2arkit_service.is_available:
            min_samples = int(0.3 * self.output_sample_rate)
            remaining_bytes = bytes(self.audio_buffer)
            if buffer_samples < min_samples:
                padding = bytes((min_samples - buffer_samples) * 2)
                remaining_bytes = remaining_bytes + padding
            self.audio_buffer.clear()
            await self.audio_chunk_queue.put(remaining_bytes)

        while not self.audio_chunk_queue.empty() and not self.is_interrupted and not self._cancel_playback:
            await asyncio.sleep(0.05)

        self.speech_ended = True

        # Wait for workers
        if self.inference_task and not self.inference_task.done():
            try:
                await asyncio.wait_for(self.inference_task, timeout=8.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Check interruption/barge-in before final waits/sends
        if self.is_interrupted or self._cancel_playback:
            return

        if self.frame_emit_task and not self.frame_emit_task.done():
            try:
                remaining_frames = self.frame_queue.qsize()
                timeout = max(5.0, remaining_frames / self.settings.blendshape_fps + 2.0)
                await asyncio.wait_for(self.frame_emit_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        if self.is_interrupted or self._cancel_playback:
            return

        await self.send_json(
            {
                "type": "audio_end",
                "sessionId": self.current_turn_session_id or self.session_id,
                "turnId": self.current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

        if transcript:
            logger.debug(
                f"Session {self.session_id}: Sending transcript_done ({len(transcript)} chars): "
                f"{transcript[:100]!r}{'...' if len(transcript) > 100 else ''}"
            )
            msg = {
                "type": "transcript_done",
                "text": transcript,
                "role": "assistant",
                "turnId": self.current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
            if item_id:
                msg["itemId"] = item_id
            await self.send_json(msg)
        else:
            logger.warning(f"Session {self.session_id}: transcript_done SKIPPED — empty transcript")

        await self.send_json({"type": "avatar_state", "state": "Listening"})

        self.current_turn_id = None
        self.speech_ended = False
        self.inference_task = None
        self.frame_emit_task = None

    async def _handle_transcript_delta(
        self,
        text: str,
        role: str = "assistant",
        item_id: str | None = None,
        previous_item_id: str | None = None,
    ) -> None:
        if self.is_interrupted or self._cancel_playback:
            logger.debug(
                f"Session {self.session_id}: transcript_delta DROPPED "
                f"(interrupted={self.is_interrupted}, cancel={self._cancel_playback}) text={text!r}"
            )
            return
        logger.debug(f"Session {self.session_id}: transcript_delta received: {text!r} role={role}")
        # LLM APIs vary in chunk granularity — some send word-level tokens,
        # others send entire sentences.  Split multi-word chunks into
        # word-level deltas so the frontend SubtitleController gets
        # manageable display units and doesn't overflow the subtitle area.
        if role == "assistant":
            # Anchor virtual cursor to actual audio position once per
            # sentence so that within-sentence words use the heuristic.
            # Only advance forward — never backward.  The filler text
            # (spoken before tool results arrive) can push the cursor
            # ahead of the actual audio; resetting it backward would
            # create overlapping startOffset values that break clients.
            actual_audio_ms = self.total_audio_received * 1000
            if actual_audio_ms > self.virtual_cursor_text_ms:
                self.virtual_cursor_text_ms = actual_audio_ms

            words = text.split()
            if len(words) > 2:
                leading_ws = text[: len(text) - len(text.lstrip())]
                for i, word in enumerate(words):
                    prefix = leading_ws if i == 0 else " "
                    await self._emit_transcript_delta(prefix + word, role, item_id, previous_item_id)
                return

        await self._emit_transcript_delta(text, role, item_id, previous_item_id)

    async def _emit_transcript_delta(
        self,
        text: str,
        role: str,
        item_id: str | None,
        previous_item_id: str | None,
    ) -> None:
        if role == "assistant":
            self.current_turn_text += text

            char_duration_ms = (len(text) / self.chars_per_second) * 1000
            start_offset = self.virtual_cursor_text_ms
            end_offset = start_offset + char_duration_ms

            self.virtual_cursor_text_ms += char_duration_ms
        else:
            start_offset = 0
            end_offset = 0

        turn_id = self.current_turn_id if role == "assistant" else f"user_{int(time.time() * 1000)}"
        session_id = (
            self.current_turn_session_id
            if role == "assistant" and self.current_turn_session_id
            else self.session_id
        )
        msg = {
            "type": "transcript_delta",
            "text": text,
            "role": role,
            "turnId": turn_id,
            "sessionId": session_id,
            "timestamp": int(time.time() * 1000),
            "startOffset": int(start_offset),
            "endOffset": int(end_offset),
        }

        if item_id:
            msg["itemId"] = item_id
        if previous_item_id:
            msg["previousItemId"] = previous_item_id
        await self.send_json(msg)

    async def _handle_user_transcript(self, transcript: str, role: str = "user") -> None:
        self._last_user_transcript_finalized_at = time.perf_counter()
        self._last_input_source = "stt"
        user_turn_id = f"{role}_{int(time.time() * 1000)}"
        await self.send_json(
            {
                "type": "transcript_done",
                "text": transcript,
                "role": role,
                "turnId": user_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _calculate_truncated_text(self) -> str:
        """Helper to calculate truncated text based on audio duration."""
        try:
            seconds_spoken = self.total_frames_emitted / self.settings.blendshape_fps
            estimated_chars = int(seconds_spoken * self.chars_per_second)

            final_text = self.current_turn_text
            if len(final_text) > estimated_chars:
                # Try to cut at word boundary
                search_buffer = 10
                cut_index = final_text.find(" ", estimated_chars)

                if cut_index != -1 and cut_index - estimated_chars < search_buffer:
                    final_text = final_text[:cut_index] + "..."
                else:
                    final_text = final_text[:estimated_chars] + "..."

            logger.debug(
                f"Session {self.session_id}: Text truncated to {len(final_text)} chars ({seconds_spoken:.2f}s spoken)"
            )
            return final_text
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error computing text truncation: {e}")
            return self.current_turn_text

    async def _execute_interruption_sequence(self) -> None:
        """
        Executes the core cleanup logic inside the interruption lock.
        Separated for clarity and maintenance.
        """
        # Double-check inside lock
        if self.is_interrupted:
            return

        self.is_interrupted = True
        interrupted_turn_id = self.current_turn_id
        logger.debug(f"Session {self.session_id}: Cancellation sequence (turn_id: {interrupted_turn_id})")

        # 1. Cancel Agent Response
        try:
            self.agent.cancel_response()
        except Exception as e:
            logger.debug(f"Session {self.session_id}: Agent cancel: {e}")

        # 2. Clear Local Buffers
        self.audio_buffer.clear()

        # 3. Drain Queues
        await self._drain_queue(self.audio_chunk_queue)
        await self._drain_queue(self.frame_queue)

        # 4. Cancel Tasks
        await self._cancel_and_wait_task(self.inference_task)
        await self._cancel_and_wait_task(self.frame_emit_task)

        self.inference_task = None
        self.frame_emit_task = None

        # 5. Reset State
        self.speech_ended = True
        self.current_turn_id = None

        # 6. Calc Truncation & Send
        final_text = await self._calculate_truncated_text()

        try:
            await self.send_json(
                {
                    "type": "transcript_done",
                    "text": final_text,
                    "role": "assistant",
                    "turnId": interrupted_turn_id,
                    "interrupted": True,
                    "timestamp": int(time.time() * 1000),
                }
            )
            logger.info(f"Session {self.session_id}: Interruption complete")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to send transcript_done: {e}")

        # Critical: Re-enable processing
        self.is_interrupted = False

    async def _handle_interrupted(self) -> None:
        """
        Production-grade interruption handler.
        """
        # Fast-path check
        if self.is_interrupted:
            return

        # [MODIFIED] Calculate offset immediately for the instant message
        current_offset_ms = 0
        if self.settings.blendshape_fps > 0:
            current_offset_ms = int((self.total_frames_emitted / self.settings.blendshape_fps) * 1000)

        # Immediate client notification (skip if already sent by _on_agent_cancel_sync)
        if not self._interrupt_sent:
            try:
                await self.send_json(
                    {
                        "type": "interrupt",
                        "timestamp": int(time.time() * 1000),
                        "turnId": self.current_turn_id,
                        "offsetMs": current_offset_ms,
                    }
                )
                await self.send_json({"type": "avatar_state", "state": "Listening"})
                self._interrupt_sent = True
            except Exception as e:
                logger.error(f"Session {self.session_id}: Failed to send interrupt: {e}")
                return
        else:
            logger.info(f"Session {self.session_id}: Interrupt already sent by barge-in, skipping duplicate")

        # Lock acquisition with timeout
        try:

            async def _locked_execution():
                async with self._interruption_lock:
                    await self._execute_interruption_sequence()

            await asyncio.wait_for(_locked_execution(), timeout=1.0)

        except asyncio.TimeoutError:
            logger.error(f"Session {self.session_id}: TIMEOUT unlocking interrupted state")
            self.is_interrupted = False
        except Exception as e:
            logger.error(f"Session {self.session_id}: EXCEPTION in interruption: {e}", exc_info=True)
            self.is_interrupted = False

    async def on_error(self, error: Any) -> None:
        """Handle error events from the agent."""
        logger.error(f"Session {self.session_id}: Agent error: {error}")

    async def _inference_worker(self) -> None:
        """Process audio chunks through Wav2Arkit model."""
        try:
            logger.info(f"Session {self.session_id}: Inference worker started")
            loop = asyncio.get_running_loop()

            while True:
                if self.is_interrupted or self._cancel_playback:
                    await asyncio.sleep(0.01)
                    continue

                try:
                    audio_bytes = await asyncio.wait_for(
                        self.audio_chunk_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    if self.speech_ended and self.audio_chunk_queue.empty():
                        break
                    continue

                if self.is_interrupted or self._cancel_playback:
                    await asyncio.sleep(0.01)
                    continue

                # Run inference in default executor
                try:
                    # Use dedicated executor to prevent blocking
                    frames = await loop.run_in_executor(
                        self._inference_executor,
                        self.wav2arkit_service.process_audio_chunk,
                        audio_bytes,
                        self.output_sample_rate,
                    )
                except Exception as e:
                    logger.error(f"Inference processing failed: {e}", exc_info=True)
                    # Prevent CPU spin loop on persistent error
                    await asyncio.sleep(0.1)
                    continue

                if self.is_interrupted or self._cancel_playback:
                    await asyncio.sleep(0.01)
                    continue

                if frames:
                    for frame in frames:
                        if self.is_interrupted or self._cancel_playback:
                            break
                        await self.frame_queue.put(frame)
                else:
                    logger.warning("Inference returned no frames")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Session {self.session_id} inference worker fatal error: {e}", exc_info=True)

    async def _emit_frames(self) -> None:
        """Emit synchronized frames."""
        try:
            while True:
                if self.is_interrupted or self._cancel_playback:
                    await asyncio.sleep(0.01)
                    continue

                # Pacing logic: reference actual_audio_start_time (when first TTS byte
                # arrived) so frames are sent at real-time rate from audio start.
                # Using speech_start_time (LLM start) causes a burst because by the
                # time inference completes, 1–2 s of "credit" has accumulated, sending
                # all frames at once and flooding the client buffer.
                # +15 frames (~500ms) gives the client enough headroom for smooth
                # playback while capping the pre-buffered audio so interruption is
                # responsive (client discards ≤500ms on interrupt, not 1–2s).
                if self.actual_audio_start_time > 0:
                    elapsed_time = time.time() - self.actual_audio_start_time
                    target_frames = int(elapsed_time * self.settings.blendshape_fps) + 15
                else:
                    target_frames = 0

                if self.total_frames_emitted >= target_frames:
                    await asyncio.sleep(0.005)
                    continue

                try:
                    frame_data = await asyncio.wait_for(
                        self.frame_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    inference_done = self.inference_task is None or self.inference_task.done()
                    if self.speech_ended and self.frame_queue.empty() and inference_done:
                        break
                    continue

                if self.is_interrupted or self._cancel_playback:
                    await asyncio.sleep(0.01)
                    continue

                if self.blendshape_frame_idx % 30 == 0:
                    audio_b64 = frame_data.get("audio", "")
                    audio_size = len(audio_b64)
                    logger.debug(
                        f"Session {self.session_id}: Sending sync_frame {self.blendshape_frame_idx} (Active). Audio payload: {audio_size} chars"
                    )

                await self.send_json(
                    {
                        "type": "sync_frame",
                        "weights": frame_data["weights"],
                        "audio": frame_data["audio"],
                        "sessionId": self.current_turn_session_id,
                        "turnId": self.current_turn_id,
                        "timestamp": int(time.time() * 1000),
                        "frameIndex": self.blendshape_frame_idx,
                    }
                )

                self.blendshape_frame_idx += 1
                self.total_frames_emitted += 1

        except asyncio.CancelledError:
            if not self.is_interrupted:
                # Drain
                while not self.frame_queue.empty():
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        await self.send_json(
                            {
                                "type": "sync_frame",
                                "weights": frame_data["weights"],
                                "audio": frame_data["audio"],
                                "sessionId": self.current_turn_session_id,
                                "turnId": self.current_turn_id,
                                "timestamp": int(time.time() * 1000),
                                "frameIndex": self.blendshape_frame_idx,
                            }
                        )
                        self.blendshape_frame_idx += 1
                        self.total_frames_emitted += 1
                    except asyncio.QueueEmpty:
                        break

    async def process_message(self, data: dict[str, Any]) -> None:
        """Handle incoming message from client."""
        msg_type = data.get("type")

        if msg_type == "text":
            self._last_user_transcript_finalized_at = time.perf_counter()
            self._last_input_source = "text"
            self.agent.send_text_message(data.get("data", ""))

        elif msg_type == "audio_stream_start":
            self.is_streaming_audio = True
            self.user_id = data.get("userId", "unknown")
            # Unstick session: If user starts speaking, previous interruption is over.
            # This ensures we recover even if OpenAI errored out in the previous turn.
            self.is_interrupted = False

        elif msg_type == "audio":
            if self.is_streaming_audio:
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                    # Debug: Save raw input audio
                    if self.debug_file_path:
                        try:
                            with open(self.debug_file_path, "ab") as f:
                                f.write(audio_bytes)
                        except Exception as e:
                            logger.warning(f"Failed to write debug audio: {e}")

                    self.agent.append_audio(audio_bytes)

        elif msg_type == "interrupt":
            logger.info(f"Session {self.session_id}: Received explicit interrupt command from client")
            await self._handle_interrupted()

        elif msg_type == "ping":
            await self.send_json(
                {
                    "type": "pong",
                    "timestamp": int(time.time() * 1000),
                }
            )
