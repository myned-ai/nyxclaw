"""STT service (faster-whisper + Silero VAD)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from core.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels above src/services/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_silero_vad_path() -> Path | None:
    """Return path to silero_vad.onnx if pre-downloaded, else None."""
    for base in (Path.cwd(), _PROJECT_ROOT):
        candidate = base / "pretrained_models" / "silero_vad.onnx"
        if candidate.exists():
            return candidate
    return None


class _SileroVADIterator:
    """Torch-free Silero VAD backed by the ONNX model via ORT.

    Replicates the ``VADIterator`` interface from the ``silero-vad`` pip
    package — same return values (``None``, ``{"start": N}``, ``{"end": N}``)
    — without pulling in PyTorch as a dependency.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        end_threshold: float | None = None,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._threshold = threshold
        self._end_threshold = end_threshold if end_threshold is not None else max(0.0, threshold - 0.15)
        self._sr = np.array(sampling_rate, dtype=np.int64)
        self._min_silence_samples = sampling_rate * min_silence_duration_ms // 1000
        self._speech_pad_samples = sampling_rate * speech_pad_ms // 1000
        self.reset_states()

    def reset_states(self) -> None:
        # v5 model: combined state tensor [2, batch, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # Context window: 64 samples @16kHz, 32 @8kHz (required by Silero v5)
        self._context_size = 64 if int(self._sr) == 16000 else 32
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0

    def __call__(self, x: np.ndarray, return_seconds: bool = False) -> dict | None:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self._current_sample += x.shape[1]

        # Silero v5 requires context from previous frame prepended to input
        x_with_ctx = np.concatenate([self._context, x], axis=1)
        out, self._state = self._session.run(None, {"input": x_with_ctx, "sr": self._sr, "state": self._state})
        self._context = x_with_ctx[:, -self._context_size :]
        prob = float(out[0, 0])

        if prob >= self._threshold:
            self._temp_end = 0
            if not self._triggered:
                self._triggered = True
                start = max(0, self._current_sample - self._speech_pad_samples - x.shape[1])
                return {"start": round(start / int(self._sr), 1) if return_seconds else start}
        elif prob < self._end_threshold and self._triggered:
            if self._temp_end == 0:
                self._temp_end = self._current_sample
            if self._current_sample - self._temp_end >= self._min_silence_samples:
                end = self._temp_end + self._speech_pad_samples
                self._temp_end = 0
                self._triggered = False
                return {"end": round(end / int(self._sr), 1) if return_seconds else end}
        return None


def _resolve_whisper_model_dir(model_name: str) -> Path | None:
    """Return a local model dir for faster-whisper if pre-downloaded files exist.

    Looks for ``model.bin`` under ``pretrained_models/faster_whisper_{name}/``
    relative to CWD and project root.  Returns ``None`` to fall back to HF Hub.
    """
    # "tiny.en" → "tiny_en"; "small.en" → "small_en"
    short = model_name.replace(".", "_").replace("-", "_")
    subdir = f"faster_whisper_{short}"
    for base in (Path.cwd(), _PROJECT_ROOT):
        candidate = base / "pretrained_models" / subdir
        if (candidate / "model.bin").exists():
            return candidate
    return None


class STTService:
    """STT service with faster-whisper transcription and Silero VAD."""

    _shared_lock = threading.Lock()
    _shared_transcribe_lock = threading.Lock()
    _shared_silero_vad_path: str | None = None
    _shared_silero_initialized = False
    _shared_whisper_models: dict[str, Any] = {}

    def __init__(
        self,
        on_word: Callable[[str, float], Awaitable[None]] | None = None,
        on_error: Callable[[Any], Awaitable[None]] | None = None,
        stt_model: str = "tiny.en",
        vad_start_threshold: float = 0.60,
        vad_end_threshold: float = 0.35,
        vad_min_silence_ms: int = 500,
        initial_prompt: str | None = None,
    ) -> None:
        self._on_word = on_word
        self._on_error = on_error

        self._stt_model = stt_model
        self._initial_prompt = initial_prompt
        self._vad_start_threshold = max(0.0, min(1.0, vad_start_threshold))
        self._vad_end_threshold = max(0.0, min(1.0, vad_end_threshold))
        self._vad_min_silence_ms = max(80, vad_min_silence_ms)

        self._connected = False

        # 768 samples @ 24 kHz = 32 ms -> resample_poly(2,3) -> 512 samples @ 16 kHz.
        # Silero VAD requires exactly 256 or 512 samples per frame at 16 kHz.
        self._frame_samples_24k = 768
        self._frame_bytes_24k = self._frame_samples_24k * 2
        self._pending = bytearray()
        self._speech_audio_24k = bytearray()

        self._has_speech = False
        self._pause_ready = False
        self._silence_frames = 0
        self._last_vad_conf = 0.0

        self._silero_iterator: Any = None
        self._silero_enabled = False
        # Tracks whether VAD considers us inside a speech segment.
        # VADIterator returns None during ongoing speech, so without this flag
        # the code falls through to the noisy RMS energy gate.
        self._vad_in_speech = False
        self._dbg_frame_count = 0

    # ── Public properties ───────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def pause_score(self) -> float:
        return self._last_vad_conf

    @property
    def has_speech(self) -> bool:
        return self._has_speech

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._connected:
            return

        self._try_init_silero()
        self._try_init_whisper_model()

        self._connected = True
        logger.info(f"STT ready (silero={self._silero_enabled}, model={self._stt_model})")

    async def disconnect(self) -> None:
        self._connected = False

        self._reset()
        logger.info("Disconnected from ONNX STT")

    def reset_turn(self, *, reset_vad_state: bool = False) -> None:
        self._has_speech = False
        self._pause_ready = False
        self._silence_frames = 0
        self._speech_audio_24k.clear()
        self._pending.clear()
        self._last_vad_conf = 0.0
        self._vad_in_speech = False
        # Only zero the Silero RNN hidden state after barge-in to prevent
        # cascading re-triggers.  On normal turn boundaries keep the RNN
        # warm so the VAD stays sensitive and can detect a quick interrupt
        # on the very next response.
        if reset_vad_state and self._silero_iterator is not None:
            self._silero_iterator.reset_states()

    # ── Audio I/O ───────────────────────────────────────────────────

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        if not self._connected:
            return

        self._pending.extend(pcm16_bytes)
        while len(self._pending) >= self._frame_bytes_24k:
            frame_bytes = bytes(self._pending[: self._frame_bytes_24k])
            del self._pending[: self._frame_bytes_24k]

            # Offload blocking scipy + ORT to a thread so the event loop stays free.
            vad_conf = await asyncio.to_thread(self._process_frame, frame_bytes)
            self._last_vad_conf = vad_conf

            if vad_conf >= self._vad_start_threshold:
                self._has_speech = True
                self._pause_ready = False
                self._silence_frames = 0
                self._speech_audio_24k.extend(frame_bytes)
                continue

            if self._has_speech:
                self._speech_audio_24k.extend(frame_bytes)

                # Silero end event (conf == 0.0) means the iterator already
                # waited min_silence_duration_ms internally.  Accept the pause
                # immediately to avoid double-counting silence.
                if vad_conf == 0.0:
                    self._pause_ready = True
                elif vad_conf <= self._vad_end_threshold:
                    self._silence_frames += 1
                else:
                    self._silence_frames = 0

                # Fallback for energy-gate VAD (Silero unavailable) which
                # doesn't have built-in silence duration tracking.
                if not self._pause_ready:
                    silence_ms = self._silence_frames * 32  # 768 samples @ 24 kHz = 32 ms
                    if silence_ms >= self._vad_min_silence_ms:
                        self._pause_ready = True

    def _process_frame(self, frame_bytes: bytes) -> float:
        """Resample + VAD in a worker thread (called via asyncio.to_thread)."""
        frame_i16 = np.frombuffer(frame_bytes, dtype=np.int16)
        frame_f32_24k = frame_i16.astype(np.float32) / 32768.0
        frame_f32_16k = self._resample_24k_to_16k(frame_f32_24k)

        vad_conf = self._detect_speech_confidence(frame_f32_16k)

        # Periodic debug: log VAD score + RMS every ~160 frames (~5 sec)
        self._dbg_frame_count += 1
        if self._dbg_frame_count % 160 == 0:
            rms = float(np.sqrt(np.mean(np.square(frame_f32_16k))))
            logger.debug(
                f"VAD probe: conf={vad_conf:.3f} rms={rms:.4f} "
                f"has_speech={self._has_speech} in_speech={self._vad_in_speech}"
            )
        return vad_conf

    async def flush_silence(self) -> None:
        if not self._connected:
            return
        if not self._speech_audio_24k:
            return

        try:
            transcript = await asyncio.to_thread(self._transcribe_current_segment)
            if transcript and self._on_word:
                for token in transcript.split():
                    await self._on_word(token, 0.0)
        except Exception as exc:
            logger.error(f"ONNX STT flush error: {exc}")
            if self._on_error:
                await self._on_error({"error": str(exc)})

    def check_pause(self) -> bool:
        """``True`` when VAD indicates the user has stopped speaking."""
        return self._has_speech and self._pause_ready

    # ── Private ─────────────────────────────────────────────────────

    def _reset(self) -> None:
        self.reset_turn(reset_vad_state=True)

    def _resample_24k_to_16k(self, audio_f32_24k: np.ndarray) -> np.ndarray:
        return resample_poly(audio_f32_24k, up=2, down=3).astype(np.float32, copy=False)

    def _try_init_silero(self) -> None:
        with self._shared_lock:
            if self._shared_silero_initialized:
                vad_path = self._shared_silero_vad_path
                if vad_path is None:
                    self._silero_enabled = False
                    return
                # Create a per-instance iterator (own RNN state) from the
                # already-validated model path.
                self._silero_iterator = _SileroVADIterator(
                    model_path=vad_path,
                    threshold=self._vad_start_threshold,
                    end_threshold=self._vad_end_threshold,
                    sampling_rate=16000,
                    min_silence_duration_ms=self._vad_min_silence_ms,
                )
                self._silero_enabled = True
                return

        try:
            resolved = _resolve_silero_vad_path()
            if resolved is None:
                raise FileNotFoundError(
                    "silero_vad.onnx not found under pretrained_models/. "
                    'Run: python -c "import shutil,pathlib,silero_vad as sv; '
                    "shutil.copy2(pathlib.Path(sv.__file__).parent/'data/silero_vad.onnx','pretrained_models/silero_vad.onnx')\""
                )
            self._silero_iterator = _SileroVADIterator(
                model_path=str(resolved),
                threshold=self._vad_start_threshold,
                end_threshold=self._vad_end_threshold,
                sampling_rate=16000,
                min_silence_duration_ms=self._vad_min_silence_ms,
            )
            # Sanity-check: feed a speech-like harmonic through the iterator.
            # A 150 Hz fundamental + harmonics mimics a vowel — a working
            # model should fire a {"start": …} event within a few frames.
            _t = np.arange(512, dtype=np.float32) / 16000
            _speech = (
                0.3 * np.sin(2 * np.pi * 150 * _t)
                + 0.2 * np.sin(2 * np.pi * 300 * _t)
                + 0.15 * np.sin(2 * np.pi * 450 * _t)
            ).astype(np.float32)
            _got_start = False
            for _ in range(5):
                ev = self._silero_iterator(_speech)
                if isinstance(ev, dict) and "start" in ev:
                    _got_start = True
                    break
            self._silero_iterator.reset_states()
            if not _got_start:
                raise RuntimeError(
                    "Silero ONNX sanity check failed — model did not detect "
                    "speech-like harmonic signal within 5 frames. "
                    "Falling back to energy-gate VAD."
                )
            self._silero_enabled = True
            logger.info(f"Silero VAD loaded from {resolved}")
            with self._shared_lock:
                self._shared_silero_vad_path = str(resolved)
                self._shared_silero_initialized = True
        except Exception as exc:
            self._silero_enabled = False
            with self._shared_lock:
                self._shared_silero_vad_path = None
                self._shared_silero_initialized = True
            logger.warning(f"Silero ONNX VAD unavailable, using energy gate fallback: {exc}")

    def _try_init_whisper_model(self) -> None:
        with self._shared_lock:
            if self._stt_model in self._shared_whisper_models:
                return

        try:
            from faster_whisper import WhisperModel

            local_dir = _resolve_whisper_model_dir(self._stt_model)
            if local_dir is not None:
                model = WhisperModel(str(local_dir), device="cpu", compute_type="int8")
                logger.info(f"faster-whisper loaded from local dir: {local_dir}")
            else:
                model = WhisperModel(self._stt_model, device="cpu", compute_type="int8")
                logger.info(f"faster-whisper loaded via HF Hub: {self._stt_model}")
            with self._shared_lock:
                self._shared_whisper_models[self._stt_model] = model
        except Exception as exc:
            logger.warning(f"faster-whisper model preload failed ({self._stt_model}): {exc}")

    def _detect_speech_confidence(self, frame_f32_16k: np.ndarray) -> float:
        if self._silero_enabled and self._silero_iterator is not None:
            try:
                event = self._silero_iterator(frame_f32_16k, return_seconds=False)
                if isinstance(event, dict):
                    if "start" in event:
                        self._vad_in_speech = True
                        return 1.0
                    if "end" in event:
                        self._vad_in_speech = False
                        return 0.0
                # VADIterator returns None inside a segment (or silence).
                # Return a stable value based on tracked state so we never
                # fall through to the noisy RMS gate during ongoing speech.
                return 0.9 if self._vad_in_speech else 0.05
            except Exception as exc:
                logger.warning(f"Silero VAD inference error: {exc}")
                self._vad_in_speech = False

        # Energy-based fallback when Silero is unavailable.
        rms = float(np.sqrt(np.mean(np.square(frame_f32_16k))))
        return min(1.0, rms * 14.0)

    # Minimum speech segment duration in samples @24kHz.
    # 500ms = 12000 samples — anything shorter is almost certainly noise or
    # a breath burst.  Raised from 300ms to reduce hallucination on very
    # short audio fragments where Whisper tends to invent text.
    _MIN_SPEECH_SAMPLES_24K = 12000

    def _transcribe_current_segment(self) -> str:
        if not self._speech_audio_24k:
            return ""

        # Filter 1: reject segments too short to be real speech
        num_samples = len(self._speech_audio_24k) // 2  # PCM16 = 2 bytes/sample
        if num_samples < self._MIN_SPEECH_SAMPLES_24K:
            duration_ms = num_samples * 1000 // 24000
            logger.debug(f"STT: rejecting segment too short ({duration_ms}ms < 500ms)")
            self._speech_audio_24k.clear()
            return ""

        audio_i16 = np.frombuffer(bytes(self._speech_audio_24k), dtype=np.int16)
        audio_f32_24k = audio_i16.astype(np.float32) / 32768.0
        audio_f32_16k = self._resample_24k_to_16k(audio_f32_24k)
        try:
            self._try_init_whisper_model()
            with self._shared_lock:
                cached_model = self._shared_whisper_models.get(self._stt_model)
            if cached_model is None:
                logger.error(f"Whisper model {self._stt_model!r} not available")
                return ""

            with self._shared_transcribe_lock:
                segments, info = cached_model.transcribe(
                    audio_f32_16k,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=200,
                        speech_pad_ms=100,
                    ),
                    no_speech_threshold=0.3,
                    log_prob_threshold=-0.7,
                    compression_ratio_threshold=2.0,
                    condition_on_previous_text=False,
                    initial_prompt=self._initial_prompt,
                    word_timestamps=True,
                    hallucination_silence_threshold=1.0,
                )
                good_parts: list[str] = []
                for seg in segments:
                    t = seg.text.strip()
                    if not t:
                        continue
                    if seg.compression_ratio > 2.4 or seg.avg_logprob < -1.0 or seg.no_speech_prob > 0.3:
                        logger.debug(
                            f"STT: dropping segment {t!r} "
                            f"(cr={seg.compression_ratio:.1f}, lp={seg.avg_logprob:.2f}, nsp={seg.no_speech_prob:.2f})"
                        )
                        continue
                    good_parts.append(t)
                text = " ".join(good_parts)

                if not text:
                    return ""

                # Filter 2: reject ultra-short text on short audio
                if len(text.strip()) < 2 and info.duration < 1.2:
                    logger.debug(
                        f"STT: rejecting short text {text!r} (len={len(text.strip())}, dur={info.duration:.1f}s)"
                    )
                    return ""

                logger.debug(
                    f"Transcription: {text!r} "
                    f"(duration={info.duration:.1f}s, lang_prob={info.language_probability:.2f})"
                )
                return text
        finally:
            self._speech_audio_24k.clear()
