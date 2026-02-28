"""TTS service (piper-tts, CPU-optimised)."""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels above src/services/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TTSService:
    """Piper TTS service using piper-tts VITS ONNX models.

    Loads a Piper ONNX model via ``PiperVoice.load()`` and synthesises
    speech via ``synthesize()``, which runs in a thread pool.
    Audio is resampled from Piper's native rate to 24000 Hz via scipy
    resample_poly (GCD-reduced ratio for minimal artefacts).
    """

    _shared_lock = threading.Lock()
    _shared_voices: dict[str, Any] = {}

    def __init__(
        self,
        model_dir: str = "./pretrained_models/piper",
        voice_path: str | None = None,
        voice_name: str | None = "en_US-hfc_female-medium",
        noise_scale: float = 0.75,
        noise_w_scale: float = 0.8,
        length_scale: float = 0.95,
    ) -> None:
        self._model_dir = self._resolve_model_dir(model_dir)
        self._voice_path = voice_path
        self._voice_name = voice_name or "en_US-hfc_female-medium"
        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale
        self._length_scale = length_scale

        self._voice: Any = None
        self._syn_config: Any = None  # SynthesisConfig, set after load
        self._native_sr: int = 22050  # overridden after load
        self._sample_rate: int = 24000  # output rate after resampling
        self._loaded = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise RuntimeError("piper-tts not installed. Run: uv sync --extra local_voice") from exc

        model_path = self._resolve_model_path()
        cache_key = str(model_path)

        with self._shared_lock:
            voice = self._shared_voices.get(cache_key)
            if voice is None:
                voice = PiperVoice.load(str(model_path))
                self._shared_voices[cache_key] = voice

        self._voice = voice
        self._native_sr = voice.config.sample_rate

        try:
            from piper.config import SynthesisConfig

            self._syn_config = SynthesisConfig(
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w_scale,
                length_scale=self._length_scale,
            )
        except ImportError:
            self._syn_config = None

        self._loaded = True

        logger.info(
            f"Piper TTS ready (model={model_path.name}, "
            f"native_sr={self._native_sr}, output_sr={self._sample_rate}, "
            f"noise_scale={self._noise_scale}, noise_w={self._noise_w_scale}, "
            f"length_scale={self._length_scale})"
        )

    # ── Path resolution ──────────────────────────────────────────────

    def _resolve_model_dir(self, model_dir: str) -> Path:
        raw = Path(model_dir)
        if raw.is_absolute():
            return raw.resolve()

        # Try CWD, project root, then src root — pick first with .onnx files.
        bases = [Path.cwd(), _PROJECT_ROOT, Path(__file__).resolve().parents[1]]
        for base in bases:
            candidate = (base / raw).resolve()
            if any(candidate.glob("*.onnx")):
                return candidate

        # Fallback: project-root-relative (will raise a clear error later)
        return (_PROJECT_ROOT / raw).resolve()

    def _resolve_model_path(self) -> Path:
        if self._voice_path:
            p = Path(self._voice_path)
            if not p.is_absolute():
                p = (self._model_dir / p).resolve()
            if p.exists():
                return p
            raise RuntimeError(f"Configured TTS voice path not found: {p}")

        model_path = self._model_dir / f"{self._voice_name}.onnx"
        if model_path.exists():
            return model_path

        # Fallback: first available .onnx in model dir
        for f in sorted(self._model_dir.glob("*.onnx")):
            logger.warning(f"Voice '{self._voice_name}.onnx' not found — using {f.name}")
            return f

        raise RuntimeError(f"No Piper ONNX model found under {self._model_dir}. Expected {self._voice_name}.onnx")

    # ── Synthesis ────────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        cancelled: asyncio.Event | None = None,
    ) -> AsyncIterator[bytes]:
        if not self._loaded or self._voice is None:
            raise RuntimeError("Piper TTS not loaded — call load() first")

        text = text.strip()
        if not text:
            return

        def _synthesize_sync() -> bytes:
            # Collect all Piper chunks for this utterance, then resample
            # the concatenated signal once. This avoids FIR filter
            # transients at chunk boundaries (resample_poly zero-pads
            # edges, causing clicks when applied per-chunk).
            # Piper typically yields 1 chunk per sentence so this adds
            # no latency in practice.
            float_parts: list[np.ndarray] = []
            for chunk in self._voice.synthesize(text, syn_config=self._syn_config):
                if cancelled is not None and cancelled.is_set():
                    break
                float_parts.append(chunk.audio_float_array)

            if not float_parts:
                return b""

            combined = np.concatenate(float_parts) if len(float_parts) > 1 else float_parts[0]
            return self._resample_to_pcm16(combined)

        try:
            pcm = await asyncio.to_thread(_synthesize_sync)
        except Exception as exc:
            logger.error(f"Piper TTS synthesis error: {exc}")
            return

        if not pcm or (cancelled is not None and cancelled.is_set()):
            return

        # Split the resampled PCM into ~100ms delivery chunks for smoother
        # streaming to the frontend.  The resampling was already done on the
        # full concatenated signal (no artefacts), so slicing the resulting
        # PCM16 bytes is safe.  100ms @ 24kHz = 2400 samples x 2 bytes.
        delivery_bytes = self._sample_rate // 10 * 2  # 4800 bytes = 100ms
        offset = 0
        while offset < len(pcm):
            if cancelled is not None and cancelled.is_set():
                return
            end = min(offset + delivery_bytes, len(pcm))
            yield pcm[offset:end]
            offset = end

    def _resample_to_pcm16(self, arr: np.ndarray) -> bytes:
        """Resample float32 audio from native rate to output rate, return PCM16."""
        if arr.size == 0:
            return b""
        if self._native_sr != self._sample_rate:
            from scipy.signal import resample_poly

            gcd = math.gcd(self._native_sr, self._sample_rate)
            arr = resample_poly(
                arr,
                self._sample_rate // gcd,
                self._native_sr // gcd,
            ).astype(np.float32)
        return (arr * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
