"""
ZeroClaw Agent Configuration

Settings for the ZeroClaw agent with local STT/TTS support.
ZeroClaw chat uses a WebSocket endpoint at `/ws/chat`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent.parent.parent.parent


class ZeroClawSettings(BaseSettings):
    """ZeroClaw-specific settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    zeroclaw_base_url: str = "http://127.0.0.1:5555"
    zeroclaw_ws_token: str | None = None
    zeroclaw_model: str = "zeroclaw:main"
    zeroclaw_user_id: str | None = None
    zeroclaw_thinking_mode: str = "minimal"
    zeroclaw_history_max_messages: int = 12
    zeroclaw_transcript_speed: float = 14.0
    zeroclaw_connect_timeout: float = 5.0
    zeroclaw_read_timeout: float = 120.0

    stt_enabled: bool = True
    stt_model: str = "small.en"
    stt_vad_start_threshold: float = 0.60
    stt_vad_end_threshold: float = 0.35
    stt_vad_min_silence_ms: int = 280
    stt_initial_prompt: str | None = "Hi. Hello. Hey. Bye. Yes. No. Okay."

    tts_enabled: bool = True
    tts_voice_path: str | None = None
    tts_voice_name: str | None = "en_US-hfc_female-medium"
    tts_onnx_model_dir: str = "./pretrained_models/piper"
    tts_sentence_max_chars: int = 200
    # VITS synthesis knobs (Piper)
    tts_noise_scale: float = 0.75  # Audio variation (0=flat, 1=expressive)
    tts_noise_w_scale: float = 0.8  # Phoneme duration variation (0=robotic, 1=natural)
    tts_length_scale: float = 0.95  # Speech speed (<1=faster, >1=slower)


@lru_cache
def get_zeroclaw_settings() -> ZeroClawSettings:
    """Get cached ZeroClaw settings."""
    return ZeroClawSettings()
