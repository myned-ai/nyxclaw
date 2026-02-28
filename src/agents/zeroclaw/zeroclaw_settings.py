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

    # ── Agent Backend ─────────────────────────────────────────────
    base_url: str = "http://127.0.0.1:5555"
    auth_token: str | None = None
    agent_model: str = "zeroclaw:main"
    user_id: str | None = None
    thinking_mode: str = "minimal"
    history_max_messages: int = 12
    transcript_speed: float = 14.0
    connect_timeout: float = 5.0
    read_timeout: float = 120.0

    # ── STT ───────────────────────────────────────────────────────
    stt_enabled: bool = True
    stt_model: str = "small.en"
    stt_vad_start_threshold: float = 0.60
    stt_vad_end_threshold: float = 0.35
    stt_vad_min_silence_ms: int = 280
    stt_initial_prompt: str | None = "Hi. Hello. Hey. Bye. Yes. No. Okay."

    # ── TTS ───────────────────────────────────────────────────────
    tts_enabled: bool = True
    tts_voice_path: str | None = None
    tts_voice_name: str | None = "en_US-hfc_female-medium"
    tts_onnx_model_dir: str = "./pretrained_models/piper"
    tts_sentence_max_chars: int = 200
    tts_noise_scale: float = 0.75
    tts_noise_w_scale: float = 0.8
    tts_length_scale: float = 0.95


@lru_cache
def get_zeroclaw_settings() -> ZeroClawSettings:
    """Get cached ZeroClaw settings."""
    return ZeroClawSettings()
