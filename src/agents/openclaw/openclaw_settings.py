"""
OpenClaw Agent Configuration

Settings for the OpenClaw agent with local ONNX STT / TTS.

OpenClaw exposes an OpenAI-compatible HTTP gateway at /v1/chat/completions.
Since NyxClaw typically coexists on the same machine as OpenClaw,
the default endpoint is http://127.0.0.1:19001 (loopback, no TLS overhead).

STT : faster-whisper (CTranslate2, int8) + Silero VAD
TTS : Pocket TTS ONNX
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory for .env file (project root)
_CONFIG_DIR = Path(__file__).parent.parent.parent.parent


class OpenClawSettings(BaseSettings):
    """
    OpenClaw-specific settings.

    Loaded from environment variables when the OpenClaw agent is used.
    All STT / TTS settings have sensible defaults for local-machine deployment.
    """

    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Agent Backend ─────────────────────────────────────────────
    base_url: str = "http://127.0.0.1:19001"
    auth_token: str = ""
    agent_model: str = "openclaw:main"
    user_id: str | None = None
    thinking_mode: str = "minimal"
    history_max_messages: int = 12
    transcript_speed: float = 14.0
    connect_timeout: float = 5.0
    read_timeout: float = 120.0

    # OpenClaw-specific
    session_key: str | None = None
    agent_id: str | None = None
    max_retries: int = 2

    # ── STT: faster-whisper + Silero VAD ────────────────────────────
    stt_enabled: bool = True
    stt_model: str = "small.en"
    stt_vad_start_threshold: float = 0.60
    stt_vad_end_threshold: float = 0.35
    stt_vad_min_silence_ms: int = 280
    stt_initial_prompt: str | None = None

    # ── TTS: Piper VITS ONNX ────────────────────────────────────────
    tts_enabled: bool = True
    tts_voice_path: str | None = None
    tts_voice_name: str | None = "en_US-hfc_female-medium"
    tts_onnx_model_dir: str = "./pretrained_models/piper"
    tts_noise_scale: float = 0.75
    tts_noise_w_scale: float = 0.8
    tts_length_scale: float = 0.95
    tts_sentence_max_chars: int = 200


@lru_cache
def get_openclaw_settings() -> OpenClawSettings:
    """Get cached OpenClaw settings."""
    return OpenClawSettings()
