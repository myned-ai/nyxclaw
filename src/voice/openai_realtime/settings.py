"""
OpenAI Voice Configuration

Settings for the OpenAI voice layer: Realtime API (VAD + STT) and TTS API.
The LLM backend (OpenClaw / ZeroClaw) is configured via base_url / auth_token / agent_model.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent.parent.parent.parent


class OpenAIRealtimeSettings(BaseSettings):
    """OpenAI voice layer settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI API ─────────────────────────────────────────────────
    openai_api_key: str = ""

    # ── Realtime API (VAD + STT) ───────────────────────────────────
    openai_realtime_model: str = "gpt-realtime"
    openai_vad_type: str = "semantic_vad"  # "semantic_vad" or "server_vad"
    openai_transcription_model: str = "gpt-4o-transcribe"
    openai_transcription_language: str = "en"

    # ── TTS API ────────────────────────────────────────────────────
    openai_tts_model: str = "tts-1"
    openai_voice: str = "alloy"
    openai_tts_speed: float = 1.0  # 0.25 to 4.0

    # ── LLM Backend (OpenClaw HTTP SSE / ZeroClaw WebSocket) ───────
    base_url: str = "http://127.0.0.1:19001"
    auth_token: str = ""
    agent_model: str = "openclaw:main"
    user_id: str | None = None
    thinking_mode: str = "minimal"
    history_max_messages: int = 12
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    session_key: str | None = None
    agent_id: str | None = None
    max_retries: int = 2

    # ── Sentence / Transcript ──────────────────────────────────────
    tts_sentence_max_chars: int = 200
    transcript_speed: float = 20.0  # chars/sec


@lru_cache
def get_openai_realtime_settings() -> OpenAIRealtimeSettings:
    """Get cached OpenAI Realtime settings."""
    return OpenAIRealtimeSettings()
