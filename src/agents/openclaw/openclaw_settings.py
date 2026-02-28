"""
OpenClaw Agent Configuration

Settings for the OpenClaw agent with local ONNX STT / TTS.

OpenClaw exposes an OpenAI-compatible HTTP gateway at /v1/chat/completions.
Since the avatar-chat-server typically coexists on the same machine as openclawd,
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

    # ── OpenClaw Gateway ────────────────────────────────────────────
    # Default: loopback on same machine as openclawd (port 19001)
    openclaw_base_url: str = "http://127.0.0.1:19001"

    # Authentication token (must match gateway.auth.token in openclaw.json)
    openclaw_api_token: str = ""

    # Model/agent identifier
    # "openclaw:main" = default agent, "openclaw:<agentId>" = specific agent
    openclaw_model: str = "openclaw:main"

    # Optional: stable user ID for session persistence across requests
    openclaw_user_id: str | None = None

    # Optional: explicit session key for session routing
    openclaw_session_key: str | None = None

    # Optional: agent ID override (sent via x-openclaw-agent-id header)
    openclaw_agent_id: str | None = None

    # Response thinking mode hint for lower latency / shorter responses.
    # Supported values: "off", "minimal", "default"
    openclaw_thinking_mode: str = "minimal"

    # Maximum number of messages kept in local conversation history
    # (includes the leading system prompt when present).
    openclaw_history_max_messages: int = 12

    # Transcript timing (chars per second)
    openclaw_transcript_speed: float = 14.0

    # HTTP client configuration (tuned for loopback)
    openclaw_connect_timeout: float = 5.0  # Connection timeout (seconds)
    openclaw_read_timeout: float = 120.0  # Read timeout for SSE stream (seconds)
    openclaw_max_retries: int = 2  # Max retries on transient failures

    # ── STT: faster-whisper + Silero VAD ────────────────────────────
    # Enable server-side STT.
    # When disabled, append_audio() is a no-op (client must handle STT).
    stt_enabled: bool = True

    # STT model (faster-whisper model name or local dir)
    stt_model: str = "small.en"
    stt_vad_start_threshold: float = 0.60
    stt_vad_end_threshold: float = 0.35
    stt_vad_min_silence_ms: int = 280
    stt_initial_prompt: str | None = None

    # ── TTS: Piper VITS ONNX ────────────────────────────────────────
    # Enable server-side TTS.
    # When disabled, only transcript deltas are emitted (client handles TTS).
    tts_enabled: bool = True

    # Path to a WAV file for voice cloning (optional, slower first load)
    tts_voice_path: str | None = None

    # Piper voice name (e.g. en_US-hfc_female-medium, en_US-lessac-medium)
    tts_voice_name: str | None = "en_US-hfc_female-medium"

    # ONNX TTS settings
    tts_onnx_model_dir: str = "./pretrained_models/piper"
    # VITS synthesis knobs (Piper)
    tts_noise_scale: float = 0.75  # Audio variation (0=flat, 1=expressive)
    tts_noise_w_scale: float = 0.8  # Phoneme duration variation (0=robotic, 1=natural)
    tts_length_scale: float = 0.95  # Speech speed (<1=faster, >1=slower)

    # Maximum characters to buffer before forcing a TTS chunk
    # (even without sentence-ending punctuation)
    tts_sentence_max_chars: int = 200


@lru_cache
def get_openclaw_settings() -> OpenClawSettings:
    """Get cached OpenClaw settings."""
    return OpenClawSettings()
