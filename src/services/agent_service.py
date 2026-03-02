"""
Agent Service

Provides agent instances based on configuration.
"""

from backend import BaseAgent, OpenClawBackend, ZeroClawBackend
from core.settings import get_settings
from voice.openai_realtime import OpenAIRealtimeBackend


def create_agent_instance() -> BaseAgent:
    """
    Create a new agent instance based on configuration.

    VOICE_MODE selects the voice pipeline:
      - "openai"  → OpenAI Realtime (VAD+STT) + OpenAI TTS API
      - "local"   → local Silero VAD + faster-whisper STT + Piper TTS

    AGENT_TYPE selects the LLM backend (openclaw / zeroclaw).
    When voice_mode is "openai", the OpenAIRealtimeBackend makes its own
    HTTP SSE calls using BASE_URL / AUTH_TOKEN / AGENT_MODEL from .env.
    """
    settings = get_settings()

    if settings.voice_mode == "openai":
        return OpenAIRealtimeBackend()

    agent_type = settings.agent_type
    if agent_type == "openclaw":
        return OpenClawBackend()
    elif agent_type == "zeroclaw":
        return ZeroClawBackend()
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Supported: openclaw, zeroclaw")
