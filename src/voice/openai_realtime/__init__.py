"""
OpenAI Realtime Voice Package

Server-side VAD + STT via OpenAI Realtime API, TTS via OpenAI TTS API.
"""

from .backend import OpenAIRealtimeBackend
from .settings import OpenAIRealtimeSettings, get_openai_realtime_settings

__all__ = [
    "OpenAIRealtimeBackend",
    "OpenAIRealtimeSettings",
    "get_openai_realtime_settings",
]
