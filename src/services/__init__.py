"""
Services Package

Contains service layer classes for:
- Wav2Arkit blendshape inference
- Agent management (factory pattern)
- STT: moshi-server WebSocket client  (server-side speech-to-text)
- TTS: Pocket TTS wrapper             (server-side text-to-speech)
"""

from services.agent_service import create_agent_instance
from services.wav2arkit_service import Wav2ArkitService, get_wav2arkit_service

__all__ = [
    "Wav2ArkitService",
    "create_agent_instance",
    "get_wav2arkit_service",
    # STT / TTS imported lazily in the OpenClaw agent to avoid
    # hard dependency when using other agents:
    #   from services.stt_service import STTService
    #   from services.tts_service import TTSService
]
