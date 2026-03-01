"""
Backend Package

Provides modular backend implementations for Claw-based conversational AI.

Backends:
- OpenClawBackend: Uses OpenClaw HTTP Gateway (text-only, SSE streaming)
- ZeroClawBackend: Uses ZeroClaw WebSocket Gateway
"""

from .base_agent import BaseAgent, ConversationState
from .openclaw import OpenClawBackend
from .zeroclaw import ZeroClawBackend

__all__ = [
    "BaseAgent",
    "ConversationState",
    "OpenClawBackend",
    "ZeroClawBackend",
]
