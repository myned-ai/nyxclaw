"""
Agents Package

Provides modular agent implementations for Claw-based conversational AI.

Agents:
- SampleOpenClawAgent: Uses OpenClaw HTTP Gateway (text-only, SSE streaming)
- SampleZeroClawAgent: Uses ZeroClaw WebSocket Gateway
"""

from .base_agent import BaseAgent, ConversationState
from .openclaw import SampleOpenClawAgent
from .zeroclaw import SampleZeroClawAgent

__all__ = [
    "BaseAgent",
    "ConversationState",
    "SampleOpenClawAgent",
    "SampleZeroClawAgent",
]
