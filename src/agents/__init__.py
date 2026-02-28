"""
Agents Package

Provides modular agent implementations for conversational AI.
Supports both local sample agents and remote (inter-container) agents.

Sample Agents:
- SampleOpenAIAgent: Uses OpenAI Realtime API
- SampleGeminiAgent: Uses Google Gemini Live API
- SampleOpenClawAgent: Uses OpenClaw HTTP Gateway (text-only, SSE streaming)

Remote Agent:
- RemoteAgent: Connects to external agent services via WebSocket
"""

from .base_agent import BaseAgent, ConversationState
from .gemini import SampleGeminiAgent
from .openclaw import SampleOpenClawAgent
from .openai import SampleOpenAIAgent
from .remote_agent import RemoteAgent
from .zeroclaw import SampleZeroClawAgent

__all__ = [
    "BaseAgent",
    "ConversationState",
    "RemoteAgent",
    "SampleGeminiAgent",
    "SampleOpenAIAgent",
    "SampleOpenClawAgent",
    "SampleZeroClawAgent",
]
