"""
OpenClaw Agent Package

Provides the sample OpenClaw agent implementation using the OpenClaw
HTTP gateway's OpenAI-compatible /v1/chat/completions endpoint.
"""

from .openclaw_settings import OpenClawSettings, get_openclaw_settings
from .sample_agent import SampleOpenClawAgent

__all__ = [
    "OpenClawSettings",
    "SampleOpenClawAgent",
    "get_openclaw_settings",
]
