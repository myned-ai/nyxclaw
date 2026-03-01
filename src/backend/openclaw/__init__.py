"""
OpenClaw Backend Package

Provides the OpenClaw backend implementation using the OpenClaw
HTTP gateway's OpenAI-compatible /v1/chat/completions endpoint.
"""

from .backend import OpenClawBackend
from .settings import OpenClawSettings, get_openclaw_settings

__all__ = [
    "OpenClawBackend",
    "OpenClawSettings",
    "get_openclaw_settings",
]
