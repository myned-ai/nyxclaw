"""
Agent Service

Provides agent instances based on configuration.
"""

from backend import (
    BaseAgent,
    OpenClawBackend,
    ZeroClawBackend,
)
from core.settings import get_settings


def create_agent_instance() -> BaseAgent:
    """
    Create a new agent instance based on configuration.
    Used for creating a fresh agent for each session.

    Returns:
        New Agent instance
    """
    settings = get_settings()
    agent_type = settings.agent_type

    if agent_type == "openclaw":
        return OpenClawBackend()
    elif agent_type == "zeroclaw":
        return ZeroClawBackend()
    else:
        raise ValueError(
            f"Unknown agent_type: {agent_type}. "
            "Supported: openclaw, zeroclaw"
        )
