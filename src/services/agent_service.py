"""
Agent Service

Provides agent instances based on configuration.
"""

from agents import (
    BaseAgent,
    SampleOpenClawAgent,
    SampleZeroClawAgent,
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

    if agent_type == "sample_openclaw":
        return SampleOpenClawAgent()
    elif agent_type == "sample_zeroclaw":
        return SampleZeroClawAgent()
    else:
        raise ValueError(
            f"Unknown agent_type: {agent_type}. "
            "Supported: sample_openclaw, sample_zeroclaw"
        )
