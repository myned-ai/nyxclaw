"""Tests for challenge contextual metadata binding."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.challenge_service import ChallengeService
from auth.store import InMemoryAuthStore


def test_issue_challenge_binds_optional_client_context() -> None:
    service = ChallengeService(store=InMemoryAuthStore(), challenge_ttl_sec=120)

    challenge = service.issue_challenge(
        device_id="device-1",
        public_key="pk-1",
        algorithm="ed25519",
        client_platform="android",
        client_app_version="1.2.3",
        client_device_model="Pixel 8",
        client_ip="10.0.0.5",
        user_agent="nyxclaw-test-agent",
    )

    assert challenge.client_platform == "android"
    assert challenge.client_app_version == "1.2.3"
    assert challenge.client_device_model == "Pixel 8"
    assert challenge.client_ip == "10.0.0.5"
    assert challenge.user_agent == "nyxclaw-test-agent"
