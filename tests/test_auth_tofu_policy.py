"""Tests for registration and TOFU policy behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.challenge_service import ChallengeService
from auth.models import AuthChallengeRequest, AuthClientInfo, AuthRegisterRequest
from auth.store import InMemoryAuthStore
from core.settings import Settings
from routers.auth_router import challenge, register_device


def _settings(policy: str) -> Settings:
    return Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_secret_key="test-auth-secret",
        jwt_signing_key="test-jwt-key",
        auth_registration_policy=policy,
    )


def _client() -> AuthClientInfo:
    return AuthClientInfo(platform="android", appVersion="test", deviceModel="pixel")


@pytest.mark.asyncio
async def test_register_open_policy_returns_pending_approval() -> None:
    store = InMemoryAuthStore()
    response = await register_device(
        payload=AuthRegisterRequest(
            deviceId="device-open",
            publicKey="pk-open",
            algorithm="ed25519",
            client=_client(),
        ),
        settings=_settings("open"),
        store=store,
    )

    assert response.status == "pending_approval"
    record = store.get_device("device-open")
    assert record is not None
    assert record.status == "pending"


@pytest.mark.asyncio
async def test_register_invite_policy_returns_pending_approval() -> None:
    store = InMemoryAuthStore()
    response = await register_device(
        payload=AuthRegisterRequest(
            deviceId="device-invite",
            publicKey="pk-invite",
            algorithm="ed25519",
            client=_client(),
        ),
        settings=_settings("invite"),
        store=store,
    )

    assert response.status == "pending_approval"
    record = store.get_device("device-invite")
    assert record is not None
    assert record.status == "pending"


@pytest.mark.asyncio
async def test_register_admin_approval_policy_returns_pending_approval() -> None:
    store = InMemoryAuthStore()
    response = await register_device(
        payload=AuthRegisterRequest(
            deviceId="device-admin",
            publicKey="pk-admin",
            algorithm="ed25519",
            client=_client(),
        ),
        settings=_settings("admin-approval"),
        store=store,
    )

    assert response.status == "pending_approval"
    record = store.get_device("device-admin")
    assert record is not None
    assert record.status == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["invite", "admin-approval"])
async def test_challenge_unknown_device_denied_when_policy_not_open(policy: str) -> None:
    store = InMemoryAuthStore()
    response = await challenge(
        payload=AuthChallengeRequest(deviceId="missing", publicKey="pk", client=_client()),
        settings=_settings(policy),
        store=store,
        challenge_service=ChallengeService(store=store, challenge_ttl_sec=120),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert b'"code":"device_not_approved"' in response.body


@pytest.mark.asyncio
async def test_challenge_unknown_device_creates_pending_when_policy_open() -> None:
    store = InMemoryAuthStore()
    response = await challenge(
        payload=AuthChallengeRequest(deviceId="new-open", publicKey="pk-open", client=_client()),
        settings=_settings("open"),
        store=store,
        challenge_service=ChallengeService(store=store, challenge_ttl_sec=120),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert b'"code":"device_not_approved"' in response.body
    record = store.get_device("new-open")
    assert record is not None
    assert record.status == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["open", "invite", "admin-approval"])
async def test_register_policy_keeps_devices_pending_until_approved(policy: str) -> None:
    store = InMemoryAuthStore()
    response = await register_device(
        payload=AuthRegisterRequest(
            deviceId=f"device-{policy}",
            publicKey=f"pk-{policy}",
            algorithm="ed25519",
            client=_client(),
        ),
        settings=_settings(policy),
        store=store,
    )

    assert response.status == "pending_approval"
    record = store.get_device(f"device-{policy}")
    assert record is not None
    assert record.status == "pending"


@pytest.mark.asyncio
async def test_challenge_pending_device_is_denied() -> None:
    store = InMemoryAuthStore()
    await register_device(
        payload=AuthRegisterRequest(
            deviceId="pending-device",
            publicKey="pk-pending",
            algorithm="ed25519",
            client=_client(),
        ),
        settings=_settings("admin-approval"),
        store=store,
    )

    response = await challenge(
        payload=AuthChallengeRequest(deviceId="pending-device", publicKey="pk-pending", client=_client()),
        settings=_settings("admin-approval"),
        store=store,
        challenge_service=ChallengeService(store=store, challenge_ttl_sec=120),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert b'"code":"device_not_approved"' in response.body
