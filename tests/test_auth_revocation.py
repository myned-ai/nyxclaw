"""Tests for auth device/session revocation endpoints."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import (
    AuthRevokeDeviceRequest,
    AuthRevokeSessionRequest,
    DeviceRecord,
    SessionRecord,
    utc_now,
)
from auth.store import InMemoryAuthStore
from core.settings import Settings
from routers.auth_router import revoke_device, revoke_session


def _settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_secret_key="test-auth-secret",
        jwt_signing_key="test-jwt-key",
        auth_admin_token="admin-secret",
    )


def _request(admin_token: str = "admin-secret") -> SimpleNamespace:
    return SimpleNamespace(headers={"x-auth-admin-token": admin_token})


@pytest.mark.asyncio
async def test_revoke_device_revokes_device_and_active_sessions() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="device-1",
            public_key="pk",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    store.save_session(
        SessionRecord(
            session_id="s1",
            device_id="device-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=5),
            refresh_jti="j1",
            refresh_token_hash="h1",
            revoked=False,
        )
    )
    store.save_session(
        SessionRecord(
            session_id="s2",
            device_id="device-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=5),
            refresh_jti="j2",
            refresh_token_hash="h2",
            revoked=False,
        )
    )

    result = await revoke_device(
        payload=AuthRevokeDeviceRequest(deviceId="device-1"),
        request=_request(),
        settings=_settings(),
        store=store,
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200
    assert b'"revokedSessions":2' in result.body

    device = store.get_device("device-1")
    assert device is not None
    assert device.status == "revoked"
    assert store.get_session("s1").revoked is True
    assert store.get_session("s2").revoked is True


@pytest.mark.asyncio
async def test_revoke_session_revokes_single_session() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.save_session(
        SessionRecord(
            session_id="s1",
            device_id="device-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=5),
            refresh_jti="j1",
            refresh_token_hash="h1",
            revoked=False,
        )
    )

    result = await revoke_session(
        payload=AuthRevokeSessionRequest(sessionId="s1"),
        request=_request(),
        settings=_settings(),
        store=store,
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200
    assert b'"sessionId":"s1"' in result.body
    assert store.get_session("s1").revoked is True


@pytest.mark.asyncio
async def test_revoke_device_returns_not_found_for_unknown_device() -> None:
    result = await revoke_device(
        payload=AuthRevokeDeviceRequest(deviceId="missing"),
        request=_request(),
        settings=_settings(),
        store=InMemoryAuthStore(),
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 404
    assert b'"code":"device_not_found"' in result.body


@pytest.mark.asyncio
async def test_revoke_session_returns_not_found_for_unknown_session() -> None:
    result = await revoke_session(
        payload=AuthRevokeSessionRequest(sessionId="missing"),
        request=_request(),
        settings=_settings(),
        store=InMemoryAuthStore(),
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 404
    assert b'"code":"session_not_found"' in result.body


@pytest.mark.asyncio
async def test_revoke_device_requires_admin_token() -> None:
    result = await revoke_device(
        payload=AuthRevokeDeviceRequest(deviceId="device-1"),
        request=_request(admin_token="wrong-secret"),
        settings=_settings(),
        store=InMemoryAuthStore(),
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    assert b'"code":"forbidden"' in result.body
