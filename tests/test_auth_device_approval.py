"""Tests for admin approval workflow endpoints."""

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
    AuthApproveDeviceRequest,
    AuthRejectDeviceRequest,
    DeviceRecord,
    SessionRecord,
    utc_now,
)
from auth.store import InMemoryAuthStore
from core.settings import Settings
from routers.auth_router import approve_device, reject_device


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
async def test_approve_device_sets_device_active() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="device-1",
            public_key="pk",
            algorithm="ed25519",
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )

    response = await approve_device(
        payload=AuthApproveDeviceRequest(deviceId="device-1"),
        request=_request(),
        settings=_settings(),
        store=store,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert store.get_device("device-1").status == "active"


@pytest.mark.asyncio
async def test_reject_device_sets_rejected_and_revokes_sessions() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="device-1",
            public_key="pk",
            algorithm="ed25519",
            status="pending",
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

    response = await reject_device(
        payload=AuthRejectDeviceRequest(deviceId="device-1"),
        request=_request(),
        settings=_settings(),
        store=store,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert b'"revokedSessions":1' in response.body
    assert store.get_device("device-1").status == "rejected"
    assert store.get_session("s1").revoked is True


@pytest.mark.asyncio
async def test_approve_device_missing_returns_not_found() -> None:
    response = await approve_device(
        payload=AuthApproveDeviceRequest(deviceId="missing"),
        request=_request(),
        settings=_settings(),
        store=InMemoryAuthStore(),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 404
    assert b'"code":"device_not_found"' in response.body


@pytest.mark.asyncio
async def test_approve_device_requires_admin_token() -> None:
    response = await approve_device(
        payload=AuthApproveDeviceRequest(deviceId="device-1"),
        request=_request(admin_token="wrong-secret"),
        settings=_settings(),
        store=InMemoryAuthStore(),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert b'"code":"forbidden"' in response.body
