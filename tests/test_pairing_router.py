"""Tests for pairing router approval/token behavior."""

from __future__ import annotations

import sys
import importlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import DeviceRecord, DeviceTokenRecord, PairingRequestRecord, SessionRecord, utc_now
from auth.store import InMemoryAuthStore
from core.settings import Settings
from routers.pairing_router import (
    PairingApproveRequest,
    PairingRejectRequest,
    PairingRevokeDeviceRequest,
    approve_pairing_request,
    reject_pairing_request,
    revoke_approved_device,
)

pairing_router_module = importlib.import_module("routers.pairing_router")


def _settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_admin_token="admin-secret",
        jwt_signing_key="jwt-secret",
    )


class _DummyRequest:
    def __init__(self, *, admin_token: str = "") -> None:
        self.headers = {
            "x-auth-admin-token": admin_token,
            "x-forwarded-for": "10.0.0.1",
        }
        self.client = SimpleNamespace(host="127.0.0.1")


@pytest.mark.asyncio
async def test_approve_pairing_request_issues_and_stores_device_token() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-1",
            device_id="dev-1",
            public_key_fingerprint="sha256:abc",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
            platform="android",
            app_version="1.0.0",
        )
    )

    response = await approve_pairing_request(
        payload=PairingApproveRequest(requestId="pr-1"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert b'"deviceToken"' in response.body

    token_record = store.get_device_token("dev-1", "operator")
    assert token_record is not None
    assert token_record.revoked is False
    assert token_record.expires_at > now


@pytest.mark.asyncio
async def test_reject_pairing_request_marks_device_rejected() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk",
            algorithm="ed25519",
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-2",
            device_id="dev-1",
            public_key_fingerprint="sha256:abc",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )

    response = await reject_pairing_request(
        payload=PairingRejectRequest(requestId="pr-2", reason="unknown device"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert store.get_pairing_request("pr-2").status == "rejected"
    assert store.get_device("dev-1").status == "rejected"


@pytest.mark.asyncio
async def test_revoke_approved_device_revokes_session_and_device_token() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-3",
            device_id="dev-1",
            public_key_fingerprint="sha256:abc",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )
    await approve_pairing_request(
        payload=PairingApproveRequest(requestId="pr-3"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    store.save_session(
        SessionRecord(
            session_id="s1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="j1",
            refresh_token_hash="h1",
            revoked=False,
        )
    )
    store.save_device_token(
        DeviceTokenRecord(
            device_id="dev-1",
            role="operator",
            token_hash="hash-1",
            issued_at=now,
            expires_at=now + timedelta(days=10),
            revoked=False,
        )
    )

    response = await revoke_approved_device(
        payload=PairingRevokeDeviceRequest(deviceId="dev-1", reason="lost"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert store.get_device("dev-1").status == "revoked"
    assert store.get_session("s1").revoked is True
    assert store.get_device_token("dev-1", "operator").revoked is True


@pytest.mark.asyncio
async def test_pairing_router_writes_audit_events_for_approve_reject_and_revoke() -> None:
    store = InMemoryAuthStore()
    now = utc_now()

    # Approve path audit event.
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-audit-approve",
            device_id="dev-audit-approve",
            public_key_fingerprint="sha256:approve",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )
    await approve_pairing_request(
        payload=PairingApproveRequest(requestId="pr-audit-approve"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    # Reject path audit event.
    store.upsert_device(
        DeviceRecord(
            device_id="dev-audit-reject",
            public_key="pk",
            algorithm="ed25519",
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-audit-reject",
            device_id="dev-audit-reject",
            public_key_fingerprint="sha256:reject",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )
    await reject_pairing_request(
        payload=PairingRejectRequest(requestId="pr-audit-reject", reason="not trusted"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    # Revoke path audit event.
    store.upsert_device(
        DeviceRecord(
            device_id="dev-audit-revoke",
            public_key="pk",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-audit-revoke",
            device_id="dev-audit-revoke",
            public_key_fingerprint="sha256:revoke",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )
    await approve_pairing_request(
        payload=PairingApproveRequest(requestId="pr-audit-revoke"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )
    await revoke_approved_device(
        payload=PairingRevokeDeviceRequest(deviceId="dev-audit-revoke", reason="lost"),
        request=_DummyRequest(admin_token="admin-secret"),
        settings=_settings(),
        store=store,
    )

    event_types = [event.event_type for event in store.list_pairing_audit_events(limit=20)]
    assert "pairing.approved" in event_types
    assert "pairing.rejected" in event_types
    assert "pairing.revoked" in event_types


@pytest.mark.asyncio
async def test_pairing_approve_rate_limited_returns_429_and_keeps_request_pending() -> None:
    store = InMemoryAuthStore()
    now = utc_now()
    store.save_pairing_request(
        PairingRequestRecord(
            request_id="pr-rate-limit",
            device_id="dev-rate-limit",
            public_key_fingerprint="sha256:rate",
            role="operator",
            requested_scopes=["chat.read"],
            status="pending",
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
    )

    original_get_auth_middleware = pairing_router_module.get_auth_middleware

    class _BlockedAuthMiddleware:
        def check_endpoint_rate_limit(self, endpoint: str, *key_parts: str) -> tuple[bool, str | None]:
            return False, "blocked"

    pairing_router_module.get_auth_middleware = lambda: _BlockedAuthMiddleware()
    try:
        response = await approve_pairing_request(
            payload=PairingApproveRequest(requestId="pr-rate-limit"),
            request=_DummyRequest(admin_token="admin-secret"),
            settings=_settings(),
            store=store,
        )
    finally:
        pairing_router_module.get_auth_middleware = original_get_auth_middleware

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert b'"code":"rate_limited"' in response.body
    assert store.get_pairing_request("pr-rate-limit").status == "pending"
