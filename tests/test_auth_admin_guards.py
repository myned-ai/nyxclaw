"""HTTP-level tests for admin guards on destructive auth endpoints."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import DeviceRecord, SessionRecord, utc_now
from auth.store import InMemoryAuthStore, get_auth_store
from core.settings import Settings, get_settings
from routers.auth_router import auth_router


@pytest.fixture
def app_and_store() -> tuple[FastAPI, InMemoryAuthStore]:
    app = FastAPI()
    app.include_router(auth_router)
    store = InMemoryAuthStore()

    settings = Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_secret_key="test-auth-secret",
        jwt_signing_key="test-jwt-key",
        auth_admin_token="admin-secret",
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_auth_store] = lambda: store
    return app, store


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/auth/revoke-device", {"deviceId": "device-1"}),
        ("/api/auth/revoke-session", {"sessionId": "session-1"}),
        ("/api/auth/approve-device", {"deviceId": "device-1"}),
        ("/api/auth/reject-device", {"deviceId": "device-1"}),
    ],
)
def test_destructive_auth_endpoints_require_admin_token(
    app_and_store: tuple[FastAPI, InMemoryAuthStore],
    path: str,
    payload: dict[str, str],
) -> None:
    app, _ = app_and_store
    client = TestClient(app)

    response = client.post(path, json=payload)

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_destructive_auth_endpoints_accept_valid_admin_token(
    app_and_store: tuple[FastAPI, InMemoryAuthStore],
) -> None:
    app, store = app_and_store
    client = TestClient(app)

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
            session_id="session-1",
            device_id="device-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=5),
            refresh_jti="j1",
            refresh_token_hash="h1",
            revoked=False,
        )
    )

    headers = {"x-auth-admin-token": "admin-secret"}

    assert client.post("/api/auth/revoke-device", json={"deviceId": "device-1"}, headers=headers).status_code == 200
    assert client.post("/api/auth/revoke-session", json={"sessionId": "session-1"}, headers=headers).status_code in {
        200,
        404,
    }
    assert client.post("/api/auth/approve-device", json={"deviceId": "device-1"}, headers=headers).status_code == 200
    assert client.post("/api/auth/reject-device", json={"deviceId": "device-1"}, headers=headers).status_code == 200
