"""Targeted tests for Phase 2 auth services and websocket gating."""

from __future__ import annotations

import base64
import sys
import importlib.util
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.challenge_service import ChallengeService
from auth.jwt_service import JWTError, JWTService
from auth.models import ApprovedDeviceRecord, DeviceRecord, DeviceTokenRecord, SessionRecord, utc_now
from auth.store import InMemoryAuthStore

_CHAT_ROUTER_PATH = Path(__file__).resolve().parents[1] / "src" / "routers" / "chat_router.py"
_CHAT_ROUTER_SPEC = importlib.util.spec_from_file_location("chat_router_module", _CHAT_ROUTER_PATH)
if _CHAT_ROUTER_SPEC is None or _CHAT_ROUTER_SPEC.loader is None:
    raise RuntimeError("Failed to load chat_router module for tests")
_CHAT_ROUTER_MODULE = importlib.util.module_from_spec(_CHAT_ROUTER_SPEC)
_CHAT_ROUTER_SPEC.loader.exec_module(_CHAT_ROUTER_MODULE)
_authenticate_jwt_websocket = _CHAT_ROUTER_MODULE._authenticate_jwt_websocket


class DummySettings:
    auth_allow_query_token = False


class DummyWebSocket:
    def __init__(
        self,
        authorization: str = "",
        query_token: str | None = None,
        device_token: str | None = None,
    ) -> None:
        self.headers = {"authorization": authorization}
        if device_token is not None:
            self.headers["x-device-token"] = device_token
        self.query_params: dict[str, str] = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        if query_token is not None:
            self.query_params["accessToken"] = query_token


def _build_jwt() -> JWTService:
    return JWTService(
        signing_key="test-signing-key",
        issuer="nyxclaw-tests",
        audience="nyxclaw-mobile-tests",
        access_ttl_sec=120,
        refresh_ttl_sec=600,
    )


def test_challenge_one_time_use() -> None:
    store = InMemoryAuthStore()
    service = ChallengeService(store=store, challenge_ttl_sec=60)

    issued = service.issue_challenge(device_id="dev-1", public_key="pk-1", algorithm="ed25519")
    valid_once = store.get_challenge(issued.challenge_id)
    assert valid_once is not None
    assert valid_once.used is False

    service.mark_used(issued.challenge_id)
    valid_twice = store.get_challenge(issued.challenge_id)
    assert valid_twice is not None
    assert valid_twice.used is True


def test_refresh_reuse_detection_revokes_session() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    initial_refresh = jwt_service.mint_refresh_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=utc_now(),
            refresh_expires_at=utc_now() + timedelta(minutes=10),
            refresh_jti=initial_refresh.jti,
            refresh_token_hash=jwt_service.hash_refresh_token(initial_refresh.token),
            revoked=False,
        )
    )

    # First valid rotation succeeds.
    rotated = jwt_service.mint_refresh_token(session_id="sess-1", device_id="dev-1")
    ok = store.rotate_refresh(
        session_id="sess-1",
        presented_jti=initial_refresh.jti,
        presented_hash=jwt_service.hash_refresh_token(initial_refresh.token),
        new_jti=rotated.jti,
        new_hash=jwt_service.hash_refresh_token(rotated.token),
        new_exp=utc_now() + timedelta(minutes=10),
    )
    assert ok.ok is True

    # Reuse of old refresh should revoke the session.
    reused = store.rotate_refresh(
        session_id="sess-1",
        presented_jti=initial_refresh.jti,
        presented_hash=jwt_service.hash_refresh_token(initial_refresh.token),
        new_jti="unused",
        new_hash="unused",
        new_exp=utc_now() + timedelta(minutes=10),
    )
    assert reused.ok is False
    assert reused.code == "refresh_reuse_detected"

    revoked = store.get_session("sess-1")
    assert revoked is not None and revoked.revoked is True


def test_websocket_jwt_auth_accepts_active_device() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk-1",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )

    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="refresh-1",
            refresh_token_hash="hash-1",
            revoked=False,
        )
    )

    websocket = DummyWebSocket(authorization=f"Bearer {access.token}")
    ok, close_code, close_reason, auth_context = _authenticate_jwt_websocket(
        websocket=websocket,
        settings=DummySettings(),
        jwt_service=jwt_service,
        auth_store=store,
    )

    assert ok is True
    assert close_code == 1000
    assert close_reason == "ok"
    assert auth_context == {"sessionId": "sess-1", "deviceId": "dev-1", "userId": "dev-1"}


def test_websocket_jwt_auth_rejects_pending_device() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk-1",
            algorithm="ed25519",
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )

    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="refresh-1",
            refresh_token_hash="hash-1",
            revoked=False,
        )
    )

    websocket = DummyWebSocket(authorization=f"Bearer {access.token}")
    ok, close_code, close_reason, auth_context = _authenticate_jwt_websocket(
        websocket=websocket,
        settings=DummySettings(),
        jwt_service=jwt_service,
        auth_store=store,
    )

    assert ok is False
    assert close_code == 4403
    assert close_reason.startswith("pairing_required:")
    assert auth_context is None
    pending = store.list_pairing_requests(status="pending")
    assert len(pending) == 1
    assert close_reason == f"pairing_required:{pending[0].request_id}"


def test_websocket_jwt_auth_requires_device_token_for_approved_device() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk-1",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    store.upsert_approved_device(
        ApprovedDeviceRecord(
            device_id="dev-1",
            public_key_fingerprint="sha256:abc",
            role="operator",
            approved_scopes=["chat.read"],
            approved_at=now,
            revoked=False,
            revoked_at=None,
        )
    )
    store.save_device_token(
        DeviceTokenRecord(
            device_id="dev-1",
            role="operator",
            token_hash="expected-hash",
            issued_at=now,
            expires_at=now + timedelta(minutes=10),
            revoked=False,
        )
    )

    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="refresh-1",
            refresh_token_hash="hash-1",
            revoked=False,
        )
    )

    websocket = DummyWebSocket(authorization=f"Bearer {access.token}")
    ok, close_code, close_reason, auth_context = _authenticate_jwt_websocket(
        websocket=websocket,
        settings=DummySettings(),
        jwt_service=jwt_service,
        auth_store=store,
    )

    assert ok is False
    assert close_code == 4403
    assert close_reason == "device_token_required"
    assert auth_context is None


def test_websocket_jwt_auth_accepts_valid_device_token_for_approved_device() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk-1",
            algorithm="ed25519",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    store.upsert_approved_device(
        ApprovedDeviceRecord(
            device_id="dev-1",
            public_key_fingerprint="sha256:abc",
            role="operator",
            approved_scopes=["chat.read"],
            approved_at=now,
            revoked=False,
            revoked_at=None,
        )
    )
    raw_token = "device-token-1"
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    store.save_device_token(
        DeviceTokenRecord(
            device_id="dev-1",
            role="operator",
            token_hash=token_hash,
            issued_at=now,
            expires_at=now + timedelta(minutes=10),
            revoked=False,
        )
    )

    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="refresh-1",
            refresh_token_hash="hash-1",
            revoked=False,
        )
    )

    websocket = DummyWebSocket(authorization=f"Bearer {access.token}", device_token=raw_token)
    ok, close_code, close_reason, auth_context = _authenticate_jwt_websocket(
        websocket=websocket,
        settings=DummySettings(),
        jwt_service=jwt_service,
        auth_store=store,
    )

    assert ok is True
    assert close_code == 1000
    assert close_reason == "ok"
    assert auth_context == {"sessionId": "sess-1", "deviceId": "dev-1", "userId": "dev-1"}


def test_websocket_jwt_auth_reject_status_continues_to_block_connect() -> None:
    store = InMemoryAuthStore()
    jwt_service = _build_jwt()

    now = utc_now()
    store.upsert_device(
        DeviceRecord(
            device_id="dev-1",
            public_key="pk-1",
            algorithm="ed25519",
            status="rejected",
            created_at=now,
            updated_at=now,
        )
    )

    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    store.save_session(
        SessionRecord(
            session_id="sess-1",
            device_id="dev-1",
            created_at=now,
            refresh_expires_at=now + timedelta(minutes=10),
            refresh_jti="refresh-1",
            refresh_token_hash="hash-1",
            revoked=False,
        )
    )

    websocket = DummyWebSocket(authorization=f"Bearer {access.token}")
    ok, close_code, close_reason, auth_context = _authenticate_jwt_websocket(
        websocket=websocket,
        settings=DummySettings(),
        jwt_service=jwt_service,
        auth_store=store,
    )

    assert ok is False
    assert close_code == 4403
    assert close_reason.startswith("pairing_required:")
    assert auth_context is None


def _b64url_json(data: dict[str, str]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def test_jwt_rejects_non_hs256_algorithm_header() -> None:
    jwt_service = _build_jwt()
    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    _, encoded_payload, encoded_signature = access.token.split(".")

    tampered_header = _b64url_json({"alg": "none", "typ": "JWT"})
    tampered = f"{tampered_header}.{encoded_payload}.{encoded_signature}"

    with pytest.raises(JWTError) as exc_info:
        jwt_service.verify_access(tampered)

    assert str(exc_info.value) == "invalid_token_algorithm"


def test_jwt_rejects_malformed_numeric_claims() -> None:
    jwt_service = _build_jwt()
    access = jwt_service.mint_access_token(session_id="sess-1", device_id="dev-1")
    _, encoded_payload, _ = access.token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(encoded_payload + "==").decode("utf-8"))
    payload["nbf"] = "not-a-number"
    tampered = jwt_service._encode(payload)

    with pytest.raises(JWTError) as exc_info:
        jwt_service.verify_access(tampered)

    assert str(exc_info.value) == "invalid_token_claims"
