"""Tests for auth signature success/failure/replay behavior."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.challenge_service import ChallengeService
from auth.device_verifier import DeviceVerificationError
from auth.jwt_service import JWTService
from auth.models import AuthCompleteRequest, ChallengeRecord, DeviceRecord, utc_now
from auth.store import InMemoryAuthStore
from core.settings import Settings
from routers.auth_router import complete


class _VerifierOk:
    def verify_signature(self, signed_payload: str, signature_b64url: str, public_key_b64url: str, algorithm: str) -> bool:
        return True


class _VerifierFail:
    def verify_signature(self, signed_payload: str, signature_b64url: str, public_key_b64url: str, algorithm: str) -> bool:
        raise DeviceVerificationError("signature_verification_failed")


def _settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_secret_key="test-auth-secret",
        jwt_signing_key="test-jwt-key",
        jwt_issuer="nyxclaw-tests",
        jwt_audience="nyxclaw-mobile-tests",
        jwt_access_ttl_sec=120,
        jwt_refresh_ttl_sec=600,
        auth_registration_policy="open",
    )


def _setup_challenge() -> tuple[InMemoryAuthStore, ChallengeService, AuthCompleteRequest]:
    store = InMemoryAuthStore()
    now = utc_now()
    device = DeviceRecord(
        device_id="device-1",
        public_key="public-key-1",
        algorithm="ed25519",
        status="active",
        created_at=now,
        updated_at=now,
    )
    store.upsert_device(device)

    challenge_service = ChallengeService(store=store, challenge_ttl_sec=120)
    issued = challenge_service.issue_challenge(device_id=device.device_id, public_key=device.public_key, algorithm=device.algorithm)
    canonical = challenge_service.build_canonical_payload(issued)

    payload = AuthCompleteRequest(
        challengeId=issued.challenge_id,
        deviceId=device.device_id,
        publicKey=device.public_key,
        signature="dummy-signature",
        signedPayload=canonical,
        algorithm="ed25519",
    )

    return store, challenge_service, payload


@pytest.mark.asyncio
async def test_signature_success_mints_tokens() -> None:
    settings = _settings()
    store, challenge_service, payload = _setup_challenge()
    jwt_service = JWTService(
        signing_key=settings.jwt_signing_key,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        access_ttl_sec=settings.jwt_access_ttl_sec,
        refresh_ttl_sec=settings.jwt_refresh_ttl_sec,
    )

    response = await complete(
        payload=payload,
        settings=settings,
        store=store,
        challenge_service=challenge_service,
        verifier=_VerifierOk(),
        jwt_service=jwt_service,
    )

    assert not isinstance(response, JSONResponse)
    assert response.accessToken
    assert response.refreshToken
    assert response.sessionId
    assert response.deviceId == "device-1"


@pytest.mark.asyncio
async def test_signature_failure_returns_bad_signature() -> None:
    settings = _settings()
    store, challenge_service, payload = _setup_challenge()
    jwt_service = JWTService(
        signing_key=settings.jwt_signing_key,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        access_ttl_sec=settings.jwt_access_ttl_sec,
        refresh_ttl_sec=settings.jwt_refresh_ttl_sec,
    )

    response = await complete(
        payload=payload,
        settings=settings,
        store=store,
        challenge_service=challenge_service,
        verifier=_VerifierFail(),
        jwt_service=jwt_service,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 401
    assert b'"code":"bad_signature"' in response.body


@pytest.mark.asyncio
async def test_signature_replay_rejected_after_success() -> None:
    settings = _settings()
    store, challenge_service, payload = _setup_challenge()
    jwt_service = JWTService(
        signing_key=settings.jwt_signing_key,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        access_ttl_sec=settings.jwt_access_ttl_sec,
        refresh_ttl_sec=settings.jwt_refresh_ttl_sec,
    )

    first = await complete(
        payload=payload,
        settings=settings,
        store=store,
        challenge_service=challenge_service,
        verifier=_VerifierOk(),
        jwt_service=jwt_service,
    )
    assert not isinstance(first, JSONResponse)

    replay = await complete(
        payload=payload,
        settings=settings,
        store=store,
        challenge_service=challenge_service,
        verifier=_VerifierOk(),
        jwt_service=jwt_service,
    )
    assert isinstance(replay, JSONResponse)
    assert replay.status_code == 401
    assert b'"code":"expired_challenge"' in replay.body


@pytest.mark.asyncio
async def test_signature_rejected_when_challenge_timestamp_outside_skew() -> None:
    settings = _settings()
    store = InMemoryAuthStore()
    now = utc_now()
    device = DeviceRecord(
        device_id="device-1",
        public_key="public-key-1",
        algorithm="ed25519",
        status="active",
        created_at=now,
        updated_at=now,
    )
    store.upsert_device(device)

    # Future-issued challenge outside allowed skew window.
    issued_at_ms = int(time.time() * 1000) + 180_000
    expires_at_ms = issued_at_ms + 120_000
    challenge = ChallengeRecord(
        challenge_id="future-challenge",
        device_id=device.device_id,
        public_key=device.public_key,
        algorithm="ed25519",
        nonce="nonce-1",
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
        used=False,
    )
    store.save_challenge(challenge)

    challenge_service = ChallengeService(store=store, challenge_ttl_sec=120)
    payload = AuthCompleteRequest(
        challengeId=challenge.challenge_id,
        deviceId=device.device_id,
        publicKey=device.public_key,
        signature="dummy-signature",
        signedPayload=challenge_service.build_canonical_payload(challenge),
        algorithm="ed25519",
    )
    jwt_service = JWTService(
        signing_key=settings.jwt_signing_key,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        access_ttl_sec=settings.jwt_access_ttl_sec,
        refresh_ttl_sec=settings.jwt_refresh_ttl_sec,
    )

    response = await complete(
        payload=payload,
        settings=settings,
        store=store,
        challenge_service=challenge_service,
        verifier=_VerifierOk(),
        jwt_service=jwt_service,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 401
    assert b'"code":"timestamp_skew"' in response.body
