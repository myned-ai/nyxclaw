"""Tests for pairing setup-code service and endpoint behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import AuthPairingSetupCodeRequest  # noqa: E402
from auth.setup_code_service import SetupCodeError, SetupCodeService  # noqa: E402
from core.settings import Settings  # noqa: E402
from routers.pairing_router import generate_pairing_setup_code  # noqa: E402


def _settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_mode="device_challenge_jwt",
        auth_admin_token="admin-secret",
        jwt_signing_key="jwt-secret",
        server_host="127.0.0.1",
        server_port=8080,
        pairing_setup_code_ttl_sec=120,
    )


def _service() -> SetupCodeService:
    return SetupCodeService(secret="jwt-secret", ttl_sec=120)


def _context(settings: Settings) -> str:
    return f"{settings.server_host}:{settings.server_port}|{settings.jwt_issuer}|{settings.jwt_audience}"


def test_setup_code_service_requires_non_empty_secret() -> None:
    with pytest.raises(SetupCodeError, match="setup_code_secret_missing"):
        SetupCodeService(secret="", ttl_sec=120)


class _DummyRequest:
    def __init__(self, *, admin_token: str = "", scheme: str = "https", host: str = "nyx.example") -> None:
        self.headers = {
            "x-auth-admin-token": admin_token,
            "host": host,
        }
        self.url = SimpleNamespace(scheme=scheme)


def test_setup_code_roundtrip_and_context_validation() -> None:
    settings = _settings()
    service = _service()
    code, issued = service.generate_setup_code(
        gateway_url="wss://nyx.example/ws",
        bootstrap_token="bootstrap-1",
        bootstrap_password=None,
        role="operator",
        scopes=["chat.read"],
        context=_context(settings),
    )

    decoded = service.decode_setup_code(code)
    assert decoded.url == "wss://nyx.example/ws"
    assert decoded.bootstrap_token == "bootstrap-1"
    assert decoded.bootstrap_password is None

    # Should not raise.
    service.validate_setup_code(decoded, expected_context=_context(settings))
    assert issued.expires_at > issued.issued_at


def test_setup_code_context_mismatch_rejected() -> None:
    settings = _settings()
    service = _service()
    code, _ = service.generate_setup_code(
        gateway_url="wss://nyx.example/ws",
        bootstrap_token="bootstrap-1",
        bootstrap_password=None,
        role="operator",
        scopes=[],
        context=_context(settings),
    )

    decoded = service.decode_setup_code(code)
    with pytest.raises(SetupCodeError, match="setup_code_context_mismatch"):
        service.validate_setup_code(decoded, expected_context="different-context")


@pytest.mark.asyncio
async def test_generate_pairing_setup_code_requires_admin_token() -> None:
    settings = _settings()
    request = _DummyRequest(admin_token="wrong-secret")

    response = await generate_pairing_setup_code(
        payload=AuthPairingSetupCodeRequest(),
        request=request,
        settings=settings,
        setup_code_service=_service(),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert b'"code":"forbidden"' in response.body


@pytest.mark.asyncio
async def test_generate_pairing_setup_code_returns_valid_payload() -> None:
    settings = _settings()
    service = _service()
    request = _DummyRequest(admin_token="admin-secret")

    response = await generate_pairing_setup_code(
        payload=AuthPairingSetupCodeRequest(scopes=["chat.read", "chat.write"]),
        request=request,
        settings=settings,
        setup_code_service=service,
    )

    assert response.gatewayUrl == "wss://nyx.example/ws"
    assert response.expiresIn > 0
    assert response.auth == "token"

    decoded = service.decode_setup_code(response.setupCode)
    assert decoded.url == response.gatewayUrl
    assert decoded.bootstrap_token is not None
    assert decoded.bootstrap_password is None
    assert decoded.scopes == ["chat.read", "chat.write"]
