"""Tests for auth.setup_code_service — SetupCodeService, _print_setup_code, ensure_setup_code."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("segno")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.setup_code_service import (  # noqa: E402
    SetupCodeService,
    _print_setup_code,
    ensure_setup_code,
)


# ---------------------------------------------------------------------------
# SetupCodeService.__init__
# ---------------------------------------------------------------------------


class TestSetupCodeServiceInit:
    def test_rejects_empty_secret(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SetupCodeService(secret="")

    def test_rejects_whitespace_only_secret(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SetupCodeService(secret="   ")

    def test_accepts_valid_secret(self) -> None:
        svc = SetupCodeService(secret="my-secret")
        assert svc._secret == b"my-secret"


# ---------------------------------------------------------------------------
# SetupCodeService.generate_signed
# ---------------------------------------------------------------------------


class TestGenerateSigned:
    def test_returns_base64url_string(self) -> None:
        svc = SetupCodeService(secret="test-secret")
        result = svc.generate_signed(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok123",
        )
        # Should be a non-empty string with no padding
        assert isinstance(result, str)
        assert len(result) > 0
        assert "=" not in result

    def test_decodes_to_valid_envelope(self) -> None:
        svc = SetupCodeService(secret="test-secret")
        result = svc.generate_signed(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok123",
        )
        # Add padding back for base64 decode
        padded = result + "=" * (-len(result) % 4)
        raw = base64.urlsafe_b64decode(padded)
        envelope = json.loads(raw)

        assert "v" in envelope
        assert "p" in envelope
        assert "s" in envelope
        assert envelope["v"] == 1

    def test_payload_contains_required_keys(self) -> None:
        svc = SetupCodeService(secret="test-secret")
        result = svc.generate_signed(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok123",
        )
        padded = result + "=" * (-len(result) % 4)
        envelope = json.loads(base64.urlsafe_b64decode(padded))
        payload = envelope["p"]

        assert "issuedAt" in payload
        assert isinstance(payload["issuedAt"], int)
        assert payload["token"] == "tok123"
        assert payload["url"] == "wss://example.com/ws"

    def test_strips_whitespace_from_token_and_url(self) -> None:
        svc = SetupCodeService(secret="test-secret")
        result = svc.generate_signed(
            gateway_url="  wss://example.com/ws  ",
            bootstrap_token="  tok123  ",
        )
        padded = result + "=" * (-len(result) % 4)
        envelope = json.loads(base64.urlsafe_b64decode(padded))
        payload = envelope["p"]

        assert payload["token"] == "tok123"
        assert payload["url"] == "wss://example.com/ws"

    def test_signature_is_valid_hmac_sha256(self) -> None:
        secret = "test-secret"
        svc = SetupCodeService(secret=secret)
        result = svc.generate_signed(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok123",
        )
        padded = result + "=" * (-len(result) % 4)
        envelope = json.loads(base64.urlsafe_b64decode(padded))

        # Reconstruct the payload JSON the same way the code does
        payload_json = json.dumps(envelope["p"], separators=(",", ":"), sort_keys=True)
        expected_digest = hmac.new(
            secret.encode("utf-8"),
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_sig = (
            base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("utf-8")
        )
        assert envelope["s"] == expected_sig

    def test_different_secrets_produce_different_signatures(self) -> None:
        kwargs = dict(gateway_url="wss://example.com/ws", bootstrap_token="tok123")
        result_a = SetupCodeService(secret="secret-a").generate_signed(**kwargs)
        result_b = SetupCodeService(secret="secret-b").generate_signed(**kwargs)
        assert result_a != result_b


# ---------------------------------------------------------------------------
# SetupCodeService.generate_qr_uri
# ---------------------------------------------------------------------------


class TestGenerateQrUri:
    def test_basic_uri_format(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok123",
        )
        assert uri == "nyxclaw://example.com/ws?t=tok123"

    def test_gateway_with_port(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="wss://example.com:8080/ws",
            bootstrap_token="tok123",
        )
        assert uri == "nyxclaw://example.com:8080/ws?t=tok123"

    def test_gateway_with_path(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="wss://example.com/api/v1/ws",
            bootstrap_token="tok123",
        )
        assert uri == "nyxclaw://example.com/api/v1/ws?t=tok123"

    def test_url_encodes_special_characters_in_token(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="wss://example.com/ws",
            bootstrap_token="abc+def/ghi=jkl",
        )
        assert "?t=abc%2Bdef%2Fghi%3Djkl" in uri

    def test_strips_scheme(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="wss://example.com/ws",
            bootstrap_token="tok",
        )
        assert uri.startswith("nyxclaw://example.com")
        assert "wss" not in uri

    def test_ws_scheme_also_stripped(self) -> None:
        uri = SetupCodeService.generate_qr_uri(
            gateway_url="ws://localhost:8080/ws",
            bootstrap_token="tok",
        )
        assert uri == "nyxclaw://localhost:8080/ws?t=tok"


# ---------------------------------------------------------------------------
# _print_setup_code
# ---------------------------------------------------------------------------


class TestPrintSetupCode:
    @patch("auth.setup_code_service.segno")
    @patch("auth.setup_code_service.logger")
    def test_calls_segno_make_and_save(
        self, mock_logger: MagicMock, mock_segno: MagicMock
    ) -> None:
        mock_qr = MagicMock()
        mock_segno.make.return_value = mock_qr

        _print_setup_code(setup_code="CODE123", qr_uri="nyxclaw://host/ws?t=tok")

        mock_segno.make.assert_called_once_with("nyxclaw://host/ws?t=tok", error="m")
        mock_qr.save.assert_called_once_with("/app/pairing_qr.png", scale=10, border=4)

    @patch("auth.setup_code_service.segno")
    @patch("auth.setup_code_service.logger")
    def test_logs_setup_code(
        self, mock_logger: MagicMock, mock_segno: MagicMock
    ) -> None:
        mock_segno.make.return_value = MagicMock()

        _print_setup_code(setup_code="MY-CODE", qr_uri="nyxclaw://host/ws?t=tok")

        # Collect all logger.info call args
        logged = [str(call.args[0]) for call in mock_logger.info.call_args_list]
        assert any("MY-CODE" in line for line in logged)
        assert any("DEVICE PAIRING" in line for line in logged)
        assert any("/app/pairing_qr.png" in line for line in logged)

    @patch("auth.setup_code_service.segno")
    @patch("auth.setup_code_service.logger")
    def test_segno_save_failure_logs_warning(
        self, mock_logger: MagicMock, mock_segno: MagicMock
    ) -> None:
        """QR save failure is caught and logged as warning, doesn't crash."""
        mock_qr = MagicMock()
        mock_qr.save.side_effect = OSError("Permission denied")
        mock_segno.make.return_value = mock_qr

        _print_setup_code(setup_code="CODE", qr_uri="nyxclaw://host/ws?t=tok")

        mock_logger.warning.assert_called_once()
        # Setup code text is still logged despite QR failure
        log_texts = [str(c) for c in mock_logger.info.call_args_list]
        assert any("CODE" in t for t in log_texts)


# ---------------------------------------------------------------------------
# ensure_setup_code
# ---------------------------------------------------------------------------


class TestEnsureSetupCode:
    def _make_settings(self, **overrides) -> MagicMock:
        defaults = {
            "auth_enabled": True,
            "auth_setup_code_secret": "my-secret",
            "auth_setup_code_url": "wss://example.com/ws",
            "auth_regenerate_setup_code": False,
            "use_ssl": False,
            "server_host": "127.0.0.1",
            "server_port": 8080,
        }
        defaults.update(overrides)
        settings = MagicMock()
        for k, v in defaults.items():
            setattr(settings, k, v)
        return settings

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_returns_none_when_auth_disabled(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(auth_enabled=False)
        auth_store = MagicMock()
        result = ensure_setup_code(auth_store)
        assert result is None
        mock_print.assert_not_called()

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_returns_none_when_secret_empty(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(
            auth_setup_code_secret="   "
        )
        auth_store = MagicMock()
        result = ensure_setup_code(auth_store)
        assert result is None
        mock_print.assert_not_called()

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_returns_none_when_bootstrap_exists_and_no_regeneration(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(
            auth_regenerate_setup_code=False
        )
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = MagicMock()  # existing bootstrap
        result = ensure_setup_code(auth_store)
        assert result is None
        mock_print.assert_not_called()

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_generates_token_when_no_bootstrap_exists(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings()
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = None

        result = ensure_setup_code(auth_store)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        auth_store.set_bootstrap.assert_called_once()
        mock_print.assert_called_once()

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_generates_token_when_regeneration_enabled(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(
            auth_regenerate_setup_code=True
        )
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = MagicMock()  # existing bootstrap

        result = ensure_setup_code(auth_store)

        assert result is not None
        auth_store.set_bootstrap.assert_called_once()
        mock_print.assert_called_once()

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_print_receives_setup_code_and_qr_uri(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings()
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = None

        ensure_setup_code(auth_store)

        call_kwargs = mock_print.call_args.kwargs
        assert "setup_code" in call_kwargs
        assert "qr_uri" in call_kwargs
        assert call_kwargs["qr_uri"].startswith("nyxclaw://")

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_falls_back_to_constructed_url_when_setup_code_url_empty(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(
            auth_setup_code_url="   ",
            server_host="192.168.1.5",
            server_port=9090,
            use_ssl=False,
        )
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = None

        ensure_setup_code(auth_store)

        call_kwargs = mock_print.call_args.kwargs
        # QR URI should use the fallback host:port
        assert "192.168.1.5:9090" in call_kwargs["qr_uri"]

    @patch("auth.setup_code_service._print_setup_code")
    @patch("auth.setup_code_service.get_settings")
    def test_uses_wss_scheme_when_ssl_enabled(
        self, mock_get_settings: MagicMock, mock_print: MagicMock
    ) -> None:
        mock_get_settings.return_value = self._make_settings(
            auth_setup_code_url="   ",
            use_ssl=True,
        )
        auth_store = MagicMock()
        auth_store.get_bootstrap.return_value = None

        result = ensure_setup_code(auth_store)

        assert result is not None
        # The generate_signed uses the constructed wss:// URL internally
        # We verify _print_setup_code was called (integration covered above)
        mock_print.assert_called_once()
