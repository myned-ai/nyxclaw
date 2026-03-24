"""Tests for ZeroClawBackend._build_ws_chat_url — user_id in query params."""

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse


class TestBuildWsChatUrl:
    """Test the URL builder without instantiating the full backend."""

    def _call_build(self, base_url: str, auth_token: str | None = None, user_id: str | None = None) -> str:
        """Import and call _build_ws_chat_url with mocked settings."""
        from backend.zeroclaw.backend import ZeroClawBackend

        # Create a minimal mock backend — avoid __init__ entirely
        backend = object.__new__(ZeroClawBackend)

        mock_zc = MagicMock()
        mock_zc.base_url = base_url
        mock_zc.auth_token = auth_token
        mock_zc.user_id = user_id

        with patch("backend.zeroclaw.settings.get_zeroclaw_settings", return_value=mock_zc):
            backend._zc = mock_zc
            return backend._build_ws_chat_url()

    def test_no_auth_no_user_id(self):
        url = self._call_build("http://localhost:5555")
        parsed = urlparse(url)
        assert parsed.scheme == "ws"
        assert parsed.path == "/ws/avatar"
        assert parsed.query == ""

    def test_auth_token_only(self):
        url = self._call_build("https://example.com", auth_token="secret123")
        parsed = urlparse(url)
        assert parsed.scheme == "wss"
        qs = parse_qs(parsed.query)
        assert qs["token"] == ["secret123"]
        assert "user_id" not in qs

    def test_user_id_only(self):
        url = self._call_build("http://localhost:5555", user_id="uid-42")
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        assert qs["user_id"] == ["uid-42"]
        assert "token" not in qs

    def test_auth_token_and_user_id(self):
        url = self._call_build("https://example.com", auth_token="tok", user_id="uid-1")
        parsed = urlparse(url)
        assert parsed.scheme == "wss"
        qs = parse_qs(parsed.query)
        assert qs["token"] == ["tok"]
        assert qs["user_id"] == ["uid-1"]

    def test_existing_path_preserved(self):
        url = self._call_build("http://localhost:5555/api/v2", user_id="u1")
        parsed = urlparse(url)
        assert parsed.path == "/api/v2/ws/avatar"

    def test_existing_query_preserved(self):
        url = self._call_build("http://localhost:5555?existing=yes", user_id="u1")
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        assert qs["existing"] == ["yes"]
        assert qs["user_id"] == ["u1"]
