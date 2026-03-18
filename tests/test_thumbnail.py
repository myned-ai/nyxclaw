"""Tests for utils.thumbnail — OpenGraph thumbnail extraction."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.thumbnail import _OG_IMAGE_RE, _OG_IMAGE_RE_ALT, _fetch_thumbnail_sync, get_link_thumbnail


# ── Regex unit tests ──────────────────────────────────────────────


class TestOgImageRegex:
    def test_standard_og_image_tag(self):
        html = '<meta property="og:image" content="https://example.com/img.jpg">'
        match = _OG_IMAGE_RE.search(html)
        assert match is not None
        assert match.group(1) == "https://example.com/img.jpg"

    def test_single_quotes(self):
        html = "<meta property='og:image' content='https://example.com/img.png'>"
        match = _OG_IMAGE_RE.search(html)
        assert match is not None
        assert match.group(1) == "https://example.com/img.png"

    def test_alt_order_content_before_property(self):
        html = '<meta content="https://example.com/thumb.jpg" property="og:image">'
        match = _OG_IMAGE_RE_ALT.search(html)
        assert match is not None
        assert match.group(1) == "https://example.com/thumb.jpg"

    def test_extra_attributes_between(self):
        html = '<meta name="foo" property="og:image" class="bar" content="https://example.com/a.jpg">'
        match = _OG_IMAGE_RE.search(html)
        assert match is not None
        assert match.group(1) == "https://example.com/a.jpg"

    def test_no_match_on_unrelated_meta(self):
        html = '<meta property="og:title" content="Page Title">'
        assert _OG_IMAGE_RE.search(html) is None
        assert _OG_IMAGE_RE_ALT.search(html) is None

    def test_case_insensitive(self):
        html = '<META PROPERTY="og:image" CONTENT="https://example.com/img.jpg">'
        match = _OG_IMAGE_RE.search(html)
        assert match is not None


# ── _fetch_thumbnail_sync tests ───────────────────────────────────


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, size=-1):
        if size < 0:
            return self._body
        return self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestFetchThumbnailSync:
    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_returns_absolute_og_image(self, mock_urlopen, _mock_safe):
        html = b'<html><head><meta property="og:image" content="https://cdn.example.com/pic.jpg"></head></html>'
        mock_urlopen.return_value = FakeResponse(html)

        result = _fetch_thumbnail_sync("https://example.com/page")
        assert result == "https://cdn.example.com/pic.jpg"

    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_resolves_relative_url(self, mock_urlopen, _mock_safe):
        html = b'<html><head><meta property="og:image" content="/images/thumb.png"></head></html>'
        mock_urlopen.return_value = FakeResponse(html)

        result = _fetch_thumbnail_sync("https://example.com/page")
        assert result == "https://example.com/images/thumb.png"

    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_returns_none_when_no_og_image(self, mock_urlopen, _mock_safe):
        html = b"<html><head><title>No OG</title></head></html>"
        mock_urlopen.return_value = FakeResponse(html)

        result = _fetch_thumbnail_sync("https://example.com")
        assert result is None

    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_returns_none_on_network_error(self, mock_urlopen, _mock_safe):
        mock_urlopen.side_effect = OSError("Connection refused")

        result = _fetch_thumbnail_sync("https://example.com")
        assert result is None

    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_returns_none_on_timeout(self, mock_urlopen, _mock_safe):
        mock_urlopen.side_effect = TimeoutError("timed out")

        result = _fetch_thumbnail_sync("https://example.com")
        assert result is None

    @patch("utils.thumbnail._is_safe_url", return_value=True)
    @patch("utils.thumbnail.urllib.request.urlopen")
    def test_handles_non_utf8_gracefully(self, mock_urlopen, _mock_safe):
        # Binary content that isn't valid HTML
        mock_urlopen.return_value = FakeResponse(b"\x80\x81\x82\x83")

        result = _fetch_thumbnail_sync("https://example.com")
        assert result is None

    def test_blocks_unsafe_url(self):
        """SSRF protection: private IPs and non-http schemes are rejected."""
        result = _fetch_thumbnail_sync("file:///etc/passwd")
        assert result is None

    @patch("utils.thumbnail._is_safe_url", return_value=False)
    def test_blocks_when_is_safe_url_returns_false(self, _mock_safe):
        result = _fetch_thumbnail_sync("https://example.com")
        assert result is None


# ── Async wrapper tests ───────────────────────────────────────────


class TestGetLinkThumbnail:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_url(self):
        result = await get_link_thumbnail("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_none_url(self):
        result = await get_link_thumbnail(None)
        assert result is None

    @pytest.mark.asyncio
    @patch("utils.thumbnail._fetch_thumbnail_sync")
    async def test_delegates_to_sync_via_to_thread(self, mock_sync):
        mock_sync.return_value = "https://example.com/thumb.jpg"

        result = await get_link_thumbnail("https://example.com")
        assert result == "https://example.com/thumb.jpg"
        mock_sync.assert_called_once_with("https://example.com")
