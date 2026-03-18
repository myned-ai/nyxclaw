"""OpenGraph thumbnail extraction for link card enrichment."""

import asyncio
import ipaddress
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse

from core.logger import get_logger

logger = get_logger(__name__)

_OG_IMAGE_RE = re.compile(
    r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]',
    re.IGNORECASE,
)
_OG_IMAGE_RE_ALT = re.compile(
    r'<meta[^>]*content=[\'"]([^\'"]+)[\'"][^>]*property=[\'"]og:image[\'"]',
    re.IGNORECASE,
)

_MAX_RESPONSE_BYTES = 256 * 1024  # 256 KB — enough for <head> OG tags


def _is_safe_url(url: str) -> bool:
    """Reject non-HTTPS URLs and private/loopback IP ranges (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, ValueError):
        return False
    return True


def _fetch_thumbnail_sync(url: str) -> str | None:
    """Fetch og:image from a URL (blocking)."""
    if not _is_safe_url(url):
        logger.debug(f"Thumbnail fetch blocked (unsafe URL): {url}")
        return None
    try:
        req = urllib.request.Request(
            url,
            data=None,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                ),
            },
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            html = response.read(_MAX_RESPONSE_BYTES).decode("utf-8", errors="ignore")
            match = _OG_IMAGE_RE.search(html) or _OG_IMAGE_RE_ALT.search(html)
            if match:
                img_url = match.group(1)
                if img_url.startswith("/"):
                    img_url = urljoin(url, img_url)
                return img_url
    except Exception as e:
        logger.debug(f"Failed to fetch thumbnail for {url}: {e}")
    return None


async def get_link_thumbnail(url: str) -> str | None:
    """Asynchronously fetch the OpenGraph image for a URL."""
    if not url:
        return None
    return await asyncio.to_thread(_fetch_thumbnail_sync, url)
