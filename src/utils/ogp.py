"""OpenGraph metadata extraction for link card enrichment."""

import asyncio
import http.client
import ipaddress
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from core.logger import get_logger

logger = get_logger(__name__)

_MAX_RESPONSE_BYTES = 256 * 1024  # 256 KB — enough for <head> OG tags
_CACHE_TTL_SECONDS = 48 * 3600  # 48 hours (successful fetches)
_CACHE_NEGATIVE_TTL_SECONDS = 5 * 60  # 5 minutes (failed fetches)
_CACHE_MAX_SIZE = 256
_MAX_URLS_PER_MESSAGE = 5
_FETCH_SEMAPHORE = asyncio.Semaphore(3)  # max concurrent OGP fetches across all sessions
_MAX_REDIRECTS = 3

# OGP meta tag patterns — handle both attribute orderings
_OG_RE = {}
for _prop in ("title", "description", "image", "site_name"):
    _OG_RE[_prop] = re.compile(
        rf'<meta[^>]*property=[\'"]og:{_prop}[\'"][^>]*content=[\'"]([^\'"]+)[\'"]',
        re.IGNORECASE,
    )
    _OG_RE[f"{_prop}_alt"] = re.compile(
        rf'<meta[^>]*content=[\'"]([^\'"]+)[\'"][^>]*property=[\'"]og:{_prop}[\'"]',
        re.IGNORECASE,
    )

# Fallback: <title> tag
_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)

# URL extraction from markdown/text
_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")


@dataclass(frozen=True, slots=True)
class OGPMetadata:
    """OpenGraph metadata for a URL."""

    url: str
    title: str | None = None
    description: str | None = None
    image: str | None = None
    site_name: str | None = None

    def to_card(self) -> dict:
        """Convert to a link_card dict for the client protocol."""
        card: dict = {"type": "link_card", "url": self.url}
        if self.title:
            card["title"] = self.title
        if self.description:
            card["summary"] = self.description
        if self.image:
            card["thumbnail"] = self.image
        if self.site_name:
            card["site_name"] = self.site_name
        return card


# --- In-memory cache with TTL ---

_cache: dict[str, tuple[float, OGPMetadata | None]] = {}
_cache_lock = threading.Lock()


def _cache_get(url: str) -> tuple[bool, OGPMetadata | None]:
    """Return (hit, metadata). hit=False means cache miss."""
    with _cache_lock:
        entry = _cache.get(url)
        if entry is None:
            return False, None
        ts, meta = entry
        ttl = _CACHE_TTL_SECONDS if meta is not None else _CACHE_NEGATIVE_TTL_SECONDS
        if time.monotonic() - ts > ttl:
            del _cache[url]
            return False, None
        return True, meta


def _cache_put(url: str, meta: OGPMetadata | None) -> None:
    with _cache_lock:
        # Evict oldest if full
        if len(_cache) >= _CACHE_MAX_SIZE and url not in _cache:
            oldest_key = next(iter(_cache))
            del _cache[oldest_key]
        _cache[url] = (time.monotonic(), meta)


# --- SSRF protection ---

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


def _resolve_and_validate(hostname: str) -> str:
    """Resolve hostname to IP, reject private/loopback/link-local/reserved ranges.

    Returns the first safe IP as a string, or raises ValueError.
    """
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {hostname}") from e

    for _family, _type, _proto, _canonname, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"Blocked private/reserved IP {ip} for {hostname}")

    # All IPs passed — return the first one to connect to
    return addr_info[0][4][0]


def _fetch_html(url: str) -> tuple[str, str]:
    """Fetch HTML from URL using http.client, connecting to pre-resolved IP.

    Handles redirects manually, re-validating each hop against SSRF checks.
    Returns (html_content, final_url).
    """
    current_url = url

    for _ in range(_MAX_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported scheme: {parsed.scheme}")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("No hostname in URL")

        # Resolve DNS and validate IP — single resolution used for connection
        resolved_ip = _resolve_and_validate(hostname)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # Connect directly to the resolved IP, with correct SNI for HTTPS.
        # Raw socket created first — must be closed on any failure before
        # it's handed to conn (which owns cleanup via conn.close() after).
        raw_sock = socket.create_connection((resolved_ip, port), timeout=3)
        try:
            if parsed.scheme == "https":
                context = ssl.create_default_context()
                tls_sock = context.wrap_socket(raw_sock, server_hostname=hostname)
                conn = http.client.HTTPSConnection(hostname, port, timeout=3, context=context)
                conn.sock = tls_sock
            else:
                conn = http.client.HTTPConnection(hostname, port, timeout=3)
                conn.sock = raw_sock
        except Exception:
            raw_sock.close()
            raise

        try:
            conn.request("GET", path, headers={"Host": hostname, "User-Agent": _USER_AGENT})
            resp = conn.getresponse()

            # Handle redirects — re-validate the new target
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader("Location")
                if not location:
                    raise ValueError(f"Redirect {resp.status} with no Location header")
                # Resolve relative redirects
                current_url = urljoin(current_url, location)
                resp.read()  # drain body before reuse
                continue

            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status}")

            html = resp.read(_MAX_RESPONSE_BYTES).decode("utf-8", errors="ignore")
            return html, current_url
        finally:
            conn.close()

    raise ValueError(f"Too many redirects (>{_MAX_REDIRECTS})")


def _extract_ogp(html: str, url: str) -> OGPMetadata:
    """Extract OGP metadata from HTML content."""

    def _extract(prop: str) -> str | None:
        m = _OG_RE[prop].search(html) or _OG_RE[f"{prop}_alt"].search(html)
        return m.group(1).strip() if m else None

    title = _extract("title")
    if not title:
        m = _TITLE_RE.search(html)
        if m:
            title = m.group(1).strip()

    image = _extract("image")
    if image and image.startswith("/"):
        image = urljoin(url, image)

    # Favicon fallback when no og:image found
    if not image:
        parsed = urlparse(url)
        if parsed.hostname:
            image = f"https://www.google.com/s2/favicons?domain={parsed.hostname}&sz=128"

    return OGPMetadata(
        url=url,
        title=title,
        description=_extract("description"),
        image=image,
        site_name=_extract("site_name"),
    )


# --- Sync fetch ---


def _fetch_ogp_sync(url: str) -> OGPMetadata | None:
    """Fetch OpenGraph metadata from a URL (blocking, SSRF-safe)."""
    try:
        html, final_url = _fetch_html(url)
        return _extract_ogp(html, final_url)
    except ValueError as e:
        logger.debug(f"OGP fetch blocked for {url}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch OGP for {url}: {e}")
        return None


# --- Public API ---

_THUMBNAIL_HINTS_SENTINEL = "---THUMBNAIL_HINTS---"


def extract_thumbnail_hints(text: str) -> dict[str, str]:
    """Parse URL→thumbnail mappings from a tool output containing THUMBNAIL_HINTS footer.

    Any tool can append this footer to its output:
        ---THUMBNAIL_HINTS---
        https://page-url.com\\thttps://thumbnail-url.com

    Returns {page_url: thumbnail_url}. Empty dict if no hints found.
    """
    idx = text.find(_THUMBNAIL_HINTS_SENTINEL)
    if idx == -1:
        return {}
    hints: dict[str, str] = {}
    for line in text[idx + len(_THUMBNAIL_HINTS_SENTINEL) :].split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].startswith("http") and parts[1].startswith("https://"):
            hints[parts[0]] = parts[1]
    return hints


def extract_urls(text: str) -> list[str]:
    """Extract unique HTTP(S) URLs from text/markdown."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


async def fetch_ogp(url: str) -> OGPMetadata | None:
    """Fetch OGP metadata for a single URL (cached, async, concurrency-limited)."""
    hit, meta = _cache_get(url)
    if hit:
        return meta
    async with _FETCH_SEMAPHORE:
        # Re-check cache — another task may have fetched while we waited
        hit, meta = _cache_get(url)
        if hit:
            return meta
        meta = await asyncio.to_thread(_fetch_ogp_sync, url)
        _cache_put(url, meta)
        return meta


async def fetch_ogp_batch(urls: list[str]) -> list[OGPMetadata]:
    """Fetch OGP metadata for multiple URLs in parallel (capped). Returns only successful results."""
    if not urls:
        return []
    # Cap URLs to prevent resource exhaustion from AI-generated content
    capped = urls[:_MAX_URLS_PER_MESSAGE]
    results = await asyncio.gather(*(fetch_ogp(url) for url in capped), return_exceptions=True)
    return [r for r in results if isinstance(r, OGPMetadata)]
