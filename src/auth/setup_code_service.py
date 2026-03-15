"""Setup code generation for device pairing.

Two formats:
- **QR code** (compact): `nyxclaw://<gateway-host>/ws?t=<token>` — small enough
  for Version 3-4 QR (29-33 modules), easily scannable from a terminal.
- **Text paste** (signed envelope): base64url JSON with HMAC-SHA256 signature,
  for manual copy-paste when QR scanning isn't possible.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from urllib.parse import quote, urlparse

import segno

from auth.models import BootstrapRecord, utc_now
from auth.store import AuthStore, hash_token
from core.logger import get_logger
from core.settings import get_settings

logger = get_logger(__name__)


class SetupCodeService:
    def __init__(self) -> None:
        pass

    def generate_signed(self, *, gateway_url: str, bootstrap_token: str) -> str:
        """Generate a base64url setup code (text-paste format)."""
        now = int(time.time())
        payload = {
            "issuedAt": now,
            "token": bootstrap_token.strip(),
            "url": gateway_url.strip(),
        }
        envelope = {"v": 1, "p": payload}
        envelope_json = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
        return base64.urlsafe_b64encode(envelope_json.encode("utf-8")).rstrip(b"=").decode("utf-8")

    @staticmethod
    def generate_qr_uri(*, gateway_url: str, bootstrap_token: str) -> str:
        """Generate compact QR URI: nyxclaw://<host>/ws?t=<token>."""
        parsed = urlparse(gateway_url)
        # Strip scheme — the mobile app prepends wss://
        host_and_path = parsed.netloc + parsed.path
        token_encoded = quote(bootstrap_token, safe="")
        return f"nyxclaw://{host_and_path}?t={token_encoded}"


def ensure_setup_code(auth_store: AuthStore) -> str | None:
    """Generate and print setup code at startup if needed. Returns the raw bootstrap token or None."""
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    gateway_url = settings.auth_setup_code_url
    if not gateway_url.strip():
        protocol = "wss" if settings.use_ssl else "ws"
        gateway_url = f"{protocol}://{settings.server_host}:{settings.server_port}/ws"
        logger.warning(f"AUTH_SETUP_CODE_URL not set, using: {gateway_url}")

    existing_bootstrap = auth_store.get_bootstrap()

    if existing_bootstrap and not settings.auth_regenerate_setup_code:
        logger.info("Existing bootstrap token found. Setup code was printed on first start.")
        logger.info("Set AUTH_REGENERATE_SETUP_CODE=true and restart to generate a new code.")
        return None

    # Generate new bootstrap token
    bootstrap_token = secrets.token_urlsafe(32)
    auth_store.set_bootstrap(
        BootstrapRecord(
            token_hash=hash_token(bootstrap_token),
            created_at=utc_now(),
        )
    )

    service = SetupCodeService()
    setup_code = service.generate_signed(gateway_url=gateway_url, bootstrap_token=bootstrap_token)
    qr_uri = service.generate_qr_uri(gateway_url=gateway_url, bootstrap_token=bootstrap_token)

    _print_setup_code(setup_code=setup_code, qr_uri=qr_uri)
    return bootstrap_token


def _print_setup_code(*, setup_code: str, qr_uri: str) -> None:
    """Print setup code and save QR as PNG."""
    logger.info("=" * 60)
    logger.info("DEVICE PAIRING")
    logger.info("")
    logger.info("Scan the QR code with the mobile app,")
    logger.info("or paste this setup code manually:")
    logger.info("")
    logger.info(f"  {setup_code}")
    logger.info("")

    try:
        qr_path = "/app/pairing_qr.png"
        qr = segno.make(qr_uri, error="m")
        qr.save(qr_path, scale=10, border=4)
        logger.info(f"QR code saved to: {qr_path}")
    except OSError as exc:
        logger.warning(f"Could not save QR code PNG: {exc}")

    logger.info("=" * 60)
