"""Setup code generation — produces a signed, base64url-encoded setup code.

The setup code encodes the server's WebSocket URL and a bootstrap token.
Clients decode it to discover the server and pair for the first time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from auth.models import BootstrapRecord, utc_now
from auth.store import AuthStore, hash_token
from core.logger import get_logger
from core.settings import get_settings

logger = get_logger(__name__)


class SetupCodeService:
    def __init__(self, secret: str) -> None:
        if not secret.strip():
            raise ValueError("Setup code secret must not be empty")
        self._secret = secret.encode("utf-8")

    def generate(self, *, gateway_url: str, bootstrap_token: str) -> str:
        """Generate a signed setup code containing the gateway URL and bootstrap token."""
        now = int(time.time())
        payload = {
            "issuedAt": now,
            "token": bootstrap_token.strip(),
            "url": gateway_url.strip(),
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        signature = self._sign(payload_json)
        envelope = {"v": 1, "p": payload, "s": signature}
        envelope_json = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
        return base64.urlsafe_b64encode(envelope_json.encode("utf-8")).rstrip(b"=").decode("utf-8")

    def _sign(self, payload_json: str) -> str:
        digest = hmac.new(self._secret, payload_json.encode("utf-8"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def ensure_setup_code(auth_store: AuthStore) -> str | None:
    """Generate and print setup code at startup if needed. Returns the raw bootstrap token or None."""
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    secret = settings.auth_setup_code_secret
    if not secret.strip():
        logger.warning("AUTH_SETUP_CODE_SECRET is not set — cannot generate setup code")
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
    auth_store.set_bootstrap(BootstrapRecord(
        token_hash=hash_token(bootstrap_token),
        created_at=utc_now(),
    ))

    service = SetupCodeService(secret=secret)
    setup_code = service.generate(gateway_url=gateway_url, bootstrap_token=bootstrap_token)

    _print_setup_code(setup_code)
    return bootstrap_token


def _print_setup_code(setup_code: str) -> None:
    """Print setup code to logs, with QR code if qrcode library is available."""
    logger.info("=" * 60)
    logger.info("SETUP CODE (copy-paste or scan QR):")
    logger.info("")
    logger.info(f"  {setup_code}")
    logger.info("")

    try:
        import qrcode  # type: ignore[import-untyped]

        qr = qrcode.QRCode(border=1)
        qr.add_data(setup_code)
        qr.make(fit=True)
        from io import StringIO

        buf = StringIO()
        qr.print_ascii(out=buf, invert=True)
        for line in buf.getvalue().splitlines():
            logger.info(line)
    except ImportError:
        logger.info("(Install 'qrcode' package to also display a scannable QR code)")

    logger.info("")
    logger.info("=" * 60)
