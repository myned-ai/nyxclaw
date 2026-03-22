"""Cloudflare Tunnel provisioning service.

Handles device ID generation, tunnel provisioning via api.nyxclaw.ai,
and stale token detection. The tunnel itself is run by cloudflared
(Docker sidecar or local daemon) — this module manages the config.
"""

import json
import stat
import uuid
from pathlib import Path

import httpx

from core.logger import get_logger
from core.settings import get_settings

logger = get_logger(__name__)


class TunnelConfig:
    """Stored tunnel configuration (persisted to data/tunnel.json)."""

    def __init__(self, tunnel_token: str, hostname: str, tunnel_id: str) -> None:
        self.tunnel_token = tunnel_token
        self.hostname = hostname
        self.tunnel_id = tunnel_id

    def to_dict(self) -> dict:
        return {
            "tunnel_token": self.tunnel_token,
            "hostname": self.hostname,
            "tunnel_id": self.tunnel_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TunnelConfig":
        return cls(
            tunnel_token=data["tunnel_token"],
            hostname=data["hostname"],
            tunnel_id=data["tunnel_id"],
        )


def _resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    settings = get_settings()
    raw = Path(relative)
    if raw.is_absolute():
        return raw
    from core.settings import _CONFIG_DIR
    return _CONFIG_DIR / raw


def load_or_generate_device_id(device_id_path: str = "./data/device_id") -> str:
    """Load existing device ID or generate a new one.

    Device ID format: 8 hex characters from uuid4 (e.g. "a3f7b2c1").
    Stored as a plain text file.
    """
    path = _resolve_path(device_id_path)

    if path.exists():
        device_id = path.read_text().strip()
        if device_id:
            logger.info(f"Device ID: {device_id}")
            return device_id

    # Generate new ID
    device_id = uuid.uuid4().hex[:8]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(device_id)
    logger.info(f"Generated new device ID: {device_id}")
    return device_id


def load_tunnel_config(tunnel_config_path: str = "./data/tunnel.json") -> TunnelConfig | None:
    """Load stored tunnel configuration, or None if not provisioned."""
    path = _resolve_path(tunnel_config_path)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        config = TunnelConfig.from_dict(data)
        logger.info(f"Loaded tunnel config: {config.hostname}")
        return config
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning(f"Invalid tunnel config at {path}, will re-provision: {exc}")
        return None


def save_tunnel_config(config: TunnelConfig, tunnel_config_path: str = "./data/tunnel.json") -> None:
    """Save tunnel configuration to disk. Also writes the raw token file for cloudflared."""
    path = _resolve_path(tunnel_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    # Write plain token file for cloudflared to read
    token_path = path.parent / "tunnel_token"
    token_path.write_text(config.tunnel_token)
    token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    logger.info(f"Tunnel config saved: {config.hostname}")


def clear_tunnel_config(tunnel_config_path: str = "./data/tunnel.json") -> None:
    """Delete stored tunnel config and token file (triggers re-provisioning on next boot)."""
    path = _resolve_path(tunnel_config_path)
    token_path = path.parent / "tunnel_token"

    for f in (path, token_path):
        if f.exists():
            f.unlink()
            logger.info(f"Deleted {f}")


async def provision_tunnel(
    device_id: str,
    provisioning_api_url: str = "https://api.nyxclaw.ai",
) -> TunnelConfig:
    """Call the provisioning API to create a Cloudflare Tunnel.

    The API is idempotent — same device_id returns the same tunnel.
    """
    url = f"{provisioning_api_url.rstrip('/')}/provision"
    logger.info(f"Provisioning tunnel for device {device_id}...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json={"device_id": device_id})
        response.raise_for_status()
        data = response.json()

    config = TunnelConfig(
        tunnel_token=data["tunnel_token"],
        hostname=data["hostname"],
        tunnel_id=data["tunnel_id"],
    )
    logger.info(f"Tunnel provisioned: {config.hostname}")
    return config


async def check_tunnel_status(
    device_id: str,
    provisioning_api_url: str = "https://api.nyxclaw.ai",
) -> bool:
    """Check if the tunnel still exists on Cloudflare. Returns True if active."""
    url = f"{provisioning_api_url.rstrip('/')}/provision/{device_id}/status"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 404:
                logger.warning(f"Tunnel for {device_id} no longer exists (cleaned up)")
                return False
            response.raise_for_status()
            return True
    except httpx.HTTPError as exc:
        logger.warning(f"Tunnel status check failed: {exc}")
        # Assume tunnel is OK if we can't reach the API (network issue)
        return True


async def ensure_tunnel(
    device_id_path: str = "./data/device_id",
    tunnel_config_path: str = "./data/tunnel.json",
    provisioning_api_url: str = "https://api.nyxclaw.ai",
) -> TunnelConfig | None:
    """Full provisioning flow: load or generate device ID, load or provision tunnel.

    Returns TunnelConfig if successful, None if provisioning failed (server
    should fall back to local-only mode).
    """
    device_id = load_or_generate_device_id(device_id_path)

    # Try loading existing config
    config = load_tunnel_config(tunnel_config_path)

    if config:
        # Verify the tunnel still exists (may have been cleaned up after 5 days inactive)
        if await check_tunnel_status(device_id, provisioning_api_url):
            return config
        else:
            logger.info("Stored tunnel was cleaned up, re-provisioning...")
            clear_tunnel_config(tunnel_config_path)

    # Provision new tunnel
    try:
        config = await provision_tunnel(device_id, provisioning_api_url)
        save_tunnel_config(config, tunnel_config_path)
        return config
    except httpx.HTTPError as exc:
        logger.error(f"Tunnel provisioning failed: {exc}")
        logger.warning("Server will start in local-only mode (no wss:// access)")
        return None
