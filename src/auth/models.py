"""Authentication domain models for WebSocket device auth."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DeviceRecord:
    device_id: str
    public_key: str
    algorithm: str
    status: str  # "active", "revoked"
    created_at: datetime


@dataclass
class DeviceTokenRecord:
    device_id: str
    token_hash: str
    issued_at: datetime
    revoked: bool = False


@dataclass
class BootstrapRecord:
    token_hash: str
    created_at: datetime


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)
