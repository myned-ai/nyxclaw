"""Authentication persistence — JSON file-backed store."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

from auth.models import BootstrapRecord, DeviceRecord, DeviceTokenRecord, utc_now
from core.logger import get_logger
from core.settings import get_settings

logger = get_logger(__name__)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    """Thread-safe auth store backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = Lock()
        self._devices: dict[str, DeviceRecord] = {}
        self._device_tokens: dict[str, DeviceTokenRecord] = {}
        self._bootstrap: BootstrapRecord | None = None
        # Ephemeral — not persisted
        self._challenge_nonces: dict[str, _ChallengeNonce] = {}
        self._load()

    # ── Device operations ─────────────────────────────────────────────

    def upsert_device(self, record: DeviceRecord) -> DeviceRecord:
        with self._lock:
            self._devices[record.device_id] = record
            self._save()
            return record

    def get_device(self, device_id: str) -> DeviceRecord | None:
        with self._lock:
            return self._devices.get(device_id)

    def revoke_device(self, device_id: str) -> bool:
        with self._lock:
            device = self._devices.get(device_id)
            if not device:
                return False
            device.status = "revoked"
            token = self._device_tokens.get(device_id)
            if token:
                token.revoked = True
            self._save()
            return True

    # ── Device token operations ───────────────────────────────────────

    def save_device_token(self, record: DeviceTokenRecord) -> None:
        with self._lock:
            self._device_tokens[record.device_id] = record
            self._save()

    def get_device_token(self, device_id: str) -> DeviceTokenRecord | None:
        with self._lock:
            return self._device_tokens.get(device_id)

    def verify_device_token(self, device_id: str, token: str) -> bool:
        with self._lock:
            record = self._device_tokens.get(device_id)
            if not record or record.revoked:
                return False
            return record.token_hash == hash_token(token)

    def list_device_ids(self) -> list[str]:
        with self._lock:
            return list(self._device_tokens.keys())

    # ── Bootstrap token operations ────────────────────────────────────

    def get_bootstrap(self) -> BootstrapRecord | None:
        with self._lock:
            return self._bootstrap

    def set_bootstrap(self, record: BootstrapRecord) -> None:
        with self._lock:
            self._bootstrap = record
            self._save()

    def verify_bootstrap_token(self, token: str) -> bool:
        with self._lock:
            if not self._bootstrap:
                return False
            return self._bootstrap.token_hash == hash_token(token)

    # ── Challenge nonce operations (ephemeral) ────────────────────────

    def save_challenge_nonce(self, nonce_id: str, nonce: str, expires_at_ms: int) -> None:
        with self._lock:
            self._challenge_nonces[nonce_id] = _ChallengeNonce(nonce=nonce, expires_at_ms=expires_at_ms, used=False)

    def consume_challenge_nonce(self, nonce_id: str) -> str | None:
        with self._lock:
            entry = self._challenge_nonces.pop(nonce_id, None)
            if not entry or entry.used:
                return None
            import time
            if int(time.time() * 1000) > entry.expires_at_ms:
                return None
            entry.used = True
            return entry.nonce

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text("utf-8"))
            for did, d in raw.get("devices", {}).items():
                self._devices[did] = DeviceRecord(
                    device_id=d["device_id"],
                    public_key=d["public_key"],
                    algorithm=d["algorithm"],
                    status=d["status"],
                    created_at=datetime.fromisoformat(d["created_at"]),
                )
            for did, t in raw.get("device_tokens", {}).items():
                self._device_tokens[did] = DeviceTokenRecord(
                    device_id=t["device_id"],
                    token_hash=t["token_hash"],
                    issued_at=datetime.fromisoformat(t["issued_at"]),
                    revoked=t.get("revoked", False),
                )
            bs = raw.get("bootstrap")
            if bs:
                self._bootstrap = BootstrapRecord(
                    token_hash=bs["token_hash"],
                    created_at=datetime.fromisoformat(bs["created_at"]),
                )
            logger.info(f"Auth store loaded: {len(self._devices)} devices, {len(self._device_tokens)} tokens")
        except Exception:
            logger.warning("Failed to load auth store, starting fresh", exc_info=True)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "devices": {
                did: {
                    "device_id": d.device_id,
                    "public_key": d.public_key,
                    "algorithm": d.algorithm,
                    "status": d.status,
                    "created_at": d.created_at.isoformat(),
                }
                for did, d in self._devices.items()
            },
            "device_tokens": {
                did: {
                    "device_id": t.device_id,
                    "token_hash": t.token_hash,
                    "issued_at": t.issued_at.isoformat(),
                    "revoked": t.revoked,
                }
                for did, t in self._device_tokens.items()
            },
        }
        if self._bootstrap:
            data["bootstrap"] = {
                "token_hash": self._bootstrap.token_hash,
                "created_at": self._bootstrap.created_at.isoformat(),
            }
        self._path.write_text(json.dumps(data, indent=2), "utf-8")


class _ChallengeNonce:
    __slots__ = ("nonce", "expires_at_ms", "used")

    def __init__(self, nonce: str, expires_at_ms: int, used: bool) -> None:
        self.nonce = nonce
        self.expires_at_ms = expires_at_ms
        self.used = used


@lru_cache(maxsize=1)
def get_auth_store() -> AuthStore:
    settings = get_settings()
    return AuthStore(path=settings.auth_store_path)
