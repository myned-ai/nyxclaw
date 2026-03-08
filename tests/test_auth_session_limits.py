"""Tests for per-device concurrent session limits."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import SessionRecord, utc_now
from auth.store import InMemoryAuthStore


def _session(session_id: str, device_id: str, created_offset_sec: int) -> SessionRecord:
    base = utc_now()
    return SessionRecord(
        session_id=session_id,
        device_id=device_id,
        created_at=base + timedelta(seconds=created_offset_sec),
        refresh_expires_at=base + timedelta(days=1),
        refresh_jti=f"jti-{session_id}",
        refresh_token_hash=f"hash-{session_id}",
        revoked=False,
    )


@pytest.mark.asyncio
async def test_enforce_max_sessions_revokes_oldest_overflow() -> None:
    store = InMemoryAuthStore()
    store.save_session(_session("s1", "d1", 1))
    store.save_session(_session("s2", "d1", 2))
    store.save_session(_session("s3", "d1", 3))

    revoked = store.enforce_max_sessions_per_device("d1", max_active_sessions=2)

    assert revoked == 1
    assert store.get_session("s1").revoked is True
    assert store.get_session("s2").revoked is False
    assert store.get_session("s3").revoked is False


@pytest.mark.asyncio
async def test_enforce_max_sessions_is_scoped_per_device() -> None:
    store = InMemoryAuthStore()
    store.save_session(_session("s1", "d1", 1))
    store.save_session(_session("s2", "d1", 2))
    store.save_session(_session("s3", "d2", 3))

    revoked = store.enforce_max_sessions_per_device("d1", max_active_sessions=1)

    assert revoked == 1
    assert store.get_session("s1").revoked is True
    assert store.get_session("s2").revoked is False
    assert store.get_session("s3").revoked is False


@pytest.mark.asyncio
async def test_enforce_max_sessions_noop_when_under_limit() -> None:
    store = InMemoryAuthStore()
    store.save_session(_session("s1", "d1", 1))

    revoked = store.enforce_max_sessions_per_device("d1", max_active_sessions=2)

    assert revoked == 0
    assert store.get_session("s1").revoked is False
