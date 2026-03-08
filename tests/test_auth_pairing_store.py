"""Tests for pairing persistence behavior in InMemoryAuthStore."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.models import (  # noqa: E402
    ApprovedDeviceRecord,
    DeviceTokenRecord,
    PairingAuditEventRecord,
    PairingRequestRecord,
    utc_now,
)
from auth.store import InMemoryAuthStore  # noqa: E402


def test_pairing_request_roundtrip_and_status_filter() -> None:
    store = InMemoryAuthStore()
    now = utc_now()

    pending = PairingRequestRecord(
        request_id="pr-1",
        device_id="dev-1",
        public_key_fingerprint="sha256:abc",
        role="operator",
        requested_scopes=["chat.read"],
        status="pending",
        requested_at=now,
        expires_at=now + timedelta(hours=1),
    )
    approved = PairingRequestRecord(
        request_id="pr-2",
        device_id="dev-2",
        public_key_fingerprint="sha256:def",
        role="operator",
        requested_scopes=["chat.read"],
        status="approved",
        requested_at=now + timedelta(seconds=1),
        expires_at=now + timedelta(hours=1),
    )

    store.save_pairing_request(pending)
    store.save_pairing_request(approved)

    got = store.get_pairing_request("pr-1")
    assert got is not None
    assert got.device_id == "dev-1"

    only_pending = store.list_pairing_requests(status="pending")
    assert len(only_pending) == 1
    assert only_pending[0].request_id == "pr-1"


def test_approved_device_roundtrip_and_revoke() -> None:
    store = InMemoryAuthStore()
    now = utc_now()

    approved = ApprovedDeviceRecord(
        device_id="dev-1",
        public_key_fingerprint="sha256:abc",
        role="operator",
        approved_scopes=["chat.read", "chat.write"],
        approved_at=now,
        revoked=False,
    )
    store.upsert_approved_device(approved)

    loaded = store.get_approved_device("dev-1")
    assert loaded is not None
    assert loaded.revoked is False

    assert store.revoke_approved_device("dev-1") is True
    loaded2 = store.get_approved_device("dev-1")
    assert loaded2 is not None
    assert loaded2.revoked is True
    assert loaded2.revoked_at is not None


def test_device_token_roundtrip_and_revoke() -> None:
    store = InMemoryAuthStore()
    now = utc_now()

    token = DeviceTokenRecord(
        device_id="dev-1",
        role="operator",
        token_hash="hash-1",
        issued_at=now,
        expires_at=now + timedelta(days=30),
        revoked=False,
    )
    store.save_device_token(token)

    loaded = store.get_device_token("dev-1", "operator")
    assert loaded is not None
    assert loaded.token_hash == "hash-1"
    assert loaded.revoked is False

    assert store.revoke_device_token("dev-1", "operator") is True
    loaded2 = store.get_device_token("dev-1", "operator")
    assert loaded2 is not None
    assert loaded2.revoked is True


def test_pairing_audit_events_order_and_filter() -> None:
    store = InMemoryAuthStore()
    now = utc_now()

    store.add_pairing_audit_event(
        PairingAuditEventRecord(
            event_id="e1",
            event_type="pairing.requested",
            device_id="dev-1",
            request_id="pr-1",
            actor_id=None,
            reason=None,
            created_at=now,
        )
    )
    store.add_pairing_audit_event(
        PairingAuditEventRecord(
            event_id="e2",
            event_type="pairing.approved",
            device_id="dev-2",
            request_id="pr-2",
            actor_id="owner-1",
            reason=None,
            created_at=now + timedelta(seconds=1),
        )
    )

    latest_first = store.list_pairing_audit_events(limit=10)
    assert [event.event_id for event in latest_first] == ["e2", "e1"]

    dev2_only = store.list_pairing_audit_events(device_id="dev-2", limit=10)
    assert len(dev2_only) == 1
    assert dev2_only[0].event_id == "e2"
