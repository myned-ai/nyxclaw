"""Tests for auth.store — list_device_ids method."""

from datetime import datetime, timezone

from auth.models import DeviceTokenRecord
from auth.store import AuthStore, hash_token


class TestListDeviceIds:
    def test_returns_empty_when_no_tokens(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        assert store.list_device_ids() == []

    def test_returns_device_ids_with_tokens(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        now = datetime.now(tz=timezone.utc)

        store.save_device_token(
            DeviceTokenRecord(device_id="device-a", token_hash=hash_token("tok-a"), issued_at=now)
        )
        store.save_device_token(
            DeviceTokenRecord(device_id="device-b", token_hash=hash_token("tok-b"), issued_at=now)
        )

        ids = store.list_device_ids()
        assert sorted(ids) == ["device-a", "device-b"]

    def test_returns_ids_even_if_token_revoked(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        now = datetime.now(tz=timezone.utc)

        store.save_device_token(
            DeviceTokenRecord(device_id="dev-1", token_hash=hash_token("t"), issued_at=now, revoked=True)
        )

        assert store.list_device_ids() == ["dev-1"]
