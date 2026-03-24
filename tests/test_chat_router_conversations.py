"""Tests for GET /conversations endpoint in routers.chat_router."""

import importlib
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from auth.models import DeviceTokenRecord
from auth.store import AuthStore, hash_token
from core.settings import Settings
from services.conversation_store import ConversationStore

# Import the actual module object (not the re-exported APIRouter from routers/__init__)
_router_module = importlib.import_module("routers.chat_router")


def _make_app(settings: Settings, auth_store: AuthStore) -> FastAPI:
    """Build a minimal FastAPI app with dependency overrides for testing."""
    from auth.store import get_auth_store
    from core.settings import get_settings

    app = FastAPI()
    app.include_router(_router_module.chat_router)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_auth_store] = lambda: auth_store
    return app


@pytest.fixture
def settings_auth_off():
    s = Settings.__new__(Settings)
    object.__setattr__(s, "auth_enabled", False)
    return s


@pytest.fixture
def settings_auth_on():
    s = Settings.__new__(Settings)
    object.__setattr__(s, "auth_enabled", True)
    return s


@pytest.fixture
def auth_store(tmp_path):
    return AuthStore(path=tmp_path / "auth.json")


@pytest.mark.asyncio(loop_scope="function")
class TestGetConversations:
    async def test_returns_entries_no_auth(self, tmp_path, settings_auth_off, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        conv_store.append("user", "hello")
        conv_store.append("assistant", "hi there")

        app = _make_app(settings_auth_off, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["text"] == "hello"
        assert data[1]["text"] == "hi there"

    async def test_limit_caps_at_200(self, tmp_path, settings_auth_off, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        for i in range(5):
            conv_store.append("user", f"msg-{i}")

        app = _make_app(settings_auth_off, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", params={"limit": 999})

        assert resp.status_code == 200
        assert len(resp.json()) == 5  # only 5 exist, capped at 200

    async def test_limit_minimum_is_1(self, tmp_path, settings_auth_off, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        for i in range(5):
            conv_store.append("user", f"msg-{i}")

        app = _make_app(settings_auth_off, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", params={"limit": -10})

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_before_pagination(self, tmp_path, settings_auth_off, auth_store):
        path = tmp_path / "conv.jsonl"
        conv_store = ConversationStore(path=str(path))
        for ts in [100.0, 200.0, 300.0]:
            entry = {"ts": ts, "role": "user", "text": f"ts-{int(ts)}"}
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        app = _make_app(settings_auth_off, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", params={"before": 250.0})

        data = resp.json()
        assert len(data) == 2
        assert data[-1]["text"] == "ts-200"

    async def test_auth_required_rejects_missing_header(self, tmp_path, settings_auth_on, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        app = _make_app(settings_auth_on, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations")

        assert resp.status_code == 401

    async def test_auth_required_rejects_bad_token(self, tmp_path, settings_auth_on, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        app = _make_app(settings_auth_on, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", headers={"Authorization": "Bearer wrong-token"})

        assert resp.status_code == 401

    async def test_auth_accepts_valid_token(self, tmp_path, settings_auth_on, auth_store):
        now = datetime.now(tz=timezone.utc)
        token = "valid-secret-token"
        auth_store.save_device_token(
            DeviceTokenRecord(device_id="dev-1", token_hash=hash_token(token), issued_at=now)
        )

        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        conv_store.append("user", "authed message")

        app = _make_app(settings_auth_on, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_empty_store_returns_empty_list(self, tmp_path, settings_auth_off, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        app = _make_app(settings_auth_off, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations")

        assert resp.status_code == 200
        assert resp.json() == []

    async def test_auth_rejects_non_bearer_scheme(self, tmp_path, settings_auth_on, auth_store):
        conv_store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        app = _make_app(settings_auth_on, auth_store)

        with patch.object(_router_module, "get_conversation_store", return_value=conv_store):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/conversations", headers={"Authorization": "Basic abc123"})

        assert resp.status_code == 401
