"""Tests that legacy /api/auth/token endpoint is removed."""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

from fastapi import APIRouter
from fastapi.testclient import TestClient

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_MAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "main.py"
_ROUTERS_STUB = types.ModuleType("routers")
_ROUTERS_STUB.auth_router = APIRouter()
_ROUTERS_STUB.chat_router = APIRouter()
sys.modules.setdefault("routers", _ROUTERS_STUB)

_MAIN_SPEC = importlib.util.spec_from_file_location("main_module_for_test", _MAIN_PATH)
if _MAIN_SPEC is None or _MAIN_SPEC.loader is None:
    raise RuntimeError("Failed to load main module for tests")
main = importlib.util.module_from_spec(_MAIN_SPEC)
_MAIN_SPEC.loader.exec_module(main)


def test_legacy_token_endpoint_removed() -> None:
    client = TestClient(main.app)
    response = client.post("/api/auth/token", headers={"origin": "http://localhost:5173"})

    assert response.status_code == 404
