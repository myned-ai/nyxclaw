import asyncio
import uuid

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect

from auth.device_verifier import get_device_verifier
from auth.store import AuthStore, get_auth_store
from auth.ws_auth import authenticate_websocket
from chat import ConnectionManager, get_connection_manager
from core.logger import get_logger
from core.settings import Settings, get_settings
from services import Wav2ArkitService, get_wav2arkit_service
from services.conversation_store import get_conversation_store

logger = get_logger(__name__)

chat_router = APIRouter(tags=["chat"])


def _verify_bearer(
    auth_store: AuthStore,
    settings: Settings,
    authorization: str | None,
) -> None:
    """Verify Bearer token against the auth store. Raises 401 on failure."""
    if not settings.auth_enabled:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[7:]
    # Check against all registered device tokens
    for device_id in auth_store.list_device_ids():
        if auth_store.verify_device_token(device_id, token):
            return
    raise HTTPException(status_code=401, detail="Invalid token")


@chat_router.get("/conversations")
async def get_conversations(
    limit: int = 20,
    before: float | None = None,
    settings: Settings = Depends(get_settings),
    auth_store: AuthStore = Depends(get_auth_store),
    authorization: str | None = Header(None),
) -> list[dict]:
    """Return recent conversation history.

    Args:
        limit: Max entries to return (default 50, max 200).
        before: Unix timestamp — only return entries older than this (for pagination).

    Returns entries oldest-first. Requires Bearer token auth when auth is enabled.
    """
    _verify_bearer(auth_store, settings, authorization)
    store = get_conversation_store()
    capped_limit = max(1, min(limit, 200))
    await asyncio.to_thread(store.trim)
    return await asyncio.to_thread(store.get_recent, capped_limit, before)


@chat_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    auth_store: AuthStore = Depends(get_auth_store),
    wav2arkit_service: Wav2ArkitService = Depends(get_wav2arkit_service),
    chat_connection_manager: ConnectionManager = Depends(get_connection_manager),
) -> None:
    session_id = str(uuid.uuid4())
    auth_context: dict[str, str] | None = None

    # Accept the WebSocket first (auth happens inside the connection via challenge-response)
    await websocket.accept()

    if settings.auth_enabled:
        verifier = get_device_verifier()
        auth_context = await authenticate_websocket(
            websocket=websocket,
            auth_store=auth_store,
            verifier=verifier,
        )
        if auth_context is None:
            return  # auth failed — ws_auth already closed the connection

    # Auth passed (or disabled) — start the chat session
    session = await chat_connection_manager.connect(
        websocket,
        session_id,
        settings,
        wav2arkit_service,
        auth_context=auth_context,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            # Guard against oversized messages (1MB limit)
            if len(raw) > 1_048_576:
                logger.warning(f"Session {session_id}: Message too large ({len(raw)} bytes), dropping")
                continue
            try:
                data = orjson.loads(raw)
            except Exception:
                logger.warning(f"Session {session_id}: Invalid JSON message, dropping")
                continue
            await chat_connection_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        await chat_connection_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        await chat_connection_manager.disconnect(websocket)
