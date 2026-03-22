import uuid

import orjson
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from auth.device_verifier import get_device_verifier
from auth.store import AuthStore, get_auth_store
from auth.ws_auth import authenticate_websocket
from chat import ConnectionManager, get_connection_manager
from core.logger import get_logger
from core.settings import Settings, get_settings
from services import Wav2ArkitService, get_wav2arkit_service

logger = get_logger(__name__)

chat_router = APIRouter(tags=["chat"])


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
