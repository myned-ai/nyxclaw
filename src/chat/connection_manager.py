from functools import lru_cache
from typing import Any

from fastapi import WebSocket

from core.logger import get_logger
from core.settings import Settings
from services import Wav2ArkitService

from .chat_session import ChatSession

logger = get_logger(__name__)


class ConnectionManager:
    """
    Registry for active chat sessions.
    """

    def __init__(self) -> None:
        self.sessions: dict[WebSocket, ChatSession] = {}

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        settings: Settings,
        wav2arkit_service: Wav2ArkitService,
        auth_context: dict[str, str] | None = None,
    ) -> ChatSession:
        # WebSocket is already accepted by the router (auth handshake happens before this)
        session = ChatSession(websocket, session_id, settings, wav2arkit_service, auth_context=auth_context)
        self.sessions[websocket] = session

        await session.start()
        logger.info(f"Session started: {session_id}")
        return session

    async def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.sessions:
            session = self.sessions[websocket]
            await session.stop()
            del self.sessions[websocket]
            logger.info(f"Client disconnected: {session.session_id}")

    async def handle_message(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        if websocket in self.sessions:
            await self.sessions[websocket].process_message(data)


@lru_cache
def get_connection_manager() -> ConnectionManager:
    """
    Get the singleton connection manager.
    LRU cache ensures we always get the same instance.
    """
    return ConnectionManager()
