"""Tests for rich content features in ChatSession and backends.

Covers:
- _handle_tool_call with rich_content forwarding
- _handle_tool_call ignoring non-rich_content tool calls
- client_event handling (context directive, speak directive, unknown)
- Filler skip logic in TTS worker
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Mock heavy optional deps before they get imported by the services chain
for _mod in ("numpy", "onnxruntime"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from backend.base_agent import TOOL_FILLERS
from chat.chat_session import ChatSession


# ── Helpers ───────────────────────────────────────────────────────


def _make_chat_session():
    """Build a minimal ChatSession with mocked dependencies."""
    session = ChatSession.__new__(ChatSession)

    settings = MagicMock()
    settings.blendshape_fps = 30
    settings.agent_type = "zeroclaw"
    settings.voice_backend = "piper"

    session.ws = AsyncMock()
    session.settings = settings
    session.session_id = "test-session"
    session._lock = asyncio.Lock()
    session.agent = MagicMock()
    session.agent.handle_client_event = AsyncMock()
    session.send_json = AsyncMock()

    return session


# ================================================================
# _handle_tool_call tests
# ================================================================


class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_rich_content_forwarded_to_client(self):
        """rich_content tool call sends rich_content message to client."""
        session = _make_chat_session()

        await session._handle_tool_call("rich_content", {"content": "## Hello\nSome markdown"})

        session.send_json.assert_called_once()
        sent = session.send_json.call_args[0][0]
        assert sent["type"] == "rich_content"
        assert sent["content"] == "## Hello\nSome markdown"

    @pytest.mark.asyncio
    async def test_rich_content_empty_content_not_sent(self):
        """rich_content with empty content string should not send anything."""
        session = _make_chat_session()

        await session._handle_tool_call("rich_content", {"content": ""})

        session.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_rich_content_missing_content_key_not_sent(self):
        """rich_content with no content key should not send anything."""
        session = _make_chat_session()

        await session._handle_tool_call("rich_content", {})

        session.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_rich_content_tool_not_forwarded(self):
        """Other tool calls are logged but not forwarded to client."""
        session = _make_chat_session()

        await session._handle_tool_call("web_search", {"query": "weather NYC"})

        session.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_rich_content_tool_not_forwarded(self):
        """Legacy send_rich_content tool name is no longer forwarded."""
        session = _make_chat_session()

        await session._handle_tool_call("send_rich_content", {"content_type": "link_card"})

        session.send_json.assert_not_called()


# ================================================================
# handle_client_event on ZeroClawBackend
# ================================================================


class TestZeroClawClientEvent:
    def _make_zeroclaw_backend(self):
        """Minimal ZeroClawBackend for testing handle_client_event."""
        with (
            patch("backend.zeroclaw.backend.get_settings") as mock_settings,
            patch("backend.zeroclaw.backend.get_zeroclaw_settings") as mock_zc,
        ):
            settings = MagicMock()
            settings.agent_type = "zeroclaw"
            mock_settings.return_value = settings

            zc = MagicMock()
            zc.zeroclaw_ws_url = "ws://localhost:8000"
            zc.zeroclaw_api_key = "test"
            zc.zeroclaw_system_prompt = "You are helpful."
            zc.zeroclaw_model = "test-model"
            zc.history_max_messages = 20
            mock_zc.return_value = zc

            from backend.zeroclaw.backend import ZeroClawBackend

            backend = ZeroClawBackend()

        backend._messages = []
        backend._ws = MagicMock()
        backend._ws.send = MagicMock()
        backend.send_text_message = MagicMock()
        backend._append_message = MagicMock()

        return backend

    @pytest.mark.asyncio
    async def test_context_directive_absorbs_silently(self):
        backend = self._make_zeroclaw_backend()

        await backend.handle_client_event(
            name="location_update",
            data={"lat": 40.7, "lon": -74.0},
            directive="context",
        )

        backend._append_message.assert_called_once()
        call_args = backend._append_message.call_args
        assert call_args[0][0] == "user"
        assert "location_update" in call_args[0][1]
        backend.send_text_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_directive_sends_message(self):
        backend = self._make_zeroclaw_backend()

        await backend.handle_client_event(
            name="user_action",
            data={"action": "tapped_card"},
            directive="speak",
        )

        backend.send_text_message.assert_called_once()
        msg = backend.send_text_message.call_args[0][0]
        assert "user_action" in msg
        backend._append_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_directive_sends_message(self):
        backend = self._make_zeroclaw_backend()

        await backend.handle_client_event(
            name="app_event",
            directive="trigger",
        )

        backend.send_text_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_directive_sends_message(self):
        """When directive is None, treat as speak (non-silent)."""
        backend = self._make_zeroclaw_backend()

        await backend.handle_client_event(name="some_event")

        backend.send_text_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_attachments_included_in_context(self):
        backend = self._make_zeroclaw_backend()

        await backend.handle_client_event(
            name="file_shared",
            directive="context",
            attachments=[{"type": "image", "url": "https://example.com/img.jpg"}],
        )

        msg_text = backend._append_message.call_args[0][1]
        assert "attachments=" in msg_text


# ================================================================
# Filler skip logic
# ================================================================


class TestFillerSkipLogic:
    """Test that filler phrases are skipped when real content is already queued."""

    @pytest.mark.asyncio
    async def test_filler_skipped_when_queue_has_content(self):
        """Filler item should be skipped if the queue already has more items."""
        with (
            patch("voice.openai_realtime.backend.get_settings") as mock_settings,
            patch("voice.openai_realtime.backend.get_openai_realtime_settings") as mock_rt,
        ):
            mock_settings.return_value = MagicMock(agent_type="openclaw")
            rt = MagicMock()
            rt.openai_tts_model = "tts-1"
            rt.openai_voice = "alloy"
            rt.openai_tts_speed = 1.0
            rt.history_max_messages = 20
            mock_rt.return_value = rt

            from voice.openai_realtime.backend import OpenAIRealtimeBackend

            backend = OpenAIRealtimeBackend()

        # Mock TTS client
        from tests.test_openai_tts_worker import FakeStreamContext

        audio_data = b"\x00" * 4800
        mock_create = MagicMock(return_value=FakeStreamContext([audio_data]))
        mock_openai = MagicMock()
        mock_openai.audio.speech.with_streaming_response.create = mock_create
        backend._openai = mock_openai

        backend._on_audio_delta = AsyncMock()
        backend._on_transcript_delta = AsyncMock()

        # Queue: filler + real content + sentinel
        # The filler should be skipped because real content is already behind it
        queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
        await queue.put(("On it.", "On it.", True))  # filler
        await queue.put(("Here is the answer.", "Here is the answer.", False))  # real
        await queue.put(None)

        await backend._tts_worker(queue, "item_1")

        # Transcript should only have the real content, not the filler
        transcript_calls = backend._on_transcript_delta.call_args_list
        texts = [c[0][0] for c in transcript_calls]
        assert "On it." not in texts
        assert "Here is the answer." in texts

    @pytest.mark.asyncio
    async def test_filler_skipped_when_only_sentinel_remains(self):
        """Filler is skipped even when only the None sentinel remains in the queue,
        because queue.empty() checks raw queue size (sentinel counts)."""
        with (
            patch("voice.openai_realtime.backend.get_settings") as mock_settings,
            patch("voice.openai_realtime.backend.get_openai_realtime_settings") as mock_rt,
        ):
            mock_settings.return_value = MagicMock(agent_type="openclaw")
            rt = MagicMock()
            rt.openai_tts_model = "tts-1"
            rt.openai_voice = "alloy"
            rt.openai_tts_speed = 1.0
            rt.history_max_messages = 20
            mock_rt.return_value = rt

            from voice.openai_realtime.backend import OpenAIRealtimeBackend

            backend = OpenAIRealtimeBackend()

        from tests.test_openai_tts_worker import FakeStreamContext

        audio_data = b"\x00" * 4800
        mock_create = MagicMock(return_value=FakeStreamContext([audio_data]))
        mock_openai = MagicMock()
        mock_openai.audio.speech.with_streaming_response.create = mock_create
        backend._openai = mock_openai

        backend._on_audio_delta = AsyncMock()
        backend._on_transcript_delta = AsyncMock()

        # Queue: filler + sentinel. After dequeuing filler, sentinel is still
        # in the queue so queue.empty() is False -> filler is skipped.
        queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
        await queue.put(("On it.", "On it.", True))  # filler
        await queue.put(None)

        await backend._tts_worker(queue, "item_1")

        # Filler is skipped (sentinel makes queue non-empty), so no audio or transcript
        backend._on_transcript_delta.assert_not_called()
        backend._on_audio_delta.assert_not_called()


# ================================================================
# TOOL_FILLERS constant
# ================================================================


class TestToolFillers:
    def test_fillers_are_non_empty_strings(self):
        assert len(TOOL_FILLERS) > 0
        for filler in TOOL_FILLERS:
            assert isinstance(filler, str)
            assert len(filler) > 0

    def test_fillers_end_with_period(self):
        for filler in TOOL_FILLERS:
            assert filler.endswith("."), f"Filler {filler!r} does not end with period"
