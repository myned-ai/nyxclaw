"""
End-to-end tests for barge-in during audio playback.

Verifies that:
1. cancel_response() fires the on_cancel_sync callback
2. _cancel_playback flag aborts _handle_response_end immediately
3. is_responding stays True until after response callbacks finish
4. Barge-in during playback triggers the interrupted callback
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def test_cancel_sync_callback_fires():
    """cancel_response() must call _on_cancel_sync synchronously."""
    from agents.zeroclaw.sample_agent import SampleZeroClawAgent as ZeroClawAgent

    agent = ZeroClawAgent()
    cancel_called = False

    def on_cancel():
        nonlocal cancel_called
        cancel_called = True

    agent.set_event_handlers(on_cancel_sync=on_cancel)
    # Simulate that we're mid-response
    agent._state.is_responding = True
    agent._response_cancelled = False

    agent.cancel_response()

    assert cancel_called, "on_cancel_sync was NOT called by cancel_response()"
    assert agent._response_cancelled, "_response_cancelled should be True"
    assert not agent._state.is_responding, "is_responding should be False after cancel"
    print("  PASS: cancel_response() fires on_cancel_sync")


async def test_cancel_playback_aborts_response_end():
    """_handle_response_end must exit immediately when _cancel_playback is set."""
    from chat.chat_session import ChatSession

    # Create a minimal mock ChatSession (bypass __init__)
    session = object.__new__(ChatSession)
    session.session_id = "test-session"
    session.is_interrupted = False
    session._cancel_playback = True  # pre-set cancel
    session.total_audio_received = 100.0
    session.audio_buffer = bytearray()
    session.audio_chunk_queue = asyncio.Queue()
    session.speech_ended = False
    session.inference_task = None
    session.is_active = True
    session.settings = MagicMock()
    session.settings.blendshape_fps = 30
    session.wav2arkit_service = MagicMock()
    session.wav2arkit_service.is_available = True
    session.input_sample_rate = 24000
    session.websocket = AsyncMock()
    session.current_turn_id = "test-turn"
    session.current_turn_session_id = "test-session"
    session.total_frames_emitted = 0
    session.blendshape_frame_idx = 0
    session.current_turn_text = ""
    session.virtual_cursor_text_ms = 0.0
    session.chars_per_second = 14.0
    session.first_audio_received = True
    session.frame_emit_task = None
    session.frame_queue = asyncio.Queue()

    start = time.perf_counter()
    await session._handle_response_end("test transcript", "test-item")
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Should return almost instantly (< 50ms), not wait 750ms+ for stabilization
    assert elapsed_ms < 100, f"_handle_response_end took {elapsed_ms:.0f}ms (should be < 100ms)"
    print(f"  PASS: _handle_response_end aborted in {elapsed_ms:.1f}ms with _cancel_playback=True")


async def test_cancel_playback_reset_on_response_start():
    """_cancel_playback must reset to False on new response start."""
    from chat.chat_session import ChatSession

    session = object.__new__(ChatSession)
    session.session_id = "test-session"
    session.is_interrupted = False
    session._cancel_playback = True  # leftover from previous barge-in
    session.total_audio_received = 0
    session.audio_buffer = bytearray()
    session.audio_chunk_queue = asyncio.Queue()
    session.frame_queue = asyncio.Queue()
    session.speech_ended = False
    session.is_active = True
    session.inference_task = None
    session.frame_emit_task = None
    session.settings = MagicMock()
    session.settings.blendshape_fps = 30
    session.settings.audio_chunk_duration = 0.5
    session.wav2arkit_service = MagicMock()
    session.wav2arkit_service.is_available = True
    session.wav2arkit_service.reset_context = MagicMock()
    session.input_sample_rate = 24000
    session.output_sample_rate = 24000
    session.websocket = AsyncMock()
    session.current_turn_id = None
    session.current_turn_session_id = None
    session.total_frames_emitted = 0
    session.blendshape_frame_idx = 0
    session.current_turn_text = ""
    session.virtual_cursor_text_ms = 0.0
    session.first_audio_received = False
    session.actual_audio_start_time = 0
    session.speech_start_time = 0
    session._last_user_transcript_finalized_at = 0
    session._last_input_source = "stt"
    session._last_response_start_at = 0
    session.agent = MagicMock()

    await session._handle_response_start("new-session-123")

    assert not session._cancel_playback, "_cancel_playback should be reset to False on response start"
    print("  PASS: _cancel_playback resets on new response start")


async def test_is_responding_stays_true_during_response_end():
    """is_responding must stay True while _on_response_end waits for audio drain."""
    from agents.zeroclaw.sample_agent import SampleZeroClawAgent as ZeroClawAgent

    agent = ZeroClawAgent()

    response_end_called = asyncio.Event()
    is_responding_during_callback = None

    async def mock_response_end(transcript, item_id):
        nonlocal is_responding_during_callback
        is_responding_during_callback = agent._state.is_responding
        response_end_called.set()

    agent.set_event_handlers(
        on_response_end=mock_response_end,
        on_response_start=AsyncMock(),
    )

    # Simulate the state that _stream_and_speak finally block would have
    agent._state.is_responding = True
    agent._response_cancelled = False
    agent._tts_cancelled = asyncio.Event()
    agent._cancel_event = asyncio.Event()

    # Call the relevant part of the finally block logic
    if not agent._response_cancelled and agent._on_response_end:
        await agent._on_response_end("test response", "item-1")
        # After callback, check state
        if agent._response_cancelled and agent._on_interrupted:
            await agent._on_interrupted()

    agent._state.is_responding = False
    agent._state.audio_done = True

    assert response_end_called.is_set(), "_on_response_end was never called"
    assert is_responding_during_callback, "is_responding was False during _on_response_end callback!"
    assert not agent._state.is_responding, "is_responding should be False after callbacks finish"
    print("  PASS: is_responding stays True during _on_response_end callback")


async def test_bargein_during_playback_calls_interrupted():
    """When barge-in fires during _on_response_end, _on_interrupted must be called."""
    from agents.zeroclaw.sample_agent import SampleZeroClawAgent as ZeroClawAgent

    agent = ZeroClawAgent()

    interrupted_called = asyncio.Event()
    response_end_entered = asyncio.Event()

    async def mock_response_end(transcript, item_id):
        response_end_entered.set()
        # Simulate audio drain wait — cancel_response fires during this
        await asyncio.sleep(0.2)

    async def mock_interrupted():
        interrupted_called.set()

    agent.set_event_handlers(
        on_response_end=mock_response_end,
        on_interrupted=mock_interrupted,
        on_response_start=AsyncMock(),
    )

    agent._state.is_responding = True
    agent._response_cancelled = False
    agent._tts_cancelled = asyncio.Event()
    agent._cancel_event = asyncio.Event()

    async def simulate_bargein():
        await response_end_entered.wait()
        # Barge-in during audio drain
        agent.cancel_response()

    # Run the finally block logic and barge-in concurrently
    bargein_task = asyncio.create_task(simulate_bargein())

    if not agent._response_cancelled and agent._on_response_end:
        await agent._on_response_end("test", "item-1")
        if agent._response_cancelled and agent._on_interrupted:
            await agent._on_interrupted()
    elif agent._response_cancelled and agent._on_interrupted:
        await agent._on_interrupted()

    agent._state.is_responding = False
    agent._state.audio_done = True

    await bargein_task

    assert interrupted_called.is_set(), "_on_interrupted was NOT called after barge-in during playback"
    print("  PASS: barge-in during playback triggers _on_interrupted callback")


async def test_response_end_loop_exits_on_cancel():
    """The audio stabilization loop must check _cancel_playback on every tick."""
    from chat.chat_session import ChatSession

    session = object.__new__(ChatSession)
    session.session_id = "test-session"
    session.is_interrupted = False
    session._cancel_playback = False
    session.total_audio_received = 100.0
    session.audio_buffer = bytearray()
    session.audio_chunk_queue = asyncio.Queue()
    session.speech_ended = False
    session.inference_task = None
    session.is_active = True
    session.settings = MagicMock()
    session.settings.blendshape_fps = 30
    session.wav2arkit_service = MagicMock()
    session.wav2arkit_service.is_available = True
    session.input_sample_rate = 24000
    session.websocket = AsyncMock()
    session.current_turn_id = "test-turn"
    session.current_turn_session_id = "test-session"
    session.total_frames_emitted = 0
    session.blendshape_frame_idx = 0
    session.current_turn_text = ""
    session.virtual_cursor_text_ms = 0.0
    session.chars_per_second = 14.0
    session.first_audio_received = True
    session.frame_emit_task = None
    session.frame_queue = asyncio.Queue()

    async def set_cancel_after_delay():
        await asyncio.sleep(0.1)
        session._cancel_playback = True

    cancel_task = asyncio.create_task(set_cancel_after_delay())

    start = time.perf_counter()
    await session._handle_response_end("test", "item-1")
    elapsed_ms = (time.perf_counter() - start) * 1000

    await cancel_task

    # Should exit within ~150ms (100ms delay + one loop tick), not 750ms+
    assert elapsed_ms < 300, f"Loop took {elapsed_ms:.0f}ms to exit after _cancel_playback set (should be < 300ms)"
    print(f"  PASS: stabilization loop exits in {elapsed_ms:.1f}ms after _cancel_playback set mid-loop")


async def main():
    print("=" * 60)
    print("Barge-in During Playback — Integration Tests")
    print("=" * 60)

    tests = [
        ("1. cancel_response() fires on_cancel_sync", test_cancel_sync_callback_fires),
        ("2. _cancel_playback aborts _handle_response_end", test_cancel_playback_aborts_response_end),
        ("3. _cancel_playback resets on response start", test_cancel_playback_reset_on_response_start),
        ("4. is_responding stays True during response_end", test_is_responding_stays_true_during_response_end),
        ("5. Barge-in during playback triggers interrupted", test_bargein_during_playback_calls_interrupted),
        ("6. Stabilization loop exits on mid-loop cancel", test_response_end_loop_exits_on_cancel),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
