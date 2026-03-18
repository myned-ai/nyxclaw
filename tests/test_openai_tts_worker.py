"""
Tests for OpenAIRealtimeBackend._tts_worker.

Covers the three key changes:
1. Transcript emission happens BEFORE TTS synthesis (not inside audio loop)
2. Audio re-chunking: variable-size HTTP chunks -> fixed 4800-byte pieces
3. Flush of remaining buffer after stream ends
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Fake streaming response objects ─────────────────────────────────


class FakeStreamResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class FakeStreamContext:
    def __init__(self, chunks: list[bytes]):
        self._response = FakeStreamResponse(chunks)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


# ── Helper to build a minimally-mocked backend ──────────────────────


def _make_backend(tts_chunks: list[bytes] | None = None):
    """
    Build an OpenAIRealtimeBackend with mocked OpenAI client and callbacks.
    Does NOT call connect() — we only need the _tts_worker method.
    """
    if tts_chunks is None:
        tts_chunks = [b"\x00" * 4800]

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

    # Mock OpenAI client's TTS streaming endpoint
    mock_create = MagicMock(return_value=FakeStreamContext(tts_chunks))
    mock_openai = MagicMock()
    mock_openai.audio.speech.with_streaming_response.create = mock_create
    backend._openai = mock_openai

    # Mock callbacks
    backend._on_audio_delta = AsyncMock()
    backend._on_transcript_delta = AsyncMock()

    return backend


DELIVERY_BYTES = 4800  # 100ms @ 24kHz PCM16 mono


# ================================================================
# Tests
# ================================================================


@pytest.mark.asyncio
async def test_transcript_emitted_before_audio():
    """_on_transcript_delta must be called BEFORE any _on_audio_delta for each sentence."""
    audio_data = b"\x01" * DELIVERY_BYTES
    backend = _make_backend(tts_chunks=[audio_data])

    call_order: list[str] = []
    backend._on_transcript_delta = AsyncMock(side_effect=lambda *_a: call_order.append("transcript"))
    backend._on_audio_delta = AsyncMock(side_effect=lambda *_a: call_order.append("audio"))

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Hello.", "Hello.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    assert call_order[0] == "transcript", f"Expected transcript first, got: {call_order}"
    assert "audio" in call_order, "Expected at least one audio callback"


@pytest.mark.asyncio
async def test_rechunk_large_chunks():
    """Large TTS chunks (e.g. 16KB) must be re-chunked into 4800-byte pieces."""
    big_chunk = b"\xAB" * 16000  # not a multiple of 4800
    backend = _make_backend(tts_chunks=[big_chunk])

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Test.", "Test.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    audio_calls = backend._on_audio_delta.call_args_list
    # 16000 / 4800 = 3 full chunks + 1600 remainder
    # Plus a flush call for the 1600-byte remainder
    full_chunks = [c for c in audio_calls if len(c[0][0]) == DELIVERY_BYTES]
    flush_chunks = [c for c in audio_calls if 0 < len(c[0][0]) < DELIVERY_BYTES]

    assert len(full_chunks) == 3, f"Expected 3 full 4800-byte chunks, got {len(full_chunks)}"
    assert len(flush_chunks) == 1, f"Expected 1 flush chunk, got {len(flush_chunks)}"
    assert len(flush_chunks[0][0][0]) == 16000 - 3 * DELIVERY_BYTES  # 1600 bytes


@pytest.mark.asyncio
async def test_flush_remaining_buffer():
    """If TTS returns fewer than delivery_bytes total, the remainder must be flushed."""
    small_chunk = b"\x02" * 1000
    backend = _make_backend(tts_chunks=[small_chunk])

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Hi.", "Hi.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    audio_calls = backend._on_audio_delta.call_args_list
    assert len(audio_calls) == 1, f"Expected exactly 1 audio call (flush), got {len(audio_calls)}"
    assert len(audio_calls[0][0][0]) == 1000


@pytest.mark.asyncio
async def test_multiple_small_tts_chunks_rechunked():
    """Multiple sub-threshold TTS chunks should accumulate and emit 4800-byte pieces."""
    # 3 chunks of 2000 bytes = 6000 total -> 1 full (4800) + 1 flush (1200)
    chunks = [b"\x03" * 2000] * 3
    backend = _make_backend(tts_chunks=chunks)

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Test.", "Test.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    audio_calls = backend._on_audio_delta.call_args_list
    sizes = [len(c[0][0]) for c in audio_calls]
    assert sizes == [DELIVERY_BYTES, 1200], f"Expected [4800, 1200], got {sizes}"


@pytest.mark.asyncio
async def test_bargein_stops_audio_emission():
    """Setting _response_cancelled mid-stream stops further audio and transcript callbacks."""
    # Produce enough data for multiple chunks
    chunks = [b"\x04" * DELIVERY_BYTES] * 5
    backend = _make_backend(tts_chunks=chunks)

    call_count = 0

    async def audio_side_effect(data: bytes):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            backend._response_cancelled = True

    backend._on_audio_delta = AsyncMock(side_effect=audio_side_effect)

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("First sentence.", "First sentence.", False))
    await queue.put(("Second sentence.", "Second sentence.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    # Should have stopped after ~2 audio calls (barge-in)
    assert call_count <= 3, f"Expected at most 3 audio calls before barge-in, got {call_count}"
    # Second sentence's transcript should NOT have been emitted
    transcript_calls = backend._on_transcript_delta.call_args_list
    assert len(transcript_calls) == 1, (
        f"Expected only 1 transcript call (first sentence), got {len(transcript_calls)}"
    )


@pytest.mark.asyncio
async def test_multiple_sentences_each_get_transcript_before_audio():
    """Each sentence gets its own transcript_delta before its audio begins."""
    audio_data = b"\x05" * DELIVERY_BYTES
    backend = _make_backend(tts_chunks=[audio_data])

    call_order: list[str] = []
    backend._on_transcript_delta = AsyncMock(side_effect=lambda *a: call_order.append(f"transcript:{a[0]}"))
    backend._on_audio_delta = AsyncMock(side_effect=lambda *_a: call_order.append("audio"))

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("First.", "First.", False))
    await queue.put(("Second.", "Second.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    # Find indices of transcript and audio events
    first_t = call_order.index("transcript:First.")
    # Find first audio after the first transcript
    first_a = next(i for i, v in enumerate(call_order) if v == "audio" and i > first_t)
    second_t = call_order.index("transcript:Second.")
    assert second_t > first_a, "Second transcript should come after first sentence's audio"


@pytest.mark.asyncio
async def test_tts_error_continues_to_next_sentence():
    """If TTS API raises an exception, the worker should continue with the next sentence."""
    call_count = 0

    def create_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("TTS API error")
        return FakeStreamContext([b"\x06" * DELIVERY_BYTES])

    backend = _make_backend()
    backend._openai.audio.speech.with_streaming_response.create = MagicMock(side_effect=create_side_effect)

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Fails.", "Fails.", False))
    await queue.put(("Works.", "Works.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    # Both sentences should have transcript emitted (transcript is before TTS)
    transcript_calls = backend._on_transcript_delta.call_args_list
    assert len(transcript_calls) == 2, f"Expected 2 transcript calls, got {len(transcript_calls)}"
    assert transcript_calls[0][0][0] == "Fails."
    assert transcript_calls[1][0][0] == "Works."

    # Only second sentence should produce audio (first one errored)
    audio_calls = backend._on_audio_delta.call_args_list
    # Second sentence has silence pad (sentence_count=1) + audio chunk
    # But sentence_count only increments on SUCCESS, so first failure leaves it at 0
    # After error, continue -> second sentence: sentence_count is still 0 (not incremented)
    # Wait — let's check: sentence_count increments AFTER the try/except block
    # First sentence: error in try -> continue (sentence_count stays 0)
    # Second sentence: sentence_count == 0, no silence pad, just audio
    assert len(audio_calls) >= 1, "Expected at least 1 audio call from second sentence"


@pytest.mark.asyncio
async def test_silence_pad_between_sentences():
    """A 200ms silence pad is inserted between sentences (but not before the first)."""
    audio_data = b"\x07" * DELIVERY_BYTES
    backend = _make_backend(tts_chunks=[audio_data])

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("First.", "First.", False))
    await queue.put(("Second.", "Second.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    audio_calls = backend._on_audio_delta.call_args_list
    # First sentence: 1 audio chunk (4800 bytes)
    # Second sentence: silence pad + 1 audio chunk
    # Silence pad = int(24000 * 0.20) * 2 = 9600 bytes
    silence_size = int(24000 * 0.20) * 2
    sizes = [len(c[0][0]) for c in audio_calls]

    # First call should be audio (4800), second should be silence (9600),
    # then audio for second sentence
    assert sizes[0] == DELIVERY_BYTES, f"First audio should be {DELIVERY_BYTES}, got {sizes[0]}"
    assert silence_size in sizes, f"Expected silence pad of {silence_size} bytes in {sizes}"


@pytest.mark.asyncio
async def test_exact_multiple_of_delivery_bytes_no_flush():
    """If total TTS bytes are an exact multiple of 4800, no flush chunk is emitted."""
    # Exactly 2 * 4800 = 9600 bytes
    chunks = [b"\x08" * 9600]
    backend = _make_backend(tts_chunks=chunks)

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(("Test.", "Test.", False))
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    audio_calls = backend._on_audio_delta.call_args_list
    sizes = [len(c[0][0]) for c in audio_calls]
    assert sizes == [DELIVERY_BYTES, DELIVERY_BYTES], f"Expected [4800, 4800], got {sizes}"


@pytest.mark.asyncio
async def test_empty_queue_terminates_immediately():
    """Sending None immediately causes the worker to exit without any callbacks."""
    backend = _make_backend()

    queue: asyncio.Queue[tuple[str, str, bool] | None] = asyncio.Queue()
    await queue.put(None)

    await backend._tts_worker(queue, "item_1")

    backend._on_transcript_delta.assert_not_called()
    backend._on_audio_delta.assert_not_called()
