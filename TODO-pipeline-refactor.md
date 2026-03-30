# Pipeline Fix: Crunchy Audio, Transcript Sync & Filler Cleanup

**Date**: 2026-03-30
**Goal**: Fix crunchy audio, transcript desync, and consolidate filler system.

---

## Validated assumptions (test_pipeline_assumptions.py — 37/37 pass)

These are confirmed facts, not guesses:

- **wav2arkit is stateless** — processing filler audio before response audio does NOT affect response blendshapes. No hidden state leaks between calls. `reset_context()` is a no-op. (Validated in Docker with real ONNX model)
- **The audio buffer handles mixed content correctly** — silence pads + speech crossing chunk boundaries produce correct byte-exact frame slices and correct blendshapes.
- **The two-gate emission system is sound** — Gate 1 (audio-clock) + Gate 2 (wall-clock 33ms) correctly pace output at real-time. Burst audio fills the queue but emission stays at 30fps.
- **Chunk math is exact** — 500ms = 24000 bytes = 15 frames × 1600 bytes. No remainder, no alignment issues, 16kHz↔24kHz frame counts match.
- **No AudioSegment abstraction needed in ChatSession** — the frame factory pipeline works correctly as a flat byte stream.

---

## Phase 0: Isolate the crunchy audio

**Before any code changes.** Determine root cause.

- [ ] **0.1** Commit current state to `feat/filler-system` branch (safe rollback point). Then `git stash`, rebuild Docker, test audio quality on the previous commit (`7ac3992`).
- [ ] **0.2** If clean on stash → bisect: reapply changes one file at a time. Priority order:
  1. Only `chat_session.py` (frame pacing)
  2. Only `backend.py` (filler routing + TTS worker)
  3. Only `openai_realtime/backend.py` (nano classifier + cached fillers)
- [ ] **0.3** Test client-side `audio_end` handling: verify the iOS app resets `AVAudioPlayerNode` scheduling position (`headSec` → 0) on every `audio_end`. The headSec accumulation (2.338 → 8.924 between turns) is the strongest signal of a client-side bug.
- [ ] **0.4** Test TTS in isolation: call `gpt-4o-mini-tts` directly, save PCM to file, play it back. Rule out TTS source quality.

### Decision gate after Phase 0

| Finding | Action |
|---|---|
| Stash fixes crunch | Bisect to find the offending change (0.2). Likely a targeted fix, not a refactor. |
| Stash doesn't fix crunch + headSec accumulates | Client-side bug. Fix in iOS app. Server changes are optional improvements. |
| Stash doesn't fix crunch + headSec is fine | TTS source quality or a pre-existing pipeline issue. Proceed to Phase 1. |

---

## Phase 1: Transcript timing fix

**The confirmed real bug.** Transcript is emitted from the TTS worker the moment TTS produces audio, but that audio then goes through a ~530ms+ pipeline (500ms buffer fill + 30ms inference + queue wait) before reaching the client as a sync_frame. The client sees text before hearing it.

### Byte-level trace (design validation)

```
TTS worker produces sentence "Hello world." (ZeroClaw backend):
  1. silence_pad delivered:     on_audio_delta(9600 bytes)  → total_audio_received = 1.700s
  2. transcript_delta emitted:  "Hello world." at total_audio_received = 1.700s
  3. TTS audio delivered:       on_audio_delta(4800 bytes × N chunks)

Meanwhile in ChatSession:
  - audio_buffer accumulates bytes from step 1 + 3
  - 500ms chunks extracted → audio_chunk_queue
  - _inference_worker processes → 15 frames per chunk → frame_queue
  - _emit_frames sends frames at 30fps

  Frame delivery timeline (30fps = 33ms per frame):
    frame 0  at total_audio_delivered = 0.000s  (turn start)
    ...
    frame 51 at total_audio_delivered = 1.700s  ← THIS is when transcript should arrive
    ...

Current behavior: transcript arrives at step 2 (~0ms into pipeline).
Desired behavior: transcript arrives when frame 51 emits (~1700ms into pipeline).
Gap: ~1700ms for this example. Always ≥500ms (one chunk buffer fill).
```

### Tests to write BEFORE implementation

- [ ] **1.0a** `test_transcript_audio_position_recording`: Verify that `total_audio_received` at the moment `on_transcript_delta` fires reflects the correct audio position (after silence pad, before sentence audio). Instrument both backends.
- [ ] **1.0b** `test_transcript_pending_queue_ordering`: Unit test that pending transcripts are consumed in order when delivery clock advances.
- [ ] **1.0c** `test_transcript_cleared_on_interruption`: Verify pending transcripts are discarded on barge-in.
- [ ] **1.0d** `test_filler_audio_advances_clock_without_transcript`: Verify that 1.5s of filler audio advances the delivery clock by 1.5s without emitting any transcript.

### Implementation

- [ ] **1.1** Add `_pending_transcripts: list[tuple[float, str, str, str | None, str | None]]` to ChatSession (audio_position, text, role, item_id, previous_item_id).
- [ ] **1.2** In `_handle_transcript_delta`, instead of sending immediately, push to `_pending_transcripts` with `transcript_audio_position = self.total_audio_received`.
- [ ] **1.3** In `_emit_frames`, track `total_audio_delivered` (increment by `1.0 / fps` per frame). After each frame emission, check `_pending_transcripts`: pop and send all entries where `audio_position <= total_audio_delivered`.
- [ ] **1.4** Clear `_pending_transcripts` in `_handle_response_start` and `_execute_interruption_sequence`.
- [ ] **1.5** On `_handle_response_end`, after workers drain, flush any remaining `_pending_transcripts` (safety net for the last sentence).
- [ ] **1.6** Remove `virtual_cursor_text_ms`, `chars_per_second`, and `self.agent.transcript_speed` references. Remove `transcript_speed` property from BaseAgent and both backend implementations.
- [ ] **1.7** Decide on word-splitting: for now, keep it. Push the full sentence to pending queue, split into words when emitting from `_emit_frames`. `startOffset`/`endOffset` calculated from delivery clock position.

### Backend-specific note

ZeroClaw backend emits transcript BEFORE sentence audio → `total_audio_received` is at the correct sentence start.
OpenAI Realtime backend emits transcript on FIRST audio chunk → `total_audio_received` includes ~100ms of the sentence. This is a ~100ms discrepancy. Acceptable for now; document as known imprecision.

---

## Phase 2: Filler throttle consolidation

### Tests to write BEFORE implementation

- [ ] **2.0a** `test_filler_throttle_2s_gap`: Two fillers 1.5s apart → second is suppressed. Two fillers 2.5s apart → both speak.
- [ ] **2.0b** `test_filler_throttle_same_content_5s`: Same filler 4s apart → suppressed. Same filler 6s apart → speaks.
- [ ] **2.0c** `test_filler_skipped_when_real_content_queued`: Filler arrives after real content is already in TTS queue → filler skipped.

### Implementation

- [ ] **2.1** Add `_should_speak_filler(self, content: str) -> bool` method to both backends. Contains all throttle logic: `_last_filler_at` (2s gap), `_filler_history` (5s same-content). Single source of truth.
- [ ] **2.2** Update `_classify_and_filler()` in OpenAI Realtime backend to call `_should_speak_filler()` instead of inline throttle checks.
- [ ] **2.3** Update `_iter_tokens_ws()` in OpenAI Realtime backend: replace inline filler throttle with `_should_speak_filler()`.
- [ ] **2.4** Update `_stream_and_speak()` in ZeroClaw backend: replace inline filler throttle with `_should_speak_filler()`.
- [ ] **2.5** Keep the TTS worker's `if is_filler and not queue.empty(): continue` logic unchanged — it's the most important optimization.

---

## Phase 3: Stabilization loop replacement

### Byte-level trace (design validation)

```
TTS worker finishes last sentence:
  1. Last on_audio_delta(4800 bytes) called and AWAITED  → _handle_audio_delta completes
  2. tts_queue.put(None) → TTS worker exits
  3. await tts_task completes
  4. Backend calls on_response_end(full_response, item_id)
  5. ChatSession._handle_response_end runs

At step 5:
  - total_audio_received is FINAL (no more on_audio_delta calls coming)
  - audio_buffer may have 0-23999 bytes of unflushed partial chunk
  - audio_chunk_queue may have 0-15 chunks waiting for inference
  - frame_queue may have 0-N frames waiting for emission

The stabilization loop (lines 420-433) polls total_audio_received.
Since it stopped growing at step 1, stable_count increments immediately.
15 ticks × 50ms = 750ms of pure waste.
```

### Tests to write BEFORE implementation

- [ ] **3.0a** `test_response_end_audio_already_delivered`: Verify that when `_handle_response_end` is called, `total_audio_received` has already reached its final value (no more growth).
- [ ] **3.0b** `test_partial_buffer_flush_produces_correct_frames`: Verify that flushing a partial buffer (e.g., 14400 bytes with silence padding to 14400 bytes) produces the correct number of frames with correct audio slices.
- [ ] **3.0c** `test_workers_drain_after_speech_ended`: After `speech_ended = True`, inference_worker and emit_worker exit once their queues are empty.

### Implementation

- [ ] **3.1** Remove the stabilization polling loop (lines 420-433 of `_handle_response_end`).
- [ ] **3.2** Add `await asyncio.sleep(0.05)` before buffer flush — one event-loop tick to ensure any in-flight `_handle_audio_delta` coroutines complete.
- [ ] **3.3** Keep the existing partial-chunk flush code (lines 438-447) unchanged.
- [ ] **3.4** Keep the existing worker wait logic (lines 449-471) unchanged — it correctly waits for inference and emission to complete.

---

## Phase 4: Cleanup

- [ ] **4.1** Remove diagnostic `audio_delta` logging from `_handle_audio_delta`.
- [ ] **4.2** Remove the `TOOL_FILLERS` list from `base_agent.py` — unused after filler system rework.
- [ ] **4.3** Restore `.env` settings: `OPENAI_TTS_SPEED`, `OPENAI_TTS_INSTRUCTIONS`, `AUTH_ENABLED`.
- [ ] **4.4** Run `ruff check src/` and `ruff format src/`.
- [ ] **4.5** Run `pytest tests/`.

---

## Phase 5: Verification

- [ ] **5.1** Chitchat (no tools): clean audio, synced transcript, smooth blendshapes.
- [ ] **5.2** Tool call (calendar): filler plays, then real response, transcript synced.
- [ ] **5.3** Barge-in during filler: clean interruption, no residual audio/frames.
- [ ] **5.4** Barge-in during real response: truncated transcript, clean cutoff.
- [ ] **5.5** Rapid back-to-back messages: no headSec accumulation, no state leakage.
- [ ] **5.6** Long conversation (10+ turns): no progressive degradation.
- [ ] **5.7** Transcript sync measurement: enable debug logging that timestamps both `transcript_delta SEND` and the `sync_frame` containing the corresponding audio. Compare — they should be within ±33ms.

---

## Review cleanup actions (from 5-agent code review, 2026-03-30)

### Now (before commit)

- [x] Downgrade `TRANSCRIPT_AUDIT` text field to `logger.debug` and remove the `text=` content to avoid reflected PII in production logs. *(Done)*
- [ ] Remove dead `turn_with_avatar_streaming` from `claw_patches/zeroclaw/src/agent/agent.rs` — ~280 lines of copy-pasted `turn_with_streaming` without cancellation support. Never called. *(Attempted, reverted due to edit corruption — needs careful handling as a standalone task.)*

### Soon (next session)

- [ ] Move `_should_speak_filler()` from both backends into `BaseAgent` — identical logic duplicated in `ZeroClawBackend` and `OpenAIRealtimeBackend`. Single source of truth.
- [ ] Extract cancel+drain helper in `OpenAIRealtimeBackend` — the `send(cancel) → drain_ws_until_done()` sequence is copy-pasted in `_on_transcript_ready`, `_wait_and_start_stream`, and `_stream_and_speak` finally block.
- [ ] Gate nano classifier (`_classify_and_filler`) against dead TTS queue — if the turn ends before the classifier returns, `_active_tts_queue` may point to a consumed queue. Add a turn-ID check or clear the queue reference on turn end.

### Later

- [ ] Serialize `_wait_and_start_stream` — two rapid `send_text_message` calls can orphan a stream task. Use a lock or inline the wait (matching the voice input path in `_on_transcript_ready`).
- [ ] Add integration tests for `_classify_and_filler` (nano classifier) and cached filler audio delivery path in TTS worker — these are the largest untested new features.
- [ ] Consider reverting `frame_queue` maxsize from 450 back to 120 — the two-gate pacing fix should make the larger buffer unnecessary.
- [ ] Move Rust `[TIMELINE]` tool-call arg logging from `tracing::info!` to `tracing::debug!` — full tool args can contain PII (email bodies, calendar details).
- [ ] Switch `_emit_frames` from `time.time()` to `time.perf_counter()` for consistency — `time.time()` can jump on NTP adjustments, causing frame jitter.

---

## What was removed from the original plan (and why)

| Removed | Why |
|---|---|
| AudioSegment abstraction in ChatSession | Tests prove audio_buffer handles mixed content correctly. Model is stateless. |
| `reset_context()` at segment boundaries | Confirmed no-op — model has no hidden state. (Docker ONNX test) |
| Remove/replace Gate 1 emission pacing | Tests prove Gate 1 + Gate 2 correctly pace at real-time. |
| Remove silence padding between sentences | Silence pads produce correct blendshapes (closed mouth). |
| Transcript tagging on wav2arkit frames | Violates separation of concerns. Transcript queue is simpler. |
| `on_filler_request` / `on_segment_start/end` callbacks | Over-engineering. Pipeline works as flat byte stream. |
| Protocol changes (`segmentId`, `audio_segment_start/end`) | Premature. Fix server-side first. |

---

## Dependency order

```
Phase 0 (isolate crunch)  → FIRST, no code changes, determines path forward
Phase 1 (transcript sync) → independent, can start after Phase 0 decision gate
Phase 2 (filler throttle) → independent, can run in parallel with Phase 1
Phase 3 (stabilization)   → independent, can run in parallel with Phase 1
Phase 4 (cleanup)         → after Phases 1-3
Phase 5 (verification)    → after Phase 4
```

Phases 1, 2, 3 are independent and can be worked in parallel.
