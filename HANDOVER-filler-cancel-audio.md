# Handover: Filler System, Turn Cancellation & Audio Issues

**Date**: 2026-03-29
**Session scope**: Prompt caching, filler system, turn cancellation, frame pacing, audio debugging
**Status**: Filler system and cancellation are functionally complete but the implementation introduced **crunchy/distorted audio** that remains unresolved.

---

## 1. Goals

### 1.1 Prompt caching optimization (DONE, committed)
Reduce ZeroClaw LLM latency by enabling OpenAI's prompt cache (requires a stable system prompt prefix across requests).

### 1.2 Two-layer filler system (DONE, uncommitted)
Make the avatar speak during long tool executions (calendar, email, web search) instead of going silent for 5-20 seconds. Two layers:
- **Layer 1 (nano classifier)**: Fires immediately (~200ms) on user input to give an instant "On it." style acknowledgment while ZeroClaw processes.
- **Layer 2 (ZeroClaw tool fillers)**: Emits tool-specific filler phrases ("I'm checking your calendar.") as each tool_call event is dispatched.

### 1.3 Turn cancellation (DONE, uncommitted)
Allow users to interrupt long-running tool calls (e.g., a 20-second composio chain). The LLM should be able to decide whether to continue, cancel, or give a status update based on prompt guidance.

### 1.4 Frame pacing fix (DONE, uncommitted)
Fix frame bursts (200-3000fps) caused by audio arriving faster than real-time (cached fillers deliver instantly, TTS streaming outruns playback).

---

## 2. What was implemented and why

### 2.1 Prompt caching (committed in `4868479`, `7ac3992`)

**Files changed**:
- `claw_patches/zeroclaw/src/agent/prompt.rs` — Removed `DateTimeSection` from `SystemPromptBuilder::with_defaults()`. This struct injected a per-second timestamp into the system prompt, invalidating OpenAI's prompt cache on every request. User messages already carry timestamps via `prepare_turn()`, so the LLM still knows the time.
- `src/chat/chat_session.py` — Added warmup: `self.agent.send_text_message("Hi")` with 500ms delay on WebSocket connect. Primes the cache and gives the user a natural greeting.

**Research backing**:
- OpenAI prompt caching docs: requires stable prefix, 1024+ token minimum, 128-token boundary alignment, 5-10 min TTL.
- Tested: 10 sequential calls with warmup showed 93-99% cache hit rate per call, TTFT median ~780ms.
- A two-stage classifier approach (lightweight first call, full tools second call) was prototyped and **discarded** — once the cache is warm, the full tools payload is cached anyway, so there's no latency benefit from splitting.

### 2.2 Two-layer filler system (uncommitted)

**Files changed**:

#### ZeroClaw side (Rust patches):
- `claw_patches/zeroclaw/src/channels/nyxclaw.rs`:
  - Added `tool_call_filler()` function: maps tool names to spoken phrases ("I'm checking your calendar.", "I'm searching the web.", etc.)
  - `dispatch_stream_event()` now emits a `speech_chunk` with `"filler": true` on every `tool_call` event
  - Throttling: nyxclaw (Python) handles all throttle decisions; ZeroClaw always emits

#### nyxclaw side (Python):
- `src/voice/openai_realtime/backend.py`:
  - **Nano classifier** (`_classify_and_filler()`): Fires in parallel with ZeroClaw processing. Uses `gpt-4.1-nano` with a single tool definition (`_FILLER_TOOL`). If the LLM calls the tool, it returns a filler phrase; if it doesn't call it, the input is chitchat. Tool-based approach was chosen over free-text because early free-text attempts returned "CHITCHAT" as literal text that got TTS'd.
  - **Pre-cached filler audio** (`_presynthesise_fillers()`): Pre-synthesizes 6 phrases at startup via OpenAI TTS API (~536KB total PCM). On filler trigger, the cached PCM is delivered in 4800B chunks instead of making a live TTS API call.
  - **Filler throttling**: Shared `_last_filler_at` timestamp, 2s minimum between any fillers, 5s for same-content fillers (`_filler_history`). Both nano and ZeroClaw fillers check this before queuing.
  - **TTS worker changes**: Added cached filler delivery path (loop over pre-synthesized PCM, pad last chunk to 4800B boundary). Added `transcript_emitted` flag — transcript delta now emitted after first audio chunk arrives (not before TTS request).
  - **ZeroClaw filler handling** in `_iter_tokens_ws()`: Intercepts `speech_chunk` messages with `filler: true`, routes them directly to TTS queue with `is_filler=True` instead of yielding as regular content. This prevents filler text from appearing in conversation history.
  - **Active TTS queue reference** (`_active_tts_queue`): Stored on the instance so the nano classifier (running in a parallel task) can push fillers into the current turn's queue.

- `src/backend/zeroclaw/backend.py`:
  - Removed old single-filler-per-turn system (`spoke_filler`, `TOOL_FILLERS`, `random.choice`)
  - Added filler throttle (`_last_filler_at`, 2s gap)
  - Filler `speech_chunk` messages with `filler: true` now route to TTS queue as `is_filler=True`

**Research backing**:
- Hume AI, Inworld AI, Convai, and Replica Studios all use spoken fillers during processing to maintain conversational presence.
- Google's Duplex system uses "um", "uh" fillers to humanize pauses.
- The nano classifier approach (instant intent classification → filler, full LLM → real response) mirrors how production voice assistants handle perceived latency.

### 2.3 Turn cancellation (uncommitted)

**Files changed**:

#### ZeroClaw side (Rust patches):
- `claw_patches/zeroclaw/src/agent/agent.rs`:
  - `turn_with_streaming()` now accepts `cancel_token: Option<tokio_util::sync::CancellationToken>`
  - Three cancellation checkpoints: top of iteration loop, during LLM streaming (via `tokio::select!`), during tool execution (via `tokio::select!`)
  - On cancel during tool execution: pushes a synthetic `ToolExecutionResult` with `"Task interrupted by user"` to preserve history integrity (OpenAI rejects tool_call without matching tool_result)
  - Uses existing `ToolLoopCancelled` error type from `loop_.rs`
  - Added timing instrumentation (`[TIMELINE]` tracing) and tool call/result logging

- `claw_patches/zeroclaw/src/channels/nyxclaw.rs`:
  - `process_avatar_message()` now accepts a `CancellationToken`
  - Main message loop restructured with `tokio::select!` on turn future + `receiver.next()` — reads incoming WebSocket messages concurrently with the running turn
  - New "message" or "cancel" during turn triggers `cancel_token.cancel()` + queues the new message
  - Queued message processed after cancelled turn completes (using block scope to drop `turn_fut` and release borrows)
  - Cancel error handled: sends `{"type": "done", "cancelled": true}` instead of error

#### nyxclaw side (Python):
- `src/voice/openai_realtime/backend.py`:
  - `cancel_response()`: Removed direct WebSocket send (caused `ConcurrencyError` — can't send while `_iter_tokens_ws` is doing `recv()` on the same connection)
  - Cancel message to ZeroClaw now sent in `_on_transcript_ready()` and `_wait_and_start_stream()` AFTER old stream task finishes
  - `_drain_ws_until_done()`: Drains stale WebSocket messages from the cancelled turn (up to 1s timeout) to prevent message pipeline shift
  - Stream task await timeout increased from 0.5s to 2.0s

- `zeroclaw-v0.5.0/playground/AGENTS.md`:
  - Added "Handling interruptions" section providing prompt guidance for status requests, cancel requests, and new requests during a turn

**Design decision**: Reused ZeroClaw's existing `CancellationToken` pattern from `loop_.rs` rather than inventing a new mechanism. The `tokio::select!` on turn + incoming messages in `nyxclaw.rs` is the standard Rust async pattern for concurrent operations.

**Risk mitigated**: History corruption. If cancelled between `tool_call` push and `tool_result` push, OpenAI would reject the incomplete tool sequence. Solution: always push a synthetic "Task interrupted by user" tool_result before returning the cancellation error.

### 2.4 Frame pacing (uncommitted)

**File changed**: `src/chat/chat_session.py`

- `_emit_frames()`: Replaced single audio-clock pacing with a two-gate system:
  - **Gate 1 (audio-clock)**: `target_frames = int(total_audio_received * fps) + 15` — never emit more frames than audio received justifies
  - **Gate 2 (wall-clock)**: `frame_interval = 1.0 / fps` (~33ms at 30fps) — never emit faster than real-time playback rate
- `frame_queue` maxsize increased from 120 to 450 to prevent blocking during burst delivery

**Why both gates**: Audio-clock alone allowed frame bursts when cached filler audio was delivered instantly (total_audio_received jumped, unlocking hundreds of frames). Wall-clock alone caused bursts after silence gaps. Both together keep emission smooth.

### 2.5 Other changes (uncommitted)

- `src/chat/chat_session.py`:
  - `_handle_response_start()`: Added defensive worker cancellation — cancels old inference/emit tasks before starting new turn to prevent residual frame leakage
  - `_execute_interruption_sequence()`: Added `audio_end` message on barge-in — previously, interrupted turns never sent `audio_end`, so the client's audio player never got the signal to reset between turns
  - Diagnostic `audio_delta` logging added (should be removed after debugging)

- `.env`:
  - `OPENAI_TTS_MODEL=gpt-4o-mini-tts`
  - `OPENAI_TTS_SPEED` was set to 1.1, currently reset to 1.0 for debugging
  - `OPENAI_TTS_INSTRUCTIONS` was set to "Speak with energy and warmth...", currently cleared for debugging
  - VAD eagerness set to "low" in session config (semantic_vad)

---

## 3. Issues created by these changes

### 3.1 CRITICAL: Crunchy/distorted audio (UNRESOLVED)

**Symptom**: Audio playback has crackling/distortion artifacts. Initially reported only on multi-segment turns (filler + real response), but later observed on single-segment chitchat too.

**User observations**:
- "Noises started on: 'I'm checking your calendar', then no response at all"
- "No issue with chitchat" (initially, later contradicted)
- "It started while it was saying 'As good as they get in here'" (chitchat, contradicts earlier)
- iOS team reported `headSec` values growing between turns: 2.338 (turn 1) → 8.924 (turn 2), suggesting the audio player's scheduling position isn't resetting between turns

**Server-side diagnostics** (all clean):
- Audio buffer logging shows all deliveries are clean 4800B chunks
- No mixing at segment boundaries (4800B in → buffer accumulates predictably → 24000B chunks extracted)
- No errors, no race conditions visible in logs
- Workers exit normally between turns (`inference_task=gone` at response start)

**Hypotheses tested and results**:

| Hypothesis | Test | Result |
|---|---|---|
| Mixed audio chunks at segment boundary | Added diagnostic logging to `_handle_audio_delta` — logged buf_before, buf_after, chunks_extracted | **DISPROVEN** — all chunks clean 4800B, regular accumulation |
| TTS speed=1.1 + instructions causing artifacts | Reset speed to 1.0, cleared instructions, rebuilt | **INCONCLUSIVE** — crunchy audio persisted after reset |
| Missing `audio_end` on interruption causing headSec growth | Added `audio_end` to `_execute_interruption_sequence` | **INCONCLUSIVE** — crunchy audio persisted |
| Residual frames leaking across turns | Added defensive worker cancel in `_handle_response_start` | **INCONCLUSIVE** — workers were already gone per logs |
| Frame bursts from fast audio delivery | Two-gate pacing (audio-clock + wall-clock) | **PARTIALLY FIXED** — eliminated server-side frame bursts but audio quality issue remains |

**What has NOT been tested**:
- Completely reverting ALL uncommitted changes and confirming clean audio on the previous commit — this would definitively prove whether these changes caused it or it's pre-existing
- Client-side audio player debugging — does the iOS AVAudioEngine player node reset properly on `audio_end`?
- Whether the `headSec` accumulation is caused by the server or client — the server does send `audio_end` on normal turns, but does the client actually reset its player node?
- Removing ONLY the filler system (keep everything else) to isolate whether fillers specifically cause the issue
- Testing with a non-TTS audio source (e.g., sine wave) to determine if the issue is TTS quality or pipeline processing
- Testing the OpenAI TTS API directly (outside nyxclaw) to check if `gpt-4o-mini-tts` produces clean audio

**Key unknown**: The crunchy audio initially appeared only on multi-segment turns, then spread to chitchat. This could mean:
1. It was always there but only noticed with fillers (fillers drew attention to audio quality)
2. Something in the filler changes affects all turns (e.g., pre-synthesize task at startup, changed TTS parameters, queue architecture changes)
3. It's a client-side issue that worsened over the session (headSec accumulation)

### 3.2 Double user transcripts (UNRESOLVED)

**Symptom**: User reports seeing double user transcripts in the mobile app.

**Analysis**: Server-side, only one `transcript_done` message is sent per voice input (via `_handle_user_transcript`). However, the warmup "Hi" message goes through `send_text_message` which also calls `_on_user_transcript`, so the client receives a `transcript_done` for "Hi" at connect time. This might appear as an unwanted user message. Alternatively, this could be a client-side rendering issue.

**Not investigated in depth**.

### 3.3 Double ZeroClaw tool fillers within same turn (KNOWN, NOT FIXED)

When composio retries (e.g., calendar tool fails, composio calls it again), each `tool_call` emits a filler. The 2s throttle on the nyxclaw side suppresses the second one, but if retries are >2s apart, the user hears the same filler twice. A filler progression system (first specific, then generic "still working on it") was discussed but not implemented.

### 3.4 Transcript arrives before audio

All `transcript_delta SEND` entries in logs show `total_audio_received=0.000s`. The transcript delta is emitted from the TTS worker on first audio chunk, but the `_on_transcript_delta` callback fires before the `_on_audio_delta` callback delivers the first chunk to `_handle_audio_delta`. This means the client receives transcript slightly before audio. May cause display/timing issues on the client side.

---

## 4. Architecture of the filler system

```
User speaks
    │
    ▼
OpenAI Realtime (STT) ──► transcript
    │
    ├──► _classify_and_filler() ──► gpt-4.1-nano (tool-based)
    │         │                         │
    │         │   (if action intent)    │
    │         ▼                         │
    │    _active_tts_queue.put(         │
    │      filler, filler, True)        │
    │         │                         │
    │         ▼                         ▼ (if chitchat: no-op)
    │    TTS worker picks up
    │    from _cached_filler_audio
    │    delivers 4800B PCM chunks
    │         │
    │         ▼
    │    _on_audio_delta ──► chat_session audio pipeline
    │
    ├──► ZeroClaw WebSocket ──► LLM + tools
    │         │
    │    (on each tool_call event)
    │         │
    │         ▼
    │    speech_chunk {filler: true, content: "I'm checking your calendar."}
    │         │
    │         ▼
    │    _iter_tokens_ws intercepts filler
    │    checks throttle (2s gap, 5s same-content)
    │         │
    │         ▼
    │    tts_queue.put(content, content, True)
    │         │
    │         ▼
    │    TTS worker: check _cached_filler_audio
    │    or live TTS if not cached
    │         │
    │         ▼
    │    _on_audio_delta ──► chat_session audio pipeline
    │
    ├──► ZeroClaw response (speech_chunk, done)
    │         │
    │         ▼
    │    Normal TTS path (sentence extraction, live TTS)
    │         │
    │         ▼
    │    _on_audio_delta ──► chat_session audio pipeline
    │
    ▼
chat_session._handle_audio_delta
    │
    ▼
audio_buffer → 500ms chunks → wav2arkit inference → frame_queue
    │
    ▼
_emit_frames (two-gate: audio-clock + wall-clock)
    │
    ▼
sync_frame WebSocket messages to client
```

---

## 5. Files summary

| File | Status | Changes |
|---|---|---|
| `claw_patches/zeroclaw/src/agent/agent.rs` | Uncommitted | CancellationToken in `turn_with_streaming`, tool call/result logging, `turn_with_avatar_streaming` (two-stage, unused), timing instrumentation |
| `claw_patches/zeroclaw/src/agent/prompt.rs` | Committed (`4868479`) | DateTimeSection removed for prompt caching |
| `claw_patches/zeroclaw/src/channels/nyxclaw.rs` | Uncommitted | Concurrent WS read with `tokio::select!`, cancel handling, `tool_call_filler()`, filler emission on tool_call events |
| `src/voice/openai_realtime/backend.py` | Uncommitted | Nano classifier, pre-cached filler audio, filler throttle, cancel message sequencing, WS drain, TTS worker cached path, transcript timing fix, VAD eagerness "low" |
| `src/backend/zeroclaw/backend.py` | Uncommitted | Removed old filler system, added filler throttle, filler routing through TTS queue |
| `src/chat/chat_session.py` | Uncommitted | Two-gate frame pacing, frame_queue maxsize 450, defensive worker cancel, audio_end on interrupt, diagnostic logging |
| `zeroclaw-v0.5.0/playground/AGENTS.md` | On disk (not patched) | Added "Handling interruptions" prompt section |
| `.env` | Local only | TTS model, speed (currently 1.0), instructions (currently empty), VAD eagerness |

---

## 6. Recommended next steps

1. **Isolate the crunchy audio**: `git stash` all uncommitted changes, rebuild, and test. If audio is clean, bisect the changes. If still crunchy, it's pre-existing or client-side.

2. **Test client-side audio reset**: Verify the iOS app properly resets its AVAudioEngine player node on `audio_end` and `audio_start`. The `headSec` accumulation (2.338 → 8.924) strongly suggests the player isn't resetting between turns.

3. **Test TTS API in isolation**: Call `gpt-4o-mini-tts` directly and play the PCM output to confirm the TTS itself produces clean audio. This rules out the pipeline.

4. **Remove diagnostic logging**: The `audio_delta` debug logging in `_handle_audio_delta` is verbose and should be removed once debugging is complete.

5. **Re-enable .env settings**: `AUTH_ENABLED`, `DEBUG`, TTS speed/instructions need to be restored once audio is fixed.

6. **Commit working pieces separately**: Prompt caching is already committed and working. The cancellation and filler systems are functionally correct but entangled with the audio issue. Consider committing them to a feature branch for isolation.
