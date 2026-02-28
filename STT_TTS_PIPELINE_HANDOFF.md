# STT / TTS / VAD / Barge-In Pipeline — Code Review & Handoff

> **Date:** 2026-02-28
> **Purpose:** Full pipeline review for the next developer. Covers architecture, current state, changes made, open problems, and recommended fixes.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Map](#2-file-map)
3. [Component Deep-Dive](#3-component-deep-dive)
   - [3.1 STT Service (VAD + Whisper)](#31-stt-service-vad--whisper)
   - [3.2 TTS Service (Piper)](#32-tts-service-piper)
   - [3.3 Agent Audio Pipeline](#33-agent-audio-pipeline)
   - [3.4 Chat Session (Playback & Barge-In)](#34-chat-session-playback--barge-in)
4. [Changes Made (3-Day Summary)](#4-changes-made-3-day-summary)
5. [Open Problems](#5-open-problems)
   - [P1: Client Sends Silence During Playback](#p1-client-sends-silence-during-playback)
   - [P2: Whisper Transcription Accuracy](#p2-whisper-transcription-accuracy)
   - [P3: Whisper beam_size Mismatch](#p3-whisper-beam_size-mismatch)
   - [P4: Duplicate Code in stop()](#p4-duplicate-code-in-stop)
6. [Recommended Next Steps](#6-recommended-next-steps)
7. [Test Coverage](#7-test-coverage)
8. [Settings & Configuration](#8-settings--configuration)
9. [Docker & Deployment Notes](#9-docker--deployment-notes)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser/App)                     │
│   mic PCM16 24kHz ──────────┐        ┌──── sync_frame (audio +  │
│                             │        │     blendshape weights)   │
│                             ▼        │                          │
└─────────────────────── WebSocket ────┘──────────────────────────┘
                             │                    ▲
                             ▼                    │
┌────────────────────────────────────────────────────────────────┐
│                     ChatSession (chat_session.py)               │
│                                                                 │
│  process_message()          _handle_audio_delta()               │
│    └─ audio bytes ──►       └─ audio_buffer ──► audio_chunk_queue│
│       agent.append_audio()       ──► _inference_worker()        │
│                                       └─ wav2arkit ONNX         │
│                                           └─► frame_queue       │
│                                                └─► _emit_frames()│
│                                                    └─► WS send  │
│                                                                 │
│  _handle_response_start()   _handle_response_end()              │
│  _handle_interrupted()      _on_agent_cancel_sync()             │
└────────────────────────────────────────────────────────────────┘
                             │
                    agent.append_audio()
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              Agent (zeroclaw/sample_agent.py)                   │
│                                                                 │
│  _audio_input_worker() loop:                                    │
│    ├─ if is_responding:                                         │
│    │    send_audio → VAD → barge-in check (4 frames = 128ms)   │
│    │    if barge-in → cancel_response()                         │
│    └─ else:                                                     │
│         send_audio → VAD → check_pause()                        │
│         if pause → flush_silence() → transcribe → LLM request  │
│                                                                 │
│  _stream_and_speak():                                           │
│    LLM stream → sentence buffer → TTS queue → _tts_worker()    │
│    _tts_worker() → synthesize() → on_audio_delta callback      │
│                                                                 │
│  cancel_response():                                             │
│    set flags → on_cancel_sync() → stt.reset_turn(vad=True)     │
└────────────────────────────────────────────────────────────────┘
                    │                           │
         ┌─────────┘                           └──────────┐
         ▼                                                ▼
┌──────────────────┐                          ┌──────────────────┐
│   STT Service    │                          │   TTS Service    │
│  (stt_service.py)│                          │  (tts_service.py)│
│                  │                          │                  │
│  Silero VAD v5   │                          │  Piper VITS ONNX │
│  (ONNX, no torch)│                         │  22050Hz → 24kHz │
│  faster-whisper  │                          │  resample_poly   │
│  (CTranslate2    │                          └──────────────────┘
│   int8, CPU)     │
└──────────────────┘
```

**Data flow summary:**
1. Client sends PCM16 @ 24kHz via WebSocket
2. `ChatSession.process_message()` routes audio to `agent.append_audio()`
3. Agent's `_audio_input_worker()` feeds audio to STT/VAD
4. VAD detects speech start/end; on pause → Whisper transcribes → LLM request
5. LLM streams tokens → sentence buffer → TTS synthesizes per-sentence
6. TTS PCM16 @ 24kHz → `on_audio_delta` → ChatSession → wav2arkit → blendshapes + audio → client
7. During playback (`is_responding=True`), audio still feeds VAD for barge-in detection

---

## 2. File Map

| File | Role | Lines |
|------|------|-------|
| `src/services/stt_service.py` | STT + Silero VAD (ONNX) | ~503 |
| `src/services/tts_service.py` | Piper TTS (VITS ONNX) | ~221 |
| `src/agents/zeroclaw/sample_agent.py` | ZeroClaw agent (WebSocket to LLM) | ~990 |
| `src/agents/openclaw/sample_agent.py` | OpenClaw agent (HTTP SSE to LLM) | ~972 |
| `src/agents/zeroclaw/zeroclaw_settings.py` | ZeroClaw pydantic settings | ~74 |
| `src/agents/openclaw/openclaw_settings.py` | OpenClaw pydantic settings | ~131 |
| `src/agents/base_agent.py` | Abstract agent interface | ~135 |
| `src/chat/chat_session.py` | WebSocket session, audio playback, wav2arkit | ~782 |
| `src/core/settings.py` | Global settings | - |
| `src/services/wav2arkit_service.py` | Lip-sync blendshape inference | - |
| `tests/test_audio_pipeline.py` | 20 unit/integration tests | - |
| `tests/test_bargein_playback.py` | 6 barge-in integration tests | - |

---

## 3. Component Deep-Dive

### 3.1 STT Service (VAD + Whisper)

**File:** `src/services/stt_service.py`

#### Silero VAD (`_SileroVADIterator`, lines 31-102)

- Pure ONNX implementation — no PyTorch, no `silero-vad` pip package
- Model: `pretrained_models/silero_vad.onnx` (Silero v5)
- **Frame size:** 512 samples @ 16kHz (32ms) + 64-sample context window = 576 input samples
- Input audio arrives at 24kHz → resampled to 16kHz via `resample_poly(up=2, down=3)`
- Returns `{"start": N}`, `{"end": N}`, or `None`
- Internal state: `_state` tensor `[2, 1, 128]` (RNN hidden state), `_context` (last 64 samples)
- `reset_states()` zeros everything — use after barge-in to prevent stale triggers

#### VAD State Tracking (`_detect_speech_confidence`, lines 388-408)

```
Event returned    | _vad_in_speech | Confidence returned
------------------|----------------|--------------------
{"start": N}      | → True         | 1.0
{"end": N}        | → False        | 0.0
None (in speech)  | unchanged      | 0.9
None (not speech) | unchanged      | 0.05
```

**Key insight:** VADIterator returns `None` during ongoing speech (not just silence). The `_vad_in_speech` flag (line 177) distinguishes "inside speech segment, confidence 0.9" from "silence, confidence 0.05". Without this, the code falls through to the RMS energy gate (line 407) which is noisy.

#### Speech Detection Flow (`send_audio`, lines 229-258)

1. Accumulate audio in `_pending` buffer
2. Extract 768-sample frames (32ms @ 24kHz)
3. Resample each frame to 16kHz → run VAD
4. If `vad_conf >= _vad_start_threshold (0.60)` → `_has_speech = True`, accumulate in `_speech_audio_24k`
5. If already in speech and `vad_conf <= _vad_end_threshold (0.35)` → count silence frames
6. When silence exceeds `_vad_min_silence_ms (280ms)` → `_pause_ready = True`

#### Whisper Transcription (`_transcribe_current_segment`, lines 438-502)

- **Model:** `base.en` (74M params, CTranslate2 int8)
- **Current settings (PROBLEM):**
  ```python
  beam_size=5, best_of=5  # Lines 472-473 — expensive, increases hallucination
  ```
- **Filters applied:**
  - Minimum segment duration: 300ms (7200 samples @ 24kHz) — line 444
  - Hallucination blocklist: ~20 known Whisper noise phrases — lines 411-432
  - Short text filter: < 5 chars AND duration < 0.8s — line 490
- **Thread safety:** `_shared_transcribe_lock` guards `transcribe()` — not thread-safe across concurrent calls

**CRITICAL ISSUE:** `beam_size=5, best_of=5` (line 472-473) is the default in the code. This is computationally expensive on CPU AND increases hallucination risk on short audio segments. The MEMORY.md for the project says `beam_size=1` was the intended setting. This was never changed during the 3-day work session and is likely a significant contributor to poor STT quality.

#### Shared State (class-level)

```python
_shared_lock = threading.Lock()           # Guards all shared state
_shared_transcribe_lock = threading.Lock() # Guards transcribe() calls
_shared_silero_iterator = None             # Single VAD instance shared across sessions
_shared_silero_enabled = False
_shared_whisper_models = {}                # {model_name: WhisperModel}
```

**Warning:** The Silero VAD iterator is shared across ALL sessions (line 315-316). This means two concurrent sessions will interfere with each other's VAD state. For single-session use (current deployment) this is fine, but it's a latent bug for multi-session.

---

### 3.2 TTS Service (Piper)

**File:** `src/services/tts_service.py`

- **Engine:** piper-tts 1.4.1 (`PiperVoice.load()` + `synthesize()`)
- **Model:** `pretrained_models/piper/en_US-hfc_female-medium.onnx` + `.onnx.json`
- **Native sample rate:** 22050 Hz
- **Output sample rate:** 24000 Hz (resampled via `resample_poly`)
- **Resampling ratio:** 24000/22050 = 160/147 (GCD = 150)

#### Synthesis Flow (`synthesize`, lines 159-208)

1. Text → `_worker()` thread
2. Worker calls `self._voice.synthesize(text, syn_config=self._syn_config)` — collects ALL chunks
3. Concatenates all float arrays, resamples once via `resample_poly`, converts to PCM16
4. Puts single PCM16 bytes block into async queue
5. Caller yields blocks from queue

**Key detail:** Piper typically yields 1 chunk per sentence, so the concatenate-then-resample approach adds negligible latency. But the output is one large PCM16 block per sentence (e.g., 2-3 seconds of audio), not sub-sentence streaming.

#### VITS Synthesis Knobs

```python
noise_scale=0.75    # Audio variation (0=flat, 1=expressive)
noise_w_scale=0.8   # Phoneme duration variation (0=robotic, 1=natural)
length_scale=0.95   # Speech speed (<1=faster, >1=slower)
```

These were tuned during the 3-day session. Previous values were higher noise scales which sounded worse.

#### Voice Instance Caching

```python
_shared_voices: dict[str, Any] = {}  # {model_path: PiperVoice} — class-level, thread-safe via _shared_lock
```

---

### 3.3 Agent Audio Pipeline

**Files:** `src/agents/zeroclaw/sample_agent.py`, `src/agents/openclaw/sample_agent.py`

Both agents follow the same architecture. ZeroClaw uses WebSocket to LLM, OpenClaw uses HTTP SSE. The STT/TTS pipeline is identical.

#### Audio Input Worker (`_audio_input_worker`)

This is the main audio processing loop. It runs as a background asyncio task.

**Normal path (not responding):**
```
audio_bytes → stt.send_audio() → VAD processes → check_pause()
  → if pause detected → _handle_user_pause()
      → stt.flush_silence() → Whisper transcribes → LLM request
```

**Barge-in path (is_responding=True):**
```
audio_bytes → stt.send_audio() → VAD processes
  → if _has_speech: increment _bargein_speech_frames
  → if 4 consecutive frames (128ms): cancel_response()
  → else (no speech): reset counter to 0
```

**Idle detection (no audio arriving):**
```
TimeoutError after 150ms → idle_ticks++
  → if has_speech AND transcript buffer AND idle_ticks >= 3 (450ms):
      → _handle_user_pause() (treat silence as end-of-turn)
```

#### cancel_response() (ZeroClaw, lines 966-989)

```python
def cancel_response(self):
    if not self._state.is_responding:
        return
    self._response_cancelled = True
    self._tts_cancelled.set()
    self._cancel_event.set()             # ZeroClaw-specific: signal WS reader
    self._state.is_responding = False
    self._state.audio_done = True
    # ZeroClaw: does NOT cancel _active_stream_task (graceful WS drain)
    # OpenClaw: DOES cancel _active_stream_task (HTTP SSE, no drain needed)
    if self._on_cancel_sync:             # NEW: notify ChatSession synchronously
        self._on_cancel_sync()
    if self._stt:
        self._stt.reset_turn(reset_vad_state=True)  # Clear stale VAD
    self._user_transcript_buffer = ""
    self._bargein_speech_frames = 0
```

**Important difference:** ZeroClaw uses a persistent WebSocket to the LLM server. On barge-in, it sends `{"type": "cancel"}` and drains remaining messages gracefully. OpenClaw just cancels the HTTP stream task.

#### _stream_and_speak() Finally Block (lines 863-901)

This is the critical section for barge-in timing:

```python
finally:
    # ... cleanup TTS task ...

    # Keep is_responding=True during audio playback
    if not self._response_cancelled and self._on_response_end:
        await self._on_response_end(full_response, item_id)  # Waits for audio drain!
        # Barge-in may fire during the await above
        if self._response_cancelled and self._on_interrupted:
            await self._on_interrupted()
    elif self._response_cancelled and self._on_interrupted:
        await self._on_interrupted()

    self._state.is_responding = False  # Only now set to False
    self._state.audio_done = True
```

**Why this matters:** Before the fix, `is_responding = False` was set BEFORE `on_response_end`. Since `on_response_end` waits for audio playback to complete (which can take 10+ seconds), the barge-in code path was never active during playback.

---

### 3.4 Chat Session (Playback & Barge-In)

**File:** `src/chat/chat_session.py`

#### _on_agent_cancel_sync() (line 116-124)

```python
def _on_agent_cancel_sync(self) -> None:
    self._cancel_playback = True
    self.audio_buffer.clear()
```

This is called synchronously from `agent.cancel_response()`. It sets `_cancel_playback` so the `_handle_response_end()` method aborts immediately.

#### _handle_response_end() (lines 328-416)

This method orchestrates audio playback completion. It:
1. Waits for audio to stabilize (no new audio for 15 × 50ms = 750ms)
2. Flushes remaining audio buffer
3. Waits for audio_chunk_queue to drain
4. Waits for inference_task (wav2arkit) to complete
5. Waits for frame_emit_task to complete
6. Sends `audio_end` + `transcript_done` + `avatar_state: Listening`

**Cancellation checks:** `self.is_interrupted or self._cancel_playback` is checked:
- At entry (line 330)
- Inside stabilization loop (line 341)
- After stabilization (line 349)
- During queue drain (line 363)

Without `_cancel_playback`, barge-in during audio playback would have to wait up to 750ms for stabilization + full queue drain before the session could respond.

#### _handle_interrupted() (lines 553-592)

Full interruption handler:
1. Send `interrupt` + `avatar_state: Listening` to client immediately
2. Acquire `_interruption_lock` with 1s timeout
3. Cancel agent response
4. Clear buffers and queues
5. Cancel inference and frame tasks
6. Calculate truncated text and send `transcript_done` with `interrupted: True`
7. Reset `is_interrupted = False`

---

## 4. Changes Made (3-Day Summary)

### Day 1: STT/VAD Foundation
- Implemented `_SileroVADIterator` (pure ONNX, no torch)
- Fixed Silero v5 context window (64 samples prepended to each frame)
- Added `_vad_in_speech` tracking flag
- Sanity check: feeds speech-like harmonic at startup

### Day 2: TTS + Voice Quality
- Switched TTS model from `en_US-ljspeech-high` to `en_US-hfc_female-medium`
- Tuned VITS knobs: `noise_scale=0.75, noise_w=0.8, length_scale=0.95`
- Added TTS warmup synthesis at startup (removed later — no measurable benefit)
- Fixed pydantic settings env var mismatch (docker-compose passes `ZEROCLAW_TTS_VOICE_NAME` but field is `tts_voice_name` — no prefix)

### Day 3: Barge-In During Playback (Critical Fix)

**Problem:** `is_responding` was set to `False` after TTS synthesis completed (~1s), but audio playback continued for ~11s. During those ~10 seconds, the barge-in code path was completely inactive.

**Fix (3 parts):**

1. **Moved `is_responding = False` to AFTER response callbacks** in `_stream_and_speak()` finally block (both agents). Now `is_responding` stays `True` until audio playback actually finishes.

2. **Added `on_cancel_sync` callback** — synchronous callback from `agent.cancel_response()` → `chat_session._on_agent_cancel_sync()`. Sets `_cancel_playback = True` and clears audio buffer.

3. **Added `_cancel_playback` flag** checked in all loops within `_handle_response_end()`. On barge-in, the method exits immediately instead of waiting for audio stabilization.

**Files modified:**
- `src/agents/zeroclaw/sample_agent.py` — `_on_cancel_sync` field, `set_event_handlers()`, `cancel_response()`, finally block
- `src/agents/openclaw/sample_agent.py` — same changes
- `src/agents/base_agent.py` — `on_cancel_sync` in `set_event_handlers()` signature
- `src/agents/gemini/sample_agent.py` — signature update
- `src/agents/openai/sample_agent.py` — signature update
- `src/agents/remote_agent.py` — signature update
- `src/chat/chat_session.py` — `_cancel_playback` flag, `_on_agent_cancel_sync()`, cancellation checks in `_handle_response_end()`

### Additional Day 3 Changes
- Added `reset_vad_state` parameter to `stt_service.reset_turn()` — only resets Silero RNN state on barge-in, not on normal turn boundaries
- Added 4-frame (128ms) barge-in debounce — prevents single-frame noise from triggering cancel
- ZeroClaw: `cancel` message sent to WS on barge-in + `_drain_ws_until_done()` for clean reconnection
- STT state cleared in `cancel_response()` — prevents stale `_has_speech` from triggering instant re-barge-in

---

## 5. Open Problems

### P1: Client Sends Silence During Playback

**Severity:** HIGH — This is the #1 reason barge-in feels broken to the user

**Symptom:** Server logs show `rms=0.0000` for 10+ seconds while user is speaking. Barge-in eventually fires but only after client starts sending non-zero audio.

**Evidence from logs:**
```
VAD probe: conf=0.050 rms=0.0000 has_speech=False  (repeated for ~10s)
# Then suddenly:
Barge-in: sustained speech (4 frames), cancelling
```

**Root cause:** Almost certainly client-side. Likely candidates:
1. **Echo cancellation** — client mutes mic input while playing TTS audio to prevent feedback
2. **Mic permission** — client pauses audio capture during playback
3. **WebSocket backpressure** — client stops sending audio when receiving heavy frame data

**Investigation needed:** Check the client code for:
- `AudioContext` state during playback
- `MediaStream` track.enabled status
- WebSocket send behavior during heavy receive
- Any echo cancellation / noise suppression APIs being used

**This is NOT a server-side issue.** The server correctly processes audio when it arrives. The problem is that audio isn't arriving.

---

### P2: Whisper Transcription Accuracy

**Severity:** HIGH — Consistently poor throughout all 3 days

**Examples:**
- User says "Hi Nyx, How's things" → Whisper outputs "My next house, thanks."
- User says "stop stop stop" → Whisper outputs "I hope so."
- Short barge-in segments → Whisper cycles through all temperature fallbacks and hallucinates

**Contributing factors:**
1. `base.en` model is only 74M params — struggles with accented speech and short segments
2. `beam_size=5, best_of=5` increases computation AND hallucination risk (see P3)
3. Audio quality may be degraded by whatever is causing P1
4. After barge-in, the speech buffer may contain only 0.5s of noisy audio

**Options to explore:**
- Switch to `small.en` model (244M params, WER 3.4% vs base.en 5.6%) — set `STT_MODEL=small.en` in .env
- Reduce `beam_size` to 1 (greedy decode) — faster AND less hallucination on short audio
- Increase `_MIN_SPEECH_SAMPLES_24K` from 7200 (300ms) to 12000 (500ms)
- Consider alternative STT: Whisper large-v3, or external API (Deepgram, AssemblyAI)

---

### P3: Whisper beam_size Mismatch

**Severity:** MEDIUM — Easy fix but affects both accuracy and latency

**Current code (stt_service.py:472-473):**
```python
segments, info = cached_model.transcribe(
    audio_f32_16k,
    language="en",
    beam_size=5,    # <-- EXPENSIVE on CPU
    best_of=5,      # <-- INCREASES HALLUCINATION
    ...
)
```

**MEMORY.md says:** `beam_size=1` was the intended setting.

**Recommendation:** Change to `beam_size=1, best_of=1` (greedy decode). This is:
- Faster (single pass instead of 5)
- Less prone to hallucination on short segments
- Sufficient for `base.en` model (beam search mainly helps larger models)

---

### P4: Duplicate Code in ChatSession.stop()

**Severity:** LOW — Bug, but non-critical

**File:** `src/chat/chat_session.py`, lines 140-160

```python
async def stop(self) -> None:
    self.is_active = False
    self.is_interrupted = True
    await self._cancel_and_wait_task(self.inference_task)
    await self._cancel_and_wait_task(self.frame_emit_task)
    self._inference_executor.shutdown(wait=False)  # <-- duplicate
    await self.agent.disconnect()                   # <-- duplicate
    logger.info(f"Session {self.session_id} stopped.")  # <-- duplicate
    self._inference_executor.shutdown(wait=False)  # <-- duplicate
    await self.agent.disconnect()                   # <-- duplicate
    logger.info(f"Session {self.session_id} stopped.")  # <-- duplicate
```

Lines 149-153 and 155-160 are identical. The executor shutdown and agent disconnect are called twice. Should remove the duplicate block.

---

## 6. Recommended Next Steps

### Priority 1: Investigate Client Audio (P1)
This is the single biggest issue. Until the client sends audio during playback, barge-in will feel broken no matter how good the server-side code is.

**Action items:**
1. Add client-side logging for mic audio RMS during playback
2. Check if echo cancellation is zeroing the mic signal
3. Consider a "push-to-talk" mode as a fallback
4. Test with a different client that doesn't play audio (text-only) to isolate

### Priority 2: Fix Whisper Settings (P3)
**Quick win — 5 minutes:**
```python
# stt_service.py line 472-473
beam_size=1,
best_of=1,
```

### Priority 3: Evaluate STT Model (P2)
Test with `small.en`:
```env
STT_MODEL=small.en
```
This requires downloading the model:
```bash
uv run --with 'faster-whisper>=1.1.0' python -c "
from faster_whisper.utils import download_model
download_model('small.en', output_dir='pretrained_models/faster_whisper_small_en')
"
```

### Priority 4: Clean Up
- Fix duplicate `stop()` code (P4)
- Consider making `_SileroVADIterator` per-session instead of shared (class-level)
- Review hallucination blocklist — "stop" is on it (line 431) which means the user literally cannot say "stop" as valid input

---

## 7. Test Coverage

### tests/test_audio_pipeline.py (20 tests)
All passing as of last run in Docker container.

**TTS Tests (6):**
- Model load, synthesis output, PCM16 format, resampling, empty text, cancellation

**STT/VAD Tests (5):**
- Service init, VAD iterator, speech detection, Whisper transcription, hallucination filter

**Barge-In Tests (7):**
- cancel_response() state changes, STT reset on cancel, bargein counter logic, sustained speech trigger, on_cancel_sync callback, response_cancelled flag, TTS cancellation propagation

**Integration Tests (1):**
- Full pipeline: text → TTS → audio → STT/VAD → verification

**Missing test coverage:**
- Client audio silence scenario
- Multi-session VAD interference
- Whisper accuracy with different beam_size settings
- `_handle_response_end()` abort timing under `_cancel_playback`

### tests/test_bargein_playback.py (6 tests)
Tests for the barge-in during playback fix:

1. `cancel_response()` fires `on_cancel_sync`
2. `_cancel_playback` aborts `_handle_response_end` immediately
3. `_cancel_playback` resets on new response start
4. `is_responding` stays True during `_on_response_end` callback
5. Barge-in during playback triggers `_on_interrupted`
6. Stabilization loop exits on mid-loop cancel

---

## 8. Settings & Configuration

### Environment Variables (via .env)

```env
# Agent selection
AGENT_TYPE=sample_zeroclaw   # or sample_openclaw

# STT
STT_ENABLED=true
STT_MODEL=base.en            # or small.en for better accuracy
STT_PAUSE_THRESHOLD=0.50
STT_DELAY_SEC=0.2
STT_VAD_START_THRESHOLD=0.60
STT_VAD_END_THRESHOLD=0.35
STT_VAD_MIN_SILENCE_MS=280

# TTS
TTS_ENABLED=true
TTS_VOICE_NAME=en_US-hfc_female-medium
TTS_ONNX_MODEL_DIR=./pretrained_models/piper
TTS_NOISE_SCALE=0.75
TTS_NOISE_W_SCALE=0.8
TTS_LENGTH_SCALE=0.95
TTS_SENTENCE_MAX_CHARS=200

# ZeroClaw
ZEROCLAW_BASE_URL=http://127.0.0.1:5555
ZEROCLAW_HISTORY_MAX_MESSAGES=12
ZEROCLAW_TRANSCRIPT_SPEED=14.0
```

### Pydantic Settings Gotcha

The settings fields have NO prefix (e.g., `tts_voice_name`, not `zeroclaw_tts_voice_name`). Docker-compose env vars like `ZEROCLAW_TTS_VOICE_NAME` are silently ignored because pydantic looks for `TTS_VOICE_NAME`. This was a bug that caused the wrong TTS model to load.

---

## 9. Docker & Deployment Notes

- **Base image:** `python:3.10-slim`
- **Build arg:** `INSTALL_LOCAL_VOICE=true` to include piper + faster-whisper + silero
- **System deps:** `libsndfile1`, `libespeak-ng1`
- **Models:** Mounted as read-only volumes from host
- **No HF_HOME:** All models loaded from `pretrained_models/` directory
- **Package manager:** `uv` only (no pip)

### Volume Mounts (docker-compose.yml)
```yaml
volumes:
  - ./pretrained_models/piper:/app/pretrained_models/piper:ro
  - ./pretrained_models/faster_whisper_base_en:/app/pretrained_models/faster_whisper_base_en:ro
  - ./pretrained_models/silero_vad.onnx:/app/pretrained_models/silero_vad.onnx:ro
  - ./pretrained_models/wav2arkit:/app/pretrained_models/wav2arkit:ro
```

### Running Tests
```bash
docker compose exec avatar-server uv run python tests/test_audio_pipeline.py
docker compose exec avatar-server uv run python tests/test_bargein_playback.py
```

---

## Appendix: Key Line References

| What | File | Lines |
|------|------|-------|
| VAD iterator | stt_service.py | 31-102 |
| VAD state tracking | stt_service.py | 388-408 |
| Whisper transcription | stt_service.py | 438-502 |
| beam_size setting | stt_service.py | 472-473 |
| Hallucination blocklist | stt_service.py | 411-432 |
| Min segment duration | stt_service.py | 436, 444 |
| reset_turn() | stt_service.py | 212-225 |
| TTS synthesize | tts_service.py | 159-208 |
| TTS resampling | tts_service.py | 210-220 |
| Barge-in detection | zeroclaw/sample_agent.py | 314-328 |
| cancel_response() | zeroclaw/sample_agent.py | 966-989 |
| Finally block | zeroclaw/sample_agent.py | 863-901 |
| _cancel_playback check | chat_session.py | 330, 341, 349, 363 |
| _on_agent_cancel_sync | chat_session.py | 116-124 |
| _handle_response_end | chat_session.py | 328-416 |
| _handle_interrupted | chat_session.py | 553-592 |
| Duplicate stop() | chat_session.py | 140-160 |
