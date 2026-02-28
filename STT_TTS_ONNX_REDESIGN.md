# STT/TTS Pipeline Redesign (ONNX-First)

## Goal

Redesign the current audio pipeline to use:

- **STT:** Moonshine ONNX
- **VAD / speech detection / barge-in:** Silero VAD ONNX
- **TTS:** Pocket TTS ONNX (Hugging Face export)

Primary outcome: remove heavy PyTorch runtime dependency from online inference path and keep low-latency full-duplex behavior with reliable interruption handling.

---

## Current vs Target

### Current (simplified)

- Client sends 24kHz PCM16 audio
- STT service transcribes incrementally
- LLM response streams tokens
- TTS streams audio back
- Barge-in interrupts response during user speech

### Target (ONNX-first)

- Client sends 24kHz PCM16 audio
- **Silero VAD ONNX** runs continuously on short frames (after 24k -> 16k resample)
- Speech segments are cut by VAD start/end and pushed into **Moonshine ONNX STT**
- Finalized user utterance is sent to LLM
- LLM token stream is sentence-buffered and sent to **Pocket TTS ONNX stream()**
- While assistant audio is playing, VAD remains active; if speech detected, trigger **barge-in** (cancel LLM + TTS immediately)

---

## High-Level Architecture

1. **Audio Ingestion Layer**
   - Input: PCM16 mono 24kHz from websocket
   - Ring buffer for recent audio (pre-roll for VAD start padding)
   - Real-time resampler branch to 16kHz for VAD/STT

2. **VAD Engine (Silero ONNX)**
   - Frame-based inference (e.g., 20–32ms frames)
   - Stateful smoothing with hysteresis
   - Emits events:
     - `speech_start`
     - `speech_ongoing`
     - `speech_end`
   - Maintains confidence and silence counters

3. **Turn Segmentation + STT (Moonshine ONNX)**
   - On `speech_start`: open a user segment and include pre-roll
   - On `speech_end`: close segment, run STT decode, produce transcript
   - Optional interim decode cadence while segment is open
   - Output: `transcript_partial` and `transcript_final`

4. **Dialogue Controller**
   - Receives final user transcript
   - Starts/cancels LLM response stream
   - Emits assistant text deltas

5. **TTS Engine (Pocket TTS ONNX)**
   - Sentence queue from assistant deltas
   - Uses ONNX `stream()` for chunked PCM output
   - Pushes audio chunks to existing websocket output path

6. **Barge-In Controller**
   - Active during assistant speech playback
   - If VAD confirms user speech start (and debounce passes), executes:
     1) cancel TTS stream
     2) cancel LLM generation
     3) clear pending TTS queue
     4) mark conversation interrupted
     5) begin new user capture

---

## Barge-In + Speech Detection Design

### VAD State Machine

States:

- `IDLE` (no speech)
- `MAYBE_SPEECH` (short positive burst, waiting debounce)
- `IN_SPEECH`
- `MAYBE_END` (short silence burst, waiting end debounce)

Transitions (recommended starting values):

- `IDLE -> MAYBE_SPEECH` when `vad_prob >= start_threshold`
- `MAYBE_SPEECH -> IN_SPEECH` after `min_speech_ms` continuous speech
- `IN_SPEECH -> MAYBE_END` when `vad_prob < end_threshold`
- `MAYBE_END -> IDLE` after `min_silence_ms` silence
- Any false burst returns to previous stable state

Suggested defaults (tune with logs):

- `start_threshold = 0.60`
- `end_threshold = 0.35`
- `min_speech_ms = 120`
- `min_silence_ms = 280`
- `pre_roll_ms = 200`
- `max_segment_ms = 12000` (force finalize safety)

### Barge-In Trigger

While assistant is speaking, only trigger interruption when all are true:

- VAD enters `IN_SPEECH` and speech persists >= `barge_in_min_ms` (e.g. 140ms)
- Current assistant playback has started (avoid race before first chunk)
- Not within `barge_in_cooldown_ms` from previous cancel (e.g. 300ms)

Action on trigger:

- Fire `interrupt` event to client
- Cancel assistant token stream task
- Cancel ONNX TTS stream generator
- Flush TTS queue and playback buffers
- Start new user turn capture with fresh segment ID

### Echo/Double-Talk Guard

To reduce false barge-ins from speaker leakage:

- Apply optional AEC upstream if available
- Add short suppression window right after assistant chunk emission (`echo_suppress_ms` 60–100)
- Optionally require spectral energy gate in addition to VAD prob

---

## Moonshine ONNX STT Integration

## Package and API

- Install package from Moonshine ONNX docs (Git-based package)
- Use `moonshine_onnx` import path
- Base API is file-level `transcribe(...)`; streaming behavior should be implemented in our adapter using VAD segments and optional periodic partial decode.

### Adapter Interface (proposed)

`OnnxSttAdapter` methods:

- `append_audio_24k(pcm_bytes)`
- `on_vad_event(event)`
- `flush_segment() -> transcript`
- `reset_turn()`

Responsibilities:

- convert PCM16 bytes -> float32 mono
- resample 24k to 16k as required by model setup
- maintain segment buffers and decode cadence

---

## Pocket TTS ONNX Integration

## Weights and Files Needed

From `KevinAHM/pocket-tts-onnx` we need these artifacts (ONNX + tokenizer):

- `onnx/flow_lm_main_int8.onnx`
- `onnx/flow_lm_flow_int8.onnx`
- `onnx/mimi_decoder_int8.onnx`
- `onnx/mimi_encoder.onnx`
- `onnx/text_conditioner.onnx`
- `tokenizer.model`

Optional FP32 alternatives can be supported behind config flag.

### Weight Download Strategy

Use `huggingface_hub.snapshot_download` at build/startup (same pattern already used for wav2arkit):

- repo id: `KevinAHM/pocket-tts-onnx`
- local dir: `pretrained_models/pocket_tts_onnx`
- allow patterns: `onnx/*`, `tokenizer.model`, `pocket_tts_onnx.py` (if needed)

Caching:

- Docker image: bake model files into image for cold-start-free deployment
- Local dev: cache under `pretrained_models/` and skip re-download if present

License/usage note:

- Respect model card terms (CC BY 4.0 for model weights) and attribution requirements.

### TTS Adapter Interface (proposed)

`OnnxTtsAdapter` methods:

- `load(model_dir, voice_ref=None, temperature=0.7, lsd_steps=1)`
- `stream(text) -> AsyncIterator[bytes]`
- `cancel()`

Behavior:

- sentence-level streaming from assistant deltas
- produce PCM16 chunks at 24kHz to preserve existing downstream contracts

---

## Configuration Additions

Add env/config keys (names proposed):

- `STT_BACKEND=moonshine_onnx`
- `VAD_BACKEND=silero_onnx`
- `TTS_BACKEND=pocket_onnx`
- `VAD_START_THRESHOLD=0.60`
- `VAD_END_THRESHOLD=0.35`
- `VAD_MIN_SPEECH_MS=120`
- `VAD_MIN_SILENCE_MS=280`
- `BARGE_IN_MIN_MS=140`
- `BARGE_IN_COOLDOWN_MS=300`
- `POCKET_TTS_ONNX_MODEL_DIR=./pretrained_models/pocket_tts_onnx`
- `POCKET_TTS_ONNX_VOICE_REF=./voices/default.wav`
- `POCKET_TTS_ONNX_LSD_STEPS=1`
- `POCKET_TTS_ONNX_TEMPERATURE=0.7`

---

## Dependency Plan (ONNX-first)

Core runtime:

- `onnxruntime`
- `numpy`
- `scipy` (resampling)
- `soundfile` (if needed for voice ref/audio utils)
- `sentencepiece`
- Moonshine ONNX package
- Silero VAD ONNX usage path

Remove from default runtime where possible:

- `torch` in default path (currently pulled by `pocket-tts`)

Approach:

- Keep legacy PyTorch-based TTS as optional backend for fallback
- Make ONNX pipeline default and keep torch-based stack in optional extra

---

## Operational Metrics to Add

For each turn, log:

- VAD first speech ms
- VAD speech end ms
- STT decode latency
- LLM first token latency
- TTS first audio latency
- End-to-end user-end-to-audio-start latency
- Barge-in count and trigger confidence
- False-barge-in counter (cancelled but no valid transcript followed)

---

## Risks and Mitigations

1. **Moonshine ONNX not truly streaming**
   - Mitigation: segment-based pseudo-streaming with periodic partial decode.

2. **Community Pocket ONNX export compatibility drift**
   - Mitigation: pin commit hash and add startup model integrity checks.

3. **Barge-in false positives from echo**
   - Mitigation: hysteresis + debounce + optional echo suppression window.

4. **Thread contention in ONNX runtime**
   - Mitigation: set ORT session thread config (`intra_op`, `inter_op`) per model.

---

## Review TODO List

### Phase 1: Foundations

- [ ] Add backend abstraction interfaces for STT, VAD, and TTS.
- [ ] Add new env/config keys and defaults.
- [ ] Add model downloader utility for Pocket TTS ONNX weights and tokenizer.

### Phase 2: VAD + STT

- [ ] Implement `SileroOnnxVadService` with state machine and event callbacks.
- [ ] Implement `MoonshineOnnxSttService` with segment buffer + decode.
- [ ] Wire VAD events to STT segment lifecycle.
- [ ] Add tuning config for thresholds, debounce, and silence windows.

### Phase 3: TTS

- [ ] Implement `PocketOnnxTtsService` wrapper around ONNX stream API.
- [ ] Replace sentence-to-audio generation path with backend selector.
- [ ] Implement cancellation-safe TTS stream for barge-in.

### Phase 4: Barge-In Controller

- [ ] Add dedicated barge-in coordinator to cancel LLM/TTS atomically.
- [ ] Add cooldown and echo suppression guards.
- [ ] Emit explicit interruption events and metrics.

### Phase 5: Docker + Dependencies

- [ ] Move torch-based dependencies to optional extras.
- [ ] Make ONNX dependency set the default runtime install.
- [ ] Update Dockerfile apt packages to runtime-minimal set for ONNX stack.

### Phase 6: Validation

- [ ] Add integration tests for turn detection and interruption behavior.
- [ ] Add regression test for rapid back-to-back barge-ins.
- [ ] Validate latency budget and CPU usage under load.
