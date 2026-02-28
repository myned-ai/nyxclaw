# Audio Pipeline Setup: STT & TTS

This document explains how to set up the server-side Speech-to-Text (STT) and
Text-to-Speech (TTS) components used by NyxClaw.

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │                  NyxClaw                    │
                          │                                             │
  Client ──WebSocket──►   │  audio ──► STT Service ──► text             │
                          │           (faster-whisper)   │              │
                          │                    Claw Agent (LLM)         │
                          │                              │              │
                          │            text ◄── SSE/WS stream           │
                          │              │                              │
                          │  audio ◄── TTS Service                      │
                          │           (Piper VITS ONNX)                 │
  Client ◄──WebSocket──   │                                             │
                          └─────────────────────────────────────────────┘

  STT: faster-whisper (CTranslate2, int8) + Silero VAD (ONNX) → in-process
  TTS: Piper VITS ONNX                                         → in-process
  LLM: Claw Agent (OpenClaw HTTP or ZeroClaw WebSocket)
```

NyxClaw and the Claw agent backend are designed to coexist on the **same
machine**, communicating over loopback.

---

## 1. STT — Kyutai Speech-to-Text (Rust)

The STT component uses **moshi-server**, a production Rust binary from Kyutai
that exposes Kyutai STT over WebSocket. It uses the
[candle](https://github.com/huggingface/candle) ML framework and can run on
CPU or CUDA GPU.

- **Model:** `kyutai/stt-1b-en_fr-candle` (~1B params, English + French)
- **Input:** 24 kHz PCM audio streamed over WebSocket
- **Output:** Real-time word-level transcription with timestamps + semantic VAD
- **WebSocket endpoint:** `ws://127.0.0.1:8090/api/asr-streaming`

### 1.1 Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Rust toolchain | stable ≥ 1.75 | `curl https://sh.rustup.rs -sSf \| sh` or [rustup.rs](https://rustup.rs) |
| C/C++ compiler | GCC/Clang/MSVC | Linux: `apt install build-essential`, Windows: Visual Studio Build Tools |
| Python 3.10+ | (for libpython linkage) | Required even for STT — the moshi-server binary links against libpython |
| CUDA toolkit *(optional)* | 12.x | Only if using `--features cuda` for GPU acceleration |

> **Windows note:** Building moshi-server natively on Windows is supported but
> may require extra setup for the `sentencepiece` C++ dependency. If you hit
> build issues, consider using **WSL 2** (Ubuntu) which is the tested
> environment by Kyutai.

### 1.2 Install moshi-server

**CPU-only (no GPU required):**

```bash
cargo install moshi-server@0.6.4
```

**With CUDA GPU acceleration:**

```bash
cargo install --features cuda moshi-server@0.6.4
```

**With Metal (macOS Apple Silicon):**

```bash
cargo install --features metal moshi-server@0.6.4
```

The binary is installed to `~/.cargo/bin/moshi-server`. Make sure this is in
your `PATH`.

> **Build troubleshooting:**
>
> - If you see `no module named 'huggingface_hub'` or similar Python errors,
>   set `LD_LIBRARY_PATH` (Linux) before building:
>   ```bash
>   export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
>   ```
> - On GCC 15+, you may need: `export CXXFLAGS="-include cstdint"` (sentencepiece fix)
> - To force a clean rebuild: `cargo install --force moshi-server@0.6.4`

### 1.3 Create STT Configuration

Create a TOML config file for the STT server. Save this as
`configs/stt.toml` in the project root:

```toml
# configs/stt.toml — Kyutai STT server configuration
static_dir = "./static/"
log_dir = "./logs/stt"
instance_name = "stt"

# Authentication tokens accepted by the server.
# The avatar-chat-server must send one of these when connecting.
authorized_ids = ["avatar_stt_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"

# Model files — downloaded automatically from Hugging Face Hub on first run
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
asr_delay_in_tokens = 6

# Batch size: higher = more concurrent streams but more memory/latency.
# For a single-user avatar, 1 is sufficient.
batch_size = 1
conditioning_learnt_padding = true
temperature = 0.25

[modules.asr.model]
audio_vocab_size = 2049
text_in_vocab_size = 8001
text_out_vocab_size = 8000
audio_codebooks = 20

[modules.asr.model.transformer]
d_model = 2048
num_heads = 16
num_layers = 16
dim_feedforward = 8192
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 750
max_period = 100000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "Rope"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 40960

[modules.asr.model.extra_heads]
num_heads = 4
dim = 6
```

### 1.4 Run the STT Server

```bash
moshi-server worker --config configs/stt.toml --port 8090
```

On first run, model weights (~2 GB) are downloaded from Hugging Face Hub and
cached locally in `~/.cache/huggingface/`.

The server will listen on `ws://127.0.0.1:8090/api/asr-streaming`.

### 1.5 Verify STT is Running

You should see output like:

```
INFO  Loading model from hf://kyutai/stt-1b-en_fr-candle/model.safetensors
INFO  Worker listening on 0.0.0.0:8090
```

### 1.6 CPU Performance Notes

| Setup | Realtime Factor | Notes |
|---|---|---|
| CPU (modern x86, AVX2) | ~1–3× realtime | Sufficient for single-user avatar |
| CUDA GPU (e.g. RTX 3060) | ~10–20× realtime | Overkill for single user, great for multi |
| L40S GPU | 64 streams at 3× RT | Production scale (Unmute.sh) |

For a single avatar session on CPU, the 1B model should keep up with real-time
speech. If you experience lag, consider the quantized GGUF model variant:

```toml
# In stt.toml, replace the model file with the quantized version:
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model-q4k.gguf"
```

---

## 2. STT — Standalone CLI (Alternative for Testing)

If you don't need real-time WebSocket streaming (e.g. for batch testing), you
can use the standalone `stt-rs` CLI tool instead.

### 2.1 Build from Source

```bash
git clone https://github.com/kyutai-labs/delayed-streams-modeling.git
cd delayed-streams-modeling/stt-rs
cargo build --release
```

The binary is at `target/release/kyutai-stt-rs`.

### 2.2 Run on a File

```bash
# CPU mode (explicitly)
./target/release/kyutai-stt-rs --cpu --timestamps audio.wav

# With VAD (voice activity detection)
./target/release/kyutai-stt-rs --cpu --vad --timestamps audio.wav

# With quantized model for faster CPU inference
./target/release/kyutai-stt-rs --cpu --model-path model-q4k.gguf audio.wav
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--cpu` | Force CPU inference (auto-detects CUDA/Metal otherwise) |
| `--timestamps` | Show word-level timestamps |
| `--vad` | Enable voice activity detection |
| `--hf-repo <repo>` | HF repo (default: `kyutai/stt-1b-en_fr-candle`) |
| `--model-path <file>` | Model file in repo (default: `model.safetensors`) |

---

## 3. TTS — Kyutai Pocket TTS (Python)

The TTS component uses **pocket-tts**, a lightweight Python library by Kyutai
optimized for CPU inference. It runs in-process within the avatar-chat-server.

- **Model:** ~100M parameters, CPU-optimized
- **Output:** 24 kHz PCM audio
- **Latency:** ~200ms to first audio chunk, ~6× realtime throughput on CPU
- **Streaming:** Yes, via `generate_audio_stream()` yielding audio chunks

### 3.1 Install

```bash
uv add pocket-tts
```

Or with pip:

```bash
pip install pocket-tts
```

### 3.2 Voice Setup

Pocket TTS supports voice cloning from a reference audio file. Place your
voice reference WAV files in a `voices/` directory:

```
voices/
  default.wav      # Default avatar voice (5-15 seconds, clean speech)
```

**Voice requirements:**
- WAV format, mono, any sample rate (resampled internally to 24 kHz)
- 5–15 seconds of clean speech, minimal background noise
- The voice characteristics (pitch, tone, pace) will be cloned

You can find pre-made voices in the [Kyutai voice repository](https://huggingface.co/kyutai/tts-voices).

### 3.3 Python API Quick Reference

```python
from pocket_tts import TTSModel

# Load model (done once at startup — ~2-3 seconds)
model = TTSModel.load_model()

# Prepare voice state from reference audio
state = model.get_state_for_audio_prompt("voices/default.wav")

# Streaming generation (yields audio chunks as 1D torch tensors)
for chunk in model.generate_audio_stream(state, "Hello, how can I help you?"):
    pcm_bytes = (chunk.numpy() * 32767).astype("int16").tobytes()
    # Send pcm_bytes to client via WebSocket

# Or generate all at once
audio = model.generate_audio(state, "Hello!")
# audio.shape = (N,), sample_rate = model.sample_rate  (24000)
```

### 3.4 Voice Preloading (Optional Optimization)

For faster startup, export voice state to safetensors format:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("voices/default.wav")

# Export to safetensors for instant loading later
model.export_model_state(state, "voices/default.safetensors")
```

Then load the precomputed state at runtime:

```python
state = model.get_state_for_safetensors_prompt("voices/default.safetensors")
```

---

## 4. Environment Variables

Add these to your `.env` file:

```bash
# Agent type
AGENT_TYPE=sample_openclaw

# Agent backend (same machine)
BASE_URL=http://127.0.0.1:19001
AUTH_TOKEN=your_agent_token
AGENT_MODEL=openclaw:main

# STT (faster-whisper, in-process)
STT_ENABLED=true
STT_MODEL=small.en

# TTS (Piper VITS ONNX, in-process)
TTS_ENABLED=true
TTS_VOICE_NAME=en_US-hfc_female-medium
# TTS_VOICE_PATH=voices/default.wav  # Optional: WAV for voice reference
```

---

## 5. Running the Full Stack

Start both services (each in its own terminal):

**Terminal 1 — Claw Agent Backend:**

```bash
# OpenClaw example (start per OpenClaw docs)
openclawd

# Or ZeroClaw
zeroclawd
```

**Terminal 2 — NyxClaw Server:**

```bash
uv run python src/main.py
```

Both communicate over loopback:

| Service | Address | Protocol |
|---|---|---|
| Claw Agent | `http://127.0.0.1:19001` (OpenClaw) or `ws://127.0.0.1:5555` (ZeroClaw) | HTTP SSE / WebSocket |
| NyxClaw | `ws://0.0.0.0:8080` | WebSocket |

---

## 6. Request Flow

1. **Client** connects via WebSocket to NyxClaw
2. **Client** streams microphone audio (24 kHz PCM, 16-bit LE)
3. **NyxClaw** runs audio through **Silero VAD** (ONNX) for speech detection
4. When speech is detected, audio is transcribed by **faster-whisper** (CTranslate2)
5. When VAD detects end-of-turn, NyxClaw sends accumulated text to the
   **Claw Agent** (OpenClaw HTTP SSE or ZeroClaw WebSocket)
6. As text tokens arrive, they are buffered into sentences
7. Complete sentences are fed to **Piper TTS** (VITS ONNX)
8. Generated PCM audio chunks are sent back to the **client** via WebSocket
9. Client plays audio and drives avatar lip-sync (Wav2Arkit pipeline)

---

## 7. Troubleshooting

### moshi-server won't build

- **Windows:** Install Visual Studio Build Tools with C++ workload. If
  sentencepiece fails, use WSL 2 (Ubuntu) instead.
- **Linux:** `sudo apt install build-essential pkg-config libssl-dev`
- **Python linkage errors:** Ensure `LD_LIBRARY_PATH` includes libpython dir:
  ```bash
  export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
  ```

### Model download fails

- Set `HUGGING_FACE_HUB_TOKEN` if the model requires authentication:
  ```bash
  export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
  ```
- Or pre-download with the HF CLI:
  ```bash
  uvx --from 'huggingface_hub[cli]' huggingface-cli download kyutai/stt-1b-en_fr-candle
  ```

### STT is too slow on CPU

- Use the quantized GGUF model (`model-q4k.gguf`) — ~2-4× faster than fp32
- Reduce `batch_size` to 1 in `stt.toml`
- Ensure no other CPU-heavy processes are competing
- Consider adding a GPU (even a modest one like GTX 1660 helps significantly)

### pocket-tts install fails

- Requires Python ≥ 3.10 and PyTorch (CPU version is fine):
  ```bash
  uv add pocket-tts torch --extra-index-url https://download.pytorch.org/whl/cpu
  ```

---

## 8. References

- [Kyutai STT & TTS (delayed-streams-modeling)](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Unmute — full voice AI system using STT + TTS](https://github.com/kyutai-labs/unmute)
- [Pocket TTS Python API](https://kyutai-labs.github.io/pocket-tts/API%20Reference/python-api/)
- [moshi-server crate](https://crates.io/crates/moshi-server)
- [Kyutai STT models on Hugging Face](https://huggingface.co/collections/kyutai/speech-to-text-685403682cf8a23ab9466886)
- [Kyutai TTS voices](https://huggingface.co/kyutai/tts-voices)
- [Candle ML framework](https://github.com/huggingface/candle)
