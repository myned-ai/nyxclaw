# OpenClaw Avatar — Local Setup Log

Step-by-step record of everything done to get the full voice pipeline working
on **Windows** (native PowerShell, no WSL).

```
Architecture:
  Client mic → WebSocket → avatar-chat-server → STT (moshi-server) → text
                                                                       ↓
  Client ← sync_frame (audio + blendshapes) ← TTS (pocket-tts) ← OpenClaw LLM
```

---

## Prerequisites already in place

| Component | Status | Details |
|---|---|---|
| Python 3.10+ | ✅ | via `uv` |
| OpenClaw Gateway | ✅ | Docker `openclaw:local` on `localhost:18789` |
| pocket-tts (TTS) | ✅ | v1.1.1, voice "eponine", 24 kHz |
| wav2arkit (blendshapes) | ✅ | ONNX model, onnxruntime 1.23.2 |
| avatar-chat-server | ✅ | Running on port 8080, sends `sync_frame` |

---

## Step 1 — Install Rust Toolchain ✅

moshi-server is a Rust binary. We need the Rust compiler and Cargo.

### 1.1 Download and run the Rust installer

On Windows, download and run `rustup-init.exe` from https://rustup.rs :

```powershell
# Download rustup-init.exe
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe"

# Run the installer (default settings: stable toolchain, add to PATH)
& "$env:TEMP\rustup-init.exe" -y
```

### 1.2 Reload PATH

After install, reload the terminal or run:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
```

### 1.3 Verify

```powershell
rustc --version   # → rustc 1.93.1 (01f6ddf75 2026-02-11)
cargo --version   # → cargo 1.93.1 (083ac5135 2025-12-15)
```

---

## Step 2 — Build moshi-server (Kyutai STT) ✅

moshi-server wraps the `kyutai/stt-1b-en_fr-candle` model (~1B params) in a
WebSocket server. Model weights (~2 GB) are downloaded from HuggingFace Hub
automatically on first run.

### 2.1 Prerequisites check

- **C/C++ compiler**: On Windows, Visual Studio Build Tools with "Desktop
  development with C++" workload must be installed (provides MSVC).
- **Python**: Required for libpython linkage during build.

### 2.2 Install moshi-server (CPU-only)

```powershell
cargo install moshi-server@0.6.4
```

> Build takes ~5-15 minutes depending on hardware.
> Binary installs to `~/.cargo/bin/moshi-server.exe`.

### 2.3 Verify

```powershell
moshi-server --help
```

---

## Step 3 — Create STT Configuration ✅

Config was already created at `configs/stt.toml` per AUDIO_PIPELINE.md.

Create `configs/stt.toml` in the project root:

```toml
# configs/stt.toml — Kyutai STT server configuration
static_dir = "./static/"
log_dir = "./logs/stt"
instance_name = "stt"

authorized_ids = ["avatar_stt_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"

lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
asr_delay_in_tokens = 6

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

---

## Step 4 — Run moshi-server ✅

```powershell
moshi-server worker --config configs/stt.toml --port 8090
```

First run downloads ~2 GB of model weights from HuggingFace Hub to
`~/.cache/huggingface/`. Subsequent runs are instant.

Expected output:
```
INFO  Loading model from hf://kyutai/stt-1b-en_fr-candle/model.safetensors
INFO  Worker listening on 0.0.0.0:8090
```

---

## Step 5 — Enable STT in avatar-chat-server ✅

Edit `.env`:

```env
STT_ENABLED=true
```

Restart the avatar-chat-server. Look for:
```
INFO  STT service connected
INFO  OpenClaw agent ready (stt=True, tts=True, model=openclaw:main)
```

---

## Step 6 — Test End-to-End

Open `test.html` in a browser. Click the microphone button, speak, and verify:
1. Server logs show STT words being transcribed
2. OpenClaw receives the text and responds
3. Widget plays audio with lip-sync (sync_frame messages)

---

## Fixes Applied (this session)

| Fix | File | Issue |
|---|---|---|
| TTS voice quality | `src/services/tts_service.py` | `init_states()` produced silence; switched to `get_state_for_audio_prompt('eponine')` |
| wav2arkit ONNX path | `.env` | `ONNX_MODEL_PATH` had double `src/src/` prefix; fixed to `./pretrained_models/wav2arkit_cpu.onnx` |
| TTS voice config | `src/agents/openclaw/openclaw_settings.py` | Added `tts_voice_name` setting |
| Voice passthrough | `src/agents/openclaw/sample_agent.py` | Pass `voice_name` to TTSService constructor |
