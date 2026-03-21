# NyxClaw

**Voice-to-avatar server for Claw-based AI agents**


## What It Does

NyxClaw is a real-time WebSocket server that bridges a 3D avatar frontend and Claw-based AI backends (OpenClaw, ZeroClaw, etc). It runs the [**Wav2Arkit ONNX**](https://huggingface.co/myned-ai/wav2arkit_cpu) model on every audio chunk, generating 52 ARKit facial blendshapes at 30 FPS so a 3D avatar can lip-sync in real time on CPU.

Two voice pipelines are supported:

| | **OpenAI Voice** (`VOICE_MODE=openai`) | **Local Voice** (`VOICE_MODE=local`) |
|---|---|---|
| **STT** | OpenAI Realtime API (server-side) | faster-whisper + Silero VAD (CPU) |
| **TTS** | OpenAI TTS API | Piper VITS ONNX (CPU) |
| **Install** | `uv sync` (included by default) | `uv sync --extra local_voice` |
| **Requires** | `OPENAI_API_KEY` in `.env` | ~1 GB of model downloads |

Both pipelines run Wav2Arkit on every audio chunk for facial animation.

## Quick Start

### Docker (recommended)

```bash
# 1. Clone and configure
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw
cp .env.example .env
# Edit .env with your settings (see Configuration below)

# 2. Build and run (port 8081)
# The Wav2Arkit model is downloaded automatically during the Docker build.
docker compose up --build -d

# 3. View logs
docker compose logs -f
```

To enable local voice (Piper TTS + faster-whisper) in Docker, set `INSTALL_LOCAL_VOICE=true` in `.env` before building.

### Local Development

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Clone and configure
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw
cp .env.example .env

# 3. Install dependencies
uv sync                        # OpenAI voice mode
# or: uv sync --extra local_voice  # local voice mode (Piper TTS + faster-whisper)

# 4. Download models
mkdir -p pretrained_models/wav2arkit
uv run --with "huggingface_hub[cli]" huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models/wav2arkit

# 5. Run
uv run python src/main.py
```

Server starts at `http://localhost:8080`

## Backend Setup

NyxClaw supports two Claw backends. Set `AGENT_TYPE` in `.env` to switch.

### OpenClaw

Requires the nyxclaw avatar patch applied to OpenClaw. See [claw_patches/openclaw/README.md](claw_patches/openclaw/README.md) for full setup (patching, auth, AGENTS.md prompt).

```env
AGENT_TYPE=openclaw
BASE_URL=http://127.0.0.1:18789
AUTH_TOKEN=your-openclaw-gateway-token
USE_AVATAR_ENDPOINT=true
```

### ZeroClaw

Requires the nyxclaw avatar patch applied to ZeroClaw. See [claw_patches/zeroclaw/README.md](claw_patches/zeroclaw/README.md) for full setup (patching, auth, AGENTS.md prompt).

```env
AGENT_TYPE=zeroclaw
BASE_URL=http://127.0.0.1:42617
AUTH_TOKEN=zc_YOUR_TOKEN_HERE
USE_AVATAR_ENDPOINT=true
```

### Unpatched Backends

Both backends work without the avatar patch — set `USE_AVATAR_ENDPOINT=false` (the default). NyxClaw will use the standard `/v1/chat/completions` (OpenClaw) or `/ws/chat` (ZeroClaw) endpoints. Rich content (`rich_content` messages) won't be available — all LLM output is treated as speech.

## Configuration

All settings are configured via environment variables or `.env` file. See [.env.example](.env.example) for the full template.
xs
## Rich Content

When the LLM's response includes content better seen than heard (URLs, tables, structured data), the avatar patch splits the response:

- **`speech`** → avatar speaks a short phrase ("Here's the Wikipedia page, take a look.")
- **`content`** → forwarded as a `rich_content` message (markdown) to the client

```json
{"type": "rich_content", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome"}
```

This requires a patched backend (`USE_AVATAR_ENDPOINT=true`). Without the patch, all LLM output is treated as speech.

## WebSocket Protocol (`/ws`)

**Client → Server:**

| Type | Description |
|------|-------------|
| `audio_stream_start` | Start audio session |
| `audio` | Audio chunk (base64 PCM16 24kHz mono) |
| `text` | Text message to AI |
| `interrupt` | Stop AI response |

**Server → Client:**

| Type | Description |
|------|-------------|
| `config` | Audio settings (sent on connect) |
| `audio_start` | AI response started |
| `sync_frame` | Audio + 52 ARKit blendshapes (30 FPS) |
| `audio_end` | AI response finished |
| `transcript_delta` | Streaming text fragment |
| `transcript_done` | Complete turn transcript |
| `rich_content` | Markdown content for the chat view |
| `avatar_state` | `"Listening"` or `"Responding"` |

## Claw Patches

Backend-specific patches that add the avatar endpoint with structured `{speech, content}` output:

| Patch | Backend | Endpoint | Docs |
|-------|---------|----------|------|
| `claw_patches/openclaw/` | OpenClaw v2026.3.13 | `/v1/chat/completions/avatar` (HTTP SSE) | [README](claw_patches/openclaw/README.md) |
| `claw_patches/zeroclaw/` | ZeroClaw v0.5.0 | `/ws/avatar` (WebSocket) | [README](claw_patches/zeroclaw/README.md) |


## Resource Requirements

| Component | Memory |
|-----------|--------|
| Wav2Arkit ONNX | ~200 MB |
| ONNX Runtime | ~200 MB |
| Python + FastAPI | ~300 MB |
| faster-whisper small.en (local voice only) | ~500 MB |
| Piper TTS (local voice only) | ~100 MB |

**Minimum:** 2 GB RAM, 2 CPU cores. **Recommended:** 3–4 GB RAM, 4 cores.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
