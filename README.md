# NyxClaw

**Voice-to-avatar server for Claw-based AI agents**


## What It Does

NyxClaw is a real-time WebSocket server that bridges our **Any Claw** companion mobile avatar app (based on [Myned](https://myned.ai)'s Nyx) and Claw-based AI backends ([OpenClaw](https://github.com/openclaw/openclaw), [ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw)). It runs the [**Wav2Arkit ONNX**](https://huggingface.co/myned-ai/wav2arkit_cpu) model on every audio chunk, generating 52 ARKit facial blendshapes at 30 FPS for real-time lip-sync on CPU.

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
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw
cp .env.example .env
# Edit .env with your backend settings (BASE_URL, AUTH_TOKEN — see Backend Setup below)

docker compose up --build -d
```

On first boot, NyxClaw downloads models, provisions a Cloudflare Tunnel, and starts serving. Check the logs for your secure URL:

```bash
docker compose logs -f nyxclaw
# Tunnel: wss://a3f7b2c1.nyxclaw.ai/ws
```

Your mobile app connects to that `wss://` URL — no port forwarding or TLS certs needed.

To enable local voice (Piper TTS + faster-whisper), set `INSTALL_LOCAL_VOICE=true` in `.env` before building.

### Install script (Linux / macOS / Windows)

Installs NyxClaw + Cloudflare Tunnel as system services. Handles `uv`, `cloudflared`, model downloads, tunnel provisioning, and service registration (systemd / launchd / Windows service) automatically.

```bash
# Linux / macOS
./install.sh

# Windows (PowerShell as Administrator)
.\install.ps1
```

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

> **One session at a time.** NyxClaw serves a single active connection — one avatar, one audio stream. You can pair multiple devices (phone, tablet, desktop) for convenience, but only one connects at a time. Treat the setup code like a password — anyone with it can pair a device and talk to your AI agent.

## Claw Patches

Backend-specific patches that add the avatar endpoint with structured `{speech, content}` output. When the LLM's response includes content better seen than heard (URLs, tables, structured data), the patch splits the response:

- **`speech`** → avatar speaks a short phrase ("Here's the Wikipedia page, take a look.")
- **`content`** → forwarded as a `rich_content` message (markdown) to the client

| Patch | Backend | Endpoint | Docs |
|-------|---------|----------|------|
| `claw_patches/openclaw/` | OpenClaw v2026.3.13 | `/v1/chat/completions/avatar` (HTTP SSE) | [README](claw_patches/openclaw/README.md) |
| `claw_patches/zeroclaw/` | ZeroClaw v0.5.0 | `/ws/avatar` (WebSocket) | [README](claw_patches/zeroclaw/README.md) |

Without the patch, all LLM output is treated as speech — no `rich_content` messages.

## Bring Your Own Tunnel

NyxClaw auto-provisions a free Cloudflare Tunnel on first boot (`wss://<id>.nyxclaw.ai`). This service has limited capacity. You can use any reverse proxy or tunneling solution instead — NyxClaw just needs something that terminates TLS and forwards traffic to `localhost:8080`:

- **Cloudflare Tunnel** (your own account) — run `cloudflared tunnel` with your own token
- **Tailscale** — encrypted mesh VPN, stable DNS, zero config
- **nginx / Caddy** — traditional reverse proxy with Let's Encrypt
- **ngrok** — quick dev tunnels

Set `AUTH_SETUP_CODE_URL=wss://your-domain/ws` in `.env` so the QR code contains your custom URL.

## Resource Requirements

| Component | Memory | Mode |
|-----------|--------|------|
| Python + FastAPI + ONNX Runtime | ~500 MB | Both |
| Wav2Arkit (blendshape inference) | ~200 MB | Both |
| faster-whisper small.en (speech recognition) | ~500 MB | Local only |
| Piper TTS VITS (speech synthesis) | ~100 MB | Local only |
| Silero VAD (voice activity detection) | ~10 MB | Local only |

**OpenAI Voice:** 1 GB RAM, 1 core minimum. Recommended 1.5 GB, 2 cores.
**Local Voice:** 2 GB RAM, 2 cores minimum. Recommended 3–4 GB, 4 cores (STT, TTS, and blendshapes run concurrently during barge-in).

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
