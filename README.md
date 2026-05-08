<h1 align="center">NyxClaw</h1>

<h3 align="center">Every claw deserves a face. <sub>🦞</sub></h3>

<p align="center">
  Give any AI agent a real-time face and voice.<br>
  <strong>Open source. Runs on your machine. No GPU required.</strong>
</p>

<p align="center">
  <a href="https://github.com/myned-ai/nyxclaw/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-FF4D6D?style=flat-square" alt="MIT License"></a>
  <a href="https://nyxclaw.ai"><img src="https://img.shields.io/badge/website-nyxclaw.ai-FF4D6D?style=flat-square" alt="Website"></a>
  <a href="https://huggingface.co/myned-ai/wav2arkit_cpu"><img src="https://img.shields.io/badge/🤗-wav2arkit__cpu-FF4D6D?style=flat-square" alt="Hugging Face model"></a>
  <a href="https://nyxclaw.ai"><img src="https://img.shields.io/badge/App_Store-Coming_Soon-FF4D6D?style=flat-square&logo=apple&logoColor=white" alt="App Store"></a>
  <a href="https://play.google.com/store/apps/details?id=ai.nyxclaw.app"><img src="https://img.shields.io/badge/Google_Play-Get_App-FF4D6D?style=flat-square&logo=googleplay&logoColor=white" alt="Google Play"></a>
  <a href="https://buymeacoffee.com/nyxclaw"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FF4D6D?style=flat-square&logo=buymeacoffee&logoColor=white" alt="Buy me a coffee"></a>
  <a href="https://x.com/myned_ai"><img src="https://img.shields.io/badge/follow-@myned__ai-FF4D6D?style=flat-square&logo=x&logoColor=white" alt="X / Twitter"></a>
</p>

<p align="center">
  <img src="docs/nyxclaw_intro.gif" alt="NyxClaw avatar demo" width="280">
</p>

---

## What It Does

NyxClaw runs locally and turns any Claw agent into a talking, listening, lip-syncing avatar
on your phone. Audio in → 52 ARKit blendshapes out @ 30 FPS, all on CPU.

- **Real-time animation** — Wav2Arkit ONNX, 52 ARKit blendshapes @ 30 FPS, CPU-only
- **Tool-call fillers** — the avatar talks while the AI works, no awkward silence
- **Thinking silence** — keeps breathing/blinking during processing gaps (up to 5 s)
- **Synced transcripts** — text lands with the audio via delivery-clock tracking
- **Barge-in** — interrupt mid-sentence; LLM + TTS + playback cancel in ~128 ms
- **Rich content** — `{speech, content}` splits what the avatar says vs. what the app shows

## Your server. Your data.

NyxClaw runs on **your** machine, right alongside your claw. End-to-end encrypted,
cryptographically paired. **No cloud. No relay. No telemetry.**

| | |
|---|---|
| 🔐 **Ed25519 Auth** — device pairing via cryptographic challenge. No passwords on the wire. | 🔒 **End-to-end WSS** — auto-provisioned Cloudflare Tunnel. No port forwarding, no certs. |
| 📱 **QR Pairing** — scan to connect. One device at a time. Treat the code like a password. | 🏠 **Self-hosted** — Docker or install script. Your machine, your data, your rules. |

## Two voice pipelines

Two reference pipelines out of the box — **OpenAI Realtime** for cloud-grade voice quality,
**Local CPU** for total privacy. Swap or extend either one.

| | 🌐 OpenAI Voice | 🖥️ Local Voice |
|---|---|---|
| **STT** | OpenAI Realtime API | faster-whisper + Silero VAD |
| **TTS** | OpenAI TTS API | Piper VITS ONNX |
| **Install** | `uv sync` | `uv sync --extra local_voice` |
| **Footprint** | ~1 GB RAM, 1 core | ~2 GB RAM, 2 cores |
| **Privacy** | OpenAI sees the audio | Nothing leaves your machine |

Both pipelines run **Wav2Arkit ONNX** on CPU — 52 ARKit blendshapes at 30 FPS.
&nbsp;&nbsp;[🤗 Model card](https://huggingface.co/myned-ai/wav2arkit_cpu)

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw
cp .env.example .env
# Edit .env with your backend settings (BASE_URL, AUTH_TOKEN — see Backend Setup below)

docker compose up --build -d
```

On first boot, NyxClaw downloads models, provisions a Cloudflare Tunnel, and starts serving.
Check the logs for your secure URL:

```bash
docker compose logs -f nyxclaw
# Tunnel: wss://a3f7b2c1.nyxclaw.ai/ws
```

Your mobile app connects to that `wss://` URL — no port forwarding or TLS certs needed.

To enable local voice (Piper TTS + faster-whisper), set `INSTALL_LOCAL_VOICE=true` in `.env`
before building.

### Install script (Linux / macOS / Windows)

Installs NyxClaw + Cloudflare Tunnel as system services. Handles `uv`, `cloudflared`, model
downloads, tunnel provisioning, and service registration (systemd / launchd / Windows service)
automatically.

```bash
# Linux / macOS
./install.sh

# Windows (PowerShell as Administrator)
.\install.ps1
```

## Backend Setup

NyxClaw supports two Claw backends. Set `AGENT_TYPE` in `.env` to switch.

### OpenClaw

Requires the nyxclaw avatar patch applied to OpenClaw.
See [claw_patches/openclaw/README.md](claw_patches/openclaw/README.md) for full setup
(patching, auth, AGENTS.md prompt).

```env
AGENT_TYPE=openclaw
BASE_URL=http://127.0.0.1:18789
AUTH_TOKEN=your-openclaw-gateway-token
USE_AVATAR_ENDPOINT=true
```

### ZeroClaw

Requires the nyxclaw avatar patch applied to ZeroClaw.
See [claw_patches/zeroclaw/README.md](claw_patches/zeroclaw/README.md) for full setup
(patching, auth, AGENTS.md prompt).

```env
AGENT_TYPE=zeroclaw
BASE_URL=http://127.0.0.1:42617
AUTH_TOKEN=zc_YOUR_TOKEN_HERE
USE_AVATAR_ENDPOINT=true
```

### Unpatched backends

Both backends work without the avatar patch — set `USE_AVATAR_ENDPOINT=false` (the default).
NyxClaw will use the standard `/v1/chat/completions` (OpenClaw) or `/ws/chat` (ZeroClaw)
endpoints. Rich content (`rich_content` messages) won't be available — all LLM output is
treated as speech.

## Supported claws

| Claw | Notes |
|---|---|
| [**OpenClaw**](https://github.com/openclaw/openclaw) | HTTP SSE backend with `/v1/chat/completions/avatar` |
| [**ZeroClaw**](https://github.com/zeroclaw-labs/zeroclaw) | WebSocket backend with `/ws/avatar` |
| **Your claw next?** | [Open an issue](https://github.com/myned-ai/nyxclaw/issues/new) or [email us](mailto:hello@myned.ai?subject=New%20claw%20backend%20request) |

## Configuration

All settings are configured via environment variables or `.env` file.
See [.env.example](.env.example) for the full template.

> **One session at a time.** NyxClaw serves a single active connection — one avatar, one
> audio stream. You can pair multiple devices (phone, tablet, desktop) for convenience, but
> only one connects at a time. **Treat the setup code like a password** — anyone with it
> can pair a device and talk to your AI agent.

## Claw patches

Backend-specific patches that add the avatar endpoint with structured `{speech, content}`
output and tool call events. When the LLM's response includes content better seen than
heard (URLs, tables, structured data), the patch splits the response:

- **`speech`** → avatar speaks a short phrase ("Here's the Wikipedia page, take a look.")
- **`content`** → forwarded as a `rich_content` message (markdown) to the client

| Patch | Backend | Endpoint | Docs |
|---|---|---|---|
| `claw_patches/openclaw/` | OpenClaw v2026.5.6 | `/v1/chat/completions/avatar` (HTTP SSE) | [README](claw_patches/openclaw/README.md) |
| `claw_patches/zeroclaw/` | ZeroClaw v0.7.4 | `/ws/avatar` (WebSocket) | [README](claw_patches/zeroclaw/README.md) |

Without the patch, all LLM output is treated as speech — no `rich_content` messages.

## Bring your own tunnel

NyxClaw auto-provisions a free Cloudflare Tunnel on first boot (`wss://<id>.nyxclaw.ai`).
This service has limited capacity. You can use any reverse proxy or tunneling solution
instead — NyxClaw just needs something that terminates TLS and forwards traffic to
`localhost:8080`:

- **Cloudflare Tunnel** (your own account) — run `cloudflared tunnel` with your own token
- **Tailscale** — encrypted mesh VPN, stable DNS, zero config
- **nginx / Caddy** — traditional reverse proxy with Let's Encrypt
- **ngrok** — quick dev tunnels

Set `AUTH_SETUP_CODE_URL=wss://your-domain/ws` in `.env` so the QR code contains your
custom URL.

## Resource requirements

| Component | Memory | Mode |
|---|---|---|
| Python + FastAPI + ONNX Runtime | ~500 MB | Both |
| Wav2Arkit (blendshape inference) | ~200 MB | Both |
| faster-whisper small.en (speech recognition) | ~500 MB | Local only |
| Piper TTS VITS (speech synthesis) | ~100 MB | Local only |
| Silero VAD (voice activity detection) | ~10 MB | Local only |

**OpenAI Voice:** 1 GB RAM, 1 core minimum. Recommended 1.5 GB, 2 cores.
**Local Voice:** 2 GB RAM, 2 cores minimum. Recommended 3–4 GB, 4 cores (STT, TTS, and
blendshapes run concurrently during barge-in).

## WebSocket protocol (`/ws`)

**Client → Server:**

| Type | Description |
|---|---|
| `audio_stream_start` | Start audio session |
| `audio` | Audio chunk (base64 PCM16 24 kHz mono) |
| `text` | Text message to AI |
| `interrupt` | Stop AI response |

**Server → Client:**

| Type | Description |
|---|---|
| `config` | Audio settings (sent on connect) |
| `audio_start` | AI response started |
| `sync_frame` | Audio + 52 ARKit blendshapes (30 FPS) |
| `audio_end` | AI response finished |
| `transcript_delta` | Streaming text fragment |
| `transcript_done` | Complete turn transcript |
| `rich_content` | Markdown content for the chat view |
| `avatar_state` | `"Listening"` or `"Responding"` |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with <a href="https://nyxclaw.ai"><span style="color:#FF4D6D">♥</span></a> by
  <a href="https://myned.ai">Myned AI</a> &nbsp;·&nbsp;
  <a href="https://nyxclaw.ai">nyxclaw.ai</a> &nbsp;·&nbsp;
  <a href="https://buymeacoffee.com/nyxclaw">Buy me a coffee</a>
</p>
