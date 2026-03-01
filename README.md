# NyxClaw

**Voice-to-avatar server for Claw-based AI agents**

> **See it in action -> [Try Nyx](https://myned.ai)**

Real-time voice-to-avatar interaction server combining Claw agents (OpenClaw, ZeroClaw) with local STT/TTS and the Wav2Arkit model for synchronized avatar facial animation. Processes audio streams and generates ARKit blendshapes for realistic facial animations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

## Features

- **Local STT/TTS Pipeline**: faster-whisper (CTranslate2) + Silero VAD + Piper VITS ONNX — no cloud dependency
- **Facial Animation Sync**: Wav2Arkit model for ARKit-compatible blendshapes
- **Modular Agent System**: Pluggable Claw agents (OpenClaw HTTP SSE, ZeroClaw WebSocket)
- **WebSocket Communication**: Low-latency bidirectional streaming
- **CPU Acceleration**: ONNX-optimized inference for real-time performance without GPU
- **Production Ready**: Docker support, health checks, logging, authentication

## Architecture

```
Client ──WebSocket──► NyxClaw Server
                        │
                        ├── audio ──► STT (faster-whisper + Silero VAD) ──► text
                        │                                                    │
                        │                                          Claw Agent (LLM)
                        │                                                    │
                        ├── audio ◄── TTS (Piper VITS ONNX) ◄── text response
                        │
                        ├── Wav2Arkit (ONNX) ──► blendshapes
                        │
Client ◄──WebSocket──── sync_frame (audio + blendshapes @ 30 FPS)
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Docker Usage](#docker-usage)
- [Authentication](#authentication)
- [Agent Modularity](#agent-modularity)
- [Development](#development)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

### Local Development

- Python 3.10.x (exact version required)
- [uv](https://github.com/astral-sh/uv) package manager
- A running Claw agent backend (OpenClaw or ZeroClaw)
- ONNX Runtime (CPU-optimized, included in dependencies)

### Docker

- Docker 20.10+
- Docker Compose 2.0+
- A running Claw agent backend (OpenClaw or ZeroClaw)

## Quick Start

### Local Development

```bash
# 1. Install uv (if not already installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw

# 3. Install dependencies (with local voice stack)
uv sync --extra local_voice

# 4. Download the Wav2Arkit model
pip install -U "huggingface_hub[cli]"
mkdir -p pretrained_models/wav2arkit
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models/wav2arkit

# 5. Download faster-whisper model (CTranslate2 int8)
uv run --with 'faster-whisper>=1.1.0' python -c \
  "from faster_whisper.utils import download_model; download_model('small.en', output_dir='pretrained_models/faster_whisper_small_en')"

# 6. Download Piper TTS voice
mkdir -p pretrained_models/piper
uv run --with huggingface_hub python -c "
import shutil; from huggingface_hub import hf_hub_download
for s in ('', '.json'):
    shutil.copy2(
        hf_hub_download('rhasspy/piper-voices', f'en_US-hfc_female-medium.onnx{s}',
            subfolder='en/en_US/hfc_female/medium'),
        f'pretrained_models/piper/en_US-hfc_female-medium.onnx{s}')
"

# 7. Configure environment
cp .env.example .env
# Edit .env with your agent backend settings (BASE_URL, AUTH_TOKEN, etc.)

# 8. Run server
uv run python src/main.py
```

Server will start at `http://localhost:8080`

**Test the server:** Open `test.html` in your browser to test the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget) with your local server. Make sure `AUTH_ENABLED=false` in your `.env` file for testing.

### Docker

```bash
# 1. Clone and configure
git clone https://github.com/myned-ai/nyxclaw.git
cd nyxclaw
cp .env.example .env
# Edit .env with your settings

# 2. Download the ONNX model
pip install -U "huggingface_hub[cli]"
mkdir -p pretrained_models/wav2arkit
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models/wav2arkit

# Optional: enable local voice build (faster-whisper / Piper TTS / Silero VAD)
# Add in .env before build: INSTALL_LOCAL_VOICE=true

# 3. Build and run (production)
docker-compose up -d

# 4. View logs
docker-compose logs -f

# 5. Stop server
docker-compose down

# Development mode (with hot reload)
docker-compose --profile dev up
```

## Configuration

All settings can be configured via environment variables or `.env` file.

### Agent Backend Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_TYPE` | `openclaw` | Agent type: `openclaw`, `zeroclaw` |
| `BASE_URL` | `http://127.0.0.1:19001` | Agent backend URL |
| `AUTH_TOKEN` | *(none)* | Bearer token for agent authentication |
| `AGENT_MODEL` | `openclaw:main` | Model identifier for agent backend |
| `USER_ID` | *(none)* | Optional stable user ID for session continuity |
| `THINKING_MODE` | `minimal` | Thinking hint: `off`, `minimal`, `default` |
| `SESSION_KEY` | *(none)* | OpenClaw gateway routing override |
| `AGENT_ID` | *(none)* | OpenClaw agent ID override |
| `MAX_RETRIES` | `2` | Max retries for failed requests (OpenClaw) |

### STT Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_ENABLED` | `true` | Enable speech-to-text |
| `STT_MODEL` | `small.en` | faster-whisper model (`tiny.en`, `base.en`, `small.en`, `medium.en`) |
| `STT_VAD_START_THRESHOLD` | `0.60` | Silero VAD speech start threshold |
| `STT_VAD_END_THRESHOLD` | `0.35` | Silero VAD speech end threshold |
| `STT_VAD_MIN_SILENCE_MS` | `280` | Minimum silence before end-of-speech (ms) |
| `STT_INITIAL_PROMPT` | *(none)* | Whisper initial prompt for vocabulary priming |

### TTS Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENABLED` | `true` | Enable text-to-speech |
| `TTS_ONNX_MODEL_DIR` | `./pretrained_models/piper` | Piper ONNX model directory |
| `TTS_VOICE_NAME` | `en_US-hfc_female-medium` | Piper voice name |
| `TTS_VOICE_PATH` | *(none)* | Path to WAV file for voice cloning |
| `TTS_NOISE_SCALE` | `0.75` | Audio variation (0=flat, 1=expressive) |
| `TTS_NOISE_W_SCALE` | `0.8` | Phoneme duration variation |
| `TTS_LENGTH_SCALE` | `0.95` | Speech speed (<1=faster, >1=slower) |

### Other Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ASSISTANT_INSTRUCTIONS` | *see .env.example* | System prompt for the AI assistant |
| `ONNX_MODEL_PATH` | `./pretrained_models/wav2arkit/wav2arkit_cpu.onnx` | Wav2Arkit ONNX model path |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `DEBUG` | `false` | Enable debug logging |
| `AUTH_ENABLED` | `false` | Enable HMAC token authentication |
| `AUTH_SECRET_KEY` | *(auto-generated)* | HMAC signing key (`openssl rand -hex 32`) |

See [.env.example](.env.example) for the complete configuration template.

## API Documentation

### REST Endpoints

- `GET /inf` - Server info and status
- `GET /health` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)
- `GET /redoc` - ReDoc documentation
- `POST /api/auth/token` - Generate HMAC auth token (if authentication enabled)

### WebSocket Endpoint

Connect to `ws://localhost:8080/ws` (or `wss://` for production with TLS)

With authentication:
```
ws://localhost:8080/ws?token=YOUR_AUTH_TOKEN
```

#### Client -> Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `text` | `{"type": "text", "data": "Hello"}` | Send text message to AI |
| `audio_stream_start` | `{"type": "audio_stream_start", "userId": "user123"}` | Start audio streaming session |
| `audio` | `{"type": "audio", "data": "<base64>"}` | Audio chunk (PCM16, 24kHz mono, base64-encoded) |
| `interrupt` | `{"type": "interrupt"}` | Explicitly interrupt AI response |
| `ping` | `{"type": "ping"}` | Heartbeat to keep connection alive |

#### Server -> Client Messages

| Type | Description |
|------|-------------|
| `config` | Sent on connection. Contains negotiated audio settings |
| `audio_start` | AI started responding. Includes `sessionId`, `turnId`, `sampleRate`, `format` |
| `sync_frame` | Synchronized audio + blendshape frame at 30 FPS |
| `audio_chunk` | Fallback audio-only chunk when Wav2Arkit model is unavailable |
| `audio_end` | AI finished responding. Includes `sessionId`, `turnId` |
| `transcript_delta` | Streaming text fragment with timing offsets |
| `transcript_done` | Complete transcript for a turn (`role`: `"user"` or `"assistant"`) |
| `avatar_state` | Avatar state change (`"Listening"` or `"Responding"`) |
| `interrupt` | User interrupted AI response. Includes `turnId`, `offsetMs` |
| `error` | Error message |
| `pong` | Heartbeat response with `timestamp` |

## Docker Usage

### Multi-Stage Build

The Dockerfile uses a multi-stage build optimized for CPU-only production:

1. **Base Stage**: Python 3.10-slim (Debian-based) with system dependencies
2. **Dependencies Stage**: Fast dependency installation with uv
3. **Production Stage**: Minimal image with non-root user, health checks
4. **Development Stage**: Hot reload support for development

### Production Deployment

```bash
# Build image
docker build -t nyxclaw .

# Run (CPU-only)
docker run -d \
  --name nyxclaw \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/pretrained_models:/app/pretrained_models:ro \
  --restart unless-stopped \
  nyxclaw

# View logs
docker logs -f nyxclaw

# Health check
curl http://localhost:8080/health
```

### Docker Compose Profiles

- **Default (Production)**: `docker-compose up -d`
- **Development**: `docker-compose --profile dev up`

### Resource Requirements

NyxClaw runs multiple ONNX models concurrently on CPU. With the full local voice stack enabled (`INSTALL_LOCAL_VOICE=true`), expect the following memory footprint:

| Component | Memory |
|-----------|--------|
| faster-whisper small.en (CTranslate2 int8) | ~500 MB |
| Piper TTS VITS ONNX | ~100 MB |
| Wav2Arkit ONNX (blendshape inference) | ~200 MB |
| Silero VAD ONNX | ~10 MB |
| ONNX Runtime session overhead | ~200 MB |
| Python runtime + FastAPI + dependencies | ~300 MB |

|  | Minimum | Recommended |
|--|---------|-------------|
| **RAM** | 2 GB | 3-4 GB |
| **CPU** | 2 cores | 4 cores |

During an active conversation, CPU usage spikes as Wav2Arkit inference runs continuously on every audio chunk, Piper TTS synthesis is CPU-bound (sentence-by-sentence), and faster-whisper transcription bursts when the user finishes speaking. All three can overlap during barge-in scenarios.

For a single concurrent session, **2 vCPU + 3 GB RAM** is a comfortable target. Each additional concurrent WebSocket session adds roughly 200-400 MB (separate STT/VAD state) plus CPU contention on the ONNX inference.

## Authentication

The server uses **HMAC-SHA256 signed tokens** for authentication. Tokens are origin-bound and time-limited.

### Enabling Authentication

Set in `.env`:
```bash
AUTH_ENABLED=true
AUTH_SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
AUTH_ALLOWED_ORIGINS=https://yourwebsite.com,https://www.yourwebsite.com
```

### Generate Secret Key

```bash
openssl rand -hex 32
```

### Getting a Token

```bash
curl -X POST http://localhost:8080/api/auth/token \
  -H "Origin: https://yourwebsite.com"
```

### Using the Token

```
ws://localhost:8080/ws?token=YOUR_AUTH_TOKEN
```

## Backend Modularity

The server uses a modular backend system for different Claw backends:

### Available Backends

- **openclaw**: OpenClaw HTTP gateway with SSE streaming (`/v1/chat/completions`)
- **zeroclaw**: ZeroClaw WebSocket gateway (`/ws/chat`)

Both backends use the same local STT/TTS pipeline (faster-whisper + Piper VITS ONNX).

### Quick `.env` switch

```dotenv
# OpenClaw
AGENT_TYPE=openclaw
BASE_URL=http://127.0.0.1:19001
AUTH_TOKEN=your-openclaw-token
AGENT_MODEL=openclaw:main

# ZeroClaw
AGENT_TYPE=zeroclaw
BASE_URL=http://127.0.0.1:5555
AUTH_TOKEN=your-zeroclaw-token
AGENT_MODEL=zeroclaw:main
```

### Custom Backends

Implement the `BaseAgent` interface:

```python
from backend import BaseAgent, ConversationState

class MyCustomAgent(BaseAgent):
    @property
    def is_connected(self) -> bool: ...

    @property
    def state(self) -> ConversationState: ...

    @property
    def transcript_speed(self) -> float: ...

    def set_event_handlers(self, **kwargs) -> None: ...

    async def connect(self) -> None: ...

    def send_text_message(self, text: str) -> None: ...

    def append_audio(self, audio_bytes: bytes) -> None: ...

    async def disconnect(self) -> None: ...
```

### Switching Backends

1. Set `AGENT_TYPE` in `.env`
2. Update `BASE_URL` and `AUTH_TOKEN` for your backend
3. Restart the server

## Development

### Install Dev Dependencies

```bash
uv sync --group dev
```

### Code Quality

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run ty check src/

# All checks
uv run ruff check src/ --fix && uv run ruff format src/ && uv run ty check src/
```

### Running Tests

```bash
uv run pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Apple ARKit](https://developer.apple.com/augmented-reality/arkit/) for the blendshape specification standard
- [LAM Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) for facial animation model
- [Wav2Vec 2.0](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) for speech representation learning
- [Piper TTS](https://github.com/rhasspy/piper) for VITS ONNX text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for CTranslate2 speech-to-text
