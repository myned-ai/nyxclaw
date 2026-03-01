# NyxClaw - Technical Specification

## 1. Overview
**NyxClaw** is a real-time, bidirectional interaction backend designed to power 3D avatar interfaces. It acts as a bridge between a client (web/mobile) and Claw-based AI agents (OpenClaw, ZeroClaw), adding a critical layer of **audio-to-facial-animation** synthesis with local STT/TTS.

Key capability: transforming raw audio streams into synchronized ARKit blendshapes in real-time on CPU.

## 2. System Architecture

### 2.1 Technology Stack
- **Runtime**: Python 3.10+
- **Framework**: FastAPI (ASGI)
- **Protocol**: WebSockets (Real-time communication), HTTP (Health/Auth)
- **Inference Engine**: ONNX Runtime (CPU-optimized)
- **STT**: faster-whisper (CTranslate2) + Silero VAD (ONNX)
- **TTS**: Piper VITS ONNX
- **Package Manager**: uv

### 2.2 Core Modules
- **`main.py`**: Application entry point, middleware configuration (CORS, Auth), and lifecycle management.
- **`routers/chat_router.py`**: Handles WebSocket connections (`/ws`), manages session state, and orchestrates data flow between the client, the AI agent, and the inference service.
- **`backend/`**: Abstract layer (`BaseAgent`) for swappable Claw backends (`openclaw`, `zeroclaw`).
- **`wav2arkit/`**: Inference module (`Wav2ArkitInference`) that runs the `wav2arkit_cpu.onnx` model to generate 52 facial blendshapes from audio segments.
- **`services/stt_service.py`**: Silero VAD (ONNX) + faster-whisper transcription.
- **`services/tts_service.py`**: Piper VITS ONNX text-to-speech.

## 3. Interfaces & Protocols

### 3.1 HTTP Endpoints
| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root info (name, version, status). |
| `GET` | `/health` | Health check. Returns 503 if services are unhealthy. |
| `POST` | `/api/auth/token` | Generates a token for WebSocket connection. Requires valid `Origin`. |
| `GET` | `/docs` | OpenAPI / Swagger UI. |

### 3.2 WebSocket Protocol (`/ws`)
**Format**: JSON-based control messages and Base64-encoded binary data.
**Audio Spec**: PCM 16-bit, 24kHz, Mono.

#### Downstream (Server -> Client)
- **`sync_frame`**: The core event. Contains `audio` (base64) AND `weights` (52 blendshapes) for a single frame (30fps).
- **`audio_start` / `audio_end`**: Delimiters for AI response turns.
- **`transcript_delta` / `transcript_done`**: Real-time text streaming.
- **`interrupt`**: Signals that the user interrupted the bot; client must cut audio.

#### Upstream (Client -> Server)
- **`audio_stream_start`**: Usage initiator.
- **`audio`**: Raw audio chunks (base64 wrapped JSON).
- **`text`**: Text-only input.
- **`interrupt`**: Client-triggered interruption.

## 4. Feature Specification

### 4.1 Real-Time Intelligence
- **Pluggable Agents**: Switch between OpenClaw (HTTP SSE) and ZeroClaw (WebSocket) via `AGENT_TYPE` configuration.
- **Low Latency**: Streaming architecture ensures minimal delay between user speech and avatar response.

### 4.2 Facial Animation Synthesis
- **Wav2Arkit Model**: Custom ONNX model converts audio directly to ARKit blendshape coefficients.
- **CPU Optimization**: Optimized for efficient inference on standard CPUs (AVX-512 support) without needing GPUs.
- **Synchronization**: Server guarantees 1:1 mapping between audio chunks and visual frames.

### 4.3 Security & Production
- **Authentication**: Optional HMAC-based auth flow (`AUTH_ENABLED=true`).
- **Rate Limiting**: Protects against abuse (requests per minute).
- **CORS**: Configurable allowed origins.
- **Dockerized**: Multi-stage Docker build for small, secure production images.

## 5. Configuration

Configured via `.env` or Environment Variables.

**Key Variables:**
- `AGENT_TYPE`: Selector (`openclaw`, `zeroclaw`).
- `BASE_URL` / `AUTH_TOKEN`: Agent backend connection.
- `ONNX_MODEL_PATH`: Path to the Wav2Arkit model file.
- `STT_ENABLED` / `TTS_ENABLED`: Toggle local voice pipeline.
- `AUTH_ENABLED`: Toggle security features.
- `DEBUG`: Toggle verbose logging.

## 6. Development & Deployment
- **Local Dev**: Uses `uv` for fast dependency management.
- **Docker**: Simple `docker-compose up` workflow for production.
- **Quality**: Enforced via `ruff` (lint/format) and `ty` (type checking).
