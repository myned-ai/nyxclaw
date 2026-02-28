# OpenClaw Integration Architecture

## Goal
Enable a low-latency, voice-enabled avatar using OpenClaw as the intelligence backend, while handling Speech-to-Text (STT) and Text-to-Speech (TTS) on the server side using optimized local ONNX models. NyxClaw typically coexists on the same machine as OpenClaw.

## Architecture Overview

The system follows a **Voice-to-Text -> LLM Stream -> Text-to-Voice** pipeline to minimize latency.

### 1. User Input (Speech-to-Text)
**Component:** NyxClaw Server
**Model:** faster-whisper (CTranslate2, int8) + Silero VAD (ONNX)
**Task:**
- Capture user microphone audio via WebSocket.
- Detect speech with Silero VAD, transcribe with faster-whisper.
- **Output:** Text string.

### 2. Intelligence (LLM Backend)
**Component:** OpenClaw Gateway
**Protocol:** HTTP POST `/v1/chat/completions` (OpenAI-compatible)
**Configuration:**
- Enable `gateway.http.endpoints.chatCompletions` in `openclaw.json`.
- Use `stream: true` in the request to receive text tokens instantly.

**Request Example:**
```json
POST /v1/chat/completions
Authorization: Bearer <AUTH_TOKEN>
Content-Type: application/json

{
  "model": "openclaw:main",
  "messages": [{"role": "user", "content": "captured_stt_text"}],
  "stream": true
}
```

**Output:** Server-Sent Events (SSE) stream of text tokens.

### 3. Response Generation (Text-to-Speech & Lip Sync)
**Component:** NyxClaw Server
**Model:** Piper VITS ONNX
**Task:**
- Buffer incoming text tokens from OpenClaw into sentences/phrases.
- Send buffered text to Piper TTS for audio synthesis.
- Run Wav2Arkit ONNX model for blendshape generation.
- **Output:** Synchronized audio + blendshapes streamed to client.

## Prerequisites — OpenClaw Gateway Configuration

> **Important:** The OpenClaw web chat UI works independently of the HTTP API.
> The `/v1/chat/completions` endpoint must be **explicitly enabled** in
> `openclaw.json` before NyxClaw can communicate with the gateway.

In your `openclaw.json` (inside the Docker container or on the host), add/verify:

```jsonc
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": {
          "enabled": true          // required for NyxClaw
        }
      }
    },
    "auth": {
      "token": "<your-token>"      // this value goes into AUTH_TOKEN in .env
    }
  }
}
```

**After editing**, restart the OpenClaw container so the config takes effect.

You can verify the endpoint is active with:

```bash
curl -s -X POST http://localhost:19001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"model":"openclaw:main","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

A `200` with a JSON response means it's working. A `405 Method Not Allowed` means
`chatCompletions` is still disabled.

## Implementation Steps

1.  **OpenClaw Setup:**
    - Verify `openclaw.json` has `gateway.http.endpoints.chatCompletions.enabled = true`.
    - Ensure an authentication token is configured (`gateway.auth.token`).

2.  **NyxClaw Configuration:**
    - Set `AGENT_TYPE=sample_openclaw` in `.env`.
    - Set `BASE_URL=http://127.0.0.1:19001` and `AUTH_TOKEN=<your-token>`.
    - Enable STT/TTS: `STT_ENABLED=true`, `TTS_ENABLED=true`.

3.  **Audio Pipeline:**
    - NyxClaw handles the full pipeline automatically:
      audio in -> VAD -> STT -> OpenClaw SSE -> sentence buffer -> TTS -> Wav2Arkit -> sync_frame out.
