# OpenClaw Avatar Integration Architecture

## Goal
Enable a low-latency, voice-enabled avatar (running on the Nyx framework) using OpenClaw as the intelligence backend, while handling Speech-to-Text (STT) and Text-to-Speech (TTS) on the server side using optimized local/WASM models.The server might coexists on the same machine as OpenClawd.

## Architecture Overview

The system follows a **Voice-to-Text -> LLM Stream -> Text-to-Voice** pipeline to minimize latency.

### 1. User Input (Speech-to-Text)
**Component:** Client Widget / Avatar Server
**Model:** `kyutai/stt-1b-en_fr-candle` (Hugging Face)
**Task:**
- Capture user microphone audio.
- Transcribe audio to text in real-time or near real-time using the Candle/WASM implementation.
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
Authorization: Bearer <OPENCLAW_TOKEN>
Content-Type: application/json

{
  "model": "openclaw:main",
  "messages": [{"role": "user", "content": "captured_stt_text"}],
  "stream": true
}
```

**Output:** Server-Sent Events (SSE) stream of text tokens.

### 3. Response Generation (Text-to-Speech & Lip Sync)
**Component:** Client Widget / Avatar Server
**Model:** `kyutai/pocket-tts-candle` (GitHub Rust subdirectory / WASM)
**Task:**
- Buffer incoming text tokens from OpenClaw into sentences/phrases.
- Send buffered text to the local `pocket-tts-candle` model.
- Generate raw audio buffers and viseme data (for lip sync).
- **Output:** Audio playback and avatar animation.

## Prerequisites — OpenClaw Gateway Configuration

> **Important:** The OpenClaw web chat UI works independently of the HTTP API.
> The `/v1/chat/completions` endpoint must be **explicitly enabled** in
> `openclaw.json` before the avatar server can communicate with the gateway.

In your `openclaw.json` (inside the Docker container or on the host), add/verify:

```jsonc
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": {
          "enabled": true          // ← required for the avatar server
        }
      }
    },
    "auth": {
      "token": "<your-token>"      // ← this value goes into OPENCLAW_API_TOKEN in .env
    }
  }
}
```

**After editing**, restart the openclawd container so the config takes effect.

You can verify the endpoint is active with:

```bash
curl -s -X POST http://localhost:18789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"model":"openclaw:main","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

A `200` with a JSON response means it's working. A `405 Method Not Allowed` means
`chatCompletions` is still disabled.

## Implementation Steps for Code Agent

1.  **OpenClaw Setup:**
    - Verify `openclaw.json` has `gateway.http.endpoints.chatCompletions.enabled = true`.
    - Ensure an authentication token is configured (`gateway.auth.token`).

2.  **Client/Server Integration:**
    - Integrate the `stt-1b-en_fr-candle` model to transcribe microphone input.
    - Implement an HTTP client to POST the transcription to OpenClaw's `/v1/chat/completions` endpoint.
    - Implement an SSE reader to parse the streaming response (`data: {...}`).

3.  **Audio Pipeline:**
    - Create a text buffer that accumulates streaming tokens until a punctuation mark (sentence boundary) is reached.
    - Feed the complete sentence to `pocket-tts-candle` to generate audio.
    - Queue the generated audio chunks for seamless playback to the user.
