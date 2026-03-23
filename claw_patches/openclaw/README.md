# OpenClaw Avatar SSE Patch

Patches for OpenClaw **v2026.3.13** that add a dedicated avatar SSE endpoint (`/v1/chat/completions/avatar`) for nyxclaw voice+avatar integration.

## What this patch does

Adds a new HTTP SSE endpoint that forces the LLM to respond with structured JSON:

```json
{"speech": "Here's what I found, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome"}
```

- `speech` — streamed to nyxclaw as `event: speech_chunk` SSE events → avatar speaks it
- `content` — streamed as `event: rich_content` SSE event → app renders cards/links/tables
- Tool calls/results stream as `event: tool_call` / `event: tool_result` during agent execution
- The existing `/v1/chat/completions` endpoint is unchanged

## Usage

### Bash (Linux/macOS)
```bash
./patch.sh /path/to/openclaw-v2026.3.13
```

### PowerShell (Windows)
```powershell
.\patch.ps1 -OpenClawDir C:\path\to\openclaw-v2026.3.13
```

Both scripts:
- Back up original files to `.nyxclaw-patch-backup/`
- Copy patched/new files into place
- Inject route registration into `server-http.ts`
- Verify all files were applied

### After patching
```bash
cd /path/to/openclaw-v2026.3.13
npm run build      # or: pnpm build / bun build
node dist/index.js gateway --bind lan --port 18789
```

nyxclaw connects to `http://<host>:<port>/v1/chat/completions/avatar` instead of `/v1/chat/completions`.

## Authentication

OpenClaw uses a static gateway token for HTTP auth. nyxclaw sends it as an `Authorization: Bearer <token>` header on every request.

### Setting the token

Set a gateway token in OpenClaw's `.env` (or `docker-compose.yml`):

```env
OPENCLAW_GATEWAY_TOKEN=your_secret_token_here
```

If you're using Docker Compose, this is typically set in the `.env` file next to `docker-compose.yml`. OpenClaw reads it as the `OPENCLAW_GATEWAY_TOKEN` environment variable.

### Configuring nyxclaw

Add the same token to nyxclaw's `.env`:

```env
AGENT_TYPE=openclaw
BASE_URL=http://<openclaw-host>:18789
AUTH_TOKEN=your_secret_token_here
USE_AVATAR_ENDPOINT=true
```

nyxclaw will send `Authorization: Bearer your_secret_token_here` on all requests to `/v1/chat/completions/avatar`.

## AGENTS.md — Required prompt addition

You must manually add the following **Response format** section to your workspace `AGENTS.md` (located at `~/.openclaw/workspace/AGENTS.md`, or wherever `OPENCLAW_WORKSPACE_DIR` points):

```markdown
## Response format

Your responses are consumed by a voice + avatar system. Every response you generate is a JSON object with two fields:

\`\`\`json
{"speech": "...", "content": "..."}
\`\`\`

### `speech` — what the avatar says aloud
- Keep it concise and conversational — this is spoken, not read.
- Never include URLs, table data, code, or markdown syntax in speech.
- When you have rich content to show, use a brief phrase: "Check this out", "Here's what I found", "Take a look."
- For simple conversational responses (greetings, opinions, short answers), just put the full response in speech.

### `content` — what appears in the chat (rich content)
- Put URLs, links, tables, code snippets, structured data, and detailed information here.
- Use markdown formatting — the app renders it.
- Set to empty string `""` when there's nothing visual to show — including error messages, apologies, explanations, and status updates. Only use `content` for URLs, tables, code, or structured data.
- If you browsed a URL the user asked for, put the URL here.
- If you compared items, put a markdown table here.
- If you found search results, put the links here.

### Examples

Simple greeting:
\`\`\`json
{"speech": "Hey, what's up?", "content": ""}
\`\`\`

User asks for a link:
\`\`\`json
{"speech": "Here's the Wikipedia page for Rome, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome\n\nRome is the capital city of Italy."}
\`\`\`

User asks to compare things:
\`\`\`json
{"speech": "Here's the comparison, check it out.", "content": "| Feature | iPhone 15 | Samsung S24 |\n|---------|-----------|-------------|\n| Screen | 6.1\" | 6.2\" |\n| Battery | 3349mAh | 4000mAh |"}
\`\`\`

### Never do this
- Never put URLs in speech
- Never narrate table data in speech
- Never leave speech empty — always say something
- Never put raw JSON or code in speech
- Never put error messages or apologies in content — those belong in speech only
```

**Note:** The avatar endpoint also injects these instructions via `extraSystemPrompt`, so the LLM receives them even without `AGENTS.md`. However, adding them to `AGENTS.md` reinforces the format and improves reliability.

## Performance Tuning

For real-time voice, latency matters. These OpenClaw settings reduce time-to-first-token (TTFT) significantly. Add them to your `openclaw.json` (located at your `OPENCLAW_CONFIG_DIR`, e.g. `~/.openclaw/openclaw.json` on the host, mounted as `/home/node/.openclaw/openclaw.json` in Docker):

```json
{
  "agents": {
    "defaults": {
      "thinkingDefault": "off",
      "humanDelay": { "mode": "off" },
      "blockStreamingDefault": "off",
      "timeoutSeconds": 30,
      "models": {
        "openai/gpt-4.1": {
          "params": {
            "temperature": 0.4,
            "maxTokens": 400
          }
        }
      }
    }
  }
}
```

Replace `openai/gpt-4.1` with your actual model. Merge these into your existing `openclaw.json` — don't overwrite the `gateway` section.

### What each setting does

| Setting | Value | Effect |
|---------|-------|--------|
| `thinkingDefault` | `"off"` | Disables chain-of-thought reasoning — biggest TTFT win |
| `humanDelay` | `"off"` | Removes artificial 800–2500ms typing delay between responses |
| `blockStreamingDefault` | `"off"` | Streams raw tokens instead of buffering into blocks |
| `timeoutSeconds` | `30` | Fails fast instead of hanging on slow requests |
| `temperature` | `0.4` | Lower randomness = faster token selection |
| `maxTokens` | `400` | Caps response length — voice responses should be short |

### Benchmark results (gpt-4.1)

| | TTFS (time to first speech) | Total |
|---|---|---|
| **Default config** | 2.6s | 3.1s |
| **Optimized (warm)** | 1.1s | 1.4s |

### Model-specific notes

- **`fastMode: true`** — only works with reasoning models (o3, o4-mini). Sends `reasoning.effort: "low"`. Do **not** use with gpt-4.1 or claude — will cause 400 errors.
- **Anthropic models** — add `"cacheRetention": "short"` to enable prompt caching (5min TTL, reduces input processing time).

These values are optimized for snappy voice responses. Depending on your use case you may want different tradeoffs — for example, raising `maxTokens` if your agent gives detailed answers, enabling `thinkingDefault: "minimal"` if response quality matters more than speed, or increasing `temperature` for more creative output. Experiment and find what works best for your setup.

## SSE Protocol Reference

### Request (POST)

Same as `/v1/chat/completions`:
```json
{
  "model": "openclaw:main",
  "stream": true,
  "messages": [{"role": "user", "content": "Show me the Wikipedia page for Rome"}]
}
```

Auth: `Authorization: Bearer <gateway-token>` (same as existing endpoint).

### Response (SSE stream)

Standard SSE format with custom event types:

```
event: tool_call
data: {"name": "web_fetch", "args": {"url": "https://en.wikipedia.org/wiki/Rome"}}

event: tool_result
data: {"name": "web_fetch", "success": true, "duration_ms": 1200}

event: speech_chunk
data: {"content": "Here's the Wikipedia page for Rome, take a look."}

event: rich_content
data: {"content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome\n\n..."}

event: done
data: {"full_response": "Here's the Wikipedia page for Rome, take a look."}

data: [DONE]
```

### Key differences from `/v1/chat/completions`

| Feature | `/v1/chat/completions` | `/v1/chat/completions/avatar` |
|---------|----------------------|------------------------------|
| Response format | Raw text chunks | Structured `{speech, content}` JSON |
| SSE event types | `data:` only (OpenAI format) | `event: speech_chunk`, `event: rich_content`, `event: tool_call`, `event: tool_result`, `event: done` |
| System prompt | Unchanged | `extraSystemPrompt` injected with JSON format instructions |
| Tool visibility | Hidden | Streamed as events |

## How it works

OpenClaw uses an external PI agent runtime (`@mariozechner/pi-coding-agent`) that doesn't expose `response_format`. Instead:

1. **`extraSystemPrompt`** — injects the JSON response format instructions into the agent's system prompt
2. **`avatar-http.ts`** — new SSE handler that subscribes to agent events, accumulates the JSON response, incrementally extracts the `speech` field, and emits custom SSE event types
3. **`server-http.ts`** — one injection to register the `/v1/chat/completions/avatar` route

The `extraSystemPrompt` approach works because:
- OpenClaw's agent pipeline already supports `extraSystemPrompt` all the way through
- Claude and GPT-4 reliably produce JSON when instructed in the system prompt (especially with structured examples)
- No modifications needed to the agent runner, LLM provider, or session manager

### Files modified

**New files:**

| File | What it does |
|------|-------------|
| `src/gateway/avatar-http.ts` | Avatar SSE endpoint handler. Adds `extraSystemPrompt` for JSON format, incrementally extracts `speech` → `speech_chunk` events, emits `rich_content` on completion. |

**Line injections (original files preserved):**

| File | What injected |
|------|--------------|
| `server-http.ts` | Import + route registration for `/v1/chat/completions/avatar` in the request pipeline |

## Compatibility

- **OpenClaw v2026.3.13** — tested and supported
- **Other versions** — `server-http.ts` request pipeline structure may differ; manual adjustment may be needed

### Provider support

Since we use `extraSystemPrompt` (not `response_format`), this works with **any LLM provider** that OpenClaw supports — Claude, GPT-4, Gemini, etc. The LLM just needs to follow JSON instructions in the system prompt.

## Reverting

To revert all patches:
```bash
cp -r /path/to/openclaw/.nyxclaw-patch-backup/* /path/to/openclaw/
rm -rf /path/to/openclaw/.nyxclaw-patch-backup
rm /path/to/openclaw/src/gateway/avatar-http.ts
```
