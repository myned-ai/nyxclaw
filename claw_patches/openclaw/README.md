# OpenClaw Avatar SSE Patch

Patches for **OpenClaw v2026.5.6** that add a dedicated avatar SSE endpoint (`/v1/chat/completions/avatar`) for nyxclaw voice + avatar integration.

## What this patch does

Adds a new HTTP SSE endpoint that forces the LLM to respond with structured JSON:

```json
{"speech": "Here's what I found, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome"}
```

- `speech` — streamed to nyxclaw as `event: speech_chunk` SSE events → avatar speaks it
- `content` — streamed as `event: rich_content` SSE event → app renders cards/links/tables
- Tool calls/results stream as `event: tool_call` / `event: tool_result` during agent execution
- The existing `/v1/chat/completions` endpoint is unchanged

## What this patch contains

The patch ships as a directory of post-patch files under [`files/`](./files/) — `patch.sh` overlays them onto a v2026.5.6 checkout via a plain copy.

| File | Change |
|------|--------|
| `src/gateway/avatar-http.ts` | **NEW** — Avatar SSE handler. Subscribes to agent events, accumulates the streamed `{speech, content}` JSON, sentence-splits the `speech` field for low-latency `event: speech_chunk` emission, and emits `event: rich_content` / `event: tool_call` / `event: tool_result` / `event: done`. Injects an `extraSystemPrompt` to force the JSON envelope. |
| `src/gateway/server-http.ts` | Adds 4 small blocks to the upstream v2026.5.6 file: a `getAvatarHttpModule()` lazy-load factory + module promise (matches the upstream pattern for every other gateway sub-handler), an `isAvatarHttpPath()` strict-equality guard for `/v1/chat/completions/avatar`, and a `requestStages.push` block that wires the avatar handler into the request pipeline before the openai handler. Avatar reuses the same auth, config, and rate limiter as `/v1/chat/completions`. |

**Stats**: 2 files (1 new + 1 modified), ~660 lines added (avatar-http.ts) + ~25 lines added to server-http.ts.

## Apply

The patch script overlays every file under [`files/`](./files/) onto an
OpenClaw v2026.5.6 checkout. No git operations on the target — `cp` / `rsync`
under the hood — so it works equally on a `git clone` or an extracted tarball.

### Bash (Linux/macOS)

```bash
git clone https://github.com/openclaw/openclaw -b v2026.5.6 ~/openclaw-v2026.5.6
./patch.sh ~/openclaw-v2026.5.6
```

### PowerShell (Windows)

```powershell
git clone https://github.com/openclaw/openclaw -b v2026.5.6 C:\openclaw-v2026.5.6
.\patch.ps1 -OpenClawDir C:\openclaw-v2026.5.6
```

Both scripts:

1. Sanity-check the target is an OpenClaw v2026.5.6 source tree.
2. If the target is a git checkout, refuse to overlay if it has uncommitted
   changes (so a future revert via `git checkout -- .` is clean).
3. List every file about to be overlaid.
4. Copy `files/` over the target.
5. Print next-step build/run commands.

### Revert

If the target is a git checkout — one command resets everything to upstream:

```bash
git -C /path/to/openclaw-v2026.5.6 checkout -- .
```

Otherwise, re-extract the upstream source.

### After patching

```bash
cd /path/to/openclaw-v2026.5.6
npm install
npm run build
npm start -- gateway --bind lan --port 18789
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

## Rich Content Thumbnails

nyxclaw automatically enriches `rich_content` URLs with link card metadata (title, description, thumbnail) by scraping OpenGraph tags from each page. This works for most sites, but pages behind Cloudflare or bot protection will block the scrape and fall back to a favicon.

### Current limitation

OpenClaw's agent event system does not include tool output in `tool_result` SSE events. This means nyxclaw cannot receive thumbnail hints from tools via the current patch. Rich content thumbnails rely entirely on OGP scraping + favicon fallback.

### How to enable thumbnail hints (requires OpenClaw source change)

If you need provider-quality thumbnails (e.g., from Brave Search), you need to:

1. **Extend OpenClaw's agent event emitter** to include `output` in tool events with `phase: "end"`:
   ```typescript
   // In OpenClaw's agent runner, when emitting tool completion:
   emit({ stream: "tool", data: { phase: "end", name, toolCallId, isError, output: toolOutput } });
   ```

2. **Update the avatar patch** (`avatar-http.ts`) to forward the output:
   ```typescript
   } else if (data.phase === "end") {
     writeAvatarEvent(res, "tool_result", {
       name: data.name,
       tool_call_id: data.toolCallId,
       success: !data.isError,
       duration_ms: durationMs,
       output: (data as any).output ?? "",  // Forward tool output
     });
   }
   ```

3. **Append thumbnail hints** in your search tool's output (same format as ZeroClaw — see ZeroClaw README for details):
   ```
   ---THUMBNAIL_HINTS---
   https://example.com/article	https://cdn.example.com/thumb.jpg
   ```

nyxclaw already handles `tool_result` events with `output` from both backends — once OpenClaw sends the output, thumbnails will work automatically.

### Without this change

Everything works — nyxclaw fetches OGP metadata for each URL and falls back to favicons for blocked sites. No thumbnails from the search provider are available, but the cards still render with titles and descriptions from OGP.

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
3. **`server-http.ts`** — adds a `getAvatarHttpModule()` lazy-load factory and a `requestStages.push` block that registers `/v1/chat/completions/avatar` ahead of `/v1/chat/completions` in the gateway pipeline

The `extraSystemPrompt` approach works because:
- OpenClaw's agent pipeline already supports `extraSystemPrompt` all the way through
- Claude and GPT-4 reliably produce JSON when instructed in the system prompt (especially with structured examples)
- No modifications needed to the agent runner, LLM provider, or session manager

## Compatibility

- **OpenClaw v2026.5.6** — tested and supported. The overlay matches v2026.5.6's lazy-load module pattern (every gateway sub-handler — `openai-http`, `models-http`, `embeddings-http`, etc. — is lazy-loaded via a `get*Module()` factory; the avatar overlay follows the same convention).
- **v2026.3.13–v2026.4.x** — incompatible. v2026.3.x used static imports for gateway sub-handlers; the new lazy-load factory pattern was introduced between v2026.3.13 and v2026.5.6. If you need a v2026.3.13 patch, check git history for the previous overlay version.
- **Newer versions** — may require manual rebase. The patch is git-managed; resolve conflicts with the usual git tooling.

### Provider support

Since we use `extraSystemPrompt` (not `response_format`), this works with **any LLM provider** that OpenClaw supports — Claude, GPT-4, Gemini, etc. The LLM just needs to follow JSON instructions in the system prompt.

## Files

```
claw_patches/openclaw/
├── README.md                          # This file
├── patch.sh                           # Overlay on Linux/macOS
├── patch.ps1                          # Overlay on Windows
└── files/                             # Post-patch copies of every modified file
    └── src/gateway/
        ├── avatar-http.ts             # NEW — avatar SSE handler
        └── server-http.ts             # Patched — adds 4 wiring blocks for the avatar route
```
