# ZeroClaw Avatar Channel Patch

Patches for ZeroClaw **v0.5.0** that add a dedicated avatar WebSocket channel (`/ws/avatar`) for nyxclaw voice+avatar integration.

## What this patch does

When connected via `/ws/avatar`, ZeroClaw forces the LLM to respond with structured JSON:

```json
{"speech": "Here's what I found, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome"}
```

- `speech` — sent to nyxclaw as `speech_chunk` events → avatar speaks it
- `content` — sent to nyxclaw as `rich_content` event → app renders cards/links/tables
- Tool calls/results stream in real-time during agent execution
- Cancel support for barge-in interruption

The existing `/ws/chat` endpoint is unchanged — CLI and web dashboard clients work as before.

## Usage

### Bash (Linux/macOS)
```bash
./patch.sh /path/to/zeroclaw-v0.5.0
```

### PowerShell (Windows)
```powershell
.\patch.ps1 -ZeroClawDir C:\path\to\zeroclaw-v0.5.0
```

Both scripts:
- Back up original files to `.nyxclaw-patch-backup/`
- Copy patched/new files into place
- Verify all files were applied

### After patching
```bash
cd /path/to/zeroclaw-v0.5.0
cargo build
cargo test
cargo run -- gateway
```

nyxclaw connects to `ws://<host>:<port>/ws/avatar` instead of `/ws/chat`.

## Authentication

ZeroClaw uses bearer tokens for WebSocket auth. nyxclaw sends the token as a `?token=` query parameter on the WebSocket connection.

### Getting the token

Generate a pairing token inside the ZeroClaw container:

```bash
docker exec <zeroclaw-container> zeroclaw gateway get-paircode --new
```

This prints a token like `zc_abc123def456...`. Copy it.

### Configuring nyxclaw

Add the token to nyxclaw's `.env`:

```env
AGENT_TYPE=zeroclaw
BASE_URL=http://<zeroclaw-host>:<port>
AUTH_TOKEN=zc_YOUR_TOKEN_HERE
USE_AVATAR_ENDPOINT=true
```

nyxclaw will connect to `ws://<host>:<port>/ws/avatar?token=<AUTH_TOKEN>`.

## AGENTS.md — Required prompt addition

You must manually add the following **Response format** section to your `playground/AGENTS.md`:

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

## Performance Tuning

For real-time voice, latency matters. These ZeroClaw settings reduce time-to-first-token (TTFT) significantly. Edit your `config.toml` (located at `~/.zeroclaw/config.toml`, or `/zeroclaw-data/.zeroclaw/config.toml` inside Docker):

```toml
default_temperature = 0.5
provider_timeout_secs = 30

[agent]
compact_context = true
max_tool_iterations = 4
max_history_messages = 15
max_context_tokens = 12000
parallel_tools = true

[runtime]
reasoning_enabled = false
```

### What each setting does

| Setting | Default | Recommended | Effect |
|---------|---------|-------------|--------|
| `default_temperature` | `0.7` | `0.5` | Less sampling overhead, slightly faster token selection |
| `provider_timeout_secs` | `120` | `30` | Fail fast instead of hanging on slow requests |
| `compact_context` | `false` | `true` | Reduces system prompt and context payload sent to the LLM |
| `max_tool_iterations` | `10` | `4` | Limits tool-call round-trips per turn (each is a full LLM call) |
| `max_history_messages` | `50` | `15` | Less conversation history = fewer input tokens = faster TTFT |
| `max_context_tokens` | `32000` | `12000` | Triggers context compaction sooner, keeping payload lean |
| `parallel_tools` | `false` | `true` | Runs multiple tool calls concurrently instead of sequentially |
| `reasoning_enabled` | `false` | `false` | Keep disabled — reasoning adds seconds of thinking delay |

### Model selection

Model choice is the single biggest latency factor. Fast options:

| Model | Provider | TTFT | Notes |
|-------|----------|------|-------|
| `gpt-4.1-mini` | `openai` | ~1s | Good balance of speed and quality |
| `gpt-4.1-nano` | `openai` | ~0.5s | Fastest, lower quality |
| `claude-haiku-4-5` | `anthropic` | ~0.8s | Fast, good quality |
| `llama-3.3-70b-versatile` | `groq` | ~0.3s | Groq LPU hardware, very fast |

Set in `config.toml`:
```toml
default_provider = "openai"
default_model = "gpt-4.1-mini"
```

### Automatic optimizations (no config needed)

- **Streaming** — the avatar channel uses `turn_with_streaming()` which streams tokens as they arrive. Speech starts as soon as the first sentence is complete.
- **Prompt caching** — OpenAI automatically caches prompt prefixes (system prompt + tools). This patch removes the `DateTimeSection` (which changed every second and invalidated the cache). With a stable system prompt, cache hit rates of 95-99% are typical after the first call, reducing TTFT by up to 50%.
- **HTTP warmup** — ZeroClaw pre-warms provider HTTP connection pools (TLS + HTTP/2) at startup, eliminating cold-start latency on the first request.
- **Response caching** — set `response_cache_enabled = true` in `[memory]` and `temperature = 0.0` to cache responses for repeated queries. Useful for common greetings or FAQ-type questions.

### Identity prompt size

Your `playground/AGENTS.md`, `IDENTITY.md`, and `SOUL.md` files are injected into the system prompt. Larger files = more input tokens = higher TTFT. Keep them concise for voice use cases.

These values are optimized for snappy voice responses. Depending on your use case you may want different tradeoffs — for example, raising `max_history_messages` for longer conversations, enabling `reasoning_enabled` for complex tasks, or increasing `max_tool_iterations` if your agent uses many tools. Experiment and find what works best for your setup.

## WebSocket Protocol: `/ws/avatar`

### Client → Server

```json
{"type": "connect", "session_id": "...", "device_name": "...", "capabilities": ["avatar"]}
{"type": "message", "content": "What is the Wikipedia page for Rome?"}
{"type": "cancel"}
```

### Server → Client

```json
{"type": "session_start", "session_id": "abc123", "resumed": false, "message_count": 0}
{"type": "connected", "message": "Connection established"}
{"type": "tool_call", "name": "web_fetch", "args": {"url": "..."}}
{"type": "tool_result", "name": "web_fetch", "output": "...", "success": true, "duration_ms": 1200}
{"type": "speech_chunk", "content": "Here's the Wikipedia page for Rome, take a look."}
{"type": "rich_content", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome\n\n..."}
{"type": "done", "full_response": "Here's the Wikipedia page for Rome, take a look."}
{"type": "error", "message": "..."}
```

### Key differences from `/ws/chat`

| Feature | `/ws/chat` | `/ws/avatar` |
|---------|-----------|-------------|
| Response format | Raw text | Structured `{speech, content}` JSON |
| Streaming events | `done` only | `tool_call`, `tool_result`, `speech_chunk`, `rich_content`, `done` |
| Cancel support | No | Yes (`{"type": "cancel"}`) |
| LLM constraint | `response_format` not set | `response_format: json_schema` enforced |

## How it works

### Files modified

**Full file replacements (patched copies in `src/`):**

| File | What changed |
|------|-------------|
| `src/providers/traits.rs` | Added `response_format` to `ChatRequest`, `StreamEvent` enum, `stream_chat()` trait method. |
| `src/providers/openai.rs` | Added `response_format` + `stream` to `NativeChatRequest`, SSE streaming structs, `stream_chat()` impl. |
| `src/agent/agent.rs` | Added `response_format` field + setter, `turn_with_events()`, `turn_with_streaming()` (streaming agent turn). |
| `src/agent/prompt.rs` | Removed `DateTimeSection` from system prompt builder. The per-second timestamp was invalidating OpenAI's prompt cache on every request. User messages already carry timestamps, so the LLM still knows the current time. |
| `src/channels/nyxclaw.rs` | **NEW** — Avatar WebSocket channel with incremental JSON extraction: streams `speech_chunk` events as the LLM generates, not after. |

**Line injections (original files preserved):**

| File | What injected |
|------|--------------|
| `src/agent/loop_.rs` | `response_format: None,` in `ChatRequest` construction (1 location) |
| `src/providers/anthropic.rs` | `response_format: None,` in `ProviderChatRequest` construction (1 location) |
| `src/providers/reliable.rs` | `response_format: None,` in `ChatRequest` constructions (6 locations) + `stream_chat()` delegation |
| `src/providers/mod.rs` | `StreamEvent` added to re-export list |
| `Cargo.toml` | `async-stream = "0.3"` dependency added |
| `src/channels/mod.rs` | `pub mod nyxclaw;` after `pub mod notion;` |
| `src/gateway/mod.rs` | `use crate::channels::nyxclaw;` import + `/ws/avatar` route + print line |

## Compatibility

- **ZeroClaw v0.5.0** — tested and supported
- **Other versions** — may require manual adjustments to `gateway/mod.rs` and `channels/mod.rs`

### Provider support

| Provider | Structured output | Status |
|----------|------------------|--------|
| `openai` | `response_format: json_schema` | **Patched** |
| `azure_openai` | Same API as OpenAI | Not patched (see guide below) |
| `openrouter` | Passes through to model | Not patched (see guide below) |
| `compatible` | OpenAI-compatible API | Not patched (see guide below) |
| `anthropic` | Requires forced tool calling (different mechanism) | Not supported yet |
| `gemini` | `response_mime_type` + `response_schema` | Not supported yet |
| `ollama` | `format: "json"` (partial) | Not supported yet |
| Others | — | Fallback: speech-only, no rich content |

When using an unpatched provider, the avatar channel falls back gracefully — the LLM response is sent as `speech_chunk` (the avatar speaks it) but no `rich_content` cards are generated.

### Patching additional providers

If your provider uses the OpenAI-compatible API (`azure_openai`, `openrouter`, `compatible`), you can patch it yourself. The pattern is the same for all OpenAI-compatible providers:

**Step 1:** Find the `NativeChatRequest` struct in `src/providers/<your_provider>.rs`:

```rust
struct NativeChatRequest {
    model: String,
    messages: Vec<...>,
    temperature: f64,
    tools: Option<...>,
    tool_choice: Option<String>,
}
```

**Step 2:** Add the `response_format` field:

```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    // ── nyxclaw patch: structured output ──
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}
```

**Step 3:** Find the `chat()` method where `NativeChatRequest` is constructed. Look for `tool_choice: tools.as_ref().map(|_| "auto"...` and add the `response_format` field after it:

```rust
    tool_choice: tools.as_ref().map(|_| "auto".to_string()),
    // ── nyxclaw patch: pass through response_format ──
    response_format: request.response_format.cloned(),
    tools,
```

**Step 4:** Find `chat_with_tools()` (if it exists) and add `response_format: None,` to its `NativeChatRequest` construction.

**Step 5:** Build and test:
```bash
cargo build
cargo test -p zeroclaw --lib providers::<your_provider>
```

### Anthropic / Gemini — different approach needed

Anthropic and Gemini use different APIs for structured output:

- **Anthropic**: Requires `tool_choice: {"type": "tool", "name": "respond"}` with a forced tool whose schema includes `speech` and `content` fields. This is architecturally different from `response_format`.
- **Gemini**: Uses `generationConfig.response_mime_type: "application/json"` + `generationConfig.response_schema: {...}`.

These require provider-specific patches beyond the OpenAI pattern. Contributions welcome.

## Reverting

To revert all patches:
```bash
cp -r /path/to/zeroclaw/.nyxclaw-patch-backup/* /path/to/zeroclaw/
rm -rf /path/to/zeroclaw/.nyxclaw-patch-backup
rm /path/to/zeroclaw/src/channels/nyxclaw.rs
```
