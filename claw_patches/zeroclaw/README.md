# ZeroClaw Avatar Channel Patch

Patches for **ZeroClaw v0.7.4** that add a dedicated avatar WebSocket channel (`/ws/avatar`) for nyxclaw voice + avatar integration.

> Looking for the v0.5.0 version? See [`legacy_v0.5.0/`](./legacy_v0.5.0/).

## What this patch does

When connected via `/ws/avatar`, ZeroClaw forces the LLM to respond with structured JSON:

```json
{"speech": "Here's what I found, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome"}
```

- `speech` → streamed to nyxclaw as `speech_chunk` events (sentence-split mid-stream) → avatar speaks it
- `content` → sent to nyxclaw as `rich_content` event → app renders cards/links/tables
- Tool calls/results stream live during agent execution
- Tool-call fillers — avatar speaks contextual phrases (e.g. "I'm searching the web") while tools execute
- Cancel + barge-in support (mid-turn `{"type":"cancel"}` or new `{"type":"message"}` aborts in-flight response)

The existing `/ws/chat` endpoint is unchanged — CLI and web dashboard clients work exactly as before.

## What this patch contains

The patch ships as a single `git apply`-able file: [`zeroclaw-v0.7.4-nyxclaw.patch`](./zeroclaw-v0.7.4-nyxclaw.patch).

| Layer | Crate / file | Change |
|-------|--------------|--------|
| API | `zeroclaw-api/src/provider.rs` | Add `response_format: Option<&serde_json::Value>` to `ChatRequest` |
| Runtime | `zeroclaw-runtime/src/agent/agent.rs` | Add `response_format` field + builder + `set_response_format`/`set_prompt_builder` setters; thread into `Agent::turn` and `Agent::turn_streamed` |
| Runtime | `zeroclaw-runtime/src/agent/loop_.rs` | `response_format: None` defaults in `run_tool_call_loop`'s `ChatRequest` sites (orchestrator/delegate paths) |
| Runtime | `zeroclaw-runtime/src/agent/prompt.rs` | New `SystemPromptBuilder::without_section(name)` for stripping `DateTimeSection` (cache-stable system prompt) |
| Providers | `zeroclaw-providers/src/openai.rs` | Native SSE streaming impl: `stream_chat()`, tool-call delta accumulation by `index`, `parse_openai_sse_lines` helper, full streaming-with-structured-output support |
| Providers | `zeroclaw-providers/src/anthropic.rs` | `response_format: None` default (Anthropic uses forced tool-calls; not wired here) |
| Providers | `zeroclaw-providers/src/reliable.rs` | Pass-through of `response_format` in 2 reconstruction sites; `None` defaults in 7 test sites |
| Providers | `zeroclaw-providers/src/openrouter.rs`, `router.rs` | `None` defaults in 3 test sites |
| Gateway | `zeroclaw-gateway/src/lib.rs` | Register `/ws/avatar` route; add `pub mod nyxclaw` |
| Gateway | `zeroclaw-gateway/src/nyxclaw.rs` | **NEW** — Avatar WebSocket channel: incremental JSON extractor, sentence-split speech_chunk emission, tool-call fillers, barge-in with `cancel_tokens` registry, `scope_session_key` task-local, partial-content persistence on streaming chunks |
| Binary | `src/providers/traits.rs`, `tests/live/openai_codex_vision_e2e.rs` | `response_format: None` defaults in 8 test sites |

Inspectable copies of every modified file live under [`files/`](./files/), mirroring the v0.7.4 crate layout.

**Stats**: 13 files, +1671 / −1 lines, 1041 lines of which are the new `nyxclaw.rs` module.

## Apply

The patch uses `git apply`, so the target must be a git checkout of ZeroClaw at (or descended from) the v0.7.4 release tag.

### Bash (Linux/macOS)

```bash
git clone https://github.com/zeroclaw-labs/zeroclaw -b v0.7.4 ~/zeroclaw-v0.7.4
./patch.sh ~/zeroclaw-v0.7.4
```

### PowerShell (Windows)

```powershell
git clone https://github.com/zeroclaw-labs/zeroclaw -b v0.7.4 C:\zeroclaw-v0.7.4
.\patch.ps1 -ZeroClawDir C:\zeroclaw-v0.7.4
```

Both scripts:

1. Verify the target is a v0.7.4 git checkout
2. Dry-run the patch (`git apply --check`) — bail out before any tree mutation if it doesn't apply cleanly
3. Apply the patch
4. Print next-step build/test commands

### Revert

```bash
git -C /path/to/zeroclaw-v0.7.4 apply -R zeroclaw-v0.7.4-nyxclaw.patch
```

### After patching

```bash
cd /path/to/zeroclaw-v0.7.4
cargo build --workspace
cargo test -p zeroclaw-gateway --lib nyxclaw   # 10 unit tests for the avatar channel
cargo run --bin zeroclawlabs -- gateway
```

nyxclaw then connects to `ws://<host>:<port>/ws/avatar` instead of `/ws/chat`.

## Authentication

ZeroClaw uses bearer tokens for WebSocket auth. Tokens are accepted via (precedence order):

1. `Authorization: Bearer <token>` header
2. `Sec-WebSocket-Protocol: bearer.<token>` subprotocol
3. `?token=<token>` query parameter

Generate a pairing token inside the ZeroClaw container:

```bash
docker exec <zeroclaw-container> zeroclawlabs gateway get-paircode --new
```

Configure nyxclaw `.env`:

```env
AGENT_TYPE=zeroclaw
BASE_URL=http://<zeroclaw-host>:<port>
AUTH_TOKEN=zc_YOUR_TOKEN_HERE
USE_AVATAR_ENDPOINT=true
```

## Required AGENTS.md addition

You **must** manually add the following **Response format** section to your `playground/AGENTS.md`:

````markdown
## Response format

Your responses are consumed by a voice + avatar system. Every response you generate is a JSON object with two fields:

```json
{"speech": "...", "content": "..."}
```

### `speech` — what the avatar says aloud
- Keep it concise and conversational — this is spoken, not read.
- Never include URLs, table data, code, or markdown syntax in speech.
- When you have rich content to show, use a brief phrase: "Check this out", "Here's what I found", "Take a look."
- For simple conversational responses (greetings, opinions, short answers), put the full response in speech.

### `content` — what appears in the chat (rich content)
- Put URLs, links, tables, code snippets, structured data, and detailed information here.
- Use markdown formatting — the app renders it.
- Set to empty string `""` when there's nothing visual to show — including error messages, apologies, explanations, and status updates. Only use `content` for URLs, tables, code, or structured data.

### Examples

Simple greeting:
```json
{"speech": "Hey, what's up?", "content": ""}
```

User asks for a link:
```json
{"speech": "Here's the Wikipedia page for Rome, take a look.", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome\n\nRome is the capital city of Italy."}
```

User asks to compare things:
```json
{"speech": "Here's the comparison, check it out.", "content": "| Feature | iPhone 15 | Samsung S24 |\n|---------|-----------|-------------|\n| Screen | 6.1\" | 6.2\" |\n| Battery | 3349mAh | 4000mAh |"}
```

### Never do this
- Never put URLs in speech
- Never narrate table data in speech
- Never leave speech empty — always say something
- Never put raw JSON or code in speech
- Never put error messages or apologies in content — those belong in speech only
````

## Performance tuning

For real-time voice, latency matters. These ZeroClaw settings reduce time-to-first-token (TTFT). Edit your `config.toml` (in `~/.zeroclaw/config.toml`, or `/zeroclaw-data/.zeroclaw/config.toml` inside Docker):

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

| Setting | Default | Recommended | Effect |
|---------|---------|-------------|--------|
| `default_temperature` | `0.7` | `0.5` | Less sampling overhead, faster token selection |
| `provider_timeout_secs` | `120` | `30` | Fail fast instead of hanging |
| `compact_context` | `false` | `true` | Reduces system prompt + context payload |
| `max_tool_iterations` | `10` | `4` | Limits tool round-trips per turn |
| `max_history_messages` | `50` | `15` | Less history → fewer input tokens → faster TTFT |
| `max_context_tokens` | `32000` | `12000` | Triggers context compaction sooner |
| `parallel_tools` | `false` | `true` | Concurrent tool execution |
| `reasoning_enabled` | `false` | `false` | Keep disabled — adds seconds of thinking delay |

### Model selection

Model choice is the biggest single latency factor:

| Model | Provider | TTFT | Notes |
|-------|----------|------|-------|
| `gpt-4.1-mini` | `openai` | ~1s | Good speed/quality balance |
| `gpt-4.1-nano` | `openai` | ~0.5s | Fastest OpenAI option |
| `claude-haiku-4-5` | `anthropic` | ~0.8s | Fast, good quality |
| `llama-3.3-70b-versatile` | `groq` | ~0.3s | Groq LPU hardware |

```toml
default_provider = "openai"
default_model = "gpt-4.1-mini"
```

### Automatic optimizations (no config needed)

- **Native SSE streaming** — the patched `OpenAiProvider::stream_chat` emits `TurnEvent::Chunk` deltas as soon as bytes arrive. The avatar channel feeds those into a sentence splitter so speech starts on the first complete sentence (`~300–800 ms` after first byte).
- **Prompt caching** — the avatar channel calls `set_prompt_builder(SystemPromptBuilder::with_defaults().without_section("datetime"))`, removing the per-second timestamp section that otherwise busts OpenAI's automatic prompt cache. Cache hit rates jump from ~0% to ~95% after the first call.
- **HTTP warmup** — ZeroClaw pre-warms provider connection pools at startup.
- **Partial-content persistence** — assistant responses are saved every 500 ms via `update_last`, so a process crash mid-turn doesn't lose the partial reply.

## WebSocket Protocol: `/ws/avatar`

### Client → Server

```json
{"type": "connect", "session_id": "...", "device_name": "...", "capabilities": ["avatar"]}
{"type": "message", "content": "What is the Wikipedia page for Rome?"}
{"type": "cancel"}
```

A new `{"type":"message"}` arriving mid-turn cancels the in-flight turn AND queues itself as the next user message — so the user can interrupt and restart in one step.

### Server → Client

```json
{"type": "session_start", "session_id": "abc123", "resumed": false, "message_count": 0}
{"type": "connected", "message": "Avatar connection established"}
{"type": "speech_chunk", "content": "I'm searching the web.", "filler": true}
{"type": "tool_call", "id": "...", "name": "web_search", "args": {"query": "..."}}
{"type": "tool_result", "id": "...", "name": "web_search", "output": "..."}
{"type": "speech_chunk", "content": "Here's what I found."}
{"type": "rich_content", "content": "**Rome - Wikipedia**\nhttps://en.wikipedia.org/wiki/Rome\n\n..."}
{"type": "done", "full_response": "Here's what I found."}
{"type": "done", "full_response": "", "cancelled": true}
{"type": "thinking", "content": "..."}
{"type": "error", "message": "..."}
```

`speech_chunk` fields:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `string` | Text for the avatar to speak |
| `filler` | `bool` (optional) | `true` if this is a contextual filler emitted during tool execution. nyxclaw uses this to apply throttling (2s gap between fillers, 5s same-content cooldown). |

### Differences from `/ws/chat`

| Feature | `/ws/chat` | `/ws/avatar` |
|---------|-----------|--------------|
| Response format | Raw text | Structured `{speech, content}` JSON |
| Streaming events | `chunk`, `tool_call`, `tool_result`, `done` | `speech_chunk` (sentence-split), `rich_content`, `tool_call`, `tool_result`, `done`, plus `filler:true` chunks |
| Cancel handling | Via `/api/abort` HTTP | Via `{"type":"cancel"}` over WS, plus mid-turn message barge-in |
| LLM constraint | None | `response_format: json_schema` enforced |
| System prompt | Default (with datetime) | DateTimeSection stripped for cache stability |
| Concurrency | Serialized via `session_queue` | Inline cancel/queue (UX-driven barge-in) |

## Provider support

| Provider | Structured output | Streaming | Status |
|----------|-------------------|-----------|--------|
| `openai` | `response_format: json_schema` | Native SSE | **Patched** |
| `azure_openai` | Same API as OpenAI | — | Not patched (manual edit needed; see [Patching additional providers](#patching-additional-providers)) |
| `openrouter` | Passes through to model | — | Stub field only (no streaming impl) |
| `compatible` | OpenAI-compatible | — | Not patched |
| `anthropic` | Forced tool-calling (different mechanism) | — | Field stub only |
| `gemini` | `response_mime_type` + `response_schema` | — | Not supported |
| `ollama` | `format: "json"` | — | Not supported |
| Others | — | — | Fallback: speech-only, no rich content |

When using an unpatched provider, the avatar channel falls back gracefully — the response is sent as `speech_chunk` (avatar speaks it) but no `rich_content` cards are generated.

### Patching additional providers

For OpenAI-compatible providers, add the `response_format` field to the request struct and wire it through. The OpenAI implementation in `crates/zeroclaw-providers/src/openai.rs` is the reference template — search for `response_format` to find every site that needs touching (request struct, request builder, retry path, streaming path).

## Compatibility

- **ZeroClaw v0.7.4** — tested and supported (workspace builds clean, 6300+ tests pass)
- **Newer versions** — may require manual rebase. The patch is git-managed; resolve conflicts with the usual git tooling.
- **Older versions (< v0.7.4)** — incompatible. v0.6.x and earlier use a different crate layout. For v0.5.0 see [`legacy_v0.5.0/`](./legacy_v0.5.0/).

## Files

```
claw_patches/zeroclaw/
├── README.md                          # This file
├── patch.sh                           # Apply on Linux/macOS
├── patch.ps1                          # Apply on Windows
├── zeroclaw-v0.7.4-nyxclaw.patch      # The patch (git apply -able)
├── upgrade_to_zeroclaw_0.7.4.md       # Original migration plan
├── files/                             # Inspectable copies of modified files
│   ├── crates/
│   │   ├── zeroclaw-api/src/provider.rs
│   │   ├── zeroclaw-providers/src/{openai,anthropic,reliable,openrouter,router}.rs
│   │   ├── zeroclaw-runtime/src/agent/{agent,loop_,prompt}.rs
│   │   └── zeroclaw-gateway/src/{lib,nyxclaw}.rs
│   ├── src/providers/traits.rs
│   └── tests/live/openai_codex_vision_e2e.rs
└── legacy_v0.5.0/                     # Archived v0.5.0 patch + README + scripts
```
