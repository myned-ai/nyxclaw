# ZeroClaw Avatar Channel Patch

Patches for **ZeroClaw v0.7.4** that add a dedicated avatar WebSocket channel (`/ws/avatar`) for nyxclaw voice + avatar integration.

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

The patch ships as a directory of post-patch files under [`files/`](./files/) — `patch.sh` overlays them onto a v0.7.4 checkout via a plain copy.

| Layer | Crate / file | Change |
|-------|--------------|--------|
| API | `zeroclaw-api/src/provider.rs` | Add `response_format: Option<&serde_json::Value>` to `ChatRequest`; new `Provider::supports_response_format()` capability method (default `false`) |
| Runtime | `zeroclaw-runtime/src/agent/agent.rs` | Add `response_format` field + builder + `set_response_format`/`set_prompt_builder` setters; thread into `Agent::turn` and `Agent::turn_streamed`. **Tests**: 4 setter / prompt-builder tests with a `RequestCaptureProvider` mock |
| Runtime | `zeroclaw-runtime/src/agent/loop_.rs` | `response_format: None` defaults in `run_tool_call_loop`'s `ChatRequest` sites (orchestrator/delegate paths) |
| Runtime | `zeroclaw-runtime/src/agent/prompt.rs` | New `SystemPromptBuilder::without_section(name)` for stripping `DateTimeSection` (cache-stable system prompt) |
| Providers | `zeroclaw-providers/src/openai.rs` | Native SSE streaming impl: `stream_chat()`, tool-call delta accumulation by `index` (mandatory; missing-index deltas skipped, id rebinding is first-write-wins), `parse_openai_sse_lines` helper with **1 MiB per-line cap**, advertises `supports_response_format() = true` |
| Providers | `zeroclaw-providers/src/anthropic.rs` | `response_format: None` default (Anthropic doesn't honor the field — the avatar handler now skips `set_response_format` here via the capability gate) |
| Providers | `zeroclaw-providers/src/reliable.rs` | Pass-through of `response_format` + conservative AND-policy `supports_response_format()` (only true when every fallback candidate honors the field) |
| Providers | `zeroclaw-providers/src/openrouter.rs`, `router.rs` | `None` defaults; `Router` uses the same conservative AND-policy as `Reliable` |
| Gateway | `zeroclaw-gateway/src/lib.rs` | Register `/ws/avatar` route; add `pub mod nyxclaw` |
| Gateway | `zeroclaw-gateway/src/nyxclaw.rs` | **NEW** — Avatar WebSocket channel. **Hardened post-review**: `TurnOutcome` state machine (Completed / Cancelled / Disconnected / Failed) replaces error-string classification; `session_queue.acquire` serializes per-`session_id` to prevent cross-connection corruption; `session_id` charset validation (`[A-Za-z0-9_-]{1,128}`); WS frame caps (64 KiB / 256 KiB), per-message user-content cap (32 KiB), `accumulated_raw` cap (1 MiB) with cancel-on-overflow; `AvatarJsonExtractor` field caps (64 KiB); `Provider::supports_response_format()` capability gate (Anthropic etc. fall back to plain-prose narration without lying about the contract); barge-in race fix (drop `biased;` + post-loop `now_or_never` receiver poll); pure `classify_turn_event` helper for testable WS-frame contracts; `chunk_reset` + `consolidate_turn` for parity with `/ws/chat` |
| Binary | `src/providers/traits.rs`, `tests/live/openai_codex_vision_e2e.rs` | `response_format: None` defaults in 8 test sites |

All 15 modified files live under [`files/`](./files/), mirroring the v0.7.4 source-tree layout.

**Stats**: 15 files (13 source + Dockerfile + docker-compose.yml), ~3000 lines added, ~1900 lines of which are `nyxclaw.rs` (orchestration + 24 unit tests).

## Hardening (post-review)

After the initial port, a 5-agent code review surfaced 11 distinct
correctness / security / reliability issues. All have been fixed
on the same branch:

| Category | Issue | Fix |
|---|---|---|
| Correctness | Sentence extractor split prematurely on chunk-end terminators ("I see 1." + "5 inches" emitted "I see 1." as a sentence) | Required lookahead whitespace; end-of-stream flushing handled by post-loop `sentence_buf.trim()` |
| Correctness | Streaming-fallback path narrated raw JSON when parse failed (avatar would speak `{"speech":"hi","content":""}` literally) | Strict envelope check; if absent and `schema_enforced`, send `INVALID_RESPONSE_FORMAT` error frame instead |
| Correctness | WS-closed-mid-turn classified as agent error; session ended in `"error"` state, sender wrote to closed socket | New `TurnOutcome::Disconnected` arm; same persistence as `Cancelled`, no `done` frame, session ends `idle` |
| Correctness | `extractor.finalize()` skipped on cancel path (in-progress speech lost) | Finalize unconditionally before classification |
| Correctness | `biased;` + barge-in arriving in same poll cycle as turn completion = queued message lost | Dropped `biased;`; added one-shot `now_or_never` receiver poll after loop |
| Concurrency | Two clients reusing `?session_id=X` raced on `cancel_tokens` map and partial-persist `update_last` | `session_queue.acquire(&session_key)` per turn (matches `ws.rs`) |
| Security | `session_id` flowed into format!() without validation (path traversal / log injection / oversize) | Charset+length gate at upgrade and connect-frame override |
| Security | Anthropic + `response_format` was silent failure (LLM returned prose, every turn errored) | New `Provider::supports_response_format()` capability gate; non-supporting providers fall back to plain-prose narration |
| Security | Unbounded WS frames, extractor buffers, `accumulated_raw`, SSE lines = OOM vectors | WS frame cap (`max_frame_size`/`max_message_size`), per-message user-content cap, extractor field cap (64 KiB), `accumulated_raw` cap (1 MiB) with cancel-on-overflow, SSE line cap (1 MiB) |
| Security | Tool-call delta missing `index` defaulted to 0, splicing args into wrong slot; id rebinding silently overwrote | Missing index → skip with debug; id rebinding → first-write-wins with warn |
| Security | Agent-init error sent raw to client (could leak provider URLs / key fragments) | Pipe through `sanitize_api_error` |
| Parity | `/ws/avatar` skipped memory consolidation that `/ws/chat` runs after each turn | Same `consolidate_turn` fire-and-forget pattern in `handle_completed` |
| Parity | Missing `chunk_reset` frame and error-code taxonomy (`AUTH_ERROR`/`PROVIDER_ERROR`/`AGENT_ERROR`) | Both added; matches `ws.rs:620` and `ws.rs:649` |
| Test coverage | `Agent::set_response_format` / `set_prompt_builder` setters had no direct tests | 4 new tests with a `RequestCaptureProvider` mock pin the contract |
| Test coverage | TurnEvent → WS-frame mapping (especially ToolCall's two-frame ordering) untested | Refactored `handle_turn_event` to delegate to a pure `classify_turn_event`; 6 new tests pin every variant's frame shape |
| Docker | Upstream `Dockerfile` uses `COPY --parents` with the bare `1.7` syntax pragma — modern BuildKit refuses the flag | Bumped pragma to `1.7-labs` |
| Docker | Workspace lists `tools/fill-translations` and `xtask` as members but the Dockerfile copies neither — `cargo build --locked` fails parsing the workspace | Added explicit `COPY` for both manifests + stub bin sources |
| Docker | Second-pass source restore copies only `src/`, `benches/`, root `*.rs` — the `crates/*/src/lib.rs` empty stubs from pass 1 stay in place, producing 217 unresolved-import errors when the root crate links | Added `COPY crates/ crates/` after the cleanup |
| Docker | Cache-invalidation `rm` covers only `zeroclawlabs-*` (the root crate); pass-1 stub artifacts for the 14 workspace members shadow the rebuild | Expanded glob to `zeroclaw* aardvark* robot-kit*` |
| Docker | Upstream `docker-compose.yml` defaults to `image: ghcr.io/zeroclaw-labs/zeroclaw:latest` (the unpatched upstream build) | Replaced with `build: { context: ., target: dev }`, added env passthroughs and the `./playground:/zeroclaw-data/workspace` bind mount |

See `git log 78fb0a6..HEAD` in the patched repo for per-fix commits
with detailed root-cause notes.

## Apply

The patch script overlays every file under [`files/`](./files/) onto a
ZeroClaw v0.7.4 checkout. No git operations on the target — `cp` / `rsync`
under the hood — so it works equally on a `git clone` or an extracted tarball.

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

1. Sanity-check the target is a ZeroClaw v0.7.4 source tree.
2. If the target is a git checkout, refuse to overlay if it has uncommitted
   changes (so a future revert via `git checkout -- .` is clean).
3. List every file about to be overlaid.
4. Copy `files/` over the target.
5. Print next-step build/configure commands.

### Revert

If the target is a git checkout — one command resets everything to upstream:

```bash
git -C /path/to/zeroclaw-v0.7.4 checkout -- .
```

Otherwise, re-extract the upstream source.

### After patching — local cargo workflow

```bash
cd /path/to/zeroclaw-v0.7.4
cargo build --workspace
cargo test -p zeroclaw-gateway --lib nyxclaw   # 24 unit tests for the avatar channel
cargo run --bin zeroclawlabs -- gateway
```

### After patching — Docker workflow (recommended)

The patch ships a working `docker-compose.yml` and Dockerfile (the upstream
v0.7.4 versions are broken in four distinct places — the patch fixes all of
them; see commit `c60968c` for the full root-cause writeup).

```bash
cd /path/to/zeroclaw-v0.7.4

# 1. Set provider creds (.env file at repo root)
cat > .env <<EOF
PROVIDER=openai
ZEROCLAW_MODEL=gpt-4.1-mini
API_KEY=sk-...your-key-here...
OPENAI_API_KEY=sk-...your-key-here...
EOF

# 2. Make sure ./playground/ exists with your AGENTS.md, IDENTITY.md, etc.
#    (the bind mount lands here at /zeroclaw-data/workspace inside the container)
ls playground/AGENTS.md  # should exist

# 3. Build + start
docker compose up -d --build
```

Build takes ~5–10 min cold (Rust workspace compile inside the container) and
~10 s warm. The image is ~196 MB.

### Configure providers + pairing inside the container

The upstream v0.7.4 default `config.toml` ships with two settings that need
to be flipped before the avatar will work end-to-end:

```bash
# 1. Enable bearer-token authentication (default ships disabled — would let
#    any client hit the gateway with no auth at all).
docker exec zeroclaw sed -i 's/require_pairing = false/require_pairing = true/' \
    /zeroclaw-data/.zeroclaw/config.toml

# 2. Point the agent at your real provider (default is `ollama`, which means
#    the gateway tries to reach localhost:11434 inside the container and times
#    out on every turn).
docker exec zeroclaw zeroclaw config set providers.fallback openai

# 3. Restart so the new config is loaded
docker restart zeroclaw
```

### Get the bearer token

In v0.7.4, `gateway get-paircode --new` returns a **one-time 6-digit pairing
code**, not a long-lived bearer token. Exchange it via `POST /pair`:

```bash
# Generate a fresh pairing code (e.g. "325758")
CODE=$(docker exec zeroclaw zeroclaw gateway get-paircode --new 2>&1 \
       | grep -oE '[0-9]{6}' | head -1)
echo "code: $CODE"

# Exchange for a long-lived token
TOKEN=$(curl -s -X POST \
    -H "X-Pairing-Code: $CODE" \
    -H "Content-Type: application/json" \
    -d '{"device_name":"nyxclaw"}' \
    http://localhost:42617/pair \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')
echo "token: $TOKEN"
```

Save the token in nyxclaw's `.env`:

```env
AGENT_TYPE=zeroclaw
BASE_URL=http://host.docker.internal:42617
AUTH_TOKEN=zc_...the-token-from-above...
USE_AVATAR_ENDPOINT=true
```

> **Important**: `docker compose restart` does NOT re-read `.env`. After
> changing `AUTH_TOKEN`, recreate the container:
>
> ```bash
> docker compose up -d --force-recreate server
> ```

### Smoke test

```bash
# From the host
curl -i \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "http://localhost:42617/ws/avatar?token=$TOKEN" \
  --max-time 3 2>&1 | head -5
# Expect: HTTP/1.1 101 Switching Protocols + a session_start frame

# Without token, expect 401:
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  http://localhost:42617/ws/avatar
# Expect: 401
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

- **ZeroClaw v0.7.4** — tested and supported (workspace builds clean, 6403 tests pass)
- **Newer versions** — may require manual rebase. The patch is git-managed; resolve conflicts with the usual git tooling.
- **Older versions (< v0.7.4)** — incompatible. v0.6.x and earlier use a different crate layout. For v0.5.0 see [`legacy_v0.5.0/`](./legacy_v0.5.0/).

## Files

```
claw_patches/zeroclaw/
├── README.md                          # This file
├── patch.sh                           # Overlay on Linux/macOS
├── patch.ps1                          # Overlay on Windows
└── files/                             # Post-patch copies of every modified file
    ├── Dockerfile                     # Patched (1.7-labs syntax, full crate copy, etc.)
    ├── docker-compose.yml             # Local-build target=dev, playground bind mount
    ├── crates/
    │   ├── zeroclaw-api/src/provider.rs
    │   ├── zeroclaw-providers/src/{openai,anthropic,reliable,openrouter,router}.rs
    │   ├── zeroclaw-runtime/src/agent/{agent,loop_,prompt}.rs
    │   └── zeroclaw-gateway/src/{lib,nyxclaw}.rs
    ├── src/providers/traits.rs
    └── tests/live/openai_codex_vision_e2e.rs
```
