# Upgrade ZeroClaw Patches: v0.5.0 → v0.7.4

Status: **Plan ready for implementation** | Created: 2026-04-23 | Updated: 2026-04-23 (deep-read pass)

## Context

Our v0.5.0 patches add a `/ws/avatar` channel to ZeroClaw for nyxclaw voice+avatar integration. ZeroClaw v0.7.4 has significant changes from v0.5.0. This document supersedes both the v0.6.9 plan and the earlier v0.7.4 sketch — all open questions have been resolved by reading source.

## Headline structural changes (v0.5.0 → v0.7.4)

1. **Multi-crate workspace.** `src/<x>` → `crates/zeroclaw-<crate>/src/<x>`. Every patch path moves.
2. **Edition 2024, Rust MSRV 1.87.** Patches must compile against newer edition.
3. **Provider trait moved.** `crate::providers::traits::*` → `zeroclaw_api::provider::*` (re-exported via `zeroclaw_providers`).
4. **`StreamEvent` enum gained 2 variants** (`PreExecutedToolCall`, `PreExecutedToolResult`) — used only by Claude Code proxy. Pattern matches need to handle them or `_`.
5. **`Provider::stream_chat()` gained `StreamOptions` parameter** (2-field `Copy` struct: `enabled`, `count_tokens`).

## What v0.7.4 gives us free

| Feature | v0.7.4 status |
|---|---|
| Native `StreamEvent` enum + `stream_chat()` trait method | ✅ |
| `Agent::turn_streamed(msg, event_tx, Option<CancellationToken>)` | ✅ — cancel param already upstream |
| `TurnEvent` enum: `Chunk { delta }`, `Thinking { delta }`, `ToolCall { id, name, args }`, `ToolResult { id, name, output }` | ✅ |
| Anthropic SSE streaming (`supports_streaming()` + `supports_streaming_tool_events()`) | ✅ |
| `is_tool_loop_cancelled()` helper | ✅ — same path |
| `sanitize_api_error()` | ✅ |
| `AppState.cancel_tokens: Arc<Mutex<HashMap<String, CancellationToken>>>` | ✅ — per-session cancel registry |
| `AppState.event_tx`, `session_backend`, `pairing`, `model`, `provider`, `mem` | ✅ — all fields we used in v0.5.0 still exist |
| `ChatMessage::user()` / `assistant()` / `system()` / `tool()` constructors | ✅ |
| 3-tier WS auth pattern (header → subprotocol → query) | ✅ — `ws.rs::extract_ws_token` is our template |

## What we still must patch

| Feature | Reason |
|---|---|
| `response_format` field on `ChatRequest` (`zeroclaw-api`) | Forces structured `{speech, content}` JSON |
| `response_format` field + setter on `Agent` | Stores schema, threads into ChatRequest |
| `response_format` field on OpenAI `NativeChatRequest` | Serializes to OpenAI API |
| OpenAI `stream_chat()` impl + `supports_streaming()=true` | Currently non-streaming |
| `SystemPromptBuilder::without_section()` upstream method (or fork `with_defaults`) | DateTimeSection has no removal hook |
| New `crates/zeroclaw-gateway/src/nyxclaw.rs` (or `avatar.rs`) | Avatar WebSocket handler |
| `/ws/avatar` route registration in `zeroclaw-gateway/src/lib.rs` | |

---

## Resolved questions (from deep read)

### R1. `StreamOptions` semantics

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamOptions {
    pub enabled: bool,
    pub count_tokens: bool,
}
```

Two fields. `Copy`. Anthropic checks `options.enabled` first — if false, emits a single `Final` and returns. `count_tokens` is for token-counting passthrough.

**Implication:** Our OpenAI `stream_chat()` impl must early-return a `Final`-only stream when `!options.enabled`. We can ignore `count_tokens` for now (Anthropic seems to as well based on the impl).

`loop_.rs` always calls with `StreamOptions::new(true)`.

### R2. `StreamEvent::PreExecutedToolCall` / `PreExecutedToolResult`

Emitted only by Claude Code proxy provider. **Anthropic and OpenAI don't emit them.**

**Implication:** Our OpenAI streaming impl doesn't need to construct these. Our nyxclaw event consumer needs a `_ => {}` arm for completeness when matching on `StreamEvent` (but we mostly consume `TurnEvent` from agent.rs anyway, which has different variants).

### R3. `cancel_tokens` reuse pattern

```rust
// Insert when starting a turn
let cancel_token = tokio_util::sync::CancellationToken::new();
state.cancel_tokens.lock().expect("cancel_tokens lock poisoned")
    .insert(session_key.to_string(), cancel_token.clone());

// Remove on turn end
state.cancel_tokens.lock().expect("cancel_tokens lock poisoned").remove(&session_key);

// External cancel (e.g., barge-in from another connection)
if let Some(ct) = state.cancel_tokens.lock().expect("...").get(&session_key) {
    ct.cancel();
}
```

**Implication:** Our nyxclaw barge-in does NOT need a parallel registry. Use `state.cancel_tokens` directly. Same pattern as `ws.rs::handle_ws_chat`.

### R4. `SystemPromptBuilder` has no removal API

```rust
pub fn with_defaults() -> Self {
    Self {
        sections: vec![
            Box::new(DateTimeSection),  // ← we want this gone
            Box::new(IdentitySection),
            // ... 7 more
        ],
    }
}
pub fn add_section(mut self, section: Box<dyn PromptSection>) -> Self { ... }
```

No `without_section()`. **Decision: add a minimal upstream method `pub fn without_section(mut self, name: &str) -> Self`** that filters by `section.name()`. ~5 LoC. Cleaner than forking `with_defaults`. Patch then becomes:

```rust
let prompt_builder = SystemPromptBuilder::with_defaults().without_section("datetime");
```

### R5. `Agent` and `turn_streamed`

```rust
pub async fn turn_streamed(
    &mut self,
    user_message: &str,
    event_tx: tokio::sync::mpsc::Sender<TurnEvent>,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
) -> Result<String>

pub async fn from_config(config: &Config) -> Result<Self>  // async!
```

`Agent` struct has 27 fields. AgentBuilder has chained setters. `from_config` is **async** (was sync in v0.5.0).

ChatRequest constructed at 3 sites in agent.rs:
1. `turn()` non-streaming
2. `turn_streamed()` streaming path → `provider.stream_chat(..., stream_opts)`
3. `turn_streamed()` fallback non-streaming when streaming unsupported

All 3 use `tools: if self.tool_dispatcher.should_send_tool_specs() { Some(&self.tool_specs) } else { None }`.

### R6. `loop_.rs` ChatRequest sites + `run_tool_call_loop`

3 ChatRequest constructions in loop_.rs. The streaming one:
```rust
let mut provider_stream = provider.stream_chat(
    ChatRequest { messages, tools: request_tools },
    model,
    Some(temperature),
    zeroclaw_providers::traits::StreamOptions::new(true),
);
```

`run_tool_call_loop()` has **26 parameters**. To thread `response_format` through, we either:
- (a) Add a 27th parameter `response_format: Option<&serde_json::Value>` (adds churn at every call site), OR
- (b) Stash response_format on the Agent and read it via task-local context (overkill), OR
- (c) Pass it via a single new struct `LoopExtras { response_format: Option<&Value> }` (cleaner, future-proofs)

**Decision: (a)** — pragmatic, follows the existing convention in this file. ~3-line change at the loop signature + each call site.

`TurnEvent` is defined in agent.rs (NOT loop_.rs). Variants:
- `Chunk { delta: String }`
- `Thinking { delta: String }`
- `ToolCall { id: String, name: String, args: serde_json::Value }` — note `id` is **new** since v0.5.0
- `ToolResult { id: String, name: String, output: String }` — note `id` is **new**

### R7. `voice_duplex.rs` is a stub

```rust
pub enum VoiceEvent { SpeechStart, SpeechEnd, BargeIn, TtsCancel, TtsChunk { ... } }
pub fn try_parse_voice_event(text: &str) -> Option<VoiceEvent>;
pub fn handle_voice_event(event: VoiceEvent) -> Option<serde_json::Value>;
```

Feature-gated `gateway-voice-duplex`, no route, no pipeline. **Cannot be reused as a base** for nyxclaw. We build from scratch using `ws.rs::handle_ws_chat` as the structural template.

### R8. Cargo deps

- Workspace edition 2024, Rust MSRV 1.87.
- `gateway-voice-duplex` feature flag exists (empty stub).
- `axum 0.8`, `tokio 1.50`, `futures-util 0.3` already in `zeroclaw-gateway` deps.
- **Missing in gateway runtime deps**: `async-stream`, `rand`, `uuid` (verify), `tokio-util`. Likely just add `rand` (we already use `rand::random`); `tokio-util` is needed for `CancellationToken` — verify. `async-stream` may not be needed if we use `tokio::sync::mpsc` + `tokio_stream::wrappers`.

### R9. `ChatMessage` constructors

```rust
ChatMessage::system(content)
ChatMessage::user(content)
ChatMessage::assistant(content)
ChatMessage::tool(content)
```

All exist. Use as before.

### R10. Anthropic streaming reference

Pattern to mirror for our OpenAI `stream_chat()`:
1. Early return `StreamEvent::Final` only if `!options.enabled`
2. Spawn `tokio::spawn` task; communicate via `mpsc::channel::<StreamResult<StreamEvent>>(64)`
3. Build native request body with `stream: true`
4. Parse SSE line-by-line via `tokio::io::AsyncBufReadExt`
5. Emit `TextDelta(StreamChunk)`, `ToolCall(ProviderToolCall)`, `Final`
6. Cancellation is implicit — receiver drop terminates the spawn task
7. Wrap with `stream::unfold(rx, ...).boxed()` to return `BoxStream`

**No `eventsource-stream` crate** — manual SSE parser is the convention.

---

## Final patch surface

| File (v0.7.4 path) | Change | Est. LoC |
|---|---|---|
| `crates/zeroclaw-api/src/provider.rs` | Add `pub response_format: Option<&'a serde_json::Value>` to `ChatRequest<'a>` | ~3 |
| `crates/zeroclaw-providers/src/openai.rs` | Implement `stream_chat()` SSE; override `supports_streaming()=true`; add `response_format` to `NativeChatRequest`; thread through `chat()`/`chat_with_tools()` | ~280 |
| `crates/zeroclaw-providers/src/reliable.rs` | Forward `response_format` in ChatRequest reconstruction (1 site) | ~3 |
| `crates/zeroclaw-providers/src/anthropic.rs` | Add `response_format: None` to its ChatRequest constructions if any (treat as no-op for now) | ~2 |
| `crates/zeroclaw-runtime/src/agent/agent.rs` | Add `response_format` field + `set_response_format()` setter; thread into 3 ChatRequest construction sites | ~25 |
| `crates/zeroclaw-runtime/src/agent/loop_.rs` | Add 27th `response_format` param to `run_tool_call_loop`; thread to 3 ChatRequest sites; pass `StreamOptions::new(true)` | ~12 |
| `crates/zeroclaw-runtime/src/agent/prompt.rs` | Add `pub fn without_section(self, name: &str) -> Self` method | ~6 |
| `crates/zeroclaw-gateway/src/lib.rs` | Add `pub mod nyxclaw;`; add `.route("/ws/avatar", get(nyxclaw::handle_ws_nyxclaw))`; add startup print | ~5 |
| `crates/zeroclaw-gateway/src/nyxclaw.rs` | NEW — avatar WebSocket handler | ~500 |
| `crates/zeroclaw-gateway/Cargo.toml` | Add `rand`, `tokio-util`, possibly `async-stream` to runtime deps | ~3 |

**Total: ~840 LoC across 9 modified + 1 new file.**

---

## Implementation phases

### Phase 0 — Setup (0.5 day)

#### 0.1 Clone v0.7.4
```bash
git clone --branch v0.7.4 --depth 1 https://github.com/zeroclaw-labs/zeroclaw.git zeroclaw-v0.7.4
cd zeroclaw-v0.7.4
rustup show  # confirm MSRV 1.87+
cargo build --workspace
cargo test --workspace
```

#### 0.2 Verify TurnEvent location
Read `crates/zeroclaw-runtime/src/agent/agent.rs` and confirm `TurnEvent` enum with the 4 expected variants (Chunk/Thinking/ToolCall/ToolResult). Note that `ToolCall` and `ToolResult` now have `id` fields — confirm field types.

#### 0.3 Verify Cargo deps
Inspect `crates/zeroclaw-gateway/Cargo.toml`. Note which of these are missing from runtime deps: `rand`, `tokio-util`, `async-stream`, `uuid`. Plan to add only what nyxclaw.rs needs.

---

### Phase 1 — `response_format` thread (0.5 day)

Lands the field through every layer without changing behavior. Feature is silently inert until something calls `set_response_format`.

#### 1.1 Add field to ChatRequest

`crates/zeroclaw-api/src/provider.rs`:

```rust
pub struct ChatRequest<'a> {
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    pub response_format: Option<&'a serde_json::Value>,  // NEW
}
```

After this, `cargo build --workspace` will fail at every ChatRequest construction site. That's intentional — those are the sites we patch next.

#### 1.2 Fix all construction sites

Compiler-driven. Add `response_format: None` (or `response_format: self.response_format.as_ref()` where applicable) at:
- `agent.rs` — 3 sites (turn, streaming, fallback)
- `loop_.rs` — 3 sites
- `reliable.rs` — 1 site (the wrap)
- `anthropic.rs` — verify and add as needed
- (`openai.rs` will be touched in Phase 2 with the streaming impl)

#### 1.3 Add field on Agent

`crates/zeroclaw-runtime/src/agent/agent.rs`:

```rust
pub struct Agent {
    // ... existing 27 fields ...
    response_format: Option<serde_json::Value>,
}

impl Agent {
    pub fn set_response_format(&mut self, fmt: Option<serde_json::Value>) {
        self.response_format = fmt;
    }
}
```

Add corresponding field + setter to `AgentBuilder`. Initialize to `None` in `from_config()` and `AgentBuilder::default()`.

#### 1.4 Thread through `run_tool_call_loop`

Add 27th parameter `response_format: Option<&serde_json::Value>`. Pass it from `Agent::turn_streamed` and `Agent::turn` callers. Use it in the 3 ChatRequest sites within the loop function.

#### 1.5 Add to OpenAI NativeChatRequest

`crates/zeroclaw-providers/src/openai.rs`:

```rust
struct NativeChatRequest {
    // ... existing fields ...
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}
```

Populate from `request.response_format.cloned()` in `chat()` and `chat_with_tools()`.

#### Phase 1 verification
- `cargo build --workspace` clean
- `cargo test --workspace` clean (no behavior change yet)
- Smoke test: set a response_format on Agent, run a turn, intercept OpenAI request body, confirm the field is present

---

### Phase 2 — OpenAI SSE streaming (1.5-2 days)

Most complex piece. Mirror Anthropic's approach.

#### 2.1 Implement `stream_chat()` on OpenAiProvider

In `crates/zeroclaw-providers/src/openai.rs`:

```rust
fn supports_streaming(&self) -> bool { true }
fn supports_streaming_tool_events(&self) -> bool { true }  // OpenAI streams tool calls

fn stream_chat(
    &self,
    request: ChatRequest<'_>,
    model: &str,
    temperature: Option<f64>,
    options: StreamOptions,
) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
    // Mirror anthropic.rs pattern:
    // 1. If !options.enabled → return Final-only stream
    // 2. Build NativeChatRequest with stream=true, include response_format
    // 3. Spawn task with mpsc::channel::<StreamResult<StreamEvent>>(64)
    // 4. Parse OpenAI SSE format:
    //    - data: {"choices":[{"delta":{"content":"..."}}]}
    //    - data: {"choices":[{"delta":{"tool_calls":[...]}}]}
    //    - data: [DONE]
    // 5. Emit StreamEvent::TextDelta, StreamEvent::ToolCall, StreamEvent::Final
    // 6. Wrap rx in stream::unfold().boxed()
}
```

Key differences from Anthropic SSE:
- OpenAI uses `data: {...}` lines with `[DONE]` sentinel; Anthropic uses typed events
- OpenAI accumulates tool_call args across multiple deltas indexed by `tool_calls[].index`
- OpenAI's final usage stats arrive in the last `data:` line before `[DONE]`

Reference our existing v0.5.0 patch SSE parser — the logic ports directly; only the trait signature differs.

#### 2.2 Tests

Add unit tests in `openai.rs`:
- `!options.enabled` → emits only `Final`
- Mock SSE input → assert correct StreamEvent sequence
- Tool call accumulation across deltas

#### Phase 2 verification
- `cargo test -p zeroclaw-providers` clean
- Manual test against real OpenAI API:
  - Set `response_format` on agent, call `turn_streamed()`
  - Confirm `TurnEvent::Chunk` events arrive incrementally (TTFT < 1s for cache hits)
  - Confirm `TurnEvent::ToolCall` fires for tool-using prompts
  - Confirm `cached_tokens` shows in usage stats

---

### Phase 3 — Prompt caching (0.25 day)

#### 3.1 Add `without_section()`

`crates/zeroclaw-runtime/src/agent/prompt.rs`:

```rust
impl SystemPromptBuilder {
    /// Remove a section by name. No-op if section is absent.
    pub fn without_section(mut self, name: &str) -> Self {
        self.sections.retain(|s| s.name() != name);
        self
    }
}
```

#### 3.2 Use it in nyxclaw.rs (or in agent setup)

When building Agent for the avatar channel, apply:
```rust
let prompt_builder = SystemPromptBuilder::with_defaults().without_section("datetime");
```

This requires the patch site to know how Agent gets its prompt builder. If `from_config` always uses `with_defaults()` internally, we may need a config flag (`prompt_skip_datetime: bool`) or to override post-construction.

**Decision branch:**
- (a) If Agent builder accepts a custom `prompt_builder` → just pass our custom one
- (b) If Agent only uses `from_config` with builtin defaults → add `prompt_skip_datetime: bool` field to `Config` and have `from_config` consult it

Verify in Phase 0 which path is needed.

#### Phase 3 verification
- After 2+ turns, OpenAI usage shows `cached_tokens > 0`

---

### Phase 4 — Avatar channel (1.5 days)

Largest single piece, but mostly mechanical port.

#### 4.1 Write `crates/zeroclaw-gateway/src/nyxclaw.rs`

Structure mirrors `ws.rs::handle_ws_chat`:

```rust
use crate::AppState;
use axum::{
    extract::{ws::{Message, WebSocket}, Query, State, WebSocketUpgrade},
    http::HeaderMap,
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use zeroclaw_runtime::agent::{Agent, TurnEvent};

const WS_PROTOCOL: &str = "zeroclaw.v1";
const BEARER_SUBPROTO_PREFIX: &str = "bearer.";
const AVATAR_SESSION_PREFIX: &str = "avatar_";

#[derive(Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConnectParams { /* same as v0.5.0 */ }

fn avatar_response_format() -> serde_json::Value {
    // Same JSON schema as v0.5.0 patch
}

fn extract_ws_token<'a>(headers: &'a HeaderMap, query_token: Option<&'a str>) -> Option<&'a str> {
    // Mirror ws.rs::extract_ws_token verbatim
}

pub async fn handle_ws_nyxclaw(
    State(state): State<AppState>,
    Query(params): Query<WsQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Mirror ws.rs::handle_ws_chat: auth check, subprotocol echo, on_upgrade
    // Differences: AVATAR_SESSION_PREFIX, calls handle_avatar_socket
}

async fn handle_avatar_socket(socket: WebSocket, state: AppState, session_id: Option<String>) {
    // 1. Build Agent: Agent::from_config(&state.config.lock().clone()).await
    //    Apply prompt_builder.without_section("datetime") via R4 path
    //    Set response_format: agent.set_response_format(Some(avatar_response_format()))
    // 2. Resolve session_id (uuid fallback), session_key = format!("{AVATAR_SESSION_PREFIX}{session_id}")
    // 3. Hydrate from state.session_backend if available
    // 4. Send session_start
    // 5. Optional connect handshake
    // 6. Message loop: spawn cancel_token, register in state.cancel_tokens,
    //    call agent.turn_streamed via scope_session_key(...),
    //    consume TurnEvent through dispatch_stream_event,
    //    on user "cancel" or new "message" → cancel_token.cancel(),
    //    cleanup: state.cancel_tokens.remove(&session_key)
}

// AvatarJsonExtractor — port verbatim from v0.5.0 patch (pure logic)
// extract_complete_sentences, split_sentences — port verbatim
// tool_call_filler — port (will be replaced in filler refactor later)

async fn dispatch_stream_event(
    event: TurnEvent,
    extractor: &mut AvatarJsonExtractor,
    sentence_buf: &mut String,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) {
    match event {
        TurnEvent::Chunk { delta } => { /* feed extractor, emit speech_chunk */ }
        TurnEvent::Thinking { .. } => { /* ignore or log */ }
        TurnEvent::ToolCall { name, args, .. } => {
            // Emit filler + tool_call
        }
        TurnEvent::ToolResult { name, output, .. } => { /* forward */ }
    }
}
```

**Critical changes vs v0.5.0:**
- `Agent::from_config` is async — `.await` it
- `turn_streamed` takes `cancel_token` directly — no need for our v0.5.0 custom signature
- `TurnEvent` enum match (not `serde_json::Value` JSON match)
- `TurnEvent::ToolCall { id, name, args }` has new `id` field (we ignore it for filler)
- Use `state.cancel_tokens` instead of building our own
- Wrap turn invocation in `zeroclaw_runtime::agent::loop_::scope_session_key(Some(session_key.clone()), agent.turn_streamed(...))`

Pure-logic helpers port unchanged:
- `AvatarJsonExtractor` (8-state JSON parser)
- `extract_complete_sentences` / `split_sentences`
- `tool_call_filler` (will be revisited in filler refactor)
- `extract_ws_token`
- `avatar_response_format`
- `WsQuery`, `ConnectParams` structs
- WebSocket protocol message shapes (mobile app expects exact JSON)

#### 4.2 Register in lib.rs

```rust
// In crates/zeroclaw-gateway/src/lib.rs
pub mod nyxclaw;

// In run_gateway router setup, add:
.route("/ws/avatar", get(nyxclaw::handle_ws_nyxclaw))

// Also add startup print:
println!("  GET  /ws/avatar — WebSocket avatar channel (nyxclaw)");
```

#### 4.3 Cargo.toml deps

In `crates/zeroclaw-gateway/Cargo.toml`, add to `[dependencies]`:
```toml
rand = { workspace = true }
tokio-util = { version = "0.7", default-features = false }
uuid = { version = "1", features = ["v4"] }
```

(Skip `async-stream` — Anthropic doesn't use it, we won't either.)

#### Phase 4 verification (full integration test)
- [ ] Mobile app connects to `/ws/avatar`
- [ ] `session_start` sent
- [ ] Connect handshake → `connected` ack
- [ ] Plain message → sentence-by-sentence `speech_chunk` events
- [ ] Tool-using message → `speech_chunk` filler + `tool_call` + `tool_result` + content `speech_chunk` + `rich_content` + `done`
- [ ] `{"type":"cancel"}` mid-turn → turn aborts cleanly, `done` with `cancelled: true`
- [ ] New `{"type":"message"}` mid-turn → previous cancels, new turn starts
- [ ] Disconnect/reconnect with same session_id → `resumed: true`
- [ ] Auth: bare connect without token → 401

---

### Phase 5 — Patch scripts + docs (0.5 day)

#### 5.1 Rewrite `patch.sh` and `patch.ps1`
- Update all paths from `src/<x>` to `crates/zeroclaw-<crate>/src/<x>`
- Update line numbers for injections
- Add `cargo build --workspace` verification step

#### 5.2 Update `claw_patches/zeroclaw/README.md`
- v0.7.4 instead of v0.5.0
- Workspace-aware build instructions
- Edition 2024 / MSRV 1.87 note
- Updated provider compatibility table:
  ```
  | Provider  | response_format | Streaming | Status         |
  |-----------|-----------------|-----------|----------------|
  | openai    | Patched         | Patched   | Full           |
  | anthropic | No-op (ignored) | Native    | Streaming only |
  | gemini    | Not patched     | Verify    | Unverified     |
  ```

#### 5.3 Bump baseline
Replace `claw_patches/zeroclaw/src/channels/nyxclaw.rs` with the new gateway-located version. Update file structure under `claw_patches/` to mirror v0.7.4 layout.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| OpenAI SSE parser bug | Medium | Mirror anthropic.rs structure; unit-test SSE parsing in isolation |
| `run_tool_call_loop` 27-param creep breaks downstream callers | Low | Compiler will catch; `cargo build --workspace` after each change |
| Edition 2024 lints fail compile | Low-Medium | Build early in Phase 0; fix as encountered. Common: `gen` keyword, `let_chains` change, lifetime elision |
| `prompt_builder` injection path unclear | Medium | Resolve in Phase 0 by reading `Agent::from_config` source — decide between (a) custom builder param vs (b) Config flag |
| `state.cancel_tokens` lock contention | Low | We hold lock briefly (insert/remove only); turns themselves use the cloned token |
| `TurnEvent` field changes (e.g., `id`) silently break extractor logic | Low | Pure-logic helpers don't depend on TurnEvent shape; ID gets dropped intentionally |
| Anthropic doesn't honor `response_format` | Known | Documented limitation; avatar uses OpenAI primarily |

## Out of scope for this upgrade

- Filler system refactor (separate plan: `docs/VOICE_FLOW_REFACTOR_PLAN.md`)
- Migrating to v0.7.4's `voice_duplex` foundation (audio-codec only, doesn't fit our blendshape pipeline)
- OpenClaw upgrade (separate codebase, separate cycle)
- Switching from manual SSE to `eventsource-stream` crate (not used by upstream)

## Effort summary

| Phase | Estimate |
|---|---|
| Phase 0 (setup + verify) | 0.5 day |
| Phase 1 (response_format thread) | 0.5 day |
| Phase 2 (OpenAI SSE streaming) | 1.5-2 days |
| Phase 3 (DateTimeSection removal via without_section) | 0.25 day |
| Phase 4 (nyxclaw channel + integration test) | 1.5 days |
| Phase 5 (patch scripts + docs) | 0.5 day |
| **Total** | **~4-5 days focused work** |

## Implementation kickoff checklist

Before writing any patch code:

- [ ] v0.7.4 cloned and `cargo build --workspace` succeeds locally
- [ ] Rust toolchain confirmed at MSRV 1.87+
- [ ] `TurnEvent` definition read in agent.rs, all 4 variants confirmed (especially `id` fields)
- [ ] `Agent::from_config` body read to resolve prompt_builder injection path (R4 decision)
- [ ] `crates/zeroclaw-gateway/Cargo.toml` runtime deps inventoried
- [ ] Existing v0.5.0 nyxclaw.rs backed up before any port
- [ ] Branch created: `feat/zeroclaw-v0.7.4-upgrade`

When all boxes ticked, start at Phase 1.
