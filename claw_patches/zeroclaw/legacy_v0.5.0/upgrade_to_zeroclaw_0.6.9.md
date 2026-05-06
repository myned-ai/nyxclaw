# Upgrade ZeroClaw Patches: v0.5.0 → v0.6.9

Status: **Planning** | Created: 2026-04-08

## Context

Our v0.5.0 patches add a `/ws/avatar` channel to ZeroClaw for nyxclaw voice+avatar integration. ZeroClaw v0.6.9 has significant internal changes — all existing patches need rewriting. This document tracks every task required to produce a working v0.6.9 patch set with zero feature loss.

## What v0.6.9 gives us for free

These features existed only in our patches for v0.5.0 but are now built into v0.6.9:

| Feature | v0.5.0 (we patched) | v0.6.9 (built-in) |
|---------|---------------------|--------------------|
| `StreamEvent` enum | We added to `traits.rs` | Native: `TextDelta`, `ToolCall`, `Final` |
| `stream_chat()` trait method | We added to `Provider` trait | Native on `Provider` trait |
| Agent streaming | We added `turn_with_streaming()` | Native `turn_streamed(msg, event_tx)` |
| Streaming events | Custom `serde_json::Value` events | Native `TurnEvent::Chunk`, `ToolCall`, `ToolResult`, `Thinking` |
| Anthropic SSE streaming | Not patched | `AnthropicProvider` has `supports_streaming() = true` |
| Cancellation in loop | We added | `is_tool_loop_cancelled()` + `CancellationToken` in loop internals |

## What v0.6.9 still lacks (must patch)

| Feature | Why we need it |
|---------|---------------|
| `response_format` on `ChatRequest` | Forces LLM to return `{"speech": "...", "content": "..."}` — guarantees speech never contains URLs/code/markdown |
| `response_format` on `Agent` | Stores the schema, passes it to provider on every turn |
| `set_response_format()` on `Agent` | nyxclaw.rs calls this to set the avatar JSON schema |
| OpenAI SSE streaming (`stream_chat()` impl) | `OpenAiProvider` still returns `supports_streaming() = false` — without this, text arrives as one block, not sentence-by-sentence |
| `response_format` on `NativeChatRequest` (OpenAI) | Serializes to OpenAI API's `response_format` parameter |
| `DateTimeSection` removal | Injects second-precision timestamps that bust OpenAI prompt cache on every turn |
| `/ws/avatar` endpoint | The avatar WebSocket channel — entirely separate protocol from `/ws/chat` |
| `CancellationToken` on `turn_streamed()` | v0.6.9's `turn_streamed()` has NO cancel parameter — barge-in requires it |

## Breaking API changes (v0.5.0 → v0.6.9)

These are changes in ZeroClaw's internal APIs that break our existing patch code:

| What changed | v0.5.0 (our patches use) | v0.6.9 | Impact |
|---|---|---|---|
| `Agent::from_config()` | `fn from_config(config: &AgentConfig) -> Result<Self>` (sync) | `async fn from_config(config: &Config) -> Result<Self>` | nyxclaw.rs must `.await` and pass `&Config` |
| Agent streaming method | `turn_with_streaming(msg, event_tx, Some(cancel_token))` | `turn_streamed(msg, event_tx)` — no cancel token | Must add cancel param or find alternative |
| Event types | `serde_json::Value` with `"type": "content_delta"` etc. | `TurnEvent` enum: `Chunk { delta }`, `ToolCall { name, args }`, `ToolResult { name, output }`, `Thinking { delta }` | nyxclaw.rs dispatch logic must match on enum, not JSON |
| Event content field | `"content_delta"` → feed to JSON extractor, `"content_done"` → finalize | `Chunk { delta }` contains raw LLM output (which is the JSON `{"speech":..., "content":...}`) | Same extractor logic, different event matching |
| `AppState.config` | `state.config.lock().clone()` returned `AgentConfig` | `state.config.lock()` returns `Config` (parking_lot Mutex) | Adjust config access |
| `AppState.session_backend` | We added this field | May not exist in v0.6.9 — verify | May need to add or use v0.6.9's session/memory system |
| `AppState.event_tx` | We added this broadcast channel | May not exist in v0.6.9 — verify | May need to add or use v0.6.9's observer/hook system |
| `ChatRequest` fields | `messages`, `tools`, `response_format` (our addition) | `messages`, `tools` only | Must add `response_format` field |
| Loop ChatRequest sites | 2 construction sites | 3 construction sites + `consume_provider_streaming_response` | More places to thread `response_format` through |
| `crate::agent::loop_::is_tool_loop_cancelled` | We used this | Still exists at same path | No change needed |
| `crate::providers::sanitize_api_error` | We used this | Verify still exists | Check and update path if moved |

---

## Task List

### Phase 1: Setup & Verification

#### 1.1 — Clone ZeroClaw v0.6.9
```bash
git clone --branch v0.6.9 --depth 1 https://github.com/zeroclaw-labs/zeroclaw.git zeroclaw-v0.6.9
```
Verify it builds clean: `cargo build && cargo test`

#### 1.2 — Verify AppState fields
Check `src/gateway/mod.rs` for:
- Does `AppState` have `session_backend`? If not, what's the v0.6.9 equivalent for session persistence?
- Does `AppState` have `event_tx`? If not, what's the v0.6.9 equivalent for broadcasting events?
- Does `AppState` have `model`? (used in our timeline logging)
- How does `state.pairing` work? (used for bearer token auth)

These answers determine how much nyxclaw.rs needs to change beyond the API renames.

#### 1.3 — Verify helper functions still exist
Grep for these in v0.6.9:
- `crate::providers::sanitize_api_error` — used in error handling
- `crate::providers::ChatMessage::user()` / `::assistant()` — used for session persistence
- `crate::agent::loop_::is_tool_loop_cancelled` — used for barge-in detection

---

### Phase 2: Provider patches (response_format + OpenAI streaming)

#### 2.1 — Add `response_format` to `ChatRequest` (traits.rs)
**File:** `src/providers/traits.rs`
**Change:** Add one field to the `ChatRequest` struct:
```rust
pub struct ChatRequest<'a> {
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    pub response_format: Option<&'a serde_json::Value>,  // NEW
}
```
**Risk:** Low — additive change, no existing code breaks (new field is `Option`).
**Verify:** All existing `ChatRequest { messages, tools }` constructions will fail to compile — that's intentional, it forces us to find every site.

#### 2.2 — Fix all `ChatRequest` construction sites
After 2.1, `cargo build` will show every place that constructs `ChatRequest` without `response_format`. Add `response_format: None` to each:

**Known sites (from v0.6.9 analysis):**
- `src/agent/loop_.rs` — 3 sites (streaming path, non-streaming fallback, main non-streaming)
- `src/providers/reliable.rs` — reconstructs ChatRequest when delegating to inner provider
- `src/agent/agent.rs` — if `turn()` or `turn_streamed()` construct ChatRequest directly
- Any other provider files that build ChatRequest

For the agent's streaming/turn methods, pass `self.response_format` instead of `None`.

#### 2.3 — Add `response_format` field to `Agent` (agent.rs)
**File:** `src/agent/agent.rs`
**Changes:**
1. Add field to Agent struct: `response_format: Option<serde_json::Value>`
2. Add field to AgentBuilder + setter
3. Add public method:
   ```rust
   pub fn set_response_format(&mut self, fmt: Option<serde_json::Value>) {
       self.response_format = fmt;
   }
   ```
4. In `turn_streamed()` and `turn()`: pass `self.response_format.as_ref()` when constructing ChatRequest for the provider call

#### 2.4 — Add `response_format` to OpenAI `NativeChatRequest` (openai.rs)
**File:** `src/providers/openai.rs`
**Changes:**
1. Add to `NativeChatRequest`:
   ```rust
   #[serde(skip_serializing_if = "Option::is_none")]
   response_format: Option<serde_json::Value>,
   ```
2. In `chat()` / `chat_with_tools()` / `chat_with_system()`: populate from `request.response_format.cloned()`
3. In any non-request methods: set `response_format: None`

#### 2.5 — Implement `stream_chat()` on OpenAI provider (openai.rs)
**File:** `src/providers/openai.rs`
**Changes:**
1. Override `supports_streaming() -> bool` to return `true`
2. Implement `stream_chat()` — SSE parsing of OpenAI's streaming response format
3. Include `response_format` in the streaming request body
4. Add `"stream": true` to the request
5. Parse SSE events into `StreamEvent::TextDelta` / `StreamEvent::ToolCall` / `StreamEvent::Final`

**Reference:** Our v0.5.0 openai.rs patch has a working SSE implementation. Port the logic but adapt to v0.6.9's `StreamEvent` variants (which already exist in traits.rs).

**Test:** Build and run `cargo test -p zeroclaw --lib providers::openai`. Then test manually with a real OpenAI API key.

---

### Phase 3: Prompt caching patch

#### 3.1 — Remove DateTimeSection (prompt.rs)
**File:** `src/agent/prompt.rs`
**Change:** In `SystemPromptBuilder::with_defaults()`, remove or comment out the line that adds `DateTimeSection`.

In v0.6.9, `DateTimeSection` injects:
```
## CRITICAL CONTEXT: CURRENT DATE & TIME
Date: 2026-04-08
Time: 14:30:45 (CET)
ISO 8601: 2026-04-08T14:30:45+02:00
```

The second-precision timestamp changes every turn, busting OpenAI's prompt prefix cache. User messages already carry timestamps. Removing this enables 95-99% cache hit rates.

**Risk:** Low. ZeroClaw's own tests don't depend on this section.

---

### Phase 4: Avatar channel (nyxclaw.rs)

#### 4.1 — Rewrite nyxclaw.rs for v0.6.9 API

**File:** `src/channels/nyxclaw.rs` (new file, replaces v0.5.0 version)

This is the largest task. The channel logic is the same, but every agent API call must be updated.

**Changes needed:**

**a) Agent initialization:**
```rust
// v0.5.0:
let mut agent = crate::agent::Agent::from_config(&config)?;

// v0.6.9:
let mut agent = crate::agent::Agent::from_config(&config).await?;
```
Note: `from_config` now takes `&Config` (full config), not `&AgentConfig`.

**b) Streaming method:**
```rust
// v0.5.0:
agent.turn_with_streaming(content, event_tx, Some(cancel_token.clone()))

// v0.6.9:
agent.turn_streamed(content, event_tx)
// BUT: no cancel_token parameter — see task 4.2
```

**c) Event dispatch — match on TurnEvent enum instead of JSON:**
```rust
// v0.5.0 (JSON values):
match event["type"].as_str() {
    "tool_call" => { ... }
    "tool_result" => { ... }
    "content_delta" => { extractor.feed(delta); ... }
    "content_done" => { extractor.finalize(); ... }
}

// v0.6.9 (TurnEvent enum):
match event {
    TurnEvent::Chunk { delta } => {
        // delta contains raw LLM text (the JSON {"speech":..., "content":...})
        // Feed to AvatarJsonExtractor same as before
        if let Some(new_speech) = extractor.feed(&delta) { ... }
    }
    TurnEvent::ToolCall { name, args } => {
        // Emit filler + forward tool_call
    }
    TurnEvent::ToolResult { name, output } => {
        // Forward tool_result
    }
    TurnEvent::Thinking { delta } => {
        // Ignore or log — thinking tokens aren't speech
    }
}
```

**d) No more `"content_done"` event:**
v0.5.0 had an explicit `content_done` event to trigger `extractor.finalize()`. In v0.6.9, the `turn_streamed()` future completes when the turn is done — call `extractor.finalize()` after the future resolves and the event channel is drained.

**e) AppState access:**
Verify and adapt:
- `state.config.lock()` — now returns `Config`, not `AgentConfig`
- `state.session_backend` — may not exist, check Phase 1.2
- `state.event_tx` — may not exist, check Phase 1.2
- `state.pairing` — verify auth API unchanged

**f) Keep unchanged:**
- `AvatarJsonExtractor` — state machine is pure logic, no API dependencies
- `extract_complete_sentences()` / `split_sentences()` — pure string functions
- `tool_call_filler()` — pure mapping function
- `extract_ws_token()` — pure header parsing
- `avatar_response_format()` — pure JSON construction
- `WsQuery`, `ConnectParams` structs — pure deserialization
- WebSocket protocol messages — these are what the mobile app expects, don't change

#### 4.2 — Add CancellationToken support to `turn_streamed()` (agent.rs)

**Problem:** v0.6.9's `turn_streamed()` has no cancellation parameter. Our barge-in feature requires aborting a turn mid-stream when the user speaks or sends `{"type": "cancel"}`.

**File:** `src/agent/agent.rs`

**Option A — Add cancel_token parameter (preferred):**
```rust
pub async fn turn_streamed(
    &mut self,
    user_message: &str,
    event_tx: tokio::sync::mpsc::Sender<TurnEvent>,
    cancel_token: Option<tokio_util::sync::CancellationToken>,  // NEW
) -> Result<String>
```
Then pass it into the internal loop call. The loop already supports `CancellationToken` (confirmed in `loop_.rs`).

**Option B — Separate method:**
Add `turn_streamed_cancellable()` that wraps `turn_streamed` logic with a cancel token, leaving the original method untouched. Less invasive but duplicates code.

**Verify:** After this change, confirm barge-in works: send `{"type": "cancel"}` during an active turn, verify the turn aborts and returns `ToolLoopCancelled`.

#### 4.3 — Register nyxclaw channel

**File:** `src/channels/mod.rs`
**Change:** Add `pub mod nyxclaw;` after the last channel module.

**File:** `src/gateway/mod.rs`
**Changes:**
1. Add import: `use crate::channels::nyxclaw;`
2. Add route: `.route("/ws/avatar", get(nyxclaw::handle_ws_nyxclaw))`
3. Add startup print: `println!("  avatar: ws://{}:{}/ws/avatar", host, port);`

---

### Phase 5: Build & Test

#### 5.1 — Compile
```bash
cd zeroclaw-v0.6.9
cargo build 2>&1 | head -50
```
Fix any compilation errors. The most common will be:
- Missing `response_format` field in ChatRequest constructions (from 2.1)
- Changed import paths
- async/await mismatches (from_config is now async)

#### 5.2 — Run existing tests
```bash
cargo test
```
Our patches should not break any existing tests. If they do, investigate — it likely means we changed a shared struct signature incorrectly.

#### 5.3 — Manual integration test
1. Start patched ZeroClaw with OpenAI provider
2. Connect nyxclaw to `/ws/avatar`
3. Verify these work:
   - [ ] `session_start` received on connect
   - [ ] `connect` handshake → `connected` response
   - [ ] Simple message → `speech_chunk` events arrive sentence-by-sentence (streaming works)
   - [ ] Message requiring tool use → `speech_chunk` filler (`filler: true`) + `tool_call` + `tool_result` + `speech_chunk` (non-filler) + `rich_content` + `done`
   - [ ] `{"type": "cancel"}` during active turn → turn aborts, `done` with `cancelled: true`
   - [ ] New message during active turn → previous turn cancelled, new turn starts
   - [ ] Session resume: disconnect and reconnect with same `session_id` → `resumed: true` + correct `message_count`
   - [ ] Auth: connect without token when pairing enabled → 401

#### 5.4 — Verify prompt caching
After 2+ turns with the same system prompt:
- Check OpenAI API response headers for `x-openai-cache-hit: true`
- Or check ZeroClaw logs for cache hit indicators

---

### Phase 6: Patch scripts & docs

#### 6.1 — Rewrite patch.sh and patch.ps1
Update both scripts for v0.6.9:
- New file paths and line numbers for injections
- Handle the 3 ChatRequest sites in loop_.rs (was 1 in v0.5.0)
- Handle reliable.rs changes
- Update Cargo.toml injection (check if `async-stream` is still needed or if v0.6.9's `tokio-stream`/`futures-util` suffice)
- Update verification checks

#### 6.2 — Update README.md
- Change version references from v0.5.0 to v0.6.9
- Update "Files modified" table
- Update "How it works" section
- Update `turn_with_streaming` references to `turn_streamed`
- Note that Anthropic now works for streaming without patches (native support)
- Add note about new `CancellationToken` parameter

#### 6.3 — Update provider compatibility table
```
| Provider    | Structured output | Streaming | Status          |
|-------------|-------------------|-----------|-----------------|
| openai      | response_format   | Patched   | Full support    |
| anthropic   | Not yet           | Native    | Streaming only  |
| gemini      | Not yet           | Not yet   | No support      |
```

---

## File change summary

| File | Change type | Lines changed (est.) |
|------|-------------|---------------------|
| `src/providers/traits.rs` | Add 1 field | ~1 |
| `src/providers/openai.rs` | Add field + implement stream_chat() | ~150-200 |
| `src/providers/reliable.rs` | Add response_format forwarding | ~3-5 |
| `src/providers/anthropic.rs` | Add response_format: None | ~1-2 |
| `src/agent/agent.rs` | Add field + setter + cancel token on turn_streamed | ~20-30 |
| `src/agent/loop_.rs` | Add response_format to ChatRequest sites | ~3-5 |
| `src/agent/prompt.rs` | Remove DateTimeSection | ~1 |
| `src/channels/nyxclaw.rs` | New file (rewritten for v0.6.9 API) | ~900 |
| `src/channels/mod.rs` | Add module declaration | ~1 |
| `src/gateway/mod.rs` | Add import + route + print | ~3 |
| `Cargo.toml` | Add async-stream (if needed) | ~1 |
| **Total** | 8 modified + 1 new | ~1100-1150 |

## Risks

1. **OpenAI SSE parsing** — Most complex single patch. Port from v0.5.0 but adapt to v0.6.9's `StreamEvent` variants. Test with real API.
2. **CancellationToken threading** — v0.6.9's loop uses CancellationToken internally but `turn_streamed()` doesn't expose it. Adding it requires understanding how the loop receives it. May need to read `turn_streamed()` implementation carefully.
3. **AppState unknowns** — `session_backend` and `event_tx` may not exist in v0.6.9. Phase 1.2 must resolve this before writing nyxclaw.rs.
4. **`consume_provider_streaming_response`** — This function in loop_.rs bypasses ChatRequest and takes individual params. Need to verify it receives `response_format` somehow, or add a parameter.
