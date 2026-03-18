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

## Files modified

| File | Type | What changed |
|------|------|-------------|
| `src/providers/traits.rs` | MODIFIED | Added `response_format: Option<&serde_json::Value>` to `ChatRequest` struct. Removed `Copy` derive (Value isn't Copy). |
| `src/providers/openai.rs` | MODIFIED | Added `response_format` field to `NativeChatRequest`, passed through in `chat()` method to OpenAI API. |
| `src/agent/agent.rs` | MODIFIED | Added `response_format` field to Agent + setter. Added `turn_with_events()` method that streams tool_call/tool_result/chunk events via mpsc channel. |
| `src/channels/nyxclaw.rs` | **NEW** | Dedicated avatar WebSocket channel. Sets structured JSON response format, uses `turn_with_events()`, parses `{speech, content}` response, sends `speech_chunk` + `rich_content` events. |
| `src/gateway/mod.rs` | INJECTED | 3 lines added: import, `/ws/avatar` route, print line. Original file preserved. |
| `src/channels/mod.rs` | INJECTED | 1 line added: `pub mod nyxclaw;`. Original file preserved. |

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
- Set to empty string `""` when there's nothing visual to show.
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
```

## WebSocket protocol: `/ws/avatar`

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
{"type": "done", "full_response": "{\"speech\": \"...\", \"content\": \"...\"}"}
{"type": "error", "message": "..."}
```

### Key differences from `/ws/chat`

| Feature | `/ws/chat` | `/ws/avatar` |
|---------|-----------|-------------|
| Response format | Raw text | Structured `{speech, content}` JSON |
| Streaming events | `done` only | `tool_call`, `tool_result`, `speech_chunk`, `rich_content`, `done` |
| Cancel support | No | Yes (`{"type": "cancel"}`) |
| LLM constraint | `response_format` not set | `response_format: json_schema` enforced |

## Reverting

To revert all patches:
```bash
cp -r /path/to/zeroclaw/.nyxclaw-patch-backup/* /path/to/zeroclaw/
rm -rf /path/to/zeroclaw/.nyxclaw-patch-backup
rm /path/to/zeroclaw/src/channels/nyxclaw.rs
```

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
