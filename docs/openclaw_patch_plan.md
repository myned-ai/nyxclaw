# OpenClaw Avatar Channel Patch — Implementation Plan

**Target:** OpenClaw v2026.3.13-1 (TypeScript/Node.js)
**Goal:** Add a `/ws/avatar` endpoint that returns structured `{speech, content}` JSON, matching the ZeroClaw patch pattern.

## Architecture Overview

OpenClaw uses a broadcast-based WebSocket gateway. All connected clients receive events via `GatewayBroadcaster`. The avatar channel needs to **scope events to the avatar connection only**, not broadcast to every WS client.

```
Mobile App ←→ nyxclaw ←→ OpenClaw /ws/avatar
                              │
                              ├─ speech_chunk (speech text only → TTS)
                              ├─ rich_content (markdown → app cards)
                              ├─ tool_call / tool_result (forwarded)
                              └─ done (speech text only)
```

## Key Differences from ZeroClaw Patch

| Aspect | ZeroClaw (Rust) | OpenClaw (TypeScript) |
|--------|----------------|----------------------|
| Language | Rust | TypeScript/Node.js |
| Gateway | Direct WS handler | RPC broadcast system |
| LLM calls | Direct provider | Via `/v1/responses` API |
| Streaming | SSE tokens | Event broadcast |
| Channels | Separate WS endpoints | Outbound delivery targets |
| Config | TOML | TypeScript types + JSON |

## Files to Create

### 1. `src/gateway/server-methods/avatar.ts` (NEW — ~200 lines)
The avatar WebSocket handler. Equivalent to ZeroClaw's `nyxclaw.rs`.

**Responsibilities:**
- Accept WebSocket upgrade on `/ws/avatar`
- Authenticate via query param, header, or subprotocol token
- Receive text messages from nyxclaw
- Execute agent turn with `response_format` set to `{speech, content}` schema
- Parse the JSON response
- Send `speech_chunk` events (sentence-split speech text)
- Send `rich_content` event (if content is non-empty)
- Send `done` event with speech-only text
- Forward `tool_call` and `tool_result` events during agent execution
- Suppress raw chunk/delta events (avatar channel parses them itself)
- Maintain conversation history per session
- Handle cancel/barge-in messages

**Key integration points:**
- Reuse `agentCommandFromIngress()` from `src/commands/agent.ts` with `response_format` override
- Reuse session store for conversation persistence
- Reuse existing auth layer from `src/gateway/auth.ts`

## Files to Modify (Injections)

### 2. `src/gateway/open-responses.schema.ts` (~5 lines)
Add `response_format` field to the zod request schema.

```typescript
// Add to CreateResponseBodySchema:
response_format: z.object({
  type: z.literal("json_schema"),
  json_schema: z.object({
    name: z.string(),
    strict: z.boolean().optional(),
    schema: z.record(z.unknown()),
  }),
}).optional(),
```

### 3. `src/gateway/openresponses-http.ts` (~5 lines)
Pass `response_format` through to the LLM provider API call.

**Injection point:** Where the request body is constructed for the upstream LLM call.

### 4. `src/gateway/openai-http.ts` (~5 lines)
Same as above but for the legacy `/v1/chat/completions` path. Pass `response_format` through.

### 5. `src/gateway/server.impl.ts` (~10 lines)
Register the `/ws/avatar` WebSocket route.

**Injection point:** Where other WS routes are registered (near `/ws/chat` or similar).

## Response Format Schema

Same as ZeroClaw patch — enforced via `response_format: json_schema`:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "avatar_response",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "speech": {
          "type": "string",
          "description": "Brief spoken text for the avatar to say aloud via TTS"
        },
        "content": {
          "type": "string",
          "description": "Rich content (markdown with URLs, tables, etc.) shown in the chat view. Empty string if no rich content."
        }
      },
      "required": ["speech", "content"],
      "additionalProperties": false
    }
  }
}
```

## Wire Protocol (nyxclaw ← OpenClaw)

Identical to ZeroClaw patch:

| Event | Direction | Description |
|-------|-----------|-------------|
| `{"type":"message","content":"..."}` | nyxclaw → OpenClaw | User transcript |
| `{"type":"cancel"}` | nyxclaw → OpenClaw | Barge-in cancel |
| `{"type":"tool_call","name":"...","args":{}}` | OpenClaw → nyxclaw | Tool execution started |
| `{"type":"tool_result","name":"...","output":"..."}` | OpenClaw → nyxclaw | Tool execution result |
| `{"type":"speech_chunk","content":"..."}` | OpenClaw → nyxclaw | Sentence of speech text |
| `{"type":"rich_content","content":"..."}` | OpenClaw → nyxclaw | Markdown for app cards |
| `{"type":"done","full_response":"..."}` | OpenClaw → nyxclaw | Speech-only final text |

## Patch Delivery

Same structure as `claw_patches/zeroclaw/`:

```
claw_patches/openclaw/
├── README.md                              # Setup instructions
├── patch.sh                               # Bash apply script
├── patch.ps1                              # PowerShell apply script
└── src/
    └── gateway/
        └── server-methods/
            └── avatar.ts                  # New file (full copy)
```

The patch script:
- Copies `avatar.ts` into the OpenClaw source
- Injects `response_format` field into `open-responses.schema.ts`
- Injects passthrough into `openresponses-http.ts` and `openai-http.ts`
- Injects route registration into `server.impl.ts`
- Creates a backup of all modified files before patching

## AGENTS.md Prompt

Same structured output instructions as ZeroClaw — added to the OpenClaw agent's system prompt. Not included in the patch (user-specific), but documented in README.

## Dependencies

- No new npm packages required
- Reuses existing OpenClaw modules (agent, auth, sessions, broadcast)

## Risks & Considerations

1. **Broadcast scoping** — OpenClaw broadcasts events to ALL WS clients. The avatar handler must send events directly to the avatar socket, not via broadcast. This is the main architectural difference from the existing chat handler.

2. **Provider compatibility** — `response_format` works on OpenAI and compatible providers. Other providers (Anthropic, local models) may ignore it. Document as known limitation with fallback behavior (raw text as speech, no rich content).

3. **Streaming latency** — OpenClaw streams via event broadcast. The avatar channel needs to buffer the full response before parsing `{speech, content}`. Speech delivery starts only after the LLM finishes generating. For short responses this is fine; for long ones it adds latency.

4. **Session isolation** — Each avatar connection should have its own conversation history, isolated from other WS clients or channel sessions.

## TODO Checklist

- [ ] Read and understand `src/gateway/server-methods/chat.ts` (template for avatar handler)
- [ ] Read `src/commands/agent.ts` — `agentCommandFromIngress()` signature and usage
- [ ] Read `src/gateway/server.impl.ts` — route registration pattern
- [ ] Read `src/gateway/open-responses.schema.ts` — zod schema structure
- [ ] Read `src/gateway/openresponses-http.ts` — request construction
- [ ] Read `src/gateway/openai-http.ts` — legacy request construction
- [ ] Create `claw_patches/openclaw/` directory structure
- [ ] Write `src/gateway/server-methods/avatar.ts`
- [ ] Write injection for `open-responses.schema.ts` (response_format field)
- [ ] Write injection for `openresponses-http.ts` (passthrough)
- [ ] Write injection for `openai-http.ts` (passthrough)
- [ ] Write injection for `server.impl.ts` (route registration)
- [ ] Write `patch.sh` and `patch.ps1`
- [ ] Write `README.md` with setup instructions
- [ ] Test patch application on clean OpenClaw v2026.3.13-1
- [ ] Test end-to-end with nyxclaw
