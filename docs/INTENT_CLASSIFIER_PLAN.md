# Intent Classifier + Tool Call Status — ZeroClaw Patch Plan

## Problem

1. Every user message sends 63KB of tool definitions to the LLM (~25 tools), even for "Hi Nyx". This adds ~800ms to TTFT.
2. During tool execution (5-30 seconds), the avatar is silent. No feedback to the user.

## Solution: Two Phases

### Phase 1: Intent Classifier (fast chitchat, smart tool routing)

**Approach:** Add a `turn_with_avatar_streaming` method to `agent.rs` that performs a two-stage LLM call:

**Stage 1 — Classification call (fast):**
- Sends user message to LLM with ONE lightweight tool: `action`
- No real tools, no 63KB payload. ~500 byte tool spec.
- LLM either responds with chitchat directly, OR calls `action(task: "...")` + speaks an acknowledgment

**Stage 2 — Execution call (only if action tool was called):**
- Takes the `task` description from the action tool
- Sends to the existing `turn_with_streaming` with full tools
- Avatar already spoke the acknowledgment from Stage 1

**The `action` tool spec:**
```json
{
  "name": "action",
  "description": "Call this when the user wants you to DO something — check calendar, send email, search the web, look something up, compare things, fetch a page. Do NOT call for chitchat, opinions, jokes, stories, greetings, or general conversation.",
  "parameters": {
    "type": "object",
    "properties": {
      "task": {
        "type": "string",
        "description": "Brief description of what the user wants done"
      }
    },
    "required": ["task"]
  }
}
```

**Flow diagram:**
```
User speaks
  │
  ▼
Stage 1: LLM call (1 tool, ~500B payload)
  ├─ Chitchat response → speech_chunk → DONE (~700ms)
  │
  └─ action(task: "check calendar") + speech: "Let me check that."
       │
       ├─ speech_chunk: "Let me check that." → avatar speaks immediately
       │
       ▼
     Stage 2: LLM call (full tools, 63KB payload)
       → Agent loop with tool execution
       → Final response → speech_chunk → avatar speaks result
```

**Files to modify:**

| File | Change |
|------|--------|
| `src/agent/agent.rs` | Add `turn_with_avatar_streaming()` method — two-stage flow |
| `src/channels/nyxclaw.rs` | Call `turn_with_avatar_streaming()` instead of `turn_with_streaming()` |

**What stays the same:**
- `turn_with_streaming()` — unchanged, used by Stage 2 internally
- `prepare_turn()` — unchanged, called once at Stage 1
- All tool registration — unchanged
- Event channel — same `event_tx` for both stages
- JSON extractor, sentence splitter — unchanged in nyxclaw.rs
- nyxclaw Python code — zero changes

**Key design decisions:**
- Stage 1 uses the SAME `prepare_turn()` (memory store/recall happens once)
- Stage 1 and Stage 2 share the same `event_tx` — speech_chunks flow continuously
- Stage 1's response (including action tool call) goes into conversation history so Stage 2 has context
- Stage 2 receives the original user message from history, not the action tool's `task` — this preserves the full user intent
- `response_format` (JSON {speech, content}) applies to BOTH stages

**Expected latency:**
- Chitchat: ~700ms LLM + ~500ms TTS = **~1.2s** (down from 1.9s)
- Action: ~700ms Stage 1 + ~1.2s Stage 2 + ~500ms TTS = **~2.4s to first speech** (acknowledgment at ~1.2s, result at ~2.4s+tool time)

### Phase 2: Tool Call Status (future, after Phase 1 is stable)

**Approach:** In the agent loop (Stage 2), emit `speech_chunk` events between tool calls so the avatar speaks status updates during execution.

**Where the speech comes from:** The agent code generates status text from the `tool_call` event's name and args — not from the LLM, not injected by nyxclaw.

**Example flow:**
```
Agent loop iteration:
  1. tool_call: composio(GOOGLECALENDAR_EVENTS_LIST)
     → emit speech_chunk: "Checking your calendar..."
  2. Tool executes (5 seconds)
  3. tool_result: {success: true, ...}
     → emit speech_chunk: "Got your meetings."
  4. Next LLM call with results
  5. Final response: "You have one meeting at 2pm."
```

**Key concern (from earlier failure):** These speech_chunks create multiple TTS segments within one turn. The audio-clock pacing fix must work correctly for this. We validate Phase 1 first, then revisit the audio pipeline for Phase 2.

**Files to modify:**
- `src/agent/agent.rs` — emit `speech_chunk` events before/after tool execution
- Status text mapping — simple match on tool name to human-readable phrase

**Phase 2 is NOT started until Phase 1 is deployed and verified.**

## Phase 1 TODO

- [ ] 1.1 Add `action` tool spec as a const in `agent.rs`
- [ ] 1.2 Add `turn_with_avatar_streaming()` method to `Agent`
  - Calls `prepare_turn()` once
  - Stage 1: build minimal ChatRequest with only `action` tool, stream response
  - If no tool call → return (chitchat path)
  - If `action` tool called → add acknowledgment to history, run Stage 2
  - Stage 2: call existing agent loop with full `tool_specs`
- [ ] 1.3 Update `nyxclaw.rs` to call `turn_with_avatar_streaming()` instead of `turn_with_streaming()`
- [ ] 1.4 Rebuild ZeroClaw, test chitchat latency
- [ ] 1.5 Test action path (calendar check, email send)
- [ ] 1.6 Verify conversation history is coherent across both stages
- [ ] 1.7 Remove debug timeline logging (or keep behind a flag)
