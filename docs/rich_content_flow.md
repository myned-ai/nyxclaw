# Rich Content Flow — End-to-End

## Sequence Diagram

```mermaid
sequenceDiagram
    participant App as Mobile App
    participant Nyx as nyxclaw
    participant ZC as ZeroClaw
    participant LLM as OpenAI (gpt-4o)

    App->>Nyx: audio (PCM16 @ 24kHz)
    Nyx->>Nyx: VAD + STT → "What is the Wikipedia page for Rome?"
    Nyx->>ZC: {"type":"message","content":"What is the Wikipedia page for Rome?"}
    Note over Nyx,ZC: WebSocket /ws/avatar

    ZC->>LLM: chat completion (response_format: {speech, content})

    Note over ZC: Agent loop — LLM may call tools
    ZC-->>Nyx: {"type":"tool_call","name":"shell","args":{...}}
    ZC-->>Nyx: {"type":"tool_result","name":"shell","output":"..."}

    Note over ZC: LLM returns final JSON
    LLM->>ZC: {"speech":"Here's the Wikipedia page.","content":"**Rome**\nhttps://en.wikipedia.org/wiki/Rome"}

    Note over ZC: Avatar channel parses JSON, splits into events
    ZC->>Nyx: {"type":"speech_chunk","content":"Here's the Wikipedia page."}
    ZC->>Nyx: {"type":"rich_content","content":"**Rome**\nhttps://en.wikipedia.org/wiki/Rome"}
    ZC->>Nyx: {"type":"done","full_response":"Here's the Wikipedia page."}

    Note over Nyx: nyxclaw routes each event
    Nyx->>Nyx: speech_chunk → TTS → wav2arkit → sync_frame
    Nyx->>App: {"type":"sync_frame","audio":"...","blendshapes":[...]}
    Nyx->>App: {"type":"transcript_delta","text":"Here's the Wikipedia page."}
    Nyx->>App: {"type":"rich_content","content":"**Rome**\nhttps://en.wikipedia.org/wiki/Rome"}
    Nyx->>App: {"type":"transcript_done","text":"Here's the Wikipedia page."}

    Note over App: App receives events
    App->>App: sync_frame → avatar lip-syncs
    App->>App: transcript_delta → text bubble (speech only)
    App->>App: rich_content → switch to chat view + render card
```

## Event Types

### ZeroClaw → nyxclaw (WebSocket /ws/avatar)

| Event | When | Payload |
|-------|------|---------|
| `tool_call` | LLM calls a tool during agent loop | `{"type":"tool_call","name":"shell","args":{...}}` |
| `tool_result` | Tool execution completes | `{"type":"tool_result","name":"shell","output":"...","success":true}` |
| `speech_chunk` | Parsed speech sentence from final response | `{"type":"speech_chunk","content":"Here's the Wikipedia page."}` |
| `rich_content` | Parsed content from final response (non-empty) | `{"type":"rich_content","content":"**Rome**\nhttps://..."}` |
| `done` | Turn complete | `{"type":"done","full_response":"Here's the Wikipedia page."}` |

### nyxclaw → Mobile App (WebSocket /ws)

| Event | When | Payload |
|-------|------|---------|
| `sync_frame` | Audio + blendshapes ready | `{"type":"sync_frame","audio":"...","blendshapes":[...]}` |
| `transcript_delta` | Speech text streaming | `{"type":"transcript_delta","text":"Here's the Wikipedia page."}` |
| `rich_content` | Rich content to display | `{"type":"rich_content","content":"**Rome**\nhttps://..."}` |
| `transcript_done` | Speech text finalized | `{"type":"transcript_done","text":"Here's the Wikipedia page."}` |
| `audio_start` | TTS audio begins | `{"type":"audio_start"}` |
| `audio_end` | TTS audio ends | `{"type":"audio_end"}` |

## Current Bug

ZeroClaw's avatar channel event loop (nyxclaw.rs line 352) forwards raw `chunk` events
from `turn_with_events` directly to nyxclaw. These contain the full JSON string
`{"speech":"...","content":"..."}`. Then AFTER the turn completes, the channel parses the
final response and sends clean `speech_chunk` + `rich_content` events.

Result: nyxclaw receives BOTH raw JSON chunks AND clean speech_chunks. The transcript
shows both concatenated.

Fix: suppress `chunk` and `done` events in the event forwarding loop (line 352), since
the avatar channel handles parsing and event generation itself after turn completion.
