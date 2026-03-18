# Research: Triggering Rich Content Alongside Speech in Voice/Chat AI Systems

**Date:** 2026-03-18
**Context:** nyxclaw voice-to-avatar server needs to reliably display rich content (cards, links, tables, images) alongside spoken LLM responses. The core problem: when an LLM is used as the brain, it does not reliably call tools like `send_rich_content` -- it often just speaks the answer instead.

---

## Table of Contents

1. [OpenClaw on WhatsApp](#1-openclaw-on-whatsapp)
2. [OpenAI Function Calling Reliability](#2-openai-function-calling-reliability)
3. [Voice Assistant Architectures (Alexa, Google, Siri)](#3-voice-assistant-architectures)
4. [Structured Output Approach](#4-structured-output-approach)
5. [Two-Pass / Post-Processing](#5-two-pass--post-processing)
6. [Anthropic/Claude Tool Use](#6-anthropicclaude-tool-use)
7. [Community Solutions](#7-community-solutions)
8. [Comparison Matrix](#8-comparison-matrix)
9. [Recommendations for nyxclaw](#9-recommendations-for-nyxclaw)

---

## 1. OpenClaw on WhatsApp

### How It Works

OpenClaw (68k+ GitHub stars) connects to WhatsApp via the Baileys library (QR code pairing to your personal WhatsApp account). Its architecture is:

- **Dual tool exposure**: Tools are exposed to the LLM through both (1) human-readable descriptions in the system prompt and (2) structured function definitions in the tool schema. Both channels are required -- "if a tool doesn't appear in the system prompt or the schema, the model cannot call it."
- **Channel-specific `message` tool**: A unified `message` tool handles cross-platform messaging (WhatsApp, Telegram, Slack, Discord, etc.) with parameters for `text`, `media`, `card` (Adaptive Cards on MS Teams), and `buttons` (Telegram inline keyboards).
- **Skills system**: SKILL.md files with YAML frontmatter teach the agent how and when to use tools. Skills declare runtime requirements and usage patterns.
- **Tool profiles**: Access control via `tools.allow`/`tools.deny` policies and preset profiles (minimal, coding, messaging, full).

### Rich Content on WhatsApp Specifically

WhatsApp's API is limited compared to Slack/Teams. OpenClaw supports:
- Text + optional media attachments (images, audio, video, documents)
- Link previews via `generateHighQualityLinkPreview`
- **No native card/carousel support** on WhatsApp (platform limitation)
- Adaptive Cards only on MS Teams; interactive cards with buttons/forms/polls on Feishu (Lark)

### Key Insight

OpenClaw relies entirely on the LLM deciding to call the `message` tool with the right parameters. There is no forced tool calling or structured output guarantee. The reliability comes from detailed system prompts and skill descriptions, not from API-level enforcement. This means it suffers from the same "LLM ignores the tool" problem we are trying to solve.

### Sources
- [OpenClaw Tools Documentation](https://docs.openclaw.ai/tools)
- [OpenClaw WhatsApp Channel](https://docs.openclaw.ai/channels/whatsapp)
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [Feature: Webchat UI inline link preview cards](https://github.com/openclaw/openclaw/issues/29466)

---

## 2. OpenAI Function Calling Reliability

### tool_choice Parameter

OpenAI provides three modes for controlling tool use:

| Mode | Behavior | Text + Tool Same Response? |
|------|----------|---------------------------|
| `"auto"` (default) | Model decides whether to call tools | Yes, possible but not guaranteed |
| `"required"` | Must call at least one tool | No text content (tool only) |
| `{"type": "function", "name": "X"}` | Must call specific function X | No text content (tool only) |

### Getting Both Text AND Tool Calls (OpenAI)

This is the critical finding for our use case. With `tool_choice: "auto"`, OpenAI models **can** return both `content` (text) and `tool_calls` in the same response message. The trick is in the tool description:

> "Write a function where the description specifically describes the necessity for writing an initial message to the user about the intention to use the function, and then proceeding automatically to using the function after that."

Community members report this is **reliably reproducible** with GPT-4 and GPT-4 Turbo when the tool description explicitly instructs the model to speak first, then call the tool. However, it is prompt-dependent and not API-guaranteed behavior.

### Strict Mode

Setting `strict: true` in function definitions ensures tool call arguments always match the schema (no type mismatches or missing fields). This is guaranteed by constrained decoding, not just best-effort.

### parallel_tool_calls

Setting `parallel_tool_calls: false` ensures at most one tool is called per turn. Useful if you need deterministic single-tool behavior.

### Best Practices for Reliability

1. **Detailed tool descriptions** -- describe WHEN to use the tool, not just what it does
2. **System prompt emphasis** -- mention critical tools in the system prompt for salience
3. **Lower temperature** -- reduces randomness in tool selection
4. **Strict mode** -- guarantees schema conformance
5. **Few-shot examples** -- show the model correct tool usage patterns in the prompt

### Reliability Assessment

- `tool_choice: "required"` -- 100% reliable for forcing A tool call, but prevents text output
- `tool_choice: "auto"` with prompt engineering -- ~90-95% reliable for getting both text + tool call
- The gap is the core problem: that remaining 5-10% where the LLM speaks instead of calling the tool

### Sources
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [tool_choice: "required" announcement](https://community.openai.com/t/new-api-feature-forcing-function-calling-via-tool-choice-required/731488)
- [Prompting Best Practices for Tool Use](https://community.openai.com/t/prompting-best-practices-for-tool-use-function-calling/1123036)
- [Tool_calls together with normal message.content](https://community.openai.com/t/tool-calls-together-with-normal-message-content/1135132)
- [Ensure every reply includes message and tool call](https://community.openai.com/t/how-can-i-ensure-every-llm-reply-includes-exactly-one-message-and-one-tool-call/1283087)

---

## 3. Voice Assistant Architectures

### Alexa (Amazon)

**Architecture: Deterministic dual-directive system.**

Alexa does NOT use an LLM to decide what to show. The skill developer's fulfillment code explicitly constructs both outputs:

1. **Speak directive** -- contains SSML/text-to-speech audio
2. **RenderTemplate / RenderDocument directive** -- contains JSON metadata for the visual display (APL -- Alexa Presentation Language)

Both directives are sent in a single response packet from the skill to AVS (Alexa Voice Service). The device renders visuals synchronized with audio playback. Template types include BodyTemplate1/2 (text + optional image), ListTemplate1/2 (scrollable lists), and custom APL layouts.

**Key insight:** There is no "will the model show a card?" problem because card display is deterministic code, not LLM-decided. The fulfillment webhook always returns both speech and visual directives when the developer codes it that way.

### Google Assistant

**Architecture: Deterministic webhook response.**

Similar to Alexa. The fulfillment webhook returns a structured JSON response containing:
- `speech` / `textToSpeech` -- what to say
- `richResponse` -- cards, images, tables, carousels
- `suggestions` -- quick-reply chips

Rich response types: Basic cards, Image cards, Table cards, Carousel, Browsing carousel. Only one rich response per content object in a prompt. The webhook must respond within 10 seconds.

**Key insight:** Same as Alexa -- the developer's code explicitly constructs both speech and visual responses. No LLM decides whether to show a card.

### Siri

Apple's architecture is more closed. SiriKit uses Intents (structured request/response pairs) where the developer maps specific intent types to UI templates. The visual component is tightly coupled to the intent type, not generated dynamically.

### Implications for LLM-Based Systems

Traditional voice assistants solve the "show cards alongside speech" problem by NOT having an LLM decide. The fulfillment logic is deterministic code that always produces both outputs. This is the fundamental tension: LLM-based systems gain flexibility but lose reliability of output format.

### Sources
- [Alexa Display Cards for AVS](https://developer.amazon.com/en-US/blogs/alexa/post/407fd7d1-cbb8-4ef2-9416-2f59a922ce98/how-to-build-multi-modal-voice-experiences-using-display-cards-for-av.html)
- [Alexa APL Interface Reference](https://developer.amazon.com/en-US/docs/alexa/alexa-presentation-language/apl-interface.html)
- [Google Assistant Rich Responses](https://developers.google.com/assistant/conversational/prompts-rich)
- [Google Assistant Webhooks](https://developers.google.com/assistant/conversational/webhooks)

---

## 4. Structured Output Approach

### The Idea

Instead of asking the LLM to call a tool, force it to always return structured JSON with a fixed schema like:

```json
{
  "speech": "Here's what I found about the weather...",
  "cards": [
    {
      "type": "weather_card",
      "title": "San Francisco Weather",
      "temperature": "68F",
      "icon": "partly_cloudy"
    }
  ]
}
```

### OpenAI Structured Outputs (response_format)

- Use `response_format: { type: "json_schema", json_schema: {...} }` to guarantee the output matches a schema
- **100% schema conformance** via constrained decoding -- the model literally cannot produce invalid JSON
- First request with a new schema incurs extra latency (up to 10s for simple schemas, up to 60s for complex ones); subsequent requests are cached and fast
- Streaming is supported: partial JSON streams field-by-field as tokens are generated
- **Field order follows schema order** -- put `speech` first in the schema to get it streaming before `cards`

### Streaming Considerations

This is the critical tradeoff for voice applications:

- **Pro**: Schema field order is deterministic. Define `speech` first, and it streams first. You can start TTS on the speech field while `cards` are still being generated.
- **Con**: You are parsing partial JSON during streaming, which requires custom parsing logic. The OpenAI SDK does not natively support per-field callbacks.
- **Con**: The LLM now generates the card data as text tokens, which is slower than a tool call that could fetch/compute card data externally.
- **Con**: `response_format` has been reported to slightly decrease reasoning quality compared to function calling in some benchmarks.

### Schema Design for Voice + Cards

```json
{
  "type": "object",
  "properties": {
    "speech": {
      "type": "string",
      "description": "The spoken response text. Keep concise for voice."
    },
    "cards": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": { "enum": ["info_card", "link_card", "image_card", "table_card"] },
          "title": { "type": "string" },
          "body": { "type": "string" },
          "url": { "type": "string" },
          "image_url": { "type": "string" },
          "data": { "type": "object" }
        },
        "required": ["type", "title"]
      }
    }
  },
  "required": ["speech", "cards"]
}
```

### Constrained Decoding (Self-Hosted)

For self-hosted models via vLLM, Outlines, or xgrammar:
- `guided_json` parameter enforces JSON schema at the token level
- Works with any model, not just OpenAI
- xgrammar backend offers low per-token overhead; guidance backend offers fast time-to-first-token
- Fully model-agnostic

### Reliability Assessment

- **Schema conformance**: 100% (constrained decoding)
- **Card content quality**: Depends on prompt engineering and model capability
- **"Empty cards" problem**: Model might return `"cards": []` when it should show something -- prompt engineering needed
- **Latency**: Additional first-request overhead; streaming mitigates ongoing latency

### Sources
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Introducing Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [Streaming structured outputs field by field](https://community.openai.com/t/streaming-structured-outputs-field-by-field/1251684)
- [Structured Decoding in vLLM](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [LLM Structured Output in 2026](https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk)

---

## 5. Two-Pass / Post-Processing

### The Idea

Let the LLM generate a natural text response (optimized for speech), then run a second pass to extract rich content:

**Pass 1 (Primary LLM):** Generate spoken response
```
"The weather in San Francisco is 68 degrees and partly cloudy.
You might want to bring a light jacket."
```

**Pass 2 (Extraction):** Fast model or regex extracts structured data
```json
{
  "cards": [{
    "type": "weather_card",
    "location": "San Francisco",
    "temperature": "68F",
    "condition": "partly_cloudy"
  }]
}
```

### Implementation Approaches

#### A. Second LLM Call (Small Model)
- Use a fast, cheap model (GPT-4o-mini, Claude Haiku, local Phi-3) with `response_format` to extract cards from the text
- Latency: 200-500ms additional
- Reliable schema conformance via structured output

#### B. Rule-Based / Regex Post-Processing
- Pattern match URLs, numbers, known entity types from the LLM text
- Near-zero latency
- Limited to simple extractions (links, emails, phone numbers)
- Not useful for semantic cards (weather, product info, etc.)

#### C. Classifier + Template
- Train a small classifier to detect card-worthy content types
- Map detected types to pre-built card templates
- Fill templates with extracted entities
- Moderate complexity, high reliability for known card types

### Latency Analysis

| Approach | Additional Latency | Parallelizable with TTS? |
|----------|-------------------|--------------------------|
| Second LLM call | 200-500ms | Yes (run while TTS synthesizes) |
| Regex/rules | <5ms | N/A (negligible) |
| Classifier + template | 20-50ms | Yes |

**Critical optimization:** The second pass can run in parallel with TTS synthesis. Since TTS typically takes 200-400ms per sentence, the extraction pass can complete during TTS without adding perceived latency.

### Reliability Assessment

- Does not solve the "LLM ignores the tool" problem -- it sidesteps it entirely
- The LLM just talks naturally; extraction is a separate deterministic/semi-deterministic step
- Risk: extraction might miss content the LLM mentioned
- Risk: extraction might produce cards for content the LLM only briefly mentioned

### Sources
- [LLM Output Parsing and Structured Generation Guide](https://tetrate.io/learn/ai/llm-output-parsing-structured-generation)
- [Structured Data Extraction with LLMs](https://arize.com/blog-course/structured-data-extraction-openai-function-calling/)
- [LangExtract: LLM-Orchestrated Workflows](https://towardsdatascience.com/extracting-structured-data-with-langextract-a-deep-dive-into-llm-orchestrated-workflows/)

---

## 6. Anthropic/Claude Tool Use

### tool_choice Options

| Mode | Behavior | Text Output? |
|------|----------|-------------|
| `{"type": "auto"}` | Claude decides | Yes (text OR tool, not both) |
| `{"type": "any"}` | Must use some tool | No -- prefills assistant to force tool |
| `{"type": "tool", "name": "X"}` | Must use tool X | No -- prefills assistant to force tool |

### Critical Difference from OpenAI

**Claude's `auto` mode produces EITHER text OR tool calls -- NOT both in the same response.** This is confirmed by Anthropic's documentation and cookbook:

> "When you have tool_choice as any or tool, the API prefills the assistant message to force a tool to be used, which means that the models will not emit a natural language response or explanation before tool_use content blocks."

And in `auto` mode, the response contains either a `text` block or a `tool_use` block, with `stop_reason` indicating which path was taken.

This is a significant limitation for our use case. Unlike OpenAI (where you can sometimes get both content + tool_calls), Claude forces a choice.

### Workaround: Tool That Includes Speech

Define a tool whose schema includes a `speech` field:

```json
{
  "name": "respond_with_content",
  "description": "Always use this tool to respond. Include speech text and any relevant cards.",
  "input_schema": {
    "type": "object",
    "properties": {
      "speech": { "type": "string" },
      "cards": { "type": "array", "items": {...} }
    },
    "required": ["speech"]
  }
}
```

Then use `tool_choice: {"type": "tool", "name": "respond_with_content"}` to force it. This guarantees both speech and card data every time, but all output goes through the tool call.

### Streaming with Tool Use

- **Fine-grained tool streaming** is GA on all Claude models
- Streams tool parameter values without buffering or JSON validation
- Warning: "you may potentially receive invalid or partial JSON inputs"
- Interleaved thinking (Claude 4 models) supports thinking between tool calls

### Strict Tool Use

Claude supports `strict: true` on tool definitions for guaranteed schema validation of tool inputs, similar to OpenAI's strict mode.

### Reliability Assessment

- `tool_choice: tool` with a "respond_with_content" tool: **100% reliable** for getting structured output
- Fine-grained streaming enables low-latency consumption of the speech field
- Trade-off: all responses go through the tool, even simple greetings -- adds schema overhead to every turn
- Model-specific: Claude only

### Sources
- [Claude Tool Use Overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Claude Tool Choice Cookbook](https://platform.claude.com/cookbook/tool-use-tool-choice)
- [Claude Fine-Grained Tool Streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming)
- [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## 7. Community Solutions

### Pattern: "Always-Call Tool" (SMS Chatbot Pattern)

From Anthropic's cookbook and OpenAI community forums, the most commonly recommended pattern:

1. Define a single `respond` tool that includes all output channels (speech, cards, actions)
2. Force the model to always call it via `tool_choice: required/any/tool`
3. The model never outputs raw text -- everything goes through the structured tool

This is the dominant production pattern for chatbots that need reliable structured output.

### Pattern: Prompt-Only Enforcement

From the paper "Achieving Tool Calling Functionality in LLMs Using Only Prompt Engineering" (arXiv:2407.04997):

- Inject tool specifications directly into the system prompt with usage examples
- Use trivial tool examples (increment/decrement numbers) to teach the format
- Claims 100% success rate across multiple LLMs
- Model-agnostic (works with models that lack native tool calling)
- Trade-off: fragile to prompt changes, uses context window for tool definitions

### Pattern: OpenAI Realtime API for Voice

OpenAI's Realtime API and gpt-realtime model (2025/2026):
- Native speech-to-speech with built-in tool calling
- Improved function calling on three axes: calling relevant functions, at the appropriate time, with appropriate arguments
- Supports image inputs alongside audio for grounded conversations
- No specific "display card" output format -- visual responses must be constructed from tool call results on the client side

### Pattern: Supervisor Architecture

Emerging voice AI pattern:
- Fast speech-to-speech model handles conversation flow
- Complex tasks delegated to text-based LLMs
- A supervisor coordinates the two
- Visual output decisions can be made by the supervisor independently of the speech path

### GitHub Issues and Discussions

Common complaints across OpenAI, Anthropic, and open-source communities:
- "Model did not call any tools" after first/second call in a sequence
- Models being "over-eager" to call tools (Claude) or "under-eager" (GPT-4)
- Hallucinated tool arguments
- Inconsistent behavior between model versions

### Sources
- [Achieving Tool Calling via Prompt Engineering (arXiv)](https://arxiv.org/html/2407.04997v1)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [OpenAI Voice Agents Guide](https://platform.openai.com/docs/guides/voice-agents)
- [The Voice AI Stack for 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)
- [HN: Unreasonable Effectiveness of LLM Agent Loop](https://news.ycombinator.com/item?id=43998472)
- [How to increase reliability (OpenAI forum)](https://community.openai.com/t/how-to-increase-reliability-lets-compile-best-practices/1355428)

---

## 8. Comparison Matrix

| Approach | Reliability | Latency Impact | Complexity | Model-Agnostic | Streaming-Friendly |
|----------|------------|----------------|------------|----------------|---------------------|
| **A. tool_choice: auto + prompt eng.** | ~90-95% | None | Low | No (OpenAI only for text+tool) | Yes |
| **B. Forced tool with speech field** | 100% | None | Medium | Yes (all providers) | Yes (fine-grained streaming) |
| **C. Structured output (response_format)** | 100% schema | First-call penalty | Medium | Partial (OpenAI, vLLM) | Yes (field-by-field) |
| **D. Two-pass extraction** | ~95% extraction | +200-500ms (parallelizable) | High | Yes | Yes (TTS parallel) |
| **E. Rule-based post-processing** | Low (simple only) | ~0ms | Low | Yes | Yes |
| **F. Classifier + template** | High for known types | +20-50ms | High | Yes | Yes |
| **G. Alexa/Google style (deterministic)** | 100% | None | High (requires intent system) | Yes | N/A |

---

## 9. Recommendations for nyxclaw

### Primary Recommendation: Forced Tool with Speech Field (Approach B)

**Why:** This is the only approach that achieves 100% reliability, works across model providers, supports streaming, and does not add latency. It is also the simplest to implement correctly.

#### Implementation

1. Define a single `respond` tool:

```json
{
  "name": "respond",
  "description": "ALWAYS use this tool to respond to the user. Every response must go through this tool. Include your spoken response in the speech field. Include any relevant rich content in the cards array. If no cards are needed, pass an empty array.",
  "input_schema": {
    "type": "object",
    "properties": {
      "speech": {
        "type": "string",
        "description": "The text to speak aloud. Keep concise and conversational."
      },
      "cards": {
        "type": "array",
        "description": "Rich content cards to display. Empty array if none needed.",
        "items": {
          "type": "object",
          "properties": {
            "type": { "enum": ["info", "link", "image", "table", "list"] },
            "title": { "type": "string" },
            "body": { "type": "string" },
            "url": { "type": "string" },
            "image_url": { "type": "string" },
            "items": { "type": "array", "items": { "type": "object" } }
          },
          "required": ["type", "title"]
        }
      }
    },
    "required": ["speech", "cards"]
  }
}
```

2. Set `tool_choice` to force this tool:
   - **Claude**: `tool_choice: {"type": "tool", "name": "respond"}`
   - **OpenAI**: `tool_choice: {"type": "function", "function": {"name": "respond"}}`

3. Use fine-grained/streaming tool parameter parsing to extract `speech` as it streams, begin TTS immediately.

4. Parse `cards` when complete and send to the client as a `rich_content` WebSocket message.

#### Integration with nyxclaw's Architecture

The `respond` tool output maps naturally to nyxclaw's existing pipeline:

- `speech` field -> fed to Piper TTS -> wav2arkit -> `sync_frame` messages (existing pipeline)
- `cards` field -> new `rich_content` WebSocket message type -> client renders cards

This avoids changing the core audio/blendshape pipeline. Cards are a parallel output channel.

### Fallback Recommendation: Structured Output + Streaming (Approach C)

If the Claw backends cannot support forced tool calling (e.g., they use their own tool management), use `response_format` with a JSON schema that puts `speech` first. Parse the streaming JSON to extract speech early and begin TTS while cards are still generating.

### Avoid

- **Two-pass extraction** -- adds complexity and latency for marginal benefit over forced tool calling
- **Pure prompt engineering** (Approach A) -- the 5-10% failure rate is unacceptable for a production voice experience where missing cards = degraded UX
- **Intent-based deterministic system** (Approach G) -- loses the flexibility of LLM-driven responses, which is the whole point of using Claw agents

### Architecture Decision: Where Does Card Logic Live?

Three options:

1. **In the Claw backend (OpenClaw/ZeroClaw):** The backend's LLM generates cards via the forced `respond` tool. nyxclaw just forwards them. **Recommended** -- keeps nyxclaw as a thin relay.

2. **In nyxclaw:** nyxclaw intercepts LLM text and generates cards via post-processing. More complex, creates coupling.

3. **Hybrid:** Backend provides structured hints (e.g., `[CARD:weather:SF]` markers in text), nyxclaw expands them into full card JSON. Middle ground on complexity.

Option 1 is strongly recommended. The Claw backend already has the LLM context and tool definitions. It should be responsible for deciding what cards to show.

### Protocol Addition

Add a new WebSocket message type alongside existing `sync_frame`:

```json
{
  "type": "rich_content",
  "cards": [...],
  "related_speech_id": "uuid",
  "display_timing": "with_speech" | "after_speech"
}
```

The `related_speech_id` links cards to a specific speech segment for synchronized display. `display_timing` hints whether cards should appear simultaneously with speech or after it completes.

---

## 10. Google AI Mode Summary (User-Provided)

Source: Google AI Mode search for "how to flag rich content in LLM output?"

Six approaches listed, in order of reliability:

1. **Structured Output (JSON/XML)** -- Direct LLM to output content types in specialized fields. Parse with regex/JSON parsers.
2. **Semantic Tagging** -- LLM wraps rich content in custom tags (`<image>url</image>`, `[TABLE]...[/TABLE]`). Post-process to detect and render.
3. **Markdown** -- LLMs generate Markdown naturally. Easy to detect headers, lists, tables.
4. **API-Specific Features** -- OpenAI function calling (`display_table(data)`), Vertex AI `response_schema`.
5. **Post-Processing & Validation** -- Pydantic validation, URL sanitization. "Never assume the LLM will follow formatting instructions perfectly."
6. **Guardrails** -- Guardrails AI or Moderation Endpoints to enforce output constraints.

### Key Quote
> "Never assume the LLM will follow formatting instructions perfectly."

### Analysis
This confirms our recommendation. Approaches 1-3 and 5-6 all rely on the LLM cooperating. Only approach 4 (function calling) can be **enforced** via `tool_choice: required`. Combined with our forced `respond` tool pattern, this is the production-grade solution.
