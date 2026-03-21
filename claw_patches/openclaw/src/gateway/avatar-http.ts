/**
 * Avatar SSE endpoint for nyxclaw voice+avatar integration.
 *
 * POST /v1/chat/completions/avatar
 *
 * Accepts the same request format as /v1/chat/completions but:
 * - Injects extraSystemPrompt to force {speech, content} JSON responses
 * - Emits custom SSE event types: speech_chunk, rich_content, tool_call, tool_result, done
 * - Incrementally extracts the "speech" field as the LLM streams tokens
 */

import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createDefaultDeps } from "../cli/deps.js";
import { agentCommandFromIngress } from "../commands/agent.js";
import type { ImageContent } from "../commands/agent/types.js";
import type { GatewayHttpChatCompletionsConfig } from "../config/types.gateway.js";
import { emitAgentEvent, onAgentEvent } from "../infra/agent-events.js";
import { logWarn } from "../logger.js";
import { estimateBase64DecodedBytes } from "../media/base64.js";
import {
  DEFAULT_INPUT_IMAGE_MAX_BYTES,
  DEFAULT_INPUT_IMAGE_MIMES,
  DEFAULT_INPUT_MAX_REDIRECTS,
  DEFAULT_INPUT_TIMEOUT_MS,
  extractImageContentFromSource,
  normalizeMimeList,
  type InputImageLimits,
  type InputImageSource,
} from "../media/input-files.js";
import { defaultRuntime } from "../runtime.js";
import { resolveAssistantStreamDeltaText } from "./agent-event-assistant-text.js";
import {
  buildAgentMessageFromConversationEntries,
  type ConversationEntry,
} from "./agent-prompt.js";
import type { AuthRateLimiter } from "./auth-rate-limit.js";
import type { ResolvedGatewayAuth } from "./auth.js";
import { sendJson, setSseHeaders, writeDone } from "./http-common.js";
import { handleGatewayPostJsonEndpoint } from "./http-endpoint-helpers.js";
import { resolveGatewayRequestContext } from "./http-utils.js";
import { normalizeInputHostnameAllowlist } from "./input-allowlist.js";

// ── Types ────────────────────────────────────────────────────────────────

type AvatarHttpOptions = {
  auth: ResolvedGatewayAuth;
  config?: GatewayHttpChatCompletionsConfig;
  maxBodyBytes?: number;
  trustedProxies?: string[];
  allowRealIpFallback?: boolean;
  rateLimiter?: AuthRateLimiter;
};

type OpenAiChatMessage = {
  role?: unknown;
  content?: unknown;
  name?: unknown;
};

type OpenAiChatCompletionRequest = {
  model?: unknown;
  stream?: unknown;
  messages?: unknown;
  user?: unknown;
};

// ── Constants ────────────────────────────────────────────────────────────

const DEFAULT_BODY_BYTES = 20 * 1024 * 1024;
const IMAGE_ONLY_USER_MESSAGE = "User sent image(s) with no text.";
const DEFAULT_MAX_IMAGE_PARTS = 8;
const DEFAULT_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024;
const DEFAULT_IMAGE_LIMITS: InputImageLimits = {
  allowUrl: false,
  allowedMimes: new Set(DEFAULT_INPUT_IMAGE_MIMES),
  maxBytes: DEFAULT_INPUT_IMAGE_MAX_BYTES,
  maxRedirects: DEFAULT_INPUT_MAX_REDIRECTS,
  timeoutMs: DEFAULT_INPUT_TIMEOUT_MS,
};

/**
 * System prompt injected via extraSystemPrompt to force JSON output.
 * Appended after the agent's own system prompt.
 */
const AVATAR_RESPONSE_FORMAT_PROMPT = `
## Response format

Your responses are consumed by a voice + avatar system. Every response you generate MUST be a valid JSON object with exactly two fields:

\`\`\`json
{"speech": "...", "content": "..."}
\`\`\`

### \`speech\` — what the avatar says aloud
- Keep it concise and conversational — this is spoken, not read.
- Never include URLs, table data, code, or markdown syntax in speech.
- When you have rich content to show, use a brief phrase: "Check this out", "Here's what I found", "Take a look."
- For simple conversational responses (greetings, opinions, short answers), just put the full response in speech.

### \`content\` — what appears in the chat (rich content)
- Put URLs, links, tables, code snippets, structured data, and detailed information here.
- Use markdown formatting — the app renders it.
- Set to empty string \`""\` when there's nothing visual to show.

### Rules
- Output ONLY the JSON object. No text before or after.
- Never wrap the JSON in markdown code fences.
- Both fields must always be present.
- The \`speech\` field must always have content (never empty).
`.trim();

// ── Limits ───────────────────────────────────────────────────────────────

type ResolvedLimits = {
  maxBodyBytes: number;
  maxImageParts: number;
  maxTotalImageBytes: number;
  images: InputImageLimits;
};

function resolveLimits(config: GatewayHttpChatCompletionsConfig | undefined): ResolvedLimits {
  const imageConfig = config?.images;
  return {
    maxBodyBytes: config?.maxBodyBytes ?? DEFAULT_BODY_BYTES,
    maxImageParts:
      typeof config?.maxImageParts === "number"
        ? Math.max(0, Math.floor(config.maxImageParts))
        : DEFAULT_MAX_IMAGE_PARTS,
    maxTotalImageBytes:
      typeof config?.maxTotalImageBytes === "number"
        ? Math.max(1, Math.floor(config.maxTotalImageBytes))
        : DEFAULT_MAX_TOTAL_IMAGE_BYTES,
    images: {
      allowUrl: imageConfig?.allowUrl ?? DEFAULT_IMAGE_LIMITS.allowUrl,
      urlAllowlist: normalizeInputHostnameAllowlist(imageConfig?.urlAllowlist),
      allowedMimes: normalizeMimeList(imageConfig?.allowedMimes, DEFAULT_INPUT_IMAGE_MIMES),
      maxBytes: imageConfig?.maxBytes ?? DEFAULT_INPUT_IMAGE_MAX_BYTES,
      maxRedirects: imageConfig?.maxRedirects ?? DEFAULT_INPUT_MAX_REDIRECTS,
      timeoutMs: imageConfig?.timeoutMs ?? DEFAULT_INPUT_TIMEOUT_MS,
    },
  };
}

// ── Request parsing (mirrors openai-http.ts) ─────────────────────────────

function coerceRequest(body: unknown): OpenAiChatCompletionRequest {
  if (body && typeof body === "object") {
    return body as OpenAiChatCompletionRequest;
  }
  return {};
}

function asMessages(val: unknown): OpenAiChatMessage[] {
  return Array.isArray(val) ? (val as OpenAiChatMessage[]) : [];
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (!part || typeof part !== "object") return "";
        const type = (part as { type?: unknown }).type;
        const text = (part as { text?: unknown }).text;
        const inputText = (part as { input_text?: unknown }).input_text;
        if (type === "text" && typeof text === "string") return text;
        if (type === "input_text" && typeof text === "string") return text;
        if (typeof inputText === "string") return inputText;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function resolveImageUrlPart(part: unknown): string | undefined {
  if (!part || typeof part !== "object") return undefined;
  const imageUrl = (part as { image_url?: unknown }).image_url;
  if (typeof imageUrl === "string") {
    const trimmed = imageUrl.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  if (!imageUrl || typeof imageUrl !== "object") return undefined;
  const rawUrl = (imageUrl as { url?: unknown }).url;
  if (typeof rawUrl !== "string") return undefined;
  const trimmed = rawUrl.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function extractImageUrls(content: unknown): string[] {
  if (!Array.isArray(content)) return [];
  const urls: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if ((part as { type?: unknown }).type !== "image_url") continue;
    const url = resolveImageUrlPart(part);
    if (url) urls.push(url);
  }
  return urls;
}

type ActiveTurnContext = {
  activeUserMessageIndex: number;
  activeUserMessage: OpenAiChatMessage | undefined;
};

function resolveActiveTurnContext(messages: OpenAiChatMessage[]): ActiveTurnContext {
  let lastUserIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      lastUserIndex = i;
      break;
    }
  }
  return {
    activeUserMessageIndex: lastUserIndex,
    activeUserMessage: lastUserIndex >= 0 ? messages[lastUserIndex] : undefined,
  };
}

function buildAgentPrompt(
  messagesUnknown: unknown,
  activeUserMessageIndex: number,
): { message: string; extraSystemPrompt?: string } {
  const messages = asMessages(messagesUnknown);
  const systemParts: string[] = [];
  const conversationEntries: ConversationEntry[] = [];

  for (const [i, msg] of messages.entries()) {
    if (!msg || typeof msg !== "object") continue;
    const role = typeof msg.role === "string" ? String(msg.role).trim() : "";
    const content = extractTextContent(msg.content).trim();
    const hasImage = extractImageUrls(msg.content).length > 0;
    if (!role) continue;

    if (role === "system" || role === "developer") {
      if (content) systemParts.push(content);
      continue;
    }

    const normalizedRole = role === "function" ? "tool" : role;
    if (normalizedRole !== "user" && normalizedRole !== "assistant" && normalizedRole !== "tool") {
      continue;
    }

    const messageContent =
      normalizedRole === "user" && !content && hasImage && i === activeUserMessageIndex
        ? IMAGE_ONLY_USER_MESSAGE
        : content;
    if (!messageContent) continue;

    const name = typeof msg.name === "string" ? String(msg.name).trim() : "";
    const sender =
      normalizedRole === "assistant"
        ? "Assistant"
        : normalizedRole === "user"
          ? "User"
          : name
            ? `Tool:${name}`
            : "Tool";

    conversationEntries.push({
      role: normalizedRole as "user" | "assistant" | "tool",
      entry: { sender, body: messageContent },
    });
  }

  const message = buildAgentMessageFromConversationEntries(conversationEntries);
  return {
    message,
    extraSystemPrompt: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
  };
}

async function resolveImagesForRequest(
  activeTurn: ActiveTurnContext,
  limits: ResolvedLimits,
): Promise<ImageContent[]> {
  const msg = activeTurn.activeUserMessage;
  if (!msg) return [];
  const content = msg.content;
  if (!Array.isArray(content)) return [];

  const imageSources: InputImageSource[] = [];
  for (const part of content as Array<{ type?: string; image_url?: { url?: string } }>) {
    if (part?.type === "image_url" && part.image_url?.url) {
      imageSources.push({ url: part.image_url.url });
    }
  }

  if (imageSources.length === 0) return [];
  if (imageSources.length > limits.maxImageParts) {
    throw new Error(`Too many image parts (${imageSources.length} > ${limits.maxImageParts}).`);
  }

  const images: ImageContent[] = [];
  let totalBytes = 0;
  for (const source of imageSources) {
    if (source.url.startsWith("data:")) {
      const estBytes = estimateBase64DecodedBytes(source.url);
      totalBytes += estBytes;
    }
    if (totalBytes > limits.maxTotalImageBytes) {
      throw new Error("Total image data exceeds size limit.");
    }
    const result = await extractImageContentFromSource(source, limits.images);
    if (result) {
      images.push(result);
    }
  }
  return images;
}

// ── Incremental JSON speech extractor ────────────────────────────────────

/**
 * Extracts the "speech" field value incrementally from a streaming JSON response.
 * As the LLM generates tokens, we accumulate text and extract complete sentences
 * from the speech field without waiting for the full response.
 */
class AvatarJsonExtractor {
  private buffer = "";
  private speechExtracted = 0;
  private inSpeech = false;
  private speechStarted = false;

  /** Feed a new text delta from the LLM. Returns any new complete sentences from speech. */
  feed(delta: string): string[] {
    this.buffer += delta;
    const sentences: string[] = [];

    if (!this.speechStarted) {
      // Look for the start of the speech field value
      const match = this.buffer.match(/"speech"\s*:\s*"/);
      if (match) {
        this.speechStarted = true;
        this.inSpeech = true;
        this.speechExtracted = match.index! + match[0].length;
      } else {
        return sentences;
      }
    }

    if (!this.inSpeech) {
      return sentences;
    }

    // Scan forward from where we left off, looking for sentence boundaries
    let i = this.speechExtracted;
    while (i < this.buffer.length) {
      const ch = this.buffer[i];

      // Handle escape sequences
      if (ch === "\\") {
        i += 2; // skip escaped char
        continue;
      }

      // End of speech string
      if (ch === '"') {
        // Flush remaining text
        const remaining = this.unescapeJson(this.buffer.slice(this.speechExtracted, i));
        if (remaining.trim()) {
          sentences.push(remaining.trim());
        }
        this.inSpeech = false;
        this.speechExtracted = i + 1;
        break;
      }

      // Sentence boundary: .!? followed by space, quote, or end
      if (".!?".includes(ch)) {
        const next = i + 1 < this.buffer.length ? this.buffer[i + 1] : undefined;
        if (next === undefined) {
          // Need more data to decide
          break;
        }
        if (next === " " || next === '"' || next === "\\") {
          const sentence = this.unescapeJson(this.buffer.slice(this.speechExtracted, i + 1));
          if (sentence.trim()) {
            sentences.push(sentence.trim());
          }
          this.speechExtracted = i + 1;
          // Skip whitespace
          if (next === " ") {
            this.speechExtracted = i + 2;
            i += 2;
            continue;
          }
        }
      }

      i++;
    }

    return sentences;
  }

  /** After the stream ends, extract speech and content from the full accumulated text. */
  finalize(): { speech: string; content: string } | null {
    const text = this.buffer.trim();
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.speech === "string") {
        return {
          speech: parsed.speech,
          content: typeof parsed.content === "string" ? parsed.content : "",
        };
      }
    } catch {
      // JSON parse failed — try to salvage
    }

    // Fallback: treat entire response as speech
    return { speech: text, content: "" };
  }

  private unescapeJson(s: string): string {
    try {
      return JSON.parse(`"${s}"`);
    } catch {
      return s.replace(/\\n/g, " ").replace(/\\"/g, '"').replace(/\\\\/g, "\\");
    }
  }
}

// ── SSE helpers ──────────────────────────────────────────────────────────

function writeAvatarEvent(res: ServerResponse, event: string, data: unknown) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// ── Main handler ─────────────────────────────────────────────────────────

export async function handleAvatarHttpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts: AvatarHttpOptions,
): Promise<boolean> {
  const limits = resolveLimits(opts.config);
  const handled = await handleGatewayPostJsonEndpoint(req, res, {
    pathname: "/v1/chat/completions/avatar",
    auth: opts.auth,
    trustedProxies: opts.trustedProxies,
    allowRealIpFallback: opts.allowRealIpFallback,
    rateLimiter: opts.rateLimiter,
    maxBodyBytes: opts.maxBodyBytes ?? limits.maxBodyBytes,
  });
  if (handled === false) {
    return false;
  }
  if (!handled) {
    return true;
  }

  const payload = coerceRequest(handled.body);
  const model = typeof payload.model === "string" ? payload.model : "openclaw";
  const user = typeof payload.user === "string" ? payload.user : undefined;

  const { sessionKey, messageChannel } = resolveGatewayRequestContext({
    req,
    model,
    user,
    sessionPrefix: "openai",
    defaultMessageChannel: "avatar",
    useMessageChannelHeader: true,
  });
  const messages = asMessages(payload.messages);
  const activeTurnContext = resolveActiveTurnContext(messages);
  const prompt = buildAgentPrompt(messages, activeTurnContext.activeUserMessageIndex);
  let images: ImageContent[] = [];
  try {
    images = await resolveImagesForRequest(activeTurnContext, limits);
  } catch (err) {
    logWarn(`avatar: invalid image_url content: ${String(err)}`);
    sendJson(res, 400, {
      error: {
        message: "Invalid image_url content in `messages`.",
        type: "invalid_request_error",
      },
    });
    return true;
  }

  if (!prompt.message && images.length === 0) {
    sendJson(res, 400, {
      error: {
        message: "Missing user message in `messages`.",
        type: "invalid_request_error",
      },
    });
    return true;
  }

  const runId = `avatar_${randomUUID()}`;
  const deps = createDefaultDeps();

  // Merge the avatar response format prompt with any existing extraSystemPrompt
  const combinedExtraSystemPrompt = [
    prompt.extraSystemPrompt?.trim(),
    AVATAR_RESPONSE_FORMAT_PROMPT,
  ]
    .filter(Boolean)
    .join("\n\n");

  const commandInput = {
    message: prompt.message,
    extraSystemPrompt: combinedExtraSystemPrompt,
    images: images.length > 0 ? images : undefined,
    sessionKey,
    runId,
    deliver: false as const,
    messageChannel,
    bestEffortDeliver: false as const,
    senderIsOwner: true as const,
  };

  // Always stream for avatar endpoint
  setSseHeaders(res);

  const extractor = new AvatarJsonExtractor();
  let closed = false;
  const toolTimings = new Map<string, number>();

  const unsubscribe = onAgentEvent((evt) => {
    if (evt.runId !== runId || closed) return;

    // ── Assistant text deltas ──
    if (evt.stream === "assistant") {
      const content = resolveAssistantStreamDeltaText(evt) ?? "";
      if (!content) return;

      const sentences = extractor.feed(content);
      for (const sentence of sentences) {
        writeAvatarEvent(res, "speech_chunk", { content: sentence });
      }
      return;
    }

    // ── Tool events ──
    if (evt.stream === "tool") {
      const data = evt.data as {
        phase?: string;
        name?: string;
        toolCallId?: string;
        isError?: boolean;
      };
      if (data.phase === "start") {
        toolTimings.set(data.toolCallId ?? "", Date.now());
        writeAvatarEvent(res, "tool_call", {
          name: data.name,
          tool_call_id: data.toolCallId,
        });
      } else if (data.phase === "end") {
        const startTime = toolTimings.get(data.toolCallId ?? "");
        const durationMs = startTime ? Date.now() - startTime : undefined;
        toolTimings.delete(data.toolCallId ?? "");
        writeAvatarEvent(res, "tool_result", {
          name: data.name,
          tool_call_id: data.toolCallId,
          success: !data.isError,
          duration_ms: durationMs,
        });
      }
      return;
    }

    // ── Lifecycle events ──
    if (evt.stream === "lifecycle") {
      const phase = (evt.data as { phase?: string })?.phase;
      if (phase === "end" || phase === "error") {
        closed = true;
        unsubscribe();

        const result = extractor.finalize();
        if (result) {
          if (result.content) {
            writeAvatarEvent(res, "rich_content", { content: result.content });
          }
          writeAvatarEvent(res, "done", { full_response: result.speech });
        } else {
          writeAvatarEvent(res, "done", { full_response: "" });
        }

        writeDone(res);
        res.end();
      }
    }
  });

  req.on("close", () => {
    closed = true;
    unsubscribe();
  });

  void (async () => {
    try {
      const result = await agentCommandFromIngress(commandInput, defaultRuntime, deps);

      if (closed) return;

      // If no assistant deltas were streamed (non-streaming LLM), handle the full response
      const fullText = extractor["buffer"];
      if (!fullText) {
        // No streaming happened — resolve from result payloads
        const payloads = Array.isArray(result) ? result : result ? [result] : [];
        const text = payloads
          .map((p: { text?: string }) => (typeof p.text === "string" ? p.text : ""))
          .filter(Boolean)
          .join("\n\n");

        if (text) {
          // Try to parse as JSON
          let speech = text;
          let content = "";
          try {
            const parsed = JSON.parse(text);
            if (parsed && typeof parsed.speech === "string") {
              speech = parsed.speech;
              content = typeof parsed.content === "string" ? parsed.content : "";
            }
          } catch {
            // Use raw text as speech
          }

          writeAvatarEvent(res, "speech_chunk", { content: speech });
          if (content) {
            writeAvatarEvent(res, "rich_content", { content });
          }
          writeAvatarEvent(res, "done", { full_response: speech });
          writeDone(res);
          res.end();
        }
      }
      // If streaming did happen, lifecycle.end handler already closed the response
    } catch (err) {
      logWarn(`avatar: chat completion failed: ${String(err)}`);
      if (closed) return;

      writeAvatarEvent(res, "done", {
        full_response: "",
        error: "Internal error",
      });
      writeDone(res);
      res.end();

      emitAgentEvent({
        runId,
        stream: "lifecycle",
        data: { phase: "error" },
      });
    }
  })();

  return true;
}
