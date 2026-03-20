//! Nyxclaw avatar WebSocket channel.
//!
//! This channel handles WebSocket connections from the nyxclaw avatar client.
//! It sets a structured `response_format` on the agent so the LLM returns JSON
//! with separate `speech` and `content` fields, then streams tool-call events,
//! speech chunks (sentence-split), and rich content to the connected client.
//!
//! Protocol:
//! ```text
//! Client -> Server: {"type":"message","content":"Hello"}
//! Client -> Server: {"type":"cancel"}
//! Server -> Client: {"type":"tool_call","name":"shell","args":{...}}
//! Server -> Client: {"type":"tool_result","name":"shell","output":"...","success":true,"duration_ms":N}
//! Server -> Client: {"type":"speech_chunk","content":"Hi there!"}
//! Server -> Client: {"type":"rich_content","content":"## Details\n..."}
//! Server -> Client: {"type":"done","full_response":"..."}
//! ```

use crate::gateway::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::{header, HeaderMap},
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::debug;

/// The sub-protocol we support for the avatar WebSocket.
const WS_PROTOCOL: &str = "zeroclaw.v1";

/// Prefix used in `Sec-WebSocket-Protocol` to carry a bearer token.
const BEARER_SUBPROTO_PREFIX: &str = "bearer.";

/// Gateway session key prefix for avatar sessions.
const AVATAR_SESSION_PREFIX: &str = "avatar_";

#[derive(Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
    pub session_id: Option<String>,
}

/// Optional connection parameters sent as the first WebSocket message.
#[derive(Debug, Deserialize)]
struct ConnectParams {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    device_name: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
}

/// The avatar response_format JSON schema that instructs the LLM to return
/// structured output with `speech` and `content` fields.
fn avatar_response_format() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "avatar_response",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "speech": {
                        "type": "string",
                        "description": "Text for the avatar to speak aloud. Keep concise."
                    },
                    "content": {
                        "type": "string",
                        "description": "Rich content (markdown with URLs, tables, etc.) to display in the chat. Empty string if nothing to show."
                    }
                },
                "required": ["speech", "content"],
                "additionalProperties": false
            }
        }
    })
}

/// Extract a bearer token from WebSocket-compatible sources.
///
/// Precedence (first non-empty wins):
/// 1. `Authorization: Bearer <token>` header
/// 2. `Sec-WebSocket-Protocol: bearer.<token>` subprotocol
/// 3. `?token=<token>` query parameter
fn extract_ws_token<'a>(headers: &'a HeaderMap, query_token: Option<&'a str>) -> Option<&'a str> {
    // 1. Authorization header
    if let Some(t) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
    {
        if !t.is_empty() {
            return Some(t);
        }
    }

    // 2. Sec-WebSocket-Protocol: bearer.<token>
    if let Some(t) = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .and_then(|protos| {
            protos
                .split(',')
                .map(|p| p.trim())
                .find_map(|p| p.strip_prefix(BEARER_SUBPROTO_PREFIX))
        })
    {
        if !t.is_empty() {
            return Some(t);
        }
    }

    // 3. ?token= query parameter
    if let Some(t) = query_token {
        if !t.is_empty() {
            return Some(t);
        }
    }

    None
}

/// GET /ws/avatar — WebSocket upgrade for nyxclaw avatar client
pub async fn handle_ws_nyxclaw(
    State(state): State<AppState>,
    Query(params): Query<WsQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Auth: check header, subprotocol, then query param (precedence order)
    if state.pairing.require_pairing() {
        let token = extract_ws_token(&headers, params.token.as_deref()).unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                "Unauthorized — provide Authorization header, Sec-WebSocket-Protocol bearer, or ?token= query param",
            )
                .into_response();
        }
    }

    // Echo Sec-WebSocket-Protocol if the client requests our sub-protocol.
    let ws = if headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .map_or(false, |protos| {
            protos.split(',').any(|p| p.trim() == WS_PROTOCOL)
        }) {
        ws.protocols([WS_PROTOCOL])
    } else {
        ws
    };

    let session_id = params.session_id;
    ws.on_upgrade(move |socket| handle_avatar_socket(socket, state, session_id))
        .into_response()
}

async fn handle_avatar_socket(socket: WebSocket, state: AppState, session_id: Option<String>) {
    let (mut sender, mut receiver) = socket.split();

    // Resolve session ID: use provided or generate a new UUID
    let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let session_key = format!("{AVATAR_SESSION_PREFIX}{session_id}");

    // Build a persistent Agent for this connection so history is maintained across turns.
    let config = state.config.lock().clone();
    let mut agent = match crate::agent::Agent::from_config(&config) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Failed to initialise agent for avatar session: {e:#}");
            let err = serde_json::json!({"type": "error", "message": "Failed to initialise agent"});
            let _ = sender.send(Message::Text(err.to_string().into())).await;
            return;
        }
    };
    agent.set_memory_session_id(Some(session_id.clone()));

    // Set avatar response format so LLM returns structured {speech, content} JSON
    agent.set_response_format(Some(avatar_response_format()));

    // Hydrate agent from persisted session (if available)
    let mut resumed = false;
    let mut message_count: usize = 0;
    if let Some(ref backend) = state.session_backend {
        let messages = backend.load(&session_key);
        if !messages.is_empty() {
            message_count = messages.len();
            agent.seed_history(&messages);
            resumed = true;
        }
    }

    // Send session_start message to client
    let session_start = serde_json::json!({
        "type": "session_start",
        "session_id": session_id,
        "resumed": resumed,
        "message_count": message_count,
    });
    let _ = sender
        .send(Message::Text(session_start.to_string().into()))
        .await;

    // ── Optional connect handshake ──────────────────────────────────
    let mut first_msg_fallback: Option<String> = None;

    if let Some(first) = receiver.next().await {
        match first {
            Ok(Message::Text(text)) => {
                if let Ok(cp) = serde_json::from_str::<ConnectParams>(&text) {
                    if cp.msg_type == "connect" {
                        debug!(
                            session_id = ?cp.session_id,
                            device_name = ?cp.device_name,
                            capabilities = ?cp.capabilities,
                            "Avatar WebSocket connect params received"
                        );
                        if let Some(sid) = &cp.session_id {
                            agent.set_memory_session_id(Some(sid.clone()));
                        }
                        let ack = serde_json::json!({
                            "type": "connected",
                            "message": "Avatar connection established"
                        });
                        let _ = sender.send(Message::Text(ack.to_string().into())).await;
                    } else {
                        first_msg_fallback = Some(text.to_string());
                    }
                } else {
                    first_msg_fallback = Some(text.to_string());
                }
            }
            Ok(Message::Close(_)) | Err(_) => return,
            _ => {}
        }
    }

    // Process the first message if it was not a connect frame
    if let Some(ref text) = first_msg_fallback {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
            if parsed["type"].as_str() == Some("message") {
                let content = parsed["content"].as_str().unwrap_or("").to_string();
                if !content.is_empty() {
                    if let Some(ref backend) = state.session_backend {
                        let user_msg = crate::providers::ChatMessage::user(&content);
                        let _ = backend.append(&session_key, &user_msg);
                    }
                    process_avatar_message(&state, &mut agent, &mut sender, &content, &session_key)
                        .await;
                }
            }
        }
    }

    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) | Err(_) => break,
            _ => continue,
        };

        let parsed: serde_json::Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"type": "error", "message": "Invalid JSON"});
                let _ = sender.send(Message::Text(err.to_string().into())).await;
                continue;
            }
        };

        let msg_type = parsed["type"].as_str().unwrap_or("");

        match msg_type {
            "message" => {
                let content = parsed["content"].as_str().unwrap_or("").to_string();
                if content.is_empty() {
                    continue;
                }

                // Persist user message
                if let Some(ref backend) = state.session_backend {
                    let user_msg = crate::providers::ChatMessage::user(&content);
                    let _ = backend.append(&session_key, &user_msg);
                }

                process_avatar_message(&state, &mut agent, &mut sender, &content, &session_key)
                    .await;
            }
            "cancel" => {
                // Cancel is a no-op at the connection level; actual cancellation
                // would need a CancellationToken threaded through turn_with_events.
                // For now we acknowledge the intent.
                debug!("Avatar client requested cancel");
            }
            _ => continue,
        }
    }
}

// ── Incremental JSON extractor for `{speech, content}` ────────────────
//
// Parses streaming JSON tokens to extract the `speech` field value as it
// arrives, enabling sentence-split `speech_chunk` events mid-stream.
// The `content` field is accumulated and emitted as `rich_content` at the end.
//
// State machine:
//   PREFIX  → scanning for `"speech":"` or `"content":"`
//   SPEECH  → inside the speech string value
//   MIDDLE  → between speech and content fields
//   CONTENT → inside the content string value
//   DONE    → both fields extracted

/// Tracks the JSON parsing depth to know when we're inside a string value
/// that isn't one of our target fields. This prevents false matches on
/// `"speech":"` appearing inside another field's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractState {
    /// Outside any string value — looking for a field key.
    TopLevel,
    /// Inside a JSON string that is a field key.
    InKey,
    /// Just saw `"speech":` — expecting the opening `"` of the value.
    ExpectSpeechValue,
    /// Just saw `"content":` — expecting the opening `"` of the value.
    ExpectContentValue,
    /// Just saw an unknown key — expecting the opening `"` of its value to skip.
    ExpectOtherValue,
    /// Inside the `speech` string value — characters go to TTS.
    InSpeech,
    /// Inside the `content` string value.
    InContent,
    /// Inside some other string value (skip until closing `"`).
    InOtherValue,
    /// Both fields extracted.
    Done,
}

struct AvatarJsonExtractor {
    state: ExtractState,
    /// Accumulated speech text (unescaped).
    speech: String,
    /// Accumulated content text (unescaped).
    content: String,
    /// Buffer for the current field being read.
    value_buf: String,
    /// Current key name being accumulated.
    key_buf: String,
    /// Whether the previous character was a backslash (JSON escape).
    escaped: bool,
    /// Track which fields we've completed.
    found_speech: bool,
    found_content: bool,
}

impl AvatarJsonExtractor {
    fn new() -> Self {
        Self {
            state: ExtractState::TopLevel,
            speech: String::new(),
            content: String::new(),
            value_buf: String::new(),
            key_buf: String::new(),
            escaped: false,
            found_speech: false,
            found_content: false,
        }
    }

    /// Feed new characters from the stream. Returns any new speech text
    /// that should be fed to the sentence splitter.
    fn feed(&mut self, text: &str) -> Option<String> {
        let mut new_speech = String::new();

        for ch in text.chars() {
            match self.state {
                ExtractState::TopLevel => {
                    if ch == '"' {
                        self.key_buf.clear();
                        self.escaped = false;
                        self.state = ExtractState::InKey;
                    }
                    // Ignore other chars (braces, commas, colons, whitespace)
                }
                ExtractState::InKey => {
                    if self.escaped {
                        self.key_buf.push(ch);
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
                        // Key complete — check what field this is
                        if !self.found_speech && self.key_buf == "speech" {
                            self.state = ExtractState::ExpectSpeechValue;
                        } else if !self.found_content && self.key_buf == "content" {
                            self.state = ExtractState::ExpectContentValue;
                        } else {
                            // Unknown key — skip its string value if present.
                            self.state = ExtractState::ExpectOtherValue;
                        }
                    } else {
                        self.key_buf.push(ch);
                    }
                }
                ExtractState::ExpectSpeechValue => {
                    // Wait for the opening `"` of the string value (skip `:` and whitespace)
                    if ch == '"' {
                        self.value_buf.clear();
                        self.escaped = false;
                        self.state = ExtractState::InSpeech;
                    }
                }
                ExtractState::ExpectContentValue => {
                    if ch == '"' {
                        self.value_buf.clear();
                        self.escaped = false;
                        self.state = ExtractState::InContent;
                    }
                }
                ExtractState::ExpectOtherValue => {
                    // Wait for the opening `"` of the value (skip `:` and whitespace)
                    if ch == '"' {
                        self.escaped = false;
                        self.state = ExtractState::InOtherValue;
                    }
                }
                ExtractState::InSpeech => {
                    if self.escaped {
                        let unescaped = match ch {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '"' => '"',
                            '\\' => '\\',
                            '/' => '/',
                            // For \uXXXX, just pass through the 'u' — imperfect but
                            // rare in speech text and won't break sentence splitting.
                            _ => ch,
                        };
                        new_speech.push(unescaped);
                        self.value_buf.push(unescaped);
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
                        // End of speech value
                        self.speech = std::mem::take(&mut self.value_buf);
                        self.found_speech = true;
                        self.state = if self.found_content {
                            ExtractState::Done
                        } else {
                            ExtractState::TopLevel
                        };
                    } else {
                        new_speech.push(ch);
                        self.value_buf.push(ch);
                    }
                }
                ExtractState::InContent => {
                    if self.escaped {
                        let unescaped = match ch {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '"' => '"',
                            '\\' => '\\',
                            '/' => '/',
                            _ => ch,
                        };
                        self.value_buf.push(unescaped);
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
                        self.content = std::mem::take(&mut self.value_buf);
                        self.found_content = true;
                        self.state = if self.found_speech {
                            ExtractState::Done
                        } else {
                            ExtractState::TopLevel
                        };
                    } else {
                        self.value_buf.push(ch);
                    }
                }
                ExtractState::InOtherValue => {
                    // Skip non-target string values
                    if self.escaped {
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
                        self.state = ExtractState::TopLevel;
                    }
                }
                ExtractState::Done => {}
            }
        }

        if new_speech.is_empty() {
            None
        } else {
            Some(new_speech)
        }
    }

    /// Finalize: if the stream ended mid-field, capture remaining value.
    fn finalize(&mut self) {
        match self.state {
            ExtractState::InSpeech => {
                self.speech = std::mem::take(&mut self.value_buf);
                self.found_speech = true;
            }
            ExtractState::InContent => {
                self.content = std::mem::take(&mut self.value_buf);
                self.found_content = true;
            }
            _ => {}
        }
    }
}

/// Route a single streaming event from the agent turn to the WebSocket client.
///
/// Handles `tool_call`/`tool_result` (forwarded), `content_delta` (fed into
/// the JSON extractor → sentence splitter → `speech_chunk`), and `content_done`
/// (flushes the sentence buffer).
async fn dispatch_stream_event(
    event: &serde_json::Value,
    extractor: &mut AvatarJsonExtractor,
    sentence_buf: &mut String,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) {
    let Some(obj) = event.as_object() else { return };
    let Some(t) = obj.get("type").and_then(|v| v.as_str()) else { return };

    match t {
        "tool_call" | "tool_result" => {
            let _ = sender.send(Message::Text(event.to_string().into())).await;
        }
        "content_delta" => {
            if let Some(delta) = obj.get("content").and_then(|v| v.as_str()) {
                if let Some(new_speech) = extractor.feed(delta) {
                    sentence_buf.push_str(&new_speech);
                    let sentences = extract_complete_sentences(sentence_buf);
                    for sentence in sentences {
                        if !sentence.is_empty() {
                            let chunk = serde_json::json!({
                                "type": "speech_chunk",
                                "content": sentence,
                            });
                            let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                        }
                    }
                }
            }
        }
        "content_done" => {
            extractor.finalize();
            let remaining = sentence_buf.trim().to_string();
            if !remaining.is_empty() {
                let chunk = serde_json::json!({
                    "type": "speech_chunk",
                    "content": remaining,
                });
                let _ = sender.send(Message::Text(chunk.to_string().into())).await;
            }
            sentence_buf.clear();
        }
        _ => {}
    }
}

/// Process a single avatar message through the agent, streaming events back.
///
/// Uses `turn_with_streaming` to receive content deltas incrementally, then
/// runs them through `AvatarJsonExtractor` to split speech from rich content
/// in real time.
async fn process_avatar_message(
    state: &AppState,
    agent: &mut crate::agent::Agent,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    content: &str,
    session_key: &str,
) {
    let provider_label = state
        .config
        .lock()
        .default_provider
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    // Broadcast agent_start event
    let _ = state.event_tx.send(serde_json::json!({
        "type": "agent_start",
        "provider": provider_label,
        "model": state.model,
        "channel": "avatar",
    }));

    // Create event channel for streaming turn
    let (event_tx, mut event_rx) = mpsc::channel::<serde_json::Value>(64);

    // Incremental JSON extractor and sentence buffer
    let mut extractor = AvatarJsonExtractor::new();
    let mut sentence_buf = String::new();

    // Run turn_with_streaming and event forwarding concurrently via select loop.
    let turn_handle = {
        let final_response: Option<Result<String, anyhow::Error>>;

        let turn_fut = agent.turn_with_streaming(content, event_tx);
        tokio::pin!(turn_fut);

        loop {
            tokio::select! {
                result = &mut turn_fut => {
                    final_response = Some(result);
                    break;
                }
                Some(event) = event_rx.recv() => {
                    dispatch_stream_event(
                        &event, &mut extractor, &mut sentence_buf, sender,
                    ).await;
                }
            }
        }

        // Drain remaining events
        while let Ok(event) = event_rx.try_recv() {
            dispatch_stream_event(
                &event, &mut extractor, &mut sentence_buf, sender,
            ).await;
        }

        match final_response {
            Some(r) => r,
            None => Err(anyhow::anyhow!("Agent turn did not produce a response")),
        }
    };

    match turn_handle {
        Ok(response) => {
            // Persist assistant response (raw JSON for history)
            if let Some(ref backend) = state.session_backend {
                let assistant_msg = crate::providers::ChatMessage::assistant(&response);
                let _ = backend.append(session_key, &assistant_msg);
            }

            // Finalize extractor (in case it wasn't finalized during drain)
            extractor.finalize();

            // Fallback: if the streaming extractor didn't find the fields
            // (e.g., provider fell back to non-streaming), parse the full response.
            if !extractor.found_speech && !extractor.found_content {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                    let speech = parsed["speech"].as_str().unwrap_or("");
                    let rich = parsed["content"].as_str().unwrap_or("");
                    if !speech.is_empty() {
                        for sentence in split_sentences(speech) {
                            if !sentence.is_empty() {
                                let chunk = serde_json::json!({
                                    "type": "speech_chunk",
                                    "content": sentence,
                                });
                                let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                            }
                        }
                    }
                    extractor.speech = speech.to_string();
                    extractor.content = rich.to_string();
                } else {
                    // Not JSON at all — raw text fallback
                    for sentence in split_sentences(&response) {
                        if !sentence.is_empty() {
                            let chunk = serde_json::json!({
                                "type": "speech_chunk",
                                "content": sentence,
                            });
                            let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                        }
                    }
                    extractor.speech = response.clone();
                }
            }

            // Send rich content if non-empty
            if !extractor.content.is_empty() {
                let rc = serde_json::json!({
                    "type": "rich_content",
                    "content": extractor.content,
                });
                let _ = sender.send(Message::Text(rc.to_string().into())).await;
            }

            // Send done with speech text only
            let speech_text = if extractor.speech.is_empty() {
                response.clone()
            } else {
                extractor.speech.clone()
            };
            let done = serde_json::json!({
                "type": "done",
                "full_response": speech_text,
            });
            let _ = sender.send(Message::Text(done.to_string().into())).await;

            // Broadcast agent_end event
            let _ = state.event_tx.send(serde_json::json!({
                "type": "agent_end",
                "provider": provider_label,
                "model": state.model,
                "channel": "avatar",
            }));
        }
        Err(e) => {
            let sanitized = crate::providers::sanitize_api_error(&e.to_string());
            let err = serde_json::json!({
                "type": "error",
                "message": sanitized,
            });
            let _ = sender.send(Message::Text(err.to_string().into())).await;

            // Broadcast error event
            let _ = state.event_tx.send(serde_json::json!({
                "type": "error",
                "component": "ws_avatar",
                "message": sanitized,
            }));
        }
    }
}

/// Extract complete sentences from a buffer, leaving the remainder.
/// Splits at `.`, `!`, `?` followed by a space or end of string.
fn extract_complete_sentences(buf: &mut String) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut last_split = 0;

    let chars: Vec<char> = buf.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        if (ch == '.' || ch == '!' || ch == '?')
            && (i + 1 >= chars.len() || chars[i + 1] == ' ' || chars[i + 1] == '\n')
        {
            let sentence: String = chars[last_split..=i].iter().collect();
            let trimmed = sentence.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            last_split = i + 1;
        }
    }

    // Keep the remainder in the buffer
    *buf = chars[last_split..].iter().collect();

    sentences
}

/// Split text into sentences at `.`, `!`, `?` boundaries, keeping the
/// delimiter attached to the preceding sentence. Trims whitespace.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remainder (no trailing punctuation)
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}
