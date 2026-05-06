//! Nyxclaw avatar WebSocket channel.
//!
//! Bridges the nyxclaw avatar client to the runtime agent. Sets a structured
//! `response_format` so the LLM returns `{speech, content}` JSON, then streams
//! tool-call events, sentence-split speech chunks, and rich content to the
//! client. Supports barge-in: a `{"type":"cancel"}` or new `{"type":"message"}`
//! arriving mid-turn cancels the in-flight agent response.
//!
//! Connect: `ws://host:port/ws/avatar?session_id=ID`
//!
//! Protocol:
//! ```text
//! Server -> Client: {"type":"session_start","session_id":"...","resumed":true,"message_count":42}
//! Client -> Server: {"type":"connect","session_id":"...","device_name":"...","capabilities":[...]}
//! Server -> Client: {"type":"connected","message":"Avatar connection established"}
//! Client -> Server: {"type":"message","content":"Hello"}
//! Server -> Client: {"type":"speech_chunk","content":"I'm checking your calendar.","filler":true}
//! Server -> Client: {"type":"tool_call","id":"...","name":"shell","args":{...}}
//! Server -> Client: {"type":"tool_result","id":"...","name":"shell","output":"..."}
//! Server -> Client: {"type":"speech_chunk","content":"Here is the result."}
//! Server -> Client: {"type":"rich_content","content":"## Details\n..."}
//! Server -> Client: {"type":"done","full_response":"..."}
//! Client -> Server: {"type":"cancel"}
//! Server -> Client: {"type":"done","full_response":"","cancelled":true}
//! ```

use super::AppState;
use axum::{
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, header},
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tracing::debug;

/// The sub-protocol we support for the avatar WebSocket.
const WS_PROTOCOL: &str = "zeroclaw.v1";

/// Prefix used in `Sec-WebSocket-Protocol` to carry a bearer token.
const BEARER_SUBPROTO_PREFIX: &str = "bearer.";

/// Session key prefix to namespace avatar sessions away from the chat
/// gateway's `gw_` sessions in the shared session backend.
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

/// JSON schema for the avatar's structured response. The LLM emits a single
/// JSON object with two string fields:
/// - `speech` — concise text the avatar speaks aloud
/// - `content` — markdown-rich text shown in the chat panel (tables, links,
///   formatting), or empty when nothing extra needs to be displayed
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
    if let Some(t) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
        && !t.is_empty()
    {
        return Some(t);
    }

    if let Some(t) = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .and_then(|protos| {
            protos
                .split(',')
                .map(|p| p.trim())
                .find_map(|p| p.strip_prefix(BEARER_SUBPROTO_PREFIX))
        })
        && !t.is_empty()
    {
        return Some(t);
    }

    if let Some(t) = query_token
        && !t.is_empty()
    {
        return Some(t);
    }

    None
}

/// GET /ws/avatar — WebSocket upgrade for the nyxclaw avatar client.
pub async fn handle_ws_nyxclaw(
    State(state): State<AppState>,
    Query(params): Query<WsQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
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

    let ws = if headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|protos| protos.split(',').any(|p| p.trim() == WS_PROTOCOL))
    {
        ws.protocols([WS_PROTOCOL])
    } else {
        ws
    };

    let session_id = params.session_id;
    ws.on_upgrade(move |socket| handle_avatar_socket(socket, state, session_id))
        .into_response()
}

async fn handle_avatar_socket(
    socket: WebSocket,
    state: AppState,
    session_id: Option<String>,
) {
    let (mut sender, mut receiver) = socket.split();

    let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let session_key = format!("{AVATAR_SESSION_PREFIX}{session_id}");

    let config = state.config.lock().clone();
    let mut agent = match zeroclaw_runtime::agent::Agent::from_config(&config).await {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "Avatar agent initialization failed");
            let err = serde_json::json!({
                "type": "error",
                "message": format!("Failed to initialise agent: {e}"),
                "code": "AGENT_INIT_FAILED"
            });
            let _ = sender.send(Message::Text(err.to_string().into())).await;
            let _ = sender
                .send(Message::Close(Some(axum::extract::ws::CloseFrame {
                    code: 1011,
                    reason: axum::extract::ws::Utf8Bytes::from_static(
                        "Agent initialization failed",
                    ),
                })))
                .await;
            return;
        }
    };
    agent.set_memory_session_id(Some(session_id.clone()));

    // Force structured `{speech, content}` JSON output for every turn.
    agent.set_response_format(Some(avatar_response_format()));

    // Strip the per-second datetime section so the cached system-prompt
    // prefix stays stable across turns (cache-hit ratio jumps from ~0% to
    // ~95% on OpenAI's automatic prompt cache).
    agent.set_prompt_builder(
        zeroclaw_runtime::agent::prompt::SystemPromptBuilder::with_defaults()
            .without_section("datetime"),
    );

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

    let session_start = serde_json::json!({
        "type": "session_start",
        "session_id": session_id,
        "resumed": resumed,
        "message_count": message_count,
    });
    let _ = sender
        .send(Message::Text(session_start.to_string().into()))
        .await;

    // Optional connect handshake — first frame may be `{"type":"connect",...}`
    // carrying device metadata. Anything else falls through to the message loop.
    let mut first_msg_fallback: Option<String> = None;
    if let Some(first) = receiver.next().await {
        match first {
            Ok(Message::Text(text)) => {
                if let Ok(cp) = serde_json::from_str::<ConnectParams>(&text)
                    && cp.msg_type == "connect"
                {
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
            }
            Ok(Message::Close(_)) | Err(_) => return,
            _ => {}
        }
    }

    if let Some(ref text) = first_msg_fallback
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text)
        && parsed["type"].as_str() == Some("message")
    {
        let content = parsed["content"].as_str().unwrap_or("").to_string();
        if !content.is_empty() {
            if let Some(ref backend) = state.session_backend {
                let user_msg = zeroclaw_providers::ChatMessage::user(&content);
                let _ = backend.append(&session_key, &user_msg);
            }
            run_turn(&state, &mut agent, &mut sender, &mut receiver, &content, &session_key).await;
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

        match parsed["type"].as_str().unwrap_or("") {
            "message" => {
                let content = parsed["content"].as_str().unwrap_or("").to_string();
                if content.is_empty() {
                    continue;
                }

                if let Some(ref backend) = state.session_backend {
                    let user_msg = zeroclaw_providers::ChatMessage::user(&content);
                    let _ = backend.append(&session_key, &user_msg);
                }

                let mut next_content = Some(content);
                while let Some(c) = next_content.take() {
                    next_content = run_turn(
                        &state,
                        &mut agent,
                        &mut sender,
                        &mut receiver,
                        &c,
                        &session_key,
                    )
                    .await;

                    // If barge-in queued a new user message, persist it before
                    // looping back so history stays coherent.
                    if let Some(ref queued) = next_content
                        && let Some(ref backend) = state.session_backend
                    {
                        let user_msg = zeroclaw_providers::ChatMessage::user(queued);
                        let _ = backend.append(&session_key, &user_msg);
                    }
                }
            }
            "cancel" => {
                debug!("Avatar client requested cancel (no turn active)");
            }
            _ => continue,
        }
    }
}

/// Run a single agent turn while concurrently watching the receiver for
/// barge-in. Returns `Some(queued_content)` if a new user message arrived
/// mid-turn (the caller should run another turn for that content), or `None`
/// for normal completion.
async fn run_turn(
    state: &AppState,
    agent: &mut zeroclaw_runtime::agent::Agent,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    receiver: &mut futures_util::stream::SplitStream<WebSocket>,
    content: &str,
    session_key: &str,
) -> Option<String> {
    use zeroclaw_runtime::agent::TurnEvent;

    let provider_label = state
        .config
        .lock()
        .providers
        .fallback
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    let _ = state.event_tx.send(serde_json::json!({
        "type": "agent_start",
        "provider": provider_label,
        "model": state.model,
        "channel": "avatar",
    }));

    let turn_id = uuid::Uuid::new_v4().to_string();
    if let Some(ref backend) = state.session_backend {
        let _ = backend.set_session_state(session_key, "running", Some(&turn_id));
    }

    let cancel_token = tokio_util::sync::CancellationToken::new();
    {
        state
            .cancel_tokens
            .lock()
            .expect("cancel_tokens lock poisoned")
            .insert(session_key.to_string(), cancel_token.clone());
    }

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<TurnEvent>(64);

    let content_owned = content.to_string();
    let session_key_owned = session_key.to_string();
    let turn_cancel = cancel_token.clone();
    let turn_fut = async move {
        zeroclaw_runtime::agent::loop_::scope_session_key(
            Some(session_key_owned),
            agent.turn_streamed(&content_owned, event_tx, Some(turn_cancel)),
        )
        .await
    };
    tokio::pin!(turn_fut);

    let mut extractor = AvatarJsonExtractor::new();
    let mut sentence_buf = String::new();
    let mut accumulated_raw = String::new();
    let mut partial_saved = false;
    let mut last_partial_save = std::time::Instant::now();
    let partial_save_interval = std::time::Duration::from_millis(500);
    let mut queued_message: Option<String> = None;

    let result = loop {
        tokio::select! {
            biased;

            r = &mut turn_fut => break r,

            Some(event) = event_rx.recv() => {
                match event {
                    TurnEvent::Chunk { delta } => {
                        accumulated_raw.push_str(&delta);

                        if last_partial_save.elapsed() >= partial_save_interval
                            && let Some(ref backend) = state.session_backend
                        {
                            let partial =
                                zeroclaw_providers::ChatMessage::assistant(&accumulated_raw);
                            if partial_saved {
                                let _ = backend.update_last(session_key, &partial);
                            } else {
                                let _ = backend.append(session_key, &partial);
                                partial_saved = true;
                            }
                            last_partial_save = std::time::Instant::now();
                        }

                        if let Some(new_speech) = extractor.feed(&delta) {
                            sentence_buf.push_str(&new_speech);
                            for sentence in extract_complete_sentences(&mut sentence_buf) {
                                let chunk = serde_json::json!({
                                    "type": "speech_chunk",
                                    "content": sentence,
                                });
                                let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                            }
                        }
                    }
                    TurnEvent::Thinking { delta } => {
                        let msg = serde_json::json!({"type": "thinking", "content": delta});
                        let _ = sender.send(Message::Text(msg.to_string().into())).await;
                    }
                    TurnEvent::ToolCall { id, name, args } => {
                        let filler = tool_call_filler(&name, &args);
                        let chunk = serde_json::json!({
                            "type": "speech_chunk",
                            "content": filler,
                            "filler": true,
                        });
                        let _ = sender.send(Message::Text(chunk.to_string().into())).await;

                        let tc = serde_json::json!({
                            "type": "tool_call",
                            "id": id,
                            "name": name,
                            "args": args,
                        });
                        let _ = sender.send(Message::Text(tc.to_string().into())).await;
                    }
                    TurnEvent::ToolResult { id, name, output } => {
                        let tr = serde_json::json!({
                            "type": "tool_result",
                            "id": id,
                            "name": name,
                            "output": output,
                        });
                        let _ = sender.send(Message::Text(tr.to_string().into())).await;
                    }
                }
            }

            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(p) = serde_json::from_str::<serde_json::Value>(&text) {
                            match p["type"].as_str().unwrap_or("") {
                                "cancel" => {
                                    debug!("Avatar barge-in: cancel during turn");
                                    cancel_token.cancel();
                                }
                                "message" => {
                                    let new_content =
                                        p["content"].as_str().unwrap_or("").to_string();
                                    if !new_content.is_empty() {
                                        debug!("Avatar barge-in: new message during turn");
                                        cancel_token.cancel();
                                        queued_message = Some(new_content);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => {
                        cancel_token.cancel();
                        break Err(anyhow::anyhow!("WebSocket closed during turn"));
                    }
                    _ => {}
                }
            }
        }
    };

    // Drain any events the agent emitted after the loop exit so we don't
    // truncate the avatar's last sentence.
    while let Ok(event) = event_rx.try_recv() {
        if let TurnEvent::Chunk { delta } = event {
            accumulated_raw.push_str(&delta);
            if let Some(new_speech) = extractor.feed(&delta) {
                sentence_buf.push_str(&new_speech);
                for sentence in extract_complete_sentences(&mut sentence_buf) {
                    let chunk = serde_json::json!({
                        "type": "speech_chunk",
                        "content": sentence,
                    });
                    let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                }
            }
        }
    }

    {
        state
            .cancel_tokens
            .lock()
            .expect("cancel_tokens lock poisoned")
            .remove(session_key);
    }

    let was_cancelled = match &result {
        Err(e) => zeroclaw_runtime::agent::loop_::is_tool_loop_cancelled(e),
        Ok(_) => false,
    };

    if was_cancelled {
        let truncated = if accumulated_raw.is_empty() {
            "[interrupted by user]".to_string()
        } else {
            format!("{accumulated_raw}\n\n[interrupted by user]")
        };
        if let Some(ref backend) = state.session_backend {
            let assistant_msg = zeroclaw_providers::ChatMessage::assistant(&truncated);
            if partial_saved {
                let _ = backend.update_last(session_key, &assistant_msg);
            } else {
                let _ = backend.append(session_key, &assistant_msg);
            }
            let _ = backend.set_session_state(session_key, "idle", None);
        }

        let done = serde_json::json!({
            "type": "done",
            "full_response": "",
            "cancelled": true,
        });
        let _ = sender.send(Message::Text(done.to_string().into())).await;

        let _ = state.event_tx.send(serde_json::json!({
            "type": "agent_end",
            "provider": provider_label,
            "model": state.model,
            "channel": "avatar",
            "cancelled": true,
        }));

        return queued_message;
    }

    match result {
        Ok(response) => {
            if let Some(ref backend) = state.session_backend {
                let assistant_msg = zeroclaw_providers::ChatMessage::assistant(&response);
                if partial_saved {
                    let _ = backend.update_last(session_key, &assistant_msg);
                } else {
                    let _ = backend.append(session_key, &assistant_msg);
                }
                let _ = backend.set_session_state(session_key, "idle", None);
            }

            extractor.finalize();

            // Streaming-extractor fallback: if the provider didn't stream
            // (fell back to non-streaming) we still need to deliver speech
            // sentences and rich content from the final response.
            if !extractor.found_speech && !extractor.found_content {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                    let speech = parsed["speech"].as_str().unwrap_or("");
                    let rich = parsed["content"].as_str().unwrap_or("");
                    if !speech.is_empty() {
                        for sentence in split_sentences(speech) {
                            let chunk = serde_json::json!({
                                "type": "speech_chunk",
                                "content": sentence,
                            });
                            let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                        }
                    }
                    extractor.speech = speech.to_string();
                    extractor.content = rich.to_string();
                } else {
                    for sentence in split_sentences(&response) {
                        let chunk = serde_json::json!({
                            "type": "speech_chunk",
                            "content": sentence,
                        });
                        let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                    }
                    extractor.speech = response.clone();
                }
            } else {
                let remaining = sentence_buf.trim().to_string();
                if !remaining.is_empty() {
                    let chunk = serde_json::json!({
                        "type": "speech_chunk",
                        "content": remaining,
                    });
                    let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                }
            }

            if !extractor.content.is_empty() {
                let rc = serde_json::json!({
                    "type": "rich_content",
                    "content": extractor.content,
                });
                let _ = sender.send(Message::Text(rc.to_string().into())).await;
            }

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

            let _ = state.event_tx.send(serde_json::json!({
                "type": "agent_end",
                "provider": provider_label,
                "model": state.model,
                "channel": "avatar",
            }));
        }
        Err(e) => {
            if let Some(ref backend) = state.session_backend {
                let _ = backend.set_session_state(session_key, "error", Some(&turn_id));
            }
            tracing::error!(error = %e, "Avatar agent turn failed");
            let sanitized = zeroclaw_providers::sanitize_api_error(&e.to_string());
            let err = serde_json::json!({
                "type": "error",
                "message": sanitized,
            });
            let _ = sender.send(Message::Text(err.to_string().into())).await;

            let _ = state.event_tx.send(serde_json::json!({
                "type": "error",
                "component": "ws_avatar",
                "message": sanitized,
            }));
        }
    }

    queued_message
}

// ── Incremental JSON extractor for `{speech, content}` ────────────────────
//
// Parses streaming JSON tokens to emit the `speech` field value as it
// arrives, enabling sentence-split `speech_chunk` events mid-stream. The
// `content` field is accumulated and emitted once as `rich_content` at the
// end.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractState {
    TopLevel,
    InKey,
    ExpectSpeechValue,
    ExpectContentValue,
    ExpectOtherValue,
    InSpeech,
    InContent,
    InOtherValue,
    Done,
}

struct AvatarJsonExtractor {
    state: ExtractState,
    speech: String,
    content: String,
    value_buf: String,
    key_buf: String,
    escaped: bool,
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

    /// Feed new characters from the stream. Returns any new speech text that
    /// should be appended to the sentence buffer.
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
                }
                ExtractState::InKey => {
                    if self.escaped {
                        self.key_buf.push(ch);
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
                        if !self.found_speech && self.key_buf == "speech" {
                            self.state = ExtractState::ExpectSpeechValue;
                        } else if !self.found_content && self.key_buf == "content" {
                            self.state = ExtractState::ExpectContentValue;
                        } else {
                            self.state = ExtractState::ExpectOtherValue;
                        }
                    } else {
                        self.key_buf.push(ch);
                    }
                }
                ExtractState::ExpectSpeechValue => {
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
                            // \uXXXX is rare in speech — pass through the escape
                            // char rather than implement full unicode unescape.
                            _ => ch,
                        };
                        new_speech.push(unescaped);
                        self.value_buf.push(unescaped);
                        self.escaped = false;
                    } else if ch == '\\' {
                        self.escaped = true;
                    } else if ch == '"' {
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

    /// Capture any in-progress field value if the stream ended mid-string
    /// (e.g., the LLM truncated the response or the turn was cancelled).
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

/// Generate a spoken filler phrase based on the tool name and serialized args.
/// Keeps the avatar talking while a tool runs so dead air doesn't kill flow.
fn tool_call_filler(name: &str, args: &serde_json::Value) -> &'static str {
    let args_str = args.to_string().to_lowercase();

    match name {
        n if n.starts_with("composio") || n.contains("composio") => {
            if args_str.contains("calendar")
                || args_str.contains("schedule")
                || args_str.contains("event")
            {
                "I'm checking your calendar."
            } else if args_str.contains("email")
                || args_str.contains("gmail")
                || args_str.contains("mail")
            {
                "I'm going through your emails."
            } else {
                "I'm working on that."
            }
        }
        "web_search" | "search" => "I'm searching the web.",
        "web_fetch" | "fetch" | "browse" => "I'm pulling up that page.",
        "shell" | "bash" | "exec" => "I'm running that now.",
        "file_read" | "read" => "I'm reading that.",
        "file_write" | "write" => "I'm writing that down.",
        "memory_recall" | "recall" => "I'm thinking back.",
        _ => "Still working on it.",
    }
}

/// Drain complete sentences from `buf`, leaving the trailing remainder.
/// Splits at `.`, `!`, `?` followed by whitespace or end-of-string.
fn extract_complete_sentences(buf: &mut String) -> Vec<String> {
    let mut sentences = Vec::new();
    let chars: Vec<char> = buf.chars().collect();
    let mut last_split = 0;

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

    *buf = chars[last_split..].iter().collect();
    sentences
}

/// Split text into sentences at `.`, `!`, `?`. Used for non-streaming fallback
/// where the full response is available at once.
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

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_ws_token_from_authorization_header() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer zc_test123".parse().unwrap());
        assert_eq!(extract_ws_token(&headers, None), Some("zc_test123"));
    }

    #[test]
    fn extract_ws_token_subprotocol_precedence_over_query() {
        let mut headers = HeaderMap::new();
        headers.insert("sec-websocket-protocol", "bearer.zc_sub".parse().unwrap());
        assert_eq!(extract_ws_token(&headers, Some("zc_query")), Some("zc_sub"));
    }

    #[test]
    fn avatar_response_format_has_speech_and_content_fields() {
        let v = avatar_response_format();
        let props = &v["json_schema"]["schema"]["properties"];
        assert!(props["speech"].is_object());
        assert!(props["content"].is_object());
        let required = v["json_schema"]["schema"]["required"].as_array().unwrap();
        assert!(required.iter().any(|r| r == "speech"));
        assert!(required.iter().any(|r| r == "content"));
    }

    #[test]
    fn extractor_streams_speech_in_chunks() {
        let mut ex = AvatarJsonExtractor::new();
        // Feed a JSON object split across chunks. The `\n` is a JSON escape
        // (backslash + n) that the extractor unescapes to a real newline.
        let parts = [
            "{\"speech\":\"Hello",
            " world. How are you?\",\"content\":\"## Heading",
            "\\n\\nDetails.\"}",
        ];

        let mut speech_seen = String::new();
        for part in parts {
            if let Some(s) = ex.feed(part) {
                speech_seen.push_str(&s);
            }
        }

        assert_eq!(speech_seen, "Hello world. How are you?");
        assert_eq!(ex.speech, "Hello world. How are you?");
        assert_eq!(ex.content, "## Heading\n\nDetails.");
        assert!(ex.found_speech && ex.found_content);
    }

    #[test]
    fn extractor_skips_unknown_keys_at_top_level() {
        let mut ex = AvatarJsonExtractor::new();
        ex.feed(r#"{"id":"abc","speech":"Hi.","content":""}"#);
        assert_eq!(ex.speech, "Hi.");
        assert_eq!(ex.content, "");
    }

    #[test]
    fn extractor_handles_escaped_quotes_in_speech() {
        let mut ex = AvatarJsonExtractor::new();
        ex.feed(r#"{"speech":"She said \"hi\".","content":""}"#);
        assert_eq!(ex.speech, "She said \"hi\".");
    }

    #[test]
    fn extractor_finalize_captures_truncated_speech() {
        let mut ex = AvatarJsonExtractor::new();
        ex.feed(r#"{"speech":"Half a sentence"#);
        assert!(!ex.found_speech);
        ex.finalize();
        assert!(ex.found_speech);
        assert_eq!(ex.speech, "Half a sentence");
    }

    #[test]
    fn extract_complete_sentences_drains_finished_only() {
        let mut buf = String::from("First sentence. Second one! Third in progres");
        let out = extract_complete_sentences(&mut buf);
        assert_eq!(out, vec!["First sentence.", "Second one!"]);
        // The remainder retains the leading space; trimming happens when the
        // next terminator arrives and the chunk is emitted.
        assert_eq!(buf.trim(), "Third in progres");
    }

    #[test]
    fn split_sentences_emits_each_terminator() {
        let s = split_sentences("One. Two! Three? Four");
        assert_eq!(s, vec!["One.", "Two!", "Three?", "Four"]);
    }

    #[test]
    fn tool_call_filler_routes_by_name_and_args() {
        assert_eq!(
            tool_call_filler("web_search", &serde_json::json!({})),
            "I'm searching the web."
        );
        assert_eq!(
            tool_call_filler("shell", &serde_json::json!({})),
            "I'm running that now."
        );
        // Composio routing is keyed on args content (the name is generic
        // `composio_<...>` for many endpoints, so args carry the intent).
        assert_eq!(
            tool_call_filler(
                "composio_action",
                &serde_json::json!({"action": "calendar.list_events"})
            ),
            "I'm checking your calendar."
        );
        assert_eq!(
            tool_call_filler("composio_gmail_send", &serde_json::json!({"q": "email"})),
            "I'm going through your emails."
        );
        // Composio with no recognizable args falls back to the generic line.
        assert_eq!(
            tool_call_filler("composio_action", &serde_json::json!({})),
            "I'm working on that."
        );
        assert_eq!(
            tool_call_filler("custom_tool", &serde_json::json!({})),
            "Still working on it."
        );
    }
}
