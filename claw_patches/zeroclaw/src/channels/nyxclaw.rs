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

use super::AppState;
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
            let err = serde_json::json!({"type": "error", "message": format!("Failed to initialise agent: {e}")});
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

/// Process a single avatar message through the agent, streaming events back.
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

    // Create event channel for turn_with_events
    let (event_tx, mut event_rx) = mpsc::channel::<serde_json::Value>(64);

    // Run turn_with_events and event forwarding concurrently via select loop.
    // We can't move sender into a spawned task (it's borrowed), so we drive
    // both the agent turn and the event channel in the same task.
    let turn_handle = {
        let mut final_response: Option<Result<String, anyhow::Error>> = None;

        // Use tokio::select to concurrently drive both the turn and event forwarding
        let turn_fut = agent.turn_with_events(content, event_tx);
        tokio::pin!(turn_fut);

        loop {
            tokio::select! {
                result = &mut turn_fut => {
                    final_response = Some(result);
                    break;
                }
                Some(event) = event_rx.recv() => {
                    let _ = sender.send(Message::Text(event.to_string().into())).await;
                }
            }
        }

        // Drain any remaining events
        while let Ok(event) = event_rx.try_recv() {
            let _ = sender.send(Message::Text(event.to_string().into())).await;
        }

        final_response.unwrap()
    };

    match turn_handle {
        Ok(response) => {
            // Persist assistant response
            if let Some(ref backend) = state.session_backend {
                let assistant_msg = crate::providers::ChatMessage::assistant(&response);
                let _ = backend.append(session_key, &assistant_msg);
            }

            // Try to parse as structured avatar JSON {speech, content}
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                let speech = parsed["speech"].as_str().unwrap_or("");
                let rich_content = parsed["content"].as_str().unwrap_or("");

                // Send speech as sentence-split chunks
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

                // Send rich content if non-empty
                if !rich_content.is_empty() {
                    let rc = serde_json::json!({
                        "type": "rich_content",
                        "content": rich_content,
                    });
                    let _ = sender.send(Message::Text(rc.to_string().into())).await;
                }
            } else {
                // Fallback: LLM didn't follow JSON format, send raw text as speech
                for sentence in split_sentences(&response) {
                    if !sentence.is_empty() {
                        let chunk = serde_json::json!({
                            "type": "speech_chunk",
                            "content": sentence,
                        });
                        let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                    }
                }
            }

            let done = serde_json::json!({
                "type": "done",
                "full_response": response,
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
