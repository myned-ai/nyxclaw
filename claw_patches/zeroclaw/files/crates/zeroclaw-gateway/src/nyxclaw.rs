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
use futures_util::{FutureExt, SinkExt, StreamExt};
use serde::Deserialize;
use tracing::debug;

/// The sub-protocol we support for the avatar WebSocket.
const WS_PROTOCOL: &str = "zeroclaw.v1";

/// Prefix used in `Sec-WebSocket-Protocol` to carry a bearer token.
const BEARER_SUBPROTO_PREFIX: &str = "bearer.";

/// Session key prefix to namespace avatar sessions away from the chat
/// gateway's `gw_` sessions in the shared session backend.
const AVATAR_SESSION_PREFIX: &str = "avatar_";

/// Hard upper bound on a client-supplied `session_id`. 128 bytes is more
/// than enough for any reasonable identifier (UUIDs are 36 chars) and
/// keeps the session-backend key length predictable.
const MAX_SESSION_ID_LEN: usize = 128;

/// Per-frame size limit applied to the WebSocket upgrade. The avatar
/// protocol is chat-shaped (a `{"type":"message","content":"..."}` JSON
/// object); 64 KiB per frame is generous for any realistic user input
/// (long copy-pastes, transcripts) while preventing a malicious client
/// from sending a multi-GB single frame that would otherwise be parsed
/// in full by `serde_json::from_str` and persisted to the session
/// backend before any length check.
const MAX_WS_FRAME_BYTES: usize = 64 * 1024;

/// Per-message size limit (across all fragments). 256 KiB allows
/// fragmented frames to assemble a larger logical message without
/// uncapping the per-frame guarantee.
const MAX_WS_MESSAGE_BYTES: usize = 256 * 1024;

/// Per-message length cap for the user `content` field on inbound
/// `{"type":"message", ...}` frames. Stricter than [`MAX_WS_FRAME_BYTES`]
/// because the content is what gets fed into the LLM and persisted —
/// even at the WS frame ceiling, the JSON envelope adds ~30 bytes of
/// overhead.
const MAX_USER_MESSAGE_BYTES: usize = 32 * 1024;

/// Hard cap on the per-turn `accumulated_raw` buffer (the streamed
/// assistant JSON envelope, accumulated across `TurnEvent::Chunk`
/// deltas, then persisted as the assistant message). 1 MiB absorbs even
/// the largest legitimate response (~64 KB speech + ~64 KB content +
/// envelope overhead is ~130 KB), and forces a hostile/jailbroken model
/// emitting an unbounded stream to be cancelled rather than blow up the
/// session backend.
const MAX_ACCUMULATED_RAW_BYTES: usize = 1024 * 1024;

/// Validate a client-supplied `session_id`. The value must be ASCII
/// alphanumeric or `-`/`_` and within `1..=MAX_SESSION_ID_LEN` bytes.
/// Anything else could collide with internal namespacing (`avatar_` /
/// `gw_` prefixes), inject newlines into log lines, or leak through to
/// a filesystem-backed session-backend implementation as a path-traversal
/// component (`../`). Validation happens at the entry edge so all
/// downstream code can assume the value is safe.
fn is_valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_SESSION_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

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

    // Bound WS payloads at the protocol layer so axum/tungstenite refuses
    // oversized frames before they reach our handler. Without these caps
    // a single 1 GB Text frame would be fully buffered by tungstenite,
    // parsed by serde_json::from_str, and (worst case) persisted to the
    // session backend — all before the handler could check `content.len()`.
    let ws = ws
        .max_frame_size(MAX_WS_FRAME_BYTES)
        .max_message_size(MAX_WS_MESSAGE_BYTES);

    // Validate any client-supplied session_id before the upgrade so we
    // reject with HTTP 400 rather than crashing the WS connection a few
    // frames in. Empty / non-charset / oversized values are refused.
    let session_id = match params.session_id {
        Some(s) if !is_valid_session_id(&s) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                "Invalid session_id — must be 1..=128 ASCII alphanumeric / '-' / '_'",
            )
                .into_response();
        }
        other => other,
    };
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
            // Log the full error internally (provider URLs, file paths,
            // partial credentials) but only ship a sanitized version to
            // the client — `Agent::from_config` errors can leak provider
            // base URLs and key fragments otherwise.
            tracing::error!(error = %e, "Avatar agent initialization failed");
            let sanitized = zeroclaw_providers::sanitize_api_error(&e.to_string());
            let err = serde_json::json!({
                "type": "error",
                "message": format!("Failed to initialise agent: {sanitized}"),
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

    // Force structured `{speech, content}` JSON output — but only for
    // providers that actually honor `response_format` natively. Setting
    // it on a provider that ignores the field (e.g. Anthropic, which
    // requires a forced-tool approach for structured output) would let
    // the LLM return free-form prose, which the streaming extractor
    // then can't parse, and the streaming-fallback path would emit an
    // INVALID_RESPONSE_FORMAT error to the client every turn — silent
    // degradation in production. The capability gate keeps the contract
    // honest: if the provider can't enforce the schema, don't claim to.
    if state.provider.supports_response_format() {
        agent.set_response_format(Some(avatar_response_format()));
    } else {
        tracing::warn!(
            "active provider does not support response_format; avatar will accept free-form responses (no rich_content / no schema enforcement)"
        );
    }

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
                        if is_valid_session_id(sid) {
                            agent.set_memory_session_id(Some(sid.clone()));
                        } else {
                            tracing::warn!(
                                "rejecting invalid session_id from connect frame; keeping session_id from query/upgrade"
                            );
                        }
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
        if content.len() > MAX_USER_MESSAGE_BYTES {
            let err = serde_json::json!({
                "type": "error",
                "code": "MESSAGE_TOO_LARGE",
                "message": format!("message content exceeds {MAX_USER_MESSAGE_BYTES} bytes"),
            });
            let _ = sender.send(Message::Text(err.to_string().into())).await;
        } else if !content.is_empty() {
            // Serialize against any other connection sharing this
            // session_key. Without the lock, two clients reusing the
            // same session_id would interleave user_msg/assistant_msg
            // appends in the backend, both register conflicting
            // CancellationToken entries (last-write-wins, breaking
            // /api/abort), and corrupt set_session_state transitions.
            match state.session_queue.acquire(&session_key).await {
                Ok(_session_guard) => {
                    if let Some(ref backend) = state.session_backend {
                        let user_msg = zeroclaw_providers::ChatMessage::user(&content);
                        let _ = backend.append(&session_key, &user_msg);
                    }
                    run_turn(
                        &state,
                        &mut agent,
                        &mut sender,
                        &mut receiver,
                        &content,
                        &session_key,
                    )
                    .await;
                }
                Err(e) => {
                    let err = serde_json::json!({
                        "type": "error",
                        "code": "SESSION_BUSY",
                        "message": e.to_string(),
                    });
                    let _ = sender.send(Message::Text(err.to_string().into())).await;
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

        match parsed["type"].as_str().unwrap_or("") {
            "message" => {
                let content = parsed["content"].as_str().unwrap_or("").to_string();
                if content.is_empty() {
                    continue;
                }
                if content.len() > MAX_USER_MESSAGE_BYTES {
                    let err = serde_json::json!({
                        "type": "error",
                        "code": "MESSAGE_TOO_LARGE",
                        "message": format!("message content exceeds {MAX_USER_MESSAGE_BYTES} bytes"),
                    });
                    let _ = sender.send(Message::Text(err.to_string().into())).await;
                    continue;
                }

                // Serialize against any other connection sharing this
                // session_key. Held across the entire barge-in chain so
                // a queued follow-up message runs atomically with the
                // original turn, matching ws.rs's per-message acquire.
                let _session_guard = match state.session_queue.acquire(&session_key).await {
                    Ok(g) => g,
                    Err(e) => {
                        let err = serde_json::json!({
                            "type": "error",
                            "code": "SESSION_BUSY",
                            "message": e.to_string(),
                        });
                        let _ = sender.send(Message::Text(err.to_string().into())).await;
                        continue;
                    }
                };

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

/// One of four explicit terminal states for an avatar turn. Replacing the
/// previous `Result<String, anyhow::Error>` makes the four success/failure
/// modes structurally distinct so the post-loop classification can't
/// misroute one as another (e.g. WS-close-mid-turn was previously matched
/// as a generic Err and dropped into the error path, leaking session state
/// `"error"` for what was just a disconnect).
enum TurnOutcome {
    /// Turn ran to completion. String is the final assistant response.
    Completed(String),
    /// User cancelled mid-turn (cancel frame, or new message that
    /// preempted the running turn). Partial content may have been
    /// streamed; persist truncated.
    Cancelled,
    /// Client closed the WebSocket mid-turn. Same persistence treatment as
    /// Cancelled, but no `done` frame is sent (the socket is gone).
    Disconnected,
    /// Turn errored out (provider failure, agent loop error, etc.).
    Failed(anyhow::Error),
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

    // Whether the active provider promised to honor response_format. When
    // false, we cannot treat a non-envelope response as a contract
    // violation — the LLM was never bound by the schema in the first
    // place. The fallback path narrates the response as plain speech.
    let schema_enforced = state.provider.supports_response_format();

    // Drive the turn future and the WS receiver concurrently. No `biased;`
    // — the previous version privileged the turn future and races where a
    // barge-in arrived in the same poll cycle as turn completion meant the
    // cancel/queued-message was observed too late.
    let outcome = loop {
        tokio::select! {
            r = &mut turn_fut => {
                break match r {
                    Ok(response) => TurnOutcome::Completed(response),
                    Err(e) if zeroclaw_runtime::agent::loop_::is_tool_loop_cancelled(&e) => {
                        TurnOutcome::Cancelled
                    }
                    Err(e) => TurnOutcome::Failed(e),
                };
            }

            Some(event) = event_rx.recv() => {
                handle_turn_event(
                    event,
                    state,
                    sender,
                    session_key,
                    &cancel_token,
                    &mut extractor,
                    &mut sentence_buf,
                    &mut accumulated_raw,
                    &mut partial_saved,
                    &mut last_partial_save,
                    partial_save_interval,
                ).await;
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
                        break TurnOutcome::Disconnected;
                    }
                    _ => {}
                }
            }
        }
    };

    // After the loop, do a single non-blocking poll on the receiver to
    // catch a barge-in that arrived in the *same* poll cycle as turn
    // completion (otherwise we'd return Completed and the queued message
    // would sit unobserved in the receiver buffer until the outer loop
    // picks it up as a fresh user turn). Safe because tokio Streams
    // expose this via FutureExt::now_or_never.
    if matches!(outcome, TurnOutcome::Completed(_))
        && let Some(buffered) = receiver.next().now_or_never()
    {
        match buffered {
            Some(Ok(Message::Text(text))) => {
                if let Ok(p) = serde_json::from_str::<serde_json::Value>(&text)
                    && p["type"].as_str() == Some("message")
                {
                    let new_content = p["content"].as_str().unwrap_or("").to_string();
                    if !new_content.is_empty() {
                        debug!("Avatar barge-in: post-completion buffered message");
                        queued_message = Some(new_content);
                    }
                }
                // A buffered "cancel" arriving after turn completion is a
                // no-op — the turn we'd cancel is already done.
            }
            Some(Ok(Message::Close(_))) | Some(Err(_)) | None => {
                // Connection closed cleanly between turn completion and
                // our poll — nothing to recover, the outer loop will see
                // the same closure and exit.
            }
            _ => {}
        }
    }

    // Drain any events the agent emitted after loop exit so the producer
    // task can release. We feed the extractor for state continuity but do
    // NOT emit speech_chunks here — those would arrive after the turn
    // boundary, and on cancel/disconnect the user explicitly asked for
    // silence. End-of-turn flushing of any complete pending sentence
    // happens explicitly in the Completed branch.
    while let Ok(event) = event_rx.try_recv() {
        if let TurnEvent::Chunk { delta } = event {
            accumulated_raw.push_str(&delta);
            if let Some(new_speech) = extractor.feed(&delta) {
                sentence_buf.push_str(&new_speech);
            }
        }
    }

    // Always finalize so any in-progress JSON value (truncated mid-string
    // by cancellation) is captured into extractor.{speech, content} for
    // persistence and downstream introspection. Idempotent.
    extractor.finalize();

    // Always remove the cancel-token registry entry — every outcome path
    // below depends on this happening before any awaits that could yield.
    {
        state
            .cancel_tokens
            .lock()
            .expect("cancel_tokens lock poisoned")
            .remove(session_key);
    }

    match outcome {
        TurnOutcome::Completed(response) => {
            handle_completed(
                state,
                sender,
                session_key,
                content,
                response,
                &provider_label,
                &mut extractor,
                &sentence_buf,
                partial_saved,
                schema_enforced,
            )
            .await;
        }
        TurnOutcome::Cancelled => {
            handle_terminated_early(
                state,
                Some(sender),
                session_key,
                &provider_label,
                &accumulated_raw,
                partial_saved,
            )
            .await;
        }
        TurnOutcome::Disconnected => {
            // Same persistence as Cancelled but skip the `done` frame —
            // the socket is closed, sender writes are no-ops anyway, and
            // we don't want to log misleading send errors at runtime.
            handle_terminated_early(
                state,
                None,
                session_key,
                &provider_label,
                &accumulated_raw,
                partial_saved,
            )
            .await;
        }
        TurnOutcome::Failed(e) => {
            if let Some(ref backend) = state.session_backend {
                let _ = backend.set_session_state(session_key, "error", Some(&turn_id));
            }
            tracing::error!(error = %e, "Avatar agent turn failed");
            let sanitized = zeroclaw_providers::sanitize_api_error(&e.to_string());
            let code = classify_error_code(&sanitized);
            let err = serde_json::json!({
                "type": "error",
                "code": code,
                "message": sanitized,
            });
            let _ = sender.send(Message::Text(err.to_string().into())).await;

            let _ = state.event_tx.send(serde_json::json!({
                "type": "error",
                "component": "ws_avatar",
                "code": code,
                "message": sanitized,
            }));
        }
    }

    queued_message
}

/// Outcome of classifying a single TurnEvent into the frames the WS
/// client should receive plus any side-effect requests. Pure data —
/// `classify_turn_event` doesn't touch the network or any locks, which
/// makes the dispatch logic unit-testable without a live socket.
#[derive(Debug)]
struct TurnEventDispatch {
    /// Ordered WS frames to send (zero-or-more `serde_json::Value`s
    /// that will be JSON-serialized into `Message::Text` frames).
    frames: Vec<serde_json::Value>,
    /// Set when this event should trigger turn cancellation (i.e. the
    /// per-turn accumulated_raw cap was exceeded). The caller is
    /// expected to invoke `cancel_token.cancel()` and skip side effects.
    request_cancel: bool,
}

/// Pure (no I/O, no locks) classification of a TurnEvent into the WS
/// frames it produces and any cancel request it triggers. Mutates the
/// extractor / sentence_buf / accumulated_raw to track streaming state.
fn classify_turn_event(
    event: &zeroclaw_runtime::agent::TurnEvent,
    extractor: &mut AvatarJsonExtractor,
    sentence_buf: &mut String,
    accumulated_raw: &mut String,
) -> TurnEventDispatch {
    use zeroclaw_runtime::agent::TurnEvent;

    let mut frames = Vec::new();

    match event {
        TurnEvent::Chunk { delta } => {
            // Hard cap on the per-turn accumulated buffer. A hostile
            // model emitting an unbounded stream would otherwise grow
            // accumulated_raw and the persisted assistant message
            // without bound. If we cross the cap, request cancel and
            // drop this delta — the already-accumulated content stays
            // as the truncated assistant message.
            if accumulated_raw.len() + delta.len() > MAX_ACCUMULATED_RAW_BYTES {
                return TurnEventDispatch {
                    frames,
                    request_cancel: true,
                };
            }
            accumulated_raw.push_str(delta);

            if let Some(new_speech) = extractor.feed(delta) {
                sentence_buf.push_str(&new_speech);
                for sentence in extract_complete_sentences(sentence_buf) {
                    frames.push(serde_json::json!({
                        "type": "speech_chunk",
                        "content": sentence,
                    }));
                }
            }
        }
        TurnEvent::Thinking { delta } => {
            frames.push(serde_json::json!({
                "type": "thinking",
                "content": delta,
            }));
        }
        TurnEvent::ToolCall { id, name, args } => {
            // Two-frame contract: the spoken filler comes first so the
            // avatar starts talking before the tool_call event reaches
            // the client; the protocol payload follows.
            let filler = tool_call_filler(name, args);
            frames.push(serde_json::json!({
                "type": "speech_chunk",
                "content": filler,
                "filler": true,
            }));
            frames.push(serde_json::json!({
                "type": "tool_call",
                "id": id,
                "name": name,
                "args": args,
            }));
        }
        TurnEvent::ToolResult { id, name, output } => {
            frames.push(serde_json::json!({
                "type": "tool_result",
                "id": id,
                "name": name,
                "output": output,
            }));
        }
    }

    TurnEventDispatch {
        frames,
        request_cancel: false,
    }
}

/// Apply a single TurnEvent: classify, fire any cancel signal, perform
/// the partial-save side effect (Chunk only), and ship the resulting
/// frames over the WS sink. The classification is split out into
/// `classify_turn_event` so the frame-shape contracts can be
/// unit-tested without a live socket.
#[allow(clippy::too_many_arguments)] // event-handler context is genuinely wide
async fn handle_turn_event(
    event: zeroclaw_runtime::agent::TurnEvent,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    session_key: &str,
    cancel_token: &tokio_util::sync::CancellationToken,
    extractor: &mut AvatarJsonExtractor,
    sentence_buf: &mut String,
    accumulated_raw: &mut String,
    partial_saved: &mut bool,
    last_partial_save: &mut std::time::Instant,
    partial_save_interval: std::time::Duration,
) {
    let is_chunk = matches!(&event, zeroclaw_runtime::agent::TurnEvent::Chunk { .. });
    let dispatch =
        classify_turn_event(&event, extractor, sentence_buf, accumulated_raw);

    if dispatch.request_cancel {
        if !cancel_token.is_cancelled() {
            tracing::error!(
                cap = MAX_ACCUMULATED_RAW_BYTES,
                accumulated = accumulated_raw.len(),
                "avatar turn exceeded accumulated_raw cap — cancelling"
            );
            cancel_token.cancel();
        }
        return;
    }

    // Partial-save side effect runs only on Chunk events that actually
    // appended (cancel-path early-returned above). Time-gated to avoid
    // hammering the backend on every delta.
    if is_chunk
        && last_partial_save.elapsed() >= partial_save_interval
        && let Some(ref backend) = state.session_backend
    {
        let partial = zeroclaw_providers::ChatMessage::assistant(&*accumulated_raw);
        if *partial_saved {
            let _ = backend.update_last(session_key, &partial);
        } else {
            let _ = backend.append(session_key, &partial);
            *partial_saved = true;
        }
        *last_partial_save = std::time::Instant::now();
    }

    for frame in dispatch.frames {
        let _ = sender.send(Message::Text(frame.to_string().into())).await;
    }
}

/// Successful-turn finalization: persist final assistant message, deliver
/// any pending speech, deliver rich content, send `done` and `agent_end`.
#[allow(clippy::too_many_arguments)]
async fn handle_completed(
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    session_key: &str,
    user_content: &str,
    response: String,
    provider_label: &str,
    extractor: &mut AvatarJsonExtractor,
    sentence_buf: &str,
    partial_saved: bool,
    schema_enforced: bool,
) {
    if let Some(ref backend) = state.session_backend {
        let assistant_msg = zeroclaw_providers::ChatMessage::assistant(&response);
        if partial_saved {
            let _ = backend.update_last(session_key, &assistant_msg);
        } else {
            let _ = backend.append(session_key, &assistant_msg);
        }
        let _ = backend.set_session_state(session_key, "idle", None);
    }

    // Streaming-extractor fallback: if neither speech nor content was
    // observed during streaming the LLM either (a) returned the envelope
    // in one shot (non-streaming provider), or (b) returned plain prose
    // (provider doesn't enforce response_format). Try to detect the
    // envelope first; if absent, behavior depends on `schema_enforced`:
    //   - true:  the provider promised the schema → contract violation.
    //            Send INVALID_RESPONSE_FORMAT, do NOT narrate raw bytes.
    //   - false: the schema was never enforced → narrate the response as
    //            plain speech (graceful degradation; equivalent to /ws/chat
    //            but split into sentences).
    if !extractor.found_speech && !extractor.found_content {
        let parsed_envelope = serde_json::from_str::<serde_json::Value>(&response)
            .ok()
            .filter(|v| v.get("speech").and_then(|s| s.as_str()).is_some());

        match parsed_envelope {
            Some(parsed) => {
                let speech = parsed["speech"].as_str().unwrap_or("");
                let rich = parsed["content"].as_str().unwrap_or("");
                for sentence in split_sentences(speech) {
                    let chunk = serde_json::json!({
                        "type": "speech_chunk",
                        "content": sentence,
                    });
                    let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                }
                extractor.speech = speech.to_string();
                extractor.content = rich.to_string();
            }
            None if schema_enforced => {
                tracing::warn!(
                    response_preview = %response.chars().take(200).collect::<String>(),
                    "avatar turn did not return {{speech, content}} envelope; sending error frame"
                );
                let err = serde_json::json!({
                    "type": "error",
                    "code": "INVALID_RESPONSE_FORMAT",
                    "message": "Model did not return structured speech/content output",
                });
                let _ = sender.send(Message::Text(err.to_string().into())).await;
                let _ = state.event_tx.send(serde_json::json!({
                    "type": "error",
                    "component": "ws_avatar",
                    "code": "INVALID_RESPONSE_FORMAT",
                    "message": "non-envelope response from provider",
                }));
                return;
            }
            None => {
                // Schema was not enforced — treat the entire response as
                // plain speech (no rich_content). Sentence-split for
                // streaming UX.
                for sentence in split_sentences(&response) {
                    let chunk = serde_json::json!({
                        "type": "speech_chunk",
                        "content": sentence,
                    });
                    let _ = sender.send(Message::Text(chunk.to_string().into())).await;
                }
                extractor.speech = response.clone();
            }
        }
    } else {
        // Streaming path: flush any trailing partial sentence that didn't
        // hit a terminator+whitespace boundary.
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

    // Fire-and-forget memory consolidation so facts from avatar sessions
    // are extracted to long-term memory (Daily + Core categories), same
    // as ws.rs does for /ws/chat. Skipping this on /ws/avatar previously
    // was a silent regression — facts spoken during voice sessions never
    // landed in memory. The consolidation runs against the user's text
    // and the assistant's response (full envelope including markdown) so
    // the fact extractor sees both speech and rich content.
    if state.auto_save {
        let mem = state.mem.clone();
        let provider = state.provider.clone();
        let model = state.model.clone();
        let user_msg = user_content.to_string();
        let assistant_resp = response.clone();
        tokio::spawn(async move {
            if let Err(e) = zeroclaw_memory::consolidation::consolidate_turn(
                provider.as_ref(),
                &model,
                mem.as_ref(),
                &user_msg,
                &assistant_resp,
            )
            .await
            {
                tracing::debug!("avatar memory consolidation skipped: {e}");
            }
        });
    }

    // Match ws.rs:620 — send chunk_reset so any client that accumulated
    // speech_chunks for transcript display can flush its draft buffer
    // before the authoritative `done` arrives. Avatar clients that only
    // play audio can ignore this frame; it costs them nothing.
    let reset = serde_json::json!({ "type": "chunk_reset" });
    let _ = sender.send(Message::Text(reset.to_string().into())).await;

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

/// Cancellation/disconnect finalization: persist any partial content with
/// an `[interrupted by user]` marker, send `done {cancelled:true}` if the
/// socket is still alive, and broadcast `agent_end {cancelled:true}`.
async fn handle_terminated_early(
    state: &AppState,
    sender: Option<&mut futures_util::stream::SplitSink<WebSocket, Message>>,
    session_key: &str,
    provider_label: &str,
    accumulated_raw: &str,
    partial_saved: bool,
) {
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

    if let Some(sender) = sender {
        let done = serde_json::json!({
            "type": "done",
            "full_response": "",
            "cancelled": true,
        });
        let _ = sender.send(Message::Text(done.to_string().into())).await;
    }

    let _ = state.event_tx.send(serde_json::json!({
        "type": "agent_end",
        "provider": provider_label,
        "model": state.model,
        "channel": "avatar",
        "cancelled": true,
    }));
}

/// Map a sanitized provider error string to a stable error code so mobile
/// clients can branch on category instead of pattern-matching free-form
/// text. Mirrors `ws.rs::process_chat_message`'s taxonomy.
fn classify_error_code(sanitized: &str) -> &'static str {
    let s = sanitized.to_lowercase();
    if s.contains("api key") || s.contains("authentication") || s.contains("unauthorized") {
        "AUTH_ERROR"
    } else if s.contains("provider") || s.contains("model") {
        "PROVIDER_ERROR"
    } else {
        "AGENT_ERROR"
    }
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

/// Hard cap on a single string field (`speech` / `content`) and key name.
/// A normal speech response is < 4 KB; markdown content might reach ~16
/// KB; this 64 KB cap is generous in normal use but bounds the worker's
/// memory in the hostile case where a model emits an unclosed string and
/// the extractor would otherwise buffer indefinitely. Once a field hits
/// the cap we stop appending to it but stay in the same parse state, so
/// the closing `"` (when it eventually arrives) still transitions
/// correctly. The truncation is reported via `truncated_speech` /
/// `truncated_content` so callers can log audit events.
const MAX_EXTRACTOR_FIELD_BYTES: usize = 64 * 1024;

struct AvatarJsonExtractor {
    state: ExtractState,
    speech: String,
    content: String,
    value_buf: String,
    key_buf: String,
    escaped: bool,
    found_speech: bool,
    found_content: bool,
    /// Set once `value_buf` for `speech` has hit `MAX_EXTRACTOR_FIELD_BYTES`
    /// and additional chars are being dropped. Persists across the
    /// extractor's lifetime.
    truncated_speech: bool,
    /// Same as `truncated_speech` but for the `content` field.
    truncated_content: bool,
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
            truncated_speech: false,
            truncated_content: false,
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
                        if self.key_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                            self.key_buf.push(ch);
                        }
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
                    } else if self.key_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                        self.key_buf.push(ch);
                    }
                    // Beyond cap: silently drop. The closing `"` still
                    // transitions correctly via the equality check above
                    // (a key longer than 64KB won't match "speech" or
                    // "content" anyway, so it routes to ExpectOtherValue
                    // and gets skipped).
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
                        if self.value_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                            new_speech.push(unescaped);
                            self.value_buf.push(unescaped);
                        } else if !self.truncated_speech {
                            tracing::warn!(
                                cap = MAX_EXTRACTOR_FIELD_BYTES,
                                "avatar speech field exceeded cap — truncating remainder"
                            );
                            self.truncated_speech = true;
                        }
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
                    } else if self.value_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                        new_speech.push(ch);
                        self.value_buf.push(ch);
                    } else if !self.truncated_speech {
                        tracing::warn!(
                            cap = MAX_EXTRACTOR_FIELD_BYTES,
                            "avatar speech field exceeded cap — truncating remainder"
                        );
                        self.truncated_speech = true;
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
                        if self.value_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                            self.value_buf.push(unescaped);
                        } else if !self.truncated_content {
                            tracing::warn!(
                                cap = MAX_EXTRACTOR_FIELD_BYTES,
                                "avatar content field exceeded cap — truncating remainder"
                            );
                            self.truncated_content = true;
                        }
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
                    } else if self.value_buf.len() < MAX_EXTRACTOR_FIELD_BYTES {
                        self.value_buf.push(ch);
                    } else if !self.truncated_content {
                        tracing::warn!(
                            cap = MAX_EXTRACTOR_FIELD_BYTES,
                            "avatar content field exceeded cap — truncating remainder"
                        );
                        self.truncated_content = true;
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
/// A sentence is complete only when a terminator (`.`, `!`, `?`) is followed
/// by whitespace within the same buffer — never at end-of-buffer, since the
/// next streamed delta might extend the token (e.g. "1." then "5 inches"
/// must not split at "1."). End-of-stream flushing is the caller's job;
/// see the post-loop `sentence_buf.trim()` emit in `run_turn`.
fn extract_complete_sentences(buf: &mut String) -> Vec<String> {
    let mut sentences = Vec::new();
    let chars: Vec<char> = buf.chars().collect();
    let mut last_split = 0;

    for i in 0..chars.len() {
        let ch = chars[i];
        let is_terminator = ch == '.' || ch == '!' || ch == '?';
        let has_lookahead_whitespace =
            i + 1 < chars.len() && (chars[i + 1] == ' ' || chars[i + 1] == '\n');

        if is_terminator && has_lookahead_whitespace {
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
    fn session_id_validator_accepts_canonical_forms() {
        assert!(is_valid_session_id("abc123"));
        assert!(is_valid_session_id("a-b-c"));
        assert!(is_valid_session_id("a_b_c"));
        assert!(is_valid_session_id("550e8400-e29b-41d4-a716-446655440000")); // UUID
        assert!(is_valid_session_id(&"x".repeat(MAX_SESSION_ID_LEN)));
    }

    #[test]
    fn session_id_validator_rejects_dangerous_inputs() {
        // Empty or oversized
        assert!(!is_valid_session_id(""));
        assert!(!is_valid_session_id(&"x".repeat(MAX_SESSION_ID_LEN + 1)));

        // Path-traversal probes — would otherwise collide with backend
        // namespacing or, on a filesystem-backed backend, escape the
        // session directory.
        assert!(!is_valid_session_id("../gw_other"));
        assert!(!is_valid_session_id("a/b"));
        assert!(!is_valid_session_id("a\\b"));

        // Log-injection probes — newlines or control chars in session_id
        // would be reflected in tracing output (debug log includes it).
        assert!(!is_valid_session_id("a\nb"));
        assert!(!is_valid_session_id("a\rb"));
        assert!(!is_valid_session_id("a\0b"));

        // Whitespace, special chars, non-ASCII
        assert!(!is_valid_session_id("a b"));
        assert!(!is_valid_session_id("a.b"));
        assert!(!is_valid_session_id("café"));
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
    fn extractor_caps_speech_field_at_max_bytes() {
        // A model emitting a 200KB speech field with no closing quote
        // would, without the cap, blow up value_buf and the eventual
        // self.speech String. With the cap, value_buf stops growing at
        // 64KB, the extractor still tracks state correctly, and the
        // truncated_speech flag is set so callers can audit.
        let mut ex = AvatarJsonExtractor::new();
        let prefix = "{\"speech\":\"";
        ex.feed(prefix);
        // 200 KB of 'a' — well over MAX_EXTRACTOR_FIELD_BYTES (64 KB).
        let huge = "a".repeat(200 * 1024);
        ex.feed(&huge);
        assert!(ex.truncated_speech, "cap must engage");
        assert_eq!(
            ex.value_buf.len(),
            MAX_EXTRACTOR_FIELD_BYTES,
            "value_buf must stop at the cap, not grow further"
        );

        // Closing the string still transitions correctly even after
        // truncation — the parser doesn't get stuck in InSpeech.
        ex.feed("\"}");
        assert!(ex.found_speech);
        assert_eq!(ex.speech.len(), MAX_EXTRACTOR_FIELD_BYTES);
    }

    #[test]
    fn extractor_caps_content_field_at_max_bytes() {
        let mut ex = AvatarJsonExtractor::new();
        ex.feed("{\"speech\":\"hi\",\"content\":\"");
        let huge = "x".repeat(200 * 1024);
        ex.feed(&huge);
        assert!(ex.truncated_content);
        assert_eq!(ex.value_buf.len(), MAX_EXTRACTOR_FIELD_BYTES);
        ex.feed("\"}");
        assert!(ex.found_content);
        assert_eq!(ex.content.len(), MAX_EXTRACTOR_FIELD_BYTES);
    }

    #[test]
    fn extractor_oversized_unknown_key_is_skipped_safely() {
        // A 200KB key (must end up routed to ExpectOtherValue since it
        // can't equal "speech" or "content") used to grow key_buf
        // unbounded. Cap stops it; the unknown-key handling still works.
        let mut ex = AvatarJsonExtractor::new();
        ex.feed("{\"");
        ex.feed(&"k".repeat(200 * 1024));
        assert_eq!(ex.key_buf.len(), MAX_EXTRACTOR_FIELD_BYTES);
        ex.feed("\":\"value\",\"speech\":\"hi.\",\"content\":\"\"}");
        assert!(ex.found_speech);
        assert_eq!(ex.speech, "hi.");
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
    fn extract_complete_sentences_does_not_split_on_buffer_end_terminator() {
        // Terminator at end-of-buffer must not emit — the next streamed
        // delta might extend the token. Concrete: "1." followed by
        // "5 inches" must not be split into a sentence at the "1.".
        let mut buf = String::from("I see 1.");
        let out = extract_complete_sentences(&mut buf);
        assert!(out.is_empty(), "must not emit at buffer-end terminator");
        assert_eq!(buf, "I see 1.");

        // After the next chunk arrives, the lookahead disambiguates.
        buf.push_str("5 inches.");
        let out2 = extract_complete_sentences(&mut buf);
        assert!(out2.is_empty(), "no whitespace after the trailing '.'");
        assert_eq!(buf, "I see 1.5 inches.");

        // Once a real boundary appears, the full sentence flushes.
        buf.push_str(" Next");
        let out3 = extract_complete_sentences(&mut buf);
        assert_eq!(out3, vec!["I see 1.5 inches."]);
        assert_eq!(buf.trim(), "Next");
    }

    #[test]
    fn extract_complete_sentences_decimal_inside_sentence_does_not_split() {
        let mut buf = String::from("Pi is 3.14 approximately.");
        let out = extract_complete_sentences(&mut buf);
        // Only the trailing terminator at buffer-end exists, but no
        // whitespace follows it, so nothing emits yet.
        assert!(out.is_empty());
        // Next chunk delivers the lookahead.
        buf.push_str(" Done");
        let out2 = extract_complete_sentences(&mut buf);
        assert_eq!(out2, vec!["Pi is 3.14 approximately."]);
        assert_eq!(buf.trim(), "Done");
    }

    #[test]
    fn split_sentences_emits_each_terminator() {
        let s = split_sentences("One. Two! Three? Four");
        assert_eq!(s, vec!["One.", "Two!", "Three?", "Four"]);
    }

    // ── classify_turn_event: pure dispatch logic ────────────────────
    //
    // These tests pin the WS-frame shape contracts that the avatar
    // protocol exposes to mobile clients. They run against the same
    // helper that handle_turn_event uses, with no sink or async — any
    // change to the JSON shape or the ToolCall two-frame ordering must
    // update these tests deliberately, not silently drift.

    use zeroclaw_runtime::agent::TurnEvent;

    fn fresh_dispatch_state() -> (AvatarJsonExtractor, String, String) {
        (AvatarJsonExtractor::new(), String::new(), String::new())
    }

    #[test]
    fn classify_turn_event_chunk_emits_no_frames_until_sentence_completes() {
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();

        // Mid-sentence streaming — extractor sees opening, no terminator
        // yet, no frames go out. accumulated_raw still grows.
        let d = classify_turn_event(
            &TurnEvent::Chunk {
                delta: "{\"speech\":\"Hello world".into(),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert!(d.frames.is_empty());
        assert!(!d.request_cancel);
        assert_eq!(raw, "{\"speech\":\"Hello world");
    }

    #[test]
    fn classify_turn_event_chunk_emits_speech_chunk_on_complete_sentence() {
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();

        // Stream the JSON envelope opener + a complete first sentence
        // with trailing whitespace (which IS the lookahead boundary).
        // The extract_complete_sentences helper trips on `.` followed by
        // ` `, emitting "First line." in the same call.
        let d1 = classify_turn_event(
            &TurnEvent::Chunk {
                delta: "{\"speech\":\"First line. ".into(),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert_eq!(d1.frames.len(), 1, "got: {:?}", d1.frames);
        assert_eq!(d1.frames[0]["type"], "speech_chunk");
        assert_eq!(d1.frames[0]["content"], "First line.");

        // The follow-up chunk has no terminator — buffered, no frame.
        let d2 = classify_turn_event(
            &TurnEvent::Chunk {
                delta: "Second".into(),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert!(d2.frames.is_empty());
        assert!(!d2.request_cancel);
    }

    #[test]
    fn classify_turn_event_chunk_over_cap_requests_cancel() {
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();
        // Pre-fill accumulated_raw to just under the cap, then push a
        // delta that would cross it.
        raw.push_str(&"x".repeat(MAX_ACCUMULATED_RAW_BYTES - 100));
        let d = classify_turn_event(
            &TurnEvent::Chunk {
                delta: "y".repeat(200),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert!(d.request_cancel, "over-cap delta must request cancel");
        assert!(d.frames.is_empty());
        // The over-cap delta itself was NOT appended — the partial
        // assistant message stays at the pre-cap content.
        assert_eq!(raw.len(), MAX_ACCUMULATED_RAW_BYTES - 100);
    }

    #[test]
    fn classify_turn_event_thinking_emits_single_thinking_frame() {
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();
        let d = classify_turn_event(
            &TurnEvent::Thinking {
                delta: "thinking out loud".into(),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert_eq!(d.frames.len(), 1);
        assert_eq!(d.frames[0]["type"], "thinking");
        assert_eq!(d.frames[0]["content"], "thinking out loud");
    }

    #[test]
    fn classify_turn_event_tool_call_emits_filler_then_tool_call_in_order() {
        // Avatar protocol contract: ToolCall produces TWO frames — the
        // spoken filler MUST come first so the avatar starts talking
        // before the tool_call event reaches the client. If this
        // ordering ever flips, the avatar would go silent during tool
        // execution. Pinned by this test.
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();
        let d = classify_turn_event(
            &TurnEvent::ToolCall {
                id: "call_1".into(),
                name: "web_search".into(),
                args: serde_json::json!({"query": "rust"}),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert_eq!(d.frames.len(), 2);
        assert_eq!(d.frames[0]["type"], "speech_chunk");
        assert_eq!(d.frames[0]["filler"], true);
        assert_eq!(d.frames[0]["content"], "I'm searching the web.");
        assert_eq!(d.frames[1]["type"], "tool_call");
        assert_eq!(d.frames[1]["id"], "call_1");
        assert_eq!(d.frames[1]["name"], "web_search");
        assert_eq!(d.frames[1]["args"]["query"], "rust");
    }

    #[test]
    fn classify_turn_event_tool_result_emits_single_tool_result_frame() {
        let (mut ex, mut sbuf, mut raw) = fresh_dispatch_state();
        let d = classify_turn_event(
            &TurnEvent::ToolResult {
                id: "call_1".into(),
                name: "web_search".into(),
                output: "found 3 results".into(),
            },
            &mut ex,
            &mut sbuf,
            &mut raw,
        );
        assert_eq!(d.frames.len(), 1);
        assert_eq!(d.frames[0]["type"], "tool_result");
        assert_eq!(d.frames[0]["id"], "call_1");
        assert_eq!(d.frames[0]["name"], "web_search");
        assert_eq!(d.frames[0]["output"], "found 3 results");
    }

    #[test]
    fn classify_error_code_taxonomy_matches_ws_rs() {
        // Same shape as ws.rs::process_chat_message — mobile clients
        // can branch on category instead of pattern-matching free-form
        // error text.
        assert_eq!(classify_error_code("invalid api key"), "AUTH_ERROR");
        assert_eq!(classify_error_code("authentication failed"), "AUTH_ERROR");
        assert_eq!(classify_error_code("Unauthorized"), "AUTH_ERROR");
        assert_eq!(classify_error_code("provider unavailable"), "PROVIDER_ERROR");
        assert_eq!(classify_error_code("model not found"), "PROVIDER_ERROR");
        assert_eq!(classify_error_code("network timeout"), "AGENT_ERROR");
        assert_eq!(classify_error_code(""), "AGENT_ERROR");
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
