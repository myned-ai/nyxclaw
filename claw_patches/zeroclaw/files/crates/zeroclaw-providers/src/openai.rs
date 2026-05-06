use crate::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    Provider, StreamChunk, StreamError, StreamEvent, StreamOptions, StreamResult, TokenUsage,
    ToolCall as ProviderToolCall,
};
use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zeroclaw_api::tool::ToolSpec;

/// OpenAI's public API endpoint.
const BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAiProvider {
    base_url: String,
    credential: Option<String>,
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models may return output in `reasoning_content`.
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl ResponseMessage {
    fn effective_content(&self) -> String {
        match &self.content {
            Some(c) if !c.is_empty() => c.clone(),
            _ => self.reasoning_content.clone().unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NativeChatRequest {
    model: String,
    messages: Vec<NativeMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<NativeToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Structured-output schema, e.g. `{"type": "json_schema", "json_schema": {...}}`
    /// or `{"type": "json_object"}`. See OpenAI's response_format docs.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    /// Set to `Some(true)` for SSE streaming (used by `stream_chat()`).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct NativeMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<NativeToolCall>>,
    /// Raw reasoning content from thinking models; pass-through for providers
    /// that require it in assistant tool-call history messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolSpec {
    #[serde(rename = "type")]
    kind: String,
    function: NativeToolFunctionSpec,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

fn parse_native_tool_spec(value: serde_json::Value) -> anyhow::Result<NativeToolSpec> {
    let spec: NativeToolSpec = serde_json::from_value(value)
        .map_err(|e| anyhow::anyhow!("Invalid OpenAI tool specification: {e}"))?;

    if spec.kind != "function" {
        anyhow::bail!(
            "Invalid OpenAI tool specification: unsupported tool type '{}', expected 'function'",
            spec.kind
        );
    }

    Ok(spec)
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: NativeFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct NativeChatResponse {
    choices: Vec<NativeChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct NativeChoice {
    message: NativeResponseMessage,
}

#[derive(Debug, Deserialize)]
struct NativeResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models may return output in `reasoning_content`.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<NativeToolCall>>,
}

impl NativeResponseMessage {
    fn effective_content(&self) -> Option<String> {
        match &self.content {
            Some(c) if !c.is_empty() => Some(c.clone()),
            _ => self.reasoning_content.clone(),
        }
    }
}

impl OpenAiProvider {
    pub fn new(credential: Option<&str>) -> Self {
        Self::with_base_url(None, credential)
    }

    /// Create a provider with an optional custom base URL.
    /// Defaults to `https://api.openai.com/v1` when `base_url` is `None`.
    pub fn with_base_url(base_url: Option<&str>, credential: Option<&str>) -> Self {
        Self {
            base_url: base_url
                .map(|u| u.trim_end_matches('/').to_string())
                .unwrap_or_else(|| BASE_URL.to_string()),
            credential: credential.map(ToString::to_string),
            max_tokens: None,
        }
    }

    /// Set the maximum output tokens for API requests.
    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Adjust temperature for models that have specific requirements.
    /// Some OpenAI models (like gpt-5-mini, o1, o3, etc) only accept temperature=1.0.
    fn adjust_temperature_for_model(model: &str, requested_temperature: f64) -> f64 {
        // Models that require temperature=1.0
        let requires_1_0 = matches!(
            model,
            "gpt-5"
                | "gpt-5-2025-08-07"
                | "gpt-5-mini"
                | "gpt-5-mini-2025-08-07"
                | "gpt-5-nano"
                | "gpt-5-nano-2025-08-07"
                | "gpt-5.1-chat-latest"
                | "gpt-5.2-chat-latest"
                | "gpt-5.3-chat-latest"
                | "o1"
                | "o1-2024-12-17"
                | "o3"
                | "o3-2025-04-16"
                | "o3-mini"
                | "o3-mini-2025-01-31"
                | "o4-mini"
                | "o4-mini-2025-04-16"
        );

        if requires_1_0 {
            1.0
        } else {
            requested_temperature
        }
    }

    fn convert_tools(tools: Option<&[ToolSpec]>) -> Option<Vec<NativeToolSpec>> {
        tools.map(|items| {
            items
                .iter()
                .map(|tool| NativeToolSpec {
                    kind: "function".to_string(),
                    function: NativeToolFunctionSpec {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect()
        })
    }

    fn convert_messages(messages: &[ChatMessage]) -> Vec<NativeMessage> {
        messages
            .iter()
            .map(|m| {
                if m.role == "assistant"
                    && let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content)
                    && let Some(tool_calls_value) = value.get("tool_calls")
                    && let Ok(parsed_calls) =
                        serde_json::from_value::<Vec<ProviderToolCall>>(tool_calls_value.clone())
                {
                    let tool_calls = parsed_calls
                        .into_iter()
                        .map(|tc| NativeToolCall {
                            id: Some(tc.id),
                            kind: Some("function".to_string()),
                            function: NativeFunctionCall {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect::<Vec<_>>();
                    let content = value
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string);
                    let reasoning_content = value
                        .get("reasoning_content")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string);
                    return NativeMessage {
                        role: "assistant".to_string(),
                        content,
                        tool_call_id: None,
                        tool_calls: Some(tool_calls),
                        reasoning_content,
                    };
                }

                if m.role == "tool"
                    && let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content)
                {
                    let tool_call_id = value
                        .get("tool_call_id")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string);
                    let content = value
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string);
                    return NativeMessage {
                        role: "tool".to_string(),
                        content,
                        tool_call_id,
                        tool_calls: None,
                        reasoning_content: None,
                    };
                }

                NativeMessage {
                    role: m.role.clone(),
                    content: Some(m.content.clone()),
                    tool_call_id: None,
                    tool_calls: None,
                    reasoning_content: None,
                }
            })
            .collect()
    }

    fn parse_native_response(message: NativeResponseMessage) -> ProviderChatResponse {
        let text = message.effective_content();
        let reasoning_content = message.reasoning_content.clone();
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ProviderToolCall {
                id: tc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect::<Vec<_>>();

        ProviderChatResponse {
            text,
            tool_calls,
            usage: None,
            reasoning_content,
        }
    }

    fn http_client(&self) -> Client {
        zeroclaw_config::schema::build_runtime_proxy_client_with_timeouts(
            "provider.openai",
            120,
            10,
        )
    }

    /// Parse an OpenAI SSE chat-completion stream from an HTTP response,
    /// emitting [`StreamEvent`]s onto the channel.
    async fn parse_openai_sse(
        response: reqwest::Response,
        tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    ) {
        use tokio_util::io::StreamReader;

        let byte_stream = response
            .bytes_stream()
            .map(|result| result.map_err(std::io::Error::other));
        let reader = StreamReader::new(byte_stream);
        Self::parse_openai_sse_lines(reader, tx).await;
    }

    /// Parse an OpenAI SSE chat-completion stream from any [`tokio::io::AsyncBufRead`]
    /// source, emitting [`StreamEvent`]s onto the channel.
    ///
    /// OpenAI's streaming format:
    /// - Each event is a single line `data: {...}` followed by a blank line.
    /// - The terminal sentinel is `data: [DONE]`.
    /// - Each chunk has `choices[0].delta` with `content` and/or `tool_calls`.
    /// - Tool-call deltas accumulate by `index` across multiple chunks: the
    ///   first delta carries `id` + `function.name`, subsequent deltas carry
    ///   `function.arguments` fragments.
    /// - The final chunk before `[DONE]` carries `usage` (non-final chunks
    ///   typically omit it). Usage is logged but not surfaced — the
    ///   [`StreamEvent`] enum has no Usage variant.
    ///
    /// Receiver-drop on the consumer side is treated as cancellation: any send
    /// failure returns early.
    async fn parse_openai_sse_lines<R>(
        reader: R,
        tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    ) where
        R: tokio::io::AsyncBufRead + Unpin,
    {
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();

        // Tool calls are accumulated by their `index` field. Each index maps
        // to (id, name, arguments). The first delta with a given index carries
        // id+name; subsequent deltas append to arguments.
        let mut tool_calls: HashMap<u64, (String, String, String)> = HashMap::new();
        // Preserve emission order: the order in which indices first appear.
        let mut tool_order: Vec<u64> = Vec::new();

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let json_str = line["data:".len()..].trim();
            if json_str == "[DONE]" {
                break;
            }

            let chunk: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let choices = match chunk.get("choices").and_then(|v| v.as_array()) {
                Some(c) => c,
                None => continue,
            };
            for choice in choices {
                let Some(delta) = choice.get("delta") else {
                    continue;
                };

                // Text delta.
                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                    && !content.is_empty()
                {
                    let send_result = tx
                        .send(Ok(StreamEvent::TextDelta(StreamChunk::delta(
                            content.to_string(),
                        ))))
                        .await;
                    if send_result.is_err() {
                        return;
                    }
                }

                // Tool-call deltas.
                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        let index = tc
                            .get("index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or_default();
                        let entry = tool_calls.entry(index).or_insert_with(|| {
                            tool_order.push(index);
                            (String::new(), String::new(), String::new())
                        });
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str())
                            && !id.is_empty()
                        {
                            entry.0 = id.to_string();
                        }
                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str())
                                && !name.is_empty()
                            {
                                entry.1 = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                entry.2.push_str(args);
                            }
                        }
                    }
                }
            }

            if let Some(usage) = chunk.get("usage") {
                tracing::debug!(usage = %usage, "OpenAI stream usage");
            }
        }

        // Emit accumulated tool calls in the order their indices first appeared.
        for index in tool_order {
            if let Some((id, name, arguments)) = tool_calls.remove(&index)
                && (!id.is_empty() || !name.is_empty() || !arguments.is_empty())
            {
                let id = if id.is_empty() {
                    uuid::Uuid::new_v4().to_string()
                } else {
                    id
                };
                let send_result = tx
                    .send(Ok(StreamEvent::ToolCall(ProviderToolCall {
                        id,
                        name,
                        arguments,
                    })))
                    .await;
                if send_result.is_err() {
                    return;
                }
            }
        }

        let _ = tx.send(Ok(StreamEvent::Final)).await;
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    // ── Provider-family defaults ──
    fn default_base_url(&self) -> Option<&str> {
        Some(BASE_URL)
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!("OpenAI API key not set. Set OPENAI_API_KEY or edit config.toml.")
        })?;

        let temperature = temperature.unwrap_or(self.default_temperature());
        let adjusted_temperature = Self::adjust_temperature_for_model(model, temperature);

        let mut messages = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(Message {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }

        messages.push(Message {
            role: "user".to_string(),
            content: message.to_string(),
        });

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature: adjusted_temperature,
            max_tokens: self.max_tokens,
        };

        let response = self
            .http_client()
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {credential}"))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenAI", response).await);
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.effective_content())
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ProviderChatResponse> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!("OpenAI API key not set. Set OPENAI_API_KEY or edit config.toml.")
        })?;

        let temperature = temperature.unwrap_or(self.default_temperature());
        let adjusted_temperature = Self::adjust_temperature_for_model(model, temperature);

        let tools = Self::convert_tools(request.tools);
        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(request.messages),
            temperature: adjusted_temperature,
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            tools,
            max_tokens: self.max_tokens,
            response_format: request.response_format.cloned(),
            stream: None,
        };

        let response = self
            .http_client()
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {credential}"))
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenAI", response).await);
        }

        let native_response: NativeChatResponse = response.json().await?;
        let usage = native_response.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ProviderChatResponse> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!("OpenAI API key not set. Set OPENAI_API_KEY or edit config.toml.")
        })?;

        let temperature = temperature.unwrap_or(self.default_temperature());
        let adjusted_temperature = Self::adjust_temperature_for_model(model, temperature);

        let native_tools: Option<Vec<NativeToolSpec>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .cloned()
                    .map(parse_native_tool_spec)
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };

        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(messages),
            temperature: adjusted_temperature,
            tool_choice: native_tools.as_ref().map(|_| "auto".to_string()),
            tools: native_tools,
            max_tokens: self.max_tokens,
            // chat_with_tools is the legacy non-ChatRequest path used by
            // some channels; structured output flows through the newer
            // chat() entry point.
            response_format: None,
            stream: None,
        };

        let response = self
            .http_client()
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {credential}"))
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenAI", response).await);
        }

        let native_response: NativeChatResponse = response.json().await?;
        let usage = native_response.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        if let Some(credential) = self.credential.as_ref() {
            self.http_client()
                .get(format!("{}/models", self.base_url))
                .header("Authorization", format!("Bearer {credential}"))
                .send()
                .await?
                .error_for_status()?;
        }
        Ok(())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // OpenAI's /v1/models requires a credential. models.dev is the no-auth
        // path onboard uses before the user has entered a key.
        crate::models_dev::list_models_for("openai").await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_streaming_tool_events(&self) -> bool {
        // OpenAI emits tool_calls progressively in `delta.tool_calls`; we
        // accumulate fragments by index and emit StreamEvent::ToolCall once
        // each call is fully assembled (after [DONE]).
        true
    }

    fn stream_chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        if !options.enabled {
            return stream::once(async { Ok(StreamEvent::Final) }).boxed();
        }

        let credential = match self.credential.as_ref() {
            Some(c) => c.clone(),
            None => {
                return stream::once(async {
                    Err(StreamError::Provider(
                        "OpenAI API key not set. Set OPENAI_API_KEY or edit config.toml."
                            .to_string(),
                    ))
                })
                .boxed();
            }
        };

        let temperature = temperature.unwrap_or(self.default_temperature());
        let adjusted_temperature = Self::adjust_temperature_for_model(model, temperature);
        let tools = Self::convert_tools(request.tools);

        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(request.messages),
            temperature: adjusted_temperature,
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            tools,
            max_tokens: self.max_tokens,
            response_format: request.response_format.cloned(),
            stream: Some(true),
        };

        let client = self.http_client();
        let url = format!("{}/chat/completions", self.base_url);

        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);

        tokio::spawn(async move {
            let response = match client
                .post(&url)
                .header("Authorization", format!("Bearer {credential}"))
                .header("Accept", "text/event-stream")
                .json(&native_request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(StreamError::Http(e.to_string()))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP error: {status}"));
                let _ = tx
                    .send(Err(StreamError::Provider(format!("{status}: {error}"))))
                    .await;
                return;
            }

            Self::parse_openai_sse(response, &tx).await;
        });

        stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        })
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_with_key() {
        let p = OpenAiProvider::new(Some("openai-test-credential"));
        assert_eq!(p.credential.as_deref(), Some("openai-test-credential"));
    }

    #[test]
    fn creates_without_key() {
        let p = OpenAiProvider::new(None);
        assert!(p.credential.is_none());
    }

    #[test]
    fn creates_with_empty_key() {
        let p = OpenAiProvider::new(Some(""));
        assert_eq!(p.credential.as_deref(), Some(""));
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = OpenAiProvider::new(None);
        let result = p.chat_with_system(None, "hello", "gpt-4o", Some(0.7)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_key() {
        let p = OpenAiProvider::new(None);
        let result = p
            .chat_with_system(Some("You are ZeroClaw"), "test", "gpt-4o", Some(0.5))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn request_serializes_with_system_message() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: "You are ZeroClaw".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ],
            temperature: 0.7,
            max_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("gpt-4o"));
    }

    #[test]
    fn request_serializes_without_system() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: 0.0,
            max_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("system"));
        assert!(json.contains("\"temperature\":0.0"));
    }

    #[test]
    fn response_deserializes_single_choice() {
        let json = r#"{"choices":[{"message":{"content":"Hi!"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.effective_content(), "Hi!");
    }

    #[test]
    fn response_deserializes_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices.is_empty());
    }

    #[test]
    fn response_deserializes_multiple_choices() {
        let json = r#"{"choices":[{"message":{"content":"A"}},{"message":{"content":"B"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 2);
        assert_eq!(resp.choices[0].message.effective_content(), "A");
    }

    #[test]
    fn response_with_unicode() {
        let json = r#"{"choices":[{"message":{"content":"Hello \u03A9"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.effective_content(),
            "Hello \u{03A9}"
        );
    }

    #[test]
    fn response_with_long_content() {
        let long = "x".repeat(100_000);
        let json = format!(r#"{{"choices":[{{"message":{{"content":"{long}"}}}}]}}"#);
        let resp: ChatResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_ref().unwrap().len(),
            100_000
        );
    }

    #[tokio::test]
    async fn warmup_without_key_is_noop() {
        let provider = OpenAiProvider::new(None);
        let result = provider.warmup().await;
        assert!(result.is_ok());
    }

    // ----------------------------------------------------------
    // Reasoning model fallback tests (reasoning_content)
    // ----------------------------------------------------------

    #[test]
    fn reasoning_content_fallback_empty_content() {
        let json = r#"{"choices":[{"message":{"content":"","reasoning_content":"Thinking..."}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.effective_content(), "Thinking...");
    }

    #[test]
    fn reasoning_content_fallback_null_content() {
        let json =
            r#"{"choices":[{"message":{"content":null,"reasoning_content":"Thinking..."}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.effective_content(), "Thinking...");
    }

    #[test]
    fn reasoning_content_not_used_when_content_present() {
        let json = r#"{"choices":[{"message":{"content":"Hello","reasoning_content":"Ignored"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.effective_content(), "Hello");
    }

    #[test]
    fn native_response_reasoning_content_fallback() {
        let json =
            r#"{"choices":[{"message":{"content":"","reasoning_content":"Native thinking"}}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert_eq!(msg.effective_content(), Some("Native thinking".to_string()));
    }

    #[test]
    fn native_response_reasoning_content_ignored_when_content_present() {
        let json =
            r#"{"choices":[{"message":{"content":"Real answer","reasoning_content":"Ignored"}}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert_eq!(msg.effective_content(), Some("Real answer".to_string()));
    }

    #[tokio::test]
    async fn chat_with_tools_fails_without_key() {
        let p = OpenAiProvider::new(None);
        let messages = vec![ChatMessage::user("hello".to_string())];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" }
                    },
                    "required": ["command"]
                }
            }
        })];
        let result = p
            .chat_with_tools(&messages, &tools, "gpt-4o", Some(0.7))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[tokio::test]
    async fn chat_with_tools_rejects_invalid_tool_shape() {
        let p = OpenAiProvider::new(Some("openai-test-credential"));
        let messages = vec![ChatMessage::user("hello".to_string())];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" }
                    },
                    "required": ["command"]
                }
            }
        })];

        let result = p
            .chat_with_tools(&messages, &tools, "gpt-4o", Some(0.7))
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid OpenAI tool specification")
        );
    }

    #[test]
    fn native_tool_spec_deserializes_from_openai_format() {
        let json = serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" }
                    },
                    "required": ["command"]
                }
            }
        });
        let spec = parse_native_tool_spec(json).unwrap();
        assert_eq!(spec.kind, "function");
        assert_eq!(spec.function.name, "shell");
    }

    #[test]
    fn native_response_parses_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
    }

    #[test]
    fn native_response_parses_without_usage() {
        let json = r#"{"choices": [{"message": {"content": "Hello"}}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // reasoning_content pass-through tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_native_response_captures_reasoning_content() {
        let json = r#"{"choices":[{"message":{
            "content":"answer",
            "reasoning_content":"thinking step",
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{}"}}]
        }}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let message = resp.choices.into_iter().next().unwrap().message;
        let parsed = OpenAiProvider::parse_native_response(message);
        assert_eq!(parsed.reasoning_content.as_deref(), Some("thinking step"));
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[test]
    fn parse_native_response_none_reasoning_content_for_normal_model() {
        let json = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let message = resp.choices.into_iter().next().unwrap().message;
        let parsed = OpenAiProvider::parse_native_response(message);
        assert!(parsed.reasoning_content.is_none());
    }

    #[test]
    fn convert_messages_round_trips_reasoning_content() {
        use zeroclaw_api::provider::ChatMessage;

        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{}"
            }],
            "reasoning_content": "Let me think..."
        });

        let messages = vec![ChatMessage::assistant(history_json.to_string())];
        let native = OpenAiProvider::convert_messages(&messages);
        assert_eq!(native.len(), 1);
        assert_eq!(
            native[0].reasoning_content.as_deref(),
            Some("Let me think...")
        );
    }

    #[test]
    fn convert_messages_no_reasoning_content_when_absent() {
        use zeroclaw_api::provider::ChatMessage;

        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{}"
            }]
        });

        let messages = vec![ChatMessage::assistant(history_json.to_string())];
        let native = OpenAiProvider::convert_messages(&messages);
        assert_eq!(native.len(), 1);
        assert!(native[0].reasoning_content.is_none());
    }

    #[test]
    fn native_message_omits_reasoning_content_when_none() {
        let msg = NativeMessage {
            role: "assistant".to_string(),
            content: Some("hi".to_string()),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("reasoning_content"));
    }

    #[test]
    fn native_message_includes_reasoning_content_when_some() {
        let msg = NativeMessage {
            role: "assistant".to_string(),
            content: Some("hi".to_string()),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: Some("thinking...".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("reasoning_content"));
        assert!(json.contains("thinking..."));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Temperature adjustment tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn adjust_temperature_for_o1_models() {
        assert_eq!(OpenAiProvider::adjust_temperature_for_model("o1", 0.7), 1.0);
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o1-2024-12-17", 0.5),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_for_o3_models() {
        assert_eq!(OpenAiProvider::adjust_temperature_for_model("o3", 0.7), 1.0);
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o3-2025-04-16", 0.5),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o3-mini", 0.3),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o3-mini-2025-01-31", 0.8),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_for_o4_models() {
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o4-mini", 0.7),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("o4-mini-2025-04-16", 0.5),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_for_gpt5_models() {
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5", 0.7),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5-2025-08-07", 0.5),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5-mini", 0.3),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5-mini-2025-08-07", 0.8),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5-nano", 0.6),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5-nano-2025-08-07", 0.4),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_for_gpt5_chat_latest_models() {
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5.1-chat-latest", 0.7),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5.2-chat-latest", 0.5),
            1.0
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-5.3-chat-latest", 0.3),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_preserves_for_standard_models() {
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-4o", 0.7),
            0.7
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-4-turbo", 0.5),
            0.5
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-3.5-turbo", 0.3),
            0.3
        );
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-4", 1.0),
            1.0
        );
    }

    #[test]
    fn adjust_temperature_handles_edge_cases() {
        // Temperature 0.0 should be preserved for standard models
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-4o", 0.0),
            0.0
        );
        // Temperature 1.0 should be preserved for all models
        assert_eq!(OpenAiProvider::adjust_temperature_for_model("o1", 1.0), 1.0);
        assert_eq!(
            OpenAiProvider::adjust_temperature_for_model("gpt-4o", 1.0),
            1.0
        );
    }

    // ── Streaming tests ──────────────────────────────────────────────────

    /// Helper: drive `parse_openai_sse_lines` to completion against a fixed
    /// input and collect every emitted [`StreamEvent`].
    async fn collect_stream_events(input: &str) -> Vec<StreamEvent> {
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);
        let cursor = std::io::Cursor::new(input.as_bytes().to_vec());
        OpenAiProvider::parse_openai_sse_lines(cursor, &tx).await;
        drop(tx);
        let mut events = Vec::new();
        while let Some(result) = rx.recv().await {
            events.push(result.expect("parser produced an error result"));
        }
        events
    }

    #[tokio::test]
    async fn stream_chat_disabled_emits_only_final() {
        let provider = OpenAiProvider::new(Some("test-key"));
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let req = ProviderChatRequest {
            messages: &messages,
            tools: None,
            response_format: None,
        };
        let mut stream = provider.stream_chat(req, "gpt-4o-mini", None, StreamOptions::default());
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev.expect("disabled stream should not produce errors"));
        }
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Final));
    }

    #[tokio::test]
    async fn stream_chat_without_credential_returns_provider_error() {
        let provider = OpenAiProvider::new(None);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let req = ProviderChatRequest {
            messages: &messages,
            tools: None,
            response_format: None,
        };
        let mut stream = provider.stream_chat(req, "gpt-4o-mini", None, StreamOptions::new(true));
        let event = stream
            .next()
            .await
            .expect("should yield at least one event");
        match event {
            Err(StreamError::Provider(msg)) => assert!(msg.contains("API key not set")),
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_sse_lines_emits_text_deltas_in_order() {
        // Three content deltas + DONE sentinel.
        let input = concat!(
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":", "}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":"world!"}}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream_events(input).await;
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta(c) => Some(c.delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello".to_string(), ", ".into(), "world!".into()]);
        assert!(matches!(events.last(), Some(StreamEvent::Final)));
    }

    #[tokio::test]
    async fn parse_sse_lines_assembles_tool_call_across_deltas() {
        // OpenAI streams tool_call arguments as a sequence of fragments. The
        // first delta carries id+name; subsequent deltas append fragments.
        let input = concat!(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"web_search","arguments":""}}]}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"que"}}]}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ry\":\"rust\"}"}}]}}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream_events(input).await;
        let tool_calls: Vec<&ProviderToolCall> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].name, "web_search");
        assert_eq!(tool_calls[0].arguments, r#"{"query":"rust"}"#);
        assert!(matches!(events.last(), Some(StreamEvent::Final)));
    }

    #[tokio::test]
    async fn parse_sse_lines_emits_multiple_tool_calls_in_index_order() {
        // Two parallel tool calls (different indices). Server streams them
        // interleaved; we should emit them in the order their indices first
        // appeared.
        let input = concat!(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_first","type":"function","function":{"name":"alpha","arguments":"{}"}}]}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_second","type":"function","function":{"name":"beta","arguments":"{}"}}]}}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream_events(input).await;
        let names: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall(tc) => Some(tc.name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".into()]);
    }

    #[tokio::test]
    async fn parse_sse_lines_skips_malformed_chunks() {
        // A chunk with invalid JSON between two valid chunks should be
        // ignored, not abort the stream.
        let input = concat!(
            r#"data: {"choices":[{"delta":{"content":"first"}}]}"#,
            "\n\n",
            "data: {bad json}\n\n",
            r#"data: {"choices":[{"delta":{"content":"second"}}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream_events(input).await;
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta(c) => Some(c.delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first".to_string(), "second".into()]);
    }

    #[tokio::test]
    async fn parse_sse_lines_treats_done_as_terminal() {
        // Anything after [DONE] must not be emitted (server may send keepalive
        // comments after the sentinel; the parser should stop reading).
        let input = concat!(
            r#"data: {"choices":[{"delta":{"content":"before"}}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
            r#"data: {"choices":[{"delta":{"content":"after"}}]}"#,
            "\n\n",
        );
        let events = collect_stream_events(input).await;
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta(c) => Some(c.delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["before".to_string()]);
    }

    #[test]
    fn supports_streaming_capabilities_advertised() {
        let p = OpenAiProvider::new(Some("test"));
        assert!(p.supports_streaming());
        assert!(p.supports_streaming_tool_events());
    }

    #[test]
    fn native_request_serializes_response_format_when_present() {
        let schema = serde_json::json!({"type": "json_object"});
        let req = NativeChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![],
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            max_tokens: None,
            response_format: Some(schema.clone()),
            stream: Some(true),
        };
        let body = serde_json::to_value(&req).expect("serialize");
        assert_eq!(body["response_format"], schema);
        assert_eq!(body["stream"], serde_json::Value::Bool(true));
    }

    #[test]
    fn native_request_omits_response_format_when_absent() {
        let req = NativeChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![],
            temperature: 0.7,
            tools: None,
            tool_choice: None,
            max_tokens: None,
            response_format: None,
            stream: None,
        };
        let body = serde_json::to_value(&req).expect("serialize");
        assert!(body.get("response_format").is_none(), "should be omitted");
        assert!(body.get("stream").is_none(), "should be omitted");
    }
}
