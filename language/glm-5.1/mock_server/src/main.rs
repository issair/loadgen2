use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;
use tracing::info;
use uuid::Uuid;

struct AppState {
    semaphore: Arc<Semaphore>,
    stats: Arc<Stats>,
    dump_path: Option<String>,
    token_delay_ms: u64,
}

impl AppState {
    /// If `dump_path` is set, append a JSON line to that file with the given label and value.
    async fn dump(&self, label: &str, value: &Value) {
        let path = match &self.dump_path {
            Some(p) => p.clone(),
            None => return,
        };
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let line = serde_json::json!({
            "label": label,
            "timestamp": timestamp,
            "data": value,
        });
        let line_str = serde_json::to_string(&line).unwrap() + "\n";
        if let Ok(mut file) = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
        {
            let _ = file.write_all(line_str.as_bytes()).await;
        }
    }
}

#[derive(Default)]
struct Stats {
    /// Total requests received.
    total_requests: AtomicU64,
    /// Currently in-flight (acquired semaphore permits).
    in_flight: AtomicU64,
}

/// RAII guard that decrements `in_flight` on drop.
///
/// Ensures the counter stays accurate even when the SSE stream is
/// cancelled mid-flight by a client disconnect.
struct InFlightGuard {
    stats: Arc<Stats>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.stats.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Periodic stats logger. Spawned only when interval > 0.
async fn stats_logger(state: Arc<AppState>, interval_secs: u64) {
    let mut tick = tokio::time::interval(Duration::from_secs(interval_secs));
    loop {
        tick.tick().await;
        let total = state.stats.total_requests.load(Ordering::Relaxed);
        let in_flight = state.stats.in_flight.load(Ordering::Relaxed);
        let available = state.semaphore.available_permits();
        info!(
            "[STATS] total_requests={}  in_flight={}  available_permits={}",
            total, in_flight, available
        );
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<Message>,
    stream: Option<bool>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    delta: Delta,
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
struct Usage {
    completion_tokens: u32,
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ResponseChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ResponseChoice {
    index: u32,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

fn extract_input(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|m| m.content.as_deref())
        .filter(|c| !c.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

fn tokenize(text: &str, max_tokens: u32) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let limit = if max_tokens == 0 || max_tokens as usize >= chars.len() {
        chars.len()
    } else {
        max_tokens as usize
    };
    chars[..limit].iter().map(|c| c.to_string()).collect()
}

async fn chat_completions_nonstream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    state.stats.total_requests.fetch_add(1, Ordering::Relaxed);

    let _permit = match state.semaphore.acquire().await {
        Ok(p) => {
            state.stats.in_flight.fetch_add(1, Ordering::Relaxed);
            p
        }
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "server busy"})),
            )
                .into_response();
        }
    };

    state
        .dump("request", &serde_json::to_value(&req).unwrap())
        .await;

    let input = extract_input(&req.messages);
    let prompt_tokens = input.chars().count() as u32;
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(u32::MAX);
    let tokens = tokenize(&input, max_tokens);
    let completion_tokens = tokens.len() as u32;
    let model = req.model.unwrap_or_else(|| "mock-model".into());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created,
        model: model.clone(),
        choices: vec![ResponseChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content: tokens.join(""),
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            completion_tokens,
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    state
        .dump("response", &serde_json::to_value(&resp).unwrap())
        .await;

    state.stats.in_flight.fetch_sub(1, Ordering::Relaxed);

    (StatusCode::OK, Json(resp)).into_response()
}

async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    state.stats.total_requests.fetch_add(1, Ordering::Relaxed);
    let permit = state.semaphore.clone().acquire_owned().await.unwrap();
    state.stats.in_flight.fetch_add(1, Ordering::Relaxed);

    state
        .dump("request", &serde_json::to_value(&req).unwrap())
        .await;

    let input = extract_input(&req.messages);
    let prompt_tokens = input.chars().count() as u32;
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(u32::MAX);
    let tokens = tokenize(&input, max_tokens);
    let completion_tokens = tokens.len() as u32;
    let model = req.model.unwrap_or_else(|| "mock-model".into());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(true);

    let state_for_dump = Arc::clone(&state);
    let token_delay_ms = state.token_delay_ms;

    // RAII guard: decrements in_flight when the stream is dropped,
    // whether it completes normally or is cancelled by client disconnect.
    let guard = InFlightGuard {
        stats: Arc::clone(&state.stats),
    };

    let stream = async_stream::stream! {
        // Move permit and guard into the stream so they live as long as
        // the SSE connection, not just this function's stack frame.
        let _permit = permit;
        let _guard = guard;

        let id = format!("chatcmpl-{}", Uuid::new_v4());

        for token in tokens.into_iter() {
            let chunk = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    delta: Delta { content: Some(token) },
                    index: 0,
                    finish_reason: None,
                }],
                usage: None,
            };
            state_for_dump
                .dump("stream_chunk", &serde_json::to_value(&chunk).unwrap())
                .await;
            let data = serde_json::to_string(&chunk).unwrap();
            yield Ok::<_, axum::Error>(Event::default().data(data));
            if token_delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(token_delay_ms)).await;
            }
        }

        // Final chunk with finish_reason, empty content, and optional usage
        let final_chunk = StreamChunk {
            id,
            object: "chat.completion.chunk".into(),
            created,
            model,
            choices: vec![ChunkChoice {
                delta: Delta { content: None },
                index: 0,
                finish_reason: Some("stop".into()),
            }],
            usage: if include_usage {
                Some(Usage {
                    completion_tokens,
                    prompt_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                })
            } else {
                None
            },
        };
        state_for_dump
            .dump("stream_final_chunk", &serde_json::to_value(&final_chunk).unwrap())
            .await;
        let data = serde_json::to_string(&final_chunk).unwrap();
        yield Ok::<_, axum::Error>(Event::default().data(data));

        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));

        // _guard and _permit drop here on normal completion.
        // On client disconnect the stream is dropped and they release too.
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn chat_completions_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.stream.unwrap_or(false) {
        chat_completions_stream(state, Json(req))
            .await
            .into_response()
    } else {
        chat_completions_nonstream(state, Json(req))
            .await
            .into_response()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(21000);

    let concurrency: usize = std::env::var("CONCURRENCY")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(1024);

    let stats_interval: u64 = std::env::var("STATS_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let dump_path: Option<String> = std::env::var("DUMP_FILE").ok();
    if let Some(ref path) = dump_path {
        info!("Message dumping enabled, writing to {}", path);
    }

    let token_delay_ms: u64 = std::env::var("TOKEN_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let state = Arc::new(AppState {
        semaphore: Arc::new(Semaphore::new(concurrency)),
        stats: Arc::new(Stats::default()),
        dump_path,
        token_delay_ms,
    });

    if stats_interval > 0 {
        let logger_state = Arc::clone(&state);
        tokio::spawn(async move {
            stats_logger(logger_state, stats_interval).await;
        });
    }

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/chat/completions", post(chat_completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Mock OpenAI server listening on {}", addr);
    info!("Concurrency limit: {}", concurrency);
    if token_delay_ms > 0 {
        info!("Token delay: {}ms", token_delay_ms);
    } else {
        info!("Token delay disabled (set TOKEN_DELAY_MS to enable)");
    }
    if stats_interval > 0 {
        info!("Stats logging interval: {}s", stats_interval);
    } else {
        info!("Stats logging disabled (set STATS_INTERVAL to enable)");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
