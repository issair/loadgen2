//! TTFT (Time to First Token) Calculator for mock_openai server.
//!
//! Sends streaming chat completion requests to a mock server and measures
//! the elapsed time between sending the request and receiving the first
//! response token.

use clap::Parser;
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "ttft_calc", about = "Measure TTFT of a mock OpenAI server")]
struct Args {
    /// Server base URL
    #[arg(long, default_value = "http://127.0.0.1:21000")]
    url: String,

    /// Number of measurement requests to send
    #[arg(long, default_value_t = 100)]
    requests: usize,

    /// Number of warmup requests (not included in results)
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Concurrency level
    #[arg(long, default_value_t = 1)]
    concurrent: usize,

    /// Prompt text to send (overrides prompt-length)
    #[arg(long)]
    prompt: Option<String>,

    /// Length of auto-generated prompt (ignored when --prompt is set)
    #[arg(long, default_value_t = 128)]
    prompt_length: usize,

    /// Max tokens for each request
    #[arg(long, default_value_t = 50)]
    max_tokens: u32,

    /// Print per-request latencies
    #[arg(long)]
    verbose: bool,

    /// Timeout per request (seconds)
    #[arg(long, default_value_t = 30)]
    timeout: u64,

    /// Model name sent in the request body
    #[arg(long, default_value = "ttft-test-model")]
    model: String,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request / SSE response types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    max_tokens: u32,
    stream_options: StreamOptions,
}

#[derive(serde::Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(serde::Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(serde::Deserialize, Debug)]
struct StreamChunk {
    #[allow(dead_code)]
    choices: Vec<ChunkChoice>,
}

#[derive(serde::Deserialize, Debug)]
struct ChunkChoice {
    delta: Delta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct Delta {
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.2} ms", secs * 1000.0)
    } else {
        format!("{:.3} s", secs)
    }
}

fn mean(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

fn stddev(values: &[f64], mean_val: f64) -> f64 {
    let variance = values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * pct / 100.0).ceil() as usize;
    let idx = idx.max(1).min(sorted.len()) - 1;
    sorted[idx]
}

// ---------------------------------------------------------------------------
// SSE line parser (lightweight, no external dep needed)
// ---------------------------------------------------------------------------

/// Returns the first "data: ..." line content, ignoring comments and empty lines.
fn parse_sse_line(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.starts_with("data:") {
        let data = line["data:".len()..].trim();
        if data == "[DONE]" {
            return None; // stream finished
        }
        Some(data)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Single TTFT measurement
// ---------------------------------------------------------------------------

async fn measure_ttft(
    client: &reqwest::Client,
    url: &str,
    prompt: &str,
    max_tokens: u32,
    model: &str,
) -> Result<Duration, String> {
    let body = ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".into(),
            content: prompt.to_string(),
        }],
        stream: true,
        max_tokens,
        stream_options: StreamOptions {
            include_usage: false,
        },
    };

    let start = Instant::now();

    let response = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut first_content_received = false;
    let mut ttft = Duration::ZERO;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if let Some(data_str) = parse_sse_line(line) {
                // Try to parse the JSON; guard against non-JSON "data:" lines.
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data_str) {
                    if let Some(choice) = chunk.choices.first() {
                        if choice.delta.content.is_some() && !first_content_received {
                            first_content_received = true;
                            ttft = start.elapsed();
                        }
                    }
                }
            }
        }

        if first_content_received {
            // Drain the rest of the stream without processing
            while let Some(_r) = stream.next().await {
                // consume silently
            }
            break;
        }
    }

    if !first_content_received {
        return Err("stream ended without any content token".into());
    }

    Ok(ttft)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            // Generate a prompt of the requested length
            let line = "The quick brown fox jumps over the lazy dog. ";
            let mut s = String::with_capacity(args.prompt_length);
            while s.len() < args.prompt_length {
                s.push_str(line);
            }
            s.truncate(args.prompt_length);
            s
        }
    };

    let chat_url = format!("{}/v1/chat/completions", args.url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.timeout))
        .build()
        .expect("failed to build reqwest client");

    let semaphore = Arc::new(Semaphore::new(args.concurrent));

    // -----------------------------------------------------------------------
    // Warmup phase
    // -----------------------------------------------------------------------
    if args.warmup > 0 {
        println!("Warming up with {} request(s) ...", args.warmup);
        let warmup_sem = Arc::clone(&semaphore);
        let warmup_client = client.clone();
        let warmup_prompt = prompt.clone();
        let warmup_url = chat_url.clone();
        let warmup_model = args.model.clone();

        let handles: Vec<_> = (0..args.warmup)
            .map(|_i| {
                let sem = Arc::clone(&warmup_sem);
                let c = warmup_client.clone();
                let p = warmup_prompt.clone();
                let u = warmup_url.clone();
                let m = warmup_model.clone();
                tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    let _ = measure_ttft(&c, &u, &p, args.max_tokens, &m).await;
                    // Ignore warmup results
                })
            })
            .collect();

        for h in handles {
            let _ = h.await;
        }
        println!("Warmup complete.\n");
    }

    // -----------------------------------------------------------------------
    // Measurement phase
    // -----------------------------------------------------------------------

    let total_requests = args.requests;
    let results: Arc<std::sync::Mutex<Vec<Duration>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(total_requests)));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    println!(
        "Sending {} request(s) with concurrency {} ...",
        total_requests, args.concurrent
    );
    if args.verbose {
        println!("---");
    }

    let handles: Vec<_> = (0..total_requests)
        .map(|_i| {
            let sem = Arc::clone(&semaphore);
            let c = client.clone();
            let p = prompt.clone();
            let u = chat_url.clone();
            let results = Arc::clone(&results);
            let completed = Arc::clone(&completed);
            let m = args.model.clone();

            tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                match measure_ttft(&c, &u, &p, args.max_tokens, &m).await {
                    Ok(ttft) => {
                        results.lock().unwrap().push(ttft);
                        if args.verbose {
                            println!(
                                "  [{:4}/{:<4}] TTFT = {}",
                                completed.load(std::sync::atomic::Ordering::Relaxed) + 1,
                                total_requests,
                                fmt_duration(ttft)
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "  [{:4}/{:<4}] ERROR: {e}",
                            completed.load(std::sync::atomic::Ordering::Relaxed) + 1,
                            total_requests,
                        );
                    }
                }
                completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        let _ = h.await;
    }

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------

    let latencies: Vec<Duration> = results.lock().unwrap().clone();

    if latencies.is_empty() {
        eprintln!("\nNo successful measurements. Aborting.");
        std::process::exit(1);
    }

    let failed = total_requests - latencies.len();

    // Convert to milliseconds for statistics
    let mut ms_values: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_ms = mean(&ms_values);
    let std_ms = stddev(&ms_values, mean_ms);
    let min_ms = ms_values.first().copied().unwrap_or(0.0);
    let max_ms = ms_values.last().copied().unwrap_or(0.0);
    let p50 = percentile(&ms_values, 50.0);
    let p90 = percentile(&ms_values, 90.0);
    let p95 = percentile(&ms_values, 95.0);
    let p99 = percentile(&ms_values, 99.0);

    println!("\n═══════════════════════════════════════════");
    println!("       TTFT Measurement Results");
    println!("═══════════════════════════════════════════");
    println!("  Server URL:      {}", args.url);
    println!("  Prompt length:   {} chars", prompt.len());
    println!("  Max tokens:      {}", args.max_tokens);
    println!(
        "  Requests:        {}  ({} succeeded, {} failed)",
        total_requests,
        latencies.len(),
        failed
    );
    println!("  Concurrency:     {}", args.concurrent);
    println!("───────────────────────────────────────────");
    println!(
        "  Min:             {:>10}",
        fmt_duration(Duration::from_secs_f64(min_ms / 1000.0))
    );
    println!(
        "  Max:             {:>10}",
        fmt_duration(Duration::from_secs_f64(max_ms / 1000.0))
    );
    println!(
        "  Mean (Avg):      {:>10}",
        fmt_duration(Duration::from_secs_f64(mean_ms / 1000.0))
    );
    println!("  Std Dev:         {:>10.3} ms", std_ms);
    println!("───────────────────────────────────────────");
    println!(
        "  Median (P50):    {:>10}",
        fmt_duration(Duration::from_secs_f64(p50 / 1000.0))
    );
    println!(
        "  P90:             {:>10}",
        fmt_duration(Duration::from_secs_f64(p90 / 1000.0))
    );
    println!(
        "  P95:             {:>10}",
        fmt_duration(Duration::from_secs_f64(p95 / 1000.0))
    );
    println!(
        "  P99:             {:>10}",
        fmt_duration(Duration::from_secs_f64(p99 / 1000.0))
    );
    println!("═══════════════════════════════════════════");
}
