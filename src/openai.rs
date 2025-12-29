//! OpenAI embeddings client
//!
//! Uses text-embedding-3-small (1536 dimensions, $0.02/1M tokens)
//! No query/passage prefixes needed - OpenAI models don't use them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tiktoken_rs::CoreBPE;

/// OpenAI embedding dimension for text-embedding-3-small
pub const OPENAI_EMBEDDING_DIM: usize = 1536;

/// Maximum texts per batch (OpenAI limit is 2048)
const MAX_BATCH_SIZE: usize = 2048;

/// Maximum tokens per request (OpenAI limit is 300,000)
/// Using 250K to account for tiktoken estimation variance (~17% margin)
/// Example: tiktoken estimated 279,801 but OpenAI counted 302,026 (8% higher)
const MAX_TOKENS_PER_REQUEST: usize = 250_000;

/// Maximum tokens per individual text (text-embedding-3-small context limit is 8191)
/// Using 5000 to account for tiktoken estimation variance, especially for CJK text
/// tiktoken can underestimate significantly for Chinese/Japanese/Korean characters
const MAX_TOKENS_PER_TEXT: usize = 5000;

/// Singleton tokenizer for cl100k_base (used by text-embedding-3-small)
static TOKENIZER: OnceLock<CoreBPE> = OnceLock::new();

fn get_tokenizer() -> &'static CoreBPE {
    TOKENIZER.get_or_init(|| {
        tiktoken_rs::cl100k_base().expect("Failed to initialize cl100k_base tokenizer")
    })
}

/// OpenAI embeddings client
pub struct OpenAIClient {
    api_key: String,
    client: reqwest::Client,
    debug: bool,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// Count tokens using tiktoken cl100k_base (same as OpenAI uses)
fn count_tokens(text: &str) -> usize {
    get_tokenizer().encode_with_special_tokens(text).len()
}

impl OpenAIClient {
    /// Create a new OpenAI client
    pub fn new(api_key: String, debug: bool) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            debug,
        }
    }

    /// Generate embeddings for multiple texts with automatic token-aware chunking
    /// Returns embeddings in the same order as input texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u32)> {
        if texts.is_empty() {
            return Ok((vec![], 0));
        }

        // Split into token-aware chunks
        let chunks = self.split_by_tokens(texts);

        let mut all_embeddings = Vec::with_capacity(texts.len());
        let mut total_tokens = 0u32;

        for chunk in chunks {
            let (embeddings, tokens) = self.embed_single_batch(&chunk).await?;
            all_embeddings.extend(embeddings);
            total_tokens += tokens;
        }

        Ok((all_embeddings, total_tokens))
    }

    /// Split texts into chunks that fit within token limits
    fn split_by_tokens(&self, texts: &[String]) -> Vec<Vec<String>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_tokens = 0usize;

        for text in texts {
            let text_tokens = count_tokens(text);

            // If this single text exceeds per-text limit, truncate by tokens
            let (text_to_add, tokens_to_add) = if text_tokens > MAX_TOKENS_PER_TEXT {
                let tokenizer = get_tokenizer();
                let tokens = tokenizer.encode_with_special_tokens(text);
                let truncated_tokens = &tokens[..MAX_TOKENS_PER_TEXT];
                // Decode can fail for incomplete UTF-8 sequences at truncation boundary
                // Fallback: use conservative character limit (1 char ≈ 1-3 tokens for CJK)
                let truncated = tokenizer.decode(truncated_tokens.to_vec())
                    .unwrap_or_else(|_| text.chars().take(MAX_TOKENS_PER_TEXT).collect());
                (truncated, MAX_TOKENS_PER_TEXT)
            } else {
                (text.clone(), text_tokens)
            };

            // Check if adding this text would exceed limits
            let would_exceed_tokens = current_tokens + tokens_to_add > MAX_TOKENS_PER_REQUEST;
            let would_exceed_count = current_chunk.len() >= MAX_BATCH_SIZE;

            if !current_chunk.is_empty() && (would_exceed_tokens || would_exceed_count) {
                chunks.push(std::mem::take(&mut current_chunk));
                current_tokens = 0;
            }

            current_chunk.push(text_to_add);
            current_tokens += tokens_to_add;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Send a single batch to OpenAI (assumes it fits within limits)
    /// Retries on transient errors (502, 503, 504, 429) with exponential backoff
    async fn embed_single_batch(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u32)> {
        let request = EmbeddingRequest {
            model: "text-embedding-3-small",
            input: texts,
        };

        // Retry with exponential backoff for transient errors
        let max_retries = 5;
        let mut last_error = None;

        for attempt in 0..max_retries {
            if attempt > 0 {
                let delay_ms = 1000 * (1 << attempt.min(4)); // 2s, 4s, 8s, 16s
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }

            let start = std::time::Instant::now();

            let response = match self
                .client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(format!("Request failed: {}", e));
                    continue;
                }
            };

            let elapsed = start.elapsed();
            let status = response.status();

            if status.is_success() {
                let result: EmbeddingResponse = response
                    .json()
                    .await
                    .context("Failed to parse OpenAI response")?;

                // Sort by index to ensure correct order
                let mut embeddings: Vec<_> = result.data.into_iter().collect();
                embeddings.sort_by_key(|e| e.index);

                let vectors: Vec<Vec<f32>> = embeddings.into_iter().map(|e| e.embedding).collect();
                let tokens_used = result.usage.total_tokens;

                // Log in same style as discover: [timestamp] URL ... timing
                if self.debug {
                    let now = chrono::Local::now().format("%H:%M:%S%.3f");
                    let retry_info = if attempt > 0 { format!(" (retry {})", attempt) } else { String::new() };
                    eprintln!(
                        "\x1b[90m[{}] POST https://api.openai.com/v1/embeddings ({} repos) ... {}ms{}\x1b[0m",
                        now, texts.len(), elapsed.as_millis(), retry_info
                    );
                }

                return Ok((vectors, tokens_used));
            }

            let body = response.text().await.unwrap_or_default();

            // Retry on transient errors
            let is_transient = status == reqwest::StatusCode::BAD_GATEWAY
                || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS;

            if !is_transient {
                // Non-transient error, fail immediately
                return Err(anyhow::anyhow!("OpenAI API error ({}): {}", status, body));
            }

            last_error = Some(format!("OpenAI API error ({})", status));
        }

        Err(anyhow::anyhow!(
            "OpenAI API failed after {} retries: {}",
            max_retries,
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ))
    }

}
