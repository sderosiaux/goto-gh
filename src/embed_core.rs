//! Core embedding logic for generating vector embeddings
//!
//! Provides batch embedding functionality used by both CLI and server modes.

use anyhow::Result;

use crate::db::Database;
use crate::embedding::{embed_passages, EmbedProvider, EMBEDDING_DIM};
use crate::openai::{OpenAIClient, OPENAI_EMBEDDING_DIM};

/// Result from embedding a batch of repos
#[derive(Debug, Default)]
pub struct EmbedBatchResult {
    /// Number of embeddings successfully stored
    pub embedded: usize,
    /// Number of tokens used (OpenAI only)
    pub tokens: u32,
}

/// Embed a batch of repos and store the embeddings
pub async fn embed_batch(
    db: &Database,
    repos: &[(i64, String)],
    provider: &EmbedProvider,
    debug: bool,
) -> Result<EmbedBatchResult> {
    if repos.is_empty() {
        return Ok(EmbedBatchResult::default());
    }

    let texts: Vec<String> = repos.iter().map(|(_, text)| text.clone()).collect();
    let repo_ids: Vec<i64> = repos.iter().map(|(id, _)| *id).collect();

    let (embeddings, tokens) = match provider {
        EmbedProvider::Local => {
            let embs = embed_passages(&texts)?;
            (embs, 0)
        }
        EmbedProvider::OpenAI { api_key } => {
            let client = OpenAIClient::new(api_key.clone(), debug);
            let (embs, tokens) = client.embed_batch(&texts).await?;
            (embs, tokens)
        }
    };

    let mut embedded = 0;
    for (repo_id, embedding) in repo_ids.into_iter().zip(embeddings) {
        if db.upsert_embedding(repo_id, &embedding).is_ok() {
            embedded += 1;
        }
    }

    Ok(EmbedBatchResult { embedded, tokens })
}

/// Configuration for the embed runner
#[derive(Clone)]
pub struct EmbedRunnerConfig {
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub delay_ms: u64,
    pub provider: EmbedProvider,
    pub debug: bool,
    pub reset: bool,
}

impl Default for EmbedRunnerConfig {
    fn default() -> Self {
        Self {
            batch_size: 200,
            limit: None,
            delay_ms: 0,
            provider: EmbedProvider::Local,
            debug: false,
            reset: false,
        }
    }
}

/// Progress information during embedding
#[derive(Debug, Clone)]
pub struct EmbedProgress {
    pub batch_num: usize,
    pub batch_size: usize,
    pub embedded_this_batch: usize,
    pub tokens_this_batch: u32,
    pub total_embedded: usize,
    pub total_tokens: u32,
    pub total_to_process: usize,
}

/// Result from a complete embed run
#[derive(Debug, Default)]
pub struct EmbedRunResult {
    pub total_embedded: usize,
    pub total_tokens: u32,
    pub batches_processed: usize,
}

/// Check and handle dimension mismatch
///
/// Returns Ok(true) if dimensions match or table was created/reset.
/// Returns Ok(false) if there's a mismatch and reset was not requested.
pub fn check_embedding_dimension(
    db: &Database,
    provider: &EmbedProvider,
    reset: bool,
) -> Result<bool> {
    let current_dim = db.get_embedding_dimension()?;
    let target_dim = provider.dimension();

    match current_dim {
        Some(dim) if dim != target_dim => {
            // Dimension mismatch
            if reset {
                db.recreate_embeddings_table(target_dim)?;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Some(_) => {
            // Dimensions match
            if reset {
                db.recreate_embeddings_table(target_dim)?;
            }
            Ok(true)
        }
        None => {
            // Table doesn't exist or is empty - create it with correct dimension
            db.ensure_embeddings_table(target_dim)?;
            Ok(true)
        }
    }
}

/// Auto-detect provider from existing embeddings
pub fn auto_detect_provider(db: &Database, requested: &str) -> Result<EmbedProvider> {
    let current_dim = db.get_embedding_dimension()?;

    let provider_str = if let Some(dim) = current_dim {
        if dim == OPENAI_EMBEDDING_DIM {
            "openai"
        } else if dim == EMBEDDING_DIM {
            "local"
        } else {
            requested
        }
    } else {
        requested
    };

    EmbedProvider::parse(provider_str)
}

/// Run embed loop until no more work or limit reached
///
/// The `on_progress` callback is called after each batch.
/// The `should_stop` function is called before each batch to allow early termination.
pub async fn run_embed_loop<F, S>(
    db: &Database,
    config: &EmbedRunnerConfig,
    mut on_progress: F,
    should_stop: S,
) -> Result<EmbedRunResult>
where
    F: FnMut(&EmbedProgress),
    S: Fn() -> bool,
{
    // Check dimension compatibility
    if !check_embedding_dimension(db, &config.provider, config.reset)? {
        return Err(anyhow::anyhow!(
            "Embedding dimension mismatch. Use --reset to recreate embeddings table."
        ));
    }

    let mut result = EmbedRunResult::default();

    loop {
        if should_stop() {
            break;
        }

        if let Some(limit) = config.limit {
            if result.total_embedded >= limit {
                break;
            }
        }

        let remaining = config.limit
            .map(|l| l - result.total_embedded)
            .unwrap_or(usize::MAX);
        let batch_size = config.batch_size.min(remaining);

        let query_start = std::time::Instant::now();
        let repos = db.get_repos_without_embeddings(Some(batch_size))?;
        let query_ms = query_start.elapsed().as_millis();

        if repos.is_empty() {
            break;
        }

        result.batches_processed += 1;

        if config.debug {
            eprintln!(
                "\x1b[33m[embed]\x1b[0m \x1b[90mBatch {}: found {} repos in {}ms\x1b[0m",
                result.batches_processed, repos.len(), query_ms
            );
        }

        let batch_result = embed_batch(db, &repos, &config.provider, config.debug).await?;

        result.total_embedded += batch_result.embedded;
        result.total_tokens += batch_result.tokens;

        // Re-query total_to_process each batch (may have grown from fetch/discover)
        let count_start = std::time::Instant::now();
        let current_need = db.count_repos_without_embeddings()?;
        if config.debug && count_start.elapsed().as_millis() > 100 {
            eprintln!(
                "\x1b[33m[embed]\x1b[0m \x1b[90mcount query took {}ms\x1b[0m",
                count_start.elapsed().as_millis()
            );
        }
        let total_to_process = if let Some(limit) = config.limit {
            limit.min(current_need + result.total_embedded)
        } else {
            current_need + result.total_embedded
        };

        let progress = EmbedProgress {
            batch_num: result.batches_processed,
            batch_size: repos.len(),
            embedded_this_batch: batch_result.embedded,
            tokens_this_batch: batch_result.tokens,
            total_embedded: result.total_embedded,
            total_tokens: result.total_tokens,
            total_to_process,
        };

        on_progress(&progress);

        // Checkpoint periodically
        if result.batches_processed % 10 == 0 {
            let _ = db.checkpoint();
        }

        // Delay between batches if configured
        if config.delay_ms > 0 && current_need > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
        }
    }

    let _ = db.checkpoint();
    Ok(result)
}
