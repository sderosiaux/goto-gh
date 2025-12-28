//! Core fetch logic for retrieving repository metadata from GitHub
//!
//! Provides batch fetching functionality used by both CLI and server modes.

use anyhow::Result;
use std::collections::HashSet;

use crate::db::Database;
use crate::embedding::build_embedding_text;
use crate::github::GitHubClient;
use crate::papers::extract_github_repos;

/// Result from fetching a batch of repos
#[derive(Debug, Default)]
pub struct FetchBatchResult {
    /// Number of repos successfully fetched and stored
    pub fetched: usize,
    /// Number of repos marked as gone (deleted/renamed)
    pub gone: usize,
    /// Number of new repos discovered from READMEs
    pub discovered: usize,
}

/// Fetch a batch of repos from GitHub and store metadata
///
/// Returns the result with counts of fetched, gone, and discovered repos.
/// Does NOT generate embeddings - that's a separate step.
pub async fn fetch_batch(
    client: &GitHubClient,
    db: &Database,
    repo_names: &[String],
    concurrency: usize,
) -> Result<FetchBatchResult> {
    if repo_names.is_empty() {
        return Ok(FetchBatchResult::default());
    }

    let fetched_repos = client.fetch_repos_batch(repo_names, concurrency).await?;

    let fetched_count = fetched_repos.len();
    let fetched_names: HashSet<_> = fetched_repos
        .iter()
        .map(|r| r.full_name.to_lowercase())
        .collect();

    let mut discovered = 0;

    for repo in fetched_repos {
        let text = build_embedding_text(
            &repo.full_name,
            repo.description.as_deref(),
            &repo.topics,
            repo.language.as_deref(),
            repo.readme.as_deref(),
        );

        db.upsert_repo_metadata_only(&repo, &text)?;

        // Extract linked repos from README for organic growth
        if let Some(readme) = &repo.readme {
            discovered += discover_repos_from_readme(db, readme);
        }
    }

    // Mark missing repos as gone
    let missing: Vec<String> = repo_names
        .iter()
        .filter(|name| !fetched_names.contains(&name.to_lowercase()))
        .cloned()
        .collect();

    let gone_count = if !missing.is_empty() {
        db.mark_as_gone_bulk(&missing)?
    } else {
        0
    };

    Ok(FetchBatchResult {
        fetched: fetched_count,
        gone: gone_count,
        discovered,
    })
}

/// Discover and add repo stubs from a README
pub fn discover_repos_from_readme(db: &Database, readme: &str) -> usize {
    let linked = extract_github_repos(readme);
    if linked.is_empty() {
        return 0;
    }

    let unique: HashSet<_> = linked.into_iter().collect();
    let repos_vec: Vec<String> = unique.into_iter().collect();

    match db.add_repo_stubs_bulk(&repos_vec) {
        Ok((added, _)) => added,
        Err(_) => 0,
    }
}

/// Configuration for the fetch runner
#[derive(Clone)]
pub struct FetchRunnerConfig {
    pub batch_size: usize,
    pub concurrency: usize,
    pub limit: Option<usize>,
    pub debug: bool,
    pub delay_ms: u64,
}

impl Default for FetchRunnerConfig {
    fn default() -> Self {
        Self {
            batch_size: 300,
            concurrency: 2,
            limit: None,
            debug: false,
            delay_ms: 50,
        }
    }
}

/// Callback for fetch progress updates
pub type FetchProgressCallback = Box<dyn Fn(&FetchProgress) + Send>;

/// Progress information during fetch
#[derive(Debug, Clone)]
pub struct FetchProgress {
    pub batch_num: usize,
    pub batch_size: usize,
    pub fetched_this_batch: usize,
    pub gone_this_batch: usize,
    pub discovered_this_batch: usize,
    pub total_fetched: usize,
    pub total_gone: usize,
    pub total_discovered: usize,
}

/// Result from a complete fetch run
#[derive(Debug, Default)]
pub struct FetchRunResult {
    pub total_fetched: usize,
    pub total_gone: usize,
    pub total_discovered: usize,
    pub batches_processed: usize,
}

/// Run fetch loop until no more work or limit reached
///
/// The `on_progress` callback is called after each batch.
/// The `should_stop` function is called before each batch to allow early termination.
pub async fn run_fetch_loop<F, S>(
    client: &GitHubClient,
    db: &Database,
    config: &FetchRunnerConfig,
    mut on_progress: F,
    should_stop: S,
) -> Result<FetchRunResult>
where
    F: FnMut(&FetchProgress),
    S: Fn() -> bool,
{
    let mut result = FetchRunResult::default();
    let limit = config.limit.unwrap_or(usize::MAX);

    loop {
        if should_stop() {
            break;
        }

        if result.total_fetched + result.total_gone >= limit {
            break;
        }

        let remaining = limit - result.total_fetched - result.total_gone;
        let batch_size = config.batch_size.min(remaining);

        let query_start = std::time::Instant::now();
        let repos_to_fetch = db.get_repos_without_metadata(Some(batch_size))?;
        let query_ms = query_start.elapsed().as_millis();

        if repos_to_fetch.is_empty() {
            break;
        }

        result.batches_processed += 1;

        if config.debug && query_ms > 100 {
            eprintln!(
                "\x1b[36m[fetch]\x1b[0m \x1b[90mfound {} repos in {}ms\x1b[0m",
                repos_to_fetch.len(), query_ms
            );
        }

        let batch_result = fetch_batch(client, db, &repos_to_fetch, config.concurrency).await?;

        result.total_fetched += batch_result.fetched;
        result.total_gone += batch_result.gone;
        result.total_discovered += batch_result.discovered;

        let progress = FetchProgress {
            batch_num: result.batches_processed,
            batch_size: repos_to_fetch.len(),
            fetched_this_batch: batch_result.fetched,
            gone_this_batch: batch_result.gone,
            discovered_this_batch: batch_result.discovered,
            total_fetched: result.total_fetched,
            total_gone: result.total_gone,
            total_discovered: result.total_discovered,
        };

        on_progress(&progress);

        // Checkpoint periodically
        if result.batches_processed % 10 == 0 {
            let _ = db.checkpoint();
        }

        // Small delay between batches to avoid rate limits
        if config.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
        }
    }

    let _ = db.checkpoint();
    Ok(result)
}
