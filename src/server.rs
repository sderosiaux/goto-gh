//! Server mode: orchestrates fetch, discover, and embed workers concurrently
//!
//! Each worker runs in its own thread with its own tokio runtime.
//! Uses core functions from fetch, discovery, and embed_core modules.

use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::db::Database;
use crate::discovery::{self, DiscoveryConfig};
use crate::embed_core::{self, EmbedRunnerConfig};
use crate::embedding::EmbedProvider;
use crate::fetch::{self, FetchRunnerConfig};
use crate::github::GitHubClient;
use crate::proxy::ProxyManager;

/// Server configuration
#[derive(Clone)]
pub struct ServerConfig {
    // Fetch settings (GraphQL - metadata only)
    pub fetch_batch_size: usize,
    pub fetch_concurrency: usize,

    // Readme settings (REST API - README content)
    pub readme_concurrency: usize,

    // Discover settings
    pub discover_limit: usize,

    // Embed settings
    pub embed_batch_size: usize,
    pub embed_delay_ms: u64,
    pub embed_provider: EmbedProvider,

    // General
    pub debug: bool,
    pub proxy_file: Option<String>,
    pub force_proxy: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            fetch_batch_size: 300,
            fetch_concurrency: 2,
            readme_concurrency: 20,
            discover_limit: 50,
            embed_batch_size: 200,
            embed_delay_ms: 50,
            embed_provider: EmbedProvider::Local,
            debug: false,
            proxy_file: None,
            force_proxy: false,
        }
    }
}

/// Shared state for all workers
pub struct ServerState {
    pub config: ServerConfig,
    pub github_tokens: Vec<String>,
    /// Shared proxy manager (loaded once at startup)
    pub proxy_manager: Option<ProxyManager>,
}

impl ServerState {
    /// Create a new database connection (each worker gets its own)
    fn open_db(&self) -> Result<Database> {
        Database::open()
    }

    /// Create a GitHub client with a specific token
    fn create_github_client_with_token(&self, token: Option<String>) -> Result<GitHubClient> {
        Ok(GitHubClient::new_with_options(
            token,
            self.config.debug,
            self.proxy_manager.clone(),
            self.config.force_proxy,
        ))
    }

    /// Create a GitHub client with the first available token (for single-token operations)
    fn create_github_client(&self) -> Result<GitHubClient> {
        self.create_github_client_with_token(self.github_tokens.first().cloned())
    }
}

/// Start the server with all workers
pub async fn start_server(config: ServerConfig) -> Result<()> {
    let github_tokens = Config::github_tokens();
    let token_count = github_tokens.len();

    // Load proxy manager once at startup
    let proxy_manager = if let Some(ref path) = config.proxy_file {
        let path = std::path::PathBuf::from(path);
        Some(ProxyManager::from_file(&path)?)
    } else {
        None
    };

    let state = Arc::new(ServerState {
        config,
        github_tokens,
        proxy_manager,
    });

    // Initial stats
    {
        let db = state.open_db()?;
        let without_metadata = db.count_repos_without_metadata()?;
        let without_readme = db.count_repos_without_readme()?;
        let without_embeddings = db.count_repos_without_embeddings()?;
        let unexplored = db.count_unexplored_owners()?;

        eprintln!("\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
        eprintln!("\x1b[36m║       goto-gh server starting        ║\x1b[0m");
        eprintln!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
        eprintln!();
        eprintln!("  \x1b[90mQueue status:\x1b[0m");
        eprintln!("    {} repos need metadata (fetch)", without_metadata);
        eprintln!("    {} repos need README (readme)", without_readme);
        eprintln!("    {} repos need embeddings (embed)", without_embeddings);
        eprintln!("    {} owners to explore (discover)", unexplored);
        eprintln!();
        if token_count > 1 {
            eprintln!("  \x1b[90mTokens:\x1b[0m {} ({}x throughput)", token_count, token_count);
        }
        eprintln!("  \x1b[90mWorkers:\x1b[0m fetch x{}, readme, discover, embed", token_count.max(1));
        eprintln!();
        eprintln!("  \x1b[33mPress Ctrl+C to stop gracefully\x1b[0m");
        eprintln!();
    }

    // Shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));

    // Spawn one fetch worker per token (parallel throughput)
    let mut fetch_handles = Vec::new();
    if state.github_tokens.is_empty() {
        // No tokens - spawn single worker without auth
        let fetch_shutdown = shutdown.clone();
        let fetch_state = state.clone();
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(run_fetch_worker_loop(fetch_state, fetch_shutdown, None, 0));
        });
        fetch_handles.push(handle);
    } else {
        // Spawn one worker per token
        for (idx, token) in state.github_tokens.iter().enumerate() {
            let fetch_shutdown = shutdown.clone();
            let fetch_state = state.clone();
            let token = token.clone();
            let handle = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(run_fetch_worker_loop(fetch_state, fetch_shutdown, Some(token), idx));
            });
            fetch_handles.push(handle);
        }
    }

    let discover_shutdown = shutdown.clone();
    let discover_state = state.clone();
    let discover_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(run_discover_worker_loop(discover_state, discover_shutdown));
    });

    let embed_shutdown = shutdown.clone();
    let embed_state = state.clone();
    let embed_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(run_embed_worker_loop(embed_state, embed_shutdown));
    });

    // Readme worker (REST API for README content)
    let readme_shutdown = shutdown.clone();
    let readme_state = state.clone();
    let readme_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(run_readme_worker_loop(readme_state, readme_shutdown));
    });

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    eprintln!("\n\x1b[33m!\x1b[0m Shutting down workers...");

    // Signal all workers to stop
    shutdown.store(true, Ordering::SeqCst);

    // Wait for workers to finish (they check shutdown flag frequently)
    for handle in fetch_handles {
        let _ = handle.join();
    }
    let _ = discover_handle.join();
    let _ = embed_handle.join();
    let _ = readme_handle.join();

    // Final checkpoint
    if let Ok(db) = Database::open() {
        let _ = db.checkpoint();
        eprintln!("\x1b[32mok\x1b[0m WAL checkpointed");
    }

    eprintln!("\x1b[32mok\x1b[0m Server stopped");
    Ok(())
}

/// Fetch worker loop - runs continuously with a specific token
async fn run_fetch_worker_loop(
    state: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    token: Option<String>,
    worker_idx: usize,
) {
    let worker_label = if state.github_tokens.len() > 1 {
        format!("fetch:{}", worker_idx + 1)
    } else {
        "fetch".to_string()
    };

    eprintln!("\x1b[36m[{}]\x1b[0m Worker started", worker_label);

    while !shutdown.load(Ordering::SeqCst) {
        match run_fetch_cycle(&state, &shutdown, token.clone(), &worker_label).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
            Err(e) => {
                eprintln!("\x1b[36m[{}]\x1b[0m \x1b[31mError: {}\x1b[0m", worker_label, e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }

    eprintln!("\x1b[36m[{}]\x1b[0m Shutting down", worker_label);
}

/// Discover worker loop - runs continuously
async fn run_discover_worker_loop(state: Arc<ServerState>, shutdown: Arc<AtomicBool>) {
    eprintln!("\x1b[35m[discover]\x1b[0m Worker started");

    while !shutdown.load(Ordering::SeqCst) {
        match run_discover_cycle(&state, &shutdown).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
            Err(e) => {
                eprintln!("\x1b[35m[discover]\x1b[0m \x1b[31mError: {}\x1b[0m", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }

    eprintln!("\x1b[35m[discover]\x1b[0m Shutting down");
}

/// Embed worker loop - runs continuously
async fn run_embed_worker_loop(state: Arc<ServerState>, shutdown: Arc<AtomicBool>) {
    eprintln!("\x1b[33m[embed]\x1b[0m Worker started");

    while !shutdown.load(Ordering::SeqCst) {
        match run_embed_cycle(&state, &shutdown).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
            Err(e) => {
                eprintln!("\x1b[33m[embed]\x1b[0m \x1b[31mError: {}\x1b[0m", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }

    eprintln!("\x1b[33m[embed]\x1b[0m Shutting down");
}

/// Fetch cycle using core functions with a specific token
async fn run_fetch_cycle(
    state: &ServerState,
    shutdown: &AtomicBool,
    token: Option<String>,
    worker_label: &str,
) -> Result<bool> {
    let db = state.open_db()?;
    let client = state.create_github_client_with_token(token)?;

    let config = FetchRunnerConfig {
        batch_size: state.config.fetch_batch_size,
        concurrency: state.config.fetch_concurrency,
        limit: None,
        debug: state.config.debug,
        delay_ms: 50,
    };

    let label = worker_label.to_string();
    let result = fetch::run_fetch_loop(
        &client,
        &db,
        &config,
        |progress| {
            // Summary after each batch (similar to discover/embed)
            let discovered_info = if progress.discovered_this_batch > 0 {
                format!(", +{} discovered", progress.discovered_this_batch)
            } else {
                String::new()
            };
            eprintln!(
                "\x1b[36m[{}]\x1b[0m +{} repos{}",
                label, progress.fetched_this_batch, discovered_info
            );
        },
        || shutdown.load(Ordering::SeqCst),
    )
    .await;

    match result {
        Ok(run_result) => {
            if run_result.total_fetched > 0 && state.config.debug {
                eprintln!(
                    "\x1b[36m[{}]\x1b[0m Done: {} fetched, {} gone",
                    worker_label, run_result.total_fetched, run_result.total_gone
                );
            }
            Ok(run_result.total_fetched > 0 || run_result.total_gone > 0)
        }
        Err(e) => Err(e),
    }
}

/// Discover cycle using core functions
async fn run_discover_cycle(state: &ServerState, shutdown: &AtomicBool) -> Result<bool> {
    let db = state.open_db()?;
    let client = state.create_github_client()?;

    let config = DiscoveryConfig {
        limit: None,
        batch_size: state.config.discover_limit,
    };

    let result = discovery::run_discover_loop(
        &client,
        &db,
        &config,
        |progress| {
            if progress.repos_inserted > 0 || progress.followers_added > 0 {
                eprintln!(
                    "\x1b[35m[discover]\x1b[0m {} +{} repos, +{} profiles",
                    progress.owner, progress.repos_inserted, progress.followers_added
                );
            }
        },
        || shutdown.load(Ordering::SeqCst),
    )
    .await;

    match result {
        Ok(run_result) => {
            if run_result.owners_processed > 0 {
                eprintln!(
                    "\x1b[35m[discover]\x1b[0m Done: {} owners, +{} repos, +{} profiles",
                    run_result.owners_processed, run_result.total_repos, run_result.total_profiles
                );
            } else if state.config.debug {
                eprintln!("\x1b[35m[discover]\x1b[0m \x1b[90mNo owners to explore\x1b[0m");
            }
            Ok(run_result.owners_processed > 0)
        }
        Err(e) => Err(e),
    }
}

/// Embed cycle using core functions
async fn run_embed_cycle(state: &ServerState, shutdown: &AtomicBool) -> Result<bool> {
    let db = state.open_db()?;

    // Check dimension compatibility
    if !embed_core::check_embedding_dimension(&db, &state.config.embed_provider, false)? {
        eprintln!(
            "\x1b[33m[embed]\x1b[0m \x1b[31mDimension mismatch, skipping\x1b[0m"
        );
        return Ok(false);
    }

    let config = EmbedRunnerConfig {
        batch_size: state.config.embed_batch_size,
        limit: None,
        delay_ms: state.config.embed_delay_ms,
        provider: state.config.embed_provider.clone(),
        debug: state.config.debug,
        reset: false,
    };

    let result = embed_core::run_embed_loop(
        &db,
        &config,
        |progress| {
            // Summary line after each batch (similar to discover summary)
            eprintln!(
                "\x1b[33m[embed]\x1b[0m +{} repos ({} tokens)",
                progress.embedded_this_batch, progress.tokens_this_batch
            );
        },
        || shutdown.load(Ordering::SeqCst),
    )
    .await;

    match result {
        Ok(run_result) => {
            if run_result.total_embedded > 0 && state.config.debug {
                eprintln!(
                    "\x1b[33m[embed]\x1b[0m Done: {} embeddings, {} tokens",
                    run_result.total_embedded, run_result.total_tokens
                );
            }
            Ok(run_result.total_embedded > 0)
        }
        Err(e) => Err(e),
    }
}

/// Readme worker loop - fetches README content via REST API
async fn run_readme_worker_loop(state: Arc<ServerState>, shutdown: Arc<AtomicBool>) {
    eprintln!("\x1b[32m[readme]\x1b[0m Worker started");

    while !shutdown.load(Ordering::SeqCst) {
        match run_readme_cycle(&state, &shutdown).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
            Err(e) => {
                eprintln!("\x1b[32m[readme]\x1b[0m \x1b[31mError: {}\x1b[0m", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }

    eprintln!("\x1b[32m[readme]\x1b[0m Shutting down");
}

/// Readme cycle - fetch README for repos that need it
async fn run_readme_cycle(state: &ServerState, shutdown: &AtomicBool) -> Result<bool> {
    use futures::stream::{self, StreamExt};

    let db = state.open_db()?;
    let client = state.create_github_client()?;

    let need_readme = db.count_repos_without_readme()?;
    if need_readme == 0 {
        return Ok(false);
    }

    // Process in batches
    let batch_size = 100;
    let repos = db.get_repos_without_readme(Some(batch_size))?;
    if repos.is_empty() {
        return Ok(false);
    }

    let concurrency = state.config.readme_concurrency;

    #[derive(Debug)]
    enum ReadmeResult {
        Found(String),
        NotFound,
        Error(String),
    }

    let mut fetched = 0;
    let mut not_found = 0;
    let mut discovered = 0;

    let mut result_stream = stream::iter(repos)
        .map(|(repo_id, full_name)| {
            let client = client.clone();
            async move {
                match client.get_readme(&full_name).await {
                    Ok(Some(readme)) => (repo_id, full_name, ReadmeResult::Found(readme)),
                    Ok(None) => (repo_id, full_name, ReadmeResult::NotFound),
                    Err(e) => (repo_id, full_name, ReadmeResult::Error(e.to_string())),
                }
            }
        })
        .buffer_unordered(concurrency);

    while let Some((repo_id, full_name, result)) = result_stream.next().await {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        match result {
            ReadmeResult::Found(content) => {
                let len = content.len();
                // Treat empty README as "no readme" - don't retry
                if len == 0 {
                    if state.config.debug {
                        eprintln!("\x1b[32m[readme]\x1b[0m \x1b[90m- {} (empty)\x1b[0m", full_name);
                    }
                    let _ = db.mark_repo_no_readme(repo_id);
                    not_found += 1;
                    continue;
                }
                if let Err(e) = db.update_repo_readme(repo_id, &content) {
                    eprintln!("\x1b[32m[readme]\x1b[0m \x1b[31m{} db error: {}\x1b[0m", full_name, e);
                } else {
                    // Extract linked repos from README
                    let found = fetch::discover_repos_from_readme(&db, &content);
                    discovered += found;
                    let _ = db.mark_repos_extracted(repo_id);
                    fetched += 1;
                    if state.config.debug {
                        if found > 0 {
                            eprintln!("\x1b[32m[readme]\x1b[0m ✓ {} ({} bytes, +{} repos)", full_name, len, found);
                        } else {
                            eprintln!("\x1b[32m[readme]\x1b[0m ✓ {} ({} bytes)", full_name, len);
                        }
                    }
                }
            }
            ReadmeResult::NotFound => {
                if state.config.debug {
                    eprintln!("\x1b[32m[readme]\x1b[0m \x1b[90m- {} (no readme)\x1b[0m", full_name);
                }
                let _ = db.mark_repo_no_readme(repo_id);
                not_found += 1;
            }
            ReadmeResult::Error(_) => {
                // Don't mark as no_readme, we'll retry later
            }
        }
    }

    if fetched > 0 || not_found > 0 {
        let discovered_info = if discovered > 0 {
            format!(", +{} discovered", discovered)
        } else {
            String::new()
        };
        eprintln!(
            "\x1b[32m[readme]\x1b[0m +{} READMEs ({} not found{}) [{} remaining]",
            fetched, not_found, discovered_info, need_readme.saturating_sub(fetched + not_found)
        );
    }

    Ok(fetched > 0 || not_found > 0)
}
