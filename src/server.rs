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
    // Fetch settings
    pub fetch_batch_size: usize,
    pub fetch_concurrency: usize,

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
    pub github_token: Option<String>,
}

impl ServerState {
    /// Create a new database connection (each worker gets its own)
    fn open_db(&self) -> Result<Database> {
        Database::open()
    }

    /// Create a GitHub client with the configured options
    fn create_github_client(&self) -> Result<GitHubClient> {
        let proxy_manager = if let Some(ref path) = self.config.proxy_file {
            let path = std::path::PathBuf::from(path);
            Some(ProxyManager::from_file(&path)?)
        } else {
            None
        };

        Ok(GitHubClient::new_with_options(
            self.github_token.clone(),
            self.config.debug,
            proxy_manager,
            self.config.force_proxy,
        ))
    }
}

/// Start the server with all workers
pub async fn start_server(config: ServerConfig) -> Result<()> {
    let github_token = Config::github_token();

    let state = Arc::new(ServerState {
        config,
        github_token,
    });

    // Initial stats
    {
        let db = state.open_db()?;
        let without_metadata = db.count_repos_without_metadata()?;
        let without_embeddings = db.count_repos_without_embeddings()?;
        let unexplored = db.count_unexplored_owners()?;

        eprintln!("\x1b[36m╔══════════════════════════════════════╗\x1b[0m");
        eprintln!("\x1b[36m║       goto-gh server starting        ║\x1b[0m");
        eprintln!("\x1b[36m╚══════════════════════════════════════╝\x1b[0m");
        eprintln!();
        eprintln!("  \x1b[90mQueue status:\x1b[0m");
        eprintln!("    {} repos need metadata (fetch)", without_metadata);
        eprintln!("    {} repos need embeddings (embed)", without_embeddings);
        eprintln!("    {} owners to explore (discover)", unexplored);
        eprintln!();
        eprintln!("  \x1b[90mWorkers:\x1b[0m fetch, discover, embed (continuous)");
        eprintln!();
        eprintln!("  \x1b[33mPress Ctrl+C to stop gracefully\x1b[0m");
        eprintln!();
    }

    // Shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));

    // Spawn workers with separate tokio runtimes in threads
    let fetch_shutdown = shutdown.clone();
    let fetch_state = state.clone();
    let fetch_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(run_fetch_worker_loop(fetch_state, fetch_shutdown));
    });

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

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    eprintln!("\n\x1b[33m!\x1b[0m Shutting down workers...");

    // Signal all workers to stop
    shutdown.store(true, Ordering::SeqCst);

    // Wait for workers to finish (they check shutdown flag frequently)
    let _ = fetch_handle.join();
    let _ = discover_handle.join();
    let _ = embed_handle.join();

    // Final checkpoint
    if let Ok(db) = Database::open() {
        let _ = db.checkpoint();
        eprintln!("\x1b[32mok\x1b[0m WAL checkpointed");
    }

    eprintln!("\x1b[32mok\x1b[0m Server stopped");
    Ok(())
}

/// Fetch worker loop - runs continuously
async fn run_fetch_worker_loop(state: Arc<ServerState>, shutdown: Arc<AtomicBool>) {
    eprintln!("\x1b[36m[fetch]\x1b[0m Worker started");

    while !shutdown.load(Ordering::SeqCst) {
        match run_fetch_cycle(&state, &shutdown).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
            Err(e) => {
                eprintln!("\x1b[36m[fetch]\x1b[0m \x1b[31mError: {}\x1b[0m", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }

    eprintln!("\x1b[36m[fetch]\x1b[0m Shutting down");
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

/// Fetch cycle using core functions
async fn run_fetch_cycle(state: &ServerState, shutdown: &AtomicBool) -> Result<bool> {
    let db = state.open_db()?;
    let client = state.create_github_client()?;

    let config = FetchRunnerConfig {
        batch_size: state.config.fetch_batch_size,
        concurrency: state.config.fetch_concurrency,
        limit: None,
        debug: state.config.debug,
        delay_ms: 50,
    };

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
                "\x1b[36m[fetch]\x1b[0m +{} repos{}",
                progress.fetched_this_batch, discovered_info
            );
        },
        || shutdown.load(Ordering::SeqCst),
    )
    .await;

    match result {
        Ok(run_result) => {
            if run_result.total_fetched > 0 && state.config.debug {
                eprintln!(
                    "\x1b[36m[fetch]\x1b[0m Done: {} fetched, {} gone",
                    run_result.total_fetched, run_result.total_gone
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
