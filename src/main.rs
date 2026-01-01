mod cluster;
mod config;
mod db;
mod discovery;
mod embed_core;
mod embedding;
mod explore;
mod fetch;
mod formatting;
mod github;
mod http;
mod openai;
mod papers;
mod proxy;
mod search;
mod server;

use anyhow::Result;
use clap::{Parser, Subcommand};

use config::Config;
use db::Database;
use discovery::{discover_from_owners, discover_owner_repos};
use embedding::{build_embedding_text, embed_passage};
use formatting::{format_repo_link, format_stars, truncate_str};
use github::{is_gone_error, GitHubClient};
use proxy::ProxyManager;
use search::{expand_query, search};


#[derive(Parser)]
#[command(name = "goto-gh")]
#[command(about = "Semantic search for GitHub repositories")]
#[command(after_help = "\x1b[36mExamples:\x1b[0m
  goto-gh index              # Index top repos by stars
  goto-gh \"vector database\"  # Semantic search
  goto-gh find qdrant        # Fuzzy search by name")]
struct Cli {
    /// Search query (semantic search)
    #[arg(trailing_var_arg = true)]
    query: Vec<String>,

    /// Number of results to show
    #[arg(short, long, default_value = "10")]
    limit: usize,

    /// Semantic-only search (disable name/keyword boosting)
    #[arg(short, long)]
    semantic: bool,

    /// Expand query using LLM (Claude CLI) for better semantic understanding
    #[arg(short, long)]
    expand: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a specific repository by name (fetches from GitHub)
    Add {
        /// Repository full name (e.g., "qdrant/qdrant")
        repo: String,
    },

    /// Load repo names from file into DB (no GitHub fetch, just stores names)
    Load {
        /// Path to file containing repo names (one per line)
        file: String,
    },

    /// Load usernames from file as owners to discover (no GitHub fetch)
    LoadUsers {
        /// Path to file containing usernames (one per line)
        file: String,
    },

    /// Show index statistics
    Stats,

    /// Fuzzy search by repo name only (no semantic search)
    Find {
        /// Pattern to search for in repo names
        pattern: String,

        /// Number of results to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Check GitHub API rate limit
    RateLimit,

    /// Fetch metadata from GitHub for repos in DB that don't have it yet
    Fetch {
        /// Number of repos to fetch (default: all pending)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Batch size for GraphQL queries (default: 300, = 10 API chunks of 30)
        #[arg(short, long, default_value = "300")]
        batch_size: usize,

        /// Number of parallel GraphQL requests (default: 2)
        #[arg(short = 'j', long, default_value = "2")]
        concurrency: usize,

        /// Show API calls for debugging
        #[arg(long)]
        debug: bool,
    },

    /// Generate embeddings for repos that don't have them yet
    Embed {
        /// Batch size for embedding (default: 200 for local, 2000 for openai)
        #[arg(short, long)]
        batch_size: Option<usize>,

        /// Maximum repos to embed (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Delay between batches in seconds (default: 0)
        #[arg(short, long, default_value = "0")]
        delay: u64,

        /// Clear all existing embeddings first (re-embed everything)
        #[arg(long)]
        reset: bool,

        /// Embedding provider: "local" (E5, default) or "openai" (text-embedding-3-small)
        #[arg(long, default_value = "local")]
        provider: String,

        /// Show API calls for debugging
        #[arg(long)]
        debug: bool,
    },

    /// Print database file path
    DbPath,

    /// Discover more repos by exploring owners of existing repos (also fetches followers)
    #[command(name = "discover")]
    Discover {
        /// Maximum owners to explore (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Number of concurrent requests (default: 5)
        #[arg(short, long, default_value = "5")]
        concurrency: usize,

        /// Show API calls for debugging
        #[arg(long)]
        debug: bool,

        /// Path to proxy list file (one ip:port per line) for REST API calls
        #[arg(long)]
        proxy_file: Option<String>,

        /// Force using proxies for all requests (don't use token)
        #[arg(long)]
        force_proxy: bool,
    },

    /// Extract paper links (arxiv, doi, etc.) from READMEs
    #[command(name = "extract-papers")]
    ExtractPapers {
        /// Maximum repos to process (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Batch size for processing (default: 1000)
        #[arg(short, long, default_value = "1000")]
        batch_size: usize,
    },

    /// Fetch README for repos that don't have one (uses REST API /repos/{owner}/{repo}/readme)
    #[command(name = "fetch-missing-readmes")]
    FetchMissingReadmes {
        /// Maximum repos to process (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Number of concurrent requests (default: 5)
        #[arg(short, long, default_value = "5")]
        concurrency: usize,

        /// Show API calls for debugging
        #[arg(long)]
        debug: bool,

        /// Path to proxy list file (one ip:port per line)
        #[arg(long)]
        proxy_file: Option<String>,

        /// Force using proxies for all requests (don't use token)
        #[arg(long)]
        force_proxy: bool,
    },

    /// Extract linked repos from README content (discover new repos organically)
    #[command(name = "extract-repos")]
    ExtractRepos {
        /// Maximum repos to process (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Batch size for processing (default: 1000)
        #[arg(short, long, default_value = "1000")]
        batch_size: usize,
    },

    /// Start HTTP server with SQL explorer interface
    #[command(hide = true)]
    Http {
        /// Port to listen on (default: 3000)
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },

    /// Run as daemon with concurrent fetch, discover, and embed workers
    Server {
        /// Fetch batch size (default: 300)
        #[arg(long, default_value = "300")]
        fetch_batch_size: usize,

        /// Fetch concurrency (default: 2)
        #[arg(long, default_value = "2")]
        fetch_concurrency: usize,

        /// README fetch concurrency (default: 20)
        #[arg(long, default_value = "20")]
        readme_concurrency: usize,

        /// Discover limit per cycle (default: 50)
        #[arg(long, default_value = "50")]
        discover_limit: usize,

        /// Embed batch size (default: 200 for local, 2000 for openai)
        #[arg(long)]
        embed_batch_size: Option<usize>,

        /// Embedding provider: "local" (E5) or "openai"
        #[arg(long, default_value = "local")]
        provider: String,

        /// Path to proxy list file (one ip:port per line)
        #[arg(long)]
        proxy_file: Option<String>,

        /// Force using proxies for all requests
        #[arg(long)]
        force_proxy: bool,

        /// Show debug output
        #[arg(long)]
        debug: bool,
    },

    // ==================== Exploration Commands ====================

    /// Semantic random walk through repos - discover by wandering
    Walk {
        /// Starting repo (owner/repo) or empty for random start
        #[arg(default_value = "")]
        repo: String,

        /// Number of steps to take
        #[arg(short, long, default_value = "5")]
        steps: usize,

        /// Number of candidates to consider at each step
        #[arg(short, long, default_value = "10")]
        breadth: usize,

        /// Start from a random repo
        #[arg(long)]
        random_start: bool,
    },

    /// Find underrated gems similar to popular repos
    Underrated {
        /// Minimum similarity threshold (0-1)
        #[arg(long, default_value = "0.6")]
        min_sim: f32,

        /// Maximum stars for "underrated" repos
        #[arg(long, default_value = "500")]
        max_stars: u64,

        /// Reference repo to find alternatives for (optional)
        #[arg(long)]
        reference: Option<String>,

        /// Number of popular repos to sample (if no reference)
        #[arg(long, default_value = "20")]
        sample: usize,

        /// Gems to show per reference
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Find repos at the intersection of two topics
    Cross {
        /// First topic/domain
        topic1: String,

        /// Second topic/domain
        topic2: String,

        /// Minimum similarity to each topic (0-1)
        #[arg(long, default_value = "0.4")]
        min_each: f32,

        /// Number of results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Generate interactive cluster map visualization using t-SNE
    #[command(name = "cluster-map")]
    ClusterMap {
        /// Output HTML file
        #[arg(short, long, default_value = "cluster_map.html")]
        output: String,

        /// Number of repos to sample
        #[arg(short, long, default_value = "5000")]
        sample: usize,

        /// Minimum stars filter
        #[arg(long, default_value = "0")]
        min_stars: u64,

        /// Filter by language
        #[arg(long)]
        language: Option<String>,

        /// t-SNE perplexity parameter (default: 30)
        #[arg(long, default_value = "30")]
        perplexity: f32,

        /// t-SNE epochs/iterations (default: 1000)
        #[arg(long, default_value = "1000")]
        epochs: usize,
    },

    /// Find interesting profiles using semantic search
    Profiles {
        /// Number of profiles to show
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Minimum interesting repos per profile
        #[arg(long, default_value = "2")]
        min_repos: usize,

        /// Repos to fetch per seed query
        #[arg(long, default_value = "100")]
        per_seed: usize,

        /// Custom seed query (can repeat)
        #[arg(long, short)]
        seed: Vec<String>,

        /// Expand seed queries using Claude for better semantic matching
        #[arg(long, short = 'x')]
        expand: bool,
    },

    /// Backfill short embeddings from existing full embeddings (Matryoshka two-stage search)
    #[command(name = "backfill-short-embeddings")]
    BackfillShortEmbeddings {
        /// Batch size for processing (default: 10000)
        #[arg(short, long, default_value = "10000")]
        batch_size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignore if missing)
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    let db = Database::open()?;

    // Checkpoint WAL at startup (clean up from previous/crashed runs)
    if let Err(e) = db.checkpoint() {
        eprintln!("\x1b[33m!\x1b[0m WAL checkpoint at startup failed: {}", e);
    }

    // Run main logic with graceful shutdown support
    let result = run_main(cli, db).await;

    result
}

async fn run_main(cli: Cli, db: Database) -> Result<()> {
    // Set up graceful shutdown using tokio signal (safer than ctrlc)
    let db_for_shutdown = db.path().to_path_buf();

    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\n\x1b[33m!\x1b[0m Interrupted, checkpointing WAL...");
            // Use the same connection approach but with timeout awareness
            if let Ok(conn) = rusqlite::Connection::open(&db_for_shutdown) {
                // Set busy timeout to avoid deadlock
                let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
                match conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);") {
                    Ok(_) => eprintln!("\x1b[32mok\x1b[0m WAL checkpointed"),
                    Err(e) => eprintln!("\x1b[31mx\x1b[0m WAL checkpoint failed: {}", e),
                }
            }
            std::process::exit(130); // 128 + SIGINT
        }
    });

    let token = Config::github_token();
    let client = GitHubClient::new(token.clone());

    // If query provided, do semantic search
    if !cli.query.is_empty() {
        let query = cli.query.join(" ");

        // Expand query with LLM if requested
        let search_query = if cli.expand {
            expand_query(&query)?
        } else {
            query
        };

        return search(&search_query, cli.limit, cli.semantic, &db);
    }

    match cli.command {
        Some(Commands::Add { repo }) => {
            add_repo(&client, &db, &repo).await
        }
        Some(Commands::Load { file }) => {
            load_repo_stubs(&db, &file)
        }
        Some(Commands::LoadUsers { file }) => {
            load_user_stubs(&db, &file)
        }
        Some(Commands::Stats) => {
            show_stats(&db)
        }
        Some(Commands::Find { pattern, limit }) => {
            find_by_name(&db, &pattern, limit)
        }
        Some(Commands::RateLimit) => {
            check_rate_limit(&client).await
        }
        Some(Commands::Fetch { limit, batch_size, concurrency, debug }) => {
            let client = GitHubClient::new_with_options(token.clone(), debug, None, false);
            let config = fetch::FetchRunnerConfig { limit, batch_size, concurrency, debug, delay_ms: 50 };
            fetch_from_db_cli(&client, &db, &config).await
        }
        Some(Commands::Embed { batch_size, limit, delay, reset, provider, debug }) => {
            embed_missing(&db, batch_size, limit, delay, reset, &provider, debug).await
        }
        Some(Commands::DbPath) => {
            println!("{}", Config::db_path()?.display());
            Ok(())
        }
        Some(Commands::Discover { limit, concurrency, debug, proxy_file, force_proxy }) => {
            // Load proxies if specified
            let proxy_manager = if let Some(path) = proxy_file {
                let path = std::path::PathBuf::from(&path);
                match ProxyManager::from_file(&path) {
                    Ok(pm) => {
                        eprintln!("\x1b[36m..\x1b[0m {}", pm);
                        Some(pm)
                    }
                    Err(e) => {
                        eprintln!("\x1b[31mx\x1b[0m Failed to load proxies: {}", e);
                        return Err(e);
                    }
                }
            } else {
                if force_proxy {
                    eprintln!("\x1b[31mx\x1b[0m --force-proxy requires --proxy-file");
                    std::process::exit(1);
                }
                None
            };

            if force_proxy {
                eprintln!("\x1b[33m..\x1b[0m Force proxy mode (token disabled)");
            }

            let client = GitHubClient::new_with_options(token.clone(), debug, proxy_manager, force_proxy);
            discover_from_owners(&client, &db, limit, concurrency).await
        }
        Some(Commands::ExtractPapers { limit, batch_size }) => {
            extract_papers(&db, limit, batch_size)
        }
        Some(Commands::FetchMissingReadmes { limit, concurrency, debug, proxy_file, force_proxy }) => {
            let proxy_manager = if let Some(path) = proxy_file {
                let path = std::path::PathBuf::from(&path);
                match ProxyManager::from_file(&path) {
                    Ok(pm) => {
                        eprintln!("\x1b[36m..\x1b[0m {}", pm);
                        Some(pm)
                    }
                    Err(e) => {
                        eprintln!("\x1b[31mx\x1b[0m Failed to load proxies: {}", e);
                        return Err(e);
                    }
                }
            } else {
                if force_proxy {
                    eprintln!("\x1b[31mx\x1b[0m --force-proxy requires --proxy-file");
                    std::process::exit(1);
                }
                None
            };

            let client = GitHubClient::new_with_options(token.clone(), debug, proxy_manager, force_proxy);
            fetch_missing_readmes(&client, &db, limit, concurrency).await
        }
        Some(Commands::ExtractRepos { limit, batch_size }) => {
            extract_repos(&db, limit, batch_size)
        }
        Some(Commands::Http { port }) => {
            http::start_server(db.path().to_path_buf(), port).await
        }
        Some(Commands::Server {
            fetch_batch_size,
            fetch_concurrency,
            readme_concurrency,
            discover_limit,
            embed_batch_size,
            provider,
            proxy_file,
            force_proxy,
            debug,
        }) => {
            use server::ServerConfig;

            // Auto-detect provider from existing embeddings if not explicitly specified
            let embed_provider = embed_core::auto_detect_provider(&db, &provider)?;
            if embed_provider.is_openai() && provider == "local" {
                eprintln!("\x1b[36m..\x1b[0m Auto-detected OpenAI embeddings in database");
            }

            let config = ServerConfig {
                fetch_batch_size,
                fetch_concurrency,
                readme_concurrency,
                discover_limit,
                embed_batch_size: embed_provider.batch_size(embed_batch_size),
                embed_delay_ms: 50,
                embed_provider,
                debug,
                proxy_file,
                force_proxy,
            };

            // Server manages its own DB connections
            drop(db);
            server::start_server(config).await
        }
        Some(Commands::Walk { repo, steps, breadth, random_start }) => {
            run_walk(&db, &repo, steps, breadth, random_start)
        }
        Some(Commands::Underrated { min_sim, max_stars, reference, sample, limit }) => {
            run_underrated(&db, min_sim, max_stars, reference, sample, limit)
        }
        Some(Commands::Cross { topic1, topic2, min_each, limit }) => {
            run_cross(&db, &topic1, &topic2, min_each, limit)
        }
        Some(Commands::ClusterMap { output, sample, min_stars, language, perplexity, epochs }) => {
            run_cluster_map(&db, &output, sample, min_stars, language.as_deref(), perplexity, epochs)
        }
        Some(Commands::Profiles { limit, min_repos, per_seed, seed, expand }) => {
            run_profiles(&db, limit, min_repos, per_seed, seed, expand)
        }
        Some(Commands::BackfillShortEmbeddings { batch_size }) => {
            backfill_short_embeddings(&db, batch_size)
        }
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            eprintln!();
            std::process::exit(0);
        }
    }
}

/// Add a single repository
async fn add_repo(client: &GitHubClient, db: &Database, full_name: &str) -> Result<()> {
    eprintln!("\x1b[36m..\x1b[0m Fetching {}", full_name);

    let repo = client.get_repo(full_name).await?;
    let readme = client.get_readme(full_name).await.ok().flatten();

    let text = build_embedding_text(
        &repo.full_name,
        repo.description.as_deref(),
        &repo.topics,
        repo.language.as_deref(),
        readme.as_deref(),
    );

    let embedding = embed_passage(&text)?;

    let repo_id = db.upsert_repo(&repo, readme.as_deref(), &text)?;
    db.upsert_embedding(repo_id, &embedding)?;

    println!("\x1b[32mok\x1b[0m Added \x1b[1m{}\x1b[0m", repo.full_name);
    if let Some(desc) = &repo.description {
        println!("  {}", desc);
    }

    // Discover other repos from the same owner (if not already explored)
    discover_owner_repos(client, db, full_name).await?;

    Ok(())
}

/// Fuzzy search by repo name
fn find_by_name(db: &Database, pattern: &str, limit: usize) -> Result<()> {
    let repos = db.find_by_name(pattern, limit)?;

    if repos.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m No repos matching '{}'", pattern);
        return Ok(());
    }

    println!("\x1b[36m{} repos matching '{}'\x1b[0m\n", repos.len(), pattern);

    for repo in repos {
        let lang = repo.language.as_deref().unwrap_or("?");
        let stars = format_stars(repo.stars);
        let desc = repo.description.as_deref().unwrap_or("");
        let desc_truncated = truncate_str(desc, 60);
        let repo_link = format_repo_link(&repo.full_name, &repo.url);

        println!(
            "{} \x1b[33m{}\x1b[0m \x1b[90m[{}]\x1b[0m \x1b[90m{}\x1b[0m",
            repo_link, stars, lang, desc_truncated
        );
    }

    Ok(())
}

/// Show statistics
fn show_stats(db: &Database) -> Result<()> {
    let (total, indexed) = db.stats()?;
    let without_metadata = db.count_repos_without_metadata()?;
    let gone = db.count_gone()?;
    let distinct_owners = db.count_distinct_owners()?;
    let owners_to_explore = db.count_owners_to_explore()?;
    // Total visible = total minus gone repos
    let total_visible = total - gone;
    let with_metadata = total_visible - without_metadata;
    let without_embeddings = total_visible - indexed;

    println!("\x1b[36mIndex Statistics\x1b[0m\n");
    println!("  \x1b[90mTotal repos:\x1b[0m        {}", total_visible);
    println!("  \x1b[90mDistinct owners:\x1b[0m    {}", distinct_owners);
    println!("  \x1b[90mWith metadata:\x1b[0m      {}", with_metadata);
    println!("  \x1b[90mWith embeddings:\x1b[0m    {}", indexed);
    println!();
    println!("  \x1b[90mNeed metadata:\x1b[0m      {}", without_metadata);
    println!("  \x1b[90mNeed embeddings:\x1b[0m    {}", without_embeddings);
    if gone > 0 {
        println!("  \x1b[90mGone (deleted):\x1b[0m     {}", gone);
    }
    if owners_to_explore > 0 {
        println!("  \x1b[90mOwners to explore:\x1b[0m  {}", owners_to_explore);
    }

    if without_metadata > 0 {
        println!("\n  \x1b[33mTip:\x1b[0m Run: goto-gh fetch");
    } else if without_embeddings > 0 {
        println!("\n  \x1b[33mTip:\x1b[0m Run: goto-gh embed");
    }

    Ok(())
}

/// Check rate limit
async fn check_rate_limit(client: &GitHubClient) -> Result<()> {
    let rates = client.rate_limit().await?;

    let format_reset = |reset: u64| {
        chrono::DateTime::from_timestamp(reset as i64, 0)
            .map(|dt| {
                let local = dt.with_timezone(&chrono::Local);
                local.format("%H:%M:%S").to_string()
            })
            .unwrap_or_else(|| "?".to_string())
    };

    println!("\x1b[36mGitHub API Rate Limits\x1b[0m\n");

    println!("  \x1b[1mREST API\x1b[0m (discover, list repos)");
    println!("    \x1b[90mRemaining:\x1b[0m {}/{}", rates.core.remaining, rates.core.limit);
    println!("    \x1b[90mResets at:\x1b[0m {}\n", format_reset(rates.core.reset));

    println!("  \x1b[1mGraphQL API\x1b[0m (fetch repos/READMEs)");
    println!("    \x1b[90mRemaining:\x1b[0m {}/{}", rates.graphql.remaining, rates.graphql.limit);
    println!("    \x1b[90mResets at:\x1b[0m {}", format_reset(rates.graphql.reset));

    Ok(())
}

/// Load repo stubs from file (no API calls, just stores names in DB)
fn load_repo_stubs(db: &Database, file_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let all_repos: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && l.contains('/'))
        .collect();

    let total = all_repos.len();
    eprintln!(
        "\x1b[36m..\x1b[0m Loading {} repo names from {} (no GitHub API calls)",
        total, file_path
    );

    if total == 0 {
        eprintln!("\x1b[33m..\x1b[0m No repos to load");
        return Ok(());
    }

    // Process in batches for progress reporting
    const BATCH_SIZE: usize = 10000;
    let mut total_inserted = 0;
    let mut total_skipped = 0;

    for (batch_idx, batch) in all_repos.chunks(BATCH_SIZE).enumerate() {
        let batch_vec: Vec<String> = batch.to_vec();
        let (inserted, skipped) = db.add_repo_stubs_bulk(&batch_vec)?;
        total_inserted += inserted;
        total_skipped += skipped;

        if (batch_idx + 1) % 10 == 0 || batch_idx == 0 {
            let processed = (batch_idx + 1) * BATCH_SIZE;
            eprintln!(
                "  ... processed {}/{} (+{} new, {} dupe)",
                processed.min(total), total, inserted, skipped
            );
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Loaded {} new repos ({} already existed)",
        total_inserted, total_skipped
    );

    // Show next steps
    let need_fetch = db.count_repos_without_metadata()?;
    eprintln!("\x1b[36m..\x1b[0m {} repos now awaiting metadata fetch", need_fetch);
    eprintln!("    Run: goto-gh fetch --batch-size 300");

    Ok(())
}

/// Load usernames from file as owners to discover
fn load_user_stubs(db: &Database, file_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let usernames: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.contains('/'))
        .collect();

    let total = usernames.len();
    eprintln!(
        "\x1b[36m..\x1b[0m Loading {} usernames from {} (no GitHub API calls)",
        total, file_path
    );

    if total == 0 {
        eprintln!("\x1b[33m..\x1b[0m No usernames to load");
        return Ok(());
    }

    let mut added = 0;
    let mut skipped = 0;

    for username in &usernames {
        match db.add_follower_as_owner(username)? {
            true => added += 1,
            false => skipped += 1,
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Added {} new users ({} already known)",
        added, skipped
    );

    let unexplored = db.count_unexplored_owners()?;
    eprintln!("\x1b[36m..\x1b[0m {} owners now awaiting discovery", unexplored);
    eprintln!("    Run: goto-gh discover");

    Ok(())
}

/// Fetch metadata from GitHub for repos in DB - CLI wrapper
async fn fetch_from_db_cli(
    client: &GitHubClient,
    db: &Database,
    config: &fetch::FetchRunnerConfig,
) -> Result<()> {
    let need_fetch = db.count_repos_without_metadata()?;

    if need_fetch == 0 {
        eprintln!("\x1b[32mok\x1b[0m All repos already have metadata");
        return Ok(());
    }

    let to_fetch = config.limit.map(|l| l.min(need_fetch)).unwrap_or(need_fetch);
    eprintln!(
        "\x1b[36m..\x1b[0m Fetching metadata for {} repos (batch size: {}, concurrency: {}) - NO EMBEDDINGS",
        to_fetch, config.batch_size, config.concurrency
    );

    let result = fetch::run_fetch_loop(
        client,
        db,
        config,
        |progress| {
            eprintln!(
                "\x1b[36m..\x1b[0m Batch {}: fetching {} repos via GraphQL...",
                progress.batch_num, progress.batch_size
            );

            if progress.gone_this_batch > 0 {
                let gone_pct = (progress.gone_this_batch as f64 / progress.batch_size as f64 * 100.0) as u32;
                eprintln!("  ... marked {} repos as gone ({}% of batch)", progress.gone_this_batch, gone_pct);
                if gone_pct >= 80 {
                    eprintln!("  \x1b[33m⚠ Warning: {}% of batch is gone - queue may be mostly stale repos\x1b[0m", gone_pct);
                }
            }

            eprintln!("  ... stored {} repos (total: {})", progress.fetched_this_batch, progress.total_fetched);

            if progress.discovered_this_batch > 0 {
                eprintln!("  ... discovered {} new repos from READMEs", progress.discovered_this_batch);
            }
        },
        || false,
    ).await?;

    eprintln!(
        "\x1b[32mok\x1b[0m Fetched {} repos ({} gone) - ready for embedding",
        result.total_fetched, result.total_gone
    );

    let need_embed = db.count_repos_without_embeddings()?;
    eprintln!("\x1b[36m..\x1b[0m {} repos now awaiting embeddings", need_embed);
    eprintln!("    Run: goto-gh embed --batch-size 200 --delay 5");

    Ok(())
}

/// Generate embeddings for repos that don't have them yet - CLI wrapper
async fn embed_missing(
    db: &Database,
    batch_size: Option<usize>,
    limit: Option<usize>,
    delay_secs: u64,
    reset: bool,
    provider: &str,
    debug: bool,
) -> Result<()> {
    use crate::embedding::EMBEDDING_DIM;
    use crate::openai::OPENAI_EMBEDDING_DIM;

    // Auto-detect provider
    let embed_provider = if reset {
        embedding::EmbedProvider::parse(provider)?
    } else {
        embed_core::auto_detect_provider(db, provider)?
    };

    let is_openai = embed_provider.is_openai();
    let provider_name = if is_openai { "openai" } else { "local" };
    let target_dim = embed_provider.dimension();
    let batch_size = embed_provider.batch_size(batch_size);

    // Log auto-detection
    if !reset {
        let current_dim = db.get_embedding_dimension()?;
        if let Some(dim) = current_dim {
            if dim == OPENAI_EMBEDDING_DIM && provider == "local" {
                eprintln!("\x1b[36m..\x1b[0m Auto-detected OpenAI embeddings (1536d) - using --provider openai");
            } else if dim == EMBEDDING_DIM && provider == "openai" {
                eprintln!("\x1b[36m..\x1b[0m Auto-detected local E5 embeddings (384d) - using --provider local");
            }
        }
    }

    if reset {
        eprintln!("\x1b[33m!\x1b[0m Recreating embeddings table with {} dimensions ({})", target_dim, provider_name);
    }

    let need_embed = db.count_repos_without_embeddings()?;
    if need_embed == 0 && !reset {
        eprintln!("\x1b[32mok\x1b[0m All repos already have embeddings");
        return Ok(());
    }

    let to_process = limit.map(|l| l.min(need_embed)).unwrap_or(need_embed);
    eprintln!(
        "\x1b[36m..\x1b[0m Generating embeddings for {} repos (provider: {}, batch: {})",
        to_process, provider_name, batch_size
    );

    // Estimate cost for OpenAI
    if is_openai {
        let estimated_tokens = to_process as f64 * 500.0;
        let estimated_cost = estimated_tokens / 1_000_000.0 * 0.02;
        eprintln!(
            "    \x1b[33mEstimated cost: ${:.2}\x1b[0m (~{:.0}M tokens)",
            estimated_cost,
            estimated_tokens / 1_000_000.0
        );
    }

    let start_time = std::time::Instant::now();
    let config = embed_core::EmbedRunnerConfig {
        batch_size,
        limit,
        delay_ms: delay_secs * 1000,
        provider: embed_provider,
        debug,
        reset,
    };

    let result = embed_core::run_embed_loop(
        db,
        &config,
        |progress| {
            eprintln!(
                "\x1b[36m..\x1b[0m Batch {}: embedding {} repos...",
                progress.batch_num, progress.batch_size
            );

            let elapsed = start_time.elapsed();
            let eta = if progress.total_embedded > 0 {
                let rate = progress.total_embedded as f64 / elapsed.as_secs_f64();
                let remaining = progress.total_to_process - progress.total_embedded;
                let eta_secs = remaining as f64 / rate;
                format_duration(eta_secs as u64)
            } else {
                "calculating...".to_string()
            };

            if progress.total_tokens > 0 {
                eprintln!(
                    "  ... embedded {} (total: {}/{}, tokens: {}, ETA: {})",
                    progress.embedded_this_batch, progress.total_embedded,
                    progress.total_to_process, progress.total_tokens, eta
                );
            } else {
                eprintln!(
                    "  ... embedded {} (total: {}/{}, ETA: {})",
                    progress.embedded_this_batch, progress.total_embedded,
                    progress.total_to_process, eta
                );
            }
        },
        || false,
    ).await?;

    if result.total_tokens > 0 {
        let actual_cost = result.total_tokens as f64 / 1_000_000.0 * 0.02;
        eprintln!(
            "\x1b[32mok\x1b[0m Generated embeddings for {} repos ({} tokens, ${:.2})",
            result.total_embedded, result.total_tokens, actual_cost
        );
    } else {
        eprintln!(
            "\x1b[32mok\x1b[0m Generated embeddings for {} repos",
            result.total_embedded
        );
    }

    let still_need = db.count_repos_without_embeddings()?;
    if still_need > 0 {
        eprintln!("\x1b[36m..\x1b[0m {} repos still awaiting embeddings", still_need);
    }

    Ok(())
}

/// Format seconds into human-readable duration (e.g., "1h 23m", "45m 12s")
fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        format!("{}h {}m", hours, mins)
    }
}


/// Extract paper links from READMEs
fn extract_papers(db: &Database, limit: Option<usize>, batch_size: usize) -> Result<()> {
    let need_extraction = db.count_repos_needing_paper_extraction()?;

    if need_extraction == 0 {
        let total_papers = db.count_papers()?;
        let total_sources = db.count_paper_sources()?;
        eprintln!("\x1b[32mok\x1b[0m All READMEs already processed");
        eprintln!("    {} unique papers from {} repo mentions", total_papers, total_sources);
        return Ok(());
    }

    let to_process = limit.map(|l| l.min(need_extraction)).unwrap_or(need_extraction);
    eprintln!(
        "\x1b[36m..\x1b[0m Extracting papers from {} READMEs (batch size: {})",
        to_process, batch_size
    );

    let mut total_processed = 0;
    let mut total_papers_found = 0;
    let mut total_new_papers = 0;
    let start = std::time::Instant::now();

    loop {
        let remaining = limit.map(|l| l - total_processed).unwrap_or(usize::MAX);
        if remaining == 0 {
            break;
        }

        let batch_limit = batch_size.min(remaining);
        let repos = db.get_repos_needing_paper_extraction(Some(batch_limit))?;

        if repos.is_empty() {
            break;
        }

        let batch_start = std::time::Instant::now();
        let mut batch_papers = 0;
        let mut batch_new = 0;

        for (repo_id, _full_name, readme) in &repos {
            let papers = papers::extract_paper_links(readme);
            batch_papers += papers.len();

            for paper in papers {
                let paper_id = db.upsert_paper(
                    &paper.url,
                    &paper.domain,
                    paper.arxiv_id.as_deref(),
                    paper.doi.as_deref(),
                )?;

                // Check if this is a new paper (simple heuristic: if paper_id is recent)
                // Actually, upsert returns existing id, so we track via sources
                let sources_before = db.count_paper_sources()?;
                db.add_paper_source(paper_id, *repo_id, paper.context.as_deref())?;
                let sources_after = db.count_paper_sources()?;
                if sources_after > sources_before {
                    batch_new += 1;
                }
            }

            db.mark_papers_extracted(*repo_id)?;
            total_processed += 1;
        }

        total_papers_found += batch_papers;
        total_new_papers += batch_new;

        let elapsed = batch_start.elapsed();
        let repos_per_sec = repos.len() as f64 / elapsed.as_secs_f64();

        eprintln!(
            "  \x1b[90m✓ Batch: {} repos in {:.1}s ({:.0}/s) - {} papers ({} new)\x1b[0m",
            repos.len(),
            elapsed.as_secs_f64(),
            repos_per_sec,
            batch_papers,
            batch_new
        );
    }

    let total_elapsed = start.elapsed();
    let total_papers = db.count_papers()?;
    let total_sources = db.count_paper_sources()?;

    eprintln!(
        "\x1b[32mok\x1b[0m Processed {} repos in {:.1}s",
        total_processed,
        total_elapsed.as_secs_f64()
    );
    eprintln!(
        "    Found {} paper links ({} new unique papers)",
        total_papers_found, total_new_papers
    );
    eprintln!(
        "    Total: {} unique papers from {} repo mentions",
        total_papers, total_sources
    );

    Ok(())
}

/// Fetch README for repos that don't have one using REST API
async fn fetch_missing_readmes(
    client: &GitHubClient,
    db: &Database,
    limit: Option<usize>,
    concurrency: usize,
) -> Result<()> {
    use futures::stream::{self, StreamExt};

    let need_readme = db.count_repos_without_readme()?;

    if need_readme == 0 {
        eprintln!("\x1b[32mok\x1b[0m All repos already have README content");
        return Ok(());
    }

    let to_process = limit.map(|l| l.min(need_readme)).unwrap_or(need_readme);
    eprintln!(
        "\x1b[36m..\x1b[0m Fetching README for {} repos via REST API (concurrency: {})",
        to_process, concurrency
    );

    let repos = db.get_repos_without_readme(limit)?;
    let total = repos.len();
    let start = std::time::Instant::now();

    let mut fetched = 0;
    let mut not_found = 0;

    #[derive(Debug)]
    enum ReadmeResult {
        Found(String),
        NotFound,
        Error(String),
    }

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

    let mut errors = 0;

    let mut discovered = 0;

    while let Some((repo_id, full_name, result)) = result_stream.next().await {
        match result {
            ReadmeResult::Found(content) => {
                let len = content.len();
                // Treat empty README as "no readme" - don't retry
                if len == 0 {
                    eprintln!("\x1b[90m-\x1b[0m {} (empty readme)", full_name);
                    let _ = db.mark_repo_no_readme(repo_id);
                    not_found += 1;
                    continue;
                }
                if let Err(e) = db.update_repo_readme(repo_id, &content) {
                    eprintln!("\x1b[31mx\x1b[0m {} - db error: {}", full_name, e);
                } else {
                    // Extract linked repos from README
                    let found = fetch::discover_repos_from_readme(db, &content);
                    discovered += found;
                    let _ = db.mark_repos_extracted(repo_id);

                    if found > 0 {
                        eprintln!("\x1b[32m✓\x1b[0m {} ({} bytes, +{} repos)", full_name, len, found);
                    } else {
                        eprintln!("\x1b[32m✓\x1b[0m {} ({} bytes)", full_name, len);
                    }
                    fetched += 1;
                }
            }
            ReadmeResult::NotFound => {
                eprintln!("\x1b[90m-\x1b[0m {} (no readme)", full_name);
                let _ = db.mark_repo_no_readme(repo_id);
                not_found += 1;
            }
            ReadmeResult::Error(e) => {
                if is_gone_error(&e) {
                    eprintln!("\x1b[33m⚠\x1b[0m {} - gone ({})", full_name, e);
                    let _ = db.mark_as_gone(&full_name);
                    not_found += 1; // Count as processed
                } else {
                    eprintln!("\x1b[31m!\x1b[0m {} - {}", full_name, e);
                    errors += 1;
                    // Don't mark as no_readme, we'll retry later
                }
            }
        }

        // Progress every 100 repos
        let done = fetched + not_found;
        if done % 100 == 0 && done > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let eta = (total - done) as f64 / rate;
            eprintln!(
                "  \x1b[36m{}/{}\x1b[0m ({:.0}/s, ETA: {})",
                done, total, rate, format_duration(eta as u64)
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\x1b[32mok\x1b[0m Fetched {} READMEs ({} not found, {} errors) in {:.1}s",
        fetched, not_found, errors, elapsed.as_secs_f64()
    );

    if fetched > 0 {
        eprintln!("    {} repos now need re-embedding", fetched);
        if discovered > 0 {
            eprintln!("    {} new repos discovered from README links", discovered);
        }
        eprintln!("    Run: goto-gh embed");
    }

    Ok(())
}

/// Extract linked repos from README content (discover new repos organically)
fn extract_repos(db: &Database, limit: Option<usize>, batch_size: usize) -> Result<()> {
    use crate::fetch::discover_repos_from_readme;

    let need_extraction = db.count_repos_needing_repo_extraction()?;

    if need_extraction == 0 {
        eprintln!("\x1b[32mok\x1b[0m All README content has been processed for repo discovery");
        return Ok(());
    }

    let to_process = limit.map(|l| l.min(need_extraction)).unwrap_or(need_extraction);
    eprintln!(
        "\x1b[36m..\x1b[0m Extracting linked repos from {} READMEs (batch size: {})",
        to_process, batch_size
    );

    let start = std::time::Instant::now();
    let mut total_processed = 0;
    let mut total_discovered = 0;

    loop {
        let remaining = limit.map(|l| l - total_processed).unwrap_or(batch_size);
        if remaining == 0 {
            break;
        }

        let repos = db.get_repos_needing_repo_extraction(Some(batch_size.min(remaining)))?;
        if repos.is_empty() {
            break;
        }

        let batch_count = repos.len();
        let mut batch_discovered = 0;

        for (repo_id, _full_name, readme) in repos {
            let discovered = discover_repos_from_readme(db, &readme);
            batch_discovered += discovered;

            db.mark_repos_extracted(repo_id)?;
        }

        total_processed += batch_count;
        total_discovered += batch_discovered;

        let elapsed = start.elapsed().as_secs_f64();
        let rate = total_processed as f64 / elapsed;
        eprintln!(
            "  +{} repos from {} READMEs ({:.0}/s, {} total discovered)",
            batch_discovered, batch_count, rate, total_discovered
        );

        if batch_count < batch_size {
            break;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\x1b[32mok\x1b[0m Processed {} READMEs in {:.1}s",
        total_processed,
        elapsed.as_secs_f64()
    );
    eprintln!("    Discovered {} new repo stubs", total_discovered);
    eprintln!("    Run: goto-gh fetch");

    Ok(())
}

// ==================== Exploration Commands ====================

/// Random walk through embedding space
fn run_walk(db: &Database, repo: &str, steps: usize, breadth: usize, random_start: bool) -> Result<()> {
    use explore::{random_walk, WalkConfig};

    let config = WalkConfig {
        steps,
        breadth,
        random_start: random_start || repo.is_empty(),
    };

    let result = random_walk(db, repo, &config)?;

    if result.steps.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m Walk failed - no repos found");
        return Ok(());
    }

    // Print starting point
    let start = &result.steps[0];
    println!(
        "\x1b[36mRandom Walk\x1b[0m starting from: \x1b[1m{}\x1b[0m {}",
        start.repo.full_name,
        format_stars(start.repo.stars)
    );
    if let Some(desc) = &start.repo.description {
        println!("  \x1b[90m{}\x1b[0m", truncate_str(desc, 70));
    }
    println!();

    // Print each step
    for step in result.steps.iter().skip(1) {
        let lang = step.repo.language.as_deref().unwrap_or("?");
        let similarity = (-step.distance).exp();

        println!(
            "\x1b[33mStep {}\x1b[0m → \x1b[1m{}\x1b[0m {} \x1b[90m[{}]\x1b[0m",
            step.step_num,
            step.repo.full_name,
            format_stars(step.repo.stars),
            lang
        );
        if let Some(desc) = &step.repo.description {
            println!("         \x1b[90m{}\x1b[0m", truncate_str(desc, 65));
        }
        println!(
            "         \x1b[90mSimilarity: {:.0}% (from {} candidates)\x1b[0m",
            similarity * 100.0,
            step.candidates_considered
        );
        println!();
    }

    println!(
        "\x1b[36mWalk complete:\x1b[0m {} steps, total distance: {:.2}",
        result.steps.len() - 1,
        result.total_distance
    );

    Ok(())
}

/// Find underrated gems
fn run_underrated(
    db: &Database,
    min_sim: f32,
    max_stars: u64,
    reference: Option<String>,
    sample: usize,
    limit: usize,
) -> Result<()> {
    use explore::{find_underrated, UnderratedConfig};

    let config = UnderratedConfig {
        min_similarity: min_sim,
        max_stars,
        reference: reference.clone(),
        sample_popular: sample,
        min_reference_stars: 5000,
        limit_per_reference: limit,
    };

    eprintln!("\x1b[36m..\x1b[0m Finding underrated gems (min similarity: {:.0}%, max stars: {})",
        min_sim * 100.0, max_stars);

    let results = find_underrated(db, &config)?;

    if results.is_empty() {
        eprintln!("\x1b[33m!\x1b[0m No underrated gems found with current criteria");
        eprintln!("    Try: --min-sim 0.5 or --max-stars 1000");
        return Ok(());
    }

    println!();
    println!("\x1b[36mUnderrated Gems\x1b[0m\n");

    for result in &results {
        println!(
            "\x1b[1mSimilar to {}\x1b[0m {}",
            result.reference.full_name,
            format_stars(result.reference.stars)
        );

        for (i, gem) in result.gems.iter().enumerate() {
            let lang = gem.repo.language.as_deref().unwrap_or("?");
            println!(
                "  {}. {} {} \x1b[90m[{}]\x1b[0m \x1b[32m{:.0}% similar\x1b[0m",
                i + 1,
                format_repo_link(&gem.repo.full_name, &gem.repo.url),
                format_stars(gem.repo.stars),
                lang,
                gem.similarity * 100.0
            );
            if let Some(desc) = &gem.repo.description {
                println!("     \x1b[90m{}\x1b[0m", truncate_str(desc, 65));
            }
        }
        println!();
    }

    let total_gems: usize = results.iter().map(|r| r.gems.len()).sum();
    println!(
        "\x1b[36mFound {} gems\x1b[0m from {} reference repos",
        total_gems,
        results.len()
    );

    Ok(())
}

/// Find cross-pollination repos
fn run_cross(db: &Database, topic1: &str, topic2: &str, min_each: f32, limit: usize) -> Result<()> {
    use explore::{find_cross_pollination, CrossConfig};

    let config = CrossConfig {
        topic1: topic1.to_string(),
        topic2: topic2.to_string(),
        min_each,
        limit,
    };

    eprintln!(
        "\x1b[36m..\x1b[0m Finding repos at intersection of \"{}\" × \"{}\"",
        topic1, topic2
    );

    let results = find_cross_pollination(db, &config)?;

    if results.is_empty() {
        eprintln!("\x1b[33m!\x1b[0m No cross-pollination repos found");
        eprintln!("    Try: --min-each 0.3 or different topics");
        return Ok(());
    }

    println!();
    println!(
        "\x1b[36mCross-Pollination:\x1b[0m \"{}\" × \"{}\"\n",
        topic1, topic2
    );

    for (i, result) in results.iter().enumerate() {
        let lang = result.repo.language.as_deref().unwrap_or("?");
        println!(
            "  {:>2}. {} {} \x1b[90m[{}]\x1b[0m",
            i + 1,
            format_repo_link(&result.repo.full_name, &result.repo.url),
            format_stars(result.repo.stars),
            lang
        );
        println!(
            "      \x1b[32m{}: {:.0}%\x1b[0m  \x1b[34m{}: {:.0}%\x1b[0m",
            truncate_str(topic1, 15),
            result.sim_topic1 * 100.0,
            truncate_str(topic2, 15),
            result.sim_topic2 * 100.0
        );
        if let Some(desc) = &result.repo.description {
            println!("      \x1b[90m{}\x1b[0m", truncate_str(desc, 65));
        }
        println!();
    }

    println!("\x1b[36mFound {} repos\x1b[0m at the intersection", results.len());

    Ok(())
}

fn run_cluster_map(
    db: &Database,
    output: &str,
    sample: usize,
    min_stars: u64,
    language: Option<&str>,
    perplexity: f32,
    epochs: usize,
) -> Result<()> {
    use cluster::{generate_cluster_map, ClusterConfig};

    let config = ClusterConfig {
        sample,
        min_stars,
        language: language.map(String::from),
        perplexity,
        epochs,
    };

    generate_cluster_map(db, &config, output)
}

/// Find interesting developer/org profiles
fn run_profiles(
    db: &Database,
    limit: usize,
    min_repos: usize,
    per_seed: usize,
    seeds: Vec<String>,
    expand: bool,
) -> Result<()> {
    use explore::{find_interesting_profiles, ProfilesConfig};

    let config = ProfilesConfig {
        seeds,
        per_seed,
        min_repos,
        limit,
        expand,
    };

    let profiles = find_interesting_profiles(db, &config)?;

    if profiles.is_empty() {
        eprintln!("\x1b[33m!\x1b[0m No interesting profiles found");
        eprintln!("    Try: --min-repos 1 or --per-seed 200");
        return Ok(());
    }

    println!();
    println!("\x1b[36mInteresting Profiles\x1b[0m\n");

    for (i, profile) in profiles.iter().enumerate() {
        // Header: rank, owner name, stars, repo count
        let github_url = format!("https://github.com/{}", profile.owner);
        let owner_link = format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", github_url, profile.owner);

        println!(
            "\x1b[33m{:>2}.\x1b[0m \x1b[1m{}\x1b[0m {} \x1b[90m({} interesting repos, avg sim: {:.2})\x1b[0m",
            i + 1,
            owner_link,
            format_stars(profile.total_stars),
            profile.interesting_count,
            profile.avg_similarity
        );

        // Show top repos with their matched seed
        for repo in profile.repos.iter().take(3) {
            let name = repo.details.full_name.split('/').nth(1).unwrap_or(&repo.details.full_name);
            let seed_short: String = repo.matched_seed.chars().take(30).collect();
            println!(
                "    \x1b[90m└\x1b[0m {} \x1b[33m{}★\x1b[0m \x1b[90m← {}\x1b[0m",
                name,
                repo.details.stars,
                seed_short
            );
        }

        if profile.repos.len() > 3 {
            println!("    \x1b[90m  ...and {} more\x1b[0m", profile.repos.len() - 3);
        }

        println!();
    }

    println!(
        "\x1b[36mFound {} interesting profiles\x1b[0m",
        profiles.len()
    );

    Ok(())
}

/// Backfill short embeddings from existing full embeddings (Matryoshka two-stage search)
/// Truncates 1536-dim embeddings to first 256 dims for fast coarse filtering
fn backfill_short_embeddings(db: &Database, batch_size: usize) -> Result<()> {
    use crate::openai::OPENAI_EMBEDDING_DIM_SHORT;

    let total_missing = db.count_embeddings_missing_short()?;

    if total_missing == 0 {
        println!("\x1b[32m✓\x1b[0m All embeddings already have short versions");
        return Ok(());
    }

    println!(
        "\x1b[36m..\x1b[0m Backfilling {} short embeddings ({}→{} dims) in batches of {}",
        total_missing,
        1536,
        OPENAI_EMBEDDING_DIM_SHORT,
        batch_size
    );

    let start = std::time::Instant::now();
    let mut processed = 0usize;

    loop {
        // Always fetch from offset 0 since we're inserting as we go (LEFT JOIN excludes already-done)
        let batch = db.get_embeddings_missing_short(batch_size, 0)?;

        if batch.is_empty() {
            break;
        }

        for (repo_id, embedding_bytes) in &batch {
            // Convert bytes to f32 slice
            let full_embedding: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    embedding_bytes.as_ptr() as *const f32,
                    embedding_bytes.len() / 4,
                )
            };

            // Truncate to short embedding (first 256 dims)
            if full_embedding.len() >= OPENAI_EMBEDDING_DIM_SHORT {
                let short_embedding = &full_embedding[..OPENAI_EMBEDDING_DIM_SHORT];
                db.insert_short_embedding(*repo_id, short_embedding)?;
            }
        }

        processed += batch.len();

        // Progress update every batch
        let elapsed = start.elapsed();
        let rate = processed as f64 / elapsed.as_secs_f64();
        let remaining = total_missing.saturating_sub(processed);
        let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { 0.0 };

        eprint!(
            "\r\x1b[36m..\x1b[0m Processed {}/{} ({:.1}%) - {:.0}/s - ETA: {:.0}s    ",
            processed,
            total_missing,
            (processed as f64 / total_missing as f64) * 100.0,
            rate,
            eta_secs
        );
    }

    eprintln!(); // Newline after progress

    let elapsed = start.elapsed();
    println!(
        "\x1b[32m✓\x1b[0m Backfilled {} short embeddings in {:.1}s ({:.0}/s)",
        processed,
        elapsed.as_secs_f64(),
        processed as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

