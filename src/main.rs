mod config;
mod db;
mod discovery;
mod embedding;
mod formatting;
mod github;
mod http;
mod papers;
mod proxy;
mod search;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::time::Duration;

use config::Config;
use db::Database;
use discovery::{discover_from_owners, discover_owner_repos};
use embedding::{build_embedding_text, embed_passage, embed_passages};
use formatting::{format_repo_link, format_stars, truncate_str};
use github::GitHubClient;
use papers::extract_github_repos;
use proxy::ProxyManager;
use search::{expand_query, search};

/// Configuration for fetching repo metadata from GitHub
struct FetchConfig {
    limit: Option<usize>,
    batch_size: usize,
    concurrency: usize,
    debug: bool,
}

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
        /// Batch size for embedding (default: 200)
        #[arg(short, long, default_value = "200")]
        batch_size: usize,

        /// Maximum repos to embed (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Delay between batches in seconds (default: 5)
        #[arg(short, long, default_value = "5")]
        delay: u64,

        /// Clear all existing embeddings first (re-embed everything)
        #[arg(long)]
        reset: bool,
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

    /// Start HTTP server with SQL explorer interface
    #[command(hide = true)]
    Http {
        /// Port to listen on (default: 3000)
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
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
            let config = FetchConfig { limit, batch_size, concurrency, debug };
            fetch_from_db(&client, &db, &config).await
        }
        Some(Commands::Embed { batch_size, limit, delay, reset }) => {
            embed_missing(&db, batch_size, limit, delay, reset)
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
        Some(Commands::Http { port }) => {
            http::start_server(db.path().to_path_buf(), port).await
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

/// Fetch metadata from GitHub for repos in DB that don't have it yet
async fn fetch_from_db(
    client: &GitHubClient,
    db: &Database,
    config: &FetchConfig,
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

    let mut stored = 0;
    let mut errors = 0;
    let mut batch_num = 0;

    while stored + errors < to_fetch {
        batch_num += 1;
        let remaining = to_fetch - stored - errors;
        let this_batch_size = remaining.min(config.batch_size);

        // Get next batch of repos without metadata
        let repos_to_fetch = db.get_repos_without_metadata(Some(this_batch_size))?;

        if repos_to_fetch.is_empty() {
            break;
        }

        eprintln!(
            "\x1b[36m..\x1b[0m Batch {}: fetching {} repos via GraphQL...",
            batch_num, repos_to_fetch.len()
        );

        // Debug: show sample of repos being fetched
        if config.debug {
            eprintln!("  [DEBUG] Sample repos to fetch:");
            for (i, name) in repos_to_fetch.iter().take(5).enumerate() {
                eprintln!("    {}. {}", i + 1, name);
            }
            if repos_to_fetch.len() > 5 {
                eprintln!("    ... and {} more", repos_to_fetch.len() - 5);
            }
        }

        // Fetch via GraphQL
        let fetched_repos = match client.fetch_repos_batch(&repos_to_fetch, config.concurrency).await {
            Ok(repos) => repos,
            Err(e) => {
                eprintln!("\x1b[31mx\x1b[0m Batch {} failed: {}", batch_num, e);
                errors += repos_to_fetch.len();
                continue;
            }
        };

        // Store metadata without embedding
        let fetched_count = fetched_repos.len();
        let fetched_names: std::collections::HashSet<_> = fetched_repos
            .iter()
            .map(|r| r.full_name.to_lowercase())
            .collect();

        let mut total_discovered = 0;

        for repo in fetched_repos {
            let text = build_embedding_text(
                &repo.full_name,
                repo.description.as_deref(),
                &repo.topics,
                repo.language.as_deref(),
                repo.readme.as_deref(),
            );

            db.upsert_repo_metadata_only(&repo, &text)?;
            stored += 1;

            // Extract linked repos from README for organic growth (like a web scraper)
            if let Some(readme) = &repo.readme {
                total_discovered += discover_repos_from_readme(db, readme);
            }
        }

        // Mark repos that weren't found as gone (deleted/renamed)
        let missing: Vec<String> = repos_to_fetch
            .iter()
            .filter(|name| !fetched_names.contains(&name.to_lowercase()))
            .cloned()
            .collect();

        if !missing.is_empty() {
            let gone_count = db.mark_as_gone_bulk(&missing)?;
            errors += gone_count;
            let gone_pct = (gone_count as f64 / repos_to_fetch.len() as f64 * 100.0) as u32;
            eprintln!("  ... marked {} repos as gone ({}% of batch)", gone_count, gone_pct);

            // Warn if too many repos are gone
            if gone_pct >= 80 {
                eprintln!("  \x1b[33m⚠ Warning: {}% of batch is gone - queue may be mostly stale repos\x1b[0m", gone_pct);
            }

            // Debug: show sample of gone repos
            if config.debug {
                eprintln!("  [DEBUG] Sample gone repos:");
                for (i, name) in missing.iter().take(5).enumerate() {
                    eprintln!("    {}. {}", i + 1, name);
                }
                if missing.len() > 5 {
                    eprintln!("    ... and {} more", missing.len() - 5);
                }
            }
        }

        eprintln!("  ... stored {} repos (total: {})", fetched_count, stored);

        if total_discovered > 0 {
            eprintln!("  ... discovered {} new repos from READMEs", total_discovered);
        }

        // Checkpoint WAL periodically to prevent unbounded growth
        if batch_num % 10 == 0 {
            let _ = db.checkpoint();
        }

        // Small delay between batches to avoid rate limits
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Final checkpoint before exit
    let _ = db.checkpoint();

    eprintln!(
        "\x1b[32mok\x1b[0m Fetched {} repos ({} errors) - ready for embedding",
        stored, errors
    );

    // Show how many need embedding
    let need_embed = db.count_repos_without_embeddings()?;
    eprintln!("\x1b[36m..\x1b[0m {} repos now awaiting embeddings", need_embed);
    eprintln!("    Run: goto-gh embed --batch-size 200 --delay 5");

    Ok(())
}

/// Generate embeddings for repos that don't have them yet
fn embed_missing(db: &Database, batch_size: usize, limit: Option<usize>, delay_secs: u64, reset: bool) -> Result<()> {
    if reset {
        let count = db.clear_all_embeddings()?;
        eprintln!("\x1b[33m!\x1b[0m Cleared {} existing embeddings", count);
    }

    let need_embed = db.count_repos_without_embeddings()?;

    if need_embed == 0 {
        eprintln!("\x1b[32mok\x1b[0m All repos already have embeddings");
        return Ok(());
    }

    let to_process = limit.map(|l| l.min(need_embed)).unwrap_or(need_embed);
    eprintln!(
        "\x1b[36m..\x1b[0m Generating embeddings for {} repos (batch size: {})",
        to_process, batch_size
    );

    let mut total_embedded = 0;
    let mut offset = 0;

    while total_embedded < to_process {
        // Fetch a batch of repos without embeddings
        let remaining = to_process - total_embedded;
        let this_batch_size = remaining.min(batch_size);

        let repos = db.get_repos_without_embeddings(Some(this_batch_size))?;

        if repos.is_empty() {
            break;
        }

        let batch_len = repos.len();
        eprintln!(
            "\x1b[36m..\x1b[0m Batch {}: embedding {} repos...",
            offset / batch_size + 1,
            batch_len
        );

        // Extract texts for batch embedding
        let texts: Vec<String> = repos.iter().map(|(_, text)| text.clone()).collect();
        let repo_ids: Vec<i64> = repos.iter().map(|(id, _)| *id).collect();

        // Generate embeddings in batch
        let embeddings = embed_passages(&texts)?;

        // Store embeddings
        for (repo_id, embedding) in repo_ids.into_iter().zip(embeddings) {
            db.upsert_embedding(repo_id, &embedding)?;
        }

        total_embedded += batch_len;
        offset += batch_len;

        eprintln!(
            "  ... embedded {} (total: {}/{})",
            batch_len, total_embedded, to_process
        );

        // Pause between batches to avoid overloading the embedding API
        // Note: embed_missing is sync, but this runs in tokio context via block_on
        // Using std::thread::sleep is intentional here to not block other async tasks
        if total_embedded < to_process && delay_secs > 0 {
            eprintln!("  \x1b[90m⏳ Waiting {}s before next batch...\x1b[0m", delay_secs);
            std::thread::sleep(std::time::Duration::from_secs(delay_secs));
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Generated embeddings for {} repos",
        total_embedded
    );

    // Show remaining
    let still_need = db.count_repos_without_embeddings()?;
    if still_need > 0 {
        eprintln!("\x1b[36m..\x1b[0m {} repos still awaiting embeddings", still_need);
    }

    Ok(())
}

/// Discover and add repo stubs from a single README
/// Returns the number of new repos added
fn discover_repos_from_readme(db: &Database, readme: &str) -> usize {
    let linked = extract_github_repos(readme);
    if linked.is_empty() {
        return 0;
    }

    let unique: std::collections::HashSet<_> = linked.into_iter().collect();
    let repos_vec: Vec<String> = unique.into_iter().collect();

    match db.add_repo_stubs_bulk(&repos_vec) {
        Ok((added, _)) => added,
        Err(_) => 0,
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

