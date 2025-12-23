mod config;
mod db;
mod embedding;
mod github;

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::stream::{self, StreamExt};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use config::Config;
use db::Database;
use embedding::{build_embedding_text, embed_text, embed_texts};
use github::GitHubClient;

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

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Index top repositories by stars (default) or by search query
    Index {
        /// Optional search query (default: top repos by stars)
        #[arg(short, long)]
        query: Option<String>,

        /// Number of repos to index (default: 50000)
        #[arg(short, long, default_value = "50000")]
        count: u32,

        /// Start fresh (ignore checkpoint, re-index from beginning)
        #[arg(long)]
        full: bool,

        /// Number of parallel API workers (default: 1, max: 3 to avoid rate limits)
        #[arg(short, long, default_value = "1")]
        workers: usize,
    },

    /// Add a specific repository by name
    Add {
        /// Repository full name (e.g., "qdrant/qdrant")
        repo: String,
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

    /// Re-generate embeddings from stored data (no API calls)
    #[command(hide = true)]
    Revectorize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let db = Database::open()?;
    let token = Config::github_token();
    let client = GitHubClient::new(token.clone());

    // If query provided, do semantic search
    if !cli.query.is_empty() {
        let query = cli.query.join(" ");
        return search(&query, cli.limit, &db);
    }

    match cli.command {
        Some(Commands::Index { query, count, full, workers }) => {
            if token.is_none() {
                eprintln!("\x1b[33m..\x1b[0m No GitHub token found. Rate limit: 60 req/hour");
                eprintln!("  Set GITHUB_TOKEN or run: gh auth login");
            }
            index_repos(&client, &db, &query, count, full, workers).await
        }
        Some(Commands::Add { repo }) => {
            add_repo(&client, &db, &repo).await
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
        Some(Commands::Revectorize) => {
            revectorize(&db)
        }
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            eprintln!();
            std::process::exit(0);
        }
    }
}

/// Semantic search in local index
fn search(query: &str, limit: usize, db: &Database) -> Result<()> {
    let (_total, indexed) = db.stats()?;

    if indexed == 0 {
        eprintln!("\x1b[31mx\x1b[0m No repositories indexed yet.");
        eprintln!("  Run: goto-gh index \"<query>\" to index some repos first.");
        std::process::exit(1);
    }

    let dots = Dots::start(&format!("Searching {} repos", indexed));

    // Embed query
    let query_embedding = embed_text(query)?;

    // Find similar
    let results = db.find_similar(&query_embedding, limit * 2)?;

    dots.stop();

    if results.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m No matching repositories found.");
        return Ok(());
    }

    // Apply boosting and display
    let query_lower = query.to_lowercase();

    let mut boosted: Vec<_> = results
        .into_iter()
        .filter_map(|(id, distance)| {
            let repo = db.get_repo_by_id(id).ok()??;
            let base_score = (1.0 / (1.0 + distance)) * 100.0;

            // Boost for name match
            let name_lower = repo.full_name.to_lowercase();
            let boosted_score = if name_lower.contains(&query_lower) {
                (base_score + 20.0).min(100.0)
            } else if query_lower.split_whitespace().all(|w| name_lower.contains(w)) {
                (base_score + 15.0).min(100.0)
            } else {
                base_score
            };

            Some((repo, boosted_score))
        })
        .collect();

    boosted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (repo, score)) in boosted.iter().take(limit).enumerate() {
        let lang = repo.language.as_deref().unwrap_or("?");
        let stars = format_stars(repo.stars);
        let desc = repo.description.as_deref().unwrap_or("No description");
        let desc_truncated = truncate_str(desc, 60);

        eprintln!(
            "\x1b[35m{:>2}.\x1b[0m \x1b[1m{}\x1b[0m \x1b[33m{}\x1b[0m \x1b[90m[{}]\x1b[0m \x1b[90m({:.0}%)\x1b[0m",
            i + 1,
            repo.full_name,
            stars,
            lang,
            score
        );
        eprintln!("    \x1b[90m{}\x1b[0m", desc_truncated);
        eprintln!("    {}", repo.url);
        eprintln!();
    }

    Ok(())
}

/// Index repos from GitHub using GraphQL (efficient: repo + README in one call)
async fn index_repos(client: &GitHubClient, db: &Database, query: &Option<String>, count: u32, full: bool, workers: usize) -> Result<()> {
    // Star ranges for pagination (GraphQL has same 1000 limit per query)
    let star_ranges: Vec<&str> = vec![
        "stars:>100000",
        "stars:50000..100000",
        "stars:30000..50000",
        "stars:20000..30000",
        "stars:15000..20000",
        "stars:10000..15000",
        "stars:8000..10000",
        "stars:6000..8000",
        "stars:5000..6000",
        "stars:4000..5000",
        "stars:3500..4000",
        "stars:3000..3500",
        "stars:2500..3000",
        "stars:2000..2500",
        "stars:1800..2000",
        "stars:1600..1800",
        "stars:1400..1600",
        "stars:1200..1400",
        "stars:1000..1200",
        "stars:900..1000",
        "stars:800..900",
        "stars:700..800",
        "stars:600..700",
        "stars:500..600",
        "stars:450..500",
        "stars:400..450",
        "stars:350..400",
        "stars:300..350",
        "stars:250..300",
        "stars:200..250",
        "stars:180..200",
        "stars:160..180",
        "stars:140..160",
        "stars:120..140",
        "stars:100..120",
        "stars:90..100",
        "stars:80..90",
        "stars:70..80",
        "stars:60..70",
        "stars:50..60",
        "stars:45..50",
        "stars:40..45",
        "stars:35..40",
        "stars:30..35",
        "stars:25..30",
        "stars:20..25",
    ];

    // Check for checkpoint (only for default star-based indexing)
    let (start_range_idx, _start_cursor, initial_fetched) = if query.is_none() && !full {
        if let Some((range_idx, cursor, fetched)) = db.get_checkpoint()? {
            eprintln!("\x1b[36m..\x1b[0m Resuming from checkpoint: {} (fetched: {})", star_ranges.get(range_idx).unwrap_or(&"?"), fetched);
            (range_idx, cursor, fetched)
        } else {
            // No checkpoint - try to infer from existing repos
            let repo_count = db.get_repo_count()?;
            if repo_count > 0 {
                if let Some(min_stars) = db.get_min_stars()? {
                    let inferred_idx = infer_star_range_idx(&star_ranges, min_stars);
                    eprintln!(
                        "\x1b[36m..\x1b[0m No checkpoint found, but {} repos exist (min stars: {})",
                        repo_count, min_stars
                    );
                    eprintln!(
                        "\x1b[36m..\x1b[0m Inferring resume point: {} (idx: {})",
                        star_ranges.get(inferred_idx).unwrap_or(&"?"), inferred_idx
                    );
                    db.save_checkpoint(inferred_idx, None, repo_count)?;
                    (inferred_idx, None, repo_count)
                } else {
                    (0, None, 0)
                }
            } else {
                (0, None, 0)
            }
        }
    } else {
        if full && query.is_none() {
            db.clear_checkpoint()?;
            eprintln!("\x1b[36m..\x1b[0m Starting fresh index (checkpoint cleared)");
        }
        (0, None, 0)
    };

    let workers = workers.max(1).min(3); // Clamp between 1-3 (avoid secondary rate limit)
    eprintln!("\x1b[36m..\x1b[0m Fetching top {} repos (GraphQL, {} workers)", count, workers);

    // Shared state for parallel fetching
    let total_fetched = Arc::new(std::sync::atomic::AtomicUsize::new(initial_fetched));
    let indexed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let skipped = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Embedding batch accumulator (shared across workers)
    const EMBED_BATCH_SIZE: usize = 1000;
    let pending_embed: Arc<Mutex<Vec<(github::RepoWithReadme, String)>>> = Arc::new(Mutex::new(Vec::new()));

    // Helper to flush pending embeddings
    async fn flush_embeddings(
        pending: &Arc<Mutex<Vec<(github::RepoWithReadme, String)>>>,
        db: &Database,
        indexed: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<()> {
        let mut pending_guard = pending.lock().await;
        if pending_guard.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = pending_guard.iter().map(|(_, t)| t.clone()).collect();
        let batch_len = texts.len();
        let repos_to_embed: Vec<_> = pending_guard.drain(..).collect();
        drop(pending_guard); // Release lock before expensive embedding

        let embeddings = embed_texts(&texts)?;

        for ((repo, text), embedding) in repos_to_embed.into_iter().zip(embeddings) {
            let repo_id = db.upsert_repo_with_readme(&repo, repo.readme.as_deref(), &text)?;
            db.upsert_embedding(repo_id, &embedding)?;
            indexed.fetch_add(1, Ordering::Relaxed);
        }
        let total = indexed.load(Ordering::Relaxed);
        eprintln!("  ... embedded {} repos (total: {})", batch_len, total);
        Ok(())
    }

    // Build list of ranges to process
    let ranges_to_process: Vec<(usize, &str)> = if query.is_some() {
        vec![(0, query.as_ref().unwrap().as_str())]
    } else {
        star_ranges.iter().enumerate()
            .skip(start_range_idx)
            .map(|(i, s)| (i, *s))
            .collect()
    };

    let use_checkpoint = query.is_none();
    let count_usize = count as usize;

    // Process ranges in parallel batches of `workers` size
    let client = Arc::new(client);
    let db = Arc::new(db);

    for chunk in ranges_to_process.chunks(workers) {
        if total_fetched.load(Ordering::Relaxed) >= count_usize {
            break;
        }

        // Fetch multiple ranges in parallel
        let fetch_tasks: Vec<_> = chunk.iter().map(|(range_idx, range)| {
            let client = Arc::clone(&client);
            let range = range.to_string();
            let range_idx = *range_idx;
            let total_fetched = Arc::clone(&total_fetched);
            let count_usize = count_usize;

            async move {
                let mut results: Vec<github::RepoWithReadme> = Vec::new();
                let mut cursor: Option<String> = None;

                loop {
                    if total_fetched.load(Ordering::Relaxed) >= count_usize {
                        break;
                    }

                    let batch_size = 100;
                    match client.search_repos_graphql(&range, batch_size, cursor.clone()).await {
                        Ok((repos, next_cursor, has_next)) => {
                            if repos.is_empty() {
                                break;
                            }
                            results.extend(repos);

                            if !has_next {
                                break;
                            }
                            cursor = next_cursor;
                        }
                        Err(e) => {
                            eprintln!("  {} - error: {}", range, e);
                            break;
                        }
                    }

                    // Limit pages per range to avoid one range hogging everything
                    if results.len() >= 500 {
                        break;
                    }
                }

                (range_idx, range, results)
            }
        }).collect();

        // Wait for all parallel fetches
        let fetch_results: Vec<_> = stream::iter(fetch_tasks)
            .buffer_unordered(workers)
            .collect()
            .await;

        // Process results and accumulate for embedding
        for (_range_idx, range, repos) in fetch_results {
            if repos.is_empty() {
                continue;
            }

            let mut page_new = 0;
            for repo in repos {
                if repo.fork {
                    continue;
                }
                total_fetched.fetch_add(1, Ordering::Relaxed);

                if db.is_fresh(&repo.full_name, 7)? {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                let text = build_embedding_text(
                    &repo.full_name,
                    repo.description.as_deref(),
                    &repo.topics,
                    repo.language.as_deref(),
                    repo.readme.as_deref(),
                );

                pending_embed.lock().await.push((repo, text));
                page_new += 1;
            }
            eprintln!("  {} +{} new", range, page_new);

            // Flush if we have enough pending
            if pending_embed.lock().await.len() >= EMBED_BATCH_SIZE {
                flush_embeddings(&pending_embed, &db, &indexed).await?;
            }
        }

        // Save checkpoint after each batch of parallel ranges
        if use_checkpoint {
            let last_range_idx = chunk.last().map(|(i, _)| *i).unwrap_or(0);
            let fetched = total_fetched.load(Ordering::Relaxed);
            db.save_checkpoint(last_range_idx + 1, None, fetched)?;
        }

        // Small delay between batches to avoid secondary rate limit
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Flush any remaining pending embeddings
    flush_embeddings(&pending_embed, &db, &indexed).await?;

    // Clear checkpoint when done (reached count target)
    let final_fetched = total_fetched.load(Ordering::Relaxed);
    if use_checkpoint && final_fetched >= count_usize {
        db.clear_checkpoint()?;
    }

    let final_indexed = indexed.load(Ordering::Relaxed);
    let final_skipped = skipped.load(Ordering::Relaxed);
    eprintln!(
        "\x1b[32mok\x1b[0m Indexed {} new repos ({} skipped as fresh)",
        final_indexed, final_skipped
    );

    Ok(())
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

    let embedding = embed_text(&text)?;

    let repo_id = db.upsert_repo(&repo, readme.as_deref(), &text)?;
    db.upsert_embedding(repo_id, &embedding)?;

    eprintln!("\x1b[32mok\x1b[0m Added \x1b[1m{}\x1b[0m", repo.full_name);
    if let Some(desc) = &repo.description {
        eprintln!("  {}", desc);
    }

    Ok(())
}

/// Fuzzy search by repo name
fn find_by_name(db: &Database, pattern: &str, limit: usize) -> Result<()> {
    let repos = db.find_by_name(pattern, limit)?;

    if repos.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m No repos matching '{}'", pattern);
        return Ok(());
    }

    eprintln!("\x1b[36m{} repos matching '{}'\x1b[0m\n", repos.len(), pattern);

    for repo in repos {
        let lang = repo.language.as_deref().unwrap_or("?");
        let stars = format_stars(repo.stars);
        let desc = repo.description.as_deref().unwrap_or("");
        let desc_truncated = truncate_str(desc, 60);

        eprintln!(
            "\x1b[1m{}\x1b[0m \x1b[33m{}\x1b[0m \x1b[90m[{}]\x1b[0m",
            repo.full_name, stars, lang
        );
        if !desc_truncated.is_empty() {
            eprintln!("  \x1b[90m{}\x1b[0m", desc_truncated);
        }
    }

    Ok(())
}

/// Truncate string safely at char boundary
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 3).collect();
        format!("{}...", truncated)
    }
}

/// Show statistics
fn show_stats(db: &Database) -> Result<()> {
    let (total, indexed) = db.stats()?;

    eprintln!("\x1b[36mIndex Statistics\x1b[0m\n");
    eprintln!("  \x1b[90mTotal repos:\x1b[0m   {}", total);
    eprintln!("  \x1b[90mWith embeddings:\x1b[0m {}", indexed);

    Ok(())
}

/// Check rate limit
async fn check_rate_limit(client: &GitHubClient) -> Result<()> {
    let rate = client.rate_limit().await?;

    let reset_time = chrono::DateTime::from_timestamp(rate.reset as i64, 0)
        .map(|dt| dt.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "?".to_string());

    eprintln!("\x1b[36mGitHub API Rate Limit\x1b[0m\n");
    eprintln!("  \x1b[90mLimit:\x1b[0m     {}/hour", rate.limit);
    eprintln!("  \x1b[90mRemaining:\x1b[0m {}", rate.remaining);
    eprintln!("  \x1b[90mResets at:\x1b[0m {}", reset_time);

    Ok(())
}

/// Re-generate embeddings from stored data (no API calls)
fn revectorize(db: &Database) -> Result<()> {
    let repos = db.get_all_repos_raw()?;
    let total = repos.len();

    if total == 0 {
        eprintln!("\x1b[31mx\x1b[0m No repositories in database.");
        return Ok(());
    }

    eprintln!("\x1b[36m..\x1b[0m Re-vectorizing {} repos from stored data", total);

    let mut processed = 0;
    let mut errors = 0;

    for (id, full_name, description, language, topics_json, readme) in repos {
        // Parse topics from JSON
        let topics: Vec<String> = topics_json
            .as_deref()
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or_default();

        // Rebuild embedding text
        let text = build_embedding_text(
            &full_name,
            description.as_deref(),
            &topics,
            language.as_deref(),
            readme.as_deref(),
        );

        // Generate new embedding
        match embed_text(&text) {
            Ok(embedding) => {
                if let Err(e) = db.update_embedding(id, &text, &embedding) {
                    eprintln!("  \x1b[31mx\x1b[0m {} - {}", full_name, e);
                    errors += 1;
                } else {
                    processed += 1;
                }
            }
            Err(e) => {
                eprintln!("  \x1b[31mx\x1b[0m {} - {}", full_name, e);
                errors += 1;
            }
        }

        // Progress
        if processed % 100 == 0 && processed > 0 {
            eprintln!("  ... processed {}/{}", processed, total);
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Re-vectorized {} repos ({} errors)",
        processed, errors
    );

    Ok(())
}

/// Infer which star range index to resume from based on minimum stars in DB
fn infer_star_range_idx(star_ranges: &[&str], min_stars: u64) -> usize {
    // Parse star ranges to find where min_stars falls
    // Ranges are like "stars:>100000", "stars:50000..100000", etc.
    for (idx, range) in star_ranges.iter().enumerate() {
        if let Some(lower) = parse_star_range_lower(range) {
            if min_stars >= lower {
                return idx;
            }
        }
    }
    // Default to last range if not found
    star_ranges.len().saturating_sub(1)
}

/// Parse the lower bound of a star range
fn parse_star_range_lower(range: &str) -> Option<u64> {
    let range = range.strip_prefix("stars:")?;
    if range.starts_with('>') {
        // "stars:>100000" -> lower bound is 100001
        range[1..].parse().ok()
    } else if let Some((lower, _)) = range.split_once("..") {
        // "stars:50000..100000" -> lower bound is 50000
        lower.parse().ok()
    } else {
        None
    }
}

/// Format star count (e.g., 1.2k, 15k)
fn format_stars(stars: u64) -> String {
    if stars >= 1000 {
        format!("{}k", stars / 1000)
    } else {
        format!("{}", stars)
    }
}

/// Animated dots spinner
struct Dots {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Dots {
    fn start(message: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let msg = message.to_string();

        let handle = std::thread::spawn(move || {
            const FRAMES: &[&str] = &[
                "\u{28CB}", "\u{28D9}", "\u{28F9}", "\u{28F8}",
                "\u{28FC}", "\u{28F4}", "\u{28E6}", "\u{28E7}",
                "\u{28C7}", "\u{28CF}",
            ];
            let mut i = 0;
            while running_clone.load(Ordering::Relaxed) {
                eprint!("\r\x1b[36m{}\x1b[0m {}", FRAMES[i % 10], msg);
                let _ = io::stderr().flush();
                std::thread::sleep(Duration::from_millis(80));
                i += 1;
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        eprint!("\r\x1b[K"); // Clear line
    }
}

impl Drop for Dots {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
