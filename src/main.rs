mod config;
mod db;
mod embedding;
mod github;

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
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

    /// Crawl awesome-list repos to discover and index linked repositories
    Crawl {
        /// Awesome-list repo (e.g., "sindresorhus/awesome" or a URL)
        source: String,

        /// Maximum repos to index from the list
        #[arg(short, long, default_value = "1000")]
        limit: usize,
    },

    /// Import repositories from a file (one "owner/repo" per line)
    Import {
        /// Path to file containing repo names (one per line)
        file: String,

        /// Number of repos to process (default: all)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Skip first N repos (for resuming)
        #[arg(short, long, default_value = "0")]
        skip: usize,
    },

    /// Fetch metadata from GitHub for repos in DB that don't have it yet
    Fetch {
        /// Number of repos to fetch (default: all pending)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Batch size for GraphQL queries (default: 500)
        #[arg(short, long, default_value = "500")]
        batch_size: usize,
    },

    /// Generate embeddings for repos that don't have them yet
    Embed {
        /// Batch size for embedding (default: 1000)
        #[arg(short, long, default_value = "1000")]
        batch_size: usize,

        /// Maximum repos to embed (default: all)
        #[arg(short, long)]
        limit: Option<usize>,
    },
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
        Some(Commands::Load { file }) => {
            load_repo_stubs(&db, &file)
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
        Some(Commands::Crawl { source, limit }) => {
            crawl_awesome_list(&client, &db, &source, limit).await
        }
        Some(Commands::Import { file, limit, skip }) => {
            import_from_file(&client, &db, &file, limit, skip).await
        }
        Some(Commands::Fetch { limit, batch_size }) => {
            fetch_from_db(&client, &db, limit, batch_size).await
        }
        Some(Commands::Embed { batch_size, limit }) => {
            embed_missing(&db, batch_size, limit)
        }
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            eprintln!();
            std::process::exit(0);
        }
    }
}

/// Hybrid search (semantic + keyword + name match with RRF fusion)
fn search(query: &str, limit: usize, db: &Database) -> Result<()> {
    let (_total, indexed) = db.stats()?;

    if indexed == 0 {
        eprintln!("\x1b[31mx\x1b[0m No repositories indexed yet.");
        eprintln!("  Run: goto-gh index \"<query>\" to index some repos first.");
        std::process::exit(1);
    }

    let dots = Dots::start(&format!("Searching {} repos (hybrid)", indexed));

    // 1. Semantic search via embeddings
    let query_embedding = embed_text(query)?;
    let vector_results = db.find_similar(&query_embedding, limit * 3)?;

    // 2. Name match search (strongest signal - repos with query in name)
    let name_results = db.find_by_name_match(query, limit * 3)?;

    // 3. Content keyword search via LIKE on embedded_text
    let keyword_results = db.find_by_keywords(query, limit * 3)?;

    dots.stop();

    // RRF (Reciprocal Rank Fusion) with weighted signals
    // Lower k = higher weight for top ranks
    let k_name = 20.0;    // Strong boost for name matches
    let k_vector = 60.0;  // Standard weight for semantic
    let k_keyword = 80.0; // Lower weight for keyword (often noisy)

    let mut scores: HashMap<i64, f32> = HashMap::new();

    // Add name match scores (strongest signal)
    for (rank, repo_id) in name_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += 1.0 / (k_name + rank as f32 + 1.0);
    }

    // Add vector search scores
    for (rank, (repo_id, _distance)) in vector_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += 1.0 / (k_vector + rank as f32 + 1.0);
    }

    // Add keyword search scores (weakest signal)
    for (rank, repo_id) in keyword_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += 1.0 / (k_keyword + rank as f32 + 1.0);
    }

    if scores.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m No matching repositories found.");
        return Ok(());
    }

    // Sort by combined RRF score
    let mut combined: Vec<_> = scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Display results
    // Max theoretical score: 1/21 + 1/61 + 1/81 ≈ 0.072
    let max_score = 1.0 / (k_name + 1.0) + 1.0 / (k_vector + 1.0) + 1.0 / (k_keyword + 1.0);

    for (i, (repo_id, rrf_score)) in combined.iter().take(limit).enumerate() {
        let repo = match db.get_repo_by_id(*repo_id)? {
            Some(r) => r,
            None => continue,
        };

        let lang = repo.language.as_deref().unwrap_or("?");
        let stars = format_stars(repo.stars);
        let desc = repo.description.as_deref().unwrap_or("No description");
        let desc_truncated = truncate_str(desc, 60);

        let display_score = (rrf_score / max_score * 100.0).min(100.0);

        eprintln!(
            "\x1b[35m{:>2}.\x1b[0m \x1b[1m{}\x1b[0m \x1b[33m{}\x1b[0m \x1b[90m[{}]\x1b[0m \x1b[90m({:.0}%)\x1b[0m",
            i + 1,
            repo.full_name,
            stars,
            lang,
            display_score
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
    let without_metadata = db.count_repos_without_metadata()?;
    let without_embeddings = db.count_repos_without_embeddings()?;
    let gone = db.count_gone()?;

    eprintln!("\x1b[36mIndex Statistics\x1b[0m\n");
    eprintln!("  \x1b[90mTotal repos:\x1b[0m        {}", total);
    eprintln!("  \x1b[90mWith metadata:\x1b[0m      {}", total - without_metadata - gone);
    eprintln!("  \x1b[90mWith embeddings:\x1b[0m    {}", indexed);
    eprintln!();
    eprintln!("  \x1b[90mNeed metadata:\x1b[0m      {}", without_metadata);
    eprintln!("  \x1b[90mNeed embeddings:\x1b[0m    {}", without_embeddings);
    if gone > 0 {
        eprintln!("  \x1b[90mGone (deleted):\x1b[0m     {}", gone);
    }

    if without_metadata > 0 {
        eprintln!("\n  \x1b[33mTip:\x1b[0m Run: goto-gh fetch");
    } else if without_embeddings > 0 {
        eprintln!("\n  \x1b[33mTip:\x1b[0m Run: goto-gh embed");
    }

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

/// Crawl an awesome-list to discover and index linked repos
async fn crawl_awesome_list(client: &GitHubClient, db: &Database, source: &str, limit: usize) -> Result<()> {
    // Parse source - could be "owner/repo" or a full URL
    let repo_name = if source.starts_with("http") {
        // Extract owner/repo from URL like https://github.com/sindresorhus/awesome
        source
            .trim_end_matches('/')
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/")
    } else {
        source.to_string()
    };

    eprintln!("\x1b[36m..\x1b[0m Fetching README from {} via GraphQL...", repo_name);

    // Fetch the README via GraphQL (more efficient, separate rate limit)
    let source_repos = match client.fetch_repos_batch(&[repo_name.clone()]).await {
        Ok(repos) => repos,
        Err(e) => {
            eprintln!("\x1b[31mx\x1b[0m Failed to fetch source repo {}: {}", repo_name, e);
            return Ok(());
        }
    };

    let readme = match source_repos.into_iter().next() {
        Some(repo) => match repo.readme {
            Some(content) => content,
            None => {
                eprintln!("\x1b[31mx\x1b[0m No README found for {}", repo_name);
                return Ok(());
            }
        },
        None => {
            eprintln!("\x1b[31mx\x1b[0m Repository {} not found", repo_name);
            return Ok(());
        }
    };

    // Extract GitHub repo links from markdown
    let repo_links = extract_github_repos(&readme);
    let unique_repos: std::collections::HashSet<_> = repo_links.into_iter().collect();

    eprintln!("\x1b[36m..\x1b[0m Found {} unique GitHub repos in README", unique_repos.len());

    if unique_repos.is_empty() {
        return Ok(());
    }

    // Filter out already-fresh repos first
    let mut indexed = 0;
    let mut skipped = 0;
    let mut errors = 0;

    let all_repos: Vec<_> = unique_repos.into_iter().take(limit).collect();
    let repos_to_fetch: Vec<_> = all_repos
        .iter()
        .filter(|name| !db.is_fresh(name, 7).unwrap_or(false))
        .cloned()
        .collect();

    skipped = all_repos.len() - repos_to_fetch.len();

    if repos_to_fetch.is_empty() {
        eprintln!(
            "\x1b[32mok\x1b[0m Crawled {} - {} repos all fresh (skipped)",
            source, skipped
        );
        return Ok(());
    }

    eprintln!(
        "\x1b[36m..\x1b[0m Fetching {} repos via GraphQL ({} already fresh)...",
        repos_to_fetch.len(),
        skipped
    );

    // Fetch all repos via GraphQL in batches (much more efficient than REST)
    let fetched_repos = match client.fetch_repos_batch(&repos_to_fetch).await {
        Ok(repos) => repos,
        Err(e) => {
            eprintln!("\x1b[31mx\x1b[0m Failed to fetch repos: {}", e);
            return Ok(());
        }
    };

    eprintln!(
        "\x1b[36m..\x1b[0m Processing {} fetched repos...",
        fetched_repos.len()
    );

    // Convert RepoWithReadme to the format expected by db.upsert_repo
    for repo in fetched_repos {
        let text = build_embedding_text(
            &repo.full_name,
            repo.description.as_deref(),
            &repo.topics,
            repo.language.as_deref(),
            repo.readme.as_deref(),
        );

        match embed_text(&text) {
            Ok(embedding) => {
                // Create a GitHubRepo from the GraphQL response
                let github_repo = github::GitHubRepo {
                    full_name: repo.full_name.clone(),
                    description: repo.description.clone(),
                    html_url: repo.html_url.clone(),
                    stargazers_count: repo.stars,
                    language: repo.language.clone(),
                    topics: repo.topics.clone(),
                };

                let repo_id = db.upsert_repo(&github_repo, repo.readme.as_deref(), &text)?;
                db.upsert_embedding(repo_id, &embedding)?;
                indexed += 1;
            }
            Err(e) => {
                eprintln!("  \x1b[31mx\x1b[0m {} - embedding error: {}", repo.full_name, e);
                errors += 1;
            }
        }
    }

    // Count repos we couldn't find
    let not_found = repos_to_fetch.len() - indexed - errors;
    if not_found > 0 {
        errors += not_found;
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Crawled {} - indexed {} new, {} skipped, {} errors",
        source, indexed, skipped, errors
    );

    Ok(())
}

/// Import repos from a file (one "owner/repo" per line)
async fn import_from_file(
    client: &GitHubClient,
    db: &Database,
    file_path: &str,
    limit: Option<usize>,
    skip: usize,
) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Read file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let all_repos: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && l.contains('/'))
        .skip(skip)
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    let total = all_repos.len();
    eprintln!(
        "\x1b[36m..\x1b[0m Importing {} repos from {} (skip: {}, limit: {:?})",
        total, file_path, skip, limit
    );

    if total == 0 {
        eprintln!("\x1b[33m..\x1b[0m No repos to import");
        return Ok(());
    }

    // Filter out repos already in DB (fresh within 7 days)
    let repos_to_fetch: Vec<String> = all_repos
        .iter()
        .filter(|name| !db.is_fresh(name, 7).unwrap_or(false))
        .cloned()
        .collect();

    let skipped = total - repos_to_fetch.len();
    eprintln!(
        "\x1b[36m..\x1b[0m {} repos to fetch ({} already fresh)",
        repos_to_fetch.len(), skipped
    );

    if repos_to_fetch.is_empty() {
        eprintln!("\x1b[32mok\x1b[0m All {} repos already indexed", total);
        return Ok(());
    }

    // Process in batches
    const BATCH_SIZE: usize = 500;
    const EMBED_BATCH_SIZE: usize = 1000;

    let mut indexed = 0;
    let mut errors = 0;
    let mut pending_embed: Vec<(github::RepoWithReadme, String)> = Vec::new();

    for (batch_idx, batch) in repos_to_fetch.chunks(BATCH_SIZE).enumerate() {
        let batch_start = batch_idx * BATCH_SIZE + skip;
        eprintln!(
            "\x1b[36m..\x1b[0m Batch {}: fetching repos {}-{} via GraphQL...",
            batch_idx + 1,
            batch_start,
            batch_start + batch.len()
        );

        // Fetch via GraphQL
        let fetched_repos = match client.fetch_repos_batch(batch).await {
            Ok(repos) => repos,
            Err(e) => {
                eprintln!("\x1b[31mx\x1b[0m Batch {} failed: {}", batch_idx + 1, e);
                errors += batch.len();
                continue;
            }
        };

        // Accumulate for embedding
        let fetched_count = fetched_repos.len();
        for repo in fetched_repos {
            let text = build_embedding_text(
                &repo.full_name,
                repo.description.as_deref(),
                &repo.topics,
                repo.language.as_deref(),
                repo.readme.as_deref(),
            );
            pending_embed.push((repo, text));
        }

        // Record repos we couldn't find
        errors += batch.len() - fetched_count;

        // Flush embeddings if we have enough
        if pending_embed.len() >= EMBED_BATCH_SIZE {
            let batch_len = pending_embed.len();
            let texts: Vec<String> = pending_embed.iter().map(|(_, t)| t.clone()).collect();
            let repos_to_embed: Vec<_> = pending_embed.drain(..).collect();

            eprintln!("  ... embedding {} repos...", batch_len);
            let embeddings = embed_texts(&texts)?;

            for ((repo, text), embedding) in repos_to_embed.into_iter().zip(embeddings) {
                let github_repo = github::GitHubRepo {
                    full_name: repo.full_name.clone(),
                    description: repo.description.clone(),
                    html_url: repo.html_url.clone(),
                    stargazers_count: repo.stars,
                    language: repo.language.clone(),
                    topics: repo.topics.clone(),
                };

                let repo_id = db.upsert_repo(&github_repo, repo.readme.as_deref(), &text)?;
                db.upsert_embedding(repo_id, &embedding)?;
                indexed += 1;
            }
            eprintln!("  ... indexed {} total", indexed);
        }

        // Small delay between batches to avoid rate limits
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Flush remaining
    if !pending_embed.is_empty() {
        let batch_len = pending_embed.len();
        let texts: Vec<String> = pending_embed.iter().map(|(_, t)| t.clone()).collect();
        let repos_to_embed: Vec<_> = pending_embed.drain(..).collect();

        eprintln!("  ... embedding final {} repos...", batch_len);
        let embeddings = embed_texts(&texts)?;

        for ((repo, text), embedding) in repos_to_embed.into_iter().zip(embeddings) {
            let github_repo = github::GitHubRepo {
                full_name: repo.full_name.clone(),
                description: repo.description.clone(),
                html_url: repo.html_url.clone(),
                stargazers_count: repo.stars,
                language: repo.language.clone(),
                topics: repo.topics.clone(),
            };

            let repo_id = db.upsert_repo(&github_repo, repo.readme.as_deref(), &text)?;
            db.upsert_embedding(repo_id, &embedding)?;
            indexed += 1;
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Imported {} new repos ({} skipped, {} errors)",
        indexed, skipped, errors
    );

    Ok(())
}

/// Fetch repos from file without generating embeddings (metadata-only)
async fn fetch_from_file(
    client: &GitHubClient,
    db: &Database,
    file_path: &str,
    limit: Option<usize>,
    skip: usize,
) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Read file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let all_repos: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && l.contains('/'))
        .skip(skip)
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    let total = all_repos.len();
    eprintln!(
        "\x1b[36m..\x1b[0m Fetching {} repos from {} (skip: {}, limit: {:?}) - NO EMBEDDINGS",
        total, file_path, skip, limit
    );

    if total == 0 {
        eprintln!("\x1b[33m..\x1b[0m No repos to fetch");
        return Ok(());
    }

    // Filter out repos already in DB (fresh within 7 days)
    let repos_to_fetch: Vec<String> = all_repos
        .iter()
        .filter(|name| !db.is_fresh(name, 7).unwrap_or(false))
        .cloned()
        .collect();

    let skipped = total - repos_to_fetch.len();
    eprintln!(
        "\x1b[36m..\x1b[0m {} repos to fetch ({} already fresh)",
        repos_to_fetch.len(), skipped
    );

    if repos_to_fetch.is_empty() {
        eprintln!("\x1b[32mok\x1b[0m All {} repos already indexed", total);
        return Ok(());
    }

    // Process in batches (larger since we're not embedding)
    const BATCH_SIZE: usize = 500;

    let mut stored = 0;
    let mut errors = 0;

    for (batch_idx, batch) in repos_to_fetch.chunks(BATCH_SIZE).enumerate() {
        let batch_start = batch_idx * BATCH_SIZE + skip;
        eprintln!(
            "\x1b[36m..\x1b[0m Batch {}: fetching repos {}-{} via GraphQL...",
            batch_idx + 1,
            batch_start,
            batch_start + batch.len()
        );

        // Fetch via GraphQL
        let fetched_repos = match client.fetch_repos_batch(batch).await {
            Ok(repos) => repos,
            Err(e) => {
                eprintln!("\x1b[31mx\x1b[0m Batch {} failed: {}", batch_idx + 1, e);
                errors += batch.len();
                continue;
            }
        };

        // Store metadata without embedding
        let fetched_count = fetched_repos.len();
        for repo in fetched_repos {
            let text = build_embedding_text(
                &repo.full_name,
                repo.description.as_deref(),
                &repo.topics,
                repo.language.as_deref(),
                repo.readme.as_deref(),
            );

            // Store metadata + embedded_text but NO embedding
            db.upsert_repo_metadata_only(&repo, &text)?;
            stored += 1;
        }

        // Record repos we couldn't find
        errors += batch.len() - fetched_count;

        eprintln!("  ... stored {} repos (total: {})", fetched_count, stored);

        // Small delay between batches to avoid rate limits
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Fetched {} repos ({} skipped, {} errors) - ready for embedding",
        stored, skipped, errors
    );

    // Show how many need embedding
    let need_embed = db.count_repos_without_embeddings()?;
    eprintln!("\x1b[36m..\x1b[0m {} repos now awaiting embeddings", need_embed);
    eprintln!("    Run: goto-gh embed --batch-size 1000");

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
    eprintln!("    Run: goto-gh fetch --batch-size 500");

    Ok(())
}

/// Fetch metadata from GitHub for repos in DB that don't have it yet
async fn fetch_from_db(
    client: &GitHubClient,
    db: &Database,
    limit: Option<usize>,
    batch_size: usize,
) -> Result<()> {
    let need_fetch = db.count_repos_without_metadata()?;

    if need_fetch == 0 {
        eprintln!("\x1b[32mok\x1b[0m All repos already have metadata");
        return Ok(());
    }

    let to_fetch = limit.map(|l| l.min(need_fetch)).unwrap_or(need_fetch);
    eprintln!(
        "\x1b[36m..\x1b[0m Fetching metadata for {} repos (batch size: {}) - NO EMBEDDINGS",
        to_fetch, batch_size
    );

    let mut stored = 0;
    let mut errors = 0;
    let mut batch_num = 0;

    while stored + errors < to_fetch {
        batch_num += 1;
        let remaining = to_fetch - stored - errors;
        let this_batch_size = remaining.min(batch_size);

        // Get next batch of repos without metadata
        let repos_to_fetch = db.get_repos_without_metadata(Some(this_batch_size))?;

        if repos_to_fetch.is_empty() {
            break;
        }

        eprintln!(
            "\x1b[36m..\x1b[0m Batch {}: fetching {} repos via GraphQL...",
            batch_num, repos_to_fetch.len()
        );

        // Fetch via GraphQL
        let fetched_repos = match client.fetch_repos_batch(&repos_to_fetch).await {
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

        let mut discovered_repos: Vec<String> = Vec::new();

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

            // Extract linked repos from README for organic growth
            if let Some(readme) = &repo.readme {
                let linked = extract_github_repos(readme);
                discovered_repos.extend(linked);
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
            eprintln!("  ... marked {} repos as gone (deleted/renamed)", gone_count);
        }

        eprintln!("  ... stored {} repos (total: {})", fetched_count, stored);

        // Add discovered repos as stubs for organic growth
        if !discovered_repos.is_empty() {
            let unique_discovered: std::collections::HashSet<_> = discovered_repos.into_iter().collect();
            let discovered_vec: Vec<String> = unique_discovered.into_iter().collect();
            let (added, _) = db.add_repo_stubs_bulk(&discovered_vec)?;
            if added > 0 {
                eprintln!("  ... discovered {} new repos from READMEs", added);
            }
        }

        // Small delay between batches to avoid rate limits
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Report total discovered
    let total_discovered = db.count_repos_without_metadata()?;

    eprintln!(
        "\x1b[32mok\x1b[0m Fetched {} repos ({} errors) - ready for embedding",
        stored, errors
    );

    // Show how many need embedding
    let need_embed = db.count_repos_without_embeddings()?;
    eprintln!("\x1b[36m..\x1b[0m {} repos now awaiting embeddings", need_embed);
    eprintln!("    Run: goto-gh embed --batch-size 1000");

    Ok(())
}

/// Generate embeddings for repos that don't have them yet
fn embed_missing(db: &Database, batch_size: usize, limit: Option<usize>) -> Result<()> {
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
        let embeddings = embed_texts(&texts)?;

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
        if total_embedded < to_process {
            std::thread::sleep(std::time::Duration::from_secs(3));
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

/// Extract GitHub repo names from markdown content
fn extract_github_repos(markdown: &str) -> Vec<String> {
    let mut repos = Vec::new();

    // Match patterns like:
    // - https://github.com/owner/repo
    // - [text](https://github.com/owner/repo)
    // - github.com/owner/repo

    let re_patterns = [
        r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
        r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
    ];

    for pattern in re_patterns {
        for cap in regex_lite::Regex::new(pattern)
            .unwrap()
            .captures_iter(markdown)
        {
            if let Some(m) = cap.get(1) {
                let repo = m.as_str();
                // Filter out non-repo paths (issues, pulls, blob, tree, etc.)
                if !repo.contains('/')
                    || repo.ends_with(".git")
                    || repo.split('/').nth(1).map_or(true, |s| {
                        ["issues", "pull", "blob", "tree", "wiki", "releases", "actions", "discussions"]
                            .contains(&s.split('/').next().unwrap_or(""))
                            || s.contains('#')
                            || s.contains('?')
                    })
                {
                    continue;
                }

                // Clean up: take only owner/repo part
                let parts: Vec<&str> = repo.split('/').take(2).collect();
                if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                    repos.push(format!("{}/{}", parts[0], parts[1]));
                }
            }
        }
    }

    repos
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
