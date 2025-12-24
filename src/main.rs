mod config;
mod db;
mod embedding;
mod github;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

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

/// Format star count (e.g., 1.2k, 15k)
fn format_stars(stars: u64) -> String {
    if stars >= 1000 {
        format!("{}k", stars / 1000)
    } else {
        format!("{}", stars)
    }
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
            eprintln!("  ... marked {} repos as gone (deleted/renamed)", gone_count);
        }

        eprintln!("  ... stored {} repos (total: {})", fetched_count, stored);

        if total_discovered > 0 {
            eprintln!("  ... discovered {} new repos from READMEs", total_discovered);
        }

        // Small delay between batches to avoid rate limits
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

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
