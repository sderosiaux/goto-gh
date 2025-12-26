mod config;
mod db;
mod embedding;
mod github;
mod proxy;

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
use proxy::ProxyManager;

/// Format repo name as clickable hyperlink (only if stdout is a TTY)
fn format_repo_link(name: &str, url: &str) -> String {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        // OSC 8 hyperlink: \x1b]8;;URL\x1b\\TEXT\x1b]8;;\x1b\\
        format!("\x1b]8;;{}\x1b\\\x1b[1m{}\x1b[0m\x1b]8;;\x1b\\", url, name)
    } else {
        name.to_string()
    }
}

/// Format owner name as clickable hyperlink (checks stderr since discover outputs there)
fn format_owner_link(name: &str, url: &str) -> String {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        format!("\x1b]8;;{}\x1b\\\x1b[1m{}\x1b[0m\x1b]8;;\x1b\\", url, name)
    } else {
        name.to_string()
    }
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

    /// Re-generate embeddings from stored data (no API calls)
    #[command(hide = true)]
    Revectorize,

    /// Fetch metadata from GitHub for repos in DB that don't have it yet
    Fetch {
        /// Number of repos to fetch (default: all pending)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Batch size for GraphQL queries (default: 300, = 10 API chunks of 30)
        #[arg(short, long, default_value = "300")]
        batch_size: usize,

        /// Number of parallel GraphQL requests (default: 4)
        #[arg(short = 'j', long, default_value = "4")]
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
        Some(Commands::Revectorize) => {
            revectorize(&db)
        }
        Some(Commands::Fetch { limit, batch_size, concurrency, debug }) => {
            let client = GitHubClient::new_with_options(token.clone(), debug, None, false);
            fetch_from_db(&client, &db, limit, batch_size, concurrency).await
        }
        Some(Commands::Embed { batch_size, limit, delay }) => {
            embed_missing(&db, batch_size, limit, delay)
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
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            eprintln!();
            std::process::exit(0);
        }
    }
}

/// Expand a search query using Claude CLI for better semantic understanding
fn expand_query(query: &str) -> Result<String> {
    use std::process::Command;

    let prompt = format!(
        r#"Expand this GitHub search query into technical keywords.

Query: "{}"

Output ONLY comma-separated keywords (no intro, no explanation, no quotes). Include:
- Original terms
- Related tools/libraries (e.g., metasploit, nmap, burpsuite)
- Technical concepts (e.g., buffer overflow, RCE, SSRF)
- File types/languages when relevant

Example input: "web security"
Example output: web security, OWASP, XSS, SQL injection, CSRF, burpsuite, ZAP, web application firewall, penetration testing

Keywords:"#,
        query
    );

    eprintln!("\x1b[36m..\x1b[0m Expanding query with Claude...");

    // Try to find claude in common locations
    let claude_paths = [
        "claude",  // In PATH
        "/usr/local/bin/claude",
        &format!("{}/.nvm/versions/node/v22.20.0/bin/claude", std::env::var("HOME").unwrap_or_default()),
    ];

    let mut output = None;
    for claude_path in &claude_paths {
        let result = Command::new(claude_path)
            .args(["--dangerously-skip-permissions", "--model", "haiku", "-p", &prompt])
            .output();
        if result.is_ok() {
            output = Some(result);
            break;
        }
    }

    let output = match output {
        Some(o) => o,
        None => return Ok(query.to_string()),  // Fallback silently
    };

    match output {
        Ok(out) if out.status.success() => {
            let expanded = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if expanded.is_empty() {
                eprintln!("\x1b[33m!\x1b[0m Claude returned empty, using original query");
                Ok(query.to_string())
            } else {
                eprintln!("\x1b[32m✓\x1b[0m Expanded: {}", truncate_str(&expanded, 80));
                Ok(expanded)
            }
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr);
            eprintln!("\x1b[33m!\x1b[0m Claude failed ({}), using original query", err.trim());
            Ok(query.to_string())
        }
        Err(e) => {
            eprintln!("\x1b[33m!\x1b[0m Claude not available ({}), using original query", e);
            Ok(query.to_string())
        }
    }
}

/// Hybrid search (semantic + keyword + name match with RRF fusion)
fn search(query: &str, limit: usize, semantic_only: bool, db: &Database) -> Result<()> {
    let (_total, indexed) = db.stats()?;

    if indexed == 0 {
        eprintln!("\x1b[31mx\x1b[0m No repositories indexed yet.");
        eprintln!("  Run: goto-gh index \"<query>\" to index some repos first.");
        std::process::exit(1);
    }

    let mode = if semantic_only { "semantic" } else { "hybrid" };
    let dots = Dots::start(&format!("Searching {} repos ({})", indexed, mode));

    // 1. Semantic search via embeddings
    let query_embedding = embed_text(query)?;
    let vector_results = db.find_similar(&query_embedding, limit * 3)?;

    // 2. Name match search (strongest signal - repos with query in name)
    let name_results = if semantic_only {
        vec![]
    } else {
        db.find_by_name_match(query, limit * 3)?
    };

    // 3. Content keyword search via LIKE on embedded_text
    let keyword_results = if semantic_only {
        vec![]
    } else {
        db.find_by_keywords(query, limit * 3)?
    };

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
    // Max score depends on mode
    let max_score = if semantic_only {
        1.0 / (k_vector + 1.0)  // Only vector contributes
    } else {
        1.0 / (k_name + 1.0) + 1.0 / (k_vector + 1.0) + 1.0 / (k_keyword + 1.0)
    };

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
        let repo_link = format_repo_link(&repo.full_name, &repo.url);

        println!(
            "\x1b[35m{:>2}.\x1b[0m {} \x1b[33m{}\x1b[0m \x1b[90m[{}]\x1b[0m \x1b[90m({:.0}%)\x1b[0m \x1b[90m{}\x1b[0m",
            i + 1,
            repo_link,
            stars,
            lang,
            display_score,
            desc_truncated
        );
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

    println!("\x1b[32mok\x1b[0m Added \x1b[1m{}\x1b[0m", repo.full_name);
    if let Some(desc) = &repo.description {
        println!("  {}", desc);
    }

    // Discover other repos from the same owner (if not already explored)
    discover_owner_repos(client, db, full_name).await?;

    Ok(())
}

/// Discover and queue all repos from an owner (user/org) for later fetching
async fn discover_owner_repos(client: &GitHubClient, db: &Database, full_name: &str) -> Result<()> {
    let owner = full_name.split('/').next().unwrap_or("");
    if owner.is_empty() {
        return Ok(());
    }

    // Skip if already explored
    if db.is_owner_repos_fetched(owner)? {
        return Ok(());
    }

    eprintln!("\x1b[36m..\x1b[0m Discovering repos from {}", owner);

    match client.list_owner_repos(owner).await {
        Ok(repos) => {
            let count = repos.len();
            let (inserted, _skipped) = db.add_repo_stubs_bulk(&repos)?;
            db.mark_owner_explored(owner, count)?;

            if inserted > 0 {
                eprintln!(
                    "  \x1b[90m+{} new repos queued from {} ({} total)\x1b[0m",
                    inserted, owner, count
                );
            }
        }
        Err(e) => {
            // Don't fail the add if discovery fails, just log it
            eprintln!("  \x1b[33m⚠\x1b[0m Could not discover repos from {}: {}", owner, e);
        }
    }

    Ok(())
}

/// Discover more repos by exploring owners of existing repos (also fetches followers)
async fn discover_from_owners(
    client: &GitHubClient,
    db: &Database,
    limit: Option<usize>,
    _concurrency: usize,
) -> Result<()> {
    // First, check for any interrupted owners and resume them
    let in_progress_repos = db.get_in_progress_owners()?;
    let in_progress_followers = db.get_in_progress_followers_fetch()?;

    if !in_progress_repos.is_empty() || !in_progress_followers.is_empty() {
        let total = in_progress_repos.len() + in_progress_followers.len();
        eprintln!("\x1b[33m..\x1b[0m Resuming {} interrupted owner(s)", total);

        let mut all_in_progress: Vec<String> = in_progress_repos;
        all_in_progress.extend(in_progress_followers);
        all_in_progress.sort();
        all_in_progress.dedup();

        for owner in &all_in_progress {
            discover_single_owner(client, db, owner).await?;
        }
    }

    // Get owners that need either repos OR followers fetched
    let unexplored = db.count_unexplored_owners()?;
    let without_followers = db.count_owners_without_followers()?;

    if unexplored == 0 && without_followers == 0 {
        println!("\x1b[32mok\x1b[0m All owners fully explored (repos + followers)");
        return Ok(());
    }

    // Get owners: prioritize unexplored, then those missing followers
    let mut owners_to_process: Vec<String> = db.get_unexplored_owners(limit)?;

    // If limit allows, add owners that have repos but no followers
    if let Some(lim) = limit {
        if owners_to_process.len() < lim {
            let remaining = lim - owners_to_process.len();
            let more = db.get_owners_without_followers(Some(remaining))?;
            owners_to_process.extend(more);
        }
    } else {
        // No limit - add all owners without followers
        let more = db.get_owners_without_followers(None)?;
        owners_to_process.extend(more);
    }

    // Dedupe (some might be in both lists)
    owners_to_process.sort();
    owners_to_process.dedup();

    let to_process = owners_to_process.len();
    eprintln!(
        "\x1b[36m..\x1b[0m Discovering from {} owners ({} need repos, {} need followers)",
        to_process, unexplored, without_followers
    );

    let mut total_repos = 0;
    let mut total_profiles = 0;
    let mut processed = 0;

    for owner in owners_to_process {
        let (repos, profiles) = discover_single_owner(client, db, &owner).await?;
        total_repos += repos;
        total_profiles += profiles;
        processed += 1;
    }

    println!(
        "\x1b[32mok\x1b[0m Processed {} owners: +{} repos, +{} profiles",
        processed, total_repos, total_profiles
    );

    let still_unexplored = db.count_unexplored_owners()?;
    let still_without_followers = db.count_owners_without_followers()?;

    if still_unexplored > 0 || still_without_followers > 0 {
        eprintln!(
            "\x1b[36m..\x1b[0m {} still need repos, {} need followers",
            still_unexplored, still_without_followers
        );
    }

    Ok(())
}

/// Discover repos AND followers from a single owner with streaming saves
async fn discover_single_owner(
    client: &GitHubClient,
    db: &Database,
    owner: &str,
) -> Result<(usize, usize)> {
    use std::cell::RefCell;
    use std::io::IsTerminal;

    let owner_url = format!("https://github.com/{}", owner);
    let owner_link = format_owner_link(owner, &owner_url);
    let is_tty = std::io::stderr().is_terminal();

    let needs_repos = !db.is_owner_repos_fetched(owner)?;
    let needs_followers = !db.is_owner_followers_fetched(owner)?;

    let mut repos_inserted = 0usize;
    let mut followers_added = 0usize;

    // === REPOS ===
    if needs_repos {
        db.mark_owner_in_progress(owner)?;

        let inserted_count = RefCell::new(0usize);
        let total_repos = RefCell::new(0usize);
        let last_page = RefCell::new(0usize);

        let result = client
            .list_owner_repos_streaming(owner, |repos, progress| {
                let (inserted, _) = db.add_repo_stubs_bulk(&repos)?;
                *inserted_count.borrow_mut() += inserted;
                *total_repos.borrow_mut() = progress.total_so_far;
                *last_page.borrow_mut() = progress.page;

                if is_tty && progress.page > 1 {
                    eprint!(
                        "\r  {} \x1b[90mrepos: {} (+{} new)...\x1b[0m\x1b[K",
                        owner_link, progress.total_so_far, *inserted_count.borrow()
                    );
                    let _ = std::io::stderr().flush();
                }
                Ok(())
            })
            .await;

        if is_tty && *last_page.borrow() >= 1 {
            eprint!("\r\x1b[K");
        }

        repos_inserted = *inserted_count.borrow();
        let total = *total_repos.borrow();

        match result {
            Ok(_) => db.mark_owner_explored(owner, total)?,
            Err(_) => db.mark_owner_explored(owner, total)?,
        }
    }

    // === FOLLOWERS ===
    if needs_followers {
        db.mark_owner_followers_in_progress(owner)?;

        let added_count = RefCell::new(0usize);
        let total_followers = RefCell::new(0usize);
        let last_page = RefCell::new(0usize);

        let result = client
            .list_owner_followers_streaming(owner, |followers, progress| {
                let (added, _) = db.add_followers_as_owners_bulk(&followers)?;
                *added_count.borrow_mut() += added;
                *total_followers.borrow_mut() = progress.total_so_far;
                *last_page.borrow_mut() = progress.page;

                if is_tty {
                    eprint!(
                        "\r  {} \x1b[90mfollowers: {} (+{} new)...\x1b[0m\x1b[K",
                        owner_link, progress.total_so_far, *added_count.borrow()
                    );
                    let _ = std::io::stderr().flush();
                }
                Ok(())
            })
            .await;

        if is_tty && *last_page.borrow() >= 1 {
            eprint!("\r\x1b[K");
        }

        followers_added = *added_count.borrow();
        let total = *total_followers.borrow();

        match result {
            Ok(_) => db.mark_owner_followers_fetched(owner, total)?,
            Err(_) => db.mark_owner_followers_fetched(owner, total)?,
        }
    }

    // Print summary
    let mut parts = Vec::new();
    if repos_inserted > 0 {
        parts.push(format!("+{} repos", repos_inserted));
    }
    if followers_added > 0 {
        parts.push(format!("+{} profiles", followers_added));
    }

    if !parts.is_empty() {
        eprintln!("  {} \x1b[32m{}\x1b[0m", owner_link, parts.join(", "));
    } else if needs_repos || needs_followers {
        eprintln!("  {} \x1b[90m(all known)\x1b[0m", owner_link);
    }

    Ok((repos_inserted, followers_added))
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

    println!("\x1b[36mIndex Statistics\x1b[0m\n");
    println!("  \x1b[90mTotal repos:\x1b[0m        {}", total);
    println!("  \x1b[90mWith metadata:\x1b[0m      {}", total - without_metadata - gone);
    println!("  \x1b[90mWith embeddings:\x1b[0m    {}", indexed);
    println!();
    println!("  \x1b[90mNeed metadata:\x1b[0m      {}", without_metadata);
    println!("  \x1b[90mNeed embeddings:\x1b[0m    {}", without_embeddings);
    if gone > 0 {
        println!("  \x1b[90mGone (deleted):\x1b[0m     {}", gone);
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
    eprintln!("    Run: goto-gh fetch --batch-size 300");

    Ok(())
}

/// Load usernames from file as owners to discover (creates placeholder repos)
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
    limit: Option<usize>,
    batch_size: usize,
    concurrency: usize,
) -> Result<()> {
    let need_fetch = db.count_repos_without_metadata()?;

    if need_fetch == 0 {
        eprintln!("\x1b[32mok\x1b[0m All repos already have metadata");
        return Ok(());
    }

    let to_fetch = limit.map(|l| l.min(need_fetch)).unwrap_or(need_fetch);
    eprintln!(
        "\x1b[36m..\x1b[0m Fetching metadata for {} repos (batch size: {}, concurrency: {}) - NO EMBEDDINGS",
        to_fetch, batch_size, concurrency
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
        let fetched_repos = match client.fetch_repos_batch(&repos_to_fetch, concurrency).await {
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
    eprintln!("    Run: goto-gh embed --batch-size 200 --delay 5");

    Ok(())
}

/// Generate embeddings for repos that don't have them yet
fn embed_missing(db: &Database, batch_size: usize, limit: Option<usize>, delay_secs: u64) -> Result<()> {
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

