mod config;
mod db;
mod embedding;
mod github;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use config::Config;
use db::Database;
use embedding::{build_embedding_text, embed_text};
use github::GitHubClient;

#[derive(Parser)]
#[command(name = "goto-gh")]
#[command(about = "Semantic search for GitHub repositories")]
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
    },

    /// Add a specific repository by name
    Add {
        /// Repository full name (e.g., "qdrant/qdrant")
        repo: String,
    },

    /// Show index statistics
    Stats,

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
        Some(Commands::Index { query, count }) => {
            if token.is_none() {
                eprintln!("\x1b[33m..\x1b[0m No GitHub token found. Rate limit: 60 req/hour");
                eprintln!("  Set GITHUB_TOKEN or run: gh auth login");
            }
            index_repos(&client, &db, &query, count).await
        }
        Some(Commands::Add { repo }) => {
            add_repo(&client, &db, &repo).await
        }
        Some(Commands::Stats) => {
            show_stats(&db)
        }
        Some(Commands::RateLimit) => {
            check_rate_limit(&client).await
        }
        Some(Commands::Revectorize) => {
            revectorize(&db)
        }
        None => {
            eprintln!("\x1b[33mUsage:\x1b[0m goto-gh <query> or goto-gh --help");
            eprintln!("\nTo get started:");
            eprintln!("  1. goto-gh index              # Index top repos by stars");
            eprintln!("  2. goto-gh \"vector database\"  # Semantic search");
            std::process::exit(1);
        }
    }
}

/// Semantic search in local index
fn search(query: &str, limit: usize, db: &Database) -> Result<()> {
    let (total, indexed) = db.stats()?;

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
        let desc_truncated = if desc.len() > 60 {
            format!("{}...", &desc[..57])
        } else {
            desc.to_string()
        };

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
async fn index_repos(client: &GitHubClient, db: &Database, query: &Option<String>, count: u32) -> Result<()> {
    eprintln!("\x1b[36m..\x1b[0m Fetching top {} repos (GraphQL)", count);

    let search_query = match query {
        Some(q) => q.clone(),
        None => "stars:>20".to_string(), // Top repos by stars
    };

    // Star ranges for pagination (GraphQL has same 1000 limit per query)
    let star_ranges = [
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

    let mut indexed = 0;
    let mut skipped = 0;
    let mut total_fetched = 0;

    // Use query if provided, otherwise iterate through star ranges
    let ranges: Vec<&str> = if query.is_some() {
        vec![&search_query]
    } else {
        star_ranges.iter().map(|s| *s).collect()
    };

    'outer: for range in ranges {
        if total_fetched >= count as usize {
            break;
        }

        let mut cursor: Option<String> = None;

        loop {
            if total_fetched >= count as usize {
                break 'outer;
            }

            let batch_size = 100.min(count - total_fetched as u32);
            let (repos, next_cursor, has_next) = match client
                .search_repos_graphql(range, batch_size, cursor)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  {} - error: {}", range, e);
                    break;
                }
            };

            if repos.is_empty() {
                break;
            }

            eprintln!("  {} ({} repos)", range, repos.len());

            for repo in repos {
                if repo.fork {
                    continue;
                }

                total_fetched += 1;

                // Skip if already fresh
                if db.is_fresh(&repo.full_name, 7)? {
                    skipped += 1;
                    continue;
                }

                // Build embedding text (README already included!)
                let text = build_embedding_text(
                    &repo.full_name,
                    repo.description.as_deref(),
                    &repo.topics,
                    repo.language.as_deref(),
                    repo.readme.as_deref(),
                );

                // Generate embedding
                let embedding = embed_text(&text)?;

                // Store in database
                let repo_id = db.upsert_repo_with_readme(&repo, repo.readme.as_deref(), &text)?;
                db.upsert_embedding(repo_id, &embedding)?;

                indexed += 1;

                // Progress
                if indexed % 100 == 0 {
                    eprintln!("  ... indexed {}", indexed);
                }
            }

            if !has_next {
                break;
            }

            cursor = next_cursor;
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    eprintln!(
        "\x1b[32mok\x1b[0m Indexed {} new repos ({} skipped as fresh)",
        indexed, skipped
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
