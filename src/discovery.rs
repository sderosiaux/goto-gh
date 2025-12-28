//! Owner discovery and repository/follower fetching
//!
//! Provides core discovery logic used by both CLI and server modes.

use anyhow::Result;
use std::cell::RefCell;
use std::io::{IsTerminal, Write};

use crate::db::Database;
use crate::formatting::format_owner_link;
use crate::github::GitHubClient;

/// Result from discovering an owner's repos and followers
#[derive(Debug, Default, Clone)]
pub struct DiscoveryResult {
    pub repos_inserted: usize,
    pub followers_added: usize,
}

/// Result from a complete discovery run
#[derive(Debug, Default)]
pub struct DiscoveryRunResult {
    pub owners_processed: usize,
    pub total_repos: usize,
    pub total_profiles: usize,
}

/// Configuration for discovery
#[derive(Clone)]
pub struct DiscoveryConfig {
    pub limit: Option<usize>,
    pub batch_size: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            limit: None,
            batch_size: 50,
        }
    }
}

/// Progress information during discovery
#[derive(Debug, Clone)]
pub struct DiscoveryProgress {
    pub owner: String,
    pub repos_inserted: usize,
    pub followers_added: usize,
    pub owners_processed: usize,
    pub total_repos: usize,
    pub total_profiles: usize,
}

// ============================================================================
// Core functions (no TTY output, no RefCell - suitable for server mode)
// ============================================================================

/// Discover repos from an owner (core version, no progress output)
///
/// Returns Ok(count) on success, Err on API error (owner stays in_progress for retry)
pub async fn discover_owner_repos_core(
    client: &GitHubClient,
    db: &Database,
    owner: &str,
) -> Result<usize> {
    if db.is_owner_repos_fetched(owner)? {
        return Ok(0);
    }

    db.mark_owner_in_progress(owner)?;

    let repos = client.list_owner_repos(owner).await;

    match repos {
        Ok(repos) => {
            let count = repos.len();
            let (inserted, _) = db.add_repo_stubs_bulk(&repos)?;
            db.mark_owner_explored(owner, count)?;
            Ok(inserted)
        }
        Err(e) => {
            // Check if it's a "not found" error (owner doesn't exist)
            let err_str = e.to_string().to_lowercase();
            if err_str.contains("not found") || err_str.contains("404") {
                // Owner doesn't exist, mark as explored with 0 repos
                db.mark_owner_explored(owner, 0)?;
                Ok(0)
            } else {
                // Other error (rate limit, network, etc.) - don't mark, will retry
                eprintln!("\x1b[33m[discover]\x1b[0m {} repos error: {}", owner, e);
                Err(e)
            }
        }
    }
}

/// Discover followers from an owner (core version, no progress output)
///
/// Returns Ok(count) on success, Err on API error (owner stays in_progress for retry)
pub async fn discover_owner_followers_core(
    client: &GitHubClient,
    db: &Database,
    owner: &str,
) -> Result<usize> {
    if db.is_owner_followers_fetched(owner)? {
        return Ok(0);
    }

    db.mark_owner_followers_in_progress(owner)?;

    // Collect all followers first (no callback to avoid RefCell issues)
    let mut all_followers: Vec<String> = Vec::new();
    let result = client
        .list_owner_followers_streaming(owner, |followers, _progress| {
            all_followers.extend(followers);
            Ok(())
        })
        .await;

    match result {
        Ok(count) => {
            let (added, _) = db.add_followers_as_owners_bulk(&all_followers)?;
            db.mark_owner_followers_fetched(owner, count)?;
            Ok(added)
        }
        Err(e) => {
            // Check if it's a "not found" error (owner doesn't exist)
            let err_str = e.to_string().to_lowercase();
            if err_str.contains("not found") || err_str.contains("404") {
                // Owner doesn't exist, mark as fetched with 0 followers
                db.mark_owner_followers_fetched(owner, 0)?;
                Ok(0)
            } else {
                // Other error (rate limit, network, etc.) - don't mark, will retry
                eprintln!("\x1b[33m[discover]\x1b[0m {} followers error: {}", owner, e);
                Err(e)
            }
        }
    }
}

/// Discover a single owner (repos + followers) - core version
pub async fn discover_single_owner_core(
    client: &GitHubClient,
    db: &Database,
    owner: &str,
) -> Result<DiscoveryResult> {
    let repos_inserted = discover_owner_repos_core(client, db, owner).await?;
    let followers_added = discover_owner_followers_core(client, db, owner).await?;

    Ok(DiscoveryResult {
        repos_inserted,
        followers_added,
    })
}

/// Get the next batch of owners to process
pub fn get_owners_batch(db: &Database, batch_size: usize) -> Result<Vec<String>> {
    let mut owners: Vec<String> = db.get_unexplored_owners(Some(batch_size))?;

    if owners.len() < batch_size {
        let remaining = batch_size - owners.len();
        let more = db.get_owners_without_followers(Some(remaining))?;
        owners.extend(more);
    }

    owners.sort();
    owners.dedup();
    Ok(owners)
}

/// Run discovery loop until no more work or limit reached
///
/// Core version suitable for server mode.
pub async fn run_discover_loop<F, S>(
    client: &GitHubClient,
    db: &Database,
    config: &DiscoveryConfig,
    mut on_progress: F,
    should_stop: S,
) -> Result<DiscoveryRunResult>
where
    F: FnMut(&DiscoveryProgress),
    S: Fn() -> bool,
{
    let mut result = DiscoveryRunResult::default();
    let limit = config.limit.unwrap_or(usize::MAX);

    loop {
        if should_stop() {
            break;
        }

        if result.owners_processed >= limit {
            break;
        }

        let remaining = limit - result.owners_processed;
        let batch_size = config.batch_size.min(remaining);

        let query_start = std::time::Instant::now();
        let owners = get_owners_batch(db, batch_size)?;
        let query_ms = query_start.elapsed().as_millis();

        if owners.is_empty() {
            break;
        }

        // Log slow queries (>100ms) - helps identify performance issues
        if query_ms > 100 {
            eprintln!(
                "\x1b[35m[discover]\x1b[0m \x1b[90mfound {} owners in {}ms\x1b[0m",
                owners.len(), query_ms
            );
        }

        for owner in owners {
            if should_stop() {
                break;
            }

            if result.owners_processed >= limit {
                break;
            }

            let owner_result = discover_single_owner_core(client, db, &owner).await?;

            result.owners_processed += 1;
            result.total_repos += owner_result.repos_inserted;
            result.total_profiles += owner_result.followers_added;

            let progress = DiscoveryProgress {
                owner: owner.clone(),
                repos_inserted: owner_result.repos_inserted,
                followers_added: owner_result.followers_added,
                owners_processed: result.owners_processed,
                total_repos: result.total_repos,
                total_profiles: result.total_profiles,
            };

            on_progress(&progress);

            // Checkpoint periodically
            if result.owners_processed % 10 == 0 {
                let _ = db.checkpoint();
            }
        }
    }

    let _ = db.checkpoint();
    Ok(result)
}

// ============================================================================
// CLI functions (with TTY output and RefCell for streaming progress)
// ============================================================================

/// Discover and queue all repos from an owner (CLI version with output)
pub async fn discover_owner_repos(
    client: &GitHubClient,
    db: &Database,
    full_name: &str,
) -> Result<()> {
    let owner = full_name.split('/').next().unwrap_or("");
    if owner.is_empty() {
        return Ok(());
    }

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
            eprintln!(
                "  \x1b[33m⚠\x1b[0m Could not discover repos from {}: {}",
                owner, e
            );
        }
    }

    Ok(())
}

/// Discover more repos by exploring owners (CLI version with TTY progress)
pub async fn discover_from_owners(
    client: &GitHubClient,
    db: &Database,
    limit: Option<usize>,
    _concurrency: usize,
) -> Result<()> {
    // Resume interrupted owners
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
            discover_single_owner_cli(client, db, owner).await?;
        }
    }

    let unexplored = db.count_unexplored_owners()?;
    let without_followers = db.count_owners_without_followers()?;

    if unexplored == 0 && without_followers == 0 {
        println!("\x1b[32mok\x1b[0m All owners fully explored (repos + followers)");
        return Ok(());
    }

    let mut owners_to_process: Vec<String> = db.get_unexplored_owners(limit)?;

    if let Some(lim) = limit {
        if owners_to_process.len() < lim {
            let remaining = lim - owners_to_process.len();
            let more = db.get_owners_without_followers(Some(remaining))?;
            owners_to_process.extend(more);
        }
    } else {
        let more = db.get_owners_without_followers(None)?;
        owners_to_process.extend(more);
    }

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
        let result = discover_single_owner_cli(client, db, &owner).await?;
        total_repos += result.repos_inserted;
        total_profiles += result.followers_added;
        processed += 1;

        if processed % 50 == 0 {
            let _ = db.checkpoint();
        }
    }

    let _ = db.checkpoint();

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

/// Discover repos AND followers from a single owner (CLI version with streaming)
pub async fn discover_single_owner_cli(
    client: &GitHubClient,
    db: &Database,
    owner: &str,
) -> Result<DiscoveryResult> {
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
                        owner_link,
                        progress.total_so_far,
                        *inserted_count.borrow()
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
                        owner_link,
                        progress.total_so_far,
                        *added_count.borrow()
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

    Ok(DiscoveryResult {
        repos_inserted,
        followers_added,
    })
}
