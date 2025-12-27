//! Owner discovery and repository/follower fetching
//!
//! Handles cascading discovery through owner profiles and their followers.

use anyhow::Result;
use std::cell::RefCell;
use std::io::{IsTerminal, Write};

use crate::db::Database;
use crate::formatting::format_owner_link;
use crate::github::GitHubClient;

/// Result from discovering an owner's repos and followers
pub struct DiscoveryResult {
    pub repos_inserted: usize,
    pub followers_added: usize,
}

/// Discover and queue all repos from an owner (user/org) for later fetching
pub async fn discover_owner_repos(client: &GitHubClient, db: &Database, full_name: &str) -> Result<()> {
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
pub async fn discover_from_owners(
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
        let result = discover_single_owner(client, db, &owner).await?;
        total_repos += result.repos_inserted;
        total_profiles += result.followers_added;
        processed += 1;

        // Checkpoint WAL periodically to prevent unbounded growth
        if processed % 50 == 0 {
            let _ = db.checkpoint();
        }
    }

    // Final checkpoint before exit
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

/// Discover repos AND followers from a single owner with streaming saves
pub async fn discover_single_owner(
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

    Ok(DiscoveryResult { repos_inserted, followers_added })
}
