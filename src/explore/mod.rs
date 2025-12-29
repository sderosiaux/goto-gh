//! Exploration features for discovering repos through semantic similarity.
//!
//! This module provides three main discovery mechanisms:
//! - **Random Walk**: Traverse the embedding space by hopping between similar repos
//! - **Underrated Gems**: Find repos semantically similar to popular ones but with fewer stars
//! - **Cross-Pollination**: Discover repos at the intersection of two different domains

mod random_walk;
mod underrated;
mod cross_pollinate;

pub use random_walk::{random_walk, WalkConfig};
pub use underrated::{find_underrated, UnderratedConfig};
pub use cross_pollinate::{find_cross_pollination, CrossConfig};

use crate::db::{Database, RepoDetails};
use anyhow::Result;

/// Common result type for exploration features
#[derive(Debug, Clone)]
pub struct ExploreResult {
    pub repo: RepoDetails,
    pub distance: f32,
}

impl ExploreResult {
    /// Create from repo details and distance
    pub fn new(repo: RepoDetails, distance: f32) -> Self {
        Self { repo, distance }
    }

    /// Convert distance to similarity score (0-1, higher = more similar)
    pub fn similarity(&self) -> f32 {
        // sqlite-vec uses L2 distance, convert to similarity
        // Using exponential decay: sim = exp(-distance)
        (-self.distance).exp()
    }
}

/// Resolve a repo name to its ID and embedding
pub fn resolve_repo(db: &Database, full_name: &str) -> Result<Option<(i64, Vec<f32>)>> {
    let repo_id = match db.get_repo_id_by_name(full_name)? {
        Some(id) => id,
        None => return Ok(None),
    };

    let embedding = match db.get_embedding(repo_id)? {
        Some(e) => e,
        None => return Ok(None),
    };

    Ok(Some((repo_id, embedding)))
}

/// Get a random repo with embedding as starting point
pub fn random_start(db: &Database) -> Result<Option<(i64, String, Vec<f32>)>> {
    let (repo_id, full_name) = match db.random_embedded_repo()? {
        Some(r) => r,
        None => return Ok(None),
    };

    let embedding = match db.get_embedding(repo_id)? {
        Some(e) => e,
        None => return Ok(None),
    };

    Ok(Some((repo_id, full_name, embedding)))
}

/// Find similar repos and return full details
pub fn find_similar_with_details(
    db: &Database,
    embedding: &[f32],
    limit: usize,
    exclude_ids: &[i64],
) -> Result<Vec<ExploreResult>> {
    let similar = db.find_similar_excluding(embedding, limit, exclude_ids)?;

    let mut results = Vec::with_capacity(similar.len());
    for (repo_id, distance) in similar {
        if let Some(details) = db.get_repo_details(repo_id)? {
            results.push(ExploreResult::new(details, distance));
        }
    }

    Ok(results)
}
