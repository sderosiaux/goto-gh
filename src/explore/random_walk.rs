//! Random walk through the embedding space.
//!
//! Starting from a repo, explore neighbors by semantic similarity,
//! randomly selecting the next hop to discover unexpected connections.

use anyhow::{bail, Result};
use rand::Rng;
use std::collections::HashSet;

use crate::db::{Database, RepoDetails};
use super::{find_similar_with_details, random_start, resolve_repo, ExploreResult};

/// Configuration for random walk
#[derive(Debug, Clone)]
pub struct WalkConfig {
    /// Number of steps to take
    pub steps: usize,
    /// Number of candidates to consider at each step
    pub breadth: usize,
    /// Start from a random repo instead of specified one
    pub random_start: bool,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            steps: 5,
            breadth: 10,
            random_start: false,
        }
    }
}

/// A single step in the walk
#[derive(Debug, Clone)]
pub struct WalkStep {
    /// The repo at this step
    pub repo: RepoDetails,
    /// Distance from previous step (0 for starting point)
    pub distance: f32,
    /// Step number (0 = start)
    pub step_num: usize,
    /// Number of candidates considered at this step
    pub candidates_considered: usize,
}

/// Result of a random walk
#[derive(Debug)]
pub struct WalkResult {
    /// All steps including the starting point
    pub steps: Vec<WalkStep>,
    /// Total distance traveled
    pub total_distance: f32,
}

/// Extract base name from full_name (e.g., "react" from "facebook/react")
fn get_base_name(full_name: &str) -> String {
    full_name
        .split('/')
        .nth(1)
        .unwrap_or("")
        .to_lowercase()
}

/// Check if a candidate is likely a fork of any visited repo
/// Only considers exact name matches (e.g., "user/react" is a fork of "facebook/react")
fn is_likely_fork(candidate_name: &str, visited_bases: &HashSet<String>) -> bool {
    let candidate_base = get_base_name(candidate_name);
    if candidate_base.is_empty() {
        return false;
    }
    visited_bases.contains(&candidate_base)
}

/// Perform a random walk through the embedding space
pub fn random_walk(db: &Database, start: &str, config: &WalkConfig) -> Result<WalkResult> {
    let mut visited: HashSet<i64> = HashSet::new();
    let mut visited_base_names: HashSet<String> = HashSet::new();
    let mut steps = Vec::with_capacity(config.steps + 1);
    let mut total_distance = 0.0;

    // Get starting point
    let (start_id, _start_name, mut current_embedding) = if config.random_start || start.is_empty() {
        match random_start(db)? {
            Some((id, name, emb)) => (id, name, emb),
            None => bail!("No repos with embeddings found"),
        }
    } else {
        match resolve_repo(db, start)? {
            Some((id, emb)) => (id, start.to_string(), emb),
            None => bail!("Repo '{}' not found or has no embedding", start),
        }
    };

    // Add starting point
    let start_details = db.get_repo_details(start_id)?
        .ok_or_else(|| anyhow::anyhow!("Failed to get details for starting repo"))?;

    visited.insert(start_id);
    visited_base_names.insert(get_base_name(&start_details.full_name));
    steps.push(WalkStep {
        repo: start_details,
        distance: 0.0,
        step_num: 0,
        candidates_considered: 0,
    });

    let mut rng = rand::thread_rng();

    // Walk through the space
    for step in 1..=config.steps {
        let exclude_ids: Vec<i64> = visited.iter().copied().collect();

        // Find candidates (fetch more to filter out forks)
        let raw_candidates = find_similar_with_details(
            db,
            &current_embedding,
            config.breadth * 5,
            &exclude_ids,
        )?;

        // Filter out forks of already visited repos
        let candidates: Vec<_> = raw_candidates
            .into_iter()
            .filter(|c| !is_likely_fork(&c.repo.full_name, &visited_base_names))
            .take(config.breadth)
            .collect();

        if candidates.is_empty() {
            eprintln!("\x1b[33m!\x1b[0m Walk ended early at step {} - no more candidates", step);
            break;
        }

        // Select next step using weighted random (prefer closer repos)
        let selected = weighted_random_select(&candidates, &mut rng);

        // Update state
        visited.insert(selected.repo.id);
        visited_base_names.insert(get_base_name(&selected.repo.full_name));
        total_distance += selected.distance;

        // Get embedding for next step
        current_embedding = match db.get_embedding(selected.repo.id)? {
            Some(e) => e,
            None => {
                eprintln!("\x1b[33m!\x1b[0m Walk ended early - selected repo has no embedding");
                break;
            }
        };

        steps.push(WalkStep {
            repo: selected.repo.clone(),
            distance: selected.distance,
            step_num: step,
            candidates_considered: candidates.len(),
        });
    }

    Ok(WalkResult {
        steps,
        total_distance,
    })
}

/// Select a candidate using weighted random selection
/// Weights are inverse of distance (closer = more likely)
fn weighted_random_select<'a, R: Rng>(candidates: &'a [ExploreResult], rng: &mut R) -> &'a ExploreResult {
    if candidates.len() == 1 {
        return &candidates[0];
    }

    // Calculate weights (inverse distance with small epsilon to avoid division by zero)
    let weights: Vec<f64> = candidates
        .iter()
        .map(|c| 1.0 / (c.distance as f64 + 0.001))
        .collect();

    let total_weight: f64 = weights.iter().sum();

    // Random selection
    let mut random_point = rng.gen::<f64>() * total_weight;

    for (i, weight) in weights.iter().enumerate() {
        random_point -= weight;
        if random_point <= 0.0 {
            return &candidates[i];
        }
    }

    // Fallback to last (should not happen)
    candidates.last().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_config_default() {
        let config = WalkConfig::default();
        assert_eq!(config.steps, 5);
        assert_eq!(config.breadth, 10);
        assert!(!config.random_start);
    }
}
