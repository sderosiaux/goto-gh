//! Find underrated gems - repos semantically similar to popular ones but with fewer stars.
//!
//! This helps discover hidden gems that might be overlooked despite being
//! high-quality alternatives or related projects to popular repos.

use anyhow::{bail, Result};

use crate::db::{Database, RepoDetails};
use super::{find_similar_with_details, resolve_repo};

/// Configuration for finding underrated gems
#[derive(Debug, Clone)]
pub struct UnderratedConfig {
    /// Minimum similarity threshold (0-1, higher = more similar)
    pub min_similarity: f32,
    /// Maximum stars for a repo to be considered "underrated"
    pub max_stars: u64,
    /// Specific repo to find alternatives for (if None, sample popular repos)
    pub reference: Option<String>,
    /// Number of popular repos to sample (only used if reference is None)
    pub sample_popular: usize,
    /// Minimum stars for reference repos (only used if reference is None)
    pub min_reference_stars: u64,
    /// Number of gems to find per reference
    pub limit_per_reference: usize,
}

impl Default for UnderratedConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.75,
            max_stars: 500,
            reference: None,
            sample_popular: 50,
            min_reference_stars: 10000,
            limit_per_reference: 5,
        }
    }
}

/// Result for a single reference repo
#[derive(Debug, Clone)]
pub struct UnderratedResult {
    /// The popular reference repo
    pub reference: RepoDetails,
    /// Underrated gems similar to this reference
    pub gems: Vec<GemResult>,
}

/// A single underrated gem
#[derive(Debug, Clone)]
pub struct GemResult {
    /// The underrated repo
    pub repo: RepoDetails,
    /// Similarity to reference (0-1)
    pub similarity: f32,
    /// Distance in embedding space
    pub distance: f32,
    /// "Underrated score" = similarity / log(stars + 1)
    pub underrated_score: f32,
}

/// Find underrated gems similar to popular repos
pub fn find_underrated(db: &Database, config: &UnderratedConfig) -> Result<Vec<UnderratedResult>> {
    let references = get_reference_repos(db, config)?;

    if references.is_empty() {
        bail!("No reference repos found");
    }

    let mut results = Vec::with_capacity(references.len());

    for (ref_id, ref_details) in references {
        let embedding = match db.get_embedding(ref_id)? {
            Some(e) => e,
            None => continue,
        };

        // Extract base name to filter out forks (e.g., "react" from "facebook/react")
        let ref_base_name = ref_details.full_name
            .split('/')
            .nth(1)
            .unwrap_or("")
            .to_lowercase();

        // Find similar repos (get more than needed to filter by stars and forks)
        let search_limit = config.limit_per_reference * 50; // increased to account for fork filtering
        let candidates = find_similar_with_details(db, &embedding, search_limit, &[ref_id])?;

        // Filter and score
        let mut gems: Vec<GemResult> = candidates
            .into_iter()
            .filter(|c| c.repo.stars <= config.max_stars)
            .filter(|c| c.similarity() >= config.min_similarity)
            // Exclude forks: repos with the exact same base name are likely forks
            // e.g., for "react": exclude "user/react" but keep "user/react-native"
            .filter(|c| {
                let candidate_base = c.repo.full_name
                    .split('/')
                    .nth(1)
                    .unwrap_or("")
                    .to_lowercase();
                candidate_base != ref_base_name
            })
            .map(|c| {
                let similarity = c.similarity();
                let underrated_score = similarity / (c.repo.stars as f32 + 1.0).ln();
                GemResult {
                    repo: c.repo,
                    similarity,
                    distance: c.distance,
                    underrated_score,
                }
            })
            .collect();

        // Sort by underrated score (higher = more underrated given similarity)
        gems.sort_by(|a, b| b.underrated_score.partial_cmp(&a.underrated_score).unwrap());
        gems.truncate(config.limit_per_reference);

        if !gems.is_empty() {
            results.push(UnderratedResult {
                reference: ref_details,
                gems,
            });
        }
    }

    // Sort results by number of gems found (most productive references first)
    results.sort_by(|a, b| b.gems.len().cmp(&a.gems.len()));

    Ok(results)
}

/// Get reference repos based on config
fn get_reference_repos(db: &Database, config: &UnderratedConfig) -> Result<Vec<(i64, RepoDetails)>> {
    if let Some(ref name) = config.reference {
        // Single specified reference
        match resolve_repo(db, name)? {
            Some((id, _)) => {
                let details = db.get_repo_details(id)?
                    .ok_or_else(|| anyhow::anyhow!("Failed to get details for reference repo"))?;
                Ok(vec![(id, details)])
            }
            None => bail!("Reference repo '{}' not found or has no embedding", name),
        }
    } else {
        // Sample popular repos
        let repos = db.get_repos_by_star_range(
            config.min_reference_stars as i64,
            None,
            config.sample_popular,
        )?;

        let mut results = Vec::with_capacity(repos.len());
        for (id, _, _) in repos {
            if let Some(details) = db.get_repo_details(id)? {
                results.push((id, details));
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_underrated_config_default() {
        let config = UnderratedConfig::default();
        assert_eq!(config.min_similarity, 0.75);
        assert_eq!(config.max_stars, 500);
        assert!(config.reference.is_none());
    }
}
