//! Find interesting developer/org profiles using semantic embeddings.
//!
//! Instead of heuristics, we:
//! 1. Define "interesting" via seed queries (compilers, LLMs, databases, etc.)
//! 2. Find repos semantically close to these seeds
//! 3. Aggregate by owner - profiles with many "interesting" repos rank higher

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::db::{Database, RepoDetails};
use crate::embedding::embed_query;
use crate::openai::OPENAI_EMBEDDING_DIM;

/// Default seed queries for "interesting" technical work.
/// These are broad enough to find interesting profiles but specific enough
/// to avoid generic repos. Each seed search takes ~1 min on large datasets.
const DEFAULT_SEEDS: &[&str] = &[
    "compiler parser AST codegen LLVM",
    "database query engine storage B-tree",
    "LLM transformer inference attention GPU",
];

/// Configuration for profile discovery
#[derive(Debug, Clone)]
pub struct ProfilesConfig {
    /// Custom seed queries (uses defaults if empty)
    pub seeds: Vec<String>,
    /// Repos to fetch per seed query
    pub per_seed: usize,
    /// Minimum repos an owner must have to qualify
    pub min_repos: usize,
    /// Number of profiles to return
    pub limit: usize,
    /// Expand seeds with LLM
    pub expand: bool,
}

impl Default for ProfilesConfig {
    fn default() -> Self {
        Self {
            seeds: vec![],
            per_seed: 100,
            min_repos: 2,
            limit: 20,
            expand: false,
        }
    }
}

/// A scored profile result
#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub owner: String,
    /// Number of "interesting" repos found
    pub interesting_count: usize,
    /// Total stars across interesting repos
    pub total_stars: u64,
    /// Average similarity to seed queries
    pub avg_similarity: f32,
    /// Combined score
    pub score: f64,
    /// The interesting repos with their matched seeds
    pub repos: Vec<InterestingRepo>,
}

/// A repo matched by seed query
#[derive(Debug, Clone)]
pub struct InterestingRepo {
    pub details: RepoDetails,
    pub similarity: f32,
    pub matched_seed: String,
}

/// Embed a query using the appropriate provider (OpenAI or local)
fn embed_query_auto(db: &Database, query: &str) -> Result<Vec<f32>> {
    let current_dim = db.get_embedding_dimension()?;

    match current_dim {
        Some(dim) if dim == OPENAI_EMBEDDING_DIM => {
            // OpenAI embeddings - call API
            let api_key = std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY not set - required for profiles with OpenAI embeddings")?;

            let client = reqwest::blocking::Client::new();
            let response = client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": "text-embedding-3-small",
                    "input": [query]
                }))
                .send()?;

            if !response.status().is_success() {
                anyhow::bail!("OpenAI API error: {}", response.status());
            }

            let result: serde_json::Value = response.json()?;
            let embedding: Vec<f32> = result["data"][0]["embedding"]
                .as_array()
                .context("Invalid OpenAI response")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            Ok(embedding)
        }
        _ => {
            // Local E5 embeddings
            embed_query(query)
        }
    }
}

/// Find interesting profiles using embedding similarity
pub fn find_interesting_profiles(
    db: &Database,
    config: &ProfilesConfig,
) -> Result<Vec<ProfileResult>> {
    let seeds: Vec<&str> = if config.seeds.is_empty() {
        DEFAULT_SEEDS.to_vec()
    } else {
        config.seeds.iter().map(|s| s.as_str()).collect()
    };

    eprintln!(
        "\x1b[36m..\x1b[0m Searching with {} seed queries, {} repos each",
        seeds.len(),
        config.per_seed
    );

    // Collect repos matched by each seed: owner -> [(repo_id, similarity, seed)]
    let mut owner_repos: HashMap<String, Vec<(i64, f32, String)>> = HashMap::new();

    for (i, seed) in seeds.iter().enumerate() {
        eprintln!(
            "\x1b[36m..\x1b[0m [{}/{}] {}",
            i + 1, seeds.len(), &seed[..seed.len().min(40)]
        );
        // Embed the seed query (auto-detects OpenAI vs local)
        let embedding = embed_query_auto(db, seed)?;

        // Find similar repos
        let similar = db.find_similar(&embedding, config.per_seed)?;

        for (repo_id, distance) in similar {
            // Get repo details to find owner
            if let Some(details) = db.get_repo_details(repo_id)? {
                // Extract owner from full_name (e.g., "owner/repo")
                let owner = details.full_name.split('/').next().unwrap_or("").to_lowercase();
                if owner.is_empty() {
                    continue;
                }
                let similarity = (-distance).exp(); // Convert L2 distance to similarity
                owner_repos
                    .entry(owner)
                    .or_default()
                    .push((repo_id, similarity, seed.to_string()));
            }
        }
    }

    eprintln!(
        "\x1b[36m..\x1b[0m Found {} unique owners",
        owner_repos.len()
    );

    // Score and filter owners
    let mut results: Vec<ProfileResult> = Vec::new();

    for (owner, repo_matches) in owner_repos {
        // Dedupe repos (same repo may match multiple seeds - keep best similarity)
        let mut best_per_repo: HashMap<i64, (f32, String)> = HashMap::new();
        for (repo_id, sim, seed) in repo_matches {
            best_per_repo
                .entry(repo_id)
                .and_modify(|(s, sd)| {
                    if sim > *s {
                        *s = sim;
                        *sd = seed.clone();
                    }
                })
                .or_insert((sim, seed));
        }

        // Filter by minimum repos
        if best_per_repo.len() < config.min_repos {
            continue;
        }

        // Build result
        let mut interesting_repos: Vec<InterestingRepo> = Vec::new();
        let mut total_stars = 0u64;
        let mut total_sim = 0f32;

        for (repo_id, (similarity, matched_seed)) in &best_per_repo {
            if let Some(details) = db.get_repo_details(*repo_id)? {
                total_stars += details.stars;
                total_sim += similarity;
                interesting_repos.push(InterestingRepo {
                    details,
                    similarity: *similarity,
                    matched_seed: matched_seed.clone(),
                });
            }
        }

        // Sort repos by similarity
        interesting_repos.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        let count = interesting_repos.len();
        let avg_sim = total_sim / count as f32;

        // Score: count * avg_similarity * log(stars + 1)
        let score = (count as f64) * (avg_sim as f64) * (total_stars as f64 + 1.0).ln();

        results.push(ProfileResult {
            owner,
            interesting_count: count,
            total_stars,
            avg_similarity: avg_sim,
            score,
            repos: interesting_repos,
        });
    }

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Take top N
    results.truncate(config.limit);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_seeds() {
        assert!(!DEFAULT_SEEDS.is_empty());
        assert!(DEFAULT_SEEDS.len() <= 5); // Keep default fast (~1min per seed)
        assert!(DEFAULT_SEEDS.iter().any(|s| s.contains("compiler")));
    }

    #[test]
    fn test_config_default() {
        let config = ProfilesConfig::default();
        assert_eq!(config.per_seed, 100);
        assert_eq!(config.min_repos, 2);
        assert_eq!(config.limit, 20);
    }
}
