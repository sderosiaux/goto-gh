//! Cross-pollination detection - find repos at the intersection of two domains.
//!
//! This helps discover innovative projects that combine ideas from different
//! fields, like "machine learning + music" or "rust + web assembly".

use anyhow::{Context, Result};
use std::collections::HashSet;

use crate::db::{Database, RepoDetails};
use crate::embedding::embed_query;
use crate::openai::OPENAI_EMBEDDING_DIM;

/// Configuration for cross-pollination search
#[derive(Debug, Clone)]
pub struct CrossConfig {
    /// First topic/domain
    pub topic1: String,
    /// Second topic/domain
    pub topic2: String,
    /// Minimum similarity to each topic (0-1)
    pub min_each: f32,
    /// Number of results to return
    pub limit: usize,
}

impl Default for CrossConfig {
    fn default() -> Self {
        Self {
            topic1: String::new(),
            topic2: String::new(),
            min_each: 0.5,
            limit: 20,
        }
    }
}

/// A repo at the intersection of two topics
#[derive(Debug, Clone)]
pub struct CrossResult {
    /// The repo
    pub repo: RepoDetails,
    /// Similarity to first topic
    pub sim_topic1: f32,
    /// Similarity to second topic
    pub sim_topic2: f32,
    /// Cross-pollination score (higher = better balance between both topics)
    pub cross_score: f32,
    /// Distance to topic1 embedding
    pub dist_topic1: f32,
    /// Distance to topic2 embedding
    pub dist_topic2: f32,
}

/// Find repos at the intersection of two topics
pub fn find_cross_pollination(db: &Database, config: &CrossConfig) -> Result<Vec<CrossResult>> {
    // Embed both topics as queries using the correct provider
    let (emb1, emb2) = embed_topics(db, &config.topic1, &config.topic2)?;

    // Search limit - fetch many candidates to find overlaps
    // sqlite-vec has a limit of 4096 for k parameter
    let search_limit = 4096;

    // Find repos similar to topic1
    let results1 = db.find_similar(&emb1, search_limit)?;
    let results2 = db.find_similar(&emb2, search_limit)?;

    // Build lookup for topic2 distances
    let dist2_map: std::collections::HashMap<i64, f32> = results2.into_iter().collect();

    // Find repos that appear in both searches
    let mut candidates: Vec<CrossResult> = Vec::new();

    for (repo_id, dist1) in results1 {
        if let Some(&dist2) = dist2_map.get(&repo_id) {
            let sim1 = distance_to_similarity(dist1);
            let sim2 = distance_to_similarity(dist2);

            // Check minimum threshold
            if sim1 < config.min_each || sim2 < config.min_each {
                continue;
            }

            // Calculate cross-pollination score
            // Use harmonic mean to reward balance between both topics
            // Multiply by min to ensure both are relevant
            let harmonic_mean = 2.0 * sim1 * sim2 / (sim1 + sim2);
            let cross_score = harmonic_mean * sim1.min(sim2);

            if let Some(repo) = db.get_repo_details(repo_id)? {
                candidates.push(CrossResult {
                    repo,
                    sim_topic1: sim1,
                    sim_topic2: sim2,
                    cross_score,
                    dist_topic1: dist1,
                    dist_topic2: dist2,
                });
            }
        }
    }

    // Sort by cross score (higher = better)
    candidates.sort_by(|a, b| b.cross_score.partial_cmp(&a.cross_score).unwrap());

    // Deduplicate by base name (keep first = highest score)
    // This filters out forks that would otherwise clutter results
    let mut seen_bases: HashSet<String> = HashSet::new();
    candidates.retain(|c| {
        let base = get_base_name(&c.repo.full_name);
        if base.is_empty() {
            return true;
        }
        seen_bases.insert(base)
    });

    candidates.truncate(config.limit);

    Ok(candidates)
}

/// Extract base name from full_name (e.g., "react" from "facebook/react")
fn get_base_name(full_name: &str) -> String {
    full_name
        .split('/')
        .nth(1)
        .unwrap_or("")
        .to_lowercase()
}

/// Convert L2 distance to similarity (0-1)
fn distance_to_similarity(distance: f32) -> f32 {
    (-distance).exp()
}

/// Embed two topics using the correct provider based on DB dimension
fn embed_topics(db: &Database, topic1: &str, topic2: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let current_dim = db.get_embedding_dimension()?;

    match current_dim {
        Some(dim) if dim == OPENAI_EMBEDDING_DIM => {
            // OpenAI embeddings - call API
            let api_key = std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY not set - required for cross-pollination with OpenAI embeddings")?;

            let client = reqwest::blocking::Client::new();

            // Embed both topics in a single batch
            let response = client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "model": "text-embedding-3-small",
                    "input": [topic1, topic2]
                }))
                .send()?;

            if !response.status().is_success() {
                anyhow::bail!("OpenAI API error: {}", response.status());
            }

            let result: serde_json::Value = response.json()?;
            let data = result["data"].as_array()
                .context("Invalid OpenAI response")?;

            let emb1: Vec<f32> = data[0]["embedding"]
                .as_array()
                .context("Invalid embedding for topic1")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            let emb2: Vec<f32> = data[1]["embedding"]
                .as_array()
                .context("Invalid embedding for topic2")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            Ok((emb1, emb2))
        }
        _ => {
            // Local E5 embeddings
            let emb1 = embed_query(topic1)?;
            let emb2 = embed_query(topic2)?;
            Ok((emb1, emb2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_config_default() {
        let config = CrossConfig::default();
        assert_eq!(config.min_each, 0.5);
        assert_eq!(config.limit, 20);
    }

    #[test]
    fn test_distance_to_similarity() {
        // Distance 0 = similarity 1
        assert!((distance_to_similarity(0.0) - 1.0).abs() < 0.001);
        // Higher distance = lower similarity
        assert!(distance_to_similarity(1.0) < distance_to_similarity(0.5));
    }
}
