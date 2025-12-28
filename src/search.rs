//! Hybrid search implementation with RRF (Reciprocal Rank Fusion)
//!
//! Combines semantic (vector), name matching, and keyword search for optimal results.

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::db::Database;
use crate::embedding::embed_query;
use crate::openai::OPENAI_EMBEDDING_DIM;
use crate::formatting::{format_repo_link, format_stars, truncate_str, Dots};

/// Reciprocal Rank Fusion (RRF) weights for hybrid search
/// Lower k = higher weight for top ranks
pub mod weights {
    /// Strong boost for repos with query term in name
    pub const K_NAME: f32 = 20.0;
    /// Standard weight for semantic/vector similarity
    pub const K_VECTOR: f32 = 60.0;
    /// Weaker weight for keyword matches (often noisy)
    pub const K_KEYWORD: f32 = 80.0;
}

/// Calculate RRF score contribution for a single rank
/// Lower k = higher weight for top ranks
pub fn rrf_score(rank: usize, k: f32) -> f32 {
    1.0 / (k + rank as f32 + 1.0)
}

/// Compute combined RRF scores from multiple ranked lists
/// Returns sorted (id, score) pairs in descending score order
pub fn compute_rrf_scores(
    name_results: &[i64],
    vector_results: &[(i64, f32)],
    keyword_results: &[i64],
) -> Vec<(i64, f32)> {
    use weights::{K_NAME, K_VECTOR, K_KEYWORD};

    let mut scores: HashMap<i64, f32> = HashMap::new();

    // Add name match scores (strongest signal)
    for (rank, repo_id) in name_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += rrf_score(rank, K_NAME);
    }

    // Add vector search scores
    for (rank, (repo_id, _distance)) in vector_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += rrf_score(rank, K_VECTOR);
    }

    // Add keyword search scores (weakest signal)
    for (rank, repo_id) in keyword_results.iter().enumerate() {
        *scores.entry(*repo_id).or_default() += rrf_score(rank, K_KEYWORD);
    }

    // Sort by combined RRF score descending
    let mut combined: Vec<_> = scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined
}

/// Expand a search query using Claude CLI for better semantic understanding
pub fn expand_query(query: &str) -> Result<String> {
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
                eprintln!("\x1b[32m✓\x1b[0m Expanded: {}", expanded);
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
pub fn search(query: &str, limit: usize, semantic_only: bool, db: &Database) -> Result<()> {
    let (_total, indexed) = db.stats()?;

    if indexed == 0 {
        eprintln!("\x1b[31mx\x1b[0m No repositories indexed yet.");
        eprintln!("  Run: goto-gh index \"<query>\" to index some repos first.");
        std::process::exit(1);
    }

    let mode = if semantic_only { "semantic" } else { "hybrid" };
    let dots = Dots::start(&format!("Searching {} repos ({})", indexed, mode));

    // 1. Semantic search via embeddings
    // Detect provider from DB embedding dimension
    let current_dim = db.get_embedding_dimension()?;
    let query_embedding = match current_dim {
        Some(dim) if dim == OPENAI_EMBEDDING_DIM => {
            // OpenAI embeddings - no prefix needed, but we need to call OpenAI API
            // For now, fall back to returning empty if OpenAI was used
            // (sync search can't easily call async OpenAI API)
            eprintln!("\x1b[33m!\x1b[0m OpenAI embeddings detected - semantic search requires OPENAI_API_KEY");
            if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                // Use blocking reqwest for sync context
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
                embedding
            } else {
                anyhow::bail!("OPENAI_API_KEY not set - required for searching OpenAI embeddings");
            }
        }
        _ => {
            // E5 embeddings - use query prefix
            embed_query(query)?
        }
    };
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
    let combined = compute_rrf_scores(&name_results, &vector_results, &keyword_results);

    if combined.is_empty() {
        eprintln!("\x1b[31mx\x1b[0m No matching repositories found.");
        return Ok(());
    }

    // Display results
    // Max score depends on mode
    use weights::{K_NAME, K_VECTOR, K_KEYWORD};
    let max_score = if semantic_only {
        1.0 / (K_VECTOR + 1.0)  // Only vector contributes
    } else {
        1.0 / (K_NAME + 1.0) + 1.0 / (K_VECTOR + 1.0) + 1.0 / (K_KEYWORD + 1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_score_decreases_with_rank() {
        // Higher rank (worse position) should have lower score
        let score_rank_0 = rrf_score(0, 60.0);
        let score_rank_1 = rrf_score(1, 60.0);
        let score_rank_10 = rrf_score(10, 60.0);

        assert!(score_rank_0 > score_rank_1);
        assert!(score_rank_1 > score_rank_10);
    }

    #[test]
    fn test_rrf_score_lower_k_means_higher_weight() {
        // Lower k should give higher scores (more weight to top ranks)
        let score_k20 = rrf_score(0, 20.0);   // K_NAME
        let score_k60 = rrf_score(0, 60.0);   // K_VECTOR
        let score_k80 = rrf_score(0, 80.0);   // K_KEYWORD

        assert!(score_k20 > score_k60);
        assert!(score_k60 > score_k80);
    }

    #[test]
    fn test_rrf_score_formula() {
        // Verify the formula: 1 / (k + rank + 1)
        assert_eq!(rrf_score(0, 60.0), 1.0 / 61.0);
        assert_eq!(rrf_score(1, 60.0), 1.0 / 62.0);
        assert_eq!(rrf_score(0, 20.0), 1.0 / 21.0);
    }

    #[test]
    fn test_compute_rrf_scores_single_source() {
        // Only name results
        let name_results = vec![1, 2, 3];
        let vector_results: Vec<(i64, f32)> = vec![];
        let keyword_results: Vec<i64> = vec![];

        let scores = compute_rrf_scores(&name_results, &vector_results, &keyword_results);

        assert_eq!(scores.len(), 3);
        // First result should have highest score
        assert_eq!(scores[0].0, 1);
        assert!(scores[0].1 > scores[1].1);
        assert!(scores[1].1 > scores[2].1);
    }

    #[test]
    fn test_compute_rrf_scores_combines_sources() {
        // Repo 1 appears in all three sources at rank 0 - should be highest
        // Repo 2 appears only in name at rank 1
        // Repo 3 appears only in vector at rank 0
        let name_results = vec![1, 2];
        let vector_results = vec![(1, 0.5), (3, 0.8)];
        let keyword_results = vec![1];

        let scores = compute_rrf_scores(&name_results, &vector_results, &keyword_results);

        // Repo 1 should be first (appears in all 3 sources)
        assert_eq!(scores[0].0, 1);

        // Expected score for repo 1:
        // name rank 0: 1/(20+0+1) = 1/21
        // vector rank 0: 1/(60+0+1) = 1/61
        // keyword rank 0: 1/(80+0+1) = 1/81
        let expected_1 = 1.0/21.0 + 1.0/61.0 + 1.0/81.0;
        assert!((scores[0].1 - expected_1).abs() < 0.0001);
    }

    #[test]
    fn test_compute_rrf_scores_name_match_dominates() {
        // Repo 1: only in name at rank 0 (strong signal)
        // Repo 2: in vector rank 0 AND keyword rank 0 (two weak signals)
        let name_results = vec![1];
        let vector_results = vec![(2, 0.5)];
        let keyword_results = vec![2];

        let scores = compute_rrf_scores(&name_results, &vector_results, &keyword_results);

        // Name match at rank 0: 1/21 ≈ 0.0476
        // Vector rank 0 + keyword rank 0: 1/61 + 1/81 ≈ 0.0164 + 0.0123 = 0.0287
        // So repo 1 should still win due to strong name signal
        assert_eq!(scores[0].0, 1);
    }

    #[test]
    fn test_compute_rrf_scores_empty_inputs() {
        let scores = compute_rrf_scores(&[], &[], &[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_compute_rrf_scores_sorted_descending() {
        let name_results = vec![3, 2, 1];  // 3 is first, 1 is last
        let scores = compute_rrf_scores(&name_results, &[], &[]);

        // Should be sorted by score descending (3 has highest score as rank 0)
        assert_eq!(scores[0].0, 3);
        assert_eq!(scores[1].0, 2);
        assert_eq!(scores[2].0, 1);
    }
}
