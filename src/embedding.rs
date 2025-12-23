use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Mutex, OnceLock};

use crate::config::Config;

/// Vector dimension for MultilingualE5Small model
pub const EMBEDDING_DIM: usize = 384;

/// Global embedding model instance (lazy-loaded)
static MODEL: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

/// Initialize the embedding model
fn init_model() -> Result<TextEmbedding> {
    let cache_dir = Config::model_cache_dir()?;
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Failed to create cache directory: {}", cache_dir.display()))?;

    TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(true),
    )
    .context("Failed to initialize embedding model")
}

/// Generate embedding for a single text
pub fn embed_text(text: &str) -> Result<Vec<f32>> {
    let model_mutex = MODEL.get_or_init(|| {
        Mutex::new(init_model().expect("Failed to initialize embedding model"))
    });

    let mut model = model_mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock embedding model"))?;

    let embeddings = model
        .embed(vec![text], None)
        .context("Failed to generate embedding")?;

    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
}

/// Generate embeddings for multiple texts (batch processing)
pub fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let model_mutex = MODEL.get_or_init(|| {
        Mutex::new(init_model().expect("Failed to initialize embedding model"))
    });

    let mut model = model_mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock embedding model"))?;

    model
        .embed(texts.to_vec(), None)
        .context("Failed to generate embeddings")
}

/// Maximum README excerpt length
const README_MAX_CHARS: usize = 1500;

/// Build embedding text from repo metadata
pub fn build_embedding_text(
    full_name: &str,
    description: Option<&str>,
    topics: &[String],
    language: Option<&str>,
    readme: Option<&str>,
) -> String {
    let mut parts = vec![full_name.to_string()];

    if let Some(desc) = description {
        parts.push(desc.to_string());
    }

    if !topics.is_empty() {
        parts.push(topics.join(", "));
    }

    if let Some(lang) = language {
        parts.push(format!("Language: {}", lang));
    }

    if let Some(readme) = readme {
        let excerpt = extract_readme_excerpt(readme);
        if !excerpt.is_empty() {
            parts.push(excerpt);
        }
    }

    parts.join(" | ")
}

/// Extract meaningful content from README
fn extract_readme_excerpt(content: &str) -> String {
    let mut result = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip empty or short lines
        if trimmed.len() < 10 {
            continue;
        }

        // Skip non-content lines
        if trimmed.starts_with('#')
            || trimmed.starts_with('[')
            || trimmed.starts_with('!')
            || trimmed.starts_with("```")
            || trimmed.starts_with("<!--")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("- ")
            || trimmed.contains("shields.io")
            || trimmed.contains("badge")
        {
            continue;
        }

        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(trimmed);

        if result.len() >= README_MAX_CHARS {
            break;
        }
    }

    // Truncate safely
    if result.len() > README_MAX_CHARS {
        let mut end = README_MAX_CHARS;
        while !result.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        result.truncate(end);
        if let Some(last_space) = result.rfind(' ') {
            result.truncate(last_space);
        }
        result.push_str("...");
    }

    result
}
