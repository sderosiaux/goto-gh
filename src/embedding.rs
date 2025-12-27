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

/// Generic section headers that appear in most READMEs and don't add semantic value
const GENERIC_HEADERS: &[&str] = &[
    "installation", "install", "installing",
    "usage", "use", "using", "how to use",
    "features", "feature",
    "api", "api reference", "api documentation",
    "documentation", "docs",
    "getting started", "quick start", "quickstart",
    "requirements", "prerequisites", "dependencies",
    "configuration", "config", "setup",
    "contributing", "contribute", "contributors",
    "license", "licensing",
    "changelog", "changes", "history",
    "credits", "acknowledgments", "acknowledgements",
    "authors", "author", "maintainers",
    "support", "help", "faq",
    "examples", "example",
    "demo", "demos",
    "roadmap",
    "table of contents", "toc", "contents",
    "about", "overview", "introduction", "intro",
    "motivation", "why",
    "testing", "tests", "test",
    "building", "build", "compilation", "compile",
    "deployment", "deploy",
    "troubleshooting", "issues", "known issues",
    "security",
    "performance",
    "disclaimer",
];

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

/// Check if a header line contains only generic section names
fn is_generic_header(line: &str) -> bool {
    // Strip markdown header prefix (# ## ### etc.)
    let text = line.trim_start_matches('#').trim().to_lowercase();

    // Check against list of generic headers
    GENERIC_HEADERS.iter().any(|&h| text == h)
}

/// Extract meaningful content from README
fn extract_readme_excerpt(content: &str) -> String {
    let mut result = String::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Track code block state
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        // Skip everything inside code blocks
        if in_code_block {
            continue;
        }

        // Skip empty or short lines
        if trimmed.len() < 10 {
            continue;
        }

        // Skip non-content lines (images, links, comments, badges)
        if trimmed.starts_with('[')
            || trimmed.starts_with('!')
            || trimmed.starts_with("<!--")
            || trimmed.contains("shields.io")
            || trimmed.contains("badge")
        {
            continue;
        }

        // Skip generic headers that don't add semantic value
        if trimmed.starts_with('#') && is_generic_header(trimmed) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_embedding_text_minimal() {
        let text = build_embedding_text("owner/repo", None, &[], None, None);
        assert_eq!(text, "owner/repo");
    }

    #[test]
    fn test_build_embedding_text_with_description() {
        let text = build_embedding_text(
            "owner/repo",
            Some("A great project"),
            &[],
            None,
            None,
        );
        assert_eq!(text, "owner/repo | A great project");
    }

    #[test]
    fn test_build_embedding_text_with_topics() {
        let topics = vec!["rust".to_string(), "cli".to_string(), "tool".to_string()];
        let text = build_embedding_text("owner/repo", None, &topics, None, None);
        assert_eq!(text, "owner/repo | rust, cli, tool");
    }

    #[test]
    fn test_build_embedding_text_with_language() {
        let text = build_embedding_text("owner/repo", None, &[], Some("Rust"), None);
        assert_eq!(text, "owner/repo | Language: Rust");
    }

    #[test]
    fn test_build_embedding_text_full() {
        let topics = vec!["machine-learning".to_string()];
        let text = build_embedding_text(
            "owner/ml-project",
            Some("Machine learning toolkit"),
            &topics,
            Some("Python"),
            Some("This is a comprehensive ML toolkit for data scientists."),
        );
        assert!(text.contains("owner/ml-project"));
        assert!(text.contains("Machine learning toolkit"));
        assert!(text.contains("machine-learning"));
        assert!(text.contains("Language: Python"));
        assert!(text.contains("comprehensive ML toolkit"));
    }

    #[test]
    fn test_extract_readme_excerpt_filters_generic_headers() {
        // Generic headers like "Installation", "Usage", "Features" don't add value
        // But specific/unique headers should be kept
        let readme = r#"
# My Awesome ML Framework
## Installation
## Features
## Neural Network Architecture
## Usage
This framework provides state-of-the-art models.
"#;
        let excerpt = extract_readme_excerpt(readme);
        // Unique/specific headers should be included
        assert!(excerpt.contains("My Awesome ML Framework"));
        assert!(excerpt.contains("Neural Network Architecture"));
        // Generic headers should be excluded
        assert!(!excerpt.contains("# Installation"));
        assert!(!excerpt.contains("## Installation"));
        assert!(!excerpt.contains("# Features"));
        assert!(!excerpt.contains("## Features"));
        assert!(!excerpt.contains("# Usage"));
        assert!(!excerpt.contains("## Usage"));
        // Content should still be included
        assert!(excerpt.contains("state-of-the-art"));
    }

    #[test]
    fn test_is_generic_header() {
        // Generic headers
        assert!(is_generic_header("# Installation"));
        assert!(is_generic_header("## Features"));
        assert!(is_generic_header("### Usage"));
        assert!(is_generic_header("# API Reference"));
        assert!(is_generic_header("## Getting Started"));
        assert!(is_generic_header("# License"));
        assert!(is_generic_header("## Contributing"));

        // Non-generic (specific) headers - should be kept
        assert!(!is_generic_header("# My Project Name"));
        assert!(!is_generic_header("## Neural Network Architecture"));
        assert!(!is_generic_header("# How We Built a Distributed System"));
        assert!(!is_generic_header("## The Algorithm Explained"));
    }

    #[test]
    fn test_extract_readme_excerpt_skips_badges() {
        let readme = r#"
[![Build](https://shields.io/badge/build-passing)](url)
![badge](https://img.shields.io/badge)
This is the real description of the project.
"#;
        let excerpt = extract_readme_excerpt(readme);
        assert!(!excerpt.contains("shields.io"));
        assert!(!excerpt.contains("badge"));
        assert!(excerpt.contains("real description"));
    }

    #[test]
    fn test_extract_readme_excerpt_skips_code_blocks() {
        let readme = r#"
```rust
fn main() {
    println!("This is code that should be ignored");
    let very_long_line_of_code_that_exceeds_ten_chars = 42;
}
```
This explains what the code does in plain English.
"#;
        let excerpt = extract_readme_excerpt(readme);
        // Should skip the entire code block content
        assert!(!excerpt.contains("```"));
        assert!(!excerpt.contains("fn main"));
        assert!(!excerpt.contains("println"));
        assert!(!excerpt.contains("very_long_line"));
        // But should include content after the code block
        assert!(excerpt.contains("explains what the code does"));
    }

    #[test]
    fn test_extract_readme_excerpt_handles_multiple_code_blocks() {
        let readme = r#"
First paragraph of meaningful content here.

```python
def hello():
    print("Hello world this is python code")
```

Middle paragraph between code blocks.

```javascript
function greet() {
    console.log("This is JavaScript code here");
}
```

Final paragraph after all code blocks.
"#;
        let excerpt = extract_readme_excerpt(readme);
        // Should include prose paragraphs
        assert!(excerpt.contains("First paragraph"));
        assert!(excerpt.contains("Middle paragraph"));
        assert!(excerpt.contains("Final paragraph"));
        // Should exclude all code
        assert!(!excerpt.contains("def hello"));
        assert!(!excerpt.contains("function greet"));
        assert!(!excerpt.contains("print"));
        assert!(!excerpt.contains("console.log"));
    }

    #[test]
    fn test_extract_readme_excerpt_skips_short_lines() {
        let readme = "Hi\nOK\nThis is a longer line with meaningful content.";
        let excerpt = extract_readme_excerpt(readme);
        assert!(!excerpt.contains("Hi"));
        assert!(!excerpt.contains("OK"));
        assert!(excerpt.contains("longer line"));
    }

    #[test]
    fn test_extract_readme_excerpt_includes_list_items() {
        // Bullet lists often contain valuable feature descriptions
        let readme = r#"
* Feature one: supports multiple formats
- Feature two: blazingly fast performance
This paragraph explains things in detail.
"#;
        let excerpt = extract_readme_excerpt(readme);
        // List items with useful content should be included
        assert!(excerpt.contains("supports multiple formats"));
        assert!(excerpt.contains("blazingly fast performance"));
        assert!(excerpt.contains("paragraph explains"));
    }

    #[test]
    fn test_extract_readme_excerpt_truncates_long_content() {
        // Create a very long README
        let long_content = "This is a meaningful sentence. ".repeat(200);
        let excerpt = extract_readme_excerpt(&long_content);

        assert!(excerpt.len() <= README_MAX_CHARS + 3); // +3 for "..."
        assert!(excerpt.ends_with("..."));
    }

    #[test]
    fn test_extract_readme_excerpt_handles_empty() {
        assert_eq!(extract_readme_excerpt(""), "");
        assert_eq!(extract_readme_excerpt("   \n\n  "), "");
    }
}
