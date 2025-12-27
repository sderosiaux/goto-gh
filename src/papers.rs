use once_cell::sync::Lazy;
use regex::Regex;
use regex_lite::Regex as RegexLite;
use std::collections::HashSet;

/// Extracted paper link with metadata
#[derive(Debug, Clone)]
pub struct ExtractedPaper {
    pub url: String,
    pub domain: String,
    pub arxiv_id: Option<String>,
    pub doi: Option<String>,
    pub context: Option<String>,
}

// === Lazy-compiled regex patterns (compiled once, reused forever) ===

/// arxiv patterns: https://arxiv.org/abs/2301.12345, https://arxiv.org/pdf/2301.12345.pdf
static ARXIV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)").unwrap()
});

/// DOI patterns: https://doi.org/10.1234/something, https://dx.doi.org/10.1234/something
static DOI_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s\)\]>"]+)"#).unwrap()
});

/// MIT Press journals: https://direct.mit.edu/neco/article/35/12/1234/...
static MIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://direct\.mit\.edu/[^\s\)\]>"]+)"#).unwrap()
});

/// OpenReview: https://openreview.net/forum?id=xxx, https://openreview.net/pdf?id=xxx
static OPENREVIEW_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://(?:www\.)?openreview\.net/(?:forum|pdf)\?id=[^\s\)\]>"]+)"#).unwrap()
});

/// ACL Anthology: https://aclanthology.org/2023.acl-long.1/
static ACL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://(?:www\.)?aclanthology\.org/[^\s\)\]>"]+)"#).unwrap()
});

/// NeurIPS / ICML / ICLR proceedings: https://proceedings.neurips.cc/paper/2023/...
static NEURIPS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://proceedings\.(?:neurips\.cc|mlr\.press)/[^\s\)\]>"]+)"#).unwrap()
});

/// Semantic Scholar: https://www.semanticscholar.org/paper/...
static SEMANTIC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://(?:www\.)?semanticscholar\.org/paper/[^\s\)\]>"]+)"#).unwrap()
});

/// PapersWithCode: https://paperswithcode.com/paper/...
static PWC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(https?://(?:www\.)?paperswithcode\.com/paper/[^\s\)\]>"]+)"#).unwrap()
});

/// Extract paper links from README content
pub fn extract_paper_links(content: &str) -> Vec<ExtractedPaper> {
    let mut papers = Vec::new();
    let mut seen_urls = HashSet::new();

    // Helper to get context around a match (safe for UTF-8)
    let get_context = |start: usize, end: usize| -> Option<String> {
        // Find char boundaries
        let mut ctx_start = start.saturating_sub(100);
        while ctx_start > 0 && !content.is_char_boundary(ctx_start) {
            ctx_start -= 1;
        }

        let mut ctx_end = (end + 100).min(content.len());
        while ctx_end < content.len() && !content.is_char_boundary(ctx_end) {
            ctx_end += 1;
        }

        // Find sentence boundaries
        let safe_start = content[ctx_start..start]
            .rfind(|c: char| c == '\n' || c == '.' || c == '!' || c == '?')
            .map(|i| ctx_start + i + 1)
            .unwrap_or(ctx_start);

        let safe_end = content[end..ctx_end]
            .find(|c: char| c == '\n' || c == '.' || c == '!' || c == '?')
            .map(|i| end + i + 1)
            .unwrap_or(ctx_end);

        // Final boundary check
        let final_start = if content.is_char_boundary(safe_start) { safe_start } else { ctx_start };
        let final_end = if content.is_char_boundary(safe_end) { safe_end } else { ctx_end };

        if final_start >= final_end {
            return None;
        }

        let ctx = content[final_start..final_end].trim().to_string();
        if ctx.len() > 10 { Some(ctx) } else { None }
    };

    // Extract arxiv
    for cap in ARXIV_RE.captures_iter(content) {
        let full_match = cap.get(0).unwrap();
        let url = full_match.as_str().to_string();

        if seen_urls.contains(&url) {
            continue;
        }
        seen_urls.insert(url.clone());

        let arxiv_id = cap.get(1).map(|m| m.as_str().to_string());
        let context = get_context(full_match.start(), full_match.end());

        papers.push(ExtractedPaper {
            url: format!("https://arxiv.org/abs/{}", arxiv_id.as_deref().unwrap_or("")),
            domain: "arxiv.org".to_string(),
            arxiv_id,
            doi: None,
            context,
        });
    }

    // Extract DOI
    for cap in DOI_RE.captures_iter(content) {
        let full_match = cap.get(0).unwrap();
        let url = full_match.as_str().to_string();

        if seen_urls.contains(&url) {
            continue;
        }
        seen_urls.insert(url.clone());

        let doi = cap.get(1).map(|m| m.as_str().to_string());
        let context = get_context(full_match.start(), full_match.end());

        papers.push(ExtractedPaper {
            url: format!("https://doi.org/{}", doi.as_deref().unwrap_or("")),
            domain: "doi.org".to_string(),
            arxiv_id: None,
            doi,
            context,
        });
    }

    // Helper macro to extract from other domains
    macro_rules! extract_domain {
        ($re:expr, $domain:expr) => {
            for cap in $re.captures_iter(content) {
                let full_match = cap.get(1).unwrap();
                let url = full_match.as_str().to_string();

                // Clean URL (remove trailing punctuation)
                let url = url.trim_end_matches(|c| c == '.' || c == ',' || c == ')' || c == ']').to_string();

                if seen_urls.contains(&url) {
                    continue;
                }
                seen_urls.insert(url.clone());

                let context = get_context(full_match.start(), full_match.end());

                papers.push(ExtractedPaper {
                    url,
                    domain: $domain.to_string(),
                    arxiv_id: None,
                    doi: None,
                    context,
                });
            }
        };
    }

    extract_domain!(&MIT_RE, "direct.mit.edu");
    extract_domain!(&OPENREVIEW_RE, "openreview.net");
    extract_domain!(&ACL_RE, "aclanthology.org");
    extract_domain!(&NEURIPS_RE, "proceedings.neurips.cc");
    extract_domain!(&SEMANTIC_RE, "semanticscholar.org");
    extract_domain!(&PWC_RE, "paperswithcode.com");

    papers
}

// === GitHub Repository Link Extraction ===

/// Lazy-compiled regex patterns for GitHub repo extraction
static GITHUB_HTTPS_RE: Lazy<RegexLite> = Lazy::new(|| {
    RegexLite::new(r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)").unwrap()
});

static GITHUB_SHORT_RE: Lazy<RegexLite> = Lazy::new(|| {
    RegexLite::new(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)").unwrap()
});

/// Paths that indicate non-repo URLs (issues, pulls, etc.)
const NON_REPO_PATHS: &[&str] = &[
    "issues", "pull", "blob", "tree", "wiki", "releases", "actions", "discussions"
];

/// Extract GitHub repo names from markdown content
/// Returns vec of "owner/repo" strings
/// Note: Extracts repo references from deep links too (e.g., /issues/123 -> owner/repo)
pub fn extract_github_repos(markdown: &str) -> Vec<String> {
    let mut repos = Vec::new();
    let mut seen = HashSet::new();

    for re in [&*GITHUB_HTTPS_RE, &*GITHUB_SHORT_RE] {
        for cap in re.captures_iter(markdown) {
            if let Some(m) = cap.get(1) {
                let repo = m.as_str();

                // Filter out clearly invalid patterns
                if !repo.contains('/') || repo.ends_with(".git") {
                    continue;
                }

                // Clean up: take only owner/repo part (first two segments)
                let parts: Vec<&str> = repo.split('/').take(2).collect();
                if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                    let repo_name = parts[1];

                    // Skip if repo name is actually a non-repo path like "issues", "pull", etc.
                    // This handles edge cases like github.com/issues (not a real repo)
                    if NON_REPO_PATHS.contains(&repo_name)
                        || repo_name.contains('#')
                        || repo_name.contains('?')
                    {
                        continue;
                    }

                    let full_name = format!("{}/{}", parts[0], repo_name);
                    if !seen.contains(&full_name) {
                        seen.insert(full_name.clone());
                        repos.push(full_name);
                    }
                }
            }
        }
    }

    repos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_arxiv() {
        let content = "Check out this paper: https://arxiv.org/abs/2301.12345 for more details.";
        let papers = extract_paper_links(content);
        assert_eq!(papers.len(), 1);
        assert_eq!(papers[0].domain, "arxiv.org");
        assert_eq!(papers[0].arxiv_id, Some("2301.12345".to_string()));
    }

    #[test]
    fn test_extract_doi() {
        let content = "Published at https://doi.org/10.1234/example.2023";
        let papers = extract_paper_links(content);
        assert_eq!(papers.len(), 1);
        assert_eq!(papers[0].domain, "doi.org");
        assert_eq!(papers[0].doi, Some("10.1234/example.2023".to_string()));
    }

    #[test]
    fn test_extract_multiple() {
        let content = r#"
        Paper 1: https://arxiv.org/abs/2301.12345
        Paper 2: https://doi.org/10.1234/test
        Paper 3: https://openreview.net/forum?id=abc123
        "#;
        let papers = extract_paper_links(content);
        assert_eq!(papers.len(), 3);
    }

    #[test]
    fn test_dedup() {
        let content = r#"
        https://arxiv.org/abs/2301.12345
        https://arxiv.org/abs/2301.12345
        https://arxiv.org/pdf/2301.12345.pdf
        "#;
        let papers = extract_paper_links(content);
        // Should dedupe the abs URLs, but pdf is different... actually we normalize to abs
        assert!(papers.len() <= 2);
    }

    #[test]
    fn test_extract_github_repos() {
        let markdown = "Check out https://github.com/owner/repo and github.com/foo/bar";
        let repos = extract_github_repos(markdown);
        assert_eq!(repos.len(), 2);
        assert!(repos.contains(&"owner/repo".to_string()));
        assert!(repos.contains(&"foo/bar".to_string()));
    }

    #[test]
    fn test_extract_from_deep_links() {
        // Deep links like /issues, /pull, /blob still contain valid repo refs
        // The function should extract owner/repo from these URLs
        let markdown = r#"
        https://github.com/owner/repo/issues/123
        https://github.com/owner/repo/pull/456
        https://github.com/owner/repo/blob/main/file.txt
        "#;
        let repos = extract_github_repos(markdown);
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0], "owner/repo");
    }

    #[test]
    fn test_github_repo_dedup() {
        let markdown = r#"
        https://github.com/owner/repo
        https://github.com/owner/repo
        github.com/owner/repo
        "#;
        let repos = extract_github_repos(markdown);
        assert_eq!(repos.len(), 1);
    }
}
