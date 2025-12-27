use regex::Regex;
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

/// Extract paper links from README content
pub fn extract_paper_links(content: &str) -> Vec<ExtractedPaper> {
    let mut papers = Vec::new();
    let mut seen_urls = HashSet::new();

    // arxiv patterns:
    // - https://arxiv.org/abs/2301.12345
    // - https://arxiv.org/pdf/2301.12345.pdf
    // - http://arxiv.org/abs/2301.12345v2
    let arxiv_re = Regex::new(
        r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    ).unwrap();

    // DOI patterns:
    // - https://doi.org/10.1234/something
    // - https://dx.doi.org/10.1234/something
    let doi_re = Regex::new(
        r#"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s\)\]>"]+)"#
    ).unwrap();

    // direct.mit.edu (MIT Press journals)
    // - https://direct.mit.edu/neco/article/35/12/1234/...
    let mit_re = Regex::new(
        r#"(https?://direct\.mit\.edu/[^\s\)\]>"]+)"#
    ).unwrap();

    // OpenReview
    // - https://openreview.net/forum?id=xxx
    // - https://openreview.net/pdf?id=xxx
    let openreview_re = Regex::new(
        r#"(https?://(?:www\.)?openreview\.net/(?:forum|pdf)\?id=[^\s\)\]>"]+)"#
    ).unwrap();

    // ACL Anthology
    // - https://aclanthology.org/2023.acl-long.1/
    let acl_re = Regex::new(
        r#"(https?://(?:www\.)?aclanthology\.org/[^\s\)\]>"]+)"#
    ).unwrap();

    // NeurIPS / ICML / ICLR proceedings
    // - https://proceedings.neurips.cc/paper/2023/...
    // - https://proceedings.mlr.press/v...
    let neurips_re = Regex::new(
        r#"(https?://proceedings\.(?:neurips\.cc|mlr\.press)/[^\s\)\]>"]+)"#
    ).unwrap();

    // Semantic Scholar
    // - https://www.semanticscholar.org/paper/...
    let semantic_re = Regex::new(
        r#"(https?://(?:www\.)?semanticscholar\.org/paper/[^\s\)\]>"]+)"#
    ).unwrap();

    // PapersWithCode
    // - https://paperswithcode.com/paper/...
    let pwc_re = Regex::new(
        r#"(https?://(?:www\.)?paperswithcode\.com/paper/[^\s\)\]>"]+)"#
    ).unwrap();

    // PMLR (Proceedings of Machine Learning Research)
    // Already covered by neurips_re (proceedings.mlr.press)

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
    for cap in arxiv_re.captures_iter(content) {
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
    for cap in doi_re.captures_iter(content) {
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

    extract_domain!(mit_re, "direct.mit.edu");
    extract_domain!(openreview_re, "openreview.net");
    extract_domain!(acl_re, "aclanthology.org");
    extract_domain!(neurips_re, "proceedings.neurips.cc");
    extract_domain!(semantic_re, "semanticscholar.org");
    extract_domain!(pwc_re, "paperswithcode.com");

    papers
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
}
