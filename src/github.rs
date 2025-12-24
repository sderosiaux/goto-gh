use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};

/// GitHub repository metadata (REST API)
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubRepo {
    pub full_name: String,
    pub description: Option<String>,
    pub html_url: String,
    pub stargazers_count: u64,
    pub language: Option<String>,
    pub topics: Vec<String>,
}

/// Repository with README included (from GraphQL)
#[derive(Debug, Clone)]
pub struct RepoWithReadme {
    pub full_name: String,
    pub description: Option<String>,
    pub html_url: String,
    pub stars: u64,
    pub language: Option<String>,
    pub topics: Vec<String>,
    pub fork: bool,
    pub readme: Option<String>,
    pub pushed_at: Option<String>,
    pub created_at: Option<String>,
}

/// README content response
#[derive(Debug, Deserialize)]
struct ReadmeResponse {
    content: String,
    encoding: String,
}

/// GitHub API client
pub struct GitHubClient {
    client: reqwest::Client,
    token: Option<String>,
}

impl GitHubClient {
    pub fn new(token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("goto-gh/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self { client, token }
    }

    /// Build request with auth header if token available
    fn request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }
        req.header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
    }

    /// Get repository details
    pub async fn get_repo(&self, full_name: &str) -> Result<GitHubRepo> {
        let url = format!("https://api.github.com/repos/{}", full_name);

        let response = self
            .request(&url)
            .send()
            .await
            .context("Failed to send repo request")?;

        if !response.status().is_success() {
            let status = response.status();
            anyhow::bail!("GitHub API error {}: repo not found", status);
        }

        response
            .json::<GitHubRepo>()
            .await
            .context("Failed to parse repo details")
    }

    /// Get README content (decoded) with retry logic
    pub async fn get_readme(&self, full_name: &str) -> Result<Option<String>> {
        let url = format!("https://api.github.com/repos/{}/readme", full_name);

        // Retry with exponential backoff for transient errors
        let mut last_error = None;
        for attempt in 0..5 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << attempt.min(3)));
                tokio::time::sleep(delay).await;
            }

            let response = match self.request(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(format!("Request failed: {}", e));
                    continue;
                }
            };

            let status = response.status();

            if status == reqwest::StatusCode::NOT_FOUND {
                return Ok(None);
            }

            // Retry on transient errors (502, 503, 504)
            if status == reqwest::StatusCode::BAD_GATEWAY
                || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
            {
                last_error = Some(format!("GitHub API error {}", status));
                continue;
            }

            // Retry on rate limit (403) with extra delay
            if status == reqwest::StatusCode::FORBIDDEN {
                last_error = Some("Rate limited (403)".to_string());
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                continue;
            }

            if !status.is_success() {
                anyhow::bail!("GitHub API error {}: failed to fetch README", status);
            }

            let readme: ReadmeResponse = match response.json().await {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(format!("JSON parse error: {}", e));
                    continue;
                }
            };

            if readme.encoding != "base64" {
                return Ok(None);
            }

            // Decode base64 content (GitHub sends it with newlines)
            let cleaned = readme.content.replace('\n', "");
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(&cleaned)
                .context("Failed to decode base64 README")?;

            let text = String::from_utf8(decoded)
                .context("README is not valid UTF-8")?;

            return Ok(Some(text));
        }

        // All retries failed
        anyhow::bail!("Failed to fetch README after 5 retries: {}", last_error.unwrap_or_default())
    }

    /// Check rate limit status
    pub async fn rate_limit(&self) -> Result<RateLimit> {
        let url = "https://api.github.com/rate_limit";

        let response = self
            .request(url)
            .send()
            .await
            .context("Failed to check rate limit")?;

        let data: RateLimitResponse = response.json().await?;
        Ok(data.rate)
    }
}

#[derive(Debug, Deserialize)]
pub struct RateLimit {
    pub limit: u32,
    pub remaining: u32,
    pub reset: u64,
}

#[derive(Debug, Deserialize)]
struct RateLimitResponse {
    rate: RateLimit,
}

// === GraphQL Types ===

#[derive(Serialize)]
struct GraphQLRequest {
    query: String,
    variables: GraphQLVariables,
}

#[derive(Serialize)]
struct GraphQLVariables {
    query: String,
    first: u32,
    after: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GraphQLResponse {
    data: Option<GraphQLData>,
    errors: Option<Vec<GraphQLError>>,
}

#[derive(Debug, Deserialize)]
struct GraphQLError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct GraphQLData {
    search: GraphQLSearch,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GraphQLSearch {
    page_info: PageInfo,
    nodes: Vec<Option<GraphQLRepo>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageInfo {
    has_next_page: bool,
    end_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GraphQLRepo {
    name_with_owner: String,
    description: Option<String>,
    url: String,
    stargazer_count: u64,
    primary_language: Option<PrimaryLanguage>,
    repository_topics: RepositoryTopics,
    is_fork: bool,
    pushed_at: Option<String>,
    created_at: Option<String>,
    readme: Option<BlobContent>,
    readme_lower: Option<BlobContent>,
    readme_rst: Option<BlobContent>,
}

#[derive(Debug, Deserialize)]
struct PrimaryLanguage {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RepositoryTopics {
    nodes: Vec<TopicNode>,
}

#[derive(Debug, Deserialize)]
struct TopicNode {
    topic: Topic,
}

#[derive(Debug, Deserialize)]
struct Topic {
    name: String,
}

#[derive(Debug, Deserialize)]
struct BlobContent {
    text: Option<String>,
}

impl GitHubClient {
    /// GraphQL query for repos with README - gets 100 repos + READMEs in ONE call
    const GRAPHQL_QUERY: &'static str = r#"
query SearchRepos($query: String!, $first: Int!, $after: String) {
  search(query: $query, type: REPOSITORY, first: $first, after: $after) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      ... on Repository {
        nameWithOwner
        description
        url
        stargazerCount
        primaryLanguage { name }
        repositoryTopics(first: 10) {
          nodes { topic { name } }
        }
        isFork
        pushedAt
        createdAt
        readme: object(expression: "HEAD:README.md") {
          ... on Blob { text }
        }
        readme_lower: object(expression: "HEAD:readme.md") {
          ... on Blob { text }
        }
        readme_rst: object(expression: "HEAD:README.rst") {
          ... on Blob { text }
        }
      }
    }
  }
}
"#;

    /// Search repos with GraphQL - returns repos WITH readme in single call
    pub async fn search_repos_graphql(
        &self,
        query: &str,
        count: u32,
        after: Option<String>,
    ) -> Result<(Vec<RepoWithReadme>, Option<String>, bool)> {
        let token = self.token.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphQL requires authentication"))?;

        let request = GraphQLRequest {
            query: Self::GRAPHQL_QUERY.to_string(),
            variables: GraphQLVariables {
                query: format!("{} sort:stars-desc", query),
                first: count.min(100),
                after,
            },
        };

        // Retry with exponential backoff for transient errors (502, 504, etc.)
        let mut last_error = None;
        for attempt in 0..8 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(1000 * (1 << attempt.min(4)));
                tokio::time::sleep(delay).await;
            }

            let response = match self.client
                .post("https://api.github.com/graphql")
                .header("Authorization", format!("Bearer {}", token))
                .header("User-Agent", "goto-gh/0.1.0")
                .json(&request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(format!("Request failed: {}", e));
                    continue;
                }
            };

            let status = response.status();
            // Retry on transient errors (502, 503, 504)
            if status == reqwest::StatusCode::BAD_GATEWAY
                || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
            {
                last_error = Some(format!("GitHub API error {}", status));
                continue;
            }

            // Retry on secondary rate limit (403) with longer backoff
            if status == reqwest::StatusCode::FORBIDDEN {
                let body = response.text().await.unwrap_or_default();
                if body.contains("secondary rate limit") {
                    last_error = Some(format!("Secondary rate limit hit"));
                    // Extra delay for rate limit
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
                anyhow::bail!("GraphQL error {}: {}", status, body);
            }

            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("GraphQL error {}: {}", status, body);
            }

            let gql_response: GraphQLResponse = match response.json().await {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(format!("Parse error: {}", e));
                    continue;
                }
            };

            if let Some(errors) = gql_response.errors {
                let msgs: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
                anyhow::bail!("GraphQL errors: {}", msgs.join(", "));
            }

            let data = gql_response.data
                .ok_or_else(|| anyhow::anyhow!("No data in GraphQL response"))?;

            let repos: Vec<RepoWithReadme> = data.search.nodes
                .into_iter()
                .filter_map(|node| {
                    let repo = node?;

                    // Get README from any of the common locations
                    let readme = repo.readme
                        .and_then(|b| b.text)
                        .or_else(|| repo.readme_lower.and_then(|b| b.text))
                        .or_else(|| repo.readme_rst.and_then(|b| b.text));

                    Some(RepoWithReadme {
                        full_name: repo.name_with_owner,
                        description: repo.description,
                        html_url: repo.url,
                        stars: repo.stargazer_count,
                        language: repo.primary_language.map(|l| l.name),
                        topics: repo.repository_topics.nodes
                            .into_iter()
                            .map(|t| t.topic.name)
                            .collect(),
                        fork: repo.is_fork,
                        readme,
                        pushed_at: repo.pushed_at,
                        created_at: repo.created_at,
                    })
                })
                .collect();

            let has_next = data.search.page_info.has_next_page;
            let cursor = data.search.page_info.end_cursor;

            return Ok((repos, cursor, has_next));
        }

        // All retries failed
        anyhow::bail!("GraphQL request failed after 5 retries: {}", last_error.unwrap_or_default())
    }

    /// Fetch multiple repos by owner/name using GraphQL (batched, efficient)
    /// Uses the search API with "repo:" filters to fetch up to 100 repos in one call
    pub async fn fetch_repos_batch(&self, repo_names: &[String]) -> Result<Vec<RepoWithReadme>> {
        if repo_names.is_empty() {
            return Ok(vec![]);
        }

        let token = self.token.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphQL requires authentication"))?;

        // Build search query with multiple repo: filters (limit to ~30 per query to avoid URL length issues)
        let chunk_size = 30;
        let mut all_repos = Vec::new();

        for chunk in repo_names.chunks(chunk_size) {
            let query = chunk.iter()
                .map(|name| format!("repo:{}", name))
                .collect::<Vec<_>>()
                .join(" ");

            let request = GraphQLRequest {
                query: Self::GRAPHQL_QUERY.to_string(),
                variables: GraphQLVariables {
                    query,
                    first: 100,
                    after: None,
                },
            };

            // Retry with exponential backoff
            let mut last_error = None;
            for attempt in 0..5 {
                if attempt > 0 {
                    let delay = std::time::Duration::from_millis(500 * (1 << attempt.min(3)));
                    tokio::time::sleep(delay).await;
                }

                let response = match self.client
                    .post("https://api.github.com/graphql")
                    .header("Authorization", format!("Bearer {}", token))
                    .header("User-Agent", "goto-gh/0.1.0")
                    .json(&request)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_error = Some(format!("Request failed: {}", e));
                        continue;
                    }
                };

                let status = response.status();
                if status == reqwest::StatusCode::BAD_GATEWAY
                    || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    last_error = Some(format!("GitHub API error {}", status));
                    continue;
                }

                if status == reqwest::StatusCode::FORBIDDEN {
                    last_error = Some("Rate limited (403)".to_string());
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    continue;
                }

                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    anyhow::bail!("GraphQL error {}: {}", status, body);
                }

                let gql_response: GraphQLResponse = match response.json().await {
                    Ok(r) => r,
                    Err(e) => {
                        last_error = Some(format!("Parse error: {}", e));
                        continue;
                    }
                };

                if let Some(errors) = gql_response.errors {
                    let msgs: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
                    // Don't fail on partial errors - some repos might not exist
                    eprintln!("  \x1b[90mGraphQL warnings: {}\x1b[0m", msgs.join(", "));
                }

                if let Some(data) = gql_response.data {
                    let repos: Vec<RepoWithReadme> = data.search.nodes
                        .into_iter()
                        .filter_map(|node| {
                            let repo = node?;
                            let readme = repo.readme
                                .and_then(|b| b.text)
                                .or_else(|| repo.readme_lower.and_then(|b| b.text))
                                .or_else(|| repo.readme_rst.and_then(|b| b.text));

                            Some(RepoWithReadme {
                                full_name: repo.name_with_owner,
                                description: repo.description,
                                html_url: repo.url,
                                stars: repo.stargazer_count,
                                language: repo.primary_language.map(|l| l.name),
                                topics: repo.repository_topics.nodes
                                    .into_iter()
                                    .map(|t| t.topic.name)
                                    .collect(),
                                fork: repo.is_fork,
                                readme,
                                pushed_at: repo.pushed_at,
                                created_at: repo.created_at,
                            })
                        })
                        .collect();

                    all_repos.extend(repos);
                }

                break;
            }

            if last_error.is_some() && all_repos.is_empty() {
                anyhow::bail!("Failed to fetch repos: {}", last_error.unwrap_or_default());
            }

            // Small delay between chunks to avoid secondary rate limits
            if chunk_size < repo_names.len() {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        Ok(all_repos)
    }
}
