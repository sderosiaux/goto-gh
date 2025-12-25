use anyhow::{Context, Result};
use base64::Engine;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;

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

    /// Check rate limit status (returns both REST and GraphQL limits)
    pub async fn rate_limit(&self) -> Result<RateLimitResources> {
        let url = "https://api.github.com/rate_limit";

        let response = self
            .request(url)
            .send()
            .await
            .context("Failed to check rate limit")?;

        let data: RateLimitResponse = response.json().await?;
        Ok(data.resources)
    }

    /// Wait for rate limit reset if we're running low
    /// Returns true if we had to wait, false if we had enough quota
    async fn wait_for_rate_limit(&self, rate: &RateLimit, api_name: &str, min_remaining: u32) -> bool {
        if rate.remaining >= min_remaining {
            return false;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if rate.reset <= now {
            // Reset time already passed, should be fine
            return false;
        }

        let wait_secs = rate.reset - now + 5; // +5s buffer
        let wait_mins = wait_secs / 60;
        let wait_remainder = wait_secs % 60;

        eprintln!(
            "  \x1b[33m⏸ {} rate limit low ({}/{}), waiting {}m{}s for reset...\x1b[0m",
            api_name, rate.remaining, rate.limit, wait_mins, wait_remainder
        );

        tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;

        eprintln!("  \x1b[32m▶ Rate limit reset, resuming\x1b[0m");
        true
    }

    /// Check and wait for REST API rate limit if needed
    pub async fn ensure_rest_quota(&self, min_remaining: u32) -> Result<()> {
        let rates = self.rate_limit().await?;
        self.wait_for_rate_limit(&rates.core, "REST API", min_remaining).await;
        Ok(())
    }

    /// Check and wait for GraphQL API rate limit if needed
    pub async fn ensure_graphql_quota(&self, min_remaining: u32) -> Result<()> {
        let rates = self.rate_limit().await?;
        self.wait_for_rate_limit(&rates.graphql, "GraphQL API", min_remaining).await;
        Ok(())
    }

    /// List all public repos for a user or org (paginated, with retry)
    pub async fn list_owner_repos(&self, owner: &str) -> Result<Vec<String>> {
        let mut all = Vec::new();
        self.list_owner_repos_streaming(owner, |repos, _progress| {
            all.extend(repos);
            Ok(())
        }).await?;
        Ok(all)
    }

    /// List repos with streaming: calls on_page for each page of results
    /// This allows saving repos to DB as we fetch them (resumable)
    pub async fn list_owner_repos_streaming<F>(
        &self,
        owner: &str,
        mut on_page: F,
    ) -> Result<usize>
    where
        F: FnMut(Vec<String>, DiscoveryProgress) -> Result<()>,
    {
        // Try as user first
        match self.list_repos_paginated_streaming(
            owner,
            &format!(
                "https://api.github.com/users/{}/repos?per_page=100&page={{page}}&type=owner",
                owner
            ),
            &mut on_page,
        ).await {
            Ok(count) => return Ok(count),
            Err(e) => {
                // If user not found, try as org
                let err_str = e.to_string();
                if err_str.contains("404") {
                    return self.list_repos_paginated_streaming(
                        owner,
                        &format!(
                            "https://api.github.com/orgs/{}/repos?per_page=100&page={{page}}&type=public",
                            owner
                        ),
                        &mut on_page,
                    ).await;
                }
                return Err(e);
            }
        }
    }

    /// Paginated repo listing with streaming callback and retry logic
    async fn list_repos_paginated_streaming<F>(
        &self,
        owner: &str,
        url_template: &str,
        on_page: &mut F,
    ) -> Result<usize>
    where
        F: FnMut(Vec<String>, DiscoveryProgress) -> Result<()>,
    {
        let mut total_repos = 0;
        let mut page = 1;
        let per_page = 100;

        loop {
            let url = url_template.replace("{page}", &page.to_string());

            // Retry with exponential backoff for transient errors
            let mut last_error = None;
            let mut repos_page: Option<Vec<OwnerRepoInfo>> = None;

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

                // 404 = user/org not found (on first page only)
                if status == reqwest::StatusCode::NOT_FOUND && page == 1 {
                    anyhow::bail!("404 not found");
                }

                // Retry on transient errors (502, 503, 504)
                if status == reqwest::StatusCode::BAD_GATEWAY
                    || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    last_error = Some(format!("GitHub API error {}", status));
                    continue;
                }

                // On rate limit (403), wait for reset then retry
                if status == reqwest::StatusCode::FORBIDDEN {
                    last_error = Some("Rate limited (403)".to_string());
                    // Check actual rate limit and wait for reset
                    if let Ok(rates) = self.rate_limit().await {
                        self.wait_for_rate_limit(&rates.core, "REST API", 10).await;
                    } else {
                        // Fallback: wait 60s if we can't check rate limit
                        eprintln!("  \x1b[33m⏸ Rate limited, waiting 60s...\x1b[0m");
                        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    }
                    continue;
                }

                // Other errors: if we have partial results, return them; otherwise fail
                if !status.is_success() {
                    if total_repos > 0 {
                        return Ok(total_repos);
                    }
                    anyhow::bail!("GitHub API error {}", status);
                }

                match response.json::<Vec<OwnerRepoInfo>>().await {
                    Ok(repos) => {
                        repos_page = Some(repos);
                        break;
                    }
                    Err(e) => {
                        last_error = Some(format!("JSON parse error: {}", e));
                        continue;
                    }
                }
            }

            // All retries exhausted
            let repos = match repos_page {
                Some(r) => r,
                None => {
                    if total_repos > 0 {
                        return Ok(total_repos);
                    }
                    anyhow::bail!("Failed after 5 retries: {}", last_error.unwrap_or_default());
                }
            };

            if repos.is_empty() {
                break;
            }

            let repo_names: Vec<String> = repos.iter().map(|r| r.full_name.clone()).collect();
            let count = repo_names.len();
            total_repos += count;

            // Call the streaming callback with this page's repos
            let progress = DiscoveryProgress {
                owner: owner.to_string(),
                page,
                repos_this_page: count,
                total_so_far: total_repos,
            };
            on_page(repo_names, progress)?;

            if count < per_page {
                break;
            }

            page += 1;
        }

        Ok(total_repos)
    }
}

/// Minimal repo info for listing
#[derive(Debug, Deserialize)]
struct OwnerRepoInfo {
    full_name: String,
}

/// Progress update for streaming repo discovery
#[derive(Debug, Clone)]
pub struct DiscoveryProgress {
    pub owner: String,
    pub page: usize,
    pub repos_this_page: usize,
    pub total_so_far: usize,
}

#[derive(Debug, Deserialize)]
pub struct RateLimit {
    pub limit: u32,
    pub remaining: u32,
    pub reset: u64,
}

#[derive(Debug, Deserialize)]
pub struct RateLimitResources {
    pub core: RateLimit,
    pub graphql: RateLimit,
}

#[derive(Debug, Deserialize)]
struct RateLimitResponse {
    resources: RateLimitResources,
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
    nodes: Vec<Option<GraphQLRepo>>,
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

    /// Fetch multiple repos by owner/name using GraphQL (batched, efficient)
    /// Uses the search API with "repo:" filters to fetch up to 100 repos in one call
    /// Parallelizes chunk fetching with concurrency limit to maximize throughput
    pub async fn fetch_repos_batch(&self, repo_names: &[String]) -> Result<Vec<RepoWithReadme>> {
        if repo_names.is_empty() {
            return Ok(vec![]);
        }

        let token = self.token.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphQL requires authentication"))?;

        // Build search query with multiple repo: filters (limit to ~30 per query to avoid URL length issues)
        let chunk_size = 30;
        let batch_start = std::time::Instant::now();
        let chunks: Vec<_> = repo_names.chunks(chunk_size).collect();
        let total_chunks = chunks.len();

        // Concurrency limit: 3 parallel requests
        let semaphore = Arc::new(Semaphore::new(3));
        let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Create futures for all chunks
        let chunk_futures = chunks.into_iter().enumerate().map(|(chunk_idx, chunk)| {
            let client = self.client.clone();
            let token = token.clone();
            let semaphore = Arc::clone(&semaphore);
            let completed_count = Arc::clone(&completed_count);
            let chunk = chunk.to_vec();

            async move {
                // Acquire semaphore permit (limits concurrency)
                let _permit = semaphore.acquire().await.unwrap();

                let query = chunk.iter()
                    .map(|name| format!("repo:{}", name))
                    .collect::<Vec<_>>()
                    .join(" ");

                let request = GraphQLRequest {
                    query: GitHubClient::GRAPHQL_QUERY.to_string(),
                    variables: GraphQLVariables {
                        query,
                        first: 100,
                        after: None,
                    },
                };

                // Retry with exponential backoff
                let mut last_error = None;
                let chunk_start = std::time::Instant::now();

                for attempt in 0..5 {
                    if attempt > 0 {
                        let delay_ms = 500 * (1 << attempt.min(3));
                        let delay = std::time::Duration::from_millis(delay_ms);
                        eprintln!("  \x1b[33m⟳ Retry {}/5 for chunk {} after {}ms ({})\x1b[0m",
                            attempt + 1, chunk_idx + 1, delay_ms,
                            last_error.as_deref().unwrap_or("unknown"));
                        tokio::time::sleep(delay).await;
                    }

                    let req_start = std::time::Instant::now();
                    let response = match client
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
                    let req_elapsed = req_start.elapsed();

                    let status = response.status();
                    if status == reqwest::StatusCode::BAD_GATEWAY
                        || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    {
                        last_error = Some(format!("GitHub API error {} ({}ms)", status, req_elapsed.as_millis()));
                        continue;
                    }

                    if status == reqwest::StatusCode::FORBIDDEN {
                        last_error = Some(format!("Rate limited (403) after {}ms", req_elapsed.as_millis()));
                        // Check rate limit and wait for reset
                        let rate_url = "https://api.github.com/rate_limit";
                        if let Ok(resp) = client.get(rate_url)
                            .header("Authorization", format!("Bearer {}", token))
                            .header("User-Agent", "goto-gh/0.1.0")
                            .send()
                            .await
                        {
                            if let Ok(data) = resp.json::<RateLimitResponse>().await {
                                let rate = &data.resources.graphql;
                                let now = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs();
                                if rate.remaining < 10 && rate.reset > now {
                                    let wait_secs = rate.reset - now + 5;
                                    let wait_mins = wait_secs / 60;
                                    let wait_remainder = wait_secs % 60;
                                    eprintln!(
                                        "  \x1b[33m⏸ GraphQL rate limit low ({}/{}), waiting {}m{}s for reset...\x1b[0m",
                                        rate.remaining, rate.limit, wait_mins, wait_remainder
                                    );
                                    tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                                    eprintln!("  \x1b[32m▶ Rate limit reset, resuming\x1b[0m");
                                    continue;
                                }
                            }
                        }
                        // Fallback: short wait
                        eprintln!("  \x1b[33m⏸ Rate limited, waiting 30s...\x1b[0m");
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                        continue;
                    }

                    if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        return Err(anyhow::anyhow!("GraphQL error {}: {}", status, body));
                    }

                    let gql_response: GraphQLResponse = match response.json().await {
                        Ok(r) => r,
                        Err(e) => {
                            last_error = Some(format!("Parse error: {}", e));
                            continue;
                        }
                    };

                    // Log chunk timing (with completed count for parallel visibility)
                    let chunk_elapsed = chunk_start.elapsed();
                    let done = completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    eprintln!("  \x1b[90m✓ Chunk {}/{} completed in {}ms (req: {}ms{})\x1b[0m",
                        done, total_chunks, chunk_elapsed.as_millis(), req_elapsed.as_millis(),
                        if attempt > 0 { format!(", {} retries", attempt) } else { String::new() });

                    if let Some(errors) = gql_response.errors {
                        let msgs: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
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
                                    readme,
                                    pushed_at: repo.pushed_at,
                                    created_at: repo.created_at,
                                })
                            })
                            .collect();

                        return Ok(repos);
                    }

                    return Ok(vec![]);
                }

                Err(anyhow::anyhow!("Failed to fetch chunk after 5 retries: {}",
                    last_error.unwrap_or_else(|| "unknown error".to_string())))
            }
        });

        // Execute all chunks in parallel with buffer_unordered (respects semaphore limit)
        let results: Vec<Result<Vec<RepoWithReadme>>> = stream::iter(chunk_futures)
            .buffer_unordered(8) // Allow up to 8 futures to be polled, semaphore limits actual concurrency
            .collect()
            .await;

        // Collect results
        let mut all_repos = Vec::new();
        let mut had_error = false;
        for result in results {
            match result {
                Ok(repos) => all_repos.extend(repos),
                Err(e) => {
                    eprintln!("  \x1b[31m⚠ Chunk error: {}\x1b[0m", e);
                    had_error = true;
                }
            }
        }

        // Log batch summary
        let batch_elapsed = batch_start.elapsed();
        if batch_elapsed.as_secs() >= 2 || total_chunks > 1 {
            let throughput = if batch_elapsed.as_secs_f32() > 0.0 {
                (repo_names.len() as f32 / batch_elapsed.as_secs_f32()) as u32
            } else {
                0
            };
            eprintln!("  \x1b[90m⏱ GraphQL batch: {} repos in {:.1}s ({} chunks parallel, ~{} repos/s)\x1b[0m",
                repo_names.len(), batch_elapsed.as_secs_f32(), total_chunks, throughput);
        }

        if had_error && all_repos.is_empty() {
            anyhow::bail!("All chunks failed");
        }

        Ok(all_repos)
    }
}
