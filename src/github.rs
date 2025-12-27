use anyhow::{Context, Result};
use base64::Engine;
use futures::stream::{self, StreamExt};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::proxy::ProxyManager;

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
#[derive(Clone)]
pub struct GitHubClient {
    client: reqwest::Client,
    token: Option<String>,
    debug: bool,
    proxy_manager: Option<ProxyManager>,
    force_proxy: bool,
}

impl GitHubClient {
    pub fn new(token: Option<String>) -> Self {
        Self::new_with_options(token, false, None, false)
    }

    pub fn new_with_options(token: Option<String>, debug: bool, proxy_manager: Option<ProxyManager>, force_proxy: bool) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("goto-gh/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self { client, token, debug, proxy_manager, force_proxy }
    }

    /// Create a client configured to use a specific proxy
    fn client_with_proxy(proxy_url: &str) -> Option<reqwest::Client> {
        reqwest::Client::builder()
            .user_agent("goto-gh/0.1.0")
            .proxy(reqwest::Proxy::all(proxy_url).ok()?)
            .timeout(std::time::Duration::from_secs(15)) // 15s total timeout (some proxies are slow)
            .connect_timeout(std::time::Duration::from_secs(3)) // 3s connect timeout
            .build()
            .ok()
    }

    /// Build REST request with auth header if token available
    fn request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }
        req.header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
    }

    /// Send REST request with optional debug timing (uses main token)
    async fn send_request(&self, url: &str) -> Result<reqwest::Response, reqwest::Error> {
        use std::io::Write;
        if self.debug {
            eprint!("\x1b[90m[API] GET {} ... \x1b[0m", url);
            std::io::stderr().flush().ok();
        }
        let start = std::time::Instant::now();
        let result = self.request(url).send().await;
        if self.debug {
            eprintln!("\x1b[90m{}ms\x1b[0m", start.elapsed().as_millis());
        }
        result
    }

    /// Send REST request via proxy (no auth token - proxies are unauthenticated)
    /// If show_debug is false, suppresses per-request debug output (for parallel racing)
    async fn send_request_via_proxy_inner(&self, url: &str, proxy: &str, show_debug: bool) -> Result<reqwest::Response, String> {
        use std::io::Write;

        let proxy_url = format!("http://{}", proxy);
        let proxy_client = Self::client_with_proxy(&proxy_url)
            .ok_or_else(|| format!("Failed to create proxy client for {}", proxy))?;

        if self.debug && show_debug {
            eprint!("\x1b[90m[PROXY {}] GET {} ... \x1b[0m", proxy, url);
            std::io::stderr().flush().ok();
        }

        let start = std::time::Instant::now();

        // Build request without auth (unauthenticated proxy request)
        let result = proxy_client
            .get(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await;

        if self.debug && show_debug {
            eprintln!("\x1b[90m{}ms\x1b[0m", start.elapsed().as_millis());
        }

        result.map_err(|e| e.to_string())
    }

    /// Send REST request via proxy with debug output
    async fn send_request_via_proxy(&self, url: &str, proxy: &str) -> Result<reqwest::Response, String> {
        self.send_request_via_proxy_inner(url, proxy, true).await
    }

    /// Try request through proxies - uses sticky proxy if available, otherwise races multiple proxies
    async fn try_with_proxies(&self, url: &str, pm: &ProxyManager) -> Result<(reqwest::Response, Option<String>), String> {
        // First, try the sticky proxy if we have one (with retry for timeouts)
        if let Some(sticky_proxy) = pm.get_current_proxy() {
            for retry in 0..2 {
                match self.send_request_via_proxy(url, &sticky_proxy).await {
                    Ok(response) => {
                        let status = response.status();

                        // Success or expected errors (404, etc.) - proxy worked, keep it sticky
                        if status.is_success() || status == reqwest::StatusCode::NOT_FOUND {
                            return Ok((response, Some(sticky_proxy)));
                        }

                        // Rate limit via this proxy - clear sticky and try others
                        if status == reqwest::StatusCode::FORBIDDEN
                            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        {
                            if self.debug {
                                eprintln!("\x1b[33m[STICKY] {} rate limited, rotating...\x1b[0m", sticky_proxy);
                            }
                            pm.clear_current();
                            break;
                        }
                    }
                    Err(e) => {
                        let is_timeout = e.contains("timed out") || e.contains("timeout");
                        if is_timeout && retry == 0 {
                            // Retry once on timeout
                            if self.debug {
                                eprintln!("\x1b[33m[STICKY] {} timeout, retrying...\x1b[0m", sticky_proxy);
                            }
                            continue;
                        }
                        // Connection error or second timeout - clear sticky proxy
                        if self.debug {
                            eprintln!("\x1b[33m[STICKY] {} failed ({}), rotating...\x1b[0m", sticky_proxy, e);
                        }
                        pm.clear_current();
                        break;
                    }
                }
            }
        }

        // Race multiple proxies in parallel - first successful response wins
        let parallel_count = 5; // Try 5 proxies at once
        let max_rounds = 10; // Up to 10 rounds = 50 proxies total

        for _round in 0..max_rounds {
            // Get batch of proxies to try
            let mut proxies_to_try = Vec::with_capacity(parallel_count);
            for _ in 0..parallel_count {
                if let Some(p) = pm.get_next() {
                    proxies_to_try.push(p);
                }
            }

            if proxies_to_try.is_empty() {
                return Err("No proxies available".to_string());
            }

            // Show what we're racing (compact)
            if self.debug {
                eprint!("\x1b[90m[RACE] {} proxies... \x1b[0m", proxies_to_try.len());
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }

            let race_start = std::time::Instant::now();

            // Race all proxies in parallel (quiet mode - no per-request output)
            let futures: Vec<_> = proxies_to_try
                .iter()
                .map(|proxy| {
                    let proxy = proxy.clone();
                    let url = url.to_string();
                    async move {
                        let result = self.send_request_via_proxy_inner(&url, &proxy, false).await;
                        (proxy, result)
                    }
                })
                .collect();

            let results = futures::future::join_all(futures).await;

            // Find first successful response
            for (proxy, result) in results {
                if let Ok(response) = result {
                    let status = response.status();

                    // Success or expected errors (404, etc.) - proxy worked
                    if status.is_success() || status == reqwest::StatusCode::NOT_FOUND {
                        if self.debug {
                            eprintln!("\x1b[32m{} won ({}ms)\x1b[0m", proxy, race_start.elapsed().as_millis());
                        }
                        pm.mark_working(&proxy);
                        return Ok((response, Some(proxy)));
                    }
                }
            }

            if self.debug {
                eprintln!("\x1b[90mall failed ({}ms)\x1b[0m", race_start.elapsed().as_millis());
            }
        }

        Err(format!("All {} proxy attempts failed", max_rounds * parallel_count))
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

            let response = match self.send_request(&url).await {
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

        // Format reset time as HH:MM local time
        let reset_time = chrono::DateTime::from_timestamp(rate.reset as i64, 0)
            .map(|dt| dt.with_timezone(&chrono::Local).format("%H:%M").to_string())
            .unwrap_or_else(|| "??:??".to_string());

        eprintln!(
            "  \x1b[33m⏸ {} rate limit low ({}/{}), waiting {}m{}s for reset (at {})...\x1b[0m",
            api_name, rate.remaining, rate.limit, wait_mins, wait_remainder, reset_time
        );

        tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;

        eprintln!("  \x1b[32m▶ Rate limit reset, resuming\x1b[0m");
        true
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
        _owner: &str,
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

                // If force_proxy is enabled, go directly to proxy rotation
                if self.force_proxy {
                    if let Some(ref pm) = self.proxy_manager {
                        match self.try_with_proxies(&url, pm).await {
                            Ok((proxy_response, _proxy)) => {
                                let proxy_status = proxy_response.status();
                                if proxy_status.is_success() {
                                    match proxy_response.json::<Vec<OwnerRepoInfo>>().await {
                                        Ok(repos) => {
                                            repos_page = Some(repos);
                                            break;
                                        }
                                        Err(e) => {
                                            last_error = Some(format!("JSON parse error: {}", e));
                                            continue;
                                        }
                                    }
                                } else if proxy_status == reqwest::StatusCode::NOT_FOUND && page == 1 {
                                    anyhow::bail!("404 not found");
                                } else {
                                    last_error = Some(format!("Proxy returned {}", proxy_status));
                                    continue;
                                }
                            }
                            Err(e) => {
                                last_error = Some(format!("Proxy rotation failed: {}", e));
                                continue;
                            }
                        }
                    } else {
                        anyhow::bail!("force_proxy enabled but no proxy_manager configured");
                    }
                }

                let response = match self.send_request(&url).await {
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

                // On rate limit (403), try proxies if available, otherwise wait
                if status == reqwest::StatusCode::FORBIDDEN {
                    if let Some(ref pm) = self.proxy_manager {
                        // Try via proxy rotation
                        eprintln!("  \x1b[33m⏸ Rate limited, switching to proxy rotation...\x1b[0m");
                        match self.try_with_proxies(&url, pm).await {
                            Ok((proxy_response, _proxy)) => {
                                let proxy_status = proxy_response.status();
                                if proxy_status.is_success() {
                                    match proxy_response.json::<Vec<OwnerRepoInfo>>().await {
                                        Ok(repos) => {
                                            repos_page = Some(repos);
                                            break;
                                        }
                                        Err(e) => {
                                            last_error = Some(format!("JSON parse error: {}", e));
                                            continue;
                                        }
                                    }
                                } else if proxy_status == reqwest::StatusCode::NOT_FOUND && page == 1 {
                                    anyhow::bail!("404 not found");
                                }
                            }
                            Err(_) => {}
                        }
                    }

                    // No proxies or proxy failed - wait for rate limit reset
                    last_error = Some("Rate limited (403)".to_string());
                    if let Ok(rates) = self.rate_limit().await {
                        self.wait_for_rate_limit(&rates.core, "REST API", 10).await;
                    } else {
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
                page,
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

    /// List all followers for a user with streaming callback
    pub async fn list_owner_followers_streaming<F>(
        &self,
        owner: &str,
        mut on_page: F,
    ) -> Result<usize>
    where
        F: FnMut(Vec<String>, FollowerProgress) -> Result<()>,
    {
        let mut total_followers = 0;
        let mut page = 1;
        let per_page = 100;

        loop {
            let url = format!(
                "https://api.github.com/users/{}/followers?per_page={}&page={}",
                owner, per_page, page
            );

            // Retry with exponential backoff for transient errors
            let mut last_error = None;
            let mut followers_page: Option<Vec<FollowerInfo>> = None;

            for attempt in 0..5 {
                if attempt > 0 {
                    let delay = std::time::Duration::from_millis(500 * (1 << attempt.min(3)));
                    tokio::time::sleep(delay).await;
                }

                // If force_proxy is enabled, go directly to proxy rotation
                if self.force_proxy {
                    if let Some(ref pm) = self.proxy_manager {
                        match self.try_with_proxies(&url, pm).await {
                            Ok((proxy_response, _proxy)) => {
                                let proxy_status = proxy_response.status();
                                if proxy_status.is_success() {
                                    match proxy_response.json::<Vec<FollowerInfo>>().await {
                                        Ok(followers) => {
                                            followers_page = Some(followers);
                                            break;
                                        }
                                        Err(e) => {
                                            last_error = Some(format!("JSON parse error: {}", e));
                                            continue;
                                        }
                                    }
                                } else if proxy_status == reqwest::StatusCode::NOT_FOUND && page == 1 {
                                    anyhow::bail!("User {} not found (404)", owner);
                                } else {
                                    last_error = Some(format!("Proxy returned {}", proxy_status));
                                    continue;
                                }
                            }
                            Err(e) => {
                                last_error = Some(format!("Proxy rotation failed: {}", e));
                                continue;
                            }
                        }
                    } else {
                        anyhow::bail!("force_proxy enabled but no proxy_manager configured");
                    }
                }

                let response = match self.send_request(&url).await {
                    Ok(r) => r,
                    Err(e) => {
                        last_error = Some(format!("Request failed: {}", e));
                        continue;
                    }
                };

                let status = response.status();

                // 404 = user not found
                if status == reqwest::StatusCode::NOT_FOUND {
                    if page == 1 {
                        anyhow::bail!("User {} not found (404)", owner);
                    }
                    break;
                }

                // Retry on transient errors (502, 503, 504)
                if status == reqwest::StatusCode::BAD_GATEWAY
                    || status == reqwest::StatusCode::GATEWAY_TIMEOUT
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    last_error = Some(format!("GitHub API error {}", status));
                    continue;
                }

                // On rate limit (403), try proxies if available, otherwise wait
                if status == reqwest::StatusCode::FORBIDDEN {
                    if let Some(ref pm) = self.proxy_manager {
                        // Try via proxy rotation
                        eprintln!("  \x1b[33m⏸ Rate limited, switching to proxy rotation...\x1b[0m");
                        match self.try_with_proxies(&url, pm).await {
                            Ok((proxy_response, _proxy)) => {
                                let proxy_status = proxy_response.status();
                                if proxy_status.is_success() {
                                    match proxy_response.json::<Vec<FollowerInfo>>().await {
                                        Ok(followers) => {
                                            followers_page = Some(followers);
                                            break;
                                        }
                                        Err(e) => {
                                            last_error = Some(format!("JSON parse error: {}", e));
                                            continue;
                                        }
                                    }
                                } else if proxy_status == reqwest::StatusCode::NOT_FOUND && page == 1 {
                                    anyhow::bail!("User {} not found (404)", owner);
                                }
                            }
                            Err(_) => {}
                        }
                    }

                    // No proxies or proxy failed - wait for rate limit reset
                    last_error = Some("Rate limited (403)".to_string());
                    if let Ok(rates) = self.rate_limit().await {
                        self.wait_for_rate_limit(&rates.core, "REST API", 10).await;
                    } else {
                        eprintln!("  \x1b[33m⏸ Rate limited, waiting 60s...\x1b[0m");
                        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    }
                    continue;
                }

                if !status.is_success() {
                    if total_followers > 0 {
                        return Ok(total_followers);
                    }
                    anyhow::bail!("GitHub API error {}", status);
                }

                match response.json::<Vec<FollowerInfo>>().await {
                    Ok(followers) => {
                        followers_page = Some(followers);
                        break;
                    }
                    Err(e) => {
                        last_error = Some(format!("JSON parse error: {}", e));
                        continue;
                    }
                }
            }

            // All retries exhausted
            let followers = match followers_page {
                Some(f) => f,
                None => {
                    if total_followers > 0 {
                        return Ok(total_followers);
                    }
                    anyhow::bail!("Failed after 5 retries: {}", last_error.unwrap_or_default());
                }
            };

            if followers.is_empty() {
                break;
            }

            let follower_logins: Vec<String> = followers.iter().map(|f| f.login.clone()).collect();
            let count = follower_logins.len();
            total_followers += count;

            // Call the streaming callback with this page's followers
            let progress = FollowerProgress {
                page,
                total_so_far: total_followers,
            };
            on_page(follower_logins, progress)?;

            if count < per_page {
                break;
            }

            page += 1;
        }

        Ok(total_followers)
    }
}

/// Minimal repo info for listing
#[derive(Debug, Deserialize)]
struct OwnerRepoInfo {
    full_name: String,
}

/// Follower info from GitHub API
#[derive(Debug, Deserialize)]
struct FollowerInfo {
    login: String,
}

/// Progress update for streaming repo discovery
#[derive(Debug, Clone)]
pub struct DiscoveryProgress {
    pub page: usize,
    pub total_so_far: usize,
}

/// Progress update for streaming follower discovery
#[derive(Debug, Clone)]
pub struct FollowerProgress {
    pub page: usize,
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

#[derive(Debug, Deserialize)]
struct GraphQLError {
    message: String,
}

// Response type for direct repository queries (dynamic JSON)
#[derive(Debug, Deserialize)]
struct DirectRepoResponse {
    data: Option<serde_json::Value>,
    errors: Option<Vec<GraphQLError>>,
}

impl GitHubClient {
    /// Build a direct repository GraphQL query with aliases
    /// Returns query like: query { r0: repository(owner: "x", name: "y") { ... } r1: ... }
    fn build_direct_repo_query(repo_names: &[String]) -> String {
        let repo_fragment = r#"
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
        "#;

        let repo_queries: Vec<String> = repo_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                let parts: Vec<&str> = name.splitn(2, '/').collect();
                if parts.len() != 2 {
                    return None;
                }
                let owner = parts[0];
                let repo = parts[1];
                Some(format!(
                    "r{}: repository(owner: \"{}\", name: \"{}\") {{ {} }}",
                    i, owner, repo, repo_fragment
                ))
            })
            .collect();

        format!("query {{ {} }}", repo_queries.join("\n"))
    }

    /// Parse a GraphQL repo from JSON value
    fn parse_repo_from_json(value: &serde_json::Value) -> Option<RepoWithReadme> {
        let name_with_owner = value.get("nameWithOwner")?.as_str()?.to_string();
        let description = value.get("description").and_then(|v| v.as_str()).map(String::from);
        let url = value.get("url")?.as_str()?.to_string();
        let stars = value.get("stargazerCount")?.as_u64().unwrap_or(0);
        let language = value
            .get("primaryLanguage")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let topics: Vec<String> = value
            .get("repositoryTopics")
            .and_then(|v| v.get("nodes"))
            .and_then(|v| v.as_array())
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|n| n.get("topic").and_then(|t| t.get("name")).and_then(|n| n.as_str()))
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        let pushed_at = value.get("pushedAt").and_then(|v| v.as_str()).map(String::from);
        let created_at = value.get("createdAt").and_then(|v| v.as_str()).map(String::from);

        // Try multiple README locations
        let readme = value
            .get("readme")
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                value
                    .get("readme_lower")
                    .and_then(|v| v.get("text"))
                    .and_then(|v| v.as_str())
            })
            .or_else(|| {
                value
                    .get("readme_rst")
                    .and_then(|v| v.get("text"))
                    .and_then(|v| v.as_str())
            })
            .map(String::from);

        Some(RepoWithReadme {
            full_name: name_with_owner,
            description,
            html_url: url,
            stars,
            language,
            topics,
            readme,
            pushed_at,
            created_at,
        })
    }

    /// Fetch multiple repos by owner/name using GraphQL (batched, efficient)
    /// Uses direct repository queries instead of search for accurate results
    /// Parallelizes chunk fetching with concurrency limit to maximize throughput
    pub async fn fetch_repos_batch(&self, repo_names: &[String], concurrency: usize) -> Result<Vec<RepoWithReadme>> {
        if repo_names.is_empty() {
            return Ok(vec![]);
        }

        if self.token.is_none() {
            anyhow::bail!("GraphQL requires authentication");
        }

        // Use direct repository queries (50 repos per query for efficiency)
        let chunk_size = 50;
        let batch_start = std::time::Instant::now();
        let chunks: Vec<_> = repo_names.chunks(chunk_size).collect();
        let total_chunks = chunks.len();

        // Concurrency limit for parallel GraphQL requests
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Create futures for all chunks
        let chunk_futures = chunks.into_iter().enumerate().map(|(chunk_idx, chunk)| {
            let gh_client = self.clone();
            let semaphore = Arc::clone(&semaphore);
            let completed_count = Arc::clone(&completed_count);
            let chunk = chunk.to_vec();

            async move {
                // Acquire semaphore permit (limits concurrency)
                let _permit = semaphore.acquire().await.unwrap();

                // Small stagger delay to avoid burst requests hitting rate limits
                // Chunks start ~200ms apart instead of all at once
                if chunk_idx > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(200 * chunk_idx as u64)).await;
                }

                // Build direct repository query with aliases
                let query = Self::build_direct_repo_query(&chunk);

                // Retry with exponential backoff
                let mut last_error = None;

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

                    // Send raw GraphQL query (no variables needed)
                    let request_body = serde_json::json!({ "query": query });
                    let response = match gh_client.client
                        .post("https://api.github.com/graphql")
                        .header("Authorization", format!("Bearer {}", gh_client.token.as_ref().unwrap()))
                        .header("User-Agent", "goto-gh")
                        .json(&request_body)
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
                        if let Ok(resp) = gh_client.send_request(rate_url).await {
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
                                    let reset_time = chrono::DateTime::from_timestamp(rate.reset as i64, 0)
                                        .map(|dt| dt.with_timezone(&chrono::Local).format("%H:%M").to_string())
                                        .unwrap_or_else(|| "??:??".to_string());
                                    eprintln!(
                                        "  \x1b[33m⏸ GraphQL rate limit low ({}/{}), waiting {}m{}s for reset (at {})...\x1b[0m",
                                        rate.remaining, rate.limit, wait_mins, wait_remainder, reset_time
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

                    let gql_response: DirectRepoResponse = match response.json().await {
                        Ok(r) => r,
                        Err(e) => {
                            last_error = Some(format!("Parse error: {}", e));
                            continue;
                        }
                    };

                    if let Some(errors) = &gql_response.errors {
                        let msgs: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();

                        // Check if any error is a rate limit error
                        let is_rate_limit = errors.iter().any(|e|
                            e.message.to_lowercase().contains("rate limit")
                        );

                        if is_rate_limit {
                            last_error = Some("Rate limited (GraphQL error)".to_string());
                            // Check rate limit and wait for reset
                            let rate_url = "https://api.github.com/rate_limit";
                            if let Ok(resp) = gh_client.send_request(rate_url).await {
                                if let Ok(data) = resp.json::<RateLimitResponse>().await {
                                    let rate = &data.resources.graphql;
                                    let now = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs();
                                    if rate.reset > now {
                                        let wait_secs = rate.reset - now + 5;
                                        let wait_mins = wait_secs / 60;
                                        let wait_remainder = wait_secs % 60;
                                        let reset_time = chrono::DateTime::from_timestamp(rate.reset as i64, 0)
                                            .map(|dt| dt.with_timezone(&chrono::Local).format("%H:%M").to_string())
                                            .unwrap_or_else(|| "??:??".to_string());
                                        eprintln!(
                                            "  \x1b[33m⏸ GraphQL rate limit exceeded ({}/{}), waiting {}m{}s for reset (at {})...\x1b[0m",
                                            rate.remaining, rate.limit, wait_mins, wait_remainder, reset_time
                                        );
                                        tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                                        eprintln!("  \x1b[32m▶ Rate limit reset, resuming\x1b[0m");
                                        continue;
                                    }
                                }
                            }
                            // Fallback: wait 60s if we can't get rate limit info
                            eprintln!("  \x1b[33m⏸ GraphQL rate limit exceeded, waiting 60s...\x1b[0m");
                            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                            continue;
                        }

                        // Log non-fatal errors (e.g., "Could not resolve to a Repository")
                        // These are expected for deleted/renamed repos
                        if gh_client.debug && !msgs.is_empty() {
                            let not_found_count = errors.iter()
                                .filter(|e| e.message.contains("Could not resolve"))
                                .count();
                            if not_found_count > 0 {
                                eprintln!("  \x1b[90m({} repos not found)\x1b[0m", not_found_count);
                            }
                        }
                    }

                    // Parse repos from dynamic JSON response
                    let mut repos: Vec<RepoWithReadme> = Vec::new();
                    if let Some(data) = gql_response.data {
                        if let Some(obj) = data.as_object() {
                            for (_key, value) in obj {
                                // Skip null values (repos that don't exist)
                                if value.is_null() {
                                    continue;
                                }
                                if let Some(repo) = Self::parse_repo_from_json(value) {
                                    repos.push(repo);
                                }
                            }
                        }
                    }

                    // Atomic debug output (single line, parallel-safe)
                    let done = completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let with_readme = repos.iter().filter(|r| r.readme.is_some()).count();
                    eprintln!("  \x1b[90m✓ [{}/{}] {}ms - {} repos ({} README){}\x1b[0m",
                        done, total_chunks, req_elapsed.as_millis(),
                        repos.len(), with_readme,
                        if attempt > 0 { format!(", {} retries", attempt) } else { String::new() });

                    return Ok(repos);
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
