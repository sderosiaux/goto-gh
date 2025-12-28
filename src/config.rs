use anyhow::{Context, Result};
use directories::ProjectDirs;
use std::path::PathBuf;

pub struct Config;

impl Config {
    /// Get the data directory path
    fn data_dir() -> Result<PathBuf> {
        ProjectDirs::from("dev", "goto-gh", "goto-gh")
            .map(|dirs| dirs.data_dir().to_path_buf())
            .context("Could not determine data directory")
    }

    /// Get the database file path
    pub fn db_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("repos.db"))
    }

    /// Get the cache directory for embedding models
    pub fn model_cache_dir() -> Result<PathBuf> {
        ProjectDirs::from("dev", "goto-gh", "goto-gh")
            .map(|dirs| dirs.cache_dir().to_path_buf())
            .context("Could not determine cache directory")
    }

    /// Get GitHub tokens from environment or gh CLI config
    /// Returns all available tokens for rotation
    pub fn github_tokens() -> Vec<String> {
        let mut tokens = Vec::new();

        // First try GITHUB_TOKENS (comma-separated for rotation)
        if let Ok(token_list) = std::env::var("GITHUB_TOKENS") {
            for token in token_list.split(',') {
                let token = token.trim();
                if !token.is_empty() {
                    tokens.push(token.to_string());
                }
            }
        }

        // Then try single GITHUB_TOKEN
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            let token = token.trim();
            if !token.is_empty() && !tokens.contains(&token.to_string()) {
                tokens.push(token.to_string());
            }
        }

        // Try GH_TOKEN (used by gh CLI)
        if let Ok(token) = std::env::var("GH_TOKEN") {
            let token = token.trim();
            if !token.is_empty() && !tokens.contains(&token.to_string()) {
                tokens.push(token.to_string());
            }
        }

        // Try to get from gh CLI config
        if let Ok(output) = std::process::Command::new("gh")
            .args(["auth", "token"])
            .output()
        {
            if output.status.success() {
                let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !token.is_empty() && !tokens.contains(&token) {
                    tokens.push(token);
                }
            }
        }

        tokens
    }

    /// Get single GitHub token (for backward compatibility)
    pub fn github_token() -> Option<String> {
        Self::github_tokens().into_iter().next()
    }
}
