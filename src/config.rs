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

    /// Get GitHub token from environment or gh CLI config
    pub fn github_token() -> Option<String> {
        // First try environment variable
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            if !token.is_empty() {
                return Some(token);
            }
        }

        // Try GH_TOKEN (used by gh CLI)
        if let Ok(token) = std::env::var("GH_TOKEN") {
            if !token.is_empty() {
                return Some(token);
            }
        }

        // Try to get from gh CLI config
        if let Ok(output) = std::process::Command::new("gh")
            .args(["auth", "token"])
            .output()
        {
            if output.status.success() {
                let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !token.is_empty() {
                    return Some(token);
                }
            }
        }

        None
    }
}
