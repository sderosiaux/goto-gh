//! Proxy management for GitHub API requests.
//!
//! Provides round-robin proxy rotation with failure tracking, similar to ytx's ProxyManager.
//! When the main GitHub token hits rate limits, we can use proxies to continue fetching.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Manages a pool of HTTP proxies with rotation and failure tracking.
#[derive(Clone)]
pub struct ProxyManager {
    inner: Arc<Mutex<ProxyManagerInner>>,
}

struct ProxyManagerInner {
    /// All available proxies (ip:port format)
    proxies: Vec<String>,
    /// Set of proxies that have permanently failed
    failed_proxies: HashSet<String>,
    /// Failure count per proxy
    failure_counts: std::collections::HashMap<String, u32>,
    /// Current rotation index
    current_index: usize,
    /// Max failures before permanent removal
    max_failures_per_proxy: u32,
    /// Path to persist failed proxies (optional)
    failed_proxies_file: Option<PathBuf>,
    /// Whether proxy usage is enabled
    enabled: bool,
}

impl ProxyManager {
    /// Create a new ProxyManager from a list of proxies.
    ///
    /// # Arguments
    /// * `proxies` - List of proxy addresses in "ip:port" format
    /// * `shuffle` - Whether to shuffle the proxy list
    /// * `failed_proxies_file` - Optional path to persist failed proxies
    pub fn new(
        mut proxies: Vec<String>,
        shuffle: bool,
        failed_proxies_file: Option<PathBuf>,
    ) -> Self {
        if shuffle {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;

            // Simple shuffle using Fisher-Yates with time-based seed
            let seed = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            let mut rng_state = hasher.finish();

            for i in (1..proxies.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                proxies.swap(i, j);
            }
        }

        // Load previously failed proxies if file exists
        let mut failed_proxies = HashSet::new();
        if let Some(ref path) = failed_proxies_file {
            if path.exists() {
                if let Ok(file) = fs::File::open(path) {
                    let reader = BufReader::new(file);
                    for line in reader.lines().map_while(Result::ok) {
                        let line = line.trim().to_string();
                        if !line.is_empty() && !line.starts_with('#') {
                            failed_proxies.insert(line);
                        }
                    }
                }
            }
        }

        let loaded_failed = failed_proxies.len();
        let total = proxies.len();

        if loaded_failed > 0 {
            eprintln!(
                "  \x1b[90mLoaded {} previously failed proxies ({} available)\x1b[0m",
                loaded_failed,
                total.saturating_sub(loaded_failed)
            );
        }

        Self {
            inner: Arc::new(Mutex::new(ProxyManagerInner {
                proxies,
                failed_proxies,
                failure_counts: std::collections::HashMap::new(),
                current_index: 0,
                max_failures_per_proxy: 3,
                failed_proxies_file,
                enabled: true,
            })),
        }
    }

    /// Load proxies from a file (one proxy per line, ip:port format).
    pub fn from_file(path: &PathBuf, failed_proxies_file: Option<PathBuf>) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let proxies: Vec<String> = content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#') && Self::is_valid_format(line))
            .collect();

        if proxies.is_empty() {
            anyhow::bail!("No valid proxies found in {}", path.display());
        }

        eprintln!("  Loaded {} proxies from {}", proxies.len(), path.display());

        Ok(Self::new(proxies, true, failed_proxies_file))
    }

    /// Validate proxy format (ip:port)
    fn is_valid_format(proxy: &str) -> bool {
        if !proxy.contains(':') {
            return false;
        }
        let parts: Vec<&str> = proxy.split(':').collect();
        if parts.len() != 2 {
            return false;
        }
        parts[1].parse::<u16>().is_ok()
    }

    /// Get the next available proxy (round-robin, skipping failed ones).
    /// Returns None if no proxies available or all have failed.
    pub fn get_next(&self) -> Option<String> {
        let mut inner = self.inner.lock().unwrap();

        if !inner.enabled || inner.proxies.is_empty() {
            return None;
        }

        // Filter out failed proxies - collect owned strings to avoid borrow issues
        let available: Vec<String> = inner
            .proxies
            .iter()
            .filter(|p| !inner.failed_proxies.contains(*p))
            .cloned()
            .collect();

        if available.is_empty() {
            // All proxies exhausted - reset and try again
            eprintln!(
                "  \x1b[33m⟳ All {} proxies exhausted, resetting failure tracking\x1b[0m",
                inner.proxies.len()
            );
            inner.failed_proxies.clear();
            inner.failure_counts.clear();
            inner.current_index = 0;

            // Return first proxy after reset
            return inner.proxies.first().cloned();
        }

        // Round-robin selection
        let idx = inner.current_index % available.len();
        let proxy = available[idx].clone();
        inner.current_index = idx + 1;

        Some(proxy)
    }

    /// Mark a proxy as having failed. After max_failures, it's permanently removed.
    pub fn mark_failed(&self, proxy: &str, reason: &str) {
        let mut inner = self.inner.lock().unwrap();

        let count = inner.failure_counts.entry(proxy.to_string()).or_insert(0);
        *count += 1;

        if *count >= inner.max_failures_per_proxy {
            inner.failed_proxies.insert(proxy.to_string());

            // Persist to file if configured
            if let Some(ref path) = inner.failed_proxies_file {
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(file, "{}", proxy);
                }
            }

            eprintln!(
                "  \x1b[90m✗ Proxy {} permanently failed: {}\x1b[0m",
                proxy, reason
            );
        }
    }

    /// Mark a proxy as successful (resets its failure count).
    pub fn mark_success(&self, proxy: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.failure_counts.remove(proxy);
    }

    /// Get current stats
    pub fn stats(&self) -> ProxyStats {
        let inner = self.inner.lock().unwrap();
        let total = inner.proxies.len();
        let failed = inner.failed_proxies.len();

        ProxyStats {
            total,
            failed,
            available: total.saturating_sub(failed),
        }
    }

}

#[derive(Debug, Clone)]
pub struct ProxyStats {
    pub total: usize,
    pub failed: usize,
    pub available: usize,
}

impl std::fmt::Display for ProxyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} proxies ({} available, {} failed)",
            self.total, self.available, self.failed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_rotation() {
        let proxies = vec![
            "1.2.3.4:80".to_string(),
            "5.6.7.8:8080".to_string(),
            "9.10.11.12:3128".to_string(),
        ];

        let manager = ProxyManager::new(proxies.clone(), false, None);

        // Should rotate through proxies
        let first = manager.get_next().unwrap();
        let second = manager.get_next().unwrap();
        let third = manager.get_next().unwrap();
        let fourth = manager.get_next().unwrap(); // Should wrap around

        assert_eq!(first, "1.2.3.4:80");
        assert_eq!(second, "5.6.7.8:8080");
        assert_eq!(third, "9.10.11.12:3128");
        assert_eq!(fourth, "1.2.3.4:80");
    }

    #[test]
    fn test_proxy_failure_tracking() {
        let proxies = vec![
            "1.2.3.4:80".to_string(),
            "5.6.7.8:8080".to_string(),
        ];

        let manager = ProxyManager::new(proxies, false, None);

        // Mark first proxy as failed 3 times
        manager.mark_failed("1.2.3.4:80", "timeout");
        manager.mark_failed("1.2.3.4:80", "timeout");
        manager.mark_failed("1.2.3.4:80", "timeout");

        // Should now skip the failed proxy
        let stats = manager.stats();
        assert_eq!(stats.available, 1);

        let next = manager.get_next().unwrap();
        assert_eq!(next, "5.6.7.8:8080");
    }

    #[test]
    fn test_valid_format() {
        assert!(ProxyManager::is_valid_format("1.2.3.4:80"));
        assert!(ProxyManager::is_valid_format("192.168.1.1:8080"));
        assert!(!ProxyManager::is_valid_format("invalid"));
        assert!(!ProxyManager::is_valid_format("1.2.3.4"));
        assert!(!ProxyManager::is_valid_format("1.2.3.4:abc"));
    }
}
