//! Proxy management for GitHub API requests.
//!
//! Provides simple round-robin proxy rotation.
//! When the main GitHub token hits rate limits, we can use proxies to continue fetching.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Manages a pool of HTTP proxies with simple round-robin rotation.
#[derive(Clone)]
pub struct ProxyManager {
    inner: Arc<Mutex<ProxyManagerInner>>,
}

struct ProxyManagerInner {
    /// All available proxies (ip:port format)
    proxies: Vec<String>,
    /// Current rotation index
    current_index: usize,
    /// Currently sticky proxy (reused until failure)
    current_proxy: Option<String>,
}

impl ProxyManager {
    /// Create a new ProxyManager from a list of proxies.
    ///
    /// # Arguments
    /// * `proxies` - List of proxy addresses in "ip:port" format
    /// * `shuffle` - Whether to shuffle the proxy list
    pub fn new(mut proxies: Vec<String>, shuffle: bool) -> Self {
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

        Self {
            inner: Arc::new(Mutex::new(ProxyManagerInner {
                proxies,
                current_index: 0,
                current_proxy: None,
            })),
        }
    }

    /// Load proxies from a file (one proxy per line, ip:port format).
    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);

        let proxies: Vec<String> = reader
            .lines()
            .map_while(Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#') && Self::is_valid_format(line))
            .collect();

        if proxies.is_empty() {
            anyhow::bail!("No valid proxies found in {}", path.display());
        }

        eprintln!("  Loaded {} proxies from {}", proxies.len(), path.display());

        Ok(Self::new(proxies, true))
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

    /// Get current sticky proxy if set (without rotation fallback).
    pub fn get_current_proxy(&self) -> Option<String> {
        let inner = self.inner.lock().unwrap();
        inner.current_proxy.clone()
    }

    /// Get the next proxy in rotation (does NOT clear sticky proxy).
    pub fn get_next(&self) -> Option<String> {
        let mut inner = self.inner.lock().unwrap();

        if inner.proxies.is_empty() {
            return None;
        }

        let idx = inner.current_index % inner.proxies.len();
        let proxy = inner.proxies[idx].clone();
        inner.current_index = idx + 1;

        Some(proxy)
    }

    /// Mark a proxy as working (sticky - will be reused).
    pub fn mark_working(&self, proxy: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.current_proxy = Some(proxy.to_string());
    }

    /// Clear the sticky proxy (e.g., when it fails).
    pub fn clear_current(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.current_proxy = None;
    }

    /// Get total number of proxies
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.proxies.len()
    }
}

impl std::fmt::Display for ProxyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} proxies", self.len())
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

        let manager = ProxyManager::new(proxies, false);

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
    fn test_valid_format() {
        assert!(ProxyManager::is_valid_format("1.2.3.4:80"));
        assert!(ProxyManager::is_valid_format("192.168.1.1:8080"));
        assert!(!ProxyManager::is_valid_format("invalid"));
        assert!(!ProxyManager::is_valid_format("1.2.3.4"));
        assert!(!ProxyManager::is_valid_format("1.2.3.4:abc"));
    }
}
