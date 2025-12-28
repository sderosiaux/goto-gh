//! Proxy management for GitHub API requests.
//!
//! Provides round-robin proxy rotation with automatic blacklisting.
//! Proxies that fail 3 times are blacklisted and removed from rotation.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of failures before a proxy is blacklisted
const MAX_FAILURES: usize = 3;

/// Known rotating proxy hostnames (each request uses different IP, never blacklist)
const ROTATING_PROXY_HOSTS: &[&str] = &[
    "superproxy.io",      // Bright Data
    "oxylabs.io",         // Oxylabs
    "smartproxy.com",     // Smartproxy
    "proxy.webshare.io",  // Webshare
    "iproyal.com",        // IPRoyal
];

/// Manages a pool of HTTP proxies with round-robin rotation and blacklisting.
#[derive(Clone)]
pub struct ProxyManager {
    inner: Arc<Mutex<ProxyManagerInner>>,
    /// Global waiting flag to avoid duplicate "waiting" messages
    waiting: Arc<AtomicBool>,
    /// Timestamp when waiting message was last printed
    last_wait_msg: Arc<AtomicU64>,
}

struct ProxyManagerInner {
    /// All available proxies (ip:port format)
    proxies: Vec<String>,
    /// Current rotation index
    current_index: usize,
    /// Currently sticky proxy (reused until failure)
    current_proxy: Option<String>,
    /// Failure count per proxy
    failures: HashMap<String, usize>,
    /// Blacklisted proxies
    blacklisted: Vec<String>,
    /// Rate limit reset time per proxy (Unix timestamp)
    rate_limit_until: HashMap<String, u64>,
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
                failures: HashMap::new(),
                blacklisted: Vec::new(),
                rate_limit_until: HashMap::new(),
            })),
            waiting: Arc::new(AtomicBool::new(false)),
            last_wait_msg: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Load proxies from a file (one proxy per line).
    /// Supports formats: ip:port, user:pass@host:port, ip:port:user:pass (Webshare)
    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);

        let proxies: Vec<String> = reader
            .lines()
            .map_while(Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| Self::normalize_proxy(&line))
            .collect();

        if proxies.is_empty() {
            anyhow::bail!("No valid proxies found in {}", path.display());
        }

        eprintln!("  Loaded {} proxies from {}", proxies.len(), path.display());

        Ok(Self::new(proxies, true))
    }

    /// Validate proxy format and normalize to user:pass@host:port or host:port
    /// Supports:
    /// - ip:port (simple proxy)
    /// - user:pass@host:port (authenticated proxy)
    /// - ip:port:user:pass (Webshare format, converted to user:pass@ip:port)
    fn is_valid_format(proxy: &str) -> bool {
        Self::normalize_proxy(proxy).is_some()
    }

    /// Normalize proxy string to standard format for reqwest
    /// Returns None if format is invalid
    fn normalize_proxy(proxy: &str) -> Option<String> {
        if !proxy.contains(':') {
            return None;
        }

        // Format: user:pass@host:port (already normalized)
        if proxy.contains('@') {
            let parts: Vec<&str> = proxy.split('@').collect();
            if parts.len() != 2 {
                return None;
            }
            let host_port: Vec<&str> = parts[1].split(':').collect();
            if host_port.len() != 2 || host_port[1].parse::<u16>().is_err() {
                return None;
            }
            return Some(proxy.to_string());
        }

        let parts: Vec<&str> = proxy.split(':').collect();

        // Format: ip:port:user:pass (Webshare format)
        if parts.len() == 4 {
            let port = parts[1].parse::<u16>();
            if port.is_err() {
                return None;
            }
            // Convert to user:pass@ip:port
            return Some(format!("{}:{}@{}:{}", parts[2], parts[3], parts[0], parts[1]));
        }

        // Format: ip:port (simple proxy)
        if parts.len() == 2 && parts[1].parse::<u16>().is_ok() {
            return Some(proxy.to_string());
        }

        None
    }

    /// Get current sticky proxy if set (without rotation fallback).
    pub fn get_current_proxy(&self) -> Option<String> {
        let inner = self.inner.lock().unwrap();
        inner.current_proxy.clone()
    }

    /// Get the next available proxy in rotation (skips rate-limited proxies).
    /// Returns None if no proxies are available or all are rate-limited.
    pub fn get_next(&self) -> Option<String> {
        let mut inner = self.inner.lock().unwrap();

        if inner.proxies.is_empty() {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let proxy_count = inner.proxies.len();

        // Try each proxy once, looking for one that's not rate-limited
        for _ in 0..proxy_count {
            let idx = inner.current_index % proxy_count;
            let proxy = inner.proxies[idx].clone();
            inner.current_index = idx + 1;

            // Check if this proxy is rate-limited
            if let Some(&reset_time) = inner.rate_limit_until.get(&proxy) {
                if now < reset_time {
                    // Still rate-limited, skip to next
                    continue;
                } else {
                    // Rate limit expired, remove it
                    inner.rate_limit_until.remove(&proxy);
                }
            }

            return Some(proxy);
        }

        // All proxies are rate-limited
        None
    }

    /// Get seconds until the earliest rate limit expires (for waiting)
    pub fn seconds_until_available(&self) -> Option<u64> {
        let inner = self.inner.lock().unwrap();

        if inner.rate_limit_until.is_empty() {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        inner.rate_limit_until
            .values()
            .filter(|&&reset| reset > now)
            .min()
            .map(|&reset| reset - now)
    }

    /// Mark a proxy as rate-limited until the given Unix timestamp
    /// Rotating proxies are not rate-limited since each request uses different IP
    pub fn mark_rate_limited(&self, proxy: &str, reset_timestamp: u64) {
        // Rotating proxies use different IP per request, no need to track rate limits
        if Self::is_rotating_proxy(proxy) {
            return;
        }

        let mut inner = self.inner.lock().unwrap();
        inner.rate_limit_until.insert(proxy.to_string(), reset_timestamp);
    }

    /// Get count of currently rate-limited proxies
    pub fn rate_limited_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        inner.rate_limit_until
            .values()
            .filter(|&&reset| reset > now)
            .count()
    }

    /// Try to acquire the global wait lock and print message (returns true if this caller should wait)
    /// Only prints message if 5+ seconds since last message
    pub fn try_start_waiting(&self, wait_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let last_msg = self.last_wait_msg.load(Ordering::SeqCst);

        // Only print if 5+ seconds since last message
        if now >= last_msg + 5 {
            // Try to update atomically
            if self.last_wait_msg.compare_exchange(last_msg, now, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                let proxy_count = self.len();
                let rate_limited = self.rate_limited_count();
                eprintln!(
                    "\x1b[33m[proxy]\x1b[0m All {} proxies rate-limited ({} active), waiting {}s...",
                    proxy_count, rate_limited, wait_secs
                );
            }
        }

        self.waiting.store(true, Ordering::SeqCst);
        true
    }

    /// Clear the waiting flag
    pub fn done_waiting(&self) {
        self.waiting.store(false, Ordering::SeqCst);
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

    /// Check if a proxy is a rotating proxy (each request = different IP)
    /// These should never be blacklisted since failures are transient
    fn is_rotating_proxy(proxy: &str) -> bool {
        ROTATING_PROXY_HOSTS.iter().any(|host| proxy.contains(host))
    }

    /// Record a failure for a proxy. Returns true if the proxy was blacklisted.
    /// Rotating proxies (Bright Data, etc.) are never blacklisted.
    pub fn record_failure(&self, proxy: &str) -> bool {
        // Never blacklist rotating proxies - each request uses different IP
        if Self::is_rotating_proxy(proxy) {
            return false;
        }

        let mut inner = self.inner.lock().unwrap();

        let count = inner.failures.entry(proxy.to_string()).or_insert(0);
        *count += 1;

        if *count >= MAX_FAILURES {
            // Blacklist this proxy
            inner.blacklisted.push(proxy.to_string());
            inner.proxies.retain(|p| p != proxy);

            // Clear sticky if it was this proxy
            if inner.current_proxy.as_deref() == Some(proxy) {
                inner.current_proxy = None;
            }

            // Reset index if needed
            if !inner.proxies.is_empty() {
                inner.current_index = inner.current_index % inner.proxies.len();
            }

            eprintln!("\x1b[31m[proxy]\x1b[0m {} blacklisted after {} failures ({} remaining)",
                proxy, MAX_FAILURES, inner.proxies.len());
            return true;
        }
        false
    }

    /// Reset failure count for a proxy (on success)
    pub fn reset_failures(&self, proxy: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.failures.remove(proxy);
    }

    /// Get total number of active proxies
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.proxies.len()
    }

    /// Get number of blacklisted proxies
    pub fn blacklisted_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.blacklisted.len()
    }

    /// Check if any proxies are still available
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.proxies.is_empty()
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
