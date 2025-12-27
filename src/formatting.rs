//! Terminal formatting utilities for goto-gh
//!
//! Provides hyperlinks, star formatting, string truncation, and animated spinners.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Format repo name as clickable hyperlink (only if stdout is a TTY)
pub fn format_repo_link(name: &str, url: &str) -> String {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        // OSC 8 hyperlink: \x1b]8;;URL\x1b\\TEXT\x1b]8;;\x1b\\
        format!("\x1b]8;;{}\x1b\\\x1b[1m{}\x1b[0m\x1b]8;;\x1b\\", url, name)
    } else {
        name.to_string()
    }
}

/// Format owner name as clickable hyperlink (checks stderr since discover outputs there)
pub fn format_owner_link(name: &str, url: &str) -> String {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        format!("\x1b]8;;{}\x1b\\\x1b[1m{}\x1b[0m\x1b]8;;\x1b\\", url, name)
    } else {
        name.to_string()
    }
}

/// Format star count (e.g., 1.2k, 15k)
pub fn format_stars(stars: u64) -> String {
    if stars >= 1000 {
        format!("{}k", stars / 1000)
    } else {
        format!("{}", stars)
    }
}

/// Truncate string safely at char boundary
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 3).collect();
        format!("{}...", truncated)
    }
}

/// Animated dots spinner for long-running operations
pub struct Dots {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Dots {
    pub fn start(message: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let msg = message.to_string();

        let handle = std::thread::spawn(move || {
            const FRAMES: &[&str] = &[
                "\u{28CB}", "\u{28D9}", "\u{28F9}", "\u{28F8}",
                "\u{28FC}", "\u{28F4}", "\u{28E6}", "\u{28E7}",
                "\u{28C7}", "\u{28CF}",
            ];
            let mut i = 0;
            while running_clone.load(Ordering::Relaxed) {
                eprint!("\r\x1b[36m{}\x1b[0m {}", FRAMES[i % 10], msg);
                let _ = io::stderr().flush();
                std::thread::sleep(Duration::from_millis(80));
                i += 1;
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        eprint!("\r\x1b[K"); // Clear line
    }
}

impl Drop for Dots {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_stars_under_1k() {
        assert_eq!(format_stars(0), "0");
        assert_eq!(format_stars(1), "1");
        assert_eq!(format_stars(999), "999");
    }

    #[test]
    fn test_format_stars_over_1k() {
        assert_eq!(format_stars(1000), "1k");
        assert_eq!(format_stars(1500), "1k");
        assert_eq!(format_stars(9999), "9k");
        assert_eq!(format_stars(15000), "15k");
        assert_eq!(format_stars(150000), "150k");
    }

    #[test]
    fn test_truncate_str_short() {
        // Strings shorter than max should be unchanged
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("", 10), "");
        assert_eq!(truncate_str("abc", 5), "abc");
    }

    #[test]
    fn test_truncate_str_exact() {
        // Strings exactly at max should be unchanged
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        // Strings longer than max should be truncated with ellipsis
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("this is a long string", 10), "this is...");
    }

    #[test]
    fn test_truncate_str_unicode() {
        // Unicode characters should be handled correctly
        let emoji = "Hello 👋 World";
        let truncated = truncate_str(emoji, 10);
        assert!(truncated.ends_with("..."));
        assert!(truncated.chars().count() <= 10);
    }

    #[test]
    fn test_truncate_str_multibyte() {
        // Multibyte characters should not be split
        let chinese = "你好世界测试字符串";
        let truncated = truncate_str(chinese, 6);
        assert!(truncated.ends_with("..."));
        // Should be 3 chars + "..." = 6 chars max
        assert!(truncated.chars().count() <= 6);
    }
}
