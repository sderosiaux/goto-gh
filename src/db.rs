use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection, OptionalExtension};
use sqlite_vec::sqlite3_vec_init;
use zerocopy::AsBytes;

use crate::config::Config;
use crate::embedding::EMBEDDING_DIM;
use crate::github::{GitHubRepo, RepoWithReadme};

/// Stored repository with metadata
#[derive(Debug, Clone)]
pub struct Repo {
    pub full_name: String,
    pub description: Option<String>,
    pub url: String,
    pub stars: u64,
    pub language: Option<String>,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn open() -> Result<Self> {
        // Initialize sqlite-vec extension
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let db_path = Config::db_path()?;

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create data directory: {}", parent.display()))?;
        }

        let conn = Connection::open(&db_path)
            .with_context(|| format!("Failed to open database: {}", db_path.display()))?;

        let db = Self { conn };
        db.init()?;
        Ok(db)
    }

    fn init(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -2000;

            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT UNIQUE NOT NULL,
                description TEXT,
                url TEXT NOT NULL,
                stars INTEGER DEFAULT 0,
                language TEXT,
                topics TEXT,
                readme_excerpt TEXT,
                embedded_text TEXT,
                pushed_at TEXT,
                created_at TEXT,
                last_indexed TEXT NOT NULL,
                gone INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_repos_name ON repos(full_name);
            CREATE INDEX IF NOT EXISTS idx_repos_stars ON repos(stars DESC);

            CREATE TABLE IF NOT EXISTS index_checkpoints (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_star_range_idx INTEGER NOT NULL,
                last_cursor TEXT,
                total_fetched INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            ",
        )?;

        // Create vector table for embeddings
        self.conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS repo_embeddings USING vec0(
                    repo_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{}]
                )",
                EMBEDDING_DIM
            ),
            [],
        )?;

        // Migration: add gone column if it doesn't exist
        let has_gone: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('repos') WHERE name = 'gone'",
            [],
            |row| row.get(0),
        )?;
        if !has_gone {
            self.conn.execute("ALTER TABLE repos ADD COLUMN gone INTEGER DEFAULT 0", [])?;
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_gone ON repos(gone)", [])?;
        }

        // Table to track explored owners (users/orgs) - avoid re-fetching their repos
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS explored_owners (
                owner TEXT PRIMARY KEY,
                repo_count INTEGER DEFAULT 0,
                explored_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(())
    }

    /// Insert or update a repository
    pub fn upsert_repo(&self, repo: &GitHubRepo, readme_excerpt: Option<&str>, embedded_text: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let topics_json = serde_json::to_string(&repo.topics)?;

        self.conn.execute(
            "INSERT INTO repos (full_name, description, url, stars, language, topics, readme_excerpt, embedded_text, last_indexed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(full_name) DO UPDATE SET
                 description = ?2,
                 url = ?3,
                 stars = ?4,
                 language = ?5,
                 topics = ?6,
                 readme_excerpt = ?7,
                 embedded_text = ?8,
                 last_indexed = ?9",
            params![
                repo.full_name,
                repo.description,
                repo.html_url,
                repo.stargazers_count as i64,
                repo.language,
                topics_json,
                readme_excerpt,
                embedded_text,
                now,
            ],
        )?;

        let id = self.conn.query_row(
            "SELECT id FROM repos WHERE full_name = ?",
            [&repo.full_name],
            |row| row.get(0),
        )?;

        Ok(id)
    }

    /// Store embedding for a repo
    pub fn upsert_embedding(&self, repo_id: i64, embedding: &[f32]) -> Result<()> {
        self.conn.execute(
            "DELETE FROM repo_embeddings WHERE repo_id = ?",
            [repo_id],
        )?;

        self.conn.execute(
            "INSERT INTO repo_embeddings (repo_id, embedding) VALUES (?, ?)",
            params![repo_id, embedding.as_bytes()],
        )?;

        Ok(())
    }

    /// Find most similar repos to a query embedding
    pub fn find_similar(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(i64, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT repo_id, distance
             FROM repo_embeddings
             WHERE embedding MATCH ?
             ORDER BY distance
             LIMIT ?",
        )?;

        let results = stmt.query_map(params![query_embedding.as_bytes(), limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get repo by ID
    pub fn get_repo_by_id(&self, id: i64) -> Result<Option<Repo>> {
        let mut stmt = self.conn.prepare(
            "SELECT full_name, description, url, stars, language FROM repos WHERE id = ?",
        )?;

        let result = stmt
            .query_row([id], |row| {
                Ok(Repo {
                    full_name: row.get(0)?,
                    description: row.get(1)?,
                    url: row.get(2)?,
                    stars: row.get::<_, i64>(3)? as u64,
                    language: row.get(4)?,
                })
            })
            .optional()?;

        Ok(result)
    }

    /// Get statistics
    pub fn stats(&self) -> Result<(usize, usize)> {
        let total: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM repos", [], |row| row.get(0))?;
        let indexed: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repo_embeddings",
            [],
            |row| row.get(0),
        )?;
        Ok((total, indexed))
    }

    /// Get all repos for revectorization (returns raw stored data)
    pub fn get_all_repos_raw(&self) -> Result<Vec<(i64, String, Option<String>, Option<String>, Option<String>, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, full_name, description, language, topics, readme_excerpt FROM repos"
        )?;

        let results = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Update embedding and embedded_text for a repo
    pub fn update_embedding(&self, repo_id: i64, embedded_text: &str, embedding: &[f32]) -> Result<()> {
        self.conn.execute(
            "UPDATE repos SET embedded_text = ? WHERE id = ?",
            params![embedded_text, repo_id],
        )?;

        self.conn.execute(
            "DELETE FROM repo_embeddings WHERE repo_id = ?",
            [repo_id],
        )?;

        self.conn.execute(
            "INSERT INTO repo_embeddings (repo_id, embedding) VALUES (?, ?)",
            params![repo_id, embedding.as_bytes()],
        )?;

        Ok(())
    }

    /// Fuzzy search repos by name (case-insensitive LIKE)
    pub fn find_by_name(&self, pattern: &str, limit: usize) -> Result<Vec<Repo>> {
        let pattern = format!("%{}%", pattern);
        let mut stmt = self.conn.prepare(
            "SELECT full_name, description, url, stars, language
             FROM repos
             WHERE full_name LIKE ?1 COLLATE NOCASE
             ORDER BY stars DESC
             LIMIT ?2",
        )?;

        let results = stmt.query_map(params![pattern, limit as i64], |row| {
            Ok(Repo {
                full_name: row.get(0)?,
                description: row.get(1)?,
                url: row.get(2)?,
                stars: row.get::<_, i64>(3)? as u64,
                language: row.get(4)?,
            })
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Keyword search for hybrid search (searches embedded_text only, not name)
    pub fn find_by_keywords(&self, query: &str, limit: usize) -> Result<Vec<i64>> {
        // Split query into words and search for any match in embedded_text
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Build OR conditions for each word (embedded_text only)
        let conditions: Vec<String> = words
            .iter()
            .enumerate()
            .map(|(i, _)| format!("embedded_text LIKE ?{} COLLATE NOCASE", i + 1))
            .collect();

        let sql = format!(
            "SELECT id FROM repos WHERE {} ORDER BY stars DESC LIMIT ?",
            conditions.join(" OR ")
        );

        let mut stmt = self.conn.prepare(&sql)?;

        let patterns: Vec<String> = words.iter().map(|w| format!("%{}%", w)).collect();

        let mut params_vec: Vec<&dyn rusqlite::ToSql> = patterns
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();
        let limit_i64 = limit as i64;
        params_vec.push(&limit_i64);

        let results = stmt.query_map(rusqlite::params_from_iter(params_vec), |row| {
            row.get::<_, i64>(0)
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get repos that have no embeddings yet (for separate embedding step)
    /// Returns (repo_id, embedded_text) pairs
    pub fn get_repos_without_embeddings(&self, limit: Option<usize>) -> Result<Vec<(i64, String)>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT r.id, r.embedded_text FROM repos r
                 LEFT JOIN repo_embeddings e ON r.id = e.repo_id
                 WHERE e.repo_id IS NULL AND r.embedded_text IS NOT NULL
                 LIMIT {}",
                lim
            ),
            None => "SELECT r.id, r.embedded_text FROM repos r
                     LEFT JOIN repo_embeddings e ON r.id = e.repo_id
                     WHERE e.repo_id IS NULL AND r.embedded_text IS NOT NULL".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Upsert repo metadata only (no embedding) - for fetch-only mode
    pub fn upsert_repo_metadata_only(&self, repo: &RepoWithReadme, embedded_text: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let topics_json = serde_json::to_string(&repo.topics)?;

        self.conn.execute(
            "INSERT INTO repos (full_name, description, url, stars, language, topics, readme_excerpt, embedded_text, pushed_at, created_at, last_indexed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(full_name) DO UPDATE SET
                 description = ?2,
                 url = ?3,
                 stars = ?4,
                 language = ?5,
                 topics = ?6,
                 readme_excerpt = ?7,
                 embedded_text = ?8,
                 pushed_at = ?9,
                 created_at = ?10,
                 last_indexed = ?11",
            params![
                repo.full_name,
                repo.description,
                repo.html_url,
                repo.stars as i64,
                repo.language,
                topics_json,
                repo.readme.as_deref(),
                embedded_text,
                repo.pushed_at,
                repo.created_at,
                now,
            ],
        )?;

        let id = self.conn.query_row(
            "SELECT id FROM repos WHERE full_name = ?",
            [&repo.full_name],
            |row| row.get(0),
        )?;

        Ok(id)
    }

    /// Count repos without embeddings
    pub fn count_repos_without_embeddings(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos r
             LEFT JOIN repo_embeddings e ON r.id = e.repo_id
             WHERE e.repo_id IS NULL AND r.embedded_text IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Add a repo stub (just the name) - no metadata, no embedding
    /// Used for bulk loading repo names before fetching metadata
    pub fn add_repo_stub(&self, full_name: &str) -> Result<bool> {
        // Only insert if not exists - don't update existing repos
        let result = self.conn.execute(
            "INSERT OR IGNORE INTO repos (full_name, url, last_indexed)
             VALUES (?1, ?2, ?3)",
            params![
                full_name,
                format!("https://github.com/{}", full_name),
                "1970-01-01T00:00:00Z", // Ancient date = needs fetch
            ],
        )?;
        Ok(result > 0) // true if inserted, false if already existed
    }

    /// Bulk add repo stubs (optimized for large imports)
    pub fn add_repo_stubs_bulk(&self, names: &[String]) -> Result<(usize, usize)> {
        let mut inserted = 0;
        let mut skipped = 0;

        for name in names {
            if self.add_repo_stub(name)? {
                inserted += 1;
            } else {
                skipped += 1;
            }
        }

        Ok((inserted, skipped))
    }

    /// Get repos that need metadata fetch (have no embedded_text = never fetched, and not gone)
    pub fn get_repos_without_metadata(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT full_name FROM repos WHERE embedded_text IS NULL AND gone = 0 LIMIT {}",
                lim
            ),
            None => "SELECT full_name FROM repos WHERE embedded_text IS NULL AND gone = 0".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count repos without metadata (excluding gone repos)
    pub fn count_repos_without_metadata(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos WHERE embedded_text IS NULL AND gone = 0",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Mark multiple repos as gone (batch)
    pub fn mark_as_gone_bulk(&self, names: &[String]) -> Result<usize> {
        let mut count = 0;
        for name in names {
            self.conn.execute(
                "UPDATE repos SET gone = 1 WHERE full_name = ?",
                [name],
            )?;
            count += 1;
        }
        Ok(count)
    }

    /// Count repos marked as gone
    pub fn count_gone(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos WHERE gone = 1",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Find repos by name match (strongest signal for hybrid search)
    /// Prioritizes: exact repo name > word boundary matches > substring matches
    pub fn find_by_name_match(&self, query: &str, limit: usize) -> Result<Vec<i64>> {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let n = words.len();

        // Param layout (1-indexed for SQL):
        // [1..n]: exact patterns (%/word)
        // [n+1..2n]: starts-with patterns (%/word%)
        // [2n+1..3n]: anywhere patterns (%word%) - used for WHERE
        // [3n+1]: limit

        // Build conditions using the "anywhere" patterns (third set)
        let conditions: Vec<String> = (0..n)
            .map(|i| format!("full_name LIKE ?{} COLLATE NOCASE", 2 * n + i + 1))
            .collect();

        // Scoring: heavily favor when repo name (after /) matches the query word
        let score_parts: Vec<String> = (0..n)
            .flat_map(|i| {
                let exact_idx = i + 1;
                let start_idx = n + i + 1;
                let any_idx = 2 * n + i + 1;
                vec![
                    // Exact repo name match: /raft -> huge boost
                    format!("(CASE WHEN full_name LIKE ?{} COLLATE NOCASE THEN 100 ELSE 0 END)", exact_idx),
                    // Repo name starts with word: /raft- -> big boost
                    format!("(CASE WHEN full_name LIKE ?{} COLLATE NOCASE THEN 50 ELSE 0 END)", start_idx),
                    // Word anywhere in path: small boost
                    format!("(CASE WHEN full_name LIKE ?{} COLLATE NOCASE THEN 1 ELSE 0 END)", any_idx),
                ]
            })
            .collect();

        let sql = format!(
            "SELECT id, ({}) as match_score FROM repos WHERE {} ORDER BY match_score DESC, stars DESC LIMIT ?",
            score_parts.join(" + "),
            conditions.join(" OR ")
        );

        let mut stmt = self.conn.prepare(&sql)?;

        // Build patterns in order: exact, starts-with, anywhere
        let mut patterns: Vec<String> = Vec::new();
        for w in &words {
            patterns.push(format!("%/{}", w)); // exact
        }
        for w in &words {
            patterns.push(format!("%/{}%", w)); // starts-with
        }
        for w in &words {
            patterns.push(format!("%{}%", w)); // anywhere
        }

        let mut params_vec: Vec<&dyn rusqlite::ToSql> = patterns
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();
        let limit_i64 = limit as i64;
        params_vec.push(&limit_i64);

        let results = stmt.query_map(rusqlite::params_from_iter(params_vec), |row| {
            row.get::<_, i64>(0)
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Check if an owner (user/org) has already been explored
    pub fn is_owner_explored(&self, owner: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM explored_owners WHERE owner = ?1",
            params![owner.to_lowercase()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Mark an owner as explored
    pub fn mark_owner_explored(&self, owner: &str, repo_count: usize) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT OR REPLACE INTO explored_owners (owner, repo_count, explored_at) VALUES (?1, ?2, ?3)",
            params![owner.to_lowercase(), repo_count as i64, now],
        )?;
        Ok(())
    }

    /// Get distinct owners from repos that haven't been explored yet
    pub fn get_unexplored_owners(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT DISTINCT LOWER(substr(full_name, 1, instr(full_name, '/') - 1)) as owner
                 FROM repos
                 WHERE owner NOT IN (SELECT owner FROM explored_owners)
                 LIMIT {}",
                lim
            ),
            None => "SELECT DISTINCT LOWER(substr(full_name, 1, instr(full_name, '/') - 1)) as owner
                     FROM repos
                     WHERE owner NOT IN (SELECT owner FROM explored_owners)".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;
        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count unexplored owners
    pub fn count_unexplored_owners(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(DISTINCT LOWER(substr(full_name, 1, instr(full_name, '/') - 1)))
             FROM repos
             WHERE LOWER(substr(full_name, 1, instr(full_name, '/') - 1)) NOT IN (SELECT owner FROM explored_owners)",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }
}
