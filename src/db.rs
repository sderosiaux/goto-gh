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
                last_indexed TEXT NOT NULL
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

    /// Insert or update a repository from GraphQL result (includes README)
    pub fn upsert_repo_with_readme(&self, repo: &RepoWithReadme, readme_excerpt: Option<&str>, embedded_text: &str) -> Result<i64> {
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
                repo.stars as i64,
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

    /// Check if repo exists and is fresh (indexed within N days)
    pub fn is_fresh(&self, full_name: &str, max_age_days: i64) -> Result<bool> {
        let cutoff = (Utc::now() - chrono::Duration::days(max_age_days)).to_rfc3339();

        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM repos WHERE full_name = ? AND last_indexed > ?",
            params![full_name, cutoff],
            |row| row.get(0),
        )?;

        Ok(count > 0)
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

    /// Get index checkpoint (star range index, cursor, total fetched)
    pub fn get_checkpoint(&self) -> Result<Option<(usize, Option<String>, usize)>> {
        let result = self
            .conn
            .query_row(
                "SELECT last_star_range_idx, last_cursor, total_fetched FROM index_checkpoints WHERE id = 1",
                [],
                |row| Ok((
                    row.get::<_, i64>(0)? as usize,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, i64>(2)? as usize,
                )),
            )
            .optional()?;
        Ok(result)
    }

    /// Save index checkpoint
    pub fn save_checkpoint(&self, star_range_idx: usize, cursor: Option<&str>, total_fetched: usize) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO index_checkpoints (id, last_star_range_idx, last_cursor, total_fetched, updated_at)
             VALUES (1, ?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET
                 last_star_range_idx = ?1,
                 last_cursor = ?2,
                 total_fetched = ?3,
                 updated_at = ?4",
            params![star_range_idx as i64, cursor, total_fetched as i64, now],
        )?;
        Ok(())
    }

    /// Clear checkpoint (for fresh indexing)
    pub fn clear_checkpoint(&self) -> Result<()> {
        self.conn.execute("DELETE FROM index_checkpoints WHERE id = 1", [])?;
        Ok(())
    }

    /// Get minimum stars count from existing repos (for checkpoint inference)
    pub fn get_min_stars(&self) -> Result<Option<u64>> {
        let result = self
            .conn
            .query_row("SELECT MIN(stars) FROM repos", [], |row| row.get::<_, Option<i64>>(0))
            .optional()?
            .flatten();
        Ok(result.map(|s| s as u64))
    }

    /// Get count of existing repos
    pub fn get_repo_count(&self) -> Result<usize> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM repos", [], |row| row.get(0))?;
        Ok(count)
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
}
