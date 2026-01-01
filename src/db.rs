use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection, OptionalExtension};
use sqlite_vec::sqlite3_vec_init;
use zerocopy::AsBytes;

use crate::config::Config;
use crate::github::{DiscoveredRepo, GitHubRepo, RepoWithReadme};


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
    path: std::path::PathBuf,
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

        // Set busy timeout to handle concurrent access from multiple CLI instances
        // SQLite will retry for up to 30 seconds before returning SQLITE_BUSY
        conn.busy_timeout(std::time::Duration::from_secs(30))?;

        let db = Self { conn, path: db_path };
        db.init()?;
        Ok(db)
    }

    /// Open an in-memory database for testing
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        // Initialize sqlite-vec extension
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let conn = Connection::open_in_memory()
            .context("Failed to open in-memory database")?;

        let db = Self {
            conn,
            path: std::path::PathBuf::from(":memory:"),
        };
        db.init()?;
        Ok(db)
    }

    /// Get the database file path
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Checkpoint WAL to merge pending writes into main DB file
    /// Uses PASSIVE mode which doesn't block other connections
    pub fn checkpoint(&self) -> Result<()> {
        self.conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    /// Get a reference to the underlying connection (for exploration queries)
    pub fn conn_ref(&self) -> &Connection {
        &self.conn
    }

    fn init(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -307200;

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
                gone INTEGER DEFAULT 0,
                has_embedding INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_repos_name ON repos(full_name);
            CREATE INDEX IF NOT EXISTS idx_repos_name_lower ON repos(LOWER(full_name));
            CREATE INDEX IF NOT EXISTS idx_repos_stars ON repos(stars DESC);
            CREATE INDEX IF NOT EXISTS idx_repos_owner_stars ON repos(owner, stars DESC);

            -- Composite indexes for discovery queries
            CREATE INDEX IF NOT EXISTS idx_repos_gone_embedded ON repos(gone, embedded_text) WHERE embedded_text IS NULL;
            CREATE INDEX IF NOT EXISTS idx_repos_gone_id ON repos(gone, id) WHERE gone = 0;

            -- Drop legacy unused table
            DROP TABLE IF EXISTS index_checkpoints;
            ",
        )?;

        // Note: repo_embeddings table is created lazily by recreate_embeddings_table()
        // when the first embedding operation runs. This allows the dimension to be
        // determined by the provider (local=384, OpenAI=1536) instead of hardcoding.

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

        // Migration: add owner column (denormalized for fast lookups)
        let has_owner: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('repos') WHERE name = 'owner'",
            [],
            |row| row.get(0),
        )?;
        if !has_owner {
            eprintln!("\x1b[36m..\x1b[0m Adding owner column to repos table (one-time migration)...");
            self.conn.execute("ALTER TABLE repos ADD COLUMN owner TEXT", [])?;
            // Populate owner from full_name (extract part before /)
            self.conn.execute(
                "UPDATE repos SET owner = LOWER(substr(full_name, 1, instr(full_name, '/') - 1)) WHERE owner IS NULL",
                [],
            )?;
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_owner ON repos(owner)", [])?;
            eprintln!("\x1b[32mok\x1b[0m Owner column migration complete");
        }

        // Migration: add has_embedding column for fast embedding status lookups
        let has_embedding_col: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('repos') WHERE name = 'has_embedding'",
            [],
            |row| row.get(0),
        )?;
        if !has_embedding_col {
            eprintln!("\x1b[36m..\x1b[0m Adding has_embedding column (one-time migration)...");
            self.conn.execute("ALTER TABLE repos ADD COLUMN has_embedding INTEGER DEFAULT 0", [])?;

            // Populate from existing repo_embeddings table
            // This may take a while for large databases but only runs once
            let embedded_count: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM repo_embeddings",
                [],
                |row| row.get(0),
            ).unwrap_or(0);

            if embedded_count > 0 {
                eprintln!("\x1b[36m..\x1b[0m Syncing {} existing embeddings...", embedded_count);
                self.conn.execute(
                    "UPDATE repos SET has_embedding = 1 WHERE id IN (SELECT repo_id FROM repo_embeddings)",
                    [],
                )?;
            }

            // Create partial index for fast lookups
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_repos_needs_embedding ON repos(has_embedding, gone, embedded_text)
                 WHERE has_embedding = 0 AND gone = 0 AND embedded_text IS NOT NULL",
                [],
            )?;

            eprintln!("\x1b[32mok\x1b[0m has_embedding migration complete");
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

        // Add status column to explored_owners for resume support
        let has_status: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('explored_owners') WHERE name = 'status'",
            [],
            |row| row.get(0),
        )?;
        if !has_status {
            self.conn.execute("ALTER TABLE explored_owners ADD COLUMN status TEXT DEFAULT 'done'", [])?;
        }

        // Table to track owners whose followers have been fetched
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS owner_followers_fetched (
                owner TEXT PRIMARY KEY,
                follower_count INTEGER DEFAULT 0,
                fetched_at INTEGER NOT NULL,
                status TEXT DEFAULT 'done'
            )",
            [],
        )?;

        // Table to track owners we want to explore (from followers, user lists, etc.)
        // This replaces the old __placeholder__ repo hack
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS owners_to_explore (
                owner TEXT PRIMARY KEY,
                added_at INTEGER NOT NULL,
                source TEXT
            )",
            [],
        )?;

        // Migration: remove any existing __placeholder__ repos and migrate to owners_to_explore
        let placeholder_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM repos WHERE full_name LIKE '%/__placeholder__'",
            [],
            |row| row.get(0),
        )?;
        if placeholder_count > 0 {
            let now = Utc::now().timestamp();
            self.conn.execute(
                "INSERT OR IGNORE INTO owners_to_explore (owner, added_at, source)
                 SELECT LOWER(substr(full_name, 1, instr(full_name, '/') - 1)), ?1, 'migrated'
                 FROM repos WHERE full_name LIKE '%/__placeholder__'",
                params![now],
            )?;
            self.conn.execute(
                "DELETE FROM repos WHERE full_name LIKE '%/__placeholder__'",
                [],
            )?;
        }

        // Clean up owners_to_explore: remove owners who already have real repos or are already explored
        self.conn.execute(
            "DELETE FROM owners_to_explore WHERE owner IN (
                SELECT DISTINCT LOWER(substr(full_name, 1, instr(full_name, '/') - 1))
                FROM repos
            )",
            [],
        )?;
        self.conn.execute(
            "DELETE FROM owners_to_explore WHERE owner IN (
                SELECT owner FROM explored_owners
            )",
            [],
        )?;

        // Papers table - stores extracted paper links
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                arxiv_id TEXT,
                doi TEXT,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                published_at TEXT,
                fetched_at TEXT
            )",
            [],
        )?;
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)", [])?;
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)", [])?;

        // Paper sources - links papers to repos that mention them
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_sources (
                paper_id INTEGER NOT NULL,
                repo_id INTEGER NOT NULL,
                context TEXT,
                extracted_at TEXT NOT NULL,
                PRIMARY KEY (paper_id, repo_id),
                FOREIGN KEY (paper_id) REFERENCES papers(id),
                FOREIGN KEY (repo_id) REFERENCES repos(id)
            )",
            [],
        )?;

        // Migration: add papers_extracted_at column to repos
        let has_papers_extracted: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('repos') WHERE name = 'papers_extracted_at'",
            [],
            |row| row.get(0),
        )?;
        if !has_papers_extracted {
            self.conn.execute("ALTER TABLE repos ADD COLUMN papers_extracted_at TEXT", [])?;
        }

        // Migration: add repos_extracted_at column for tracking README parsing for repo discovery
        let has_repos_extracted: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('repos') WHERE name = 'repos_extracted_at'",
            [],
            |row| row.get(0),
        )?;
        if !has_repos_extracted {
            self.conn.execute("ALTER TABLE repos ADD COLUMN repos_extracted_at TEXT", [])?;
        }

        Ok(())
    }

    /// Insert or update a repository
    pub fn upsert_repo(&self, repo: &GitHubRepo, readme_excerpt: Option<&str>, embedded_text: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let topics_json = serde_json::to_string(&repo.topics)?;
        let owner = repo.full_name.split('/').next().unwrap_or("").to_lowercase();

        // Check if repo exists (case-insensitive) and get the canonical name
        let existing: Option<(i64, String)> = self.conn.query_row(
            "SELECT id, full_name FROM repos WHERE LOWER(full_name) = LOWER(?1) LIMIT 1",
            params![&repo.full_name],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;

        if let Some((id, _canonical_name)) = existing {
            // Update existing row using canonical name
            self.conn.execute(
                "UPDATE repos SET
                    description = ?2, url = ?3, stars = ?4, language = ?5, topics = ?6,
                    readme_excerpt = ?7, embedded_text = ?8, last_indexed = ?9, owner = ?10
                 WHERE id = ?1",
                params![
                    id,
                    repo.description,
                    &repo.html_url,
                    repo.stargazers_count as i64,
                    repo.language,
                    topics_json,
                    readme_excerpt,
                    embedded_text,
                    &now,
                    owner,
                ],
            )?;
            return Ok(id);
        }

        // Insert new row
        self.conn.execute(
            "INSERT INTO repos (full_name, description, url, stars, language, topics, readme_excerpt, embedded_text, last_indexed, owner)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
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
                owner,
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

        // Mark repo as having an embedding for fast lookups
        self.conn.execute(
            "UPDATE repos SET has_embedding = 1 WHERE id = ?",
            [repo_id],
        )?;

        Ok(())
    }

    /// Clear all embeddings from the database
    #[allow(dead_code)]
    pub fn clear_all_embeddings(&self) -> Result<usize> {
        let count = self.conn.execute("DELETE FROM repo_embeddings", [])?;
        self.conn.execute("UPDATE repos SET has_embedding = 0", [])?;
        Ok(count)
    }

    /// Recreate the embeddings table with a new dimension
    /// This drops all existing embeddings!
    pub fn recreate_embeddings_table(&self, dim: usize) -> Result<()> {
        self.conn.execute("DROP TABLE IF EXISTS repo_embeddings", [])?;
        self.conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE repo_embeddings USING vec0(
                    repo_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{}]
                )",
                dim
            ),
            [],
        )?;
        // Reset has_embedding flag since all embeddings are gone
        self.conn.execute("UPDATE repos SET has_embedding = 0", [])?;
        Ok(())
    }

    /// Ensure the embeddings table exists with the correct dimension
    /// Creates it if it doesn't exist, does nothing if it already exists
    pub fn ensure_embeddings_table(&self, dim: usize) -> Result<()> {
        self.conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS repo_embeddings USING vec0(
                    repo_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{}]
                )",
                dim
            ),
            [],
        )?;
        Ok(())
    }

    /// Get the current embedding dimension from the table
    pub fn get_embedding_dimension(&self) -> Result<Option<usize>> {
        // Try to get a sample embedding to check dimensions
        let result: Option<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM repo_embeddings LIMIT 1",
            [],
            |row| row.get(0),
        ).ok();

        if let Some(bytes) = result {
            // Each f32 is 4 bytes
            Ok(Some(bytes.len() / 4))
        } else {
            // No embeddings yet, check table schema
            // vec0 tables don't expose dimension easily, so return None if empty
            Ok(None)
        }
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
        // Use has_embedding flag for fast lookups (indexed)
        let sql = match limit {
            Some(lim) => format!(
                "SELECT id, embedded_text FROM repos
                 WHERE has_embedding = 0
                   AND gone = 0
                   AND embedded_text IS NOT NULL
                 LIMIT {}",
                lim
            ),
            None => "SELECT id, embedded_text FROM repos
                     WHERE has_embedding = 0
                       AND gone = 0
                       AND embedded_text IS NOT NULL".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Upsert repo metadata only (no embedding) - for fetch-only mode
    /// Preserves original case from GitHub API response
    pub fn upsert_repo_metadata_only(&self, repo: &RepoWithReadme, embedded_text: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let topics_json = serde_json::to_string(&repo.topics)?;

        // Check if repo exists (case-insensitive) and get its id
        let existing_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM repos WHERE LOWER(full_name) = LOWER(?1)",
            params![&repo.full_name],
            |row| row.get(0),
        ).ok();

        let owner = repo.full_name.split('/').next().unwrap_or("").to_lowercase();

        if let Some(id) = existing_id {
            // Update existing row (keep original full_name, update metadata)
            self.conn.execute(
                "UPDATE repos SET
                    description = ?2,
                    url = ?3,
                    stars = ?4,
                    language = ?5,
                    topics = ?6,
                    readme_excerpt = ?7,
                    embedded_text = ?8,
                    pushed_at = ?9,
                    created_at = ?10,
                    last_indexed = ?11,
                    owner = ?12
                 WHERE id = ?1",
                params![
                    id,
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
                    owner,
                ],
            )?;
            Ok(id)
        } else {
            // Insert new row with original case
            self.conn.execute(
                "INSERT INTO repos (full_name, description, url, stars, language, topics, readme_excerpt, embedded_text, pushed_at, created_at, last_indexed, owner)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    &repo.full_name,
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
                    owner,
                ],
            )?;
            Ok(self.conn.last_insert_rowid())
        }
    }

    /// Count repos without embeddings (excluding gone repos)
    pub fn count_repos_without_embeddings(&self) -> Result<usize> {
        // Use has_embedding flag for fast count (indexed)
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE has_embedding = 0
               AND gone = 0
               AND embedded_text IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Count distinct owners/users in the database
    pub fn count_distinct_owners(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(DISTINCT LOWER(SUBSTR(full_name, 1, INSTR(full_name, '/') - 1)))
             FROM repos",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Add a repo stub (just the name) - no metadata, no embedding
    /// Used for bulk loading repo names before fetching metadata
    /// Preserves original case for GitHub API calls, uses case-insensitive dedup
    #[allow(dead_code)]
    pub fn add_repo_stub(&self, full_name: &str) -> Result<bool> {
        // Check if repo already exists (case-insensitive)
        let exists: bool = self.conn.query_row(
            "SELECT 1 FROM repos WHERE LOWER(full_name) = LOWER(?1) LIMIT 1",
            params![full_name],
            |_| Ok(true),
        ).unwrap_or(false);

        if exists {
            return Ok(false);
        }

        // Insert with original case (needed for GitHub API)
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
        if names.is_empty() {
            return Ok((0, 0));
        }

        // Use a single transaction for all inserts
        self.conn.execute("BEGIN IMMEDIATE", [])?;

        let mut inserted = 0;
        let mut skipped = 0;

        for name in names {
            // Extract owner from full_name (part before /)
            let owner = name.split('/').next().unwrap_or("").to_lowercase();

            // Check if repo already exists (case-insensitive)
            let exists: bool = self.conn.query_row(
                "SELECT 1 FROM repos WHERE LOWER(full_name) = LOWER(?1) LIMIT 1",
                params![name],
                |_| Ok(true),
            ).unwrap_or(false);

            if exists {
                skipped += 1;
                continue;
            }

            // Insert new repo stub
            self.conn.execute(
                "INSERT INTO repos (full_name, url, last_indexed, owner)
                 VALUES (?1, ?2, '1970-01-01T00:00:00Z', ?3)",
                params![name, format!("https://github.com/{}", name), owner],
            )?;
            inserted += 1;
        }

        self.conn.execute("COMMIT", [])?;
        Ok((inserted, skipped))
    }

    /// Save discovered repos with full metadata from REST API
    /// Inserts new repos or updates existing ones with fresh metadata
    /// Returns (inserted_count, updated_count)
    pub fn save_discovered_repos(&self, repos: &[DiscoveredRepo]) -> Result<(usize, usize)> {
        if repos.is_empty() {
            return Ok((0, 0));
        }

        let now = Utc::now().to_rfc3339();
        let mut inserted = 0;
        let mut updated = 0;

        self.conn.execute("BEGIN IMMEDIATE", [])?;

        for repo in repos {
            let owner = repo.full_name.split('/').next().unwrap_or("").to_lowercase();
            let topics_json = serde_json::to_string(&repo.topics).unwrap_or_else(|_| "[]".to_string());

            // Build embedded_text for semantic search (without README for now)
            let embedded_text = format!(
                "{}\n{}\n{}\n{}",
                repo.full_name,
                repo.description.as_deref().unwrap_or(""),
                repo.topics.join(", "),
                repo.language.as_deref().unwrap_or("")
            );

            // Check if repo exists (case-insensitive)
            let existing_id: Option<i64> = self.conn.query_row(
                "SELECT id FROM repos WHERE LOWER(full_name) = LOWER(?1) LIMIT 1",
                params![&repo.full_name],
                |row| row.get(0),
            ).optional()?;

            if let Some(id) = existing_id {
                // Update existing row with fresh metadata
                // Use COALESCE to preserve existing values when new value is NULL
                // But always update stars (even to 0) since that's valid data
                self.conn.execute(
                    "UPDATE repos SET
                        description = COALESCE(?2, description),
                        url = ?3,
                        stars = CASE WHEN ?4 > 0 THEN ?4 ELSE COALESCE(stars, ?4) END,
                        language = COALESCE(?5, language),
                        topics = COALESCE(?6, topics),
                        pushed_at = COALESCE(?7, pushed_at),
                        created_at = COALESCE(?8, created_at),
                        last_indexed = ?9,
                        owner = ?10,
                        embedded_text = COALESCE(
                            CASE WHEN readme_excerpt IS NOT NULL AND readme_excerpt != '' AND readme_excerpt != '[NO_README]'
                                 THEN ?11 || '\n' || readme_excerpt
                                 ELSE ?11
                            END,
                            embedded_text
                        )
                     WHERE id = ?1",
                    params![
                        id,
                        repo.description,
                        &repo.html_url,
                        repo.stargazers_count as i64,
                        repo.language,
                        topics_json,
                        repo.pushed_at,
                        repo.created_at,
                        &now,
                        owner,
                        embedded_text,
                    ],
                )?;
                updated += 1;
            } else {
                // Insert new row with full metadata
                self.conn.execute(
                    "INSERT INTO repos (full_name, description, url, stars, language, topics, pushed_at, created_at, last_indexed, owner, embedded_text)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    params![
                        &repo.full_name,
                        repo.description,
                        &repo.html_url,
                        repo.stargazers_count as i64,
                        repo.language,
                        topics_json,
                        repo.pushed_at,
                        repo.created_at,
                        &now,
                        owner,
                        embedded_text,
                    ],
                )?;
                inserted += 1;
            }
        }

        self.conn.execute("COMMIT", [])?;
        Ok((inserted, updated))
    }

    /// Get repos that need metadata fetch (no stars/description/language = never fetched metadata)
    /// This targets repos that were added as stubs but never got their metadata from REST or GraphQL
    /// Prioritizes repos from owners who already have high-star repos (better ROI)
    pub fn get_repos_without_metadata(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT full_name FROM repos r
                 WHERE gone = 0
                   AND (stars IS NULL OR stars = 0)
                   AND description IS NULL
                   AND language IS NULL
                 ORDER BY (
                   SELECT MAX(stars) FROM repos r2
                   WHERE r2.owner = r.owner AND r2.stars > 0
                 ) DESC NULLS LAST
                 LIMIT {}",
                lim
            ),
            None => "SELECT full_name FROM repos r
                     WHERE gone = 0
                       AND (stars IS NULL OR stars = 0)
                       AND description IS NULL
                       AND language IS NULL
                     ORDER BY (
                       SELECT MAX(stars) FROM repos r2
                       WHERE r2.owner = r.owner AND r2.stars > 0
                     ) DESC NULLS LAST".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count repos without metadata (no stars/description/language)
    pub fn count_repos_without_metadata(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE gone = 0
               AND (stars IS NULL OR stars = 0)
               AND description IS NULL
               AND language IS NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Count repos that have README but still need GraphQL fetch for other metadata
    #[allow(dead_code)]
    pub fn count_repos_with_readme_needing_metadata(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE embedded_text IS NULL
               AND gone = 0
               AND readme_excerpt IS NOT NULL
               AND readme_excerpt != ''
               AND readme_excerpt != '[NO_README]'",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Mark a single repo as gone (deleted/private/legal takedown)
    pub fn mark_as_gone(&self, full_name: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "UPDATE repos SET gone = 1 WHERE LOWER(full_name) = LOWER(?)",
            [full_name],
        )?;
        Ok(rows > 0)
    }

    /// Mark multiple repos as gone (batch, case-insensitive match)
    pub fn mark_as_gone_bulk(&self, names: &[String]) -> Result<usize> {
        let mut count = 0;
        for name in names {
            let rows = self.conn.execute(
                "UPDATE repos SET gone = 1 WHERE LOWER(full_name) = LOWER(?)",
                [name],
            )?;
            count += rows;
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

    /// Count owners in the owners_to_explore table
    pub fn count_owners_to_explore(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM owners_to_explore",
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

    /// Check if an owner's repos have already been fetched
    pub fn is_owner_repos_fetched(&self, owner: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM explored_owners WHERE owner = ?1 AND status = 'done'",
            params![owner.to_lowercase()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Mark an owner as explored (completed)
    pub fn mark_owner_explored(&self, owner: &str, repo_count: usize) -> Result<()> {
        let owner_lower = owner.to_lowercase();
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT OR REPLACE INTO explored_owners (owner, repo_count, explored_at, status) VALUES (?1, ?2, ?3, 'done')",
            params![&owner_lower, repo_count as i64, now],
        )?;
        // Clean up from owners_to_explore since they're now explored
        self.conn.execute(
            "DELETE FROM owners_to_explore WHERE owner = ?1",
            params![&owner_lower],
        )?;
        Ok(())
    }

    /// Mark an owner as in-progress (for resume support)
    pub fn mark_owner_in_progress(&self, owner: &str) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT OR REPLACE INTO explored_owners (owner, repo_count, explored_at, status) VALUES (?1, 0, ?2, 'in_progress')",
            params![owner.to_lowercase(), now],
        )?;
        Ok(())
    }

    /// Get owners that were interrupted (in_progress state)
    pub fn get_in_progress_owners(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT owner FROM explored_owners WHERE status = 'in_progress'"
        )?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;
        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get distinct owners that haven't been explored yet
    /// Combines owners from real repos and the owners_to_explore table
    pub fn get_unexplored_owners(&self, limit: Option<usize>) -> Result<Vec<String>> {
        // Primary source: owners_to_explore table (already deduplicated, indexed)
        // This is much faster than scanning millions of repos
        let sql = match limit {
            Some(lim) => format!(
                "SELECT owner FROM owners_to_explore
                 WHERE NOT EXISTS (SELECT 1 FROM explored_owners e WHERE e.owner = owners_to_explore.owner)
                 LIMIT {}",
                lim
            ),
            None => "SELECT owner FROM owners_to_explore
                     WHERE NOT EXISTS (SELECT 1 FROM explored_owners e WHERE e.owner = owners_to_explore.owner)".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;
        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count unexplored owners
    pub fn count_unexplored_owners(&self) -> Result<usize> {
        // Use owners_to_explore table (indexed, deduplicated)
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM owners_to_explore
             WHERE NOT EXISTS (SELECT 1 FROM explored_owners e WHERE e.owner = owners_to_explore.owner)",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    // ===== Follower fetching tracking =====

    /// Check if an owner's followers have already been fetched
    pub fn is_owner_followers_fetched(&self, owner: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM owner_followers_fetched WHERE owner = ?1 AND status = 'done'",
            params![owner.to_lowercase()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Mark an owner's followers as fetched (completed)
    pub fn mark_owner_followers_fetched(&self, owner: &str, follower_count: usize) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT OR REPLACE INTO owner_followers_fetched (owner, follower_count, fetched_at, status) VALUES (?1, ?2, ?3, 'done')",
            params![owner.to_lowercase(), follower_count as i64, now],
        )?;
        Ok(())
    }

    /// Mark an owner's followers fetch as in-progress
    pub fn mark_owner_followers_in_progress(&self, owner: &str) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT OR REPLACE INTO owner_followers_fetched (owner, follower_count, fetched_at, status) VALUES (?1, 0, ?2, 'in_progress')",
            params![owner.to_lowercase(), now],
        )?;
        Ok(())
    }

    /// Get owners whose followers fetch was interrupted
    pub fn get_in_progress_followers_fetch(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT owner FROM owner_followers_fetched WHERE status = 'in_progress'"
        )?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;
        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get explored owners whose followers haven't been fetched yet
    pub fn get_owners_without_followers(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT owner FROM explored_owners
                 WHERE status = 'done'
                 AND owner NOT IN (SELECT owner FROM owner_followers_fetched)
                 ORDER BY repo_count DESC
                 LIMIT {}",
                lim
            ),
            None => "SELECT owner FROM explored_owners
                     WHERE status = 'done'
                     AND owner NOT IN (SELECT owner FROM owner_followers_fetched)
                     ORDER BY repo_count DESC".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| row.get::<_, String>(0))?;
        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count explored owners whose followers haven't been fetched
    pub fn count_owners_without_followers(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM explored_owners
             WHERE status = 'done'
             AND owner NOT IN (SELECT owner FROM owner_followers_fetched)",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Add a follower as a new owner to explore (if not already known)
    /// Returns true if it was a new owner, false if already existed
    pub fn add_follower_as_owner(&self, follower: &str) -> Result<bool> {
        let follower_lower = follower.to_lowercase();

        // Check if this owner is already in explored_owners (repos fetched)
        let in_explored: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM explored_owners WHERE owner = ?1",
            params![&follower_lower],
            |row| row.get(0),
        )?;

        if in_explored > 0 {
            return Ok(false);
        }

        // Check if this owner is already in owner_followers_fetched (followers fetched)
        let in_followers: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM owner_followers_fetched WHERE owner = ?1",
            params![&follower_lower],
            |row| row.get(0),
        )?;

        if in_followers > 0 {
            return Ok(false);
        }

        // Check if already in owners_to_explore
        let in_to_explore: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM owners_to_explore WHERE owner = ?1",
            params![&follower_lower],
            |row| row.get(0),
        )?;

        if in_to_explore > 0 {
            return Ok(false);
        }

        // Check if we already have repos for this owner (using indexed owner column)
        let has_repos: i64 = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM repos WHERE owner = ?1 LIMIT 1)",
            params![&follower_lower],
            |row| row.get(0),
        )?;

        if has_repos > 0 {
            return Ok(false);
        }

        // Add to owners_to_explore table
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT INTO owners_to_explore (owner, added_at, source) VALUES (?1, ?2, 'follower')",
            params![&follower_lower, now],
        )?;
        Ok(true)
    }

    /// Bulk add followers as owners (optimized with batch filtering)
    pub fn add_followers_as_owners_bulk(&self, followers: &[String]) -> Result<(usize, usize)> {
        if followers.is_empty() {
            return Ok((0, 0));
        }

        // Lowercase all followers upfront
        let followers_lower: Vec<String> = followers.iter().map(|f| f.to_lowercase()).collect();

        // Build a set of candidates, then filter out known owners in bulk
        let mut candidates: std::collections::HashSet<String> = followers_lower.iter().cloned().collect();

        // Filter 1: Remove those already in explored_owners
        {
            let placeholders: String = (0..followers_lower.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
            let sql = format!("SELECT owner FROM explored_owners WHERE owner IN ({})", placeholders);
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> = followers_lower.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| row.get::<_, String>(0))?;
            for row in rows {
                if let Ok(owner) = row {
                    candidates.remove(&owner);
                }
            }
        }

        if candidates.is_empty() {
            return Ok((0, followers.len()));
        }

        // Filter 2: Remove those already in owner_followers_fetched
        {
            let candidate_vec: Vec<&String> = candidates.iter().collect();
            let placeholders: String = (0..candidate_vec.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
            let sql = format!("SELECT owner FROM owner_followers_fetched WHERE owner IN ({})", placeholders);
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> = candidate_vec.iter().map(|s| *s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| row.get::<_, String>(0))?;
            for row in rows {
                if let Ok(owner) = row {
                    candidates.remove(&owner);
                }
            }
        }

        if candidates.is_empty() {
            return Ok((0, followers.len()));
        }

        // Filter 3: Remove those already in owners_to_explore
        {
            let candidate_vec: Vec<&String> = candidates.iter().collect();
            let placeholders: String = (0..candidate_vec.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
            let sql = format!("SELECT owner FROM owners_to_explore WHERE owner IN ({})", placeholders);
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> = candidate_vec.iter().map(|s| *s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| row.get::<_, String>(0))?;
            for row in rows {
                if let Ok(owner) = row {
                    candidates.remove(&owner);
                }
            }
        }

        if candidates.is_empty() {
            return Ok((0, followers.len()));
        }

        // Filter 4: Remove those who already have repos (batch query using owner column)
        {
            let candidate_vec: Vec<&String> = candidates.iter().collect();
            let placeholders: String = (0..candidate_vec.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
            let sql = format!("SELECT DISTINCT owner FROM repos WHERE owner IN ({})", placeholders);
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> = candidate_vec.iter().map(|s| *s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| row.get::<_, String>(0))?;
            for row in rows {
                if let Ok(owner) = row {
                    candidates.remove(&owner);
                }
            }
        }

        let final_candidates: Vec<String> = candidates.into_iter().collect();

        if final_candidates.is_empty() {
            return Ok((0, followers.len()));
        }

        // Insert all remaining candidates in a single transaction using prepared statement
        let now = Utc::now().timestamp();
        self.conn.execute("BEGIN IMMEDIATE", [])?;

        {
            let mut stmt = self.conn.prepare_cached(
                "INSERT INTO owners_to_explore (owner, added_at, source) VALUES (?1, ?2, 'follower')"
            )?;
            for candidate in &final_candidates {
                stmt.execute(params![candidate, now])?;
            }
        }

        self.conn.execute("COMMIT", [])?;

        let added = final_candidates.len();
        let skipped = followers.len() - added;
        Ok((added, skipped))
    }

    // ==================== Paper extraction methods ====================

    /// Get repos that need paper extraction
    /// Returns (repo_id, full_name, readme_excerpt)
    pub fn get_repos_needing_paper_extraction(&self, limit: Option<usize>) -> Result<Vec<(i64, String, String)>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT id, full_name, readme_excerpt FROM repos
                 WHERE readme_excerpt IS NOT NULL
                   AND gone = 0
                   AND (papers_extracted_at IS NULL OR papers_extracted_at < last_indexed)
                 LIMIT {}",
                lim
            ),
            None => "SELECT id, full_name, readme_excerpt FROM repos
                     WHERE readme_excerpt IS NOT NULL
                       AND gone = 0
                       AND (papers_extracted_at IS NULL OR papers_extracted_at < last_indexed)".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count repos needing paper extraction
    pub fn count_repos_needing_paper_extraction(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE readme_excerpt IS NOT NULL
               AND gone = 0
               AND (papers_extracted_at IS NULL OR papers_extracted_at < last_indexed)",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Insert or get a paper, returns paper_id
    pub fn upsert_paper(&self, url: &str, domain: &str, arxiv_id: Option<&str>, doi: Option<&str>) -> Result<i64> {
        // Try to insert, ignore if exists
        self.conn.execute(
            "INSERT OR IGNORE INTO papers (url, domain, arxiv_id, doi) VALUES (?1, ?2, ?3, ?4)",
            params![url, domain, arxiv_id, doi],
        )?;

        // Get the id (either just inserted or existing)
        let id: i64 = self.conn.query_row(
            "SELECT id FROM papers WHERE url = ?",
            [url],
            |row| row.get(0),
        )?;

        Ok(id)
    }

    /// Link a paper to a repo (source)
    pub fn add_paper_source(&self, paper_id: i64, repo_id: i64, context: Option<&str>) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_sources (paper_id, repo_id, context, extracted_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![paper_id, repo_id, context, now],
        )?;
        Ok(())
    }

    /// Mark a repo as having papers extracted
    pub fn mark_papers_extracted(&self, repo_id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE repos SET papers_extracted_at = ? WHERE id = ?",
            params![now, repo_id],
        )?;
        Ok(())
    }

    /// Count total papers
    pub fn count_papers(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM papers",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Count paper sources (repo-paper links)
    pub fn count_paper_sources(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM paper_sources",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get papers needing metadata fetch (no title yet)
    #[allow(dead_code)]
    pub fn get_papers_needing_metadata(&self, limit: Option<usize>) -> Result<Vec<(i64, String, Option<String>, Option<String>)>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT id, url, arxiv_id, doi FROM papers WHERE title IS NULL LIMIT {}",
                lim
            ),
            None => "SELECT id, url, arxiv_id, doi FROM papers WHERE title IS NULL".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Update paper metadata
    #[allow(dead_code)]
    pub fn update_paper_metadata(
        &self,
        paper_id: i64,
        title: Option<&str>,
        authors: Option<&str>,
        abstract_text: Option<&str>,
        published_at: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE papers SET title = ?1, authors = ?2, abstract = ?3, published_at = ?4, fetched_at = ?5 WHERE id = ?6",
            params![title, authors, abstract_text, published_at, now, paper_id],
        )?;
        Ok(())
    }

    // === Missing README Fetch ===

    /// Get repos that have metadata but no README content
    /// Returns (repo_id, full_name) pairs
    pub fn get_repos_without_readme(&self, limit: Option<usize>) -> Result<Vec<(i64, String)>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT id, full_name FROM repos
                 WHERE gone = 0
                   AND embedded_text IS NOT NULL
                   AND (readme_excerpt IS NULL OR readme_excerpt = '')
                 ORDER BY stars DESC
                 LIMIT {}",
                lim
            ),
            None => "SELECT id, full_name FROM repos
                     WHERE gone = 0
                       AND embedded_text IS NOT NULL
                       AND (readme_excerpt IS NULL OR readme_excerpt = '')
                     ORDER BY stars DESC".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count repos without README
    pub fn count_repos_without_readme(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE gone = 0
               AND embedded_text IS NOT NULL
               AND (readme_excerpt IS NULL OR readme_excerpt = '')",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Mark a repo as having no README (so we don't retry)
    pub fn mark_repo_no_readme(&self, repo_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE repos SET readme_excerpt = '[NO_README]' WHERE id = ?",
            [repo_id],
        )?;
        Ok(())
    }

    /// Update README content for a repo and regenerate embedded_text
    pub fn update_repo_readme(&self, repo_id: i64, readme: &str) -> Result<()> {
        // Get current repo data to rebuild embedded_text
        let (full_name, description, topics_json, language): (String, Option<String>, Option<String>, Option<String>) =
            self.conn.query_row(
                "SELECT full_name, description, topics, language FROM repos WHERE id = ?",
                [repo_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;

        // Parse topics
        let topics: Vec<String> = topics_json
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();

        // Rebuild embedded_text with the full README (embedding function handles truncation)
        let embedded_text = crate::embedding::build_embedding_text(
            &full_name,
            description.as_deref(),
            &topics,
            language.as_deref(),
            Some(readme),
        );

        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE repos SET readme_excerpt = ?1, embedded_text = ?2, last_indexed = ?3, has_embedding = 0 WHERE id = ?4",
            params![readme, embedded_text, now, repo_id],
        )?;

        Ok(())
    }

    // === Repo Extraction from READMEs ===

    /// Get repos that need repo extraction from README
    /// Returns (repo_id, full_name, readme_excerpt) for repos with README that haven't been parsed
    pub fn get_repos_needing_repo_extraction(&self, limit: Option<usize>) -> Result<Vec<(i64, String, String)>> {
        let sql = match limit {
            Some(lim) => format!(
                "SELECT id, full_name, readme_excerpt FROM repos
                 WHERE readme_excerpt IS NOT NULL
                   AND readme_excerpt != ''
                   AND readme_excerpt != '[NO_README]'
                   AND (repos_extracted_at IS NULL OR repos_extracted_at < last_indexed)
                 LIMIT {}",
                lim
            ),
            None => "SELECT id, full_name, readme_excerpt FROM repos
                     WHERE readme_excerpt IS NOT NULL
                       AND readme_excerpt != ''
                       AND readme_excerpt != '[NO_README]'
                       AND (repos_extracted_at IS NULL OR repos_extracted_at < last_indexed)".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count repos needing repo extraction
    pub fn count_repos_needing_repo_extraction(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM repos
             WHERE readme_excerpt IS NOT NULL
               AND readme_excerpt != ''
               AND readme_excerpt != '[NO_README]'
               AND (repos_extracted_at IS NULL OR repos_extracted_at < last_indexed)",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Mark a repo as having repos extracted from its README
    pub fn mark_repos_extracted(&self, repo_id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE repos SET repos_extracted_at = ? WHERE id = ?",
            params![now, repo_id],
        )?;
        Ok(())
    }

    // ==================== Exploration / Discovery Methods ====================

    /// Get embedding vector for a specific repo by ID
    /// Returns None if the repo doesn't have an embedding
    pub fn get_embedding(&self, repo_id: i64) -> Result<Option<Vec<f32>>> {
        let result: Option<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM repo_embeddings WHERE repo_id = ?",
            [repo_id],
            |row| row.get(0),
        ).optional()?;

        Ok(result.map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }))
    }

    /// Get repo ID by full_name (case-insensitive)
    pub fn get_repo_id_by_name(&self, full_name: &str) -> Result<Option<i64>> {
        let id: Option<i64> = self.conn.query_row(
            "SELECT id FROM repos WHERE LOWER(full_name) = LOWER(?)",
            [full_name],
            |row| row.get(0),
        ).optional()?;
        Ok(id)
    }

    /// Get a random repo that has an embedding
    /// Returns (repo_id, full_name)
    pub fn random_embedded_repo(&self) -> Result<Option<(i64, String)>> {
        let result: Option<(i64, String)> = self.conn.query_row(
            "SELECT r.id, r.full_name FROM repos r
             INNER JOIN repo_embeddings e ON r.id = e.repo_id
             WHERE r.gone = 0
             ORDER BY RANDOM()
             LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        Ok(result)
    }

    /// Find similar repos excluding specific IDs
    /// Returns (repo_id, distance) pairs sorted by distance
    pub fn find_similar_excluding(
        &self,
        query_embedding: &[f32],
        limit: usize,
        exclude_ids: &[i64],
    ) -> Result<Vec<(i64, f32)>> {
        if exclude_ids.is_empty() {
            return self.find_similar(query_embedding, limit);
        }

        // sqlite-vec requires k parameter in MATCH clause, so we can't filter in SQL
        // Instead, fetch more results than needed and filter in memory
        use std::collections::HashSet;
        let exclude_set: HashSet<i64> = exclude_ids.iter().copied().collect();

        // Fetch extra results to account for filtered items
        let fetch_limit = limit + exclude_ids.len() + 10;
        let all_results = self.find_similar(query_embedding, fetch_limit)?;

        // Filter out excluded IDs and take only what we need
        let filtered: Vec<(i64, f32)> = all_results
            .into_iter()
            .filter(|(id, _)| !exclude_set.contains(id))
            .take(limit)
            .collect();

        Ok(filtered)
    }

    /// Get repos by star range that have embeddings
    /// Returns (repo_id, full_name, stars)
    pub fn get_repos_by_star_range(
        &self,
        min_stars: i64,
        max_stars: Option<i64>,
        limit: usize,
    ) -> Result<Vec<(i64, String, i64)>> {
        let (sql, params_vec): (&str, Vec<i64>) = match max_stars {
            Some(max) => (
                "SELECT r.id, r.full_name, r.stars FROM repos r
                 INNER JOIN repo_embeddings e ON r.id = e.repo_id
                 WHERE r.gone = 0 AND r.stars >= ?1 AND r.stars <= ?2
                 ORDER BY r.stars DESC
                 LIMIT ?3",
                vec![min_stars, max, limit as i64],
            ),
            None => (
                "SELECT r.id, r.full_name, r.stars FROM repos r
                 INNER JOIN repo_embeddings e ON r.id = e.repo_id
                 WHERE r.gone = 0 AND r.stars >= ?1
                 ORDER BY r.stars DESC
                 LIMIT ?2",
                vec![min_stars, limit as i64],
            ),
        };

        let mut stmt = self.conn.prepare(sql)?;
        let results = stmt.query_map(rusqlite::params_from_iter(params_vec), |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get repo details by ID (extended version with more fields)
    pub fn get_repo_details(&self, id: i64) -> Result<Option<RepoDetails>> {
        let result = self.conn.query_row(
            "SELECT full_name, description, url, stars, language, topics FROM repos WHERE id = ?",
            [id],
            |row| {
                let topics_json: Option<String> = row.get(5)?;
                let topics: Vec<String> = topics_json
                    .and_then(|t| serde_json::from_str(&t).ok())
                    .unwrap_or_default();

                Ok(RepoDetails {
                    id,
                    full_name: row.get(0)?,
                    description: row.get(1)?,
                    url: row.get(2)?,
                    stars: row.get::<_, i64>(3)? as u64,
                    language: row.get(4)?,
                    topics,
                })
            },
        ).optional()?;

        Ok(result)
    }

    /// Get aggregated statistics for owners with embedded repos
    /// Used for profile exploration - finds interesting developers/orgs
    /// Returns owners sorted by total stars, with min_repos filter
    pub fn get_owner_stats(
        &self,
        min_repos: usize,
        min_embedded: usize,
        limit: usize,
    ) -> Result<Vec<OwnerStats>> {
        // First, get owners with enough embedded repos
        let mut stmt = self.conn.prepare(
            "SELECT
                owner,
                COUNT(*) as repo_count,
                SUM(stars) as total_stars,
                AVG(stars) as avg_stars,
                SUM(has_embedding) as embedded_count
             FROM repos
             WHERE gone = 0 AND owner IS NOT NULL AND owner != ''
             GROUP BY owner
             HAVING COUNT(*) >= ?1 AND SUM(has_embedding) >= ?2
             ORDER BY SUM(stars) DESC
             LIMIT ?3"
        )?;

        let owners: Vec<(String, u64, u64, f64, u64)> = stmt
            .query_map(params![min_repos as i64, min_embedded as i64, limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)? as u64,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, f64>(3)?,
                    row.get::<_, i64>(4)? as u64,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Pre-prepare statements for efficiency (reused across all owners)
        let mut lang_stmt = self.conn.prepare(
            "SELECT language, COUNT(*) as cnt
             FROM repos
             WHERE owner = ?1 AND gone = 0 AND language IS NOT NULL
             GROUP BY language
             ORDER BY cnt DESC"
        )?;

        let mut topics_stmt = self.conn.prepare(
            "SELECT topics FROM repos
             WHERE owner = ?1 AND gone = 0 AND topics IS NOT NULL AND topics != '[]'"
        )?;

        let mut embedded_stmt = self.conn.prepare(
            "SELECT id FROM repos
             WHERE owner = ?1 AND gone = 0 AND has_embedding = 1
             ORDER BY stars DESC
             LIMIT 20"
        )?;

        let mut results = Vec::with_capacity(owners.len());

        for (owner, repo_count, total_stars, avg_stars, embedded_count) in owners {
            // Get language distribution for this owner
            let languages: Vec<(String, u64)> = lang_stmt
                .query_map([&owner], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
                })?
                .filter_map(|r| r.ok())
                .collect();

            // Get all topics for this owner
            let topics_json: Vec<String> = topics_stmt
                .query_map([&owner], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();

            let mut all_topics: Vec<String> = topics_json
                .iter()
                .filter_map(|t| serde_json::from_str::<Vec<String>>(t).ok())
                .flatten()
                .collect();
            all_topics.sort();
            all_topics.dedup();

            // Get sample of embedded repo IDs (up to 20 for semantic analysis)
            let embedded_repo_ids: Vec<i64> = embedded_stmt
                .query_map([&owner], |row| row.get::<_, i64>(0))?
                .filter_map(|r| r.ok())
                .collect();

            results.push(OwnerStats {
                owner,
                repo_count,
                total_stars,
                avg_stars,
                embedded_count,
                languages,
                topics: all_topics,
                embedded_repo_ids,
            });
        }

        Ok(results)
    }

    /// Get owner stats for a specific owner
    pub fn get_owner_stats_by_name(&self, owner: &str) -> Result<Option<OwnerStats>> {
        let owner_lower = owner.to_lowercase();

        // Get basic stats
        let basic: Option<(u64, u64, f64, u64)> = self.conn
            .query_row(
                "SELECT
                    COUNT(*) as repo_count,
                    SUM(stars) as total_stars,
                    AVG(stars) as avg_stars,
                    SUM(has_embedding) as embedded_count
                 FROM repos
                 WHERE owner = ?1 AND gone = 0",
                [&owner_lower],
                |row| Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, i64>(1).unwrap_or(0) as u64,
                    row.get::<_, f64>(2).unwrap_or(0.0),
                    row.get::<_, i64>(3).unwrap_or(0) as u64,
                ))
            )
            .optional()?;

        let (repo_count, total_stars, avg_stars, embedded_count) = match basic {
            Some(b) if b.0 > 0 => b,
            _ => return Ok(None),
        };

        // Get languages
        let languages: Vec<(String, u64)> = self.conn
            .prepare(
                "SELECT language, COUNT(*) as cnt
                 FROM repos
                 WHERE owner = ?1 AND gone = 0 AND language IS NOT NULL
                 GROUP BY language
                 ORDER BY cnt DESC"
            )?
            .query_map([&owner_lower], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Get topics
        let topics_json: Vec<String> = self.conn
            .prepare(
                "SELECT topics FROM repos
                 WHERE owner = ?1 AND gone = 0 AND topics IS NOT NULL AND topics != '[]'"
            )?
            .query_map([&owner_lower], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut all_topics: Vec<String> = topics_json
            .iter()
            .filter_map(|t| serde_json::from_str::<Vec<String>>(t).ok())
            .flatten()
            .collect();
        all_topics.sort();
        all_topics.dedup();

        // Get embedded repo IDs
        let embedded_repo_ids: Vec<i64> = self.conn
            .prepare(
                "SELECT id FROM repos
                 WHERE owner = ?1 AND gone = 0 AND has_embedding = 1
                 ORDER BY stars DESC
                 LIMIT 20"
            )?
            .query_map([&owner_lower], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(Some(OwnerStats {
            owner: owner_lower,
            repo_count,
            total_stars,
            avg_stars,
            embedded_count,
            languages,
            topics: all_topics,
            embedded_repo_ids,
        }))
    }

    /// Sample random repos with embeddings
    /// Returns (repo_id, full_name, stars)
    pub fn sample_embedded_repos(&self, count: usize) -> Result<Vec<(i64, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.id, r.full_name, r.stars FROM repos r
             INNER JOIN repo_embeddings e ON r.id = e.repo_id
             WHERE r.gone = 0
             ORDER BY RANDOM()
             LIMIT ?"
        )?;

        let results = stmt.query_map([count as i64], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Export repos with embeddings for external tools (JSON format)
    /// Returns (full_name, description, stars, language, topics, embedding)
    pub fn export_embeddings(
        &self,
        sample: Option<usize>,
        min_stars: u64,
        language: Option<&str>,
    ) -> Result<Vec<ExportedRepo>> {
        let mut sql = String::from(
            "SELECT r.full_name, r.description, r.stars, r.language, r.topics, e.embedding
             FROM repos r
             INNER JOIN repo_embeddings e ON r.id = e.repo_id
             WHERE r.gone = 0 AND r.stars >= ?1"
        );

        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(min_stars as i64)];

        if let Some(lang) = language {
            sql.push_str(" AND r.language = ?2");
            params.push(Box::new(lang.to_string()));
        }

        if sample.is_some() {
            sql.push_str(" ORDER BY RANDOM()");
        }

        if let Some(limit) = sample {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let results = stmt.query_map(param_refs.as_slice(), |row| {
            let embedding_blob: Vec<u8> = row.get(5)?;
            let embedding: Vec<f32> = embedding_blob
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            Ok(ExportedRepo {
                full_name: row.get(0)?,
                description: row.get(1)?,
                stars: row.get::<_, i64>(2)? as u64,
                language: row.get(3)?,
                topics: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                embedding,
            })
        })?;

        results.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}

/// Exported repo data for cluster visualization
#[derive(Debug, Clone)]
pub struct ExportedRepo {
    pub full_name: String,
    pub description: Option<String>,
    pub stars: u64,
    pub language: Option<String>,
    pub topics: String,
    pub embedding: Vec<f32>,
}

/// Extended repo details for exploration features
#[derive(Debug, Clone)]
pub struct RepoDetails {
    pub id: i64,
    pub full_name: String,
    pub description: Option<String>,
    pub url: String,
    pub stars: u64,
    pub language: Option<String>,
    pub topics: Vec<String>,
}

/// Owner statistics for profile exploration
#[derive(Debug, Clone)]
pub struct OwnerStats {
    pub owner: String,
    pub repo_count: u64,
    pub total_stars: u64,
    pub avg_stars: f64,
    pub embedded_count: u64,
    /// Languages used by this owner (language -> repo count)
    pub languages: Vec<(String, u64)>,
    /// All topics across repos
    pub topics: Vec<String>,
    /// Sample of repo IDs with embeddings (for semantic analysis)
    pub embedded_repo_ids: Vec<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test database
    fn test_db() -> Database {
        Database::open_in_memory().expect("Failed to create test database")
    }

    // === Basic Operations ===

    #[test]
    fn test_add_repo_stub() {
        let db = test_db();

        // First insert should succeed
        let inserted = db.add_repo_stub("owner/repo").unwrap();
        assert!(inserted);

        // Duplicate should be skipped
        let inserted_again = db.add_repo_stub("owner/repo").unwrap();
        assert!(!inserted_again);
    }

    #[test]
    fn test_add_repo_stub_case_insensitive() {
        let db = test_db();

        // Add with original case
        let inserted = db.add_repo_stub("Owner/Repo").unwrap();
        assert!(inserted);

        // Same repo with different case should be deduplicated
        let inserted_lower = db.add_repo_stub("owner/repo").unwrap();
        assert!(!inserted_lower);

        let inserted_upper = db.add_repo_stub("OWNER/REPO").unwrap();
        assert!(!inserted_upper);
    }

    #[test]
    fn test_add_repo_stubs_bulk() {
        let db = test_db();

        let names = vec![
            "owner1/repo1".to_string(),
            "owner2/repo2".to_string(),
            "owner3/repo3".to_string(),
        ];

        let (inserted, skipped) = db.add_repo_stubs_bulk(&names).unwrap();
        assert_eq!(inserted, 3);
        assert_eq!(skipped, 0);

        // Adding again should skip all
        let (inserted2, skipped2) = db.add_repo_stubs_bulk(&names).unwrap();
        assert_eq!(inserted2, 0);
        assert_eq!(skipped2, 3);
    }

    #[test]
    fn test_stats() {
        let db = test_db();

        let (total, indexed) = db.stats().unwrap();
        assert_eq!(total, 0);
        assert_eq!(indexed, 0);

        // Add some stubs
        db.add_repo_stub("owner/repo1").unwrap();
        db.add_repo_stub("owner/repo2").unwrap();

        let (total, indexed) = db.stats().unwrap();
        assert_eq!(total, 2);
        assert_eq!(indexed, 0); // No embeddings yet
    }

    // === Metadata Operations ===

    #[test]
    fn test_get_repos_without_metadata() {
        let db = test_db();

        // Add stubs (no metadata)
        db.add_repo_stub("owner/repo1").unwrap();
        db.add_repo_stub("owner/repo2").unwrap();

        let without_meta = db.get_repos_without_metadata(None).unwrap();
        assert_eq!(without_meta.len(), 2);

        // Test limit
        let limited = db.get_repos_without_metadata(Some(1)).unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[test]
    fn test_count_repos_without_metadata() {
        let db = test_db();

        db.add_repo_stub("owner/repo1").unwrap();
        db.add_repo_stub("owner/repo2").unwrap();
        db.add_repo_stub("owner/repo3").unwrap();

        let count = db.count_repos_without_metadata().unwrap();
        assert_eq!(count, 3);
    }

    // === Find Operations ===

    #[test]
    fn test_find_by_name() {
        let db = test_db();

        db.add_repo_stub("tensorflow/tensorflow").unwrap();
        db.add_repo_stub("pytorch/pytorch").unwrap();
        db.add_repo_stub("keras-team/keras").unwrap();

        // Search for "tensor"
        let results = db.find_by_name("tensor", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].full_name, "tensorflow/tensorflow");

        // Search for "py"
        let results = db.find_by_name("py", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].full_name, "pytorch/pytorch");
    }

    #[test]
    fn test_find_by_name_case_insensitive() {
        let db = test_db();

        db.add_repo_stub("TensorFlow/TensorFlow").unwrap();

        let results = db.find_by_name("tensorflow", 10).unwrap();
        assert_eq!(results.len(), 1);

        let results = db.find_by_name("TENSORFLOW", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    // === Owner Exploration ===

    #[test]
    fn test_owner_exploration_tracking() {
        let db = test_db();

        // Initially not explored
        assert!(!db.is_owner_repos_fetched("testowner").unwrap());

        // Mark in progress
        db.mark_owner_in_progress("testowner").unwrap();

        // Still not "fetched" (in_progress != done)
        assert!(!db.is_owner_repos_fetched("testowner").unwrap());

        // Check in_progress list
        let in_progress = db.get_in_progress_owners().unwrap();
        assert_eq!(in_progress.len(), 1);
        assert_eq!(in_progress[0], "testowner");

        // Mark as explored/done
        db.mark_owner_explored("testowner", 42).unwrap();

        // Now should be fetched
        assert!(db.is_owner_repos_fetched("testowner").unwrap());

        // In progress list should be empty
        let in_progress = db.get_in_progress_owners().unwrap();
        assert!(in_progress.is_empty());
    }

    #[test]
    fn test_unexplored_owners() {
        let db = test_db();

        // Add repos from different owners
        db.add_repo_stub("owner1/repo1").unwrap();
        db.add_repo_stub("owner2/repo2").unwrap();
        db.add_repo_stub("owner3/repo3").unwrap();

        // All 3 owners should be unexplored
        let unexplored = db.get_unexplored_owners(None).unwrap();
        assert_eq!(unexplored.len(), 3);

        // Mark one as explored
        db.mark_owner_explored("owner1", 1).unwrap();

        // Now only 2 unexplored
        let unexplored = db.get_unexplored_owners(None).unwrap();
        assert_eq!(unexplored.len(), 2);
        assert!(!unexplored.contains(&"owner1".to_string()));
    }

    #[test]
    fn test_count_unexplored_owners() {
        let db = test_db();

        db.add_repo_stub("owner1/repo1").unwrap();
        db.add_repo_stub("owner2/repo2").unwrap();

        assert_eq!(db.count_unexplored_owners().unwrap(), 2);

        db.mark_owner_explored("owner1", 1).unwrap();
        assert_eq!(db.count_unexplored_owners().unwrap(), 1);
    }

    // === Follower Tracking ===

    #[test]
    fn test_follower_tracking() {
        let db = test_db();

        // Initially not fetched
        assert!(!db.is_owner_followers_fetched("testowner").unwrap());

        // Mark in progress
        db.mark_owner_followers_in_progress("testowner").unwrap();
        assert!(!db.is_owner_followers_fetched("testowner").unwrap());

        // Check in_progress list
        let in_progress = db.get_in_progress_followers_fetch().unwrap();
        assert_eq!(in_progress.len(), 1);

        // Mark as done
        db.mark_owner_followers_fetched("testowner", 100).unwrap();
        assert!(db.is_owner_followers_fetched("testowner").unwrap());
    }

    #[test]
    fn test_add_follower_as_owner() {
        let db = test_db();

        // Add a new follower
        let added = db.add_follower_as_owner("newfollower").unwrap();
        assert!(added);

        // Adding again should skip (placeholder exists)
        let added_again = db.add_follower_as_owner("newfollower").unwrap();
        assert!(!added_again);
    }

    // === Gone Repos ===

    #[test]
    fn test_mark_as_gone() {
        let db = test_db();

        db.add_repo_stub("owner/existing").unwrap();
        db.add_repo_stub("owner/deleted").unwrap();

        let marked = db.mark_as_gone_bulk(&["owner/deleted".to_string()]).unwrap();
        assert_eq!(marked, 1);

        let gone_count = db.count_gone().unwrap();
        assert_eq!(gone_count, 1);

        // Gone repos shouldn't appear in repos needing metadata
        let need_meta = db.get_repos_without_metadata(None).unwrap();
        assert_eq!(need_meta.len(), 1);
        assert_eq!(need_meta[0], "owner/existing");
    }

    #[test]
    fn test_mark_as_gone_case_insensitive() {
        let db = test_db();

        db.add_repo_stub("Owner/Repo").unwrap();

        // Mark with different case
        let marked = db.mark_as_gone_bulk(&["owner/repo".to_string()]).unwrap();
        assert_eq!(marked, 1);

        let gone_count = db.count_gone().unwrap();
        assert_eq!(gone_count, 1);
    }

    // === Owners to Explore ===

    #[test]
    fn test_owners_to_explore() {
        let db = test_db();

        // Add a follower as owner (should go to owners_to_explore table)
        let added = db.add_follower_as_owner("newuser").unwrap();
        assert!(added);

        // Adding again should return false
        let added_again = db.add_follower_as_owner("newuser").unwrap();
        assert!(!added_again);

        // Count should reflect the addition
        let count = db.count_owners_to_explore().unwrap();
        assert_eq!(count, 1);

        // Should appear in unexplored owners
        let unexplored = db.get_unexplored_owners(None).unwrap();
        assert!(unexplored.contains(&"newuser".to_string()));
    }

    #[test]
    fn test_owners_to_explore_cleaned_on_explore() {
        let db = test_db();

        // Add a follower as owner
        db.add_follower_as_owner("testuser").unwrap();
        assert_eq!(db.count_owners_to_explore().unwrap(), 1);

        // Mark as explored - should clean up from owners_to_explore
        db.mark_owner_explored("testuser", 5).unwrap();
        assert_eq!(db.count_owners_to_explore().unwrap(), 0);

        // Should no longer appear in unexplored owners
        let unexplored = db.get_unexplored_owners(None).unwrap();
        assert!(!unexplored.contains(&"testuser".to_string()));
    }

    // === Papers ===

    #[test]
    fn test_upsert_paper() {
        let db = test_db();

        let paper_id = db.upsert_paper(
            "https://arxiv.org/abs/2301.12345",
            "arxiv.org",
            Some("2301.12345"),
            None,
        ).unwrap();

        assert!(paper_id > 0);

        // Upserting same URL should return same ID
        let paper_id2 = db.upsert_paper(
            "https://arxiv.org/abs/2301.12345",
            "arxiv.org",
            Some("2301.12345"),
            None,
        ).unwrap();

        assert_eq!(paper_id, paper_id2);

        let count = db.count_papers().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_paper_sources() {
        let db = test_db();

        // Add a repo
        db.add_repo_stub("owner/repo").unwrap();

        // Add a paper
        let paper_id = db.upsert_paper(
            "https://arxiv.org/abs/2301.12345",
            "arxiv.org",
            Some("2301.12345"),
            None,
        ).unwrap();

        // Link paper to repo (repo_id = 1 since it's the first)
        db.add_paper_source(paper_id, 1, Some("Found in README")).unwrap();

        let source_count = db.count_paper_sources().unwrap();
        assert_eq!(source_count, 1);

        // Adding same link again should be ignored (ON CONFLICT)
        db.add_paper_source(paper_id, 1, Some("Duplicate")).unwrap();
        let source_count = db.count_paper_sources().unwrap();
        assert_eq!(source_count, 1);
    }

    // === Distinct Owners ===

    #[test]
    fn test_count_distinct_owners() {
        let db = test_db();

        db.add_repo_stub("owner1/repo1").unwrap();
        db.add_repo_stub("owner1/repo2").unwrap();
        db.add_repo_stub("owner2/repo1").unwrap();

        let count = db.count_distinct_owners().unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_owners_to_explore_not_in_distinct_count() {
        let db = test_db();

        db.add_repo_stub("owner1/repo1").unwrap();
        db.add_follower_as_owner("owner2").unwrap(); // Goes to owners_to_explore, not repos

        // owner2 is in owners_to_explore but has no repos yet
        // count_distinct_owners should only count owners with real repos
        let count = db.count_distinct_owners().unwrap();
        assert_eq!(count, 1); // Only owner1 has real repos

        // But owner2 should appear in unexplored owners
        let unexplored = db.get_unexplored_owners(None).unwrap();
        assert!(unexplored.contains(&"owner2".to_string()));
    }
}
