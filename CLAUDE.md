# goto-gh

Semantic search for GitHub repositories using local vector embeddings.

## Database

SQLite database location:
```
~/Library/Application Support/dev.goto-gh.goto-gh/repos.db
```

Query shortcut:
```bash
sqlite3 "/Users/sderosiaux/Library/Application Support/dev.goto-gh.goto-gh/repos.db" "YOUR QUERY"
```

### Schema

```sql
-- Main repos table
CREATE TABLE repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT UNIQUE NOT NULL,      -- e.g. "torvalds/linux"
    description TEXT,
    url TEXT NOT NULL,                    -- html_url
    stars INTEGER DEFAULT 0,
    language TEXT,                        -- primary language
    topics TEXT,                          -- comma-separated
    readme_excerpt TEXT,                  -- first ~2000 chars of README
    embedded_text TEXT,                   -- full text used for embedding (name|desc|topics|lang|readme)
    last_indexed TEXT NOT NULL,           -- ISO timestamp
    pushed_at TEXT,
    created_at TEXT,
    gone INTEGER DEFAULT 0,               -- 1 if repo deleted/private
    owner TEXT,                           -- extracted from full_name
    has_embedding INTEGER DEFAULT 0       -- 1 if vector exists in repo_embeddings
);

-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE repo_embeddings USING vec0(
    repo_id INTEGER PRIMARY KEY,
    embedding FLOAT[1536]                 -- OpenAI text-embedding-3-small dimension
);

-- Discovery queue
CREATE TABLE owners_to_explore (
    owner TEXT PRIMARY KEY,
    added_at INTEGER NOT NULL,
    source TEXT                           -- how we found this owner
);

CREATE TABLE explored_owners (
    owner TEXT PRIMARY KEY,
    repo_count INTEGER DEFAULT 0,
    explored_at INTEGER NOT NULL,
    status TEXT DEFAULT 'done'
);

-- Papers extracted from READMEs
CREATE TABLE papers (
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
);
```

### Common Queries

```sql
-- Stats
SELECT COUNT(*) FROM repos WHERE gone = 0;
SELECT COUNT(*) FROM repos WHERE has_embedding = 1;

-- Repos without README (for fetch-missing-readmes)
SELECT full_name FROM repos
WHERE gone = 0
  AND embedded_text IS NOT NULL
  AND (readme_excerpt IS NULL OR readme_excerpt = '');

-- Repos needing metadata fetch
SELECT COUNT(*) FROM repos WHERE gone = 0 AND embedded_text IS NULL;

-- Repos needing embeddings
SELECT COUNT(*) FROM repos
WHERE has_embedding = 0 AND gone = 0 AND embedded_text IS NOT NULL;

-- Unexplored owners in queue
SELECT COUNT(*) FROM owners_to_explore;

-- Search by name
SELECT full_name, stars, language FROM repos
WHERE lower(full_name) LIKE '%query%'
ORDER BY stars DESC LIMIT 10;
```

## Architecture

### Server Mode (`goto-gh server`)

Runs 3 concurrent workers:
- **fetch**: Fetches repo metadata + README via GraphQL (1 worker per token)
- **discover**: Explores owner profiles, finds new repos/followers
- **embed**: Generates vector embeddings for semantic search

### Key Modules

- `src/github.rs` - GitHub API client (GraphQL + REST)
- `src/db.rs` - SQLite operations
- `src/server.rs` - Concurrent worker orchestration
- `src/embed_core.rs` - Embedding generation
- `src/discovery.rs` - Profile/repo discovery
- `src/fetch.rs` - Metadata fetching

### Environment Variables

```bash
GITHUB_TOKENS=token1,token2,token3  # Comma-separated for parallel workers
GITHUB_TOKEN=single_token           # Fallback
OPENAI_API_KEY=sk-...               # For OpenAI embeddings
```

## GraphQL Query

Current query fetches per repo:
- `nameWithOwner`, `description`, `stargazerCount`
- `primaryLanguage`, `repositoryTopics` (first 10)
- `pushedAt`, `createdAt`
- `README.md` content via `object(expression: "HEAD:README.md")`

Note: Only fetches `README.md` (uppercase). Repos with other README names (readme.md, README, README.rst) will have null readme_excerpt.
