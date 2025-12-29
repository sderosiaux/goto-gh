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

-- Repos needing metadata fetch (for GraphQL backfill)
SELECT COUNT(*) FROM repos
WHERE gone = 0 AND (stars IS NULL OR stars = 0)
  AND description IS NULL AND language IS NULL;

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

## Data Flow

### Commands & What They Do

| Command | API | What it captures | Target |
|---------|-----|------------------|--------|
| `discover` | REST | Full metadata (stars, desc, lang, topics) | New repos from owners |
| `fetch` | GraphQL | Metadata only (no README) | Old repos without metadata |
| `fetch-missing-readmes` | REST | README only (rebuilds embedded_text) | Repos with metadata but no README |
| `embed` | OpenAI | Vector embeddings | Repos with embedded_text but no embedding |

### Pipeline

```
discover (REST)          fetch (GraphQL)
     │                        │
     ▼                        ▼
┌─────────────────────────────────────┐
│  repos with metadata (stars, desc)  │
│  embedded_text = name|desc|topics   │
└─────────────────────────────────────┘
                  │
                  ▼
        fetch-missing-readmes (REST)
                  │
                  ▼
┌─────────────────────────────────────┐
│  repos with README                  │
│  embedded_text = name|desc|...|README│
└─────────────────────────────────────┘
                  │
                  ▼
              embed (OpenAI)
                  │
                  ▼
┌─────────────────────────────────────┐
│  repos with vector embeddings       │
│  has_embedding = 1                  │
└─────────────────────────────────────┘
```

### Key Conditions

- **`discover`**: Adds new repos OR updates existing repos without metadata
- **`fetch` (GraphQL)**: Targets `WHERE stars IS NULL AND description IS NULL AND language IS NULL`
- **`fetch-missing-readmes`**: Targets `WHERE embedded_text IS NOT NULL AND readme_excerpt IS NULL`
- **`embed`**: Targets `WHERE has_embedding = 0 AND embedded_text IS NOT NULL`

## Architecture

### Server Mode (`goto-gh server`)

Runs 3 concurrent workers:
- **fetch**: Fetches repo metadata via GraphQL (backfill old repos)
- **discover**: Explores owner profiles, finds new repos with full metadata (REST)
- **embed**: Generates vector embeddings for semantic search

### Key Modules

- `src/github.rs` - GitHub API client (GraphQL + REST), includes `rest_get()` helper
- `src/db.rs` - SQLite operations, `save_discovered_repos()` for full metadata
- `src/server.rs` - Concurrent worker orchestration
- `src/embed_core.rs` - Embedding generation
- `src/discovery.rs` - Profile/repo discovery with full metadata capture
- `src/fetch.rs` - GraphQL metadata fetching (backfill)

### Environment Variables

```bash
GITHUB_TOKENS=token1,token2,token3  # Comma-separated for parallel workers
GITHUB_TOKEN=single_token           # Fallback
OPENAI_API_KEY=sk-...               # For OpenAI embeddings
```

## GraphQL Query

Current query fetches per repo (metadata only, no README):
- `nameWithOwner`, `description`, `stargazerCount`
- `primaryLanguage`, `repositoryTopics` (first 10)
- `pushedAt`, `createdAt`

README is fetched separately via `fetch-missing-readmes` (REST API) which handles all README variants (README.md, readme.md, README.rst, etc.).

## Exploration Commands

Three commands for discovering repos through semantic similarity:

### Walk - Random Walk Through Embedding Space

Traverse the embedding space by hopping between semantically similar repos. Each step selects a weighted random neighbor, favoring closer repos but allowing exploration.

```bash
# Start from a specific repo
goto-gh walk facebook/react --steps 5 --breadth 10

# Start from a random repo
goto-gh walk --random-start --steps 10
```

**Options:**
- `--steps N` - Number of hops (default: 5)
- `--breadth N` - Candidates to consider per step (default: 10)
- `--random-start` - Start from a random embedded repo

**Use case:** Serendipitous discovery, exploring semantic neighborhoods

### Underrated - Find Hidden Gems

Find repos semantically similar to popular ones but with fewer stars. Uses an "underrated score" = similarity / log(stars + 1) to surface overlooked projects.

```bash
# Find gems similar to a specific popular repo
goto-gh underrated --reference facebook/react --max-stars 500

# Sample from top popular repos and find their underrated alternatives
goto-gh underrated --sample 50 --min-sim 0.5 --max-stars 1000
```

**Options:**
- `--reference <repo>` - Specific popular repo to find alternatives for
- `--sample N` - Number of popular repos to sample (default: 50)
- `--min-sim F` - Minimum similarity threshold 0-1 (default: 0.75)
- `--max-stars N` - Maximum stars for a repo to be "underrated" (default: 500)
- `--limit N` - Results per reference (default: 5)

**Use case:** Finding quality alternatives to well-known projects

### Cross - Cross-Pollination Detector

Find repos at the intersection of two topics/domains. Uses harmonic mean to reward balance between both topics.

```bash
# Find repos combining ML and music
goto-gh cross "machine learning" "music" --min-each 0.2

# Find Rust + WebAssembly projects
goto-gh cross "rust systems programming" "webassembly browser" --limit 20
```

**Options:**
- `--min-each F` - Minimum similarity to each topic (default: 0.5)
- `--limit N` - Number of results (default: 20)

**Use case:** Discovering innovative projects that bridge different domains

### Cluster Map - Visual Embedding Space

Generate an interactive HTML visualization of repo embeddings using t-SNE dimensionality reduction.

```bash
# Generate cluster map (default 5000 repos)
goto-gh cluster-map -o clusters.html

# With filters
goto-gh cluster-map -s 10000 --min-stars 100 -o popular_clusters.html
goto-gh cluster-map -s 20000 --language Python -o python_clusters.html

# Tune t-SNE parameters
goto-gh cluster-map -s 5000 --perplexity 50 --epochs 500 -o tuned_clusters.html
```

**Options:**
- `-s, --sample N` - Number of repos to sample (default: 5000)
- `--min-stars N` - Minimum stars filter (default: 0)
- `--language X` - Filter by language
- `--perplexity F` - t-SNE perplexity parameter (default: 30)
- `--epochs N` - t-SNE iterations (default: 1000)
- `-o, --output FILE` - Output HTML file (default: cluster_map.html)

**Use case:** Visual exploration of repo clusters, finding thematic communities
