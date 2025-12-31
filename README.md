# goto-gh

Semantic search for GitHub repositories using local vector embeddings.

## Why

GitHub's search is keyword-based. You search for "vector database" and get repos that literally contain those words. But what if you want to find repos that are *conceptually similar* - like similarity search engines, embedding stores, or nearest neighbor libraries?

`goto-gh` indexes GitHub repos locally with semantic embeddings, letting you search by meaning, not just keywords.

## What

- **Semantic search**: Find repos by concept, not just text matching
- **Local-first**: All data stored locally in SQLite + vector index
- **Fast**: Sub-second search across hundreds of thousands of repos
- **Dual embedding support**: Local E5 model (384d) or OpenAI text-embedding-3-small (1536d)
- **Offline**: Search works without internet after indexing
- **Server mode**: Continuous discovery and indexing daemon

## How it works

```
GitHub API
    │
    ├── REST API (discover) ───────────┐
    │   Full metadata + owner discovery │
    │                                   ▼
    ├── GraphQL (fetch) ────────▶ ┌───────────────────┐
    │   Metadata backfill         │  repos with       │
    │                             │  stars, desc,     │
    │                             │  topics, lang     │
    │                             └───────────────────┘
    │                                   │
    └── REST API (readme) ──────────────┤
        /repos/{owner}/{repo}/readme    │
                                        ▼
                               ┌───────────────────┐
                               │  Build embedding  │
                               │  text             │──▶ name | desc | topics | lang | readme
                               └───────────────────┘
                                        │
                                        ▼
                               ┌───────────────────┐
                               │  Generate vector  │──▶ E5 (384d) or OpenAI (1536d)
                               │  embedding        │
                               └───────────────────┘
                                        │
                                        ▼
                               ┌───────────────────┐
                               │  Store in SQLite  │──▶ repos + sqlite-vec embeddings
                               └───────────────────┘
```

## Installation

```bash
# Clone and build
git clone https://github.com/sderosiaux/goto-gh
cd goto-gh
cargo build --release

# Add to PATH or create alias
alias goto-gh='./target/release/goto-gh'
```

## Requirements

- Rust 1.70+
- GitHub token (for API access): `export GITHUB_TOKEN=...` or `export GITHUB_TOKENS=token1,token2,token3`
- OpenAI API key (optional, for OpenAI embeddings): `export OPENAI_API_KEY=...`

## Usage

### Semantic Search

```bash
# Basic semantic search
goto-gh "vector database"
goto-gh "web framework rust async"
goto-gh "kubernetes deployment tool"

# With options
goto-gh "machine learning" --limit 20
goto-gh "graph database" --semantic        # pure semantic, no name boosting
goto-gh "real-time analytics" --expand     # LLM query expansion
```

### Building the Index

The indexing pipeline has multiple stages:

```bash
# 1. Discover repos from GitHub users (REST API)
#    Fetches full metadata + discovers followers
goto-gh discover --concurrency 5

# 2. Fetch metadata for repos without it (GraphQL - backfill)
goto-gh fetch --batch-size 300 --concurrency 2

# 3. Fetch README content for repos (REST API)
goto-gh fetch-missing-readmes --concurrency 5

# 4. Generate embeddings
goto-gh embed                           # local E5 model (default)
goto-gh embed --provider openai         # OpenAI embeddings
goto-gh embed --reset --provider openai # re-embed everything with OpenAI
```

### Server Mode (Recommended)

Run all workers continuously as a daemon:

```bash
# Start server with all workers
goto-gh server

# With custom settings
goto-gh server \
  --fetch-batch-size 300 \
  --fetch-concurrency 2 \
  --readme-concurrency 20 \
  --discover-limit 50 \
  --provider openai
```

Workers:
- **fetch** (GraphQL): Fetches metadata for repos in queue
- **readme** (REST): Fetches README content
- **discover** (REST): Explores owner profiles, finds new repos and followers
- **embed**: Generates vector embeddings

### Other Commands

```bash
# Add a specific repo
goto-gh add facebook/react

# Load repo names from file (no API calls)
goto-gh load repos.txt

# Load usernames to discover
goto-gh load-users users.txt

# Fuzzy search by name
goto-gh find react --limit 20

# Show statistics
goto-gh stats

# Check API rate limit
goto-gh rate-limit

# Print database path
goto-gh db-path

# Extract paper links from READMEs
goto-gh extract-papers

# Extract linked repos from READMEs
goto-gh extract-repos
```

## Exploration Commands

### Walk - Random Walk Through Embedding Space

Traverse the embedding space by hopping between semantically similar repos.

```bash
goto-gh walk facebook/react --steps 5 --breadth 10
goto-gh walk --random-start --steps 10
```

### Underrated - Find Hidden Gems

Find repos semantically similar to popular ones but with fewer stars.

```bash
goto-gh underrated --reference facebook/react --max-stars 500
goto-gh underrated --sample 20 --min-sim 0.6 --max-stars 1000
```

### Cross - Cross-Pollination Detector

Find repos at the intersection of two topics/domains.

```bash
goto-gh cross "machine learning" "music" --min-each 0.4
goto-gh cross "rust systems programming" "webassembly browser" --limit 20
```

### Cluster Map - Visual Embedding Space

Generate an interactive HTML visualization using t-SNE.

```bash
goto-gh cluster-map -o clusters.html
goto-gh cluster-map -s 10000 --min-stars 100 -o popular.html
goto-gh cluster-map --language Python -o python.html
```

## Proxy Support

For high-volume discovery, you can use HTTP proxies:

```bash
# With proxy file (one ip:port per line)
goto-gh discover --proxy-file proxies.txt
goto-gh fetch-missing-readmes --proxy-file proxies.txt

# Force all requests through proxies (skip token)
goto-gh discover --proxy-file proxies.txt --force-proxy
```

## Data Storage

All data is stored locally:
- **macOS**: `~/Library/Application Support/dev.goto-gh.goto-gh/`
- **Linux**: `~/.local/share/goto-gh/`

Database: `repos.db` (SQLite with sqlite-vec extension)

## Tech Stack

- **[fastembed](https://github.com/Anush008/fastembed-rs)**: Local embedding generation (MultilingualE5Small, 384d)
- **[OpenAI API](https://openai.com)**: Optional cloud embeddings (text-embedding-3-small, 1536d)
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: Vector similarity search extension for SQLite
- **[reqwest](https://github.com/seanmonstar/reqwest)**: HTTP client for GitHub API
- **[clap](https://github.com/clap-rs/clap)**: CLI argument parsing
- **[bhtsne](https://github.com/frjnn/bhtsne)**: Barnes-Hut t-SNE for cluster visualization

## License

MIT
