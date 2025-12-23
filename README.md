# goto-gh

Semantic search for GitHub repositories using local vector embeddings.

## Why

GitHub's search is keyword-based. You search for "vector database" and get repos that literally contain those words. But what if you want to find repos that are *conceptually similar* - like similarity search engines, embedding stores, or nearest neighbor libraries?

`goto-gh` indexes GitHub repos locally with semantic embeddings, letting you search by meaning, not just keywords.

## What

- **Semantic search**: Find repos by concept, not just text matching
- **Local-first**: All data stored locally in SQLite + vector index
- **Fast**: Sub-second search across thousands of repos
- **Efficient**: GraphQL API fetches 100 repos + READMEs in one call
- **Offline**: Search works without internet after indexing

## How it works

```
GitHub API (GraphQL)
        │
        ▼
┌───────────────────┐
│  Fetch repos +    │
│  README content   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Build embedding  │──▶ name | description | topics | language | readme
│  text             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Generate vector  │──▶ MultilingualE5Small (384-dim)
│  embedding        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Store in SQLite  │──▶ repos table + sqlite-vec embeddings
│  + vector index   │
└───────────────────┘
```

**Search flow:**
1. Your query gets embedded using the same model
2. Vector similarity search finds closest repos
3. Results boosted by name matching
4. Display top N results

## Usage

```bash
# Index top GitHub repos by stars (default: 50k)
goto-gh index

# Index with custom count
goto-gh index --count 5000

# Index by search query
goto-gh index --query "machine learning"

# Semantic search
goto-gh "vector database"
goto-gh "web framework rust async"
goto-gh "kubernetes deployment tool"

# Add a specific repo
goto-gh add qdrant/qdrant

# Check stats
goto-gh stats

# Check API rate limit
goto-gh rate-limit
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
- GitHub token (for higher rate limits): `export GITHUB_TOKEN=...` or `gh auth login`

## Tech Stack

- **[fastembed](https://github.com/Anush008/fastembed-rs)**: Local embedding generation (MultilingualE5Small)
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: Vector similarity search extension for SQLite
- **[reqwest](https://github.com/seanmonstar/reqwest)**: HTTP client for GitHub API
- **[clap](https://github.com/clap-rs/clap)**: CLI argument parsing

## Data Storage

All data is stored locally:
- **macOS**: `~/Library/Application Support/dev.goto-gh.goto-gh/`
- **Linux**: `~/.local/share/goto-gh/`

## License

MIT
