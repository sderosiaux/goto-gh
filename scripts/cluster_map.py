#!/usr/bin/env python3
"""
Cluster Map - Visualize GitHub repo embeddings using UMAP dimensionality reduction.

This script samples embeddings from the goto-gh database, reduces them to 2D using UMAP,
and creates an interactive visualization showing clusters of semantically similar repos.

Usage:
    python scripts/cluster_map.py --sample 10000 --output cluster_map.html
    python scripts/cluster_map.py --sample 50000 --min-stars 100 --output popular_clusters.html

Requirements:
    pip install umap-learn plotly pandas numpy sqlite3
"""

import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_db_path() -> Path:
    """Get the default database path."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".local" / "share"
    return base / "dev.goto-gh.goto-gh" / "repos.db"


def parse_embedding(blob: bytes) -> np.ndarray:
    """Parse a binary embedding blob into numpy array."""
    n_floats = len(blob) // 4
    return np.array(struct.unpack(f"{n_floats}f", blob), dtype=np.float32)


def sample_repos(
    db_path: Path,
    sample_size: int,
    min_stars: int = 0,
    language: str = None
) -> pd.DataFrame:
    """Sample repos with embeddings from the database."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            r.id,
            r.full_name,
            r.description,
            r.stars,
            r.language,
            r.topics,
            e.embedding
        FROM repos r
        JOIN repo_embeddings e ON r.id = e.repo_id
        WHERE r.gone = 0
          AND r.stars >= ?
    """
    params = [min_stars]

    if language:
        query += " AND r.language = ?"
        params.append(language)

    query += f" ORDER BY RANDOM() LIMIT ?"
    params.append(sample_size)

    print(f"Sampling {sample_size} repos from {db_path}...")

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    data = []
    embeddings = []

    for row in rows:
        repo_id, full_name, description, stars, lang, topics, emb_blob = row

        embedding = parse_embedding(emb_blob)
        embeddings.append(embedding)

        data.append({
            "id": repo_id,
            "full_name": full_name,
            "description": description or "",
            "stars": stars,
            "language": lang or "Unknown",
            "topics": topics or "",
        })

    df = pd.DataFrame(data)
    df["embedding"] = embeddings

    print(f"Loaded {len(df)} repos with embeddings")
    return df


def reduce_dimensions(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP."""
    try:
        import umap
    except ImportError:
        print("Error: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
        verbose=True
    )

    coords = reducer.fit_transform(embeddings)
    print("UMAP complete")
    return coords


def create_visualization(
    df: pd.DataFrame,
    coords: np.ndarray,
    output_path: str,
    title: str = "GitHub Repository Clusters"
):
    """Create an interactive Plotly visualization."""
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Create hover text
    df["hover_text"] = df.apply(
        lambda r: f"<b>{r['full_name']}</b><br>" +
                  f"Stars: {r['stars']:,}<br>" +
                  f"Language: {r['language']}<br>" +
                  f"{r['description'][:100]}...",
        axis=1
    )

    # Use log scale for star sizes (with minimum)
    df["point_size"] = np.log1p(df["stars"]) * 2 + 3

    # Get top languages for color coding
    top_languages = df["language"].value_counts().head(15).index.tolist()
    df["language_group"] = df["language"].apply(
        lambda x: x if x in top_languages else "Other"
    )

    print(f"Creating visualization with {len(df)} points...")

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="language_group",
        size="point_size",
        hover_name="full_name",
        hover_data={
            "stars": True,
            "language": True,
            "description": True,
            "x": False,
            "y": False,
            "point_size": False,
            "language_group": False,
        },
        title=title,
        labels={"language_group": "Language"},
    )

    fig.update_layout(
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, title=""),
        plot_bgcolor="white",
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color="white"),
            opacity=0.7
        )
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Visualization saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GitHub repo embeddings using UMAP"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to repos.db (default: auto-detect)"
    )
    parser.add_argument(
        "--sample", type=int, default=10000,
        help="Number of repos to sample (default: 10000)"
    )
    parser.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum stars filter (default: 0)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Filter by language (e.g., Python, Rust)"
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="UMAP n_neighbors parameter (default: 15)"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1,
        help="UMAP min_dist parameter (default: 0.1)"
    )
    parser.add_argument(
        "--output", type=str, default="cluster_map.html",
        help="Output HTML file (default: cluster_map.html)"
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Chart title"
    )

    args = parser.parse_args()

    # Get database path
    db_path = Path(args.db) if args.db else get_db_path()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Sample repos
    df = sample_repos(
        db_path,
        sample_size=args.sample,
        min_stars=args.min_stars,
        language=args.language
    )

    if len(df) == 0:
        print("No repos found matching criteria")
        sys.exit(1)

    # Stack embeddings
    embeddings = np.stack(df["embedding"].values)
    print(f"Embedding shape: {embeddings.shape}")

    # Reduce dimensions
    coords = reduce_dimensions(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    # Create title
    title = args.title or f"GitHub Repository Clusters ({len(df):,} repos)"
    if args.min_stars > 0:
        title += f" - {args.min_stars}+ stars"
    if args.language:
        title += f" - {args.language}"

    # Create visualization
    create_visualization(df, coords, args.output, title)

    print("\nDone! Open the HTML file in a browser to explore clusters.")


if __name__ == "__main__":
    main()
