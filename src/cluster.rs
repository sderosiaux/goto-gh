//! Cluster map visualization using t-SNE dimensionality reduction.
//!
//! Generates an interactive HTML visualization of repo embeddings.

use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;

use crate::db::Database;

/// Configuration for cluster map generation
pub struct ClusterConfig {
    pub sample: usize,
    pub min_stars: u64,
    pub language: Option<String>,
    pub perplexity: f32,
    pub epochs: usize,
}

/// Repo data for visualization
struct RepoPoint {
    full_name: String,
    description: String,
    stars: u64,
    language: String,
    x: f32,
    y: f32,
}

/// Generate cluster map HTML
pub fn generate_cluster_map(db: &Database, config: &ClusterConfig, output: &str) -> Result<()> {
    eprintln!("\x1b[36m..\x1b[0m Sampling {} repos...", config.sample);

    // Get embeddings from database
    let repos = db.export_embeddings(
        Some(config.sample),
        config.min_stars,
        config.language.as_deref(),
    )?;

    if repos.is_empty() {
        anyhow::bail!("No repos with embeddings found");
    }

    eprintln!(
        "\x1b[36m..\x1b[0m Loaded {} repos, running t-SNE (perplexity={}, epochs={})...",
        repos.len(),
        config.perplexity,
        config.epochs
    );

    // Prepare embeddings for t-SNE (f32)
    let samples: Vec<Vec<f32>> = repos.iter().map(|r| r.embedding.clone()).collect();

    // Run Barnes-Hut t-SNE
    let mut tsne = bhtsne::tSNE::new(&samples);
    tsne.embedding_dim(2)
        .perplexity(config.perplexity)
        .epochs(config.epochs)
        .barnes_hut(0.5, |a: &Vec<f32>, b: &Vec<f32>| -> f32 {
            // Euclidean distance
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    eprintln!("\x1b[36m..\x1b[0m t-SNE complete");

    // Extract coordinates
    let coords: Vec<f32> = tsne.embedding().to_vec();

    // Build visualization data
    let points: Vec<RepoPoint> = repos
        .iter()
        .enumerate()
        .map(|(i, repo)| RepoPoint {
            full_name: repo.full_name.clone(),
            description: repo.description.clone().unwrap_or_default(),
            stars: repo.stars,
            language: repo.language.clone().unwrap_or_else(|| "Unknown".to_string()),
            x: coords[i * 2],
            y: coords[i * 2 + 1],
        })
        .collect();

    eprintln!("\x1b[36m..\x1b[0m Generating HTML...");

    // Count languages for color assignment
    let mut lang_counts: HashMap<&str, usize> = HashMap::new();
    for p in &points {
        *lang_counts.entry(&p.language).or_insert(0) += 1;
    }
    let mut langs: Vec<_> = lang_counts.into_iter().collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1));
    let top_langs: Vec<&str> = langs.iter().take(15).map(|(l, _)| *l).collect();

    // Generate HTML
    let html = generate_html(&points, &top_langs, config.sample);

    let mut file = std::fs::File::create(output)?;
    file.write_all(html.as_bytes())?;

    eprintln!("\x1b[32mok\x1b[0m Saved cluster map to {}", output);

    Ok(())
}

/// Escape a string for safe embedding in JSON
fn escape_json_string(s: &str) -> String {
    s.chars()
        .take(100)
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            c if c.is_control() => vec![' '],
            c => vec![c],
        })
        .collect()
}

fn generate_html(points: &[RepoPoint], top_langs: &[&str], sample_size: usize) -> String {
    // Color palette for languages
    let colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    ];

    // Build data JSON
    let mut data_json = String::from("[");
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            data_json.push(',');
        }
        let lang_group = if top_langs.contains(&p.language.as_str()) {
            &p.language
        } else {
            "Other"
        };
        let desc_escaped = escape_json_string(&p.description);
        let name_escaped = escape_json_string(&p.full_name);
        let lang_escaped = escape_json_string(&p.language);
        let lang_group_escaped = escape_json_string(lang_group);
        data_json.push_str(&format!(
            r#"{{"x":{:.4},"y":{:.4},"name":"{}","desc":"{}","stars":{},"lang":"{}","langGroup":"{}"}}"#,
            p.x, p.y, name_escaped, desc_escaped, p.stars, lang_escaped, lang_group_escaped
        ));
    }
    data_json.push(']');

    // Build color map JSON
    let mut color_map = String::from("{");
    for (i, lang) in top_langs.iter().enumerate() {
        if i > 0 {
            color_map.push(',');
        }
        let lang_escaped = escape_json_string(lang);
        color_map.push_str(&format!("\"{}\":\"{}\"", lang_escaped, colors[i % colors.len()]));
    }
    color_map.push_str(",\"Other\":\"#cccccc\"}");

    format!(
        r##"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GitHub Repository Clusters ({sample_size} repos)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #chart {{ width: 100%; height: calc(100vh - 60px); }}
        h1 {{ margin: 0 0 10px 0; font-size: 18px; color: #333; }}
    </style>
</head>
<body>
    <h1>GitHub Repository Clusters ({sample_size} repos)</h1>
    <div id="chart"></div>
    <script>
        const data = {data_json};
        const colorMap = {color_map};

        // Group by language
        const groups = {{}};
        data.forEach(d => {{
            const g = d.langGroup;
            if (!groups[g]) groups[g] = {{x: [], y: [], text: [], customdata: [], name: g}};
            groups[g].x.push(d.x);
            groups[g].y.push(d.y);
            groups[g].text.push(d.name);
            groups[g].customdata.push(d);
        }});

        const traces = Object.entries(groups).map(([name, g]) => ({{
            x: g.x,
            y: g.y,
            text: g.text,
            customdata: g.customdata,
            name: name,
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: g.customdata.map(d => Math.log10(d.stars + 1) * 3 + 4),
                color: colorMap[name] || '#cccccc',
                opacity: 0.7,
                line: {{ width: 0.5, color: 'white' }}
            }},
            hovertemplate: '<b>%{{text}}</b><br>Stars: %{{customdata.stars}}<br>Language: %{{customdata.lang}}<br>%{{customdata.desc}}<extra></extra>'
        }}));

        const layout = {{
            showlegend: true,
            legend: {{ orientation: 'v', x: 1.02, y: 1 }},
            xaxis: {{ showgrid: false, showticklabels: false, title: '' }},
            yaxis: {{ showgrid: false, showticklabels: false, title: '' }},
            plot_bgcolor: 'white',
            margin: {{ l: 20, r: 150, t: 20, b: 20 }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('chart', traces, layout, {{responsive: true}});

        // Click to open repo
        document.getElementById('chart').on('plotly_click', function(data) {{
            const name = data.points[0].text;
            window.open('https://github.com/' + name, '_blank');
        }});
    </script>
</body>
</html>"##
    )
}
