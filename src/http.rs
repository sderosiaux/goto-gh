//! HTTP server for SQL explorer interface

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tower_http::cors::{Any, CorsLayer};

/// Shared state for the HTTP server
#[derive(Clone)]
struct AppState {
    db_path: PathBuf,
}

/// Request body for SQL queries
#[derive(Deserialize)]
struct QueryRequest {
    sql: String,
}

/// Response for SQL queries
#[derive(Serialize)]
struct QueryResponse {
    columns: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
    row_count: usize,
    error: Option<String>,
}

/// Start the HTTP server
pub async fn start_server(db_path: PathBuf, port: u16) -> Result<()> {
    let state = AppState { db_path };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/query", post(query_handler))
        .route("/api/schema", get(schema_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    eprintln!("\x1b[32mok\x1b[0m SQL Explorer running at http://{}", addr);
    eprintln!("    Press Ctrl+C to stop");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve the HTML interface
async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Execute SQL query
async fn query_handler(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let sql = req.sql.trim();

    // Basic safety check - only allow SELECT, PRAGMA, and EXPLAIN
    let sql_upper = sql.to_uppercase();
    let is_safe = sql_upper.starts_with("SELECT")
        || sql_upper.starts_with("PRAGMA")
        || sql_upper.starts_with("EXPLAIN")
        || sql_upper.starts_with("WITH");

    if !is_safe {
        return (
            StatusCode::BAD_REQUEST,
            Json(QueryResponse {
                columns: vec![],
                rows: vec![],
                row_count: 0,
                error: Some("Only SELECT, PRAGMA, EXPLAIN, and WITH queries are allowed".into()),
            }),
        );
    }

    match execute_query(&state.db_path, sql) {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(QueryResponse {
                columns: vec![],
                rows: vec![],
                row_count: 0,
                error: Some(e.to_string()),
            }),
        ),
    }
}

/// Table descriptions for the UI
fn get_table_description(name: &str) -> &'static str {
    match name {
        "repos" => "GitHub repositories with metadata (stars, language, topics, README excerpt)",
        "repo_embeddings" => "Vector embeddings (384-dim) for semantic search",
        "explored_owners" => "Owners whose repos have been fully fetched",
        "owners_to_explore" => "Queue of owners pending discovery",
        "owner_followers_fetched" => "Owners whose followers have been fetched",
        "papers" => "Academic papers extracted from READMEs (arxiv, doi)",
        "paper_sources" => "Links between papers and repos that reference them",
        _ => "",
    }
}

/// Get database schema with row counts
async fn schema_handler(State(state): State<AppState>) -> impl IntoResponse {
    let conn = match Connection::open(&state.db_path) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(QueryResponse {
                    columns: vec![],
                    rows: vec![],
                    row_count: 0,
                    error: Some(e.to_string()),
                }),
            );
        }
    };

    let sql = r#"
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
            AND name NOT LIKE 'sqlite_%'
            AND name NOT IN ('repo_embeddings_chunks', 'repo_embeddings_rowids', 'repo_embeddings_info')
            AND name NOT LIKE 'repo_embeddings_vector_chunks%'
        ORDER BY type, name
    "#;

    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(QueryResponse {
                    columns: vec![],
                    rows: vec![],
                    row_count: 0,
                    error: Some(e.to_string()),
                }),
            );
        }
    };

    let mut rows: Vec<Vec<serde_json::Value>> = Vec::new();

    let table_rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    });

    if let Ok(table_rows) = table_rows {
        for table_result in table_rows.flatten() {
            let (type_name, name, sql_def) = table_result;

            // Get row count for this table
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM \"{}\"", name), [], |r| r.get(0))
                .unwrap_or(0);

            let desc = get_table_description(&name);

            rows.push(vec![
                serde_json::json!(type_name),
                serde_json::json!(name),
                serde_json::json!(sql_def),
                serde_json::json!(desc),
                serde_json::json!(count),
            ]);
        }
    }

    let row_count = rows.len();

    (
        StatusCode::OK,
        Json(QueryResponse {
            columns: vec![
                "type".into(),
                "name".into(),
                "sql".into(),
                "description".into(),
                "count".into(),
            ],
            rows,
            row_count,
            error: None,
        }),
    )
}

/// Execute a SQL query and return results
fn execute_query(db_path: &PathBuf, sql: &str) -> Result<QueryResponse> {
    let conn = Connection::open(db_path)?;

    // Set pragmas for read-only safety
    conn.execute_batch(
        "PRAGMA query_only = ON;
         PRAGMA temp_store = MEMORY;",
    )?;

    let mut stmt = conn.prepare(sql)?;
    let column_count = stmt.column_count();
    let columns: Vec<String> = (0..column_count)
        .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
        .collect();

    let mut rows: Vec<Vec<serde_json::Value>> = Vec::new();

    let mut result_rows = stmt.query([])?;
    while let Some(row) = result_rows.next()? {
        let mut values: Vec<serde_json::Value> = Vec::new();
        for i in 0..column_count {
            let value = row.get_ref(i)?;
            let json_value = match value {
                rusqlite::types::ValueRef::Null => serde_json::Value::Null,
                rusqlite::types::ValueRef::Integer(i) => serde_json::json!(i),
                rusqlite::types::ValueRef::Real(f) => serde_json::json!(f),
                rusqlite::types::ValueRef::Text(s) => {
                    let text = String::from_utf8_lossy(s);
                    // Truncate very long text for display (UTF-8 safe)
                    if text.len() > 500 {
                        let truncated: String = text.chars().take(200).collect();
                        serde_json::json!(format!("{}...", truncated))
                    } else {
                        serde_json::json!(text)
                    }
                }
                rusqlite::types::ValueRef::Blob(b) => {
                    serde_json::json!(format!("<blob {} bytes>", b.len()))
                }
            };
            values.push(json_value);
        }
        rows.push(values);
    }

    let row_count = rows.len();

    Ok(QueryResponse {
        columns,
        rows,
        row_count,
        error: None,
    })
}

/// HTML for the SQL explorer interface
const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>goto-gh SQL Explorer</title>
    <style>
        :root {
            --bg: #fafafa;
            --surface: #ffffff;
            --border: #e5e5e5;
            --text: #171717;
            --text-muted: #737373;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --success: #16a34a;
            --error: #dc2626;
            --code-bg: #f5f5f5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        h1 {
            font-size: 20px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        h1 span {
            color: var(--text-muted);
            font-weight: 400;
        }

        .badge {
            font-size: 11px;
            padding: 2px 8px;
            background: var(--code-bg);
            border-radius: 4px;
            color: var(--text-muted);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 24px;
        }

        .sidebar {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: fit-content;
            max-height: calc(100vh - 140px);
            overflow-y: auto;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .sidebar h2 {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }

        .schema-item {
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.15s;
            margin-bottom: 4px;
        }

        .schema-item:hover {
            background: var(--code-bg);
        }

        .schema-item.active {
            background: var(--primary);
            color: white;
        }

        .schema-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }

        .schema-name {
            font-size: 13px;
            font-weight: 500;
            font-family: 'SF Mono', Monaco, monospace;
        }

        .schema-count {
            font-size: 11px;
            color: var(--text-muted);
            background: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
        }

        .schema-item.active .schema-count {
            background: rgba(255,255,255,0.2);
            color: rgba(255,255,255,0.9);
        }

        .schema-type {
            font-size: 11px;
            color: var(--text-muted);
        }

        .schema-item.active .schema-type {
            color: rgba(255,255,255,0.7);
        }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .query-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        .query-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            background: var(--code-bg);
        }

        .query-header h3 {
            font-size: 13px;
            font-weight: 500;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            padding: 16px;
            border: none;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 13px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
        }

        .query-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-top: 1px solid var(--border);
            background: var(--code-bg);
        }

        button {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
        }

        .btn-secondary {
            background: var(--code-bg);
            color: var(--text);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--border);
        }

        .status {
            font-size: 12px;
            color: var(--text-muted);
        }

        .status.success {
            color: var(--success);
        }

        .status.error {
            color: var(--error);
        }

        .results-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            flex: 1;
        }

        .results-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            background: var(--code-bg);
        }

        .results-header h3 {
            font-size: 13px;
            font-weight: 500;
        }

        .table-container {
            overflow: auto;
            max-height: 500px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th {
            position: sticky;
            top: 0;
            background: var(--code-bg);
            font-weight: 600;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }

        td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        tr:hover td {
            background: var(--code-bg);
        }

        .null-value {
            color: var(--text-muted);
            font-style: italic;
        }

        .empty-state {
            padding: 48px;
            text-align: center;
            color: var(--text-muted);
        }

        .empty-state h4 {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
        }

        .error-message {
            padding: 16px;
            background: #fef2f2;
            color: var(--error);
            font-family: monospace;
            font-size: 13px;
        }

        .quick-queries {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .quick-query {
            font-size: 11px;
            padding: 4px 10px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s;
        }

        .quick-query:hover {
            border-color: var(--primary);
            color: var(--primary);
        }

        kbd {
            font-size: 11px;
            padding: 2px 6px;
            background: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-family: inherit;
        }

        .table-description {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .table-desc-name {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            font-weight: 600;
            color: var(--primary);
            background: #eff6ff;
            padding: 4px 10px;
            border-radius: 4px;
        }

        .table-desc-text {
            font-size: 13px;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>goto-gh <span>SQL Explorer</span></h1>
            <span class="badge">Read-only</span>
        </header>

        <div class="main-grid">
            <aside class="sidebar">
                <div class="sidebar-header">
                    <h2>Tables</h2>
                    <span id="table-count" class="badge"></span>
                </div>
                <div id="schema-list">
                    <div class="empty-state">Loading...</div>
                </div>
            </aside>

            <main class="main-content">
                <div id="table-description" class="table-description" style="display: none;">
                    <span class="table-desc-name"></span>
                    <span class="table-desc-text"></span>
                </div>

                <section class="query-section">
                    <div class="query-header">
                        <h3>Query</h3>
                        <div class="quick-queries">
                            <span class="quick-query" data-query="SELECT * FROM repos ORDER BY stars DESC LIMIT 20">Top repos</span>
                            <span class="quick-query" data-query="SELECT owner, COUNT(*) as count FROM repos WHERE owner IS NOT NULL GROUP BY owner ORDER BY count DESC LIMIT 20">Top owners</span>
                            <span class="quick-query" data-query="SELECT language, COUNT(*) as count FROM repos WHERE language IS NOT NULL GROUP BY language ORDER BY count DESC">Languages</span>
                            <span class="quick-query" data-query="SELECT COUNT(*) as total, SUM(CASE WHEN stars IS NOT NULL THEN 1 ELSE 0 END) as with_metadata, SUM(CASE WHEN gone = 1 THEN 1 ELSE 0 END) as gone FROM repos">Stats</span>
                        </div>
                    </div>
                    <textarea id="sql-input" placeholder="SELECT * FROM repos LIMIT 10">SELECT * FROM repos ORDER BY stars DESC LIMIT 20</textarea>
                    <div class="query-footer">
                        <span class="status" id="status"></span>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="font-size: 12px; color: var(--text-muted);"><kbd>Ctrl</kbd>+<kbd>Enter</kbd> to run</span>
                            <button class="btn-primary" id="run-btn">Run Query</button>
                        </div>
                    </div>
                </section>

                <section class="results-section">
                    <div class="results-header">
                        <h3>Results</h3>
                        <span id="row-count" class="badge"></span>
                    </div>
                    <div id="results">
                        <div class="empty-state">
                            <h4>Run a query to see results</h4>
                            <p>Try one of the quick queries above or write your own SQL</p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    </div>

    <script>
        const sqlInput = document.getElementById('sql-input');
        const runBtn = document.getElementById('run-btn');
        const results = document.getElementById('results');
        const status = document.getElementById('status');
        const rowCount = document.getElementById('row-count');
        const schemaList = document.getElementById('schema-list');
        const tableDescEl = document.getElementById('table-description');
        const tableCountEl = document.getElementById('table-count');

        // Load schema on page load
        async function loadSchema() {
            try {
                const res = await fetch('/api/schema');
                const data = await res.json();

                if (data.error) {
                    schemaList.innerHTML = `<div class="error-message">${data.error}</div>`;
                    return;
                }

                let html = '';
                for (const row of data.rows) {
                    const [type, name, sql, description, count] = row;
                    const formattedCount = count >= 1000 ? (count / 1000).toFixed(1) + 'k' : count;
                    html += `
                        <div class="schema-item" data-sql="${escapeHtml(sql || '')}" data-name="${escapeHtml(name)}" data-description="${escapeHtml(description || '')}">
                            <div class="schema-row">
                                <span class="schema-name">${escapeHtml(name)}</span>
                                <span class="schema-count">${formattedCount}</span>
                            </div>
                            <div class="schema-type">${type}</div>
                        </div>
                    `;
                }
                schemaList.innerHTML = html;
                tableCountEl.textContent = data.rows.length;

                // Click to show table contents
                schemaList.querySelectorAll('.schema-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const name = item.dataset.name;
                        const description = item.dataset.description;

                        document.querySelectorAll('.schema-item').forEach(i => i.classList.remove('active'));
                        item.classList.add('active');

                        // Show table description
                        if (description) {
                            tableDescEl.querySelector('.table-desc-name').textContent = name;
                            tableDescEl.querySelector('.table-desc-text').textContent = description;
                            tableDescEl.style.display = 'flex';
                        } else {
                            tableDescEl.style.display = 'none';
                        }

                        sqlInput.value = `SELECT * FROM ${name} LIMIT 100`;
                        runQuery();
                    });
                });
            } catch (err) {
                schemaList.innerHTML = `<div class="error-message">${err.message}</div>`;
            }
        }

        // Run SQL query
        async function runQuery() {
            const sql = sqlInput.value.trim();
            if (!sql) return;

            status.textContent = 'Running...';
            status.className = 'status';
            rowCount.textContent = '';

            try {
                const start = performance.now();
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sql })
                });
                const elapsed = Math.round(performance.now() - start);
                const data = await res.json();

                if (data.error) {
                    status.textContent = 'Error';
                    status.className = 'status error';
                    results.innerHTML = `<div class="error-message">${escapeHtml(data.error)}</div>`;
                    return;
                }

                status.textContent = `${elapsed}ms`;
                status.className = 'status success';
                rowCount.textContent = `${data.row_count} rows`;

                if (data.rows.length === 0) {
                    results.innerHTML = `
                        <div class="empty-state">
                            <h4>No results</h4>
                            <p>Query returned 0 rows</p>
                        </div>
                    `;
                    return;
                }

                let tableHtml = '<div class="table-container"><table><thead><tr>';
                for (const col of data.columns) {
                    tableHtml += `<th>${escapeHtml(col)}</th>`;
                }
                tableHtml += '</tr></thead><tbody>';

                for (const row of data.rows) {
                    tableHtml += '<tr>';
                    for (const val of row) {
                        if (val === null) {
                            tableHtml += '<td class="null-value">NULL</td>';
                        } else {
                            tableHtml += `<td title="${escapeHtml(String(val))}">${escapeHtml(String(val))}</td>`;
                        }
                    }
                    tableHtml += '</tr>';
                }
                tableHtml += '</tbody></table></div>';
                results.innerHTML = tableHtml;

            } catch (err) {
                status.textContent = 'Error';
                status.className = 'status error';
                results.innerHTML = `<div class="error-message">${escapeHtml(err.message)}</div>`;
            }
        }

        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        // Event listeners
        runBtn.addEventListener('click', runQuery);

        sqlInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                runQuery();
            }
        });

        document.querySelectorAll('.quick-query').forEach(btn => {
            btn.addEventListener('click', () => {
                sqlInput.value = btn.dataset.query;
                runQuery();
            });
        });

        // Initial load
        loadSchema();
    </script>
</body>
</html>
"#;
