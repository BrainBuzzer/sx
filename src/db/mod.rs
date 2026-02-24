use std::fs;
use std::path::Path;

use anyhow::{Context as _, Result};
use rusqlite::Connection;

const LATEST_SCHEMA_VERSION: i64 = 3;

pub fn open(db_path: &Path) -> Result<Connection> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let conn = Connection::open(db_path).with_context(|| format!("open {}", db_path.display()))?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")
        .context("set PRAGMA journal_mode=WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .context("set PRAGMA synchronous=NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")
        .context("set PRAGMA temp_store=MEMORY")?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .context("set PRAGMA foreign_keys=ON")?;
    Ok(())
}

pub fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files(
  path TEXT PRIMARY KEY,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  language TEXT NOT NULL,
  status TEXT NOT NULL,
  indexed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  kind TEXT NOT NULL,
  symbol TEXT,
  content_hash TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_by_path_start ON chunks(path, start_line);
CREATE INDEX IF NOT EXISTS chunks_by_content_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content,
  chunk_id UNINDEXED,
  path UNINDEXED,
  start_line UNINDEXED,
  end_line UNINDEXED,
  kind UNINDEXED,
  symbol UNINDEXED,
  tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS dir_frecency(
  path TEXT PRIMARY KEY,
  rank REAL NOT NULL,
  last_accessed INTEGER NOT NULL
);
"#,
    )
    .context("create schema")?;

    let now = now_unix();
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?1)",
        (now.to_string(),),
    )
    .context("write meta.created_at")?;

    let current = schema_version(conn)?.unwrap_or(1);
    if current < 2 {
        migrate_v2(conn)?;
    }
    if current < 3 {
        migrate_v3(conn)?;
    }

    // If we created a brand new DB (or upgraded), ensure the version is current.
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?1)\nON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (LATEST_SCHEMA_VERSION.to_string(),),
    )
    .context("write meta.schema_version")?;

    Ok(())
}

fn schema_version(conn: &Connection) -> Result<Option<i64>> {
    use rusqlite::OptionalExtension as _;
    let version: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |row| row.get(0),
        )
        .optional()
        .context("read meta.schema_version")?;
    Ok(version.and_then(|s| s.parse::<i64>().ok()))
}

fn migrate_v2(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS vector_collections(
  collection TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  metric TEXT NOT NULL,
  quantization TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings(
  collection TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  key INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(collection, content_hash),
  UNIQUE(collection, key)
);
CREATE INDEX IF NOT EXISTS embeddings_by_collection_key ON embeddings(collection, key);
CREATE INDEX IF NOT EXISTS embeddings_by_collection_hash ON embeddings(collection, content_hash);

CREATE TABLE IF NOT EXISTS query_embed_cache(
  collection TEXT NOT NULL,
  query TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(collection, query)
);

CREATE TABLE IF NOT EXISTS llm_expansion_cache(
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  query TEXT NOT NULL,
  expansions_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(provider, model, query)
);

CREATE TABLE IF NOT EXISTS llm_rerank_cache(
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  query TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  score REAL NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(provider, model, query, content_hash)
);
"#,
    )
    .context("migrate v2")?;
    Ok(())
}

fn migrate_v3(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS symbols(
  symbol_id TEXT PRIMARY KEY,
  chunk_id TEXT NOT NULL,
  path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  language TEXT NOT NULL,
  kind TEXT NOT NULL,
  fq_name TEXT NOT NULL,
  short_name TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_by_short_name ON symbols(short_name);
CREATE INDEX IF NOT EXISTS symbols_by_path_start ON symbols(path, start_line);

CREATE TABLE IF NOT EXISTS trace_edges(
  edge_id TEXT PRIMARY KEY,
  src_symbol_id TEXT NOT NULL,
  dst_symbol_id TEXT,
  dst_name TEXT,
  edge_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  line INTEGER NOT NULL,
  confidence REAL NOT NULL,
  evidence TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_edges_by_src ON trace_edges(src_symbol_id);
CREATE INDEX IF NOT EXISTS trace_edges_by_dst_symbol ON trace_edges(dst_symbol_id);
CREATE INDEX IF NOT EXISTS trace_edges_by_dst_name ON trace_edges(dst_name);

CREATE TABLE IF NOT EXISTS trace_query_cache(
  query TEXT NOT NULL,
  stage TEXT NOT NULL,
  result_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(query, stage)
);
"#,
    )
    .context("migrate v3")?;
    Ok(())
}

pub fn wipe_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
DELETE FROM files;
DELETE FROM chunks;
DELETE FROM chunks_fts;
DELETE FROM symbols;
DELETE FROM trace_edges;
DELETE FROM trace_query_cache;
"#,
    )
    .context("wipe index tables")?;
    Ok(())
}

pub fn delete_path(conn: &mut Connection, path: &str) -> Result<()> {
    let tx = conn.transaction().context("begin transaction")?;
    tx.execute("DELETE FROM trace_edges WHERE path=?1", (path,))
        .context("delete trace_edges")?;
    tx.execute("DELETE FROM symbols WHERE path=?1", (path,))
        .context("delete symbols")?;
    tx.execute("DELETE FROM chunks WHERE path=?1", (path,))
        .context("delete chunks")?;
    tx.execute("DELETE FROM chunks_fts WHERE path=?1", (path,))
        .context("delete chunks_fts")?;
    tx.execute("DELETE FROM files WHERE path=?1", (path,))
        .context("delete files")?;
    tx.commit().context("commit delete transaction")?;
    Ok(())
}

pub fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_secs() as i64
}

pub fn check_fts5_available() -> Result<()> {
    let conn = Connection::open_in_memory().context("open in-memory sqlite")?;
    conn.execute_batch(
        r#"
CREATE VIRTUAL TABLE t USING fts5(content);
INSERT INTO t(content) VALUES('hello');
SELECT rowid FROM t WHERE t MATCH 'hello';
"#,
    )
    .context("FTS5 check failed")?;
    Ok(())
}
