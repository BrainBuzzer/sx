use anyhow::{Context as _, Result};
use rusqlite::{Connection, ToSql, params_from_iter};
use serde::Serialize;

use crate::index::scan::is_test_case_path;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub limit: usize,
    pub literal: bool,
    pub fts: bool,
    pub langs: Vec<String>,
    pub path_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub kind: String,
    pub symbol: Option<String>,
    pub score: f64,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkRecord {
    pub chunk_id: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub kind: String,
    pub symbol: Option<String>,
    pub content_hash: String,
    pub content: String,
}

pub fn search(conn: &Connection, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
    let q = if opts.fts || !opts.literal {
        opts.query.clone()
    } else {
        build_literal_query(&opts.query)
    };

    if q.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut sql = String::from(
        r#"
SELECT
  chunks_fts.chunk_id,
  chunks_fts.path,
  chunks_fts.start_line,
  chunks_fts.end_line,
  chunks_fts.kind,
  chunks_fts.symbol,
  bm25(chunks_fts) AS bm,
  snippet(chunks_fts, 0, '[', ']', '…', 10) AS snip
FROM chunks_fts
JOIN files ON files.path = chunks_fts.path
WHERE chunks_fts MATCH ?1
"#,
    );

    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    params.push(Box::new(q));

    if !opts.langs.is_empty() {
        sql.push_str(" AND files.language IN (");
        for i in 0..opts.langs.len() {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push('?');
            sql.push_str(&(params.len() + 1).to_string());
            params.push(Box::new(opts.langs[i].clone()));
        }
        sql.push(')');
    }

    if !opts.path_prefixes.is_empty() {
        sql.push_str(" AND (");
        for (i, pfx) in opts.path_prefixes.iter().enumerate() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            sql.push_str("chunks_fts.path LIKE ?");
            sql.push_str(&(params.len() + 1).to_string());
            params.push(Box::new(format!("{}%", pfx)));
        }
        sql.push(')');
    }

    let query_limit = opts.limit.saturating_mul(5).max(opts.limit).max(1);
    sql.push_str(" ORDER BY bm LIMIT ?");
    sql.push_str(&(params.len() + 1).to_string());
    params.push(Box::new(query_limit as i64));

    let mut stmt = conn.prepare(&sql).context("prepare search SQL")?;
    let mut rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            let bm: f64 = row.get(6)?;
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                path: row.get(1)?,
                start_line: row.get(2)?,
                end_line: row.get(3)?,
                kind: row.get(4)?,
                symbol: row.get(5)?,
                score: -bm,
                snippet: row.get(7)?,
            })
        })
        .context("run search query")?;

    let mut out = Vec::new();
    while let Some(row) = rows.next() {
        let item = row.context("read search row")?;
        if is_test_case_path(&item.path) {
            continue;
        }
        out.push(item);
        if out.len() >= opts.limit {
            break;
        }
    }
    Ok(out)
}

pub fn get_chunk_by_id(conn: &Connection, chunk_id: &str) -> Result<Option<ChunkRecord>> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT f.chunk_id, f.path, f.start_line, f.end_line, f.kind, f.symbol, c.content_hash, f.content
FROM chunks_fts f
JOIN chunks c ON c.chunk_id = f.chunk_id
WHERE f.chunk_id=?1
LIMIT 1
"#,
        )
        .context("prepare get chunk SQL")?;

    let mut rows = stmt
        .query_map([chunk_id], |row| {
            Ok(ChunkRecord {
                chunk_id: row.get(0)?,
                path: row.get(1)?,
                start_line: row.get(2)?,
                end_line: row.get(3)?,
                kind: row.get(4)?,
                symbol: row.get(5)?,
                content_hash: row.get(6)?,
                content: row.get(7)?,
            })
        })
        .context("query get chunk")?;

    if let Some(row) = rows.next() {
        let chunk = row.context("read chunk row")?;
        if is_test_case_path(&chunk.path) {
            return Ok(None);
        }
        return Ok(Some(chunk));
    }
    Ok(None)
}

pub fn find_chunk_covering_line(
    conn: &Connection,
    path: &str,
    line: i64,
) -> Result<Option<String>> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT chunk_id
FROM chunks
WHERE path=?1 AND start_line<=?2 AND end_line>=?2
ORDER BY (end_line - start_line) ASC
LIMIT 1
"#,
        )
        .context("prepare find chunk covering line")?;

    let mut rows = stmt
        .query_map(rusqlite::params![path, line], |row| row.get::<_, String>(0))
        .context("query find chunk covering line")?;

    if let Some(row) = rows.next() {
        return Ok(Some(row.context("read chunk_id")?));
    }
    Ok(None)
}

fn build_literal_query(input: &str) -> String {
    let mut parts = Vec::new();
    for tok in input.split_whitespace() {
        let escaped = tok.replace('"', "\"\"");
        parts.push(format!("\"{}\"", escaped));
    }
    parts.join(" AND ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_query_splits_and_quotes() {
        let q = build_literal_query("foo bar");
        assert_eq!(q, "\"foo\" AND \"bar\"");
    }

    #[test]
    fn literal_query_escapes_quotes() {
        let q = build_literal_query("a\"b");
        assert_eq!(q, "\"a\"\"b\"");
    }
}
