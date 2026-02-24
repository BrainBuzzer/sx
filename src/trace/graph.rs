use std::collections::HashMap;

use anyhow::{Context as _, Result};
use rusqlite::{Connection, OptionalExtension, ToSql, params_from_iter};

use crate::index::chunk::Chunk;
use crate::index::scan::{Language, is_test_case_path};

use super::extract;
use super::types::{EdgeRecord, SymbolRecord, TraceStageResult};

pub fn upsert_file(
    conn: &Connection,
    repo_path: &str,
    language: Language,
    chunks: &[Chunk],
    now: i64,
) -> Result<()> {
    conn.execute("DELETE FROM trace_edges WHERE path=?1", [repo_path])
        .context("delete trace_edges for file")?;
    conn.execute("DELETE FROM symbols WHERE path=?1", [repo_path])
        .context("delete symbols for file")?;

    let (symbols, edges) = extract::extract_for_file(repo_path, language, chunks);
    if symbols.is_empty() {
        return Ok(());
    }

    let mut sym_stmt = conn
        .prepare(
            r#"
INSERT INTO symbols(symbol_id, chunk_id, path, start_line, end_line, language, kind, fq_name, short_name, updated_at)
VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
"#,
        )
        .context("prepare symbols insert")?;

    for s in &symbols {
        sym_stmt
            .execute(rusqlite::params![
                &s.symbol_id,
                &s.chunk_id,
                &s.path,
                s.start_line,
                s.end_line,
                &s.language,
                &s.kind,
                &s.fq_name,
                &s.short_name,
                now
            ])
            .with_context(|| format!("insert symbol {}", s.symbol_id))?;
    }

    let mut edge_stmt = conn
        .prepare(
            r#"
INSERT INTO trace_edges(edge_id, src_symbol_id, dst_symbol_id, dst_name, edge_kind, path, line, confidence, evidence, updated_at)
VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
"#,
        )
        .context("prepare edge insert")?;

    for e in edges {
        let dst_symbol_id = if let Some(dst_name) = e.dst_name.as_deref() {
            resolve_symbol_id(conn, repo_path, dst_name)?
        } else {
            None
        };
        let edge_id = edge_id_for(
            e.src_symbol_id.as_str(),
            e.edge_kind.as_str(),
            e.path.as_str(),
            e.line,
            dst_symbol_id.as_deref(),
            e.dst_name.as_deref(),
            e.evidence.as_str(),
        );
        edge_stmt
            .execute(rusqlite::params![
                edge_id,
                e.src_symbol_id,
                dst_symbol_id,
                e.dst_name,
                e.edge_kind,
                e.path,
                e.line,
                e.confidence,
                e.evidence,
                now
            ])
            .context("insert trace edge")?;
    }

    Ok(())
}

pub fn symbols_for_chunk_ids(
    conn: &Connection,
    chunk_ids: &[String],
) -> Result<HashMap<String, Vec<SymbolRecord>>> {
    let mut out: HashMap<String, Vec<SymbolRecord>> = HashMap::new();
    if chunk_ids.is_empty() {
        return Ok(out);
    }

    let mut sql = String::from(
        "SELECT symbol_id, chunk_id, path, start_line, end_line, language, kind, fq_name, short_name FROM symbols WHERE chunk_id IN (",
    );
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();

    for (i, id) in chunk_ids.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push('?');
        sql.push_str(&(params.len() + 1).to_string());
        params.push(Box::new(id.clone()));
    }
    sql.push(')');

    let mut stmt = conn
        .prepare(&sql)
        .context("prepare symbols_for_chunk_ids")?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            Ok(SymbolRecord {
                symbol_id: row.get(0)?,
                chunk_id: row.get(1)?,
                path: row.get(2)?,
                start_line: row.get(3)?,
                end_line: row.get(4)?,
                language: row.get(5)?,
                kind: row.get(6)?,
                fq_name: row.get(7)?,
                short_name: row.get(8)?,
            })
        })
        .context("query symbols_for_chunk_ids")?;

    for row in rows {
        let s = row.context("read symbol row")?;
        if is_test_case_path(&s.path) {
            continue;
        }
        out.entry(s.chunk_id.clone()).or_default().push(s);
    }
    Ok(out)
}

pub fn symbol_by_id(conn: &Connection, symbol_id: &str) -> Result<Option<SymbolRecord>> {
    let out: Option<SymbolRecord> = conn
        .query_row(
            r#"
SELECT symbol_id, chunk_id, path, start_line, end_line, language, kind, fq_name, short_name
FROM symbols
WHERE symbol_id=?1
LIMIT 1
"#,
            [symbol_id],
            |row| {
                Ok(SymbolRecord {
                    symbol_id: row.get(0)?,
                    chunk_id: row.get(1)?,
                    path: row.get(2)?,
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                    language: row.get(5)?,
                    kind: row.get(6)?,
                    fq_name: row.get(7)?,
                    short_name: row.get(8)?,
                })
            },
        )
        .optional()
        .context("query symbol_by_id")?;
    Ok(out.filter(|s| !is_test_case_path(&s.path)))
}

pub fn symbols_for_path_line(
    conn: &Connection,
    path: &str,
    line: i64,
) -> Result<Vec<SymbolRecord>> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT symbol_id, chunk_id, path, start_line, end_line, language, kind, fq_name, short_name
FROM symbols
WHERE path=?1 AND start_line<=?2 AND end_line>=?2
ORDER BY (end_line - start_line) ASC
"#,
        )
        .context("prepare symbols_for_path_line")?;
    let rows = stmt
        .query_map(rusqlite::params![path, line], |row| {
            Ok(SymbolRecord {
                symbol_id: row.get(0)?,
                chunk_id: row.get(1)?,
                path: row.get(2)?,
                start_line: row.get(3)?,
                end_line: row.get(4)?,
                language: row.get(5)?,
                kind: row.get(6)?,
                fq_name: row.get(7)?,
                short_name: row.get(8)?,
            })
        })
        .context("query symbols_for_path_line")?;
    let mut out = Vec::new();
    for row in rows {
        let sym = row.context("read symbols_for_path_line row")?;
        if is_test_case_path(&sym.path) {
            continue;
        }
        out.push(sym);
    }
    Ok(out)
}

pub fn outgoing_edges(
    conn: &Connection,
    symbol_id: &str,
    edge_kinds: Option<&[String]>,
) -> Result<Vec<EdgeRecord>> {
    let mut sql = String::from(
        "SELECT edge_id, src_symbol_id, dst_symbol_id, dst_name, edge_kind, path, line, confidence, evidence FROM trace_edges WHERE src_symbol_id=?1",
    );
    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(symbol_id.to_string())];

    if let Some(kinds) = edge_kinds {
        if !kinds.is_empty() {
            sql.push_str(" AND edge_kind IN (");
            for (i, k) in kinds.iter().enumerate() {
                if i > 0 {
                    sql.push_str(", ");
                }
                sql.push('?');
                sql.push_str(&(params.len() + 1).to_string());
                params.push(Box::new(k.clone()));
            }
            sql.push(')');
        }
    }

    sql.push_str(" ORDER BY confidence DESC, line ASC");

    let mut stmt = conn.prepare(&sql).context("prepare outgoing_edges")?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            Ok(EdgeRecord {
                edge_id: row.get(0)?,
                src_symbol_id: row.get(1)?,
                dst_symbol_id: row.get(2)?,
                dst_name: row.get(3)?,
                edge_kind: row.get(4)?,
                path: row.get(5)?,
                line: row.get(6)?,
                confidence: row.get(7)?,
                evidence: row.get(8)?,
            })
        })
        .context("query outgoing_edges")?;

    let mut out = Vec::new();
    for row in rows {
        let edge = row.context("read outgoing edge row")?;
        if is_test_case_path(&edge.path) {
            continue;
        }
        out.push(edge);
    }
    Ok(out)
}

pub fn resolve_symbol_id(
    conn: &Connection,
    src_path: &str,
    dst_name: &str,
) -> Result<Option<String>> {
    {
        let mut stmt = conn
            .prepare(
                "SELECT symbol_id FROM symbols WHERE fq_name=?1 ORDER BY path ASC, start_line ASC LIMIT 64",
            )
            .context("prepare resolve symbol by fq_name")?;
        let rows = stmt
            .query_map([dst_name], |row| row.get::<_, String>(0))
            .context("query resolve symbol by fq_name")?;
        for row in rows {
            let symbol_id = row.context("read resolve symbol by fq_name row")?;
            if symbol_by_id(conn, &symbol_id)?.is_some() {
                return Ok(Some(symbol_id));
            }
        }
    }

    let short = dst_name.rsplit("::").next().unwrap_or(dst_name);
    let mut stmt = conn
        .prepare(
            r#"
SELECT symbol_id
FROM symbols
WHERE short_name=?1
ORDER BY (path=?2) DESC, path ASC, start_line ASC
LIMIT 128
"#,
        )
        .context("prepare resolve symbol by short_name")?;
    let rows = stmt
        .query_map(rusqlite::params![short, src_path], |row| {
            row.get::<_, String>(0)
        })
        .context("query resolve symbol by short_name")?;
    for row in rows {
        let symbol_id = row.context("read resolve symbol by short_name row")?;
        if symbol_by_id(conn, &symbol_id)?.is_some() {
            return Ok(Some(symbol_id));
        }
    }
    Ok(None)
}

pub fn get_cached_stage(
    conn: &Connection,
    query: &str,
    stage: &str,
) -> Result<Option<TraceStageResult>> {
    let body: Option<String> = conn
        .query_row(
            "SELECT result_json FROM trace_query_cache WHERE query=?1 AND stage=?2 LIMIT 1",
            rusqlite::params![query, stage],
            |row| row.get(0),
        )
        .optional()
        .context("read trace_query_cache")?;
    let Some(body) = body else {
        return Ok(None);
    };
    let parsed = serde_json::from_str::<TraceStageResult>(&body).ok();
    Ok(parsed)
}

pub fn put_cached_stage(
    conn: &Connection,
    query: &str,
    stage: &str,
    result: &TraceStageResult,
) -> Result<()> {
    let body = serde_json::to_string(result).context("serialize trace stage cache")?;
    conn.execute(
        r#"
INSERT INTO trace_query_cache(query, stage, result_json, updated_at)
VALUES(?1, ?2, ?3, strftime('%s','now'))
ON CONFLICT(query, stage) DO UPDATE SET
  result_json=excluded.result_json,
  updated_at=excluded.updated_at
"#,
        rusqlite::params![query, stage, body],
    )
    .context("upsert trace_query_cache")?;
    Ok(())
}

pub fn count_symbols(conn: &Connection) -> Result<usize> {
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))
        .context("count symbols")?;
    Ok(n as usize)
}

fn edge_id_for(
    src_symbol_id: &str,
    edge_kind: &str,
    path: &str,
    line: i64,
    dst_symbol_id: Option<&str>,
    dst_name: Option<&str>,
    evidence: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(src_symbol_id.as_bytes());
    hasher.update(b"|");
    hasher.update(edge_kind.as_bytes());
    hasher.update(b"|");
    hasher.update(path.as_bytes());
    hasher.update(b"|");
    hasher.update(line.to_string().as_bytes());
    hasher.update(b"|");
    if let Some(id) = dst_symbol_id {
        hasher.update(id.as_bytes());
    }
    hasher.update(b"|");
    if let Some(name) = dst_name {
        hasher.update(name.as_bytes());
    }
    hasher.update(b"|");
    hasher.update(evidence.as_bytes());
    hasher.finalize().to_hex().to_string()
}
