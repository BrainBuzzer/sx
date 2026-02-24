pub mod chunk;
pub mod scan;

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context as _, Result};
use rusqlite::{Connection, OptionalExtension, params};

use crate::{cli, config, db, trace};

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub scanned: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub chunks_written: usize,
    pub removed: usize,
    pub duration_ms: u128,
}

pub fn run(
    conn: &mut Connection,
    root: &Path,
    cfg: &config::Config,
    args: cli::IndexArgs,
) -> Result<IndexStats> {
    let started = Instant::now();

    if args.jobs != 1 {
        tracing::warn!(
            jobs = args.jobs,
            "Phase 1 indexer runs single-threaded; using 1 job"
        );
    }

    if args.full {
        db::wipe_index(conn)?;
    }

    let scanned_files = scan::scan_repo(root, cfg)?;
    let scanned_set: HashSet<String> = scanned_files.iter().map(|f| f.repo_path.clone()).collect();

    let mut stats = IndexStats {
        scanned: 0,
        indexed: 0,
        skipped: 0,
        chunks_written: 0,
        removed: 0,
        duration_ms: 0,
    };

    let mut chunker = chunk::Chunker::new().context("initialize chunker")?;

    for scanned in scanned_files {
        stats.scanned += 1;

        let existing: Option<(i64, i64, String)> = {
            let mut stmt = conn
                .prepare("SELECT size, mtime, content_hash FROM files WHERE path=?1")
                .context("prepare files lookup")?;
            stmt.query_row((scanned.repo_path.as_str(),), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()
            .context("query existing file row")?
        };

        if let Some((size, mtime, _hash)) = &existing {
            if *size == scanned.size as i64 && *mtime == scanned.mtime {
                stats.skipped += 1;
                continue;
            }
        }

        let file_bytes = std::fs::read(&scanned.abs_path)
            .with_context(|| format!("read {}", scanned.abs_path.display()))?;

        if is_probably_binary(&file_bytes) {
            tracing::debug!(path = %scanned.repo_path, "skipping binary-like file");
            if existing.is_some() {
                db::delete_path(conn, &scanned.repo_path)?;
            }
            stats.skipped += 1;
            continue;
        }

        let file_text = String::from_utf8_lossy(&file_bytes).to_string();

        let file_hash = blake3::hash(file_text.as_bytes()).to_hex().to_string();

        if let Some((_size, _mtime, old_hash)) = &existing {
            if old_hash == &file_hash {
                conn.execute(
                    r#"
INSERT INTO files(path, size, mtime, content_hash, language, status, indexed_at)
VALUES(?1, ?2, ?3, ?4, ?5, 'indexed', ?6)
ON CONFLICT(path) DO UPDATE SET
  size=excluded.size,
  mtime=excluded.mtime,
  content_hash=excluded.content_hash,
  language=excluded.language,
  status=excluded.status,
  indexed_at=excluded.indexed_at
"#,
                    params![
                        &scanned.repo_path,
                        scanned.size as i64,
                        scanned.mtime,
                        file_hash,
                        scanned.language.as_str(),
                        db::now_unix()
                    ],
                )
                .context("update files row (no content change)")?;
                stats.skipped += 1;
                continue;
            }
        }

        let chunks = chunker
            .chunk_file(&scanned.repo_path, scanned.language, &file_text, &cfg.index)
            .with_context(|| format!("chunk {}", scanned.repo_path))?;

        let now = db::now_unix();
        let tx = conn.transaction().context("begin transaction")?;

        tx.execute(
            r#"
INSERT INTO files(path, size, mtime, content_hash, language, status, indexed_at)
VALUES(?1, ?2, ?3, ?4, ?5, 'indexed', ?6)
ON CONFLICT(path) DO UPDATE SET
  size=excluded.size,
  mtime=excluded.mtime,
  content_hash=excluded.content_hash,
  language=excluded.language,
  status=excluded.status,
  indexed_at=excluded.indexed_at
"#,
            params![
                &scanned.repo_path,
                scanned.size as i64,
                scanned.mtime,
                file_hash,
                scanned.language.as_str(),
                now
            ],
        )
        .context("upsert files row")?;

        tx.execute(
            "DELETE FROM chunks WHERE path=?1",
            params![&scanned.repo_path],
        )
        .context("delete old chunks")?;
        tx.execute(
            "DELETE FROM chunks_fts WHERE path=?1",
            params![&scanned.repo_path],
        )
        .context("delete old chunks_fts")?;

        {
            let mut chunk_stmt = tx
                .prepare(
                    r#"
INSERT INTO chunks(chunk_id, path, start_line, end_line, kind, symbol, content_hash, updated_at)
VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
"#,
                )
                .context("prepare chunks insert")?;
            let mut fts_stmt = tx
                .prepare(
                    r#"
INSERT INTO chunks_fts(content, chunk_id, path, start_line, end_line, kind, symbol)
VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
"#,
                )
                .context("prepare chunks_fts insert")?;

            for chunk in &chunks {
                chunk_stmt
                    .execute(params![
                        &chunk.chunk_id,
                        &chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        &chunk.kind,
                        &chunk.symbol,
                        &chunk.content_hash,
                        now
                    ])
                    .with_context(|| format!("insert chunk {}", chunk.chunk_id))?;
                fts_stmt
                    .execute(params![
                        &chunk.content,
                        &chunk.chunk_id,
                        &chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        &chunk.kind,
                        &chunk.symbol
                    ])
                    .with_context(|| format!("insert fts chunk {}", chunk.chunk_id))?;
            }
        }

        trace::graph::upsert_file(&tx, &scanned.repo_path, scanned.language, &chunks, now)
            .with_context(|| format!("update trace graph {}", scanned.repo_path))?;

        tx.commit().context("commit transaction")?;
        stats.indexed += 1;
        stats.chunks_written += chunks.len();
    }

    let existing_paths: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT path FROM files")
            .context("prepare files list")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("query files list")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("read files.path")?);
        }
        out
    };

    for path in existing_paths {
        if !scanned_set.contains(&path) {
            db::delete_path(conn, &path)?;
            stats.removed += 1;
        }
    }

    stats.duration_ms = started.elapsed().as_millis();
    Ok(stats)
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    let sample_len = bytes.len().min(8 * 1024);
    let sample = &bytes[..sample_len];
    if sample.iter().any(|b| *b == 0) {
        return true;
    }

    let lossy = String::from_utf8_lossy(sample);
    let mut total = 0usize;
    let mut replaced = 0usize;
    for ch in lossy.chars() {
        total += 1;
        if ch == '\u{FFFD}' {
            replaced += 1;
        }
    }

    if total == 0 {
        return false;
    }

    (replaced as f64 / total as f64) > 0.10
}
