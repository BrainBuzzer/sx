use std::collections::HashMap;

use anyhow::{Context as _, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, ToSql, params_from_iter};

#[derive(Debug, Clone)]
pub struct OrphanEmbedding {
    pub content_hash: String,
    pub key_i64: i64,
}

#[derive(Debug, Clone)]
pub struct ResolvedCollection {
    pub collection: String,
    pub model: String,
    pub dim: i64,
}

#[derive(Debug, Clone)]
pub struct ChunkMeta {
    pub chunk_id: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub kind: String,
    pub symbol: Option<String>,
}

pub fn upsert_collection(
    conn: &Connection,
    collection: &str,
    provider: &str,
    model: &str,
    dim: i64,
    metric: &str,
    quantization: &str,
) -> Result<()> {
    conn.execute(
        r#"
INSERT INTO vector_collections(collection, provider, model, dim, metric, quantization, updated_at)
VALUES(?1, ?2, ?3, ?4, ?5, ?6, strftime('%s','now'))
ON CONFLICT(collection) DO UPDATE SET
  provider=excluded.provider,
  model=excluded.model,
  dim=excluded.dim,
  metric=excluded.metric,
  quantization=excluded.quantization,
  updated_at=excluded.updated_at
"#,
        rusqlite::params![collection, provider, model, dim, metric, quantization],
    )
    .context("upsert vector_collections")?;
    Ok(())
}

pub fn resolve_collection(
    conn: &Connection,
    provider: &str,
    model: &str,
    preferred_dim: Option<usize>,
) -> Result<Option<ResolvedCollection>> {
    if let Some(dim) = preferred_dim {
        let exact: Option<ResolvedCollection> = conn
            .query_row(
                r#"
SELECT collection, model, dim
FROM vector_collections
WHERE provider=?1 AND model=?2 AND dim=?3
LIMIT 1
"#,
                rusqlite::params![provider, model, dim as i64],
                |row| {
                    Ok(ResolvedCollection {
                        collection: row.get(0)?,
                        model: row.get(1)?,
                        dim: row.get(2)?,
                    })
                },
            )
            .optional()
            .context("resolve collection (exact dim)")?;
        if exact.is_some() {
            return Ok(exact);
        }
    }

    let latest: Option<ResolvedCollection> = conn
        .query_row(
            r#"
SELECT collection, model, dim
FROM vector_collections
WHERE provider=?1 AND model=?2
ORDER BY updated_at DESC
LIMIT 1
"#,
            rusqlite::params![provider, model],
            |row| {
                Ok(ResolvedCollection {
                    collection: row.get(0)?,
                    model: row.get(1)?,
                    dim: row.get(2)?,
                })
            },
        )
        .optional()
        .context("resolve collection (latest)")?;
    Ok(latest)
}

pub fn latest_collection_dim(
    conn: &Connection,
    provider: &str,
    model: &str,
) -> Result<Option<i64>> {
    let dim: Option<i64> = conn
        .query_row(
            r#"
SELECT dim
FROM vector_collections
WHERE provider=?1 AND model=?2
ORDER BY updated_at DESC
LIMIT 1
"#,
            rusqlite::params![provider, model],
            |row| row.get(0),
        )
        .optional()
        .context("read latest dim")?;
    Ok(dim)
}

pub fn drop_collection(conn: &Connection, collection: &str) -> Result<()> {
    conn.execute("DELETE FROM embeddings WHERE collection=?1", [collection])
        .context("delete embeddings")?;
    conn.execute(
        "DELETE FROM query_embed_cache WHERE collection=?1",
        [collection],
    )
    .context("delete query_embed_cache")?;
    conn.execute(
        "DELETE FROM vector_collections WHERE collection=?1",
        [collection],
    )
    .context("delete vector_collections")?;
    Ok(())
}

pub fn count_embeddings(conn: &Connection, collection: &str) -> Result<usize> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE collection=?1",
            [collection],
            |row| row.get(0),
        )
        .context("count embeddings")?;
    Ok(n as usize)
}

pub fn list_orphan_embeddings(conn: &Connection, collection: &str) -> Result<Vec<OrphanEmbedding>> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT e.content_hash, e.key
FROM embeddings e
LEFT JOIN (SELECT DISTINCT content_hash FROM chunks) c ON c.content_hash = e.content_hash
WHERE e.collection=?1 AND c.content_hash IS NULL
ORDER BY e.content_hash
"#,
        )
        .context("prepare orphan embeddings query")?;

    let rows = stmt
        .query_map([collection], |row| {
            Ok(OrphanEmbedding {
                content_hash: row.get(0)?,
                key_i64: row.get(1)?,
            })
        })
        .context("query orphan embeddings")?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("read orphan row")?);
    }
    Ok(out)
}

pub fn delete_embedding(conn: &Connection, collection: &str, content_hash: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM embeddings WHERE collection=?1 AND content_hash=?2",
        rusqlite::params![collection, content_hash],
    )
    .context("delete embedding")?;
    Ok(())
}

pub fn list_missing_content_hashes(conn: &Connection, collection: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT DISTINCT c.content_hash
FROM chunks c
LEFT JOIN embeddings e
  ON e.collection=?1 AND e.content_hash=c.content_hash
WHERE e.content_hash IS NULL
ORDER BY c.content_hash
"#,
        )
        .context("prepare missing content_hash query")?;

    let rows = stmt
        .query_map([collection], |row| row.get::<_, String>(0))
        .context("query missing content_hash")?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("read content_hash")?);
    }
    Ok(out)
}

pub fn fetch_representative_content_for_hash(
    conn: &Connection,
    content_hash: &str,
) -> Result<String> {
    let content: String = conn
        .query_row(
            r#"
SELECT f.content
FROM chunks_fts f
JOIN chunks c ON c.chunk_id=f.chunk_id
WHERE c.content_hash=?1
ORDER BY c.path ASC, c.start_line ASC
LIMIT 1
"#,
            [content_hash],
            |row| row.get(0),
        )
        .context("fetch chunk content")?;
    Ok(content)
}

pub fn fetch_any_chunk_content(conn: &Connection) -> Result<String> {
    let content: Option<String> = conn
        .query_row("SELECT content FROM chunks_fts LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .context("fetch any chunk content")?;
    content.ok_or_else(|| anyhow!("no indexed chunks found"))
}

pub fn content_hash_for_key(
    conn: &Connection,
    collection: &str,
    key_i64: i64,
) -> Result<Option<String>> {
    let h: Option<String> = conn
        .query_row(
            "SELECT content_hash FROM embeddings WHERE collection=?1 AND key=?2 LIMIT 1",
            rusqlite::params![collection, key_i64],
            |row| row.get(0),
        )
        .optional()
        .context("lookup content_hash for key")?;
    Ok(h)
}

pub fn upsert_embedding(
    conn: &Connection,
    collection: &str,
    content_hash: &str,
    key_i64: i64,
    updated_at: i64,
) -> Result<()> {
    conn.execute(
        r#"
INSERT INTO embeddings(collection, content_hash, key, updated_at)
VALUES(?1, ?2, ?3, ?4)
ON CONFLICT(collection, content_hash) DO UPDATE SET
  key=excluded.key,
  updated_at=excluded.updated_at
"#,
        rusqlite::params![collection, content_hash, key_i64, updated_at],
    )
    .context("upsert embedding")?;
    Ok(())
}

pub fn representative_chunks_for_keys(
    conn: &Connection,
    collection: &str,
    keys: &[u64],
    langs: &[String],
    path_prefixes: &[String],
) -> Result<HashMap<u64, ChunkMeta>> {
    if keys.is_empty() {
        return Ok(HashMap::new());
    }

    let mut sql = String::from(
        r#"
SELECT
  e.key,
  c.chunk_id,
  c.path,
  c.start_line,
  c.end_line,
  c.kind,
  c.symbol
FROM embeddings e
JOIN chunks c ON c.content_hash = e.content_hash
JOIN files f ON f.path = c.path
WHERE e.collection=?1 AND e.key IN (
"#,
    );

    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    params.push(Box::new(collection.to_string()));

    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push('?');
        sql.push_str(&(params.len() + 1).to_string());
        params.push(Box::new(*k as i64));
    }
    sql.push_str(")\n");

    if !langs.is_empty() {
        sql.push_str(" AND f.language IN (");
        for i in 0..langs.len() {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push('?');
            sql.push_str(&(params.len() + 1).to_string());
            params.push(Box::new(langs[i].clone()));
        }
        sql.push(')');
    }

    if !path_prefixes.is_empty() {
        sql.push_str(" AND (");
        for (i, pfx) in path_prefixes.iter().enumerate() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            sql.push_str("c.path LIKE ?");
            sql.push_str(&(params.len() + 1).to_string());
            params.push(Box::new(format!("{pfx}%")));
        }
        sql.push(')');
    }

    sql.push_str(" ORDER BY e.key ASC, c.path ASC, c.start_line ASC");

    let mut stmt = conn.prepare(&sql).context("prepare representative query")?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                ChunkMeta {
                    chunk_id: row.get(1)?,
                    path: row.get(2)?,
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                    kind: row.get(5)?,
                    symbol: row.get(6)?,
                },
            ))
        })
        .context("query representative chunks")?;

    let mut out: HashMap<u64, ChunkMeta> = HashMap::new();
    for row in rows {
        let (key_i64, meta) = row.context("read representative row")?;
        let key = key_i64 as u64;
        out.entry(key).or_insert(meta);
    }
    Ok(out)
}

pub fn get_cached_query_embedding(
    conn: &Connection,
    collection: &str,
    query: &str,
    expected_dim: usize,
) -> Result<Option<Vec<f32>>> {
    let row: Option<(i64, Vec<u8>)> = conn
        .query_row(
            "SELECT dim, vector FROM query_embed_cache WHERE collection=?1 AND query=?2 LIMIT 1",
            rusqlite::params![collection, query],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .context("read query_embed_cache")?;

    let Some((dim, blob)) = row else {
        return Ok(None);
    };
    if dim as usize != expected_dim {
        return Ok(None);
    }

    if blob.len() != expected_dim * 4 {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(expected_dim);
    for chunk in blob.chunks_exact(4) {
        let b: [u8; 4] = chunk.try_into().unwrap_or([0, 0, 0, 0]);
        out.push(f32::from_le_bytes(b));
    }
    Ok(Some(out))
}

pub fn put_cached_query_embedding(
    conn: &Connection,
    collection: &str,
    query: &str,
    dim: usize,
    vector: &[f32],
    updated_at: i64,
) -> Result<()> {
    if vector.len() != dim {
        return Err(anyhow!("vector length does not match dim"));
    }
    let mut blob = Vec::with_capacity(dim * 4);
    for v in vector {
        blob.extend_from_slice(&v.to_le_bytes());
    }

    conn.execute(
        r#"
INSERT INTO query_embed_cache(collection, query, dim, vector, updated_at)
VALUES(?1, ?2, ?3, ?4, ?5)
ON CONFLICT(collection, query) DO UPDATE SET
  dim=excluded.dim,
  vector=excluded.vector,
  updated_at=excluded.updated_at
"#,
        rusqlite::params![collection, query, dim as i64, blob, updated_at],
    )
    .context("upsert query_embed_cache")?;
    Ok(())
}

pub fn keys_for_chunk_ids(
    conn: &Connection,
    collection: &str,
    chunk_ids: &[String],
) -> Result<HashMap<String, u64>> {
    if chunk_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let mut sql = String::from(
        r#"
SELECT c.chunk_id, e.key
FROM chunks c
JOIN embeddings e ON e.collection=?1 AND e.content_hash=c.content_hash
WHERE c.chunk_id IN (
"#,
    );
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    params.push(Box::new(collection.to_string()));

    for (i, id) in chunk_ids.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push('?');
        sql.push_str(&(params.len() + 1).to_string());
        params.push(Box::new(id.clone()));
    }
    sql.push_str(")\n");

    let mut stmt = conn.prepare(&sql).context("prepare keys_for_chunk_ids")?;
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |row| {
            let chunk_id: String = row.get(0)?;
            let key_i64: i64 = row.get(1)?;
            Ok((chunk_id, key_i64 as u64))
        })
        .context("query keys_for_chunk_ids")?;

    let mut out = HashMap::new();
    for row in rows {
        let (chunk_id, key) = row.context("read keys_for_chunk_ids row")?;
        out.insert(chunk_id, key);
    }
    Ok(out)
}
