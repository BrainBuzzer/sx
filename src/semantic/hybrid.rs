use std::collections::HashMap;

use anyhow::{Context as _, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::search;

use super::{store, vector_index};

pub struct RankedList {
    pub weight: f64,
    pub items: Vec<String>,
}

pub fn rrf_fuse(rrf_k: usize, lists: &[RankedList]) -> HashMap<String, f64> {
    let k = rrf_k as f64;
    let mut out: HashMap<String, f64> = HashMap::new();

    for list in lists {
        if list.weight == 0.0 {
            continue;
        }
        for (rank0, id) in list.items.iter().enumerate() {
            let rank = (rank0 + 1) as f64;
            let score = list.weight / (k + rank);
            *out.entry(id.clone()).or_insert(0.0) += score;
        }
    }

    out
}

pub fn deterministic_rerank(
    scored: &[(String, f64)],
    rerank_top: usize,
    bm25_results: &[search::SearchResult],
    vec_sim_by_chunk: &HashMap<String, f64>,
    query: &str,
    query_vec: Option<&[f32]>,
    index: Option<&vector_index::VectorIndex>,
    conn: &Connection,
    collection: Option<&str>,
    metric: &str,
) -> Result<Vec<(String, f64)>> {
    let rerank_top = rerank_top.max(1).min(scored.len());
    let head = &scored[..rerank_top];
    let tail = &scored[rerank_top..];

    let mut bm25_by_id: HashMap<&str, f64> = HashMap::new();
    let mut max_bm25 = 0.0f64;
    for r in bm25_results {
        bm25_by_id.insert(r.chunk_id.as_str(), r.score);
        max_bm25 = max_bm25.max(r.score);
    }
    if max_bm25 <= 0.0 {
        max_bm25 = 1.0;
    }

    let query_terms = query_term_set(query);

    // Optional: compute cosine similarity for top candidates by fetching vectors from the index.
    let mut cos_by_id: HashMap<String, f64> = HashMap::new();
    if metric.trim().eq_ignore_ascii_case("cos") {
        if let (Some(qv), Some(ix), Some(coll)) = (query_vec, index, collection) {
            let ids: Vec<String> = head.iter().map(|(id, _)| id.clone()).collect();
            let keys = store::keys_for_chunk_ids(conn, coll, &ids)
                .context("lookup embedding keys for rerank")?;
            for id in ids {
                let Some(key) = keys.get(&id).copied() else {
                    continue;
                };
                let Some(v) = ix.get(key)? else {
                    continue;
                };
                let mut dot = 0.0f64;
                for (a, b) in qv.iter().zip(v.iter()) {
                    dot += (*a as f64) * (*b as f64);
                }
                cos_by_id.insert(id, dot.clamp(-1.0, 1.0));
            }
        }
    }

    let mut reranked: Vec<(String, f64)> = Vec::with_capacity(head.len());
    for (id, rrf_score) in head {
        let bm25_norm = bm25_by_id.get(id.as_str()).copied().unwrap_or(0.0) / max_bm25;

        let vec_sim = cos_by_id
            .get(id)
            .copied()
            .or_else(|| vec_sim_by_chunk.get(id).copied())
            .unwrap_or(0.0);
        let vec_norm = vec_sim.max(0.0).min(1.0);

        let sym_bonus = symbol_bonus(conn, id, &query_terms).unwrap_or(0.0);

        let final_score = *rrf_score + 0.2 * bm25_norm + 0.2 * vec_norm + sym_bonus;
        reranked.push((id.clone(), final_score));
    }

    reranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut out = reranked;
    out.extend_from_slice(tail);
    Ok(out)
}

fn query_term_set(query: &str) -> HashMap<String, ()> {
    let mut out = HashMap::new();
    for tok in query
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_'))
    {
        if tok.is_empty() {
            continue;
        }
        out.insert(tok.to_ascii_lowercase(), ());
    }
    out
}

fn symbol_bonus(
    conn: &Connection,
    chunk_id: &str,
    query_terms: &HashMap<String, ()>,
) -> Result<f64> {
    let symbol: Option<String> = conn
        .query_row(
            "SELECT symbol FROM chunks WHERE chunk_id=?1 LIMIT 1",
            [chunk_id],
            |row| row.get(0),
        )
        .optional()
        .context("read chunks.symbol")?;

    let Some(sym) = symbol else {
        return Ok(0.0);
    };

    let sym_norm = sym
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase());

    for part in sym_norm {
        if query_terms.contains_key(&part) {
            return Ok(0.05);
        }
    }

    Ok(0.0)
}
