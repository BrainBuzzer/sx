mod deep;
mod embedder;
mod hybrid;
mod llm;
mod store;
mod vector_index;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow};
use rusqlite::Connection;

use crate::cli::{EmbedArgs, QueryArgs, VsearchArgs};
use crate::{config, db, search};

pub struct EmbedStats {
    pub collection: String,
    pub embedded_new: usize,
    pub embedded_kept: usize,
    pub pruned: usize,
    pub dim: usize,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
struct CollectionInfo {
    collection: String,
    provider: EmbedProvider,
    model: String,
    dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedProvider {
    Ollama,
    OpenAi,
    Voyage,
}

impl EmbedProvider {
    fn as_str(&self) -> &'static str {
        match self {
            EmbedProvider::Ollama => "ollama",
            EmbedProvider::OpenAi => "openai",
            EmbedProvider::Voyage => "voyage",
        }
    }

    fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(EmbedProvider::Ollama),
            "openai" => Some(EmbedProvider::OpenAi),
            "voyage" | "voyageai" => Some(EmbedProvider::Voyage),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbedConnectivity {
    pub provider: String,
    pub model: String,
    pub dim: usize,
}

pub fn probe_embedding_connectivity(
    cfg: &config::Config,
    provider_override: Option<&str>,
    model_override: Option<&str>,
    dimensions_override: Option<usize>,
) -> Result<EmbedConnectivity> {
    let provider_raw = provider_override.unwrap_or(cfg.embed.provider.as_str());
    if provider_raw.trim().eq_ignore_ascii_case("none") {
        return Err(anyhow!(
            "embedding provider is 'none' (set [embed].provider first)"
        ));
    }
    let provider = EmbedProvider::parse(provider_raw)
        .ok_or_else(|| anyhow!("unsupported embedding provider: {provider_raw}"))?;
    let model = model_override
        .unwrap_or(cfg.embed.model.as_str())
        .to_string();

    let preferred_dim = match provider {
        EmbedProvider::OpenAi => Some(dimensions_override.unwrap_or(cfg.embed.dimensions).max(1)),
        EmbedProvider::Ollama | EmbedProvider::Voyage => None,
    };

    let embedder =
        embedder::build_embedder(cfg, provider, &model, preferred_dim).context("build embedder")?;
    let probe_text = "sx embedding connectivity probe".to_string();
    let vectors = embedder
        .embed_batch(&[probe_text])
        .context("request embedding")?;
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    if dim == 0 {
        return Err(anyhow!("embedding probe returned empty vector"));
    }

    Ok(EmbedConnectivity {
        provider: provider.as_str().to_string(),
        model,
        dim,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlmProvider {
    None,
    Ollama,
    OpenAi,
    Zhipu,
}

impl LlmProvider {
    fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "none" => Some(LlmProvider::None),
            "ollama" => Some(LlmProvider::Ollama),
            "openai" => Some(LlmProvider::OpenAi),
            "zhipu" => Some(LlmProvider::Zhipu),
            "codex" => Some(LlmProvider::Zhipu),
            _ => None,
        }
    }
}

pub fn embed(
    conn: &mut Connection,
    db_path: &Path,
    cfg: &config::Config,
    args: EmbedArgs,
) -> Result<EmbedStats> {
    let started = Instant::now();

    let provider_raw = args
        .provider
        .clone()
        .unwrap_or_else(|| cfg.embed.provider.clone());
    if provider_raw.trim().eq_ignore_ascii_case("none") {
        return Err(anyhow!(
            "embedding provider is 'none' (set [embed].provider or pass --provider)"
        ));
    }
    let provider = EmbedProvider::parse(&provider_raw)
        .ok_or_else(|| anyhow!("unsupported embedding provider: {provider_raw}"))?;

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| cfg.embed.model.clone());
    let batch_size = args.batch_size.unwrap_or(cfg.embed.batch_size).max(1);
    let max_chars = cfg.embed.max_chars.max(1);

    let preferred_dim = match provider {
        EmbedProvider::OpenAi => Some(args.dimensions.unwrap_or(cfg.embed.dimensions).max(1)),
        EmbedProvider::Ollama | EmbedProvider::Voyage => None,
    };

    let dim = resolve_dim_for_embed(conn, cfg, provider, &model, preferred_dim)
        .context("resolve embedding dimension")?;

    let collection = format!("{}:{}:{dim}", provider.as_str(), model);
    let vectors_dir = vectors_dir_for_db(db_path)?;
    let collection_dir = vectors_dir.join(slugify(&collection));
    let index_path = collection_dir.join("index.usearch");
    let meta_path = collection_dir.join("meta.toml");

    std::fs::create_dir_all(&collection_dir)
        .with_context(|| format!("create {}", collection_dir.display()))?;

    if args.full {
        store::drop_collection(conn, &collection).context("drop collection rows")?;
        if index_path.exists() {
            std::fs::remove_file(&index_path)
                .with_context(|| format!("remove {}", index_path.display()))?;
        }
    }

    // If the index file is missing but rows exist, force a rebuild by clearing rows.
    if !index_path.exists() {
        let existing = store::count_embeddings(conn, &collection).unwrap_or(0);
        if existing > 0 {
            store::drop_collection(conn, &collection).context("drop stale collection rows")?;
        }
    }

    let embedder =
        embedder::build_embedder(cfg, provider, &model, preferred_dim).context("build embedder")?;

    let mut index = vector_index::open_for_write(cfg, dim, &index_path)
        .with_context(|| format!("open vector index {}", index_path.display()))?;

    store::upsert_collection(
        conn,
        &collection,
        provider.as_str(),
        &model,
        dim as i64,
        &cfg.vector.metric,
        &cfg.vector.quantization,
    )
    .context("upsert vector_collections")?;

    let before_total = store::count_embeddings(conn, &collection).unwrap_or(0);

    // Prune embeddings for content_hashes that no longer exist in chunks.
    let orphans =
        store::list_orphan_embeddings(conn, &collection).context("list orphan embeddings")?;
    for orphan in &orphans {
        let key_u64 = orphan.key_i64 as u64;
        let _ = index.remove(key_u64).ok();
        store::delete_embedding(conn, &collection, &orphan.content_hash)
            .with_context(|| format!("delete embedding {}", orphan.content_hash))?;
    }

    let embedded_kept = store::count_embeddings(conn, &collection).unwrap_or(0);
    let pruned = before_total.saturating_sub(embedded_kept);

    // Embed missing unique content hashes.
    let missing =
        store::list_missing_content_hashes(conn, &collection).context("list missing hashes")?;
    if missing.is_empty() {
        index
            .save(index_path.to_string_lossy().as_ref())
            .context("save vector index")?;
        write_collection_meta(&meta_path, &collection, provider, &model, dim, cfg)
            .context("write meta.toml")?;
        return Ok(EmbedStats {
            collection,
            embedded_new: 0,
            embedded_kept,
            pruned,
            dim,
            duration_ms: started.elapsed().as_millis(),
        });
    }

    index
        .reserve((embedded_kept + missing.len()).max(1))
        .context("reserve vector index capacity")?;

    let mut embedded_new = 0usize;
    let now = db::now_unix();

    let mut batch_hashes: Vec<String> = Vec::new();
    let mut batch_texts: Vec<String> = Vec::new();

    for content_hash in missing {
        let content = store::fetch_representative_content_for_hash(conn, &content_hash)
            .with_context(|| format!("fetch content for {content_hash}"))?;
        let input = truncate_chars(&content, max_chars);
        batch_hashes.push(content_hash);
        batch_texts.push(input);

        if batch_texts.len() >= batch_size {
            embedded_new += embed_batch_into_index(
                conn,
                &collection,
                dim,
                now,
                embedder.as_ref(),
                &mut index,
                std::mem::take(&mut batch_hashes),
                std::mem::take(&mut batch_texts),
            )?;
        }
    }

    if !batch_texts.is_empty() {
        embedded_new += embed_batch_into_index(
            conn,
            &collection,
            dim,
            now,
            embedder.as_ref(),
            &mut index,
            batch_hashes,
            batch_texts,
        )?;
    }

    index
        .save(index_path.to_string_lossy().as_ref())
        .context("save vector index")?;

    write_collection_meta(&meta_path, &collection, provider, &model, dim, cfg)
        .context("write meta.toml")?;

    Ok(EmbedStats {
        collection,
        embedded_new,
        embedded_kept,
        pruned,
        dim,
        duration_ms: started.elapsed().as_millis(),
    })
}

fn embed_batch_into_index(
    conn: &Connection,
    collection: &str,
    dim: usize,
    now: i64,
    embedder: &dyn embedder::Embedder,
    index: &mut vector_index::VectorIndex,
    content_hashes: Vec<String>,
    texts: Vec<String>,
) -> Result<usize> {
    if content_hashes.is_empty() {
        return Ok(0);
    }
    let vectors = embedder.embed_batch(&texts).context("embed batch")?;

    if vectors.len() != content_hashes.len() {
        return Err(anyhow!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            content_hashes.len()
        ));
    }

    let mut embedded = 0usize;

    for (hash, mut vec) in content_hashes.into_iter().zip(vectors.into_iter()) {
        if vec.len() != dim {
            return Err(anyhow!(
                "embedding dimension mismatch for {hash}: expected {dim}, got {}",
                vec.len()
            ));
        }
        vector_index::normalize_in_place(&mut vec);

        // Collision-resistant deterministic key.
        let mut salt = 0usize;
        let key_u64 = loop {
            let k = key_for_hash(&hash, salt);
            if let Some(existing) = store::content_hash_for_key(conn, collection, k as i64)? {
                if existing == hash {
                    break k;
                }
                salt += 1;
                continue;
            }
            break k;
        };

        index.add(key_u64, &vec).context("index.add")?;
        store::upsert_embedding(conn, collection, &hash, key_u64 as i64, now)
            .with_context(|| format!("upsert embedding row for {hash}"))?;
        embedded += 1;
    }

    Ok(embedded)
}

pub fn vsearch(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    args: VsearchArgs,
) -> Result<Vec<search::SearchResult>> {
    let provider_raw = cfg.embed.provider.trim();
    if provider_raw.eq_ignore_ascii_case("none") {
        return Err(anyhow!(
            "vectors not built; run `sx embed` (config [embed].provider is 'none')"
        ));
    }
    let provider = EmbedProvider::parse(provider_raw)
        .ok_or_else(|| anyhow!("unsupported embedding provider: {provider_raw}"))?;

    let model = cfg.embed.model.clone();
    let collection =
        resolve_collection_for_query(conn, provider, &model, Some(cfg.embed.dimensions))
            .ok_or_else(|| anyhow!("vectors not built; run `sx embed`"))?;

    let vectors_dir = vectors_dir_for_db(db_path)?;
    let index_path = vectors_dir
        .join(slugify(&collection.collection))
        .join("index.usearch");
    if !index_path.exists() {
        return Err(anyhow!("vectors not built; run `sx embed`"));
    }

    let embedder = embedder::build_embedder(cfg, provider, &model, Some(collection.dim))
        .context("build embedder")?;

    let query_vec = get_or_embed_query(
        conn,
        &collection.collection,
        collection.dim,
        embedder.as_ref(),
        &args.query,
        cfg.embed.max_chars,
    )
    .context("embed query")?;

    let index = vector_index::open_for_read(cfg, collection.dim, &index_path)
        .with_context(|| format!("open vector index {}", index_path.display()))?;

    let oversample = args.limit.saturating_mul(5).max(args.limit).max(1);
    let matches = index
        .search(&query_vec, oversample)
        .context("vector search")?;

    let key_order: Vec<u64> = matches.iter().map(|(k, _)| *k).collect();
    let sim_by_key: HashMap<u64, f64> = matches
        .into_iter()
        .map(|(k, dist)| (k, distance_to_similarity(dist, &cfg.vector.metric)))
        .collect();

    let selected = store::representative_chunks_for_keys(
        conn,
        &collection.collection,
        &key_order,
        &args.lang,
        &args.path_prefix,
    )
    .context("lookup representative chunks")?;

    let mut out = Vec::new();
    for key in key_order {
        if out.len() >= args.limit {
            break;
        }
        let Some(chunk) = selected.get(&key) else {
            continue;
        };
        let content = search::get_chunk_by_id(conn, &chunk.chunk_id)?
            .map(|c| c.content)
            .unwrap_or_default();
        let snippet = snippet_from_content(&content);
        let score = sim_by_key.get(&key).copied().unwrap_or(0.0);

        out.push(search::SearchResult {
            chunk_id: chunk.chunk_id.clone(),
            path: chunk.path.clone(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            kind: chunk.kind.clone(),
            symbol: chunk.symbol.clone(),
            score,
            snippet,
        });
    }

    Ok(out)
}

pub fn query(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    args: QueryArgs,
) -> Result<Vec<search::SearchResult>> {
    let started = Instant::now();

    let bm25_opts = search::SearchOptions {
        query: args.query.clone(),
        limit: args.bm25_limit,
        literal: true,
        fts: false,
        langs: args.lang.clone(),
        path_prefixes: args.path_prefix.clone(),
    };
    let bm25_results = search::search(conn, &bm25_opts).context("bm25 search")?;

    let provider_raw = cfg.embed.provider.trim();
    let provider = EmbedProvider::parse(provider_raw);
    let model = cfg.embed.model.clone();
    let collection = provider
        .and_then(|p| resolve_collection_for_query(conn, p, &model, Some(cfg.embed.dimensions)));

    let mut vec_results: Vec<search::SearchResult> = Vec::new();
    let mut query_vec: Option<Vec<f32>> = None;
    let mut index: Option<vector_index::VectorIndex> = None;
    let mut vec_sim_by_chunk: HashMap<String, f64> = HashMap::new();

    if let Some(collection) = &collection {
        let vectors_dir = vectors_dir_for_db(db_path)?;
        let index_path = vectors_dir
            .join(slugify(&collection.collection))
            .join("index.usearch");
        if index_path.exists() {
            if let Ok(embedder) = embedder::build_embedder(
                cfg,
                collection.provider,
                &collection.model,
                Some(collection.dim),
            ) {
                if let Ok(qv) = get_or_embed_query(
                    conn,
                    &collection.collection,
                    collection.dim,
                    embedder.as_ref(),
                    &args.query,
                    cfg.embed.max_chars,
                ) {
                    if let Ok(ix) = vector_index::open_for_read(cfg, collection.dim, &index_path) {
                        let matches = ix.search(&qv, args.vec_limit.max(1)).unwrap_or_default();
                        let key_order: Vec<u64> = matches.iter().map(|(k, _)| *k).collect();
                        let sim_by_key: HashMap<u64, f64> = matches
                            .into_iter()
                            .map(|(k, dist)| (k, distance_to_similarity(dist, &cfg.vector.metric)))
                            .collect();

                        let selected = store::representative_chunks_for_keys(
                            conn,
                            &collection.collection,
                            &key_order,
                            &args.lang,
                            &args.path_prefix,
                        )
                        .unwrap_or_default();

                        for key in key_order {
                            if vec_results.len() >= args.vec_limit {
                                break;
                            }
                            let Some(chunk) = selected.get(&key) else {
                                continue;
                            };
                            let content = search::get_chunk_by_id(conn, &chunk.chunk_id)?
                                .map(|c| c.content)
                                .unwrap_or_default();
                            let snippet = snippet_from_content(&content);
                            let score = sim_by_key.get(&key).copied().unwrap_or(0.0);
                            vec_sim_by_chunk.insert(chunk.chunk_id.clone(), score);
                            vec_results.push(search::SearchResult {
                                chunk_id: chunk.chunk_id.clone(),
                                path: chunk.path.clone(),
                                start_line: chunk.start_line,
                                end_line: chunk.end_line,
                                kind: chunk.kind.clone(),
                                symbol: chunk.symbol.clone(),
                                score,
                                snippet,
                            });
                        }

                        query_vec = Some(qv);
                        index = Some(ix);
                    }
                }
            }
        }
    }

    let mut fused = hybrid::rrf_fuse(
        cfg.query.rrf_k,
        &[
            hybrid::RankedList {
                weight: cfg.query.bm25_weight,
                items: bm25_results.iter().map(|r| r.chunk_id.clone()).collect(),
            },
            hybrid::RankedList {
                weight: cfg.query.vec_weight,
                items: vec_results.iter().map(|r| r.chunk_id.clone()).collect(),
            },
        ],
    );

    let mut expansion_results: Vec<search::SearchResult> = Vec::new();
    let mut expansion_lists: Vec<Vec<String>> = Vec::new();

    if args.deep && !bm25_results.is_empty() {
        let seed_limit = cfg.deep.seed_limit.max(1);
        let mut seed_chunks = Vec::new();
        for r in bm25_results.iter().take(seed_limit) {
            if let Some(ch) = search::get_chunk_by_id(conn, &r.chunk_id)? {
                seed_chunks.push(ch);
            }
        }

        let prf_terms = deep::prf_terms(&args.query, &seed_chunks, cfg.deep.max_terms);
        if !prf_terms.is_empty() {
            let fts_query = deep::fts_or_query(&prf_terms);
            let opts = search::SearchOptions {
                query: fts_query,
                limit: args.bm25_limit,
                literal: false,
                fts: true,
                langs: args.lang.clone(),
                path_prefixes: args.path_prefix.clone(),
            };
            expansion_results = search::search(conn, &opts).unwrap_or_default();
            expansion_lists.push(
                expansion_results
                    .iter()
                    .map(|r| r.chunk_id.clone())
                    .collect(),
            );
        }

        if cfg.deep.llm_expand && !cfg.llm.provider.trim().eq_ignore_ascii_case("none") {
            let llm_provider = LlmProvider::parse(&cfg.llm.provider).unwrap_or(LlmProvider::None);
            if llm_provider != LlmProvider::None {
                let hints: Vec<String> = prf_terms.iter().take(12).cloned().collect();
                if let Ok(expansions) =
                    llm::expand_queries_cached(conn, cfg, llm_provider, &args.query, &hints)
                {
                    for q in expansions {
                        let opts = search::SearchOptions {
                            query: q,
                            limit: args.bm25_limit,
                            literal: true,
                            fts: false,
                            langs: args.lang.clone(),
                            path_prefixes: args.path_prefix.clone(),
                        };
                        let res = search::search(conn, &opts).unwrap_or_default();
                        expansion_lists.push(res.iter().map(|r| r.chunk_id.clone()).collect());
                    }
                }
            }
        }
    }

    if !expansion_lists.is_empty() {
        let extra = hybrid::rrf_fuse(
            cfg.query.rrf_k,
            &expansion_lists
                .into_iter()
                .map(|items| hybrid::RankedList {
                    weight: cfg.query.expansion_bm25_weight,
                    items,
                })
                .collect::<Vec<_>>(),
        );
        for (k, v) in extra {
            *fused.entry(k).or_insert(0.0) += v;
        }
    }

    // Build best-available templates for output.
    let mut by_id: HashMap<String, search::SearchResult> = HashMap::new();
    for r in bm25_results
        .iter()
        .chain(vec_results.iter())
        .chain(expansion_results.iter())
    {
        by_id.entry(r.chunk_id.clone()).or_insert_with(|| r.clone());
    }

    let mut scored: Vec<(String, f64)> = fused.into_iter().collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    if args.deep {
        if cfg.deep.llm_rerank {
            let candidates = build_voyage_rerank_candidates(
                conn,
                &scored,
                cfg.deep.rerank_top,
                cfg.embed.max_chars.min(2000),
            )
            .unwrap_or_default();
            if !candidates.is_empty() {
                if let Ok(scores_by_chunk) =
                    llm::rerank_with_voyage_cached(conn, cfg, &args.query, &candidates)
                {
                    scored =
                        apply_llm_rerank_scores(&scored, cfg.deep.rerank_top, &scores_by_chunk);
                } else {
                    scored = hybrid::deterministic_rerank(
                        &scored,
                        cfg.deep.rerank_top,
                        &bm25_results,
                        &vec_sim_by_chunk,
                        &args.query,
                        query_vec.as_deref(),
                        index.as_ref(),
                        conn,
                        collection.as_ref().map(|c| c.collection.as_str()),
                        &cfg.vector.metric,
                    )
                    .unwrap_or(scored);
                }
            } else {
                scored = hybrid::deterministic_rerank(
                    &scored,
                    cfg.deep.rerank_top,
                    &bm25_results,
                    &vec_sim_by_chunk,
                    &args.query,
                    query_vec.as_deref(),
                    index.as_ref(),
                    conn,
                    collection.as_ref().map(|c| c.collection.as_str()),
                    &cfg.vector.metric,
                )
                .unwrap_or(scored);
            }
        } else {
            scored = hybrid::deterministic_rerank(
                &scored,
                cfg.deep.rerank_top,
                &bm25_results,
                &vec_sim_by_chunk,
                &args.query,
                query_vec.as_deref(),
                index.as_ref(),
                conn,
                collection.as_ref().map(|c| c.collection.as_str()),
                &cfg.vector.metric,
            )
            .unwrap_or(scored);
        }
    }

    let mut out = Vec::new();
    for (chunk_id, score) in scored.into_iter() {
        if out.len() >= args.limit {
            break;
        }
        let mut base = if let Some(t) = by_id.get(&chunk_id).cloned() {
            t
        } else if let Some(ch) = search::get_chunk_by_id(conn, &chunk_id)? {
            let snippet = snippet_from_content(&ch.content);
            search::SearchResult {
                chunk_id: ch.chunk_id,
                path: ch.path,
                start_line: ch.start_line,
                end_line: ch.end_line,
                kind: ch.kind,
                symbol: ch.symbol,
                score,
                snippet,
            }
        } else {
            continue;
        };
        base.score = score;
        if base.snippet.is_empty() {
            let content = search::get_chunk_by_id(conn, &chunk_id)?
                .map(|c| c.content)
                .unwrap_or_default();
            base.snippet = snippet_from_content(&content);
        }
        out.push(base);
    }

    tracing::debug!(
        deep = args.deep,
        bm25 = bm25_results.len(),
        vec = vec_results.len(),
        out = out.len(),
        duration_ms = started.elapsed().as_millis(),
        "query"
    );

    Ok(out)
}

fn build_voyage_rerank_candidates(
    conn: &Connection,
    scored: &[(String, f64)],
    rerank_top: usize,
    max_chars: usize,
) -> Result<Vec<llm::RerankCandidate>> {
    if scored.is_empty() {
        return Ok(Vec::new());
    }

    let top = rerank_top.max(1).min(scored.len());
    let mut out = Vec::with_capacity(top);

    for (chunk_id, _) in scored.iter().take(top) {
        let Some(chunk) = search::get_chunk_by_id(conn, chunk_id)? else {
            continue;
        };
        let symbol = chunk.symbol.as_deref().unwrap_or("");
        let content = truncate_chars(&chunk.content, max_chars.max(1));
        let text = format!(
            "path: {}\nlines: {}-{}\nkind: {}\nsymbol: {}\n\n{}",
            chunk.path, chunk.start_line, chunk.end_line, chunk.kind, symbol, content
        );
        out.push(llm::RerankCandidate {
            chunk_id: chunk.chunk_id,
            content_hash: if chunk.content_hash.trim().is_empty() {
                chunk_id.clone()
            } else {
                chunk.content_hash
            },
            text,
        });
    }

    Ok(out)
}

fn apply_llm_rerank_scores(
    scored: &[(String, f64)],
    rerank_top: usize,
    scores_by_chunk: &HashMap<String, f64>,
) -> Vec<(String, f64)> {
    if scored.is_empty() {
        return Vec::new();
    }

    let top = rerank_top.max(1).min(scored.len());
    let head = &scored[..top];
    let tail = &scored[top..];

    let mut reranked: Vec<(String, f64, f64)> = head
        .iter()
        .map(|(id, base_score)| {
            let llm = scores_by_chunk
                .get(id)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            (id.clone(), *base_score, llm)
        })
        .collect();

    reranked.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut out: Vec<(String, f64)> = reranked
        .into_iter()
        .map(|(id, base, llm)| {
            let score = if llm.is_finite() { llm } else { base };
            (id, score)
        })
        .collect();
    out.extend_from_slice(tail);
    out
}

fn resolve_dim_for_embed(
    conn: &Connection,
    cfg: &config::Config,
    provider: EmbedProvider,
    model: &str,
    preferred_dim: Option<usize>,
) -> Result<usize> {
    if let Some(dim) = preferred_dim {
        return Ok(dim);
    }

    if let Some(dim) = store::latest_collection_dim(conn, provider.as_str(), model)? {
        return Ok(dim as usize);
    }

    infer_dim_from_provider(conn, cfg, provider, model)
}

fn infer_dim_from_provider(
    conn: &Connection,
    cfg: &config::Config,
    provider: EmbedProvider,
    model: &str,
) -> Result<usize> {
    // No existing collection and no preferred dimension (Ollama).
    // Use the first available chunk to infer.
    let sample = store::fetch_any_chunk_content(conn)
        .context("no chunks to infer embedding dimension (run `sx index` first)")?;

    let embedder = embedder::build_embedder(cfg, provider, model, None)?;
    let vecs = embedder.embed_batch(&[truncate_chars(&sample, 8000)])?;
    let dim = vecs.get(0).map(|v| v.len()).unwrap_or(0);
    if dim == 0 {
        return Err(anyhow!("failed to infer embedding dimension"));
    }
    Ok(dim)
}

fn resolve_collection_for_query(
    conn: &Connection,
    provider: EmbedProvider,
    model: &str,
    preferred_dim: Option<usize>,
) -> Option<CollectionInfo> {
    store::resolve_collection(conn, provider.as_str(), model, preferred_dim)
        .ok()
        .flatten()
        .map(|c| CollectionInfo {
            collection: c.collection,
            provider,
            model: c.model,
            dim: c.dim as usize,
        })
}

fn vectors_dir_for_db(db_path: &Path) -> Result<PathBuf> {
    let dir = db_path
        .parent()
        .ok_or_else(|| anyhow!("db path has no parent directory"))?;
    Ok(dir.join("vectors"))
}

fn slugify(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        if ok {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "collection".to_string()
    } else {
        out
    }
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect()
}

fn key_for_hash(content_hash: &str, salt: usize) -> u64 {
    let s = if salt == 0 {
        content_hash.to_string()
    } else {
        format!("{content_hash}#{salt}")
    };
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(blake3::hash(s.as_bytes()).as_bytes());
    let mut u = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or_default());
    u &= 0x7fff_ffff_ffff_ffff;
    u
}

fn distance_to_similarity(distance: f32, metric: &str) -> f64 {
    if metric.trim().eq_ignore_ascii_case("cos") {
        (1.0 - distance as f64).clamp(-1.0, 1.0)
    } else {
        -(distance as f64)
    }
}

fn snippet_from_content(content: &str) -> String {
    let mut s = String::new();
    for line in content.lines().take(3) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(t);
        if s.len() > 200 {
            truncate_to_char_boundary(&mut s, 200);
            s.push('…');
            break;
        }
    }
    s
}

fn truncate_to_char_boundary(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut idx = max_bytes.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    s.truncate(idx);
}

fn get_or_embed_query(
    conn: &Connection,
    collection: &str,
    dim: usize,
    embedder: &dyn embedder::Embedder,
    query: &str,
    max_chars: usize,
) -> Result<Vec<f32>> {
    if let Some(v) = store::get_cached_query_embedding(conn, collection, query, dim)? {
        return Ok(v);
    }

    let q = truncate_chars(query, max_chars.max(1));
    let vecs = embedder.embed_batch(&[q]).context("embed query batch")?;
    let mut v = vecs.into_iter().next().unwrap_or_default();
    if v.len() != dim {
        return Err(anyhow!(
            "query embedding dimension mismatch: expected {dim}, got {}",
            v.len()
        ));
    }
    vector_index::normalize_in_place(&mut v);
    store::put_cached_query_embedding(conn, collection, query, dim, &v, db::now_unix())?;
    Ok(v)
}

fn write_collection_meta(
    path: &Path,
    collection: &str,
    provider: EmbedProvider,
    model: &str,
    dim: usize,
    cfg: &config::Config,
) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Meta<'a> {
        collection: &'a str,
        provider: &'a str,
        model: &'a str,
        dim: usize,
        metric: &'a str,
        quantization: &'a str,
    }

    let meta = Meta {
        collection,
        provider: provider.as_str(),
        model,
        dim,
        metric: cfg.vector.metric.as_str(),
        quantization: cfg.vector.quantization.as_str(),
    };
    let body = toml::to_string_pretty(&meta).context("serialize meta")?;
    std::fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
