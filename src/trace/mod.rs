pub mod extract;
pub mod graph;
pub mod rank;
pub mod summarize;
pub mod types;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context as _, Result};
use rusqlite::Connection;

use crate::cli::{OutputArgs, QueryArgs, TraceArgs};
use crate::{config, semantic};

use self::types::{Citation, SymbolRecord, TraceResponse, TraceStageResult};

pub fn run(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    args: TraceArgs,
) -> Result<TraceResponse> {
    let (fast, fast_status) = run_stage(
        conn,
        db_path,
        cfg,
        &args.query,
        &args.lang,
        &args.path_prefix,
        args.limit_traces,
        args.max_hops_fast,
        cfg.trace.beam_fast,
        cfg.trace.fast_timeout_ms,
        false,
    )?;

    let (deep, deep_status) = if args.deep {
        let (deep_stage, status) = run_stage(
            conn,
            db_path,
            cfg,
            &args.query,
            &args.lang,
            &args.path_prefix,
            args.limit_traces,
            args.max_hops_deep,
            cfg.trace.beam_deep,
            cfg.trace.deep_timeout_ms,
            true,
        )?;
        (Some(deep_stage), status)
    } else {
        (None, "disabled".to_string())
    };

    let deep_status = if args.deep {
        if deep_status == "timeout" || fast_status == "timeout" {
            "timeout".to_string()
        } else {
            deep_status
        }
    } else {
        deep_status
    };

    Ok(TraceResponse {
        query: args.query,
        fast,
        deep,
        deep_status,
    })
}

pub fn run_fast_stage(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    query: &str,
    limit_traces: usize,
    max_hops_fast: usize,
    langs: &[String],
    path_prefixes: &[String],
) -> Result<(TraceStageResult, String)> {
    run_stage(
        conn,
        db_path,
        cfg,
        query,
        langs,
        path_prefixes,
        limit_traces,
        max_hops_fast,
        cfg.trace.beam_fast,
        cfg.trace.fast_timeout_ms,
        false,
    )
}

pub fn run_deep_stage(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    query: &str,
    limit_traces: usize,
    max_hops_deep: usize,
    langs: &[String],
    path_prefixes: &[String],
) -> Result<(TraceStageResult, String)> {
    run_stage(
        conn,
        db_path,
        cfg,
        query,
        langs,
        path_prefixes,
        limit_traces,
        max_hops_deep,
        cfg.trace.beam_deep,
        cfg.trace.deep_timeout_ms,
        true,
    )
}

fn run_stage(
    conn: &Connection,
    db_path: &Path,
    cfg: &config::Config,
    query: &str,
    langs: &[String],
    path_prefixes: &[String],
    limit_traces: usize,
    max_hops: usize,
    beam: usize,
    timeout_ms: u64,
    use_llm_summary: bool,
) -> Result<(TraceStageResult, String)> {
    let stage_name = if use_llm_summary { "deep" } else { "fast" };
    if let Some(cached) = graph::get_cached_stage(conn, query, stage_name)? {
        if should_use_cached_stage(cfg, use_llm_summary, &cached) {
            return Ok((cached, "ready".to_string()));
        }
    }

    let roots = collect_roots(conn, db_path, cfg, query, langs, path_prefixes)?;
    if roots.is_empty() {
        let symbols = graph::count_symbols(conn).unwrap_or(0);
        let summary = if symbols == 0 {
            "Trace graph is empty. Run `sx index` to populate symbols and edges.".to_string()
        } else {
            format!("No confident semantic flow was found for \"{query}\".")
        };
        let stage = TraceStageResult {
            traces: Vec::new(),
            summary,
            citations: Vec::new(),
            summary_source: "deterministic".to_string(),
            summary_model: None,
            summary_error: None,
        };
        return Ok((stage, "ready".to_string()));
    }

    let search_opts = rank::SearchOptions {
        limit_traces,
        max_hops,
        beam: beam.max(1),
        timeout_ms,
        edge_weights: cfg.trace.edge_weights.clone(),
    };
    let (traces, status) = rank::find_paths(conn, query, &roots, &search_opts)?;
    let citations = collect_citations(&traces);
    let deterministic = summarize::deterministic_summary(query, &traces, &citations);
    let decision = if use_llm_summary {
        summarize::maybe_llm_summary(cfg, query, &traces, &citations, &deterministic)
    } else {
        summarize::SummaryDecision::deterministic(deterministic)
    };

    let stage = TraceStageResult {
        traces,
        summary: decision.summary,
        citations,
        summary_source: decision.source,
        summary_model: decision.model,
        summary_error: decision.error,
    };
    let _ = graph::put_cached_stage(conn, query, stage_name, &stage);

    let st = match status {
        rank::SearchStatus::Ready => "ready",
        rank::SearchStatus::Timeout => "timeout",
    };
    Ok((stage, st.to_string()))
}

fn should_use_cached_stage(
    cfg: &config::Config,
    use_llm_summary: bool,
    cached: &TraceStageResult,
) -> bool {
    if !use_llm_summary {
        return true;
    }
    if !cfg.trace.llm_summary {
        return true;
    }
    if cfg.llm.provider.trim().eq_ignore_ascii_case("none") {
        return true;
    }
    cached.summary_source.eq_ignore_ascii_case("llm")
}

fn collect_roots(
    conn: &Connection,
    _db_path: &Path,
    cfg: &config::Config,
    query: &str,
    langs: &[String],
    path_prefixes: &[String],
) -> Result<Vec<(SymbolRecord, f64)>> {
    let q_args = QueryArgs {
        query: query.to_string(),
        limit: cfg.trace.candidate_roots.max(1),
        bm25_limit: (cfg.trace.candidate_roots * 5).max(20),
        vec_limit: (cfg.trace.candidate_roots * 5).max(20),
        deep: false,
        lang: langs.to_vec(),
        path_prefix: path_prefixes.to_vec(),
        output: OutputArgs::default(),
    };
    let ranked = semantic::query(conn, _db_path, cfg, q_args).context("trace root query")?;
    let chunk_ids: Vec<String> = ranked.iter().map(|r| r.chunk_id.clone()).collect();
    let by_chunk =
        graph::symbols_for_chunk_ids(conn, &chunk_ids).context("root symbols by chunk")?;

    let mut roots: HashMap<String, (SymbolRecord, f64)> = HashMap::new();

    for hit in &ranked {
        let mut candidates = by_chunk.get(&hit.chunk_id).cloned().unwrap_or_default();
        if candidates.is_empty() {
            candidates =
                graph::symbols_for_path_line(conn, &hit.path, hit.start_line).unwrap_or_default();
        }
        for sym in candidates {
            let entry = roots
                .entry(sym.symbol_id.clone())
                .or_insert((sym.clone(), hit.score));
            if hit.score > entry.1 {
                *entry = (sym, hit.score);
            }
        }
    }

    let mut roots_vec: Vec<(SymbolRecord, f64)> = roots.into_values().collect();
    if roots_vec.is_empty() {
        roots_vec = fallback_symbol_roots(conn, query, cfg.trace.candidate_roots)?;
    }
    roots_vec.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.path.cmp(&b.0.path)));
    roots_vec.truncate(cfg.trace.candidate_roots.max(1));
    Ok(roots_vec)
}

fn fallback_symbol_roots(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(SymbolRecord, f64)>> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect();
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut stmt = conn.prepare(
        r#"
SELECT symbol_id, chunk_id, path, start_line, end_line, language, kind, fq_name, short_name
FROM symbols
ORDER BY path ASC, start_line ASC
"#,
    )?;
    let rows = stmt.query_map([], |row| {
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
    })?;
    for row in rows {
        let sym = row?;
        let low = sym.short_name.to_ascii_lowercase();
        let mut score: f64 = 0.0;
        for t in &terms {
            if low.contains(t) {
                score += 0.5;
            }
        }
        if score > 0.0 && seen.insert(sym.symbol_id.clone()) {
            out.push((sym, score));
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.path.cmp(&b.0.path)));
    out.truncate(limit.max(1));
    Ok(out)
}

fn collect_citations(traces: &[types::TracePath]) -> Vec<Citation> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();

    for trace in traces {
        let root_key = format!("{}:{}", trace.root_path, trace.root_line);
        if seen.insert(root_key) {
            out.push(Citation {
                path: trace.root_path.clone(),
                line: trace.root_line,
            });
        }

        for step in &trace.steps {
            let k = format!("{}:{}", step.path, step.line);
            if seen.insert(k) {
                out.push(Citation {
                    path: step.path.clone(),
                    line: step.line,
                });
            }
        }
    }
    out
}
