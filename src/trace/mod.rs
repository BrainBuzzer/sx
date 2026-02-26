pub mod edge_provider;
pub mod extract;
pub mod graph;
pub mod rank;
pub mod summarize;
pub mod types;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context as _, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::cli::{OutputArgs, QueryArgs, TraceArgs};
use crate::index::scan::is_test_case_path;
use crate::{config, semantic};

use self::types::{Citation, SymbolRecord, TraceResponse, TraceStageResult};

pub fn run(
    conn: &Connection,
    root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    args: TraceArgs,
) -> Result<TraceResponse> {
    let go_lsp_mode = resolve_go_lsp_mode(root_dir, cfg, args.lsp, args.no_lsp)?;
    let (fast, fast_status) = run_stage(
        conn,
        root_dir,
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
        go_lsp_mode,
    )?;

    let (deep, deep_status) = if args.deep {
        let (deep_stage, status) = run_stage(
            conn,
            root_dir,
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
            go_lsp_mode,
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
    root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    query: &str,
    limit_traces: usize,
    max_hops_fast: usize,
    langs: &[String],
    path_prefixes: &[String],
) -> Result<(TraceStageResult, String)> {
    let go_lsp_mode = resolve_go_lsp_mode(root_dir, cfg, false, false)?;
    run_stage(
        conn,
        root_dir,
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
        go_lsp_mode,
    )
}

pub fn run_deep_stage(
    conn: &Connection,
    root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    query: &str,
    limit_traces: usize,
    max_hops_deep: usize,
    langs: &[String],
    path_prefixes: &[String],
) -> Result<(TraceStageResult, String)> {
    let go_lsp_mode = resolve_go_lsp_mode(root_dir, cfg, false, false)?;
    run_stage(
        conn,
        root_dir,
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
        go_lsp_mode,
    )
}

fn run_stage(
    conn: &Connection,
    root_dir: &Path,
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
    go_lsp_mode: GoLspMode,
) -> Result<(TraceStageResult, String)> {
    let stage_name = if use_llm_summary { "deep" } else { "fast" };
    let stage_key = stage_cache_key(
        conn,
        stage_name,
        go_lsp_mode,
        limit_traces,
        max_hops,
        beam,
        timeout_ms,
        langs,
        path_prefixes,
        cfg,
    )?;

    if let Some(cached) = graph::get_cached_stage(conn, query, stage_key.as_str())? {
        if !stage_contains_test_paths(&cached)
            && should_use_cached_stage(cfg, use_llm_summary, &cached)
        {
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

    let mut db_edges = edge_provider::DbEdgeProvider;
    let mut lsp_edges = edge_provider::HybridEdgeProvider::new(root_dir, cfg);
    let forced_lsp = matches!(go_lsp_mode, GoLspMode::Forced);
    let gopls_timeout_ms = if forced_lsp || use_llm_summary {
        cfg.lsp.go.timeout_ms
    } else {
        // Keep fast-stage tracing responsive under load by bounding per-call gopls time (auto mode).
        cfg.lsp.go.timeout_ms.min((timeout_ms / 2).max(250))
    };
    lsp_edges.set_gopls_timeout_ms(gopls_timeout_ms);
    let use_lsp_edges =
        matches!(go_lsp_mode, GoLspMode::Enabled | GoLspMode::Forced) && lsp_edges.gopls_available();
    let provider: &mut dyn rank::EdgeProvider = if use_lsp_edges {
        &mut lsp_edges
    } else {
        &mut db_edges
    };

    let (traces, status) = rank::find_paths(conn, query, &roots, &search_opts, provider)?;
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
    let _ = graph::put_cached_stage(conn, query, stage_key.as_str(), &stage);

    let st = match status {
        rank::SearchStatus::Ready => "ready",
        rank::SearchStatus::Timeout => "timeout",
    };
    Ok((stage, st.to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoLspMode {
    Disabled,
    Enabled,
    Forced,
}

fn resolve_go_lsp_mode(
    root_dir: &Path,
    cfg: &config::Config,
    force_lsp: bool,
    force_no_lsp: bool,
) -> Result<GoLspMode> {
    if force_no_lsp {
        return Ok(GoLspMode::Disabled);
    }

    if !cfg.lsp.enabled || !cfg.lsp.go.enabled {
        if force_lsp {
            return Err(anyhow::anyhow!(
                "Go LSP is disabled. Enable it in .sx/config.toml under [lsp] and [lsp.go]."
            ));
        }
        return Ok(GoLspMode::Disabled);
    }

    let want = if force_lsp {
        true
    } else {
        cfg.trace.go_use_gopls_calls
    };

    if !want {
        return Ok(GoLspMode::Disabled);
    }

    let probe = crate::lsp::go::GoplsRunner::from_config(root_dir, cfg);
    if probe.is_available() {
        if force_lsp {
            Ok(GoLspMode::Forced)
        } else {
            Ok(GoLspMode::Enabled)
        }
    } else if force_lsp {
        Err(anyhow::anyhow!(
            "gopls is not available (expected `{}`). Install it with: `go install golang.org/x/tools/gopls@latest`",
            cfg.lsp.go.gopls_path
        ))
    } else {
        eprintln!(
            "warning: gopls not found; falling back to heuristic Go trace edges (install with: `go install golang.org/x/tools/gopls@latest`)"
        );
        Ok(GoLspMode::Disabled)
    }
}

fn stage_cache_key(
    conn: &Connection,
    stage: &str,
    go_lsp_mode: GoLspMode,
    limit_traces: usize,
    max_hops: usize,
    beam: usize,
    timeout_ms: u64,
    langs: &[String],
    path_prefixes: &[String],
    cfg: &config::Config,
) -> Result<String> {
    #[derive(serde::Serialize)]
    struct Key<'a> {
        stage: &'a str,
        index_generation: i64,
        go_lsp_mode: u8,
        limit_traces: usize,
        max_hops: usize,
        beam: usize,
        timeout_ms: u64,
        langs: &'a [String],
        path_prefixes: &'a [String],
        candidate_roots: usize,
        edge_weights: &'a config::TraceEdgeWeights,
    }

    let generation = index_generation(conn)?;
    let key = Key {
        stage,
        index_generation: generation,
        go_lsp_mode: match go_lsp_mode {
            GoLspMode::Disabled => 0,
            GoLspMode::Enabled => 1,
            GoLspMode::Forced => 2,
        },
        limit_traces,
        max_hops,
        beam,
        timeout_ms,
        langs,
        path_prefixes,
        candidate_roots: cfg.trace.candidate_roots,
        edge_weights: &cfg.trace.edge_weights,
    };
    let body = serde_json::to_vec(&key).context("serialize trace stage cache key")?;
    let hash = blake3::hash(&body).to_hex().to_string();
    Ok(format!(
        "{stage}:g{generation}:go_lsp{}:{}",
        key.go_lsp_mode,
        &hash[..12]
    ))
}

fn index_generation(conn: &Connection) -> Result<i64> {
    let generation: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='index_generation'",
            [],
            |row| row.get(0),
        )
        .optional()
        .context("read meta.index_generation")?;
    Ok(generation
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0))
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
        if is_test_case_path(&sym.path) {
            continue;
        }
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
        if is_test_case_path(&trace.root_path) {
            continue;
        }
        let root_key = format!("{}:{}", trace.root_path, trace.root_line);
        if seen.insert(root_key) {
            out.push(Citation {
                path: trace.root_path.clone(),
                line: trace.root_line,
            });
        }

        for step in &trace.steps {
            if is_test_case_path(&step.path) {
                continue;
            }
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

fn stage_contains_test_paths(stage: &TraceStageResult) -> bool {
    for trace in &stage.traces {
        if is_test_case_path(&trace.root_path) {
            return true;
        }
        for step in &trace.steps {
            if is_test_case_path(&step.path) {
                return true;
            }
        }
    }
    stage.citations.iter().any(|c| is_test_case_path(&c.path))
}
