use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::TraceEdgeWeights;

use super::graph;
use super::types::{EdgeRecord, SymbolRecord, TracePath, TraceStep};
use rusqlite::Connection;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit_traces: usize,
    pub max_hops: usize,
    pub beam: usize,
    pub timeout_ms: u64,
    pub edge_weights: TraceEdgeWeights,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchStatus {
    Ready,
    Timeout,
}

pub trait EdgeProvider {
    fn outgoing_edges(&mut self, conn: &Connection, symbol_id: &str) -> Result<Vec<EdgeRecord>>;
}

pub fn find_paths(
    conn: &Connection,
    query: &str,
    roots: &[(SymbolRecord, f64)],
    opts: &SearchOptions,
    edge_provider: &mut dyn EdgeProvider,
) -> Result<(Vec<TracePath>, SearchStatus)> {
    let query_terms = query_terms(query);
    let mut all_paths = Vec::new();
    let mut symbol_cache: HashMap<String, Option<SymbolRecord>> = HashMap::new();
    let deadline = Instant::now() + Duration::from_millis(opts.timeout_ms.max(50));
    let mut timed_out = false;

    let mut per_root_active: Vec<Vec<State>> = roots
        .iter()
        .map(|(root, root_score)| {
            vec![State {
                root: root.clone(),
                current_symbol_id: Some(root.symbol_id.clone()),
                score: *root_score,
                steps: Vec::new(),
                visited: {
                    let mut set = HashSet::new();
                    set.insert(root.symbol_id.clone());
                    set
                },
            }]
        })
        .collect();

    // Expand roots in lockstep by depth. This avoids starving later roots when early roots are
    // expensive to expand (e.g., LSP-backed call hierarchy), improving coverage in the fast stage.
    for _depth in 0..opts.max_hops.max(1) {
        if Instant::now() >= deadline {
            timed_out = true;
            break;
        }

        let mut any_active = false;
        for active in per_root_active.iter_mut() {
            if Instant::now() >= deadline {
                timed_out = true;
                break;
            }
            if active.is_empty() {
                continue;
            }

            let mut next = Vec::new();
            for state in std::mem::take(active) {
                if Instant::now() >= deadline {
                    timed_out = true;
                    break;
                }

                let Some(cur_id) = &state.current_symbol_id else {
                    if !state.steps.is_empty() {
                        all_paths.push(state.to_trace_path());
                    }
                    continue;
                };
                let from_symbol = if cur_id == &state.root.symbol_id {
                    state.root.short_name.clone()
                } else {
                    if !symbol_cache.contains_key(cur_id) {
                        symbol_cache.insert(
                            cur_id.clone(),
                            graph::symbol_by_id(conn, cur_id).ok().flatten(),
                        );
                    }
                    symbol_cache
                        .get(cur_id)
                        .and_then(|o| o.as_ref())
                        .map(|s| s.short_name.clone())
                        .unwrap_or_else(|| state.root.short_name.clone())
                };

                let outgoing = edge_provider.outgoing_edges(conn, cur_id)?;
                if outgoing.is_empty() {
                    if !state.steps.is_empty() {
                        all_paths.push(state.to_trace_path());
                    }
                    continue;
                }

                let mut produced_any = false;
                for edge in outgoing.iter().take(opts.beam.max(1)) {
                    if Instant::now() >= deadline {
                        timed_out = true;
                        break;
                    }

                    let mut next_state = state.clone();
                    let resolved_dst = if let Some(dst_id) = edge.dst_symbol_id.as_deref() {
                        Some(dst_id.to_string())
                    } else if let Some(name) = edge.dst_name.as_deref() {
                        graph::resolve_symbol_id(conn, state.root.path.as_str(), name)?
                    } else {
                        None
                    };

                    let to_symbol = if let Some(dst_id) = resolved_dst.as_deref() {
                        if !symbol_cache.contains_key(dst_id) {
                            symbol_cache.insert(
                                dst_id.to_string(),
                                graph::symbol_by_id(conn, dst_id).ok().flatten(),
                            );
                        }
                        symbol_cache
                            .get(dst_id)
                            .and_then(|o| o.clone())
                            .map(|s| s.short_name)
                    } else {
                        None
                    };

                    let target_name = to_symbol.clone().or_else(|| edge.dst_name.clone());
                    let semantic = semantic_bonus(
                        query_terms.as_slice(),
                        target_name.as_deref(),
                        edge.evidence.as_str(),
                    );
                    let hop_penalty = 0.06 * (next_state.steps.len() as f64 + 1.0);
                    let weight = edge_weight(&opts.edge_weights, edge.edge_kind.as_str());

                    next_state.score += weight * edge.confidence + semantic - hop_penalty;
                    next_state.steps.push(TraceStep {
                        edge_kind: edge.edge_kind.clone(),
                        from_symbol: from_symbol.clone(),
                        to_symbol: to_symbol.clone(),
                        target_name,
                        path: edge.path.clone(),
                        line: edge.line,
                        confidence: edge.confidence,
                        evidence: edge.evidence.clone(),
                    });

                    produced_any = true;
                    all_paths.push(next_state.to_trace_path());

                    if let Some(dst_id) = resolved_dst {
                        if !next_state.visited.contains(&dst_id) {
                            next_state.visited.insert(dst_id.clone());
                            next_state.current_symbol_id = Some(dst_id);
                            next.push(next_state);
                        }
                    }
                }

                if !produced_any && !state.steps.is_empty() {
                    all_paths.push(state.to_trace_path());
                }
            }

            next.sort_by(|a, b| b.score.total_cmp(&a.score));
            next.truncate(opts.beam.max(1));
            if !next.is_empty() {
                any_active = true;
            }
            *active = next;
        }

        if timed_out || !any_active {
            break;
        }
    }

    all_paths.sort_by(|a, b| b.score.total_cmp(&a.score));
    let traces = non_overlapping(all_paths, opts.limit_traces.max(1));
    let status = if timed_out {
        SearchStatus::Timeout
    } else {
        SearchStatus::Ready
    };
    Ok((traces, status))
}

#[derive(Debug, Clone)]
struct State {
    root: SymbolRecord,
    current_symbol_id: Option<String>,
    score: f64,
    steps: Vec<TraceStep>,
    visited: HashSet<String>,
}

impl State {
    fn to_trace_path(&self) -> TracePath {
        TracePath {
            score: self.score,
            root_symbol_id: self.root.symbol_id.clone(),
            root_symbol: self.root.short_name.clone(),
            root_path: self.root.path.clone(),
            root_line: self.root.start_line,
            steps: self.steps.clone(),
        }
    }
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn semantic_bonus(query_terms: &[String], target_name: Option<&str>, evidence: &str) -> f64 {
    let mut bonus: f64 = 0.0;
    if let Some(name) = target_name {
        let low = name.to_ascii_lowercase();
        for t in query_terms {
            if low.contains(t) {
                bonus += 0.08;
            }
        }
    }
    let ev = evidence.to_ascii_lowercase();
    for t in query_terms {
        if ev.contains(t) {
            bonus += 0.04;
        }
    }
    bonus.min(0.25)
}

fn edge_weight(w: &TraceEdgeWeights, kind: &str) -> f64 {
    match kind {
        "call" => w.call,
        "import" => w.import,
        "route" => w.route,
        "sql_read" => w.sql_read,
        "sql_write" => w.sql_write,
        _ => 0.4,
    }
}

fn non_overlapping(paths: Vec<TracePath>, limit: usize) -> Vec<TracePath> {
    if limit == 0 {
        return Vec::new();
    }

    // Prefer diversity across roots first, then fill remaining slots with additional distinct paths.
    let mut out = Vec::new();
    let mut seen_roots = HashSet::new();
    let mut seen_endpoints = HashSet::new();
    let mut remaining: Vec<(String, TracePath)> = Vec::new();

    for p in paths {
        let endpoint = path_endpoint_key(&p);
        if out.len() < limit && seen_roots.insert(p.root_symbol_id.clone()) && seen_endpoints.insert(endpoint.clone()) {
            out.push(p);
        } else {
            remaining.push((endpoint, p));
        }
    }

    for (endpoint, p) in remaining {
        if out.len() >= limit {
            break;
        }
        if seen_endpoints.insert(endpoint) {
            out.push(p);
        }
    }

    out
}

fn path_endpoint_key(p: &TracePath) -> String {
    let last = p
        .steps
        .last()
        .map(|s| {
            let target = s
                .to_symbol
                .as_deref()
                .or(s.target_name.as_deref())
                .unwrap_or("-");
            format!("{}:{}:{}:{}", s.edge_kind, s.path, s.line, target)
        })
        .unwrap_or_else(|| "root".to_string());
    format!("{}|{}", p.root_symbol_id, last)
}
