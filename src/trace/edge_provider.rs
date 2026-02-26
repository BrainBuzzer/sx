use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::{config, lsp, root};

use super::graph;
use super::rank::EdgeProvider;
use super::types::{EdgeRecord, SymbolRecord};

pub struct DbEdgeProvider;

impl EdgeProvider for DbEdgeProvider {
    fn outgoing_edges(&mut self, conn: &Connection, symbol_id: &str) -> Result<Vec<EdgeRecord>> {
        graph::outgoing_edges(conn, symbol_id, None)
    }
}

pub struct HybridEdgeProvider {
    root_dir: PathBuf,
    cache_dir: PathBuf,
    gopls: lsp::go::GoplsRunner,
    allow_fresh_gopls: bool,
    file_lines: HashMap<String, Vec<String>>,
}

impl HybridEdgeProvider {
    pub fn new(root_dir: &Path, cfg: &config::Config) -> Self {
        let sx_dir = root_dir.join(".sx");
        let cache_dir = sx_dir
            .join(cfg.lsp.go.cache_dir.trim())
            .join("call_hierarchy");
        Self {
            root_dir: root_dir.to_path_buf(),
            cache_dir,
            gopls: lsp::go::GoplsRunner::from_config(root_dir, cfg),
            allow_fresh_gopls: true,
            file_lines: HashMap::new(),
        }
    }

    pub fn gopls_available(&self) -> bool {
        self.gopls.is_available()
    }

    pub fn set_gopls_timeout_ms(&mut self, timeout_ms: u64) {
        self.gopls.set_timeout_ms(timeout_ms);
    }

    pub fn set_allow_fresh_gopls(&mut self, allow: bool) {
        self.allow_fresh_gopls = allow;
    }

    fn call_edges_for_symbol(
        &mut self,
        conn: &Connection,
        sym: &SymbolRecord,
    ) -> Result<Vec<EdgeRecord>> {
        let Some(file_hash) = file_content_hash(conn, sym.path.as_str())? else {
            return Ok(Vec::new());
        };
        let generation = super::index_generation(conn)?;

        let cache_path = self
            .cache_dir
            .join(format!("{}-{}.json", sym.symbol_id, file_hash));

        let callees = match load_call_hierarchy_cache(&cache_path) {
            Ok(Some(c))
                if c.symbol_id == sym.symbol_id
                    && c.content_hash == file_hash
                    && c.index_generation == generation =>
            {
                c.callees
            }
            _ => {
                if !self.allow_fresh_gopls {
                    return Err(anyhow::anyhow!("gopls cache miss (fresh disabled)"));
                }
                let generated = self.compute_call_hierarchy_callees(sym)?;
                let cache = CallHierarchyCache {
                    symbol_id: sym.symbol_id.clone(),
                    path: sym.path.clone(),
                    content_hash: file_hash.clone(),
                    index_generation: generation,
                    generated_at: crate::db::now_unix(),
                    callees: generated.clone(),
                };
                let _ = store_call_hierarchy_cache(&cache_path, &cache);
                generated
            }
        };

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for c in callees {
            if is_external_path(&c.callee_path) {
                continue;
            }
            let dst_symbol_id = graph::symbols_for_path_line(conn, &c.callee_path, c.callee_def_line)
                .ok()
                .and_then(|v| v.first().map(|s| s.symbol_id.clone()));

            let evidence = self
                .read_repo_line(&c.path, c.callsite_line)
                .unwrap_or_default();

            let (dst_symbol_id, dst_name) = if let Some(id) = dst_symbol_id.as_deref() {
                (Some(id), None)
            } else {
                // Keep the edge even if we can't resolve it to an indexed symbol. This helps trace
                // call flows to repo-local code that isn't symbolized for some reason.
                (None, Some(c.callee_name.as_str()))
            };

            let edge = EdgeRecord {
                edge_id: edge_id_for(
                    sym.symbol_id.as_str(),
                    "call",
                    c.path.as_str(),
                    c.callsite_line,
                    dst_symbol_id,
                    dst_name,
                    evidence.as_str(),
                ),
                src_symbol_id: sym.symbol_id.clone(),
                dst_symbol_id: dst_symbol_id.map(|s| s.to_string()),
                dst_name: dst_name.map(|s| s.to_string()),
                edge_kind: "call".to_string(),
                path: c.path.clone(),
                line: c.callsite_line,
                confidence: 0.96,
                evidence,
            };

            if seen.insert(edge.edge_id.clone()) {
                out.push(edge);
            }
        }

        Ok(out)
    }

    fn compute_call_hierarchy_callees(&mut self, sym: &SymbolRecord) -> Result<Vec<CallHierarchyCallee>> {
        if sym.start_line <= 0 {
            return Ok(Vec::new());
        }

        let abs_file = root::from_repo_path(&self.root_dir, sym.path.as_str());
        let col = self
            .symbol_ident_column(sym.path.as_str(), sym.start_line, sym.short_name.as_str())
            .unwrap_or(1);

        let res = self
            .gopls
            .call_hierarchy(&abs_file, sym.start_line as u32, col)?;

        let mut out = Vec::new();
        for link in res.callees {
            let callsite_path = link.callsite.path;
            let callee_path = link.target.location.path;
            if is_external_path(&callsite_path) || is_external_path(&callee_path) {
                continue;
            }
            out.push(CallHierarchyCallee {
                path: callsite_path,
                callsite_line: link.callsite.range.start.line as i64,
                callee_path,
                callee_def_line: link.target.location.range.start.line as i64,
                callee_name: link.target.name,
            });
        }
        Ok(out)
    }

    fn symbol_ident_column(&mut self, repo_path: &str, line: i64, ident: &str) -> Option<u32> {
        let lines = self.read_repo_lines(repo_path).ok()?;
        let idx = (line - 1) as usize;
        let text = lines.get(idx)?.as_str();
        find_ident_column(text, ident)
    }

    fn read_repo_lines(&mut self, repo_path: &str) -> Result<&Vec<String>> {
        if !self.file_lines.contains_key(repo_path) {
            let abs = root::from_repo_path(&self.root_dir, repo_path);
            let body =
                fs::read_to_string(&abs).with_context(|| format!("read {}", abs.display()))?;
            let lines = body.lines().map(|s| s.to_string()).collect::<Vec<_>>();
            self.file_lines.insert(repo_path.to_string(), lines);
        }
        Ok(self
            .file_lines
            .get(repo_path)
            .expect("inserted above"))
    }

    fn read_repo_line(&mut self, repo_path: &str, line: i64) -> Option<String> {
        let lines = self.read_repo_lines(repo_path).ok()?;
        if line <= 0 {
            return None;
        }
        let idx = (line - 1) as usize;
        let txt = lines.get(idx)?.trim_end().to_string();
        Some(txt)
    }
}

impl EdgeProvider for HybridEdgeProvider {
    fn outgoing_edges(&mut self, conn: &Connection, symbol_id: &str) -> Result<Vec<EdgeRecord>> {
        let mut edges = graph::outgoing_edges(conn, symbol_id, None)?;

        let Some(sym) = graph::symbol_by_id(conn, symbol_id)? else {
            return Ok(edges);
        };

        if sym.language.trim().eq_ignore_ascii_case("go") && self.gopls_available() {
            let call_edges: Vec<&EdgeRecord> =
                edges.iter().filter(|e| e.edge_kind == "call").collect();

            // Avoid running gopls call hierarchy for symbols that have no call edges at all, or
            // that appear to only call symbols outside the repo (heuristic resolution failed and no
            // similarly-named symbol exists in the repo). This keeps fast-stage tracing responsive
            // while still using gopls for intra-repo call edges.
            let should_attempt_gopls = if call_edges.is_empty() {
                false
            } else if call_edges.iter().any(|e| e.dst_symbol_id.is_some()) {
                true
            } else {
                call_edges
                    .iter()
                    .filter_map(|e| e.dst_name.as_deref())
                    .filter_map(go_call_candidate_name)
                    .any(|candidate| symbol_short_name_exists(conn, candidate).unwrap_or(false))
            };

            if should_attempt_gopls {
                match self.call_edges_for_symbol(conn, &sym) {
                    Ok(lsp_calls) => {
                        // Replace heuristic call edges with LSP-derived ones (even if empty).
                        edges.retain(|e| e.edge_kind != "call");
                        edges.extend(lsp_calls);
                    }
                    Err(_) => {
                        // gopls failed; keep heuristic call edges as a fallback.
                    }
                }
            } else if !call_edges.is_empty() {
                // We intentionally skip gopls here; drop heuristic call edges to avoid leaving
                // noisy/external call edges in the trace.
                edges.retain(|e| e.edge_kind != "call");
            }
        }

        edges.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| a.line.cmp(&b.line))
                .then_with(|| a.edge_kind.cmp(&b.edge_kind))
                .then_with(|| a.dst_symbol_id.cmp(&b.dst_symbol_id))
                .then_with(|| a.dst_name.cmp(&b.dst_name))
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| a.evidence.cmp(&b.evidence))
        });

        Ok(edges)
    }
}

fn file_content_hash(conn: &Connection, path: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT content_hash FROM files WHERE path=?1 LIMIT 1",
        [path],
        |row| row.get(0),
    )
    .optional()
    .context("read files.content_hash")
}

fn is_external_path(path: &str) -> bool {
    let p = Path::new(path);
    p.is_absolute() || path.starts_with("file://")
}

fn go_call_candidate_name(raw: &str) -> Option<&str> {
    // Heuristic call edges for Go sometimes include package-qualified names (e.g., "pkg.Foo").
    // We treat the last segment as the likely symbol name to probe for existence in the repo.
    let name = raw
        .rsplit_once('.')
        .map(|(_, last)| last)
        .unwrap_or(raw)
        .trim();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn symbol_short_name_exists(conn: &Connection, short_name: &str) -> Result<bool> {
    let exists: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM symbols WHERE short_name=?1 LIMIT 1",
            [short_name],
            |row| row.get(0),
        )
        .optional()
        .context("query symbols.short_name exists")?;
    Ok(exists.is_some())
}

fn find_ident_column(line: &str, ident: &str) -> Option<u32> {
    if ident.trim().is_empty() {
        return None;
    }

    for (idx, _) in line.match_indices(ident) {
        if idx > 0 {
            let prev = line.as_bytes()[idx - 1];
            if is_ident_byte(prev) {
                continue;
            }
        }
        let after = idx + ident.len();
        if after < line.len() {
            let next = line.as_bytes()[after];
            if is_ident_byte(next) {
                continue;
            }
        }
        return Some((idx as u32) + 1);
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallHierarchyCache {
    symbol_id: String,
    path: String,
    content_hash: String,
    #[serde(default)]
    index_generation: i64,
    generated_at: i64,
    callees: Vec<CallHierarchyCallee>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallHierarchyCallee {
    path: String,
    callsite_line: i64,
    callee_path: String,
    callee_def_line: i64,
    callee_name: String,
}

fn load_call_hierarchy_cache(path: &Path) -> Result<Option<CallHierarchyCache>> {
    if !path.exists() {
        return Ok(None);
    }
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed = serde_json::from_str::<CallHierarchyCache>(&body)
        .with_context(|| format!("parse cache {}", path.display()))?;
    Ok(Some(parsed))
}

fn store_call_hierarchy_cache(path: &Path, cache: &CallHierarchyCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(cache).context("serialize call hierarchy cache")?;
    fs::write(&tmp, body).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_hierarchy_cache_backcompat_missing_index_generation() {
        let json = r#"
{
  "symbol_id": "sym",
  "path": "src/main.go",
  "content_hash": "hash",
  "generated_at": 123,
  "callees": []
}
"#;
        let parsed: CallHierarchyCache = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.index_generation, 0);
    }
}
