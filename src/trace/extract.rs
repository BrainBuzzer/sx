use std::collections::HashSet;

use regex::Regex;

use crate::index::chunk::Chunk;
use crate::index::scan::Language;

use super::types::{EdgeCandidate, SymbolRecord};

pub fn extract_for_file(
    repo_path: &str,
    language: Language,
    chunks: &[Chunk],
) -> (Vec<SymbolRecord>, Vec<EdgeCandidate>) {
    let mut symbols = Vec::new();
    let mut edges = Vec::new();

    for chunk in chunks {
        let Some(short_name_raw) = &chunk.symbol else {
            continue;
        };
        let short_name = short_name_raw.trim().to_string();
        if short_name.is_empty() {
            continue;
        }

        let symbol_id = symbol_id_for(chunk.chunk_id.as_str(), short_name.as_str());
        let sym = SymbolRecord {
            symbol_id: symbol_id.clone(),
            chunk_id: chunk.chunk_id.clone(),
            path: repo_path.to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            language: language.as_str().to_string(),
            kind: chunk.kind.clone(),
            fq_name: format!("{repo_path}::{short_name}"),
            short_name: short_name.clone(),
        };
        symbols.push(sym);

        edges.extend(extract_call_edges(chunk, &symbol_id, &short_name));
        edges.extend(extract_import_edges(chunk, &symbol_id, language));
        edges.extend(extract_route_edges(
            chunk,
            &symbol_id,
            &short_name,
            language,
        ));
        edges.extend(extract_sql_edges(chunk, &symbol_id));
    }

    let edges = dedupe_edges(edges);
    (symbols, edges)
}

fn symbol_id_for(chunk_id: &str, short_name: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sym:");
    hasher.update(chunk_id.as_bytes());
    hasher.update(b":");
    hasher.update(short_name.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn edge_id_for(
    src_symbol_id: &str,
    edge_kind: &str,
    path: &str,
    line: i64,
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
    if let Some(name) = dst_name {
        hasher.update(name.as_bytes());
    }
    hasher.update(b"|");
    hasher.update(evidence.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn dedupe_edges(edges: Vec<EdgeCandidate>) -> Vec<EdgeCandidate> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for e in edges {
        let key = edge_id_for(
            e.src_symbol_id.as_str(),
            e.edge_kind.as_str(),
            e.path.as_str(),
            e.line,
            e.dst_name.as_deref(),
            e.evidence.as_str(),
        );
        if seen.insert(key) {
            deduped.push(e);
        }
    }
    deduped
}

fn extract_call_edges(chunk: &Chunk, src_symbol_id: &str, short_name: &str) -> Vec<EdgeCandidate> {
    let re = Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(").expect("compile call regex");
    let mut out = Vec::new();

    for caps in re.captures_iter(&chunk.content) {
        let Some(m) = caps.get(1) else {
            continue;
        };
        let name = m.as_str();
        if is_call_keyword(name) {
            continue;
        }
        if name == short_name {
            // Keep recursion with lower confidence.
            let local_line = line_for_offset(&chunk.content, m.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(name.to_string()),
                edge_kind: "call".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.45,
                evidence: line_snippet(&chunk.content, local_line),
            });
            continue;
        }
        let local_line = line_for_offset(&chunk.content, m.start());
        let line = chunk.start_line + local_line - 1;
        out.push(EdgeCandidate {
            src_symbol_id: src_symbol_id.to_string(),
            dst_name: Some(name.to_string()),
            edge_kind: "call".to_string(),
            path: chunk.path.clone(),
            line,
            confidence: 0.62,
            evidence: line_snippet(&chunk.content, local_line),
        });
    }
    out
}

fn extract_import_edges(
    chunk: &Chunk,
    src_symbol_id: &str,
    language: Language,
) -> Vec<EdgeCandidate> {
    let mut out = Vec::new();
    let patterns: Vec<Regex> = match language {
        Language::Rust => {
            vec![Regex::new(r"(?m)^\s*use\s+([A-Za-z0-9_:]+)").expect("rust use regex")]
        }
        Language::Go => {
            vec![Regex::new(r#"(?m)^\s*import\s+(?:\(|)(?:"([^"]+)")"#).expect("go import regex")]
        }
        Language::Ts | Language::Tsx | Language::Js | Language::Jsx => vec![
            Regex::new(r#"(?m)^\s*import\s+.*?\s+from\s+["']([^"']+)["']"#)
                .expect("js import regex"),
            Regex::new(r#"require\(\s*["']([^"']+)["']\s*\)"#).expect("js require regex"),
        ],
        Language::Python => vec![
            Regex::new(r"(?m)^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+").expect("py from regex"),
            Regex::new(r"(?m)^\s*import\s+([A-Za-z0-9_\.]+)").expect("py import regex"),
        ],
        _ => Vec::new(),
    };

    for re in patterns {
        for caps in re.captures_iter(&chunk.content) {
            let Some(m) = caps.get(1) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, m.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(m.as_str().to_string()),
                edge_kind: "import".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.75,
                evidence: line_snippet(&chunk.content, local_line),
            });
        }
    }

    out
}

fn extract_route_edges(
    chunk: &Chunk,
    src_symbol_id: &str,
    short_name: &str,
    language: Language,
) -> Vec<EdgeCandidate> {
    let mut out = Vec::new();
    let generic = Regex::new(
        r#"(?i)\b(?:router|app|r|e)\.(?:get|post|put|delete|patch|options|head)\s*\(\s*["']([^"']+)["']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)"#,
    )
    .expect("generic route regex");
    for caps in generic.captures_iter(&chunk.content) {
        let (Some(route), Some(handler)) = (caps.get(1), caps.get(2)) else {
            continue;
        };
        let local_line = line_for_offset(&chunk.content, route.start());
        let line = chunk.start_line + local_line - 1;
        out.push(EdgeCandidate {
            src_symbol_id: src_symbol_id.to_string(),
            dst_name: Some(handler.as_str().to_string()),
            edge_kind: "route".to_string(),
            path: chunk.path.clone(),
            line,
            confidence: 0.86,
            evidence: format!("route {} -> {}", route.as_str(), handler.as_str()),
        });
    }

    if language == Language::Go {
        let go_handle = Regex::new(
            r#"(?i)\b(?:handlefunc|handle)\s*\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)"#,
        )
        .expect("go route regex");
        for caps in go_handle.captures_iter(&chunk.content) {
            let (Some(route), Some(handler)) = (caps.get(1), caps.get(2)) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, route.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(handler.as_str().to_string()),
                edge_kind: "route".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.88,
                evidence: format!("route {} -> {}", route.as_str(), handler.as_str()),
            });
        }
    }

    if language == Language::Rust {
        let rust_route =
            Regex::new(r#"(?i)\broute\s*\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_:]*)"#)
                .expect("rust route regex");
        for caps in rust_route.captures_iter(&chunk.content) {
            let (Some(route), Some(handler)) = (caps.get(1), caps.get(2)) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, route.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(
                    handler
                        .as_str()
                        .rsplit("::")
                        .next()
                        .unwrap_or("")
                        .to_string(),
                ),
                edge_kind: "route".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.84,
                evidence: format!("route {} -> {}", route.as_str(), handler.as_str()),
            });
        }
    }

    if language == Language::Python {
        let py_decorator = Regex::new(
            r#"(?m)^\s*@(?:app|router)\.(?:get|post|put|delete|patch)\s*\(\s*["']([^"']+)["']"#,
        )
        .expect("python route regex");
        for caps in py_decorator.captures_iter(&chunk.content) {
            let Some(route) = caps.get(1) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, route.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(short_name.to_string()),
                edge_kind: "route".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.82,
                evidence: format!("route {} -> {}", route.as_str(), short_name),
            });
        }
    }

    out
}

fn extract_sql_edges(chunk: &Chunk, src_symbol_id: &str) -> Vec<EdgeCandidate> {
    let mut out = Vec::new();
    let reads = [
        Regex::new(r"(?i)\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)").expect("sql from regex"),
        Regex::new(r"(?i)\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)").expect("sql join regex"),
        Regex::new(r"(?i)\bselect\s+.+?\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)")
            .expect("sql select regex"),
    ];
    let writes = [
        Regex::new(r"(?i)\binsert\s+into\s+([A-Za-z_][A-Za-z0-9_]*)").expect("sql insert regex"),
        Regex::new(r"(?i)\bupdate\s+([A-Za-z_][A-Za-z0-9_]*)").expect("sql update regex"),
        Regex::new(r"(?i)\bdelete\s+from\s+([A-Za-z_][A-Za-z0-9_]*)").expect("sql delete regex"),
    ];

    for re in reads {
        for caps in re.captures_iter(&chunk.content) {
            let Some(tbl) = caps.get(1) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, tbl.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(tbl.as_str().to_string()),
                edge_kind: "sql_read".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.77,
                evidence: line_snippet(&chunk.content, local_line),
            });
        }
    }
    for re in writes {
        for caps in re.captures_iter(&chunk.content) {
            let Some(tbl) = caps.get(1) else {
                continue;
            };
            let local_line = line_for_offset(&chunk.content, tbl.start());
            let line = chunk.start_line + local_line - 1;
            out.push(EdgeCandidate {
                src_symbol_id: src_symbol_id.to_string(),
                dst_name: Some(tbl.as_str().to_string()),
                edge_kind: "sql_write".to_string(),
                path: chunk.path.clone(),
                line,
                confidence: 0.82,
                evidence: line_snippet(&chunk.content, local_line),
            });
        }
    }

    out
}

fn line_for_offset(text: &str, offset: usize) -> i64 {
    let mut line = 1i64;
    for (idx, b) in text.as_bytes().iter().enumerate() {
        if idx >= offset {
            break;
        }
        if *b == b'\n' {
            line += 1;
        }
    }
    line
}

fn line_snippet(text: &str, line: i64) -> String {
    if line <= 0 {
        return String::new();
    }
    let idx = (line as usize).saturating_sub(1);
    text.lines()
        .nth(idx)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn is_call_keyword(name: &str) -> bool {
    matches!(
        name,
        "if" | "for"
            | "while"
            | "switch"
            | "match"
            | "return"
            | "fn"
            | "func"
            | "def"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "let"
            | "new"
            | "use"
            | "import"
            | "from"
            | "catch"
            | "throw"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::chunk::Chunk;

    #[test]
    fn extracts_calls_routes_and_sql() {
        let chunk = Chunk {
            chunk_id: "abc".to_string(),
            path: "src/a.rs".to_string(),
            start_line: 10,
            end_line: 20,
            kind: "function".to_string(),
            symbol: Some("serve".to_string()),
            content_hash: "h".to_string(),
            content: r#"
router.get("/v1/missions", show_missions);
let q = "select * from driver_missions";
store_driver();
"#
            .to_string(),
        };
        let (symbols, edges) = extract_for_file("src/a.rs", Language::Rust, &[chunk]);
        assert_eq!(symbols.len(), 1);
        assert!(edges.iter().any(|e| e.edge_kind == "call"));
        assert!(edges.iter().any(|e| e.edge_kind == "route"));
        assert!(edges.iter().any(|e| e.edge_kind == "sql_read"));
    }
}
