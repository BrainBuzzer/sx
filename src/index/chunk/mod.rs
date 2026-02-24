mod treesitter;

use anyhow::Result;

use crate::config::IndexConfig;
use crate::index::scan::Language;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub chunk_id: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub kind: String,
    pub symbol: Option<String>,
    pub content_hash: String,
    pub content: String,
}

pub struct Chunker {
    ts: treesitter::TreesitterChunker,
}

impl Chunker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ts: treesitter::TreesitterChunker::new()?,
        })
    }

    pub fn chunk_file(
        &mut self,
        repo_path: &str,
        language: Language,
        text: &str,
        cfg: &IndexConfig,
    ) -> Result<Vec<Chunk>> {
        let mut candidates = match language {
            Language::Markdown => markdown_chunks(repo_path, text),
            Language::Rust
            | Language::Ts
            | Language::Tsx
            | Language::Js
            | Language::Jsx
            | Language::Python
            | Language::Go => self.ts.chunk(repo_path, language, text, cfg)?,
            Language::Unknown => Vec::new(),
        };

        if candidates.is_empty() {
            candidates = fallback_chunks(repo_path, text, 1, cfg, "block", None);
        }

        Ok(candidates)
    }
}

fn make_chunk(
    repo_path: &str,
    start_line: i64,
    end_line: i64,
    kind: &str,
    symbol: Option<String>,
    content: String,
) -> Chunk {
    let mut id_hasher = blake3::Hasher::new();
    id_hasher.update(repo_path.as_bytes());
    id_hasher.update(&start_line.to_le_bytes());
    id_hasher.update(&end_line.to_le_bytes());
    id_hasher.update(kind.as_bytes());
    if let Some(s) = &symbol {
        id_hasher.update(s.as_bytes());
    }
    let chunk_id = id_hasher.finalize().to_hex().to_string();

    let content_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    Chunk {
        chunk_id,
        path: repo_path.to_string(),
        start_line,
        end_line,
        kind: kind.to_string(),
        symbol,
        content_hash,
        content,
    }
}

fn fallback_chunks(
    repo_path: &str,
    text: &str,
    base_start_line: i64,
    cfg: &IndexConfig,
    kind: &str,
    symbol: Option<String>,
) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let chunk_lines = cfg.fallback_chunk_lines.max(1);
    let overlap = cfg
        .fallback_overlap_lines
        .min(chunk_lines.saturating_sub(1));

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + chunk_lines).min(lines.len());
        let content = lines[start..end].join("\n");
        let start_line = base_start_line + start as i64;
        let end_line = base_start_line + end as i64 - 1;
        out.push(make_chunk(
            repo_path,
            start_line,
            end_line.max(start_line),
            kind,
            symbol.clone(),
            content,
        ));

        if end == lines.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }

    out
}

fn markdown_chunks(repo_path: &str, text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut headings: Vec<(usize, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(h) = parse_md_heading(line) {
            headings.push((i, h));
        }
    }

    if headings.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();

    if headings[0].0 > 0 {
        let end = headings[0].0;
        let content = lines[0..end].join("\n");
        out.push(make_chunk(
            repo_path,
            1,
            end as i64,
            "md_section",
            Some("Preamble".to_string()),
            content,
        ));
    }

    for idx in 0..headings.len() {
        let (start_idx, title) = &headings[idx];
        let end_idx = if idx + 1 < headings.len() {
            headings[idx + 1].0
        } else {
            lines.len()
        };

        if *start_idx >= end_idx {
            continue;
        }
        let content = lines[*start_idx..end_idx].join("\n");
        out.push(make_chunk(
            repo_path,
            (*start_idx as i64) + 1,
            end_idx as i64,
            "md_section",
            Some(title.clone()),
            content,
        ));
    }

    out
}

fn parse_md_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let after = &trimmed[hashes..];
    if !after.starts_with(' ') {
        return None;
    }
    let title = after.trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_chunks_overlap_progresses() {
        let cfg = IndexConfig {
            max_file_bytes: 1_000_000,
            max_chunk_bytes: 16_384,
            fallback_chunk_lines: 3,
            fallback_overlap_lines: 1,
        };
        let text = "a\nb\nc\nd\ne\nf\n";
        let chunks = fallback_chunks("x.txt", text, 1, &cfg, "block", None);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_line, 1);
        assert!(chunks[1].start_line < chunks[1].end_line + 1);
    }

    #[test]
    fn parse_md_heading_accepts_hashes_and_space() {
        assert_eq!(parse_md_heading("# Hello"), Some("Hello".to_string()));
        assert_eq!(parse_md_heading("###   World  "), Some("World".to_string()));
    }

    #[test]
    fn parse_md_heading_rejects_missing_space() {
        assert_eq!(parse_md_heading("#Hello"), None);
        assert_eq!(parse_md_heading("####### Too many"), None);
    }

    #[test]
    fn markdown_chunks_split_by_heading() {
        let text = "# A\nline1\n## B\nline2\n";
        let chunks = markdown_chunks("doc.md", text);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].symbol.as_deref(), Some("A"));
        assert_eq!(chunks[1].symbol.as_deref(), Some("B"));
    }
}
