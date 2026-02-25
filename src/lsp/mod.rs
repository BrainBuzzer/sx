pub mod go;

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::root;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    pub path: String,
    pub range: Range,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePosition {
    pub path: String,
    pub line: u32,
    pub column: u32,
}

pub fn parse_file_position(input: &str) -> Option<FilePosition> {
    let (path, line, column) = input.rsplit_once(':').and_then(|(left, col)| {
        let (path, line) = left.rsplit_once(':')?;
        Some((path, line, col))
    })?;

    let line: u32 = line.parse().ok()?;
    let column: u32 = column.parse().ok()?;
    if line == 0 || column == 0 {
        return None;
    }

    Some(FilePosition {
        path: path.to_string(),
        line,
        column,
    })
}

pub fn resolve_path(root_dir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root::from_repo_path(root_dir, path)
    }
}

pub fn normalize_path(root_dir: &Path, abs_path: &Path) -> Result<String> {
    let canon = abs_path.canonicalize().unwrap_or_else(|_| abs_path.to_path_buf());
    match root::to_repo_path(root_dir, &canon) {
        Ok(repo) => Ok(repo),
        Err(_) => Ok(abs_path.display().to_string()),
    }
}
