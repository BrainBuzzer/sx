use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use ignore::{DirEntry, WalkBuilder};

use crate::{config, root};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Ts,
    Tsx,
    Js,
    Jsx,
    Python,
    Go,
    Markdown,
    Unknown,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Ts => "ts",
            Language::Tsx => "tsx",
            Language::Js => "js",
            Language::Jsx => "jsx",
            Language::Python => "python",
            Language::Go => "go",
            Language::Markdown => "markdown",
            Language::Unknown => "unknown",
        }
    }

    pub fn from_path(path: &Path) -> Language {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        match ext.as_str() {
            "rs" => Language::Rust,
            "ts" => Language::Ts,
            "tsx" => Language::Tsx,
            "js" => Language::Js,
            "jsx" => Language::Jsx,
            "py" => Language::Python,
            "go" => Language::Go,
            "md" | "markdown" => Language::Markdown,
            _ => Language::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScannedFile {
    pub abs_path: PathBuf,
    pub repo_path: String,
    pub size: u64,
    pub mtime: i64,
    pub language: Language,
}

pub fn scan_repo(root_dir: &Path, cfg: &config::Config) -> Result<Vec<ScannedFile>> {
    let excludes = build_excludes(cfg);
    let root_dir = root_dir.to_path_buf();
    let root_for_filter = root_dir.clone();

    let mut out = Vec::new();

    let mut builder = WalkBuilder::new(&root_dir);
    builder.hidden(false);
    builder.git_ignore(true);
    builder.git_exclude(true);
    builder.git_global(false);
    builder.filter_entry(move |entry| filter_entry(&root_for_filter, entry, &excludes));

    for result in builder.build() {
        let entry = match result {
            Ok(e) => e,
            Err(err) => {
                tracing::debug!(error = %err, "walk error");
                continue;
            }
        };

        let ft = match entry.file_type() {
            Some(ft) => ft,
            None => continue,
        };

        if ft.is_dir() {
            continue;
        }
        if ft.is_symlink() {
            continue;
        }

        let abs_path = entry.path().to_path_buf();
        if is_likely_binary_ext(&abs_path) {
            continue;
        }

        let metadata = match abs_path.metadata() {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(path = %abs_path.display(), error = %err, "metadata failed");
                continue;
            }
        };

        let size = metadata.len();
        if size > cfg.index.max_file_bytes {
            continue;
        }

        let mtime = metadata_modified_unix(&metadata).unwrap_or(0);

        let repo_path = root::to_repo_path(&root_dir, &abs_path)
            .with_context(|| format!("repo path {}", abs_path.display()))?;

        let language = Language::from_path(&abs_path);

        out.push(ScannedFile {
            abs_path,
            repo_path,
            size,
            mtime,
            language,
        });
    }

    Ok(out)
}

fn build_excludes(cfg: &config::Config) -> Vec<String> {
    let mut out = vec![
        ".git/".to_string(),
        ".sx/".to_string(),
        "target/".to_string(),
        "node_modules/".to_string(),
        "dist/".to_string(),
        "build/".to_string(),
        ".next/".to_string(),
        ".turbo/".to_string(),
        "vendor/".to_string(),
        ".venv/".to_string(),
    ];
    out.extend(cfg.scan.exclude.iter().cloned());

    for s in &mut out {
        if !s.ends_with('/') {
            s.push('/');
        }
        while s.starts_with("./") {
            *s = s.trim_start_matches("./").to_string();
        }
    }

    out.sort();
    out.dedup();
    out
}

fn filter_entry(root_dir: &Path, entry: &DirEntry, excludes: &[String]) -> bool {
    let path = entry.path();
    if path == root_dir {
        return true;
    }

    let rel = match root::to_repo_path(root_dir, path) {
        Ok(r) => r,
        Err(_) => return true,
    };

    let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
    let mut rel_norm = rel;
    if is_dir && !rel_norm.ends_with('/') {
        rel_norm.push('/');
    }

    for ex in excludes {
        if rel_norm.starts_with(ex) {
            return false;
        }
    }
    true
}

fn is_likely_binary_ext(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    matches!(
        ext.as_str(),
        "png"
            | "jpg"
            | "jpeg"
            | "gif"
            | "bmp"
            | "ico"
            | "icns"
            | "pdf"
            | "zip"
            | "gz"
            | "tgz"
            | "tar"
            | "7z"
            | "rar"
            | "jar"
            | "class"
            | "o"
            | "a"
            | "so"
            | "dylib"
            | "exe"
            | "dll"
            | "wasm"
            | "mp3"
            | "mp4"
            | "mov"
            | "avi"
            | "mkv"
    )
}

fn metadata_modified_unix(metadata: &std::fs::Metadata) -> Option<i64> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let modified = metadata.modified().ok()?;
    let dur = modified.duration_since(UNIX_EPOCH).unwrap_or_else(|_| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    });
    Some(dur.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> crate::config::Config {
        crate::config::Config::default()
    }

    #[test]
    fn scan_respects_builtin_excludes() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".git")).expect("create .git");
        std::fs::create_dir_all(dir.path().join(".sx")).expect("create .sx");
        std::fs::create_dir_all(dir.path().join("src")).expect("create src");
        std::fs::write(dir.path().join("src/lib.rs"), "fn x() {}\n").expect("write lib.rs");
        std::fs::write(dir.path().join(".sx/ignored.txt"), "nope").expect("write ignored");

        let files = scan_repo(dir.path(), &test_config()).expect("scan");
        let paths: Vec<String> = files.into_iter().map(|f| f.repo_path).collect();
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(!paths.iter().any(|p| p.starts_with(".sx/")));
        assert!(!paths.iter().any(|p| p.starts_with(".git/")));
    }

    #[test]
    fn scan_respects_gitignore() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".git")).expect("create .git");
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").expect("write .gitignore");
        std::fs::write(dir.path().join("ignored.rs"), "fn ignored() {}\n").expect("write ignored");
        std::fs::write(dir.path().join("kept.rs"), "fn kept() {}\n").expect("write kept");

        let files = scan_repo(dir.path(), &test_config()).expect("scan");
        let paths: Vec<String> = files.into_iter().map(|f| f.repo_path).collect();
        assert!(paths.contains(&"kept.rs".to_string()));
        assert!(!paths.contains(&"ignored.rs".to_string()));
    }

    #[test]
    fn scan_skips_large_files() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".git")).expect("create .git");
        std::fs::write(dir.path().join("big.txt"), "0123456789ABCDEF").expect("write big");

        let mut cfg = test_config();
        cfg.index.max_file_bytes = 8;
        let files = scan_repo(dir.path(), &cfg).expect("scan");
        assert!(files.is_empty());
    }

    #[test]
    fn language_from_path_maps_extensions() {
        assert_eq!(Language::from_path(Path::new("a.rs")), Language::Rust);
        assert_eq!(Language::from_path(Path::new("a.ts")), Language::Ts);
        assert_eq!(Language::from_path(Path::new("a.tsx")), Language::Tsx);
        assert_eq!(Language::from_path(Path::new("a.js")), Language::Js);
        assert_eq!(Language::from_path(Path::new("a.jsx")), Language::Jsx);
        assert_eq!(Language::from_path(Path::new("a.py")), Language::Python);
        assert_eq!(Language::from_path(Path::new("a.go")), Language::Go);
        assert_eq!(Language::from_path(Path::new("a.md")), Language::Markdown);
    }
}
