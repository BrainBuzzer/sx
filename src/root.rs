use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, anyhow};

pub fn discover_root(cwd: &Path) -> anyhow::Result<PathBuf> {
    for ancestor in cwd.ancestors() {
        let git = ancestor.join(".git");
        if git.exists() {
            return ancestor
                .canonicalize()
                .with_context(|| format!("canonicalize root {}", ancestor.display()));
        }
    }

    cwd.canonicalize()
        .with_context(|| format!("canonicalize cwd {}", cwd.display()))
}

pub fn normalize_root(cwd: &Path, root: &Path) -> anyhow::Result<PathBuf> {
    let resolved = if root.is_absolute() {
        root.to_path_buf()
    } else {
        cwd.join(root)
    };

    resolved
        .canonicalize()
        .with_context(|| format!("canonicalize root {}", resolved.display()))
}

pub fn normalize_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

pub fn to_repo_path(root: &Path, path: &Path) -> anyhow::Result<String> {
    let rel = path
        .strip_prefix(root)
        .map_err(|_| anyhow!("path {} escapes root {}", path.display(), root.display()))?;

    let mut parts: Vec<String> = Vec::new();
    for component in rel.components() {
        match component {
            Component::Normal(p) => parts.push(p.to_string_lossy().to_string()),
            Component::CurDir => {}
            _ => return Err(anyhow!("unsupported path component in {}", rel.display())),
        }
    }

    Ok(parts.join("/"))
}

pub fn from_repo_path(root: &Path, repo_path: &str) -> PathBuf {
    let mut out = root.to_path_buf();
    for part in repo_path.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        out.push(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_repo_path_normalizes_separators() {
        let root = PathBuf::from("/tmp/sx_root_test");
        let path = root.join("a").join("b").join("c.txt");
        let repo = to_repo_path(&root, &path).expect("to_repo_path");
        assert_eq!(repo, "a/b/c.txt");
    }

    #[test]
    fn to_repo_path_rejects_escape() {
        let root = PathBuf::from("/tmp/sx_root_test");
        let path = PathBuf::from("/tmp/other/place.txt");
        let err = to_repo_path(&root, &path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("escapes root"));
    }

    #[test]
    fn from_repo_path_roundtrip() {
        let root = PathBuf::from("/tmp/sx_root_test");
        let abs = from_repo_path(&root, "a/b/c.txt");
        assert_eq!(abs, root.join("a").join("b").join("c.txt"));
    }
}
