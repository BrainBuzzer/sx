use std::path::Path;

use anyhow::{Context as _, Result, anyhow};

use crate::config;

#[cfg(not(windows))]
pub struct VectorIndex {
    inner: usearch::Index,
    dim: usize,
}

#[cfg(windows)]
pub struct VectorIndex {
}

#[cfg(windows)]
fn unsupported() -> anyhow::Error {
    anyhow!("vector index is not supported on Windows builds")
}

#[cfg(not(windows))]
pub fn open_for_write(cfg: &config::Config, dim: usize, path: &Path) -> Result<VectorIndex> {
    let options = index_options(cfg, dim)?;
    let index = usearch::Index::new(&options).context("create usearch index")?;
    let ix = VectorIndex { inner: index, dim };

    if path.exists() {
        ix.inner
            .load(path.to_string_lossy().as_ref())
            .with_context(|| format!("load {}", path.display()))?;
    }

    Ok(ix)
}

#[cfg(windows)]
pub fn open_for_write(_cfg: &config::Config, _dim: usize, _path: &Path) -> Result<VectorIndex> {
    Err(unsupported())
}

#[cfg(not(windows))]
pub fn open_for_read(cfg: &config::Config, dim: usize, path: &Path) -> Result<VectorIndex> {
    if !path.exists() {
        return Err(anyhow!("missing vector index {}", path.display()));
    }
    let options = index_options(cfg, dim)?;
    let index = usearch::Index::new(&options).context("create usearch index")?;
    let ix = VectorIndex { inner: index, dim };
    ix.inner
        .view(path.to_string_lossy().as_ref())
        .with_context(|| format!("view {}", path.display()))?;
    Ok(ix)
}

#[cfg(windows)]
pub fn open_for_read(_cfg: &config::Config, _dim: usize, _path: &Path) -> Result<VectorIndex> {
    Err(unsupported())
}

impl VectorIndex {
    #[cfg(not(windows))]
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        self.inner.reserve(capacity).context("usearch reserve")?;
        Ok(())
    }

    #[cfg(windows)]
    pub fn reserve(&self, _capacity: usize) -> Result<()> {
        Err(unsupported())
    }

    #[cfg(not(windows))]
    pub fn add(&mut self, key: u64, vector: &[f32]) -> Result<()> {
        self.inner.add(key, vector).context("usearch add")?;
        Ok(())
    }

    #[cfg(windows)]
    pub fn add(&mut self, _key: u64, _vector: &[f32]) -> Result<()> {
        Err(unsupported())
    }

    #[cfg(not(windows))]
    pub fn remove(&mut self, key: u64) -> Result<usize> {
        let removed = self.inner.remove(key).context("usearch remove")?;
        Ok(removed)
    }

    #[cfg(windows)]
    pub fn remove(&mut self, _key: u64) -> Result<usize> {
        Err(unsupported())
    }

    #[cfg(not(windows))]
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let matches = self.inner.search(query, k).context("usearch search")?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    #[cfg(windows)]
    pub fn search(&self, _query: &[f32], _k: usize) -> Result<Vec<(u64, f32)>> {
        Err(unsupported())
    }

    #[cfg(not(windows))]
    pub fn get(&self, key: u64) -> Result<Option<Vec<f32>>> {
        if !self.inner.contains(key) {
            return Ok(None);
        }
        let mut buf = vec![0.0f32; self.dim];
        let found = self.inner.get(key, &mut buf).context("usearch get")?;
        if found == 0 {
            return Ok(None);
        }
        Ok(Some(buf))
    }

    #[cfg(windows)]
    pub fn get(&self, _key: u64) -> Result<Option<Vec<f32>>> {
        Err(unsupported())
    }

    #[cfg(not(windows))]
    pub fn save(&self, path: &str) -> Result<()> {
        self.inner.save(path).context("usearch save")?;
        Ok(())
    }

    #[cfg(windows)]
    pub fn save(&self, _path: &str) -> Result<()> {
        Err(unsupported())
    }
}

pub fn normalize_in_place(v: &mut [f32]) {
    let mut sum = 0.0f64;
    for x in v.iter() {
        sum += (*x as f64) * (*x as f64);
    }
    let norm = sum.sqrt();
    if norm <= 0.0 {
        return;
    }
    let inv = (1.0 / norm) as f32;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(not(windows))]
fn index_options(cfg: &config::Config, dim: usize) -> Result<usearch::IndexOptions> {
    let metric = match cfg.vector.metric.trim().to_ascii_lowercase().as_str() {
        "cos" => usearch::MetricKind::Cos,
        other => return Err(anyhow!("unsupported vector metric: {other}")),
    };

    let quantization = match cfg.vector.quantization.trim().to_ascii_lowercase().as_str() {
        "f32" => usearch::ScalarKind::F32,
        other => return Err(anyhow!("unsupported vector quantization: {other}")),
    };

    let mut options = usearch::IndexOptions::default();
    options.dimensions = dim;
    options.metric = metric;
    options.quantization = quantization;
    options.connectivity = cfg.vector.connectivity;
    options.expansion_add = cfg.vector.expansion_add;
    options.expansion_search = cfg.vector.expansion_search;
    options.multi = false;
    Ok(options)
}
