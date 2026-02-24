use std::path::Path;

use anyhow::{Context as _, Result, anyhow};

use crate::config;

pub struct VectorIndex {
    inner: usearch::Index,
    dim: usize,
}

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

impl VectorIndex {
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        self.inner.reserve(capacity).context("usearch reserve")?;
        Ok(())
    }

    pub fn add(&mut self, key: u64, vector: &[f32]) -> Result<()> {
        self.inner.add(key, vector).context("usearch add")?;
        Ok(())
    }

    pub fn remove(&mut self, key: u64) -> Result<usize> {
        let removed = self.inner.remove(key).context("usearch remove")?;
        Ok(removed)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let matches = self.inner.search(query, k).context("usearch search")?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

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

    pub fn save(&self, path: &str) -> Result<()> {
        self.inner.save(path).context("usearch save")?;
        Ok(())
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
