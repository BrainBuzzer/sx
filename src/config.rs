use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub scan: ScanConfig,
    pub index: IndexConfig,
    pub open: OpenConfig,
    pub providers: ProvidersConfig,
    pub embed: EmbedConfig,
    pub vector: VectorConfig,
    pub query: QueryConfig,
    pub deep: DeepConfig,
    pub llm: LlmConfig,
    pub trace: TraceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScanConfig {
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub max_file_bytes: u64,
    pub max_chunk_bytes: usize,
    pub fallback_chunk_lines: usize,
    pub fallback_overlap_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenConfig {
    pub editor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProvidersConfig {
    pub ollama: OllamaProviderConfig,
    pub openai: OpenAiProviderConfig,
    pub voyage: VoyageProviderConfig,
    #[serde(alias = "codex")]
    pub zhipu: ZhipuProviderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaProviderConfig {
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiProviderConfig {
    pub base_url: String,
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoyageProviderConfig {
    pub base_url: String,
    pub api_key_env: String,
    pub rerank_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ZhipuProviderConfig {
    pub base_url: String,
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    /// "ollama" | "openai" | "voyage" | "none"
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub batch_size: usize,
    pub max_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorConfig {
    /// "cos"
    pub metric: String,
    /// "f32"
    pub quantization: String,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QueryConfig {
    pub rrf_k: usize,
    pub bm25_weight: f64,
    pub vec_weight: f64,
    pub expansion_bm25_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeepConfig {
    pub seed_limit: usize,
    pub max_terms: usize,
    pub rerank_top: usize,
    pub llm_expand: bool,
    pub llm_rerank: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// "none" | "ollama" | "openai" | "zhipu"
    pub provider: String,
    /// OpenAI Responses model
    pub model: String,
    /// Zhipu Chat Completions model
    #[serde(alias = "codex_model")]
    pub zhipu_model: String,
    /// Ollama generation model
    pub ollama_model: String,
    pub max_output_tokens: usize,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TraceConfig {
    pub enabled: bool,
    pub candidate_roots: usize,
    pub beam_fast: usize,
    pub beam_deep: usize,
    pub fast_timeout_ms: u64,
    pub deep_timeout_ms: u64,
    pub edge_weights: TraceEdgeWeights,
    pub llm_summary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TraceEdgeWeights {
    pub call: f64,
    pub import: f64,
    pub route: f64,
    pub sql_read: f64,
    pub sql_write: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GlobalCredentials {
    pub zhipu_api_key: Option<String>,
    pub voyage_api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 3,
            scan: ScanConfig {
                exclude: vec!["target/".into(), "node_modules/".into(), ".sx/".into()],
            },
            index: IndexConfig {
                max_file_bytes: 2 * 1024 * 1024,
                max_chunk_bytes: 16 * 1024,
                fallback_chunk_lines: 200,
                fallback_overlap_lines: 20,
            },
            open: OpenConfig {
                editor: String::new(),
            },
            providers: ProvidersConfig::default(),
            embed: EmbedConfig::default(),
            vector: VectorConfig::default(),
            query: QueryConfig::default(),
            deep: DeepConfig::default(),
            llm: LlmConfig::default(),
            trace: TraceConfig::default(),
        }
    }
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            exclude: vec!["target/".into(), "node_modules/".into(), ".sx/".into()],
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_file_bytes: 2 * 1024 * 1024,
            max_chunk_bytes: 16 * 1024,
            fallback_chunk_lines: 200,
            fallback_overlap_lines: 20,
        }
    }
}

impl Default for OpenConfig {
    fn default() -> Self {
        Self {
            editor: String::new(),
        }
    }
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            ollama: OllamaProviderConfig::default(),
            openai: OpenAiProviderConfig::default(),
            voyage: VoyageProviderConfig::default(),
            zhipu: ZhipuProviderConfig::default(),
        }
    }
}

impl Default for OllamaProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
        }
    }
}

impl Default for OpenAiProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key_env: "OPENAI_API_KEY".to_string(),
        }
    }
}

impl Default for VoyageProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.voyageai.com/v1".to_string(),
            api_key_env: "VOYAGE_API_KEY".to_string(),
            rerank_model: "rerank-2.5".to_string(),
        }
    }
}

impl Default for ZhipuProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            api_key_env: "ZAI_API_KEY".to_string(),
        }
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 512,
            batch_size: 32,
            max_chars: 8000,
        }
    }
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            metric: "cos".to_string(),
            quantization: "f32".to_string(),
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60,
            bm25_weight: 1.0,
            vec_weight: 1.0,
            expansion_bm25_weight: 0.6,
        }
    }
}

impl Default for DeepConfig {
    fn default() -> Self {
        Self {
            seed_limit: 10,
            max_terms: 24,
            rerank_top: 50,
            llm_expand: true,
            llm_rerank: false,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "zhipu".to_string(),
            model: "gpt-4o-mini".to_string(),
            zhipu_model: "glm-5".to_string(),
            ollama_model: "llama3.1".to_string(),
            max_output_tokens: 256,
            temperature: 0.2,
        }
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            candidate_roots: 20,
            beam_fast: 32,
            beam_deep: 64,
            fast_timeout_ms: 1_800,
            deep_timeout_ms: 8_000,
            edge_weights: TraceEdgeWeights::default(),
            llm_summary: true,
        }
    }
}

impl Default for TraceEdgeWeights {
    fn default() -> Self {
        Self {
            call: 1.0,
            import: 0.55,
            route: 0.9,
            sql_read: 0.8,
            sql_write: 0.9,
        }
    }
}

pub fn load_or_default(config_path: &Path) -> Result<Config> {
    if !config_path.exists() {
        return Ok(Config::default());
    }
    load(config_path)
}

pub fn ensure_config(config_path: &Path) -> Result<Config> {
    let parent = config_path
        .parent()
        .context("config path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    fs::create_dir_all(parent.join("cache"))
        .with_context(|| format!("create {}", parent.join("cache").display()))?;
    fs::create_dir_all(parent.join("vectors"))
        .with_context(|| format!("create {}", parent.join("vectors").display()))?;

    if config_path.exists() {
        return load(config_path);
    }

    let cfg = Config::default();
    let body = toml::to_string_pretty(&cfg).context("serialize default config")?;
    fs::write(config_path, body).with_context(|| format!("write {}", config_path.display()))?;
    Ok(cfg)
}

pub fn save_config(config_path: &Path, cfg: &Config) -> Result<()> {
    let parent = config_path
        .parent()
        .context("config path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let body = toml::to_string_pretty(cfg).context("serialize config")?;
    fs::write(config_path, body).with_context(|| format!("write {}", config_path.display()))?;
    Ok(())
}

fn load(config_path: &Path) -> Result<Config> {
    let body = fs::read_to_string(config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let cfg: Config =
        toml::from_str(&body).with_context(|| format!("parse TOML {}", config_path.display()))?;
    Ok(cfg)
}

pub fn global_sx_dir() -> Result<PathBuf> {
    let home = home_dir().ok_or_else(|| anyhow!("unable to resolve home directory"))?;
    Ok(home.join(".sx"))
}

pub fn global_credentials_path() -> Result<PathBuf> {
    Ok(global_sx_dir()?.join("credentials.toml"))
}

pub fn load_global_credentials() -> Result<GlobalCredentials> {
    let path = global_credentials_path()?;
    if !path.exists() {
        return Ok(GlobalCredentials::default());
    }
    let body = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let creds: GlobalCredentials =
        toml::from_str(&body).with_context(|| format!("parse TOML {}", path.display()))?;
    Ok(creds)
}

pub fn store_global_zhipu_api_key(key: &str) -> Result<PathBuf> {
    let key = key.trim();
    if key.is_empty() {
        return Err(anyhow!("empty key"));
    }

    let path = global_credentials_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("credentials path has no parent"))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;

    let mut creds = load_global_credentials().unwrap_or_default();
    creds.zhipu_api_key = Some(key.to_string());

    let body = toml::to_string_pretty(&creds).context("serialize global credentials")?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    set_private_permissions_if_possible(&path);
    Ok(path)
}

pub fn clear_global_zhipu_api_key() -> Result<PathBuf> {
    let path = global_credentials_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("credentials path has no parent"))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;

    let mut creds = load_global_credentials().unwrap_or_default();
    creds.zhipu_api_key = None;

    let body = toml::to_string_pretty(&creds).context("serialize global credentials")?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    set_private_permissions_if_possible(&path);
    Ok(path)
}

pub fn has_global_zhipu_api_key() -> Result<bool> {
    let creds = load_global_credentials()?;
    Ok(creds
        .zhipu_api_key
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false))
}

pub fn store_global_voyage_api_key(key: &str) -> Result<PathBuf> {
    let key = key.trim();
    if key.is_empty() {
        return Err(anyhow!("empty key"));
    }

    let path = global_credentials_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("credentials path has no parent"))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;

    let mut creds = load_global_credentials().unwrap_or_default();
    creds.voyage_api_key = Some(key.to_string());

    let body = toml::to_string_pretty(&creds).context("serialize global credentials")?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    set_private_permissions_if_possible(&path);
    Ok(path)
}

pub fn clear_global_voyage_api_key() -> Result<PathBuf> {
    let path = global_credentials_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("credentials path has no parent"))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;

    let mut creds = load_global_credentials().unwrap_or_default();
    creds.voyage_api_key = None;

    let body = toml::to_string_pretty(&creds).context("serialize global credentials")?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    set_private_permissions_if_possible(&path);
    Ok(path)
}

pub fn has_global_voyage_api_key() -> Result<bool> {
    let creds = load_global_credentials()?;
    Ok(creds
        .voyage_api_key
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false))
}

pub fn resolve_zhipu_api_key(env_name: &str) -> Result<String> {
    if !env_name.trim().is_empty() {
        if let Ok(v) = std::env::var(env_name) {
            if !v.trim().is_empty() {
                return Ok(v);
            }
        }
    }

    let creds = load_global_credentials()?;
    if let Some(v) = creds.zhipu_api_key {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    let env_hint = if env_name.trim().is_empty() {
        "ZAI_API_KEY".to_string()
    } else {
        env_name.to_string()
    };
    Err(anyhow!(
        "missing Zhipu API key (set {} or run `sx auth zhipu`)",
        env_hint
    ))
}

pub fn resolve_voyage_api_key(env_name: &str) -> Result<String> {
    if !env_name.trim().is_empty() {
        if let Ok(v) = std::env::var(env_name) {
            if !v.trim().is_empty() {
                return Ok(v);
            }
        }
    }

    let creds = load_global_credentials()?;
    if let Some(v) = creds.voyage_api_key {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    let env_hint = if env_name.trim().is_empty() {
        "VOYAGE_API_KEY".to_string()
    } else {
        env_name.to_string()
    };
    Err(anyhow!(
        "missing Voyage API key (set {} or run `sx auth voyage`)",
        env_hint
    ))
}

fn home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        if !profile.is_empty() {
            return Some(PathBuf::from(profile));
        }
    }
    let drive = std::env::var_os("HOMEDRIVE");
    let path = std::env::var_os("HOMEPATH");
    match (drive, path) {
        (Some(d), Some(p)) if !d.is_empty() && !p.is_empty() => {
            let mut buf = PathBuf::from(d);
            buf.push(p);
            Some(buf)
        }
        _ => None,
    }
}

fn set_private_permissions_if_possible(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
}
