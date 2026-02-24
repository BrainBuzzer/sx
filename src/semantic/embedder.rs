use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use serde_json::Value;

use crate::config;

pub trait Embedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub fn build_embedder(
    cfg: &config::Config,
    provider: super::EmbedProvider,
    model: &str,
    dimensions: Option<usize>,
) -> Result<Box<dyn Embedder>> {
    match provider {
        super::EmbedProvider::Ollama => Ok(Box::new(OllamaEmbedder::new(
            &cfg.providers.ollama.base_url,
            model,
        )?)),
        super::EmbedProvider::OpenAi => Ok(Box::new(OpenAiEmbedder::new(
            &cfg.providers.openai.base_url,
            &cfg.providers.openai.api_key_env,
            model,
            dimensions,
        )?)),
        super::EmbedProvider::Voyage => Ok(Box::new(VoyageEmbedder::new(
            &cfg.providers.voyage.base_url,
            &cfg.providers.voyage.api_key_env,
            model,
        )?)),
    }
}

#[derive(Clone)]
struct OllamaEmbedder {
    agent: ureq::Agent,
    base_url: String,
    model: String,
}

impl OllamaEmbedder {
    fn new(base_url: &str, model: &str) -> Result<Self> {
        let cfg = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        let agent: ureq::Agent = cfg.into();
        Ok(Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        })
    }

    fn embed_v2(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);
        let payload = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });
        let json = post_json_with_retry(&self.agent, &url, &payload, None)?;
        parse_embeddings_response(&json, inputs.len())
    }

    fn embed_v1_single(&self, input: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": input,
        });
        let json = post_json_with_retry(&self.agent, &url, &payload, None)?;
        let emb = json
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("ollama /api/embeddings: missing embedding array"))?;
        parse_f32_array(emb)
    }

    fn embed_single_with_shrinking(&self, input: &str) -> Result<Vec<f32>> {
        const MIN_BYTES: usize = 256;

        let mut text = input.to_string();
        let mut last_err: Option<anyhow::Error> = None;

        for _ in 0..8 {
            match self.embed_v2(std::slice::from_ref(&text)) {
                Ok(mut vecs) => {
                    return Ok(vecs.pop().unwrap_or_default());
                }
                Err(err) if is_status_code(&err, 404) => {
                    return self.embed_v1_single_with_shrinking(&text);
                }
                Err(err) if is_context_length_error(&err) && text.len() > MIN_BYTES => {
                    let target = (text.len() / 2).max(MIN_BYTES);
                    truncate_to_char_boundary(&mut text, target);
                    last_err = Some(err);
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("ollama embed: failed"))).context(
            "ollama embed: input too long even after truncation (try lowering [embed].max_chars)",
        )
    }

    fn embed_v1_single_with_shrinking(&self, input: &str) -> Result<Vec<f32>> {
        const MIN_BYTES: usize = 256;

        let mut text = input.to_string();
        let mut last_err: Option<anyhow::Error> = None;

        for _ in 0..8 {
            match self.embed_v1_single(&text) {
                Ok(v) => return Ok(v),
                Err(err) if is_context_length_error(&err) && text.len() > MIN_BYTES => {
                    let target = (text.len() / 2).max(MIN_BYTES);
                    truncate_to_char_boundary(&mut text, target);
                    last_err = Some(err);
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("ollama embed: failed"))).context(
            "ollama embed: input too long even after truncation (try lowering [embed].max_chars)",
        )
    }
}

impl Embedder for OllamaEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        self.embed_batch_robust(inputs)
    }
}

impl OllamaEmbedder {
    fn embed_batch_robust(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        match self.embed_v2(inputs) {
            Ok(v) => Ok(v),
            Err(err) if is_status_code(&err, 404) => {
                let mut out = Vec::with_capacity(inputs.len());
                for t in inputs {
                    out.push(self.embed_v1_single_with_shrinking(t)?);
                }
                Ok(out)
            }
            Err(err) if should_split_on_error(&err) => {
                if inputs.len() == 1 {
                    let v = self.embed_single_with_shrinking(&inputs[0])?;
                    return Ok(vec![v]);
                }

                let mid = inputs.len() / 2;
                let mut left = self.embed_batch_robust(&inputs[..mid])?;
                let right = self.embed_batch_robust(&inputs[mid..])?;
                left.extend(right);
                Ok(left)
            }
            Err(err) => Err(err),
        }
    }
}

#[derive(Clone)]
struct OpenAiEmbedder {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: String,
    model: String,
    dimensions: Option<usize>,
}

impl OpenAiEmbedder {
    fn new(
        base_url: &str,
        api_key_env: &str,
        model: &str,
        dimensions: Option<usize>,
    ) -> Result<Self> {
        let cfg = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        let agent: ureq::Agent = cfg.into();
        Ok(Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key_env: api_key_env.to_string(),
            model: model.to_string(),
            dimensions,
        })
    }
}

impl Embedder for OpenAiEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let api_key = std::env::var(&self.api_key_env)
            .with_context(|| format!("missing OpenAI API key env var {}", self.api_key_env))?;

        let url = format!("{}/embeddings", self.base_url);
        let mut payload = serde_json::json!({
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        });
        if let Some(dim) = self.dimensions {
            payload["dimensions"] = Value::from(dim as i64);
        }

        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json_with_retry(&self.agent, &url, &payload, Some(&headers))?;

        let data = json
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("openai embeddings: missing data array"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("openai embeddings: missing embedding array"))?;
            out.push(parse_f32_array(emb)?);
        }
        if out.len() != inputs.len() {
            return Err(anyhow!(
                "openai embeddings: expected {} vectors, got {}",
                inputs.len(),
                out.len()
            ));
        }
        Ok(out)
    }
}

#[derive(Clone)]
struct VoyageEmbedder {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: String,
    model: String,
}

impl VoyageEmbedder {
    fn new(base_url: &str, api_key_env: &str, model: &str) -> Result<Self> {
        let cfg = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        let agent: ureq::Agent = cfg.into();
        Ok(Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key_env: api_key_env.to_string(),
            model: model.to_string(),
        })
    }
}

impl Embedder for VoyageEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let api_key = config::resolve_voyage_api_key(&self.api_key_env)?;

        let url = format!("{}/embeddings", self.base_url);
        let payload = serde_json::json!({
            "model": self.model,
            "input": inputs,
            "input_type": "document",
            "output_dtype": "float",
        });
        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json_with_retry(&self.agent, &url, &payload, Some(&headers))?;

        if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
            let mut out = Vec::with_capacity(data.len());
            for item in data {
                let emb = item
                    .get("embedding")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| anyhow!("voyage embeddings: missing embedding array"))?;
                out.push(parse_f32_array(emb)?);
            }
            if out.len() != inputs.len() {
                return Err(anyhow!(
                    "voyage embeddings: expected {} vectors, got {}",
                    inputs.len(),
                    out.len()
                ));
            }
            return Ok(out);
        }

        parse_embeddings_response(&json, inputs.len())
    }
}

fn post_json_with_retry(
    agent: &ureq::Agent,
    url: &str,
    payload: &Value,
    headers: Option<&[(&str, String)]>,
) -> Result<Value> {
    let mut delay_ms = 200u64;
    for attempt in 0..3 {
        let mut req = agent.post(url);
        if let Some(hs) = headers {
            for (k, v) in hs {
                req = req.header(*k, v.as_str());
            }
        }

        match req.send_json(payload) {
            Ok(mut response) => {
                let status = response.status().as_u16();
                let body = response
                    .body_mut()
                    .read_to_string()
                    .context("read response body")?;

                if status >= 400 {
                    let msg = truncate_for_error(&extract_error_message(&body), 800);
                    let err = HttpStatusError { status, body: msg };

                    let retryable = status == 429 || status >= 500;
                    if attempt + 1 < 3 && retryable {
                        std::thread::sleep(Duration::from_millis(delay_ms));
                        delay_ms *= 2;
                        continue;
                    }

                    return Err(anyhow!(err)).with_context(|| format!("POST {url}"));
                }

                let json: Value = serde_json::from_str(&body).context("parse json body")?;
                return Ok(json);
            }
            Err(e) => {
                let retryable = matches!(
                    e,
                    ureq::Error::Timeout(_)
                        | ureq::Error::ConnectionFailed
                        | ureq::Error::Io(_)
                        | ureq::Error::HostNotFound
                );

                if attempt + 1 < 3 && retryable {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                    delay_ms *= 2;
                    continue;
                }

                return Err(anyhow!(e)).with_context(|| format!("POST {url}"));
            }
        };
    }
    Err(anyhow!("failed POST {url}"))
}

#[derive(Debug)]
struct HttpStatusError {
    status: u16,
    body: String,
}

impl std::fmt::Display for HttpStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.body.trim().is_empty() {
            write!(f, "http status: {}", self.status)
        } else {
            write!(f, "http status: {} body: {}", self.status, self.body.trim())
        }
    }
}

impl std::error::Error for HttpStatusError {}

fn extract_error_message(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if let Some(s) = v.get("error").and_then(|x| x.as_str()) {
            return s.to_string();
        }
        if let Some(s) = v.pointer("/error/message").and_then(|x| x.as_str()) {
            return s.to_string();
        }
        if let Some(s) = v.get("message").and_then(|x| x.as_str()) {
            return s.to_string();
        }
    }

    trimmed.to_string()
}

fn truncate_for_error(input: &str, max_len: usize) -> String {
    let mut s = input.trim().to_string();
    if s.len() <= max_len {
        return s;
    }
    truncate_to_char_boundary(&mut s, max_len);
    s.push('…');
    s
}

fn truncate_to_char_boundary(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut idx = max_bytes.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    s.truncate(idx);
}

fn parse_embeddings_response(json: &Value, expected: usize) -> Result<Vec<Vec<f32>>> {
    if let Some(embs) = json.get("embeddings").and_then(|v| v.as_array()) {
        let mut out = Vec::with_capacity(embs.len());
        for e in embs {
            let arr = e
                .as_array()
                .ok_or_else(|| anyhow!("ollama /api/embed: embedding is not an array"))?;
            out.push(parse_f32_array(arr)?);
        }
        if out.len() != expected {
            return Err(anyhow!(
                "ollama /api/embed: expected {expected} embeddings, got {}",
                out.len()
            ));
        }
        return Ok(out);
    }

    if let Some(emb) = json.get("embedding").and_then(|v| v.as_array()) {
        let one = parse_f32_array(emb)?;
        return Ok(vec![one]);
    }

    Err(anyhow!("embedding response missing embeddings/embedding"))
}

fn parse_f32_array(arr: &[Value]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let f = v
            .as_f64()
            .ok_or_else(|| anyhow!("embedding contains non-numeric value"))?;
        out.push(f as f32);
    }
    Ok(out)
}

fn is_status_code(err: &anyhow::Error, code: u16) -> bool {
    if let Some(e) = err.downcast_ref::<HttpStatusError>() {
        return e.status == code;
    }
    err.downcast_ref::<ureq::Error>()
        .is_some_and(|e| matches!(e, ureq::Error::StatusCode(c) if *c == code))
}

fn is_context_length_error(err: &anyhow::Error) -> bool {
    let Some(e) = err.downcast_ref::<HttpStatusError>() else {
        return false;
    };
    if e.status != 400 {
        return false;
    }
    let msg = e.body.to_ascii_lowercase();
    msg.contains("context length") || msg.contains("input length exceeds")
}

fn should_split_on_error(err: &anyhow::Error) -> bool {
    if is_context_length_error(err) {
        return true;
    }
    let Some(e) = err.downcast_ref::<HttpStatusError>() else {
        return false;
    };
    e.status == 413
}
