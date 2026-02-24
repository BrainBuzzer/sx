use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use rusqlite::{Connection, OptionalExtension};
use serde_json::Value;

use crate::config;

pub fn expand_queries_cached(
    conn: &Connection,
    cfg: &config::Config,
    provider: super::LlmProvider,
    query: &str,
    hints: &[String],
) -> Result<Vec<String>> {
    let provider_str = match provider {
        super::LlmProvider::None => return Ok(Vec::new()),
        super::LlmProvider::Ollama => "ollama",
        super::LlmProvider::OpenAi => "openai",
        super::LlmProvider::Zhipu => "zhipu",
    };

    let model = match provider {
        super::LlmProvider::Ollama => cfg.llm.ollama_model.clone(),
        super::LlmProvider::OpenAi => cfg.llm.model.clone(),
        super::LlmProvider::Zhipu => cfg.llm.zhipu_model.clone(),
        super::LlmProvider::None => String::new(),
    };

    if let Some(cached) = get_cached_expansions(conn, provider_str, &model, query)? {
        return Ok(cached);
    }

    let client: Box<dyn LlmClient> = match provider {
        super::LlmProvider::Ollama => Box::new(OllamaClient::new(
            &cfg.providers.ollama.base_url,
            &model,
            cfg.llm.temperature,
            cfg.llm.max_output_tokens,
        )?),
        super::LlmProvider::OpenAi => Box::new(OpenAiClient::new(
            &cfg.providers.openai.base_url,
            &cfg.providers.openai.api_key_env,
            &model,
            cfg.llm.temperature,
            cfg.llm.max_output_tokens,
        )?),
        super::LlmProvider::Zhipu => Box::new(ZhipuClient::new(
            &cfg.providers.zhipu.base_url,
            &cfg.providers.zhipu.api_key_env,
            &model,
            cfg.llm.temperature,
            cfg.llm.max_output_tokens,
        )?),
        super::LlmProvider::None => return Ok(Vec::new()),
    };

    let expansions = client.expand_queries(query, hints)?;
    put_cached_expansions(conn, provider_str, &model, query, &expansions)?;
    Ok(expansions)
}

trait LlmClient {
    fn expand_queries(&self, query: &str, hints: &[String]) -> Result<Vec<String>>;
}

#[derive(Clone)]
struct OllamaClient {
    agent: ureq::Agent,
    base_url: String,
    model: String,
    temperature: f64,
    max_output_tokens: usize,
}

impl OllamaClient {
    fn new(
        base_url: &str,
        model: &str,
        temperature: f64,
        max_output_tokens: usize,
    ) -> Result<Self> {
        let cfg = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        let agent: ureq::Agent = cfg.into();
        Ok(Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            temperature,
            max_output_tokens,
        })
    }
}

impl LlmClient for OllamaClient {
    fn expand_queries(&self, query: &str, hints: &[String]) -> Result<Vec<String>> {
        let url = format!("{}/api/generate", self.base_url);
        let prompt = expansion_prompt(query, hints);
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens as i64,
            }
        });

        let json = post_json_with_retry(&self.agent, &url, &payload, None)?;
        let resp = json
            .get("response")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("ollama generate: missing response"))?;
        Ok(parse_expansions(resp, query))
    }
}

#[derive(Clone)]
struct OpenAiClient {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: String,
    model: String,
    temperature: f64,
    max_output_tokens: usize,
}

impl OpenAiClient {
    fn new(
        base_url: &str,
        api_key_env: &str,
        model: &str,
        temperature: f64,
        max_output_tokens: usize,
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
            temperature,
            max_output_tokens,
        })
    }
}

impl LlmClient for OpenAiClient {
    fn expand_queries(&self, query: &str, hints: &[String]) -> Result<Vec<String>> {
        let api_key = std::env::var(&self.api_key_env)
            .with_context(|| format!("missing OpenAI API key env var {}", self.api_key_env))?;

        let url = format!("{}/responses", self.base_url);
        let prompt = expansion_prompt(query, hints);
        let payload = serde_json::json!({
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens as i64,
            "temperature": self.temperature,
        });

        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json_with_retry(&self.agent, &url, &payload, Some(&headers))?;
        let text = extract_openai_output_text(&json)
            .ok_or_else(|| anyhow!("openai responses: missing output text"))?;
        Ok(parse_expansions(&text, query))
    }
}

#[derive(Clone)]
struct ZhipuClient {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: String,
    model: String,
    temperature: f64,
    max_output_tokens: usize,
}

impl ZhipuClient {
    fn new(
        base_url: &str,
        api_key_env: &str,
        model: &str,
        temperature: f64,
        max_output_tokens: usize,
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
            temperature,
            max_output_tokens,
        })
    }
}

impl LlmClient for ZhipuClient {
    fn expand_queries(&self, query: &str, hints: &[String]) -> Result<Vec<String>> {
        let api_key = config::resolve_zhipu_api_key(&self.api_key_env)?;

        let url = format!("{}/chat/completions", self.base_url);
        let prompt = expansion_prompt(query, hints);
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You expand code search queries for a local repository search tool. Return 1 to 2 alternatives, one per line. No numbering, no quotes, no extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "stream": false,
            "max_tokens": self.max_output_tokens as i64
        });

        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json_with_retry(&self.agent, &url, &payload, Some(&headers))?;
        let text = extract_chat_completion_text(&json)
            .ok_or_else(|| anyhow!("zhipu chat.completions: missing choices[0].message.content"))?;
        Ok(parse_expansions(&text, query))
    }
}

fn expansion_prompt(query: &str, hints: &[String]) -> String {
    let mut p = String::new();
    p.push_str("You expand code search queries for a local repository search tool.\n");
    p.push_str("Return 1 to 2 alternative search queries, one per line.\n");
    p.push_str("Rules: no numbering, no quotes, no extra text.\n\n");
    p.push_str("Original query:\n");
    p.push_str(query.trim());
    p.push_str("\n\n");
    if !hints.is_empty() {
        p.push_str("Hints (identifiers/symbols you may include):\n");
        for h in hints.iter().take(12) {
            p.push_str("- ");
            p.push_str(h);
            p.push('\n');
        }
        p.push('\n');
    }
    p.push_str("Alternative queries:\n");
    p
}

fn parse_expansions(body: &str, original: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in body.lines() {
        let s = line.trim().trim_matches('"').trim();
        if s.is_empty() {
            continue;
        }
        if s.eq_ignore_ascii_case(original.trim()) {
            continue;
        }
        out.push(s.to_string());
        if out.len() >= 2 {
            break;
        }
    }
    out
}

fn extract_openai_output_text(json: &Value) -> Option<String> {
    if let Some(s) = json.get("output_text").and_then(|v| v.as_str()) {
        return Some(s.to_string());
    }

    let output = json.get("output")?.as_array()?;
    let mut buf = String::new();
    for item in output {
        let Some(content) = item.get("content").and_then(|v| v.as_array()) else {
            continue;
        };
        for c in content {
            let ctype = c.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if ctype == "output_text" {
                if let Some(t) = c.get("text").and_then(|v| v.as_str()) {
                    buf.push_str(t);
                    buf.push('\n');
                }
            }
        }
    }
    let t = buf.trim().to_string();
    if t.is_empty() { None } else { Some(t) }
}

fn extract_chat_completion_text(json: &Value) -> Option<String> {
    let choices = json.get("choices")?.as_array()?;
    let first = choices.first()?;
    let content = first.pointer("/message/content")?;

    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }

    if let Some(arr) = content.as_array() {
        let mut buf = String::new();
        for item in arr {
            if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(t);
            }
        }
        if !buf.trim().is_empty() {
            return Some(buf);
        }
    }

    None
}

fn get_cached_expansions(
    conn: &Connection,
    provider: &str,
    model: &str,
    query: &str,
) -> Result<Option<Vec<String>>> {
    let json: Option<String> = conn
        .query_row(
            r#"
SELECT expansions_json
FROM llm_expansion_cache
WHERE provider=?1 AND model=?2 AND query=?3
LIMIT 1
"#,
            rusqlite::params![provider, model, query],
            |row| row.get(0),
        )
        .optional()
        .context("read llm_expansion_cache")?;
    let Some(json) = json else {
        return Ok(None);
    };
    let vals: Vec<String> = serde_json::from_str(&json).unwrap_or_default();
    Ok(Some(vals))
}

fn put_cached_expansions(
    conn: &Connection,
    provider: &str,
    model: &str,
    query: &str,
    expansions: &[String],
) -> Result<()> {
    let json = serde_json::to_string(expansions).context("serialize expansions")?;
    conn.execute(
        r#"
INSERT INTO llm_expansion_cache(provider, model, query, expansions_json, updated_at)
VALUES(?1, ?2, ?3, ?4, strftime('%s','now'))
ON CONFLICT(provider, model, query) DO UPDATE SET
  expansions_json=excluded.expansions_json,
  updated_at=excluded.updated_at
"#,
        rusqlite::params![provider, model, query, json],
    )
    .context("upsert llm_expansion_cache")?;
    Ok(())
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
