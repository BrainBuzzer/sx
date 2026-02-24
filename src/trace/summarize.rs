use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use serde_json::Value;

use crate::config;

use super::types::{Citation, TracePath};

#[derive(Debug, Clone)]
pub struct SummaryDecision {
    pub summary: String,
    pub source: String,
    pub model: Option<String>,
    pub error: Option<String>,
}

impl SummaryDecision {
    pub fn deterministic(summary: String) -> Self {
        Self {
            summary,
            source: "deterministic".to_string(),
            model: None,
            error: None,
        }
    }
}

pub fn deterministic_summary(query: &str, traces: &[TracePath], citations: &[Citation]) -> String {
    if traces.is_empty() {
        return format!("No confident semantic flow was found for \"{query}\".");
    }

    let best = &traces[0];
    let mut out = String::new();
    out.push_str(&format!(
        "Likely answer: `{}` is handled by `{}` at {}:{}.",
        query.trim(),
        best.root_symbol,
        best.root_path,
        best.root_line
    ));

    if !best.steps.is_empty() {
        out.push_str(" Main flow: ");
        let mut parts = Vec::new();
        for step in best.steps.iter().take(5) {
            let target = step
                .to_symbol
                .clone()
                .or(step.target_name.clone())
                .unwrap_or_else(|| "-".to_string());
            parts.push(format!("{} -> {}", step.from_symbol, target));
        }
        out.push_str(&parts.join(" -> "));
        out.push('.');
    }

    if traces.len() > 1 {
        out.push_str(&format!(" Alternatives reviewed: {}.", traces.len() - 1));
    }

    if !citations.is_empty() {
        out.push_str(" Evidence: ");
        for (i, c) in citations.iter().take(6).enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{}:{}", c.path, c.line));
        }
    }

    out
}

pub fn maybe_llm_summary(
    cfg: &config::Config,
    query: &str,
    traces: &[TracePath],
    citations: &[Citation],
    fallback: &str,
) -> SummaryDecision {
    if traces.is_empty() || !cfg.trace.llm_summary {
        return SummaryDecision::deterministic(fallback.to_string());
    }

    if cfg.llm.provider.trim().eq_ignore_ascii_case("none") {
        return SummaryDecision::deterministic(fallback.to_string());
    }

    match llm_summary(cfg, query, traces, citations) {
        Ok(s) if !s.trim().is_empty() => SummaryDecision {
            summary: s.trim().to_string(),
            source: "llm".to_string(),
            model: active_llm_model(cfg),
            error: None,
        },
        Ok(_) => SummaryDecision {
            summary: fallback.to_string(),
            source: "deterministic".to_string(),
            model: None,
            error: Some("llm returned empty summary".to_string()),
        },
        Err(err) => SummaryDecision {
            summary: fallback.to_string(),
            source: "deterministic".to_string(),
            model: None,
            error: Some(compact_error(&err)),
        },
    }
}

fn llm_summary(
    cfg: &config::Config,
    query: &str,
    traces: &[TracePath],
    citations: &[Citation],
) -> Result<String> {
    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    let prompt = summary_prompt(query, traces, citations);

    if provider == "openai" {
        let api_key = std::env::var(&cfg.providers.openai.api_key_env).with_context(|| {
            format!(
                "missing OpenAI API key env var {}",
                cfg.providers.openai.api_key_env
            )
        })?;
        let url = format!(
            "{}/responses",
            cfg.providers.openai.base_url.trim_end_matches('/')
        );
        let payload = serde_json::json!({
            "model": cfg.llm.model,
            "input": prompt,
            "max_output_tokens": cfg.llm.max_output_tokens as i64,
            "temperature": cfg.llm.temperature,
        });
        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json(&url, &payload, Some(&headers))?;
        return extract_openai_output_text(&json)
            .ok_or_else(|| anyhow!("openai responses: missing output text"));
    }

    if provider == "zhipu" || provider == "codex" {
        let api_key = config::resolve_zhipu_api_key(&cfg.providers.zhipu.api_key_env)?;
        let url = format!(
            "{}/chat/completions",
            cfg.providers.zhipu.base_url.trim_end_matches('/')
        );
        let payload = serde_json::json!({
            "model": cfg.llm.zhipu_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a codebase trace summarizer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": cfg.llm.max_output_tokens as i64,
            "temperature": cfg.llm.temperature,
            "stream": false,
        });
        let headers = [("Authorization", format!("Bearer {api_key}"))];
        let json = post_json(&url, &payload, Some(&headers))?;
        return extract_chat_completion_text(&json)
            .ok_or_else(|| anyhow!("zhipu chat.completions: missing choices[0].message.content"));
    }

    if provider == "ollama" {
        let url = format!(
            "{}/api/generate",
            cfg.providers.ollama.base_url.trim_end_matches('/')
        );
        let payload = serde_json::json!({
            "model": cfg.llm.ollama_model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": cfg.llm.temperature,
                "num_predict": cfg.llm.max_output_tokens as i64,
            }
        });
        let json = post_json(&url, &payload, None)?;
        return json
            .get("response")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("ollama generate: missing response"));
    }

    Err(anyhow!("unsupported llm provider {}", cfg.llm.provider))
}

fn active_llm_model(cfg: &config::Config) -> Option<String> {
    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    let model = match provider.as_str() {
        "zhipu" | "codex" => cfg.llm.zhipu_model.trim(),
        "ollama" => cfg.llm.ollama_model.trim(),
        "openai" => cfg.llm.model.trim(),
        _ => "",
    };
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}

fn compact_error(err: &anyhow::Error) -> String {
    let mut out = err.to_string();
    for cause in err.chain().skip(1) {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        if out.len() > 240 {
            break;
        }
    }
    if out.len() > 240 {
        out.truncate(240);
        out.push('…');
    }
    out
}

fn summary_prompt(query: &str, traces: &[TracePath], citations: &[Citation]) -> String {
    let mut p = String::new();
    p.push_str("You are answering a repository navigation question using trace evidence.\n");
    p.push_str("Rules:\n");
    p.push_str("1) First sentence must directly answer the query.\n");
    p.push_str("2) Mention one primary flow and at most one alternate.\n");
    p.push_str("3) Include concrete file:line citations.\n");
    p.push_str("4) Keep under 140 words.\n");
    p.push_str("Query:\n");
    p.push_str(query);
    p.push('\n');
    p.push_str("Top paths:\n");
    for (i, t) in traces.iter().take(4).enumerate() {
        p.push_str(&format!(
            "{}. root={} @ {}:{} score={:.3}\n",
            i + 1,
            t.root_symbol,
            t.root_path,
            t.root_line,
            t.score
        ));
        for step in t.steps.iter().take(5) {
            p.push_str(&format!(
                "   - {} -> {} [{}] {}:{}\n",
                step.from_symbol,
                step.target_name.clone().unwrap_or_else(|| "-".to_string()),
                step.edge_kind,
                step.path,
                step.line
            ));
        }
    }
    p.push_str("Citations:\n");
    for c in citations.iter().take(10) {
        p.push_str(&format!("- {}:{}\n", c.path, c.line));
    }
    p.push_str("Summary:");
    p
}

fn post_json(url: &str, payload: &Value, headers: Option<&[(&str, String)]>) -> Result<Value> {
    let cfg = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(25)))
        .build();
    let agent: ureq::Agent = cfg.into();
    let mut req = agent.post(url);
    if let Some(hs) = headers {
        for (k, v) in hs {
            req = req.header(*k, v.as_str());
        }
    }
    let mut resp = req.send_json(payload).context("send llm request")?;
    let status = resp.status().as_u16();
    let body = resp
        .body_mut()
        .read_to_string()
        .context("read llm response")?;
    if status >= 400 {
        return Err(anyhow!("llm http status {}: {}", status, body));
    }
    let json: Value = serde_json::from_str(&body).context("parse llm response json")?;
    Ok(json)
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
            if c.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                if let Some(t) = c.get("text").and_then(|v| v.as_str()) {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(t);
                }
            }
        }
    }
    if buf.trim().is_empty() {
        None
    } else {
        Some(buf)
    }
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
