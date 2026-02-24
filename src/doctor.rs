use std::path::Path;

use anyhow::{Context as _, Result};

use crate::{config, db};
use rusqlite::OptionalExtension;

pub fn run(root_dir: &Path, db_path: &Path, cfg: &config::Config) -> Result<()> {
    println!("root: {}", root_dir.display());
    println!("db:   {}", db_path.display());

    let cfg_path = root_dir.join(".sx/config.toml");
    if cfg_path.exists() {
        println!("config: {} (ok)", cfg_path.display());
    } else {
        println!(
            "config: {} (missing; defaults will be used)",
            cfg_path.display()
        );
    }

    let editor = if !cfg.open.editor.trim().is_empty() {
        Some(cfg.open.editor.trim().to_string())
    } else {
        std::env::var("EDITOR").ok()
    };
    match editor {
        Some(e) if !e.trim().is_empty() => println!("editor: {e}"),
        _ => println!("editor: (missing) set [open].editor or $EDITOR"),
    }

    println!("llm.provider: {}", cfg.llm.provider);
    let model = match cfg.llm.provider.trim().to_ascii_lowercase().as_str() {
        "zhipu" | "codex" => cfg.llm.zhipu_model.as_str(),
        "ollama" => cfg.llm.ollama_model.as_str(),
        _ => cfg.llm.model.as_str(),
    };
    println!("llm.model: {model}");
    match cfg.llm.provider.trim().to_ascii_lowercase().as_str() {
        "openai" => {
            let set = std::env::var(&cfg.providers.openai.api_key_env).is_ok();
            println!(
                "llm.auth: {} ({})",
                if set { "configured" } else { "missing" },
                cfg.providers.openai.api_key_env
            );
        }
        "zhipu" | "codex" => {
            let env_set = std::env::var(&cfg.providers.zhipu.api_key_env).is_ok();
            let global_set = config::has_global_zhipu_api_key().unwrap_or(false);
            println!(
                "llm.auth: {} (env={}, global=~/.sx/credentials.toml)",
                if env_set || global_set {
                    "configured"
                } else {
                    "missing"
                },
                cfg.providers.zhipu.api_key_env
            );
        }
        _ => {}
    }

    db::check_fts5_available().context("FTS5 not available")?;
    println!("fts5: ok");

    let conn = db::open(db_path)?;
    db::migrate(&conn)?;
    let schema_version: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |row| row.get(0),
        )
        .optional()
        .context("read schema_version")?;
    println!(
        "schema_version: {}",
        schema_version.as_deref().unwrap_or("(missing)")
    );

    Ok(())
}
