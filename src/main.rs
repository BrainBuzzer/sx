mod actions;
mod cli;
mod config;
mod db;
mod doctor;
mod index;
mod root;
mod search;
mod semantic;
mod trace;
mod tui;

use anyhow::Context as _;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal as _;
use std::path::Path;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    init_tracing(cli.verbose);

    let command = cli.command.unwrap_or(cli::Command::Tui);
    match command {
        cli::Command::Auth(args) => run_auth(args)?,
        command => {
            let cwd = std::env::current_dir().context("get current working directory")?;

            let root = match cli.root {
                Some(root) => root::normalize_root(&cwd, &root)?,
                None => root::discover_root(&cwd)?,
            };

            let config_path = root.join(".sx/config.toml");

            match command {
                cli::Command::Init => {
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    run_init_bootstrap(&root, &db_path, &config_path)?;
                }
                cli::Command::Onboard(args) => {
                    let mut config = config::ensure_config(&config_path)?;
                    run_onboard(&mut config, &config_path, args)?;
                }
                cli::Command::Index(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let mut conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    let stats = index::run(&mut conn, &root, &config, args)?;
                    println!(
                        "scanned={} indexed={} skipped={} chunks={} removed={} duration_ms={}",
                        stats.scanned,
                        stats.indexed,
                        stats.skipped,
                        stats.chunks_written,
                        stats.removed,
                        stats.duration_ms
                    );
                }
                cli::Command::Search(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::search(&conn, &root, &config, args)?;
                }
                cli::Command::Get(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::get(&conn, &root, &config, args)?;
                }
                cli::Command::Open(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::open(&conn, &root, &config, args)?;
                }
                cli::Command::Guide(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::guide(&conn, &root, &config, args)?;
                }
                cli::Command::Cd(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::cd(&conn, &root, &config, args)?;
                }
                cli::Command::Embed(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let mut conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::embed(&mut conn, &root, &db_path, &config, args)?;
                }
                cli::Command::Vsearch(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::vsearch(&conn, &root, &db_path, &config, args)?;
                }
                cli::Command::Query(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::query(&conn, &root, &db_path, &config, args)?;
                }
                cli::Command::Trace(args) => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    actions::trace(&conn, &root, &db_path, &config, args)?;
                }
                cli::Command::Doctor => {
                    let config = config::load_or_default(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    doctor::run(&root, &db_path, &config)?;
                }
                cli::Command::Tui => {
                    let config = config::ensure_config(&config_path)?;
                    let db_path = match cli.db {
                        Some(db_path) => root::normalize_path(&cwd, &db_path),
                        None => root.join(".sx/index.sqlite"),
                    };
                    let conn = db::open(&db_path)?;
                    db::migrate(&conn)?;
                    tui::run(&conn, &root, &db_path, &config)?;
                }
                cli::Command::Auth(_) => unreachable!("handled above"),
            }
        }
    }

    Ok(())
}

fn run_init_bootstrap(root: &Path, db_path: &Path, config_path: &Path) -> anyhow::Result<()> {
    let config_existed_before = config_path.exists();
    let progress = ProgressBar::new(5);
    let style = ProgressStyle::with_template(
        "{spinner:.cyan} [{elapsed_precise}] [{wide_bar:.green/blue}] {pos}/{len} {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    progress.set_style(style);
    progress.enable_steady_tick(Duration::from_millis(100));

    progress.set_message("Ensuring config");
    let config = match config::ensure_config(config_path) {
        Ok(cfg) => {
            progress.inc(1);
            cfg
        }
        Err(err) => {
            progress.abandon_with_message("init failed while ensuring config");
            return Err(err);
        }
    };

    let embed_is_default_ollama = config.embed.provider.trim().eq_ignore_ascii_case("ollama")
        && config.providers.ollama.base_url.trim_end_matches('/') == "http://localhost:11434";
    let should_embed = config_existed_before || !embed_is_default_ollama;

    progress.set_message("Checking embedding API connectivity");
    let embed_probe = if should_embed {
        match semantic::probe_embedding_connectivity(&config, None, None, None) {
            Ok(info) => Some(info),
            Err(err) => {
                progress.abandon_with_message("init failed while checking embedding API");
                return Err(err).context(
                    "embedding API check failed (configure provider/auth and try `sx init` again)",
                );
            }
        }
    } else {
        None
    };
    progress.inc(1);

    progress.set_message("Opening/migrating database");
    let mut conn = match db::open(db_path) {
        Ok(c) => c,
        Err(err) => {
            progress.abandon_with_message("init failed while opening DB");
            return Err(err);
        }
    };
    if let Err(err) = db::migrate(&conn) {
        progress.abandon_with_message("init failed while migrating DB");
        return Err(err);
    }
    progress.inc(1);

    progress.set_message("Indexing repository");
    let index_stats = match index::run(
        &mut conn,
        root,
        &config,
        cli::IndexArgs {
            full: false,
            jobs: 1,
        },
    ) {
        Ok(stats) => {
            progress.inc(1);
            stats
        }
        Err(err) => {
            progress.abandon_with_message("init failed while indexing");
            return Err(err);
        }
    };

    progress.set_message("Building embeddings");
    let embed_result = if should_embed {
        Some(semantic::embed(
            &mut conn,
            db_path,
            &config,
            cli::EmbedArgs {
                full: false,
                provider: None,
                model: None,
                dimensions: None,
                batch_size: None,
            },
        ))
    } else {
        None
    };
    progress.inc(1);
    progress.finish_with_message("Initialization complete");

    println!(
        "Initialized: root={} db={}",
        root.display(),
        db_path.display()
    );
    if let Some(check) = &embed_probe {
        println!(
            "embed_check: provider={} model={} dim={}",
            check.provider, check.model, check.dim
        );
    } else {
        println!("embed_check: skipped (default provider config detected)");
    }
    println!(
        "index: scanned={} indexed={} skipped={} chunks={} removed={} duration_ms={}",
        index_stats.scanned,
        index_stats.indexed,
        index_stats.skipped,
        index_stats.chunks_written,
        index_stats.removed,
        index_stats.duration_ms
    );

    match embed_result {
        Some(Ok(stats)) => {
            println!(
                "embed: collection={} embedded_new={} embedded_kept={} pruned={} dim={} duration_ms={}",
                stats.collection,
                stats.embedded_new,
                stats.embedded_kept,
                stats.pruned,
                stats.dim,
                stats.duration_ms
            );
        }
        Some(Err(err)) => {
            println!(
                "embed: skipped ({err}) - configure/start embedding provider and run `sx embed` later"
            );
        }
        None => {
            println!(
                "embed: skipped (default provider config detected) - run `sx onboard` (or edit .sx/config.toml) then run `sx embed`"
            );
        }
    }

    Ok(())
}

fn run_auth(args: cli::AuthArgs) -> anyhow::Result<()> {
    match args.command {
        cli::AuthCommand::Zhipu(args) => {
            if args.clear {
                let path = config::clear_global_zhipu_api_key()?;
                println!("Cleared stored Zhipu API key in {}", path.display());
                return Ok(());
            }

            let key = if let Some(key) = args.api_key {
                key
            } else if args.from_env {
                std::env::var("ZAI_API_KEY")
                    .context("ZAI_API_KEY is not set (cannot use --from-env)")?
            } else {
                use std::io::Write as _;
                eprint!("Enter ZAI API key: ");
                std::io::stderr().flush().ok();
                rpassword::read_password().context("read API key from terminal")?
            };

            let path = config::store_global_zhipu_api_key(&key)?;
            println!("Stored Zhipu API key in {}", path.display());
            println!("Key source will be used when ZAI_API_KEY is not exported.");
        }
        cli::AuthCommand::Voyage(args) => {
            if args.clear {
                let path = config::clear_global_voyage_api_key()?;
                println!("Cleared stored Voyage API key in {}", path.display());
                return Ok(());
            }

            let key = if let Some(key) = args.api_key {
                key
            } else if args.from_env {
                std::env::var("VOYAGE_API_KEY")
                    .context("VOYAGE_API_KEY is not set (cannot use --from-env)")?
            } else {
                use std::io::Write as _;
                eprint!("Enter VOYAGE API key: ");
                std::io::stderr().flush().ok();
                rpassword::read_password().context("read API key from terminal")?
            };

            let path = config::store_global_voyage_api_key(&key)?;
            println!("Stored Voyage API key in {}", path.display());
            println!("Key source will be used when VOYAGE_API_KEY is not exported.");
        }
        cli::AuthCommand::Status => {
            let path = config::global_credentials_path()?;
            let has_zhipu = config::has_global_zhipu_api_key().unwrap_or(false);
            let has_voyage = config::has_global_voyage_api_key().unwrap_or(false);
            println!("credentials_file: {}", path.display());
            println!(
                "zhipu_api_key: {}",
                if has_zhipu { "configured" } else { "missing" }
            );
            println!(
                "voyage_api_key: {}",
                if has_voyage { "configured" } else { "missing" }
            );
        }
    }
    Ok(())
}

fn run_onboard(
    cfg: &mut config::Config,
    config_path: &Path,
    args: cli::OnboardArgs,
) -> anyhow::Result<()> {
    use anyhow::anyhow;

    if !args.defaults && !is_interactive_terminal() {
        return Err(anyhow!(
            "`sx onboard` requires an interactive terminal; use `sx onboard --defaults` in non-interactive contexts"
        ));
    }

    if !args.defaults {
        eprintln!();
        eprintln!("SX onboarding");
        eprintln!("Quick setup with sensible defaults.");
        eprintln!();

        if prompt_bool("Set a custom editor now?", false)? {
            cfg.open.editor = prompt_line("Editor command", Some(cfg.open.editor.as_str()))?;
        }
        prompt_onboard_embedding_setup(cfg)?;
        if prompt_bool("Configure LLM provider now?", false)? {
            prompt_onboard_llm_setup(cfg)?;
            if prompt_bool("Change trace LLM summary setting now?", false)? {
                cfg.trace.llm_summary =
                    prompt_bool("Enable LLM summary in trace mode?", cfg.trace.llm_summary)?;
            }
        }
    }

    config::save_config(config_path, cfg)?;
    println!("Saved config: {}", config_path.display());
    println!(
        "embed: provider={} model={}",
        cfg.embed.provider, cfg.embed.model
    );
    println!("llm: provider={}", cfg.llm.provider);

    if args.skip_check {
        println!("embed_check: skipped (--skip-check)");
        return Ok(());
    }
    if cfg.embed.provider.trim().eq_ignore_ascii_case("none") {
        println!("embed_check: skipped (embed.provider=none)");
        return Ok(());
    }

    let check = semantic::probe_embedding_connectivity(cfg, None, None, None)
        .context("embedding connectivity check failed")?;
    println!(
        "embed_check: ok provider={} model={} dim={}",
        check.provider, check.model, check.dim
    );

    Ok(())
}

fn is_interactive_terminal() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

fn prompt_onboard_embedding_setup(cfg: &mut config::Config) -> anyhow::Result<()> {
    use anyhow::anyhow;
    use std::io::Write as _;

    let previous_provider = cfg.embed.provider.clone();
    let options = ["ollama", "voyage", "openai", "none"];
    let default_idx = options
        .iter()
        .position(|v| v.eq_ignore_ascii_case(previous_provider.as_str()))
        .unwrap_or(0);
    let choice = prompt_choice("Select embedding provider", &options, default_idx)?;
    let provider = options[choice].to_string();

    cfg.embed.provider = provider.clone();
    let default_model =
        default_embed_model(&provider, &previous_provider, cfg.embed.model.as_str()).to_string();
    if prompt_bool(
        &format!("Use default embedding model `{default_model}`?"),
        true,
    )? {
        cfg.embed.model = default_model.clone();
    } else {
        cfg.embed.model = prompt_line("Embedding model", Some(default_model.as_str()))?;
    }

    match provider.as_str() {
        "ollama" => {
            if prompt_bool("Override default Ollama base URL?", false)? {
                cfg.providers.ollama.base_url = prompt_line(
                    "Ollama base URL",
                    Some(cfg.providers.ollama.base_url.as_str()),
                )?;
            }
        }
        "voyage" => {
            if prompt_bool("Override default Voyage base URL?", false)? {
                cfg.providers.voyage.base_url = prompt_line(
                    "Voyage base URL",
                    Some(cfg.providers.voyage.base_url.as_str()),
                )?;
            }
            if prompt_bool("Use custom Voyage API key env var?", false)? {
                cfg.providers.voyage.api_key_env = prompt_line(
                    "Voyage API key env var",
                    Some(cfg.providers.voyage.api_key_env.as_str()),
                )?;
            }

            let env_name = cfg.providers.voyage.api_key_env.clone();
            let env_has_key = std::env::var(&env_name)
                .ok()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false);
            let global_has_key = config::has_global_voyage_api_key().unwrap_or(false);

            if env_has_key || global_has_key {
                let source = if env_has_key { "env" } else { "global" };
                eprintln!("Voyage API key detected via {source}.");
                if prompt_bool("Update stored global Voyage API key now?", false)? {
                    eprint!("Paste Voyage API key to store globally: ");
                    std::io::stderr().flush().ok();
                    let key = rpassword::read_password().context("read API key from terminal")?;
                    if !key.trim().is_empty() {
                        let path = config::store_global_voyage_api_key(&key)?;
                        eprintln!("Stored Voyage API key in {}", path.display());
                    }
                }
            } else {
                eprintln!("No Voyage API key detected in env ({env_name}) or global store.");
                if prompt_bool("Store Voyage API key globally now?", true)? {
                    eprint!("Enter Voyage API key (stored in ~/.sx/credentials.toml): ");
                    std::io::stderr().flush().ok();
                    let key = rpassword::read_password().context("read API key from terminal")?;
                    if key.trim().is_empty() {
                        return Err(anyhow!(
                            "Voyage API key required (set {} or run `sx auth voyage`)",
                            env_name
                        ));
                    }
                    let path = config::store_global_voyage_api_key(&key)?;
                    eprintln!("Stored Voyage API key in {}", path.display());
                }
            }
        }
        "openai" => {
            if prompt_bool("Override default OpenAI base URL?", false)? {
                cfg.providers.openai.base_url = prompt_line(
                    "OpenAI base URL",
                    Some(cfg.providers.openai.base_url.as_str()),
                )?;
            }
            if prompt_bool("Use custom OpenAI API key env var?", false)? {
                cfg.providers.openai.api_key_env = prompt_line(
                    "OpenAI API key env var",
                    Some(cfg.providers.openai.api_key_env.as_str()),
                )?;
            }
            if !prompt_bool(
                &format!("Use OpenAI embedding dimensions {}?", cfg.embed.dimensions),
                true,
            )? {
                cfg.embed.dimensions =
                    prompt_usize("OpenAI embedding dimensions", cfg.embed.dimensions)?.max(1);
            }
            let env_name = cfg.providers.openai.api_key_env.clone();
            let env_has_key = std::env::var(&env_name)
                .ok()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false);
            if !env_has_key {
                eprintln!("OpenAI API key not found in {env_name}. Set it before embedding.");
            }
        }
        "none" => {}
        _ => {}
    }

    Ok(())
}

fn prompt_onboard_llm_setup(cfg: &mut config::Config) -> anyhow::Result<()> {
    use std::io::Write as _;

    let previous_provider = cfg.llm.provider.clone();
    let options = ["zhipu", "ollama", "openai", "none"];
    let default_idx = options
        .iter()
        .position(|v| v.eq_ignore_ascii_case(previous_provider.as_str()))
        .unwrap_or(0);
    let choice = prompt_choice("Select LLM provider", &options, default_idx)?;
    let provider = options[choice].to_string();
    cfg.llm.provider = provider.clone();

    match provider.as_str() {
        "zhipu" => {
            let default_model =
                default_llm_model("zhipu", &previous_provider, cfg.llm.zhipu_model.as_str());
            if prompt_bool(&format!("Use default Zhipu model `{default_model}`?"), true)? {
                cfg.llm.zhipu_model = default_model.clone();
            } else {
                cfg.llm.zhipu_model = prompt_line("Zhipu model", Some(default_model.as_str()))?;
            }
            if prompt_bool("Override default Zhipu base URL?", false)? {
                cfg.providers.zhipu.base_url = prompt_line(
                    "Zhipu base URL",
                    Some(cfg.providers.zhipu.base_url.as_str()),
                )?;
            }
            if prompt_bool("Use custom Zhipu API key env var?", false)? {
                cfg.providers.zhipu.api_key_env = prompt_line(
                    "Zhipu API key env var",
                    Some(cfg.providers.zhipu.api_key_env.as_str()),
                )?;
            }

            let env_name = cfg.providers.zhipu.api_key_env.clone();
            let env_has_key = std::env::var(&env_name)
                .ok()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false);
            let global_has_key = config::has_global_zhipu_api_key().unwrap_or(false);

            if env_has_key || global_has_key {
                let source = if env_has_key { "env" } else { "global" };
                eprintln!("Zhipu API key detected via {source}.");
                if prompt_bool("Update stored global Zhipu API key now?", false)? {
                    eprint!("Paste Zhipu API key to store globally: ");
                    std::io::stderr().flush().ok();
                    let key = rpassword::read_password().context("read API key from terminal")?;
                    if !key.trim().is_empty() {
                        let path = config::store_global_zhipu_api_key(&key)?;
                        eprintln!("Stored Zhipu API key in {}", path.display());
                    }
                }
            } else {
                eprintln!("No Zhipu API key detected in env ({env_name}) or global store.");
                if prompt_bool("Store Zhipu API key globally now?", false)? {
                    eprint!("Enter Zhipu API key (stored in ~/.sx/credentials.toml): ");
                    std::io::stderr().flush().ok();
                    let key = rpassword::read_password().context("read API key from terminal")?;
                    if !key.trim().is_empty() {
                        let path = config::store_global_zhipu_api_key(&key)?;
                        eprintln!("Stored Zhipu API key in {}", path.display());
                    }
                }
            }
        }
        "ollama" => {
            let default_model =
                default_llm_model("ollama", &previous_provider, cfg.llm.ollama_model.as_str());
            if prompt_bool(
                &format!("Use default Ollama model `{default_model}`?"),
                true,
            )? {
                cfg.llm.ollama_model = default_model.clone();
            } else {
                cfg.llm.ollama_model =
                    prompt_line("Ollama LLM model", Some(default_model.as_str()))?;
            }
            if prompt_bool("Override default Ollama base URL?", false)? {
                cfg.providers.ollama.base_url = prompt_line(
                    "Ollama base URL",
                    Some(cfg.providers.ollama.base_url.as_str()),
                )?;
            }
        }
        "openai" => {
            let default_model =
                default_llm_model("openai", &previous_provider, cfg.llm.model.as_str());
            if prompt_bool(
                &format!("Use default OpenAI model `{default_model}`?"),
                true,
            )? {
                cfg.llm.model = default_model.clone();
            } else {
                cfg.llm.model = prompt_line("OpenAI model", Some(default_model.as_str()))?;
            }
            if prompt_bool("Override default OpenAI base URL?", false)? {
                cfg.providers.openai.base_url = prompt_line(
                    "OpenAI base URL",
                    Some(cfg.providers.openai.base_url.as_str()),
                )?;
            }
            if prompt_bool("Use custom OpenAI API key env var?", false)? {
                cfg.providers.openai.api_key_env = prompt_line(
                    "OpenAI API key env var",
                    Some(cfg.providers.openai.api_key_env.as_str()),
                )?;
            }
            let env_name = cfg.providers.openai.api_key_env.clone();
            let env_has_key = std::env::var(&env_name)
                .ok()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false);
            if !env_has_key {
                eprintln!(
                    "OpenAI API key not found in {env_name}. Set it before using LLM features."
                );
            }
        }
        "none" => {}
        _ => {}
    }

    Ok(())
}

fn default_embed_model<'a>(
    provider: &str,
    previous_provider: &str,
    current_model: &'a str,
) -> &'a str {
    if provider.eq_ignore_ascii_case(previous_provider) && !current_model.trim().is_empty() {
        return current_model;
    }
    match provider {
        "voyage" => "voyage-3.5",
        "openai" => "text-embedding-3-small",
        _ => "nomic-embed-text",
    }
}

fn default_llm_model(provider: &str, previous_provider: &str, current_model: &str) -> String {
    if provider.eq_ignore_ascii_case(previous_provider) && !current_model.trim().is_empty() {
        return current_model.to_string();
    }
    match provider {
        "zhipu" => "glm-5".to_string(),
        "ollama" => "llama3.1".to_string(),
        "openai" => "gpt-4o-mini".to_string(),
        _ => String::new(),
    }
}

fn prompt_line(prompt: &str, default: Option<&str>) -> anyhow::Result<String> {
    use std::io::Write as _;

    match default {
        Some(d) if !d.trim().is_empty() => eprint!("{prompt} [{d}]: "),
        _ => eprint!("{prompt}: "),
    }
    std::io::stderr().flush().ok();

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("read line from terminal")?;

    let trimmed = input.trim();
    if trimmed.is_empty() {
        if let Some(d) = default {
            return Ok(d.to_string());
        }
    }
    Ok(trimmed.to_string())
}

fn prompt_choice(prompt: &str, options: &[&str], default_idx: usize) -> anyhow::Result<usize> {
    use anyhow::anyhow;

    eprintln!("{prompt}:");
    for (idx, opt) in options.iter().enumerate() {
        if idx == default_idx {
            eprintln!("  {}. {} (default)", idx + 1, opt);
        } else {
            eprintln!("  {}. {}", idx + 1, opt);
        }
    }

    let raw = prompt_line("Choose option", None)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(default_idx.min(options.len().saturating_sub(1)));
    }
    if let Ok(n) = trimmed.parse::<usize>() {
        if n >= 1 && n <= options.len() {
            return Ok(n - 1);
        }
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(idx) = options
        .iter()
        .position(|o| o.eq_ignore_ascii_case(lower.as_str()))
    {
        return Ok(idx);
    }

    Err(anyhow!("invalid selection: {trimmed}"))
}

fn prompt_bool(prompt: &str, default: bool) -> anyhow::Result<bool> {
    let default_txt = if default { "Y/n" } else { "y/N" };
    let input = prompt_line(&format!("{prompt} ({default_txt})"), None)?;
    let v = input.trim().to_ascii_lowercase();
    if v.is_empty() {
        return Ok(default);
    }
    if matches!(v.as_str(), "y" | "yes" | "true" | "1") {
        return Ok(true);
    }
    if matches!(v.as_str(), "n" | "no" | "false" | "0") {
        return Ok(false);
    }
    Ok(default)
}

fn prompt_usize(prompt: &str, default: usize) -> anyhow::Result<usize> {
    let default_s = default.to_string();
    let input = prompt_line(prompt, Some(default_s.as_str()))?;
    if input.trim().is_empty() {
        return Ok(default);
    }
    input
        .trim()
        .parse::<usize>()
        .context("expected a positive integer")
}

fn init_tracing(verbosity: u8) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Layer};

    let default_level = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    let fmt = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_filter(env_filter);

    tracing_subscriber::registry().with(fmt).init();
}
