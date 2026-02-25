use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result, anyhow};
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;

use crate::cli::{
    CdArgs, EmbedArgs, GetArgs, GuideArgs, OpenArgs, OutputFormat, QueryArgs, SearchArgs,
    TraceArgs, VsearchArgs, LspArgs, LspCommand,
};
use crate::{config, db, lsp, root, search, semantic, trace};

pub fn search(
    conn: &Connection,
    _root_dir: &Path,
    _cfg: &config::Config,
    args: SearchArgs,
) -> Result<()> {
    let opts = search::SearchOptions {
        query: args.query,
        limit: args.limit,
        literal: args.literal,
        fts: args.fts,
        langs: args.lang,
        path_prefixes: args.path_prefix,
    };

    let results = search::search(conn, &opts)?;
    print_results(results, args.output.format())
}

pub fn embed(
    conn: &mut Connection,
    _root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    args: EmbedArgs,
) -> Result<()> {
    let stats = semantic::embed(conn, db_path, cfg, args)?;
    println!(
        "collection={} embedded_new={} embedded_kept={} pruned={} dim={} duration_ms={}",
        stats.collection,
        stats.embedded_new,
        stats.embedded_kept,
        stats.pruned,
        stats.dim,
        stats.duration_ms
    );
    Ok(())
}

pub fn vsearch(
    conn: &Connection,
    _root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    args: VsearchArgs,
) -> Result<()> {
    let results = semantic::vsearch(conn, db_path, cfg, args.clone())?;
    print_results(results, args.output.format())
}

pub fn query(
    conn: &Connection,
    _root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    args: QueryArgs,
) -> Result<()> {
    let results = semantic::query(conn, db_path, cfg, args.clone())?;
    print_results(results, args.output.format())
}

pub fn trace(
    conn: &Connection,
    root_dir: &Path,
    db_path: &Path,
    cfg: &config::Config,
    args: TraceArgs,
) -> Result<()> {
    let out = trace::run(conn, root_dir, db_path, cfg, args.clone())?;
    print_trace_response(&out, args.output.format())
}

pub fn lsp(root_dir: &Path, cfg: &config::Config, args: LspArgs) -> Result<()> {
    if !cfg.lsp.enabled || !cfg.lsp.go.enabled {
        return Err(anyhow!(
            "LSP is disabled. Enable it in .sx/config.toml under [lsp] and [lsp.go]."
        ));
    }

    let mut gopls = lsp::go::GoplsRunner::from_config(root_dir, cfg);
    if !gopls.is_available() {
        return Err(anyhow!(
            "gopls is not available (expected `{}`). Install it with: `go install golang.org/x/tools/gopls@latest`",
            cfg.lsp.go.gopls_path
        ));
    }

    match args.command {
        LspCommand::Def(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let res = gopls.definition(&abs, pos.line, pos.column)?;
            print_lsp_single_def(&res, a.output.format())?;
        }
        LspCommand::Refs(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let res = gopls.references(&abs, pos.line, pos.column, a.declaration)?;
            print_lsp_locations(&res, a.output.format())?;
        }
        LspCommand::Impl(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let res = gopls.implementation(&abs, pos.line, pos.column)?;
            print_lsp_locations(&res, a.output.format())?;
        }
        LspCommand::Calls(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let res = gopls.call_hierarchy(&abs, pos.line, pos.column)?;
            print_lsp_calls(&res, a.output.format())?;
        }
        LspCommand::Symbols(a) => {
            let abs = lsp::resolve_path(root_dir, &a.path);
            ensure_go_file(&abs)?;
            let res = gopls.symbols(&abs)?;
            print_lsp_symbols(&res, a.output.format(), &abs, root_dir)?;
        }
        LspCommand::WsSymbol(a) => {
            let cwds = gopls.module_cwds_for_workspace_symbol();
            let res = gopls.workspace_symbol(&a.query, &cwds)?;
            print_lsp_ws_symbols(&res, a.output.format())?;
        }
        LspCommand::Sig(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let sig = gopls.signature(&abs, pos.line, pos.column)?;
            print_lsp_signature(&sig, a.output.format())?;
        }
        LspCommand::Highlight(a) => {
            let pos = lsp::parse_file_position(&a.position)
                .ok_or_else(|| anyhow!("invalid position (expected path:line:column)"))?;
            let abs = lsp::resolve_path(root_dir, &pos.path);
            ensure_go_file(&abs)?;
            let res = gopls.highlight(&abs, pos.line, pos.column)?;
            print_lsp_locations(&res, a.output.format())?;
        }
        LspCommand::Check(a) => {
            let abs = lsp::resolve_path(root_dir, &a.path);
            ensure_go_file(&abs)?;
            let res = gopls.check(&abs)?;
            print_lsp_diagnostics(&res, a.output.format())?;
        }
    }

    Ok(())
}

fn ensure_go_file(path: &std::path::Path) -> Result<()> {
    let is_go = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("go"))
        .unwrap_or(false);
    if is_go {
        Ok(())
    } else {
        Err(anyhow!("only Go is supported for `sx lsp` right now (.go files)"))
    }
}

fn format_loc(loc: &lsp::Location) -> String {
    if loc.range.start.line == loc.range.end.line {
        if loc.range.start.column == loc.range.end.column {
            format!("{}:{}:{}", loc.path, loc.range.start.line, loc.range.start.column)
        } else {
            format!(
                "{}:{}:{}-{}",
                loc.path, loc.range.start.line, loc.range.start.column, loc.range.end.column
            )
        }
    } else {
        format!(
            "{}:{}:{}-{}:{}",
            loc.path,
            loc.range.start.line,
            loc.range.start.column,
            loc.range.end.line,
            loc.range.end.column
        )
    }
}

fn print_lsp_single_def(res: &lsp::go::DefinitionResult, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(res)?),
        OutputFormat::Files => println!("{}", res.location.path),
        OutputFormat::Md => println!(
            "- `{}` — {}",
            format_loc(&res.location),
            res.description
        ),
        OutputFormat::Text => println!(
            "{}\t{}",
            format_loc(&res.location),
            res.description.replace('\n', " ")
        ),
    }
    Ok(())
}

fn print_lsp_locations(locs: &[lsp::Location], format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(locs)?),
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            for l in locs {
                uniq.insert(l.path.clone());
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            for l in locs {
                println!("- `{}`", format_loc(l));
            }
        }
        OutputFormat::Text => {
            for l in locs {
                println!("{}", format_loc(l));
            }
        }
    }
    Ok(())
}

fn print_lsp_symbols(
    syms: &[lsp::go::DocumentSymbol],
    format: OutputFormat,
    abs_file: &std::path::Path,
    root_dir: &std::path::Path,
) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(syms)?),
        OutputFormat::Files => {
            let p = lsp::normalize_path(root_dir, abs_file)?;
            println!("{p}");
        }
        OutputFormat::Md => {
            for s in syms {
                println!(
                    "- `{}` {} {}:{}-{}:{}",
                    s.name,
                    s.kind,
                    s.range.start.line,
                    s.range.start.column,
                    s.range.end.line,
                    s.range.end.column
                );
            }
        }
        OutputFormat::Text => {
            for s in syms {
                println!(
                    "{}\t{}\t{}:{}-{}:{}",
                    s.name,
                    s.kind,
                    s.range.start.line,
                    s.range.start.column,
                    s.range.end.line,
                    s.range.end.column
                );
            }
        }
    }
    Ok(())
}

fn print_lsp_ws_symbols(
    syms: &[lsp::go::WorkspaceSymbol],
    format: OutputFormat,
) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(syms)?),
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            for s in syms {
                uniq.insert(s.location.path.clone());
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            for s in syms {
                println!(
                    "- `{}` {} — `{}`",
                    s.name,
                    s.kind,
                    format_loc(&s.location)
                );
            }
        }
        OutputFormat::Text => {
            for s in syms {
                println!(
                    "{}\t{}\t{}",
                    format_loc(&s.location),
                    s.kind,
                    s.name
                );
            }
        }
    }
    Ok(())
}

fn print_lsp_diagnostics(diags: &[lsp::go::Diagnostic], format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(diags)?),
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            for d in diags {
                uniq.insert(d.location.path.clone());
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            for d in diags {
                println!(
                    "- `{}` — {}",
                    format_loc(&d.location),
                    d.message.replace('\n', " ")
                );
            }
        }
        OutputFormat::Text => {
            for d in diags {
                println!(
                    "{}\t{}",
                    format_loc(&d.location),
                    d.message.replace('\n', " ")
                );
            }
        }
    }
    Ok(())
}

fn print_lsp_signature(sig: &str, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(sig)?),
        OutputFormat::Files => println!("{sig}"),
        OutputFormat::Md => println!("`{}`", sig.trim()),
        OutputFormat::Text => println!("{}", sig.trim()),
    }
    Ok(())
}

fn print_lsp_calls(res: &lsp::go::CallHierarchyResult, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(res)?),
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            if let Some(id) = &res.identifier {
                uniq.insert(id.location.path.clone());
            }
            for c in &res.callers {
                uniq.insert(c.callsite.path.clone());
                uniq.insert(c.target.location.path.clone());
            }
            for c in &res.callees {
                uniq.insert(c.callsite.path.clone());
                uniq.insert(c.target.location.path.clone());
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            if let Some(id) = &res.identifier {
                println!(
                    "- identifier: {} `{}` @ `{}`",
                    id.kind,
                    id.name,
                    format_loc(&id.location)
                );
            }
            if !res.callers.is_empty() {
                println!();
                println!("## Callers");
                for c in &res.callers {
                    println!(
                        "- `{}` -> `{}` (`{}`)",
                        format_loc(&c.callsite),
                        c.target.name,
                        format_loc(&c.target.location)
                    );
                }
            }
            if !res.callees.is_empty() {
                println!();
                println!("## Callees");
                for c in &res.callees {
                    println!(
                        "- `{}` -> `{}` (`{}`)",
                        format_loc(&c.callsite),
                        c.target.name,
                        format_loc(&c.target.location)
                    );
                }
            }
        }
        OutputFormat::Text => {
            if let Some(id) = &res.identifier {
                println!(
                    "identifier\t{}\t{}\t{}",
                    id.kind,
                    id.name,
                    format_loc(&id.location)
                );
            }
            for c in &res.callers {
                println!(
                    "caller\t{}\t{}\t{}",
                    format_loc(&c.callsite),
                    c.target.name,
                    format_loc(&c.target.location)
                );
            }
            for c in &res.callees {
                println!(
                    "callee\t{}\t{}\t{}",
                    format_loc(&c.callsite),
                    c.target.name,
                    format_loc(&c.target.location)
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum Target {
    ChunkId(String),
    Path { path: String },
    PathLine { path: String, line: i64 },
}

fn parse_target(input: &str) -> Target {
    if let Some((path, line)) = parse_path_line(input) {
        return Target::PathLine { path, line };
    }
    if looks_like_path(input) {
        return Target::Path {
            path: input.to_string(),
        };
    }
    Target::ChunkId(input.to_string())
}

fn parse_path_line(input: &str) -> Option<(String, i64)> {
    let (path, line) = input.rsplit_once(':')?;
    if path.is_empty() {
        return None;
    }
    let line: i64 = line.parse().ok()?;
    if line <= 0 {
        return None;
    }
    Some((path.to_string(), line))
}

fn looks_like_path(input: &str) -> bool {
    input.contains('/')
        || input.contains('.')
        || input.starts_with("./")
        || input.starts_with("../")
}

pub fn get(conn: &Connection, root_dir: &Path, _cfg: &config::Config, args: GetArgs) -> Result<()> {
    let target = parse_target(&args.target);

    let chunk = match target {
        Target::ChunkId(id) => {
            search::get_chunk_by_id(conn, &id)?.ok_or_else(|| anyhow!("chunk not found: {id}"))?
        }
        Target::Path { path } => {
            let Some(chunk_id) = conn
                .query_row(
                    "SELECT chunk_id FROM chunks WHERE path=?1 ORDER BY start_line ASC LIMIT 1",
                    [path.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .context("find first chunk for path")?
            else {
                return Err(anyhow!("no indexed chunks for {path}"));
            };
            search::get_chunk_by_id(conn, &chunk_id)?
                .ok_or_else(|| anyhow!("chunk not found: {chunk_id}"))?
        }
        Target::PathLine { path, line } => {
            let Some(chunk_id) = search::find_chunk_covering_line(conn, &path, line)? else {
                return Err(anyhow!("no indexed chunk covers {path}:{line}"));
            };
            search::get_chunk_by_id(conn, &chunk_id)?
                .ok_or_else(|| anyhow!("chunk not found: {chunk_id}"))?
        }
    };

    let context_lines = if args.context > 0 {
        read_context(
            root_dir,
            &chunk.path,
            chunk.start_line,
            chunk.end_line,
            args.context,
        )
        .unwrap_or_default()
    } else {
        Vec::new()
    };

    match args.output.format() {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct Out {
                chunk: search::ChunkRecord,
                context: Vec<ContextLine>,
            }
            let out = Out {
                chunk,
                context: context_lines,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text | OutputFormat::Md | OutputFormat::Files => {
            if !context_lines.is_empty() {
                for cl in &context_lines {
                    let prefix = if cl.in_chunk { ">" } else { " " };
                    println!("{prefix}{:>6} | {}", cl.line, cl.text);
                }
                println!();
            }
            print!("{}", chunk.content);
            if !chunk.content.ends_with('\n') {
                println!();
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct ContextLine {
    line: i64,
    in_chunk: bool,
    text: String,
}

fn read_context(
    root_dir: &Path,
    repo_path: &str,
    start_line: i64,
    end_line: i64,
    context: usize,
) -> Result<Vec<ContextLine>> {
    let abs = root::from_repo_path(root_dir, repo_path);
    let body = std::fs::read_to_string(&abs).with_context(|| format!("read {}", abs.display()))?;
    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }

    let start = (start_line - context as i64).max(1);
    let end = (end_line + context as i64).min(lines.len() as i64);

    let mut out = Vec::new();
    for line_no in start..=end {
        let idx = (line_no - 1) as usize;
        let txt = lines.get(idx).copied().unwrap_or("").to_string();
        out.push(ContextLine {
            line: line_no,
            in_chunk: line_no >= start_line && line_no <= end_line,
            text: txt,
        });
    }
    Ok(out)
}

pub fn open(
    conn: &Connection,
    root_dir: &Path,
    cfg: &config::Config,
    args: OpenArgs,
) -> Result<()> {
    let target = parse_target(&args.target);

    let (repo_path, line) = match target {
        Target::ChunkId(id) => {
            let chunk = search::get_chunk_by_id(conn, &id)?
                .ok_or_else(|| anyhow!("chunk not found: {id}"))?;
            (chunk.path, chunk.start_line)
        }
        Target::Path { path } => (path, 1),
        Target::PathLine { path, line } => (path, line),
    };

    if Path::new(&repo_path).is_absolute() {
        return Err(anyhow!("path must be repo-relative (got absolute path)"));
    }

    let abs_path = root::from_repo_path(root_dir, &repo_path);
    let editor = resolve_editor(&cfg.open.editor)?;
    let (program, cmd_args) = build_editor_args(&editor, &abs_path, line);

    if args.dry_run {
        let mut rendered = Vec::new();
        rendered.push(shell_escape(&program));
        rendered.extend(cmd_args.iter().map(|a| shell_escape(a)));
        println!("{}", rendered.join(" "));
        return Ok(());
    }

    let status = Command::new(&program)
        .args(&cmd_args)
        .status()
        .with_context(|| format!("spawn editor {}", program))?;

    if !status.success() {
        return Err(anyhow!("editor exited with status {}", status));
    }
    Ok(())
}

fn resolve_editor(config_editor: &str) -> Result<Vec<String>> {
    let raw = if !config_editor.trim().is_empty() {
        config_editor.trim().to_string()
    } else {
        std::env::var("EDITOR").unwrap_or_default()
    };

    let parts: Vec<String> = raw
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if parts.is_empty() {
        return Err(anyhow!(
            "no editor configured (set [open].editor or $EDITOR)"
        ));
    }
    Ok(parts)
}

fn build_editor_args(editor: &[String], file: &Path, line: i64) -> (String, Vec<String>) {
    let program = editor[0].clone();
    let mut args: Vec<String> = editor[1..].to_vec();

    let base = PathBuf::from(&program)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let file_str = file.display().to_string();
    if base == "code" || base == "cursor" {
        args.push("--goto".to_string());
        args.push(format!("{file_str}:{line}"));
        return (program, args);
    }

    if base.contains("vim") || base == "vi" {
        args.push(format!("+{line}"));
        args.push(file_str);
        return (program, args);
    }

    args.push(file_str);
    (program, args)
}

fn shell_escape(input: &str) -> String {
    if input.is_empty() {
        return "''".to_string();
    }

    let safe = input.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':' | '@' | '+' | '=')
    });
    if safe {
        return input.to_string();
    }
    let escaped = input.replace('\'', "'\\''");
    format!("'{escaped}'")
}

pub fn guide(
    conn: &Connection,
    _root_dir: &Path,
    _cfg: &config::Config,
    args: GuideArgs,
) -> Result<()> {
    let opts = search::SearchOptions {
        query: args.query,
        limit: args.limit_results,
        literal: true,
        fts: false,
        langs: Vec::new(),
        path_prefixes: Vec::new(),
    };
    let results = search::search(conn, &opts)?;

    let mut best_by_file: HashMap<String, search::SearchResult> = HashMap::new();
    for r in results {
        best_by_file
            .entry(r.path.clone())
            .and_modify(|cur| {
                if r.score > cur.score {
                    *cur = r.clone();
                }
            })
            .or_insert(r);
    }

    let mut files: Vec<search::SearchResult> = best_by_file.into_values().collect();
    files.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    files.truncate(args.limit_files);

    match args.output.format() {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct GuideHit {
                path: String,
                chunk_id: String,
                start_line: i64,
                end_line: i64,
                kind: String,
                symbol: Option<String>,
                score: f64,
                snippet: String,
            }
            let out: Vec<GuideHit> = files
                .into_iter()
                .map(|r| GuideHit {
                    path: r.path,
                    chunk_id: r.chunk_id,
                    start_line: r.start_line,
                    end_line: r.end_line,
                    kind: r.kind,
                    symbol: r.symbol,
                    score: r.score,
                    snippet: r.snippet,
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Md => {
            for r in files {
                let sym = r.symbol.as_deref().unwrap_or("-");
                println!(
                    "- `{}`:{} ({}, {}) — {}",
                    r.path, r.start_line, r.kind, sym, r.snippet
                );
            }
        }
        OutputFormat::Text | OutputFormat::Files => {
            for r in files {
                let sym = r.symbol.as_deref().unwrap_or("-");
                println!(
                    "{}:{}\t{}\t{}\t{}",
                    r.path, r.start_line, r.kind, sym, r.snippet
                );
            }
        }
    }

    Ok(())
}

pub fn cd(conn: &Connection, root_dir: &Path, _cfg: &config::Config, args: CdArgs) -> Result<()> {
    let opts = search::SearchOptions {
        query: args.query,
        limit: args.limit_results,
        literal: true,
        fts: false,
        langs: Vec::new(),
        path_prefixes: Vec::new(),
    };
    let results = search::search(conn, &opts)?;
    if results.is_empty() {
        return Err(anyhow!("no results"));
    }

    let mut dir_weight: HashMap<String, f64> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let weight = 1.0 / (i as f64 + 1.0);
        let dir = match r.path.rsplit_once('/') {
            Some((d, _)) if !d.is_empty() => d.to_string(),
            _ => ".".to_string(),
        };
        *dir_weight.entry(dir).or_insert(0.0) += weight;
    }

    let now = db::now_unix();
    let mut best: Option<(String, f64)> = None;

    for (dir, w) in &dir_weight {
        let (rank, last_accessed) = conn
            .query_row(
                "SELECT rank, last_accessed FROM dir_frecency WHERE path=?1",
                [dir.as_str()],
                |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()
            .context("query dir_frecency")?
            .unwrap_or((0.0, 0));

        let age_days = ((now - last_accessed).max(0) as f64) / 86_400.0;
        let frecency_score = rank * (-age_days / 14.0).exp();
        let final_score = *w + 0.5 * frecency_score;

        match &mut best {
            None => best = Some((dir.clone(), final_score)),
            Some((best_dir, best_score)) => {
                if final_score > *best_score || (final_score == *best_score && dir < best_dir) {
                    *best_dir = dir.clone();
                    *best_score = final_score;
                }
            }
        }
    }

    let Some((chosen, _score)) = best else {
        return Err(anyhow!("no candidate directories"));
    };

    conn.execute(
        r#"
INSERT INTO dir_frecency(path, rank, last_accessed)
VALUES(?1, 1.0, ?2)
ON CONFLICT(path) DO UPDATE SET
  rank = rank + 1.0,
  last_accessed = excluded.last_accessed
"#,
        rusqlite::params![chosen, now],
    )
    .context("update dir_frecency")?;

    if args.relative {
        println!("{chosen}");
        return Ok(());
    }

    let abs = if chosen == "." {
        root_dir.to_path_buf()
    } else {
        root::from_repo_path(root_dir, &chosen)
    };
    println!("{}", abs.display());
    Ok(())
}

fn print_results(results: Vec<search::SearchResult>, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            for r in results {
                uniq.insert(r.path);
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            for r in results {
                let sym = r.symbol.as_deref().unwrap_or("-");
                println!(
                    "- `{}`:{} ({}, {}) — {}",
                    r.path, r.start_line, r.kind, sym, r.snippet
                );
            }
        }
        OutputFormat::Text => {
            for r in results {
                let sym = r.symbol.as_deref().unwrap_or("-");
                println!(
                    "{:.4}\t{}:{}-{}\t{}\t{}\t{}",
                    r.score, r.path, r.start_line, r.end_line, r.kind, sym, r.snippet
                );
            }
        }
    }
    Ok(())
}

fn print_trace_response(out: &trace::types::TraceResponse, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(out)?);
        }
        OutputFormat::Files => {
            let mut uniq = BTreeSet::new();
            for c in &out.fast.citations {
                uniq.insert(c.path.clone());
            }
            if let Some(deep) = &out.deep {
                for c in &deep.citations {
                    uniq.insert(c.path.clone());
                }
            }
            for p in uniq {
                println!("{p}");
            }
        }
        OutputFormat::Md => {
            println!("# Trace");
            println!();
            println!("Query: `{}`", out.query);
            println!();
            println!("## Fast");
            println!("source: `{}`", out.fast.summary_source);
            if let Some(model) = &out.fast.summary_model {
                println!("model: `{}`", model);
            }
            if let Some(err) = &out.fast.summary_error {
                println!("note: `{}`", err);
            }
            println!("{}", out.fast.summary);
            for (i, t) in out.fast.traces.iter().enumerate() {
                println!(
                    "{}. `{}`:{} `{}` score={:.3}",
                    i + 1,
                    t.root_path,
                    t.root_line,
                    t.root_symbol,
                    t.score
                );
                for step in &t.steps {
                    let target = step
                        .to_symbol
                        .clone()
                        .or(step.target_name.clone())
                        .unwrap_or_else(|| "-".to_string());
                    println!(
                        "   - {} -> {} [{}] {}:{}",
                        step.from_symbol, target, step.edge_kind, step.path, step.line
                    );
                }
            }
            println!();
            println!("## Deep ({})", out.deep_status);
            if let Some(deep) = &out.deep {
                println!("source: `{}`", deep.summary_source);
                if let Some(model) = &deep.summary_model {
                    println!("model: `{}`", model);
                }
                if let Some(err) = &deep.summary_error {
                    println!("note: `{}`", err);
                }
                println!("{}", deep.summary);
                for (i, t) in deep.traces.iter().enumerate() {
                    println!(
                        "{}. `{}`:{} `{}` score={:.3}",
                        i + 1,
                        t.root_path,
                        t.root_line,
                        t.root_symbol,
                        t.score
                    );
                    for step in &t.steps {
                        let target = step
                            .to_symbol
                            .clone()
                            .or(step.target_name.clone())
                            .unwrap_or_else(|| "-".to_string());
                        println!(
                            "   - {} -> {} [{}] {}:{}",
                            step.from_symbol, target, step.edge_kind, step.path, step.line
                        );
                    }
                }
            } else {
                println!("(disabled)");
            }
        }
        OutputFormat::Text => {
            println!("query={}", out.query);
            println!("fast_summary={}", out.fast.summary.replace('\n', " "));
            println!("fast_summary_source={}", out.fast.summary_source);
            if let Some(model) = &out.fast.summary_model {
                println!("fast_summary_model={model}");
            }
            if let Some(err) = &out.fast.summary_error {
                println!("fast_summary_error={}", err.replace('\n', " "));
            }
            for (i, t) in out.fast.traces.iter().enumerate() {
                println!(
                    "FAST\t{}\t{:.3}\t{}:{}\t{}",
                    i + 1,
                    t.score,
                    t.root_path,
                    t.root_line,
                    t.root_symbol
                );
                for step in &t.steps {
                    let target = step
                        .to_symbol
                        .clone()
                        .or(step.target_name.clone())
                        .unwrap_or_else(|| "-".to_string());
                    println!(
                        "  {}\t{}\t{}\t{}:{}\t{}",
                        step.edge_kind,
                        step.confidence,
                        step.from_symbol,
                        step.path,
                        step.line,
                        target
                    );
                }
            }
            println!("deep_status={}", out.deep_status);
            if let Some(deep) = &out.deep {
                println!("deep_summary={}", deep.summary.replace('\n', " "));
                println!("deep_summary_source={}", deep.summary_source);
                if let Some(model) = &deep.summary_model {
                    println!("deep_summary_model={model}");
                }
                if let Some(err) = &deep.summary_error {
                    println!("deep_summary_error={}", err.replace('\n', " "));
                }
                for (i, t) in deep.traces.iter().enumerate() {
                    println!(
                        "DEEP\t{}\t{:.3}\t{}:{}\t{}",
                        i + 1,
                        t.score,
                        t.root_path,
                        t.root_line,
                        t.root_symbol
                    );
                    for step in &t.steps {
                        let target = step
                            .to_symbol
                            .clone()
                            .or(step.target_name.clone())
                            .unwrap_or_else(|| "-".to_string());
                        println!(
                            "  {}\t{}\t{}\t{}:{}\t{}",
                            step.edge_kind,
                            step.confidence,
                            step.from_symbol,
                            step.path,
                            step.line,
                            target
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
