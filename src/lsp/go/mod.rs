use std::collections::HashMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config;
use crate::lsp::{Location, Position, Range, normalize_path};

#[derive(Debug, Error)]
pub enum GoplsError {
    #[error("gopls binary not found or not runnable: {0}")]
    Missing(String),
    #[error("gopls timed out after {0}ms")]
    Timeout(u64),
    #[error("gopls failed (exit={status}) stderr={stderr}")]
    Failed { status: String, stderr: String },
    #[error("failed to parse gopls output: {0}")]
    Parse(String),
}

#[derive(Debug, Clone)]
pub struct GoWorkspaceResolver {
    root_dir: PathBuf,
    has_go_work: bool,
    cache: HashMap<PathBuf, PathBuf>,
}

impl GoWorkspaceResolver {
    pub fn new(root_dir: &Path) -> Self {
        let has_go_work = root_dir.join("go.work").exists();
        Self {
            root_dir: root_dir.to_path_buf(),
            has_go_work,
            cache: HashMap::new(),
        }
    }

    pub fn workspace_for_file(&mut self, abs_file: &Path) -> PathBuf {
        if self.has_go_work {
            return self.root_dir.clone();
        }

        if !abs_file.starts_with(&self.root_dir) {
            return self.root_dir.clone();
        }

        let dir = abs_file
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.root_dir.clone());

        if let Some(hit) = self.cache.get(&dir) {
            return hit.clone();
        }

        let mut cur: &Path = dir.as_path();
        while cur.starts_with(&self.root_dir) {
            if cur.join("go.mod").exists() {
                let found = cur.to_path_buf();
                self.cache.insert(dir.clone(), found.clone());
                return found;
            }
            if cur == self.root_dir.as_path() {
                break;
            }
            let Some(parent) = cur.parent() else {
                break;
            };
            cur = parent;
        }

        self.cache.insert(dir.clone(), self.root_dir.clone());
        self.root_dir.clone()
    }

    pub fn workspace_root_default(&self) -> PathBuf {
        if self.root_dir.join("go.work").exists() {
            return self.root_dir.clone();
        }
        if self.root_dir.join("go.mod").exists() {
            return self.root_dir.clone();
        }
        self.root_dir.clone()
    }
}

#[derive(Debug, Clone)]
pub struct GoplsRunner {
    gopls_path: String,
    remote: Option<String>,
    timeout: Duration,
    root_dir: PathBuf,
    resolver: GoWorkspaceResolver,
    available: bool,
}

impl GoplsRunner {
    pub fn from_config(root_dir: &Path, cfg: &config::Config) -> Self {
        let go_cfg = &cfg.lsp.go;
        let remote = go_cfg.remote.trim();
        let remote = if remote.is_empty() {
            None
        } else {
            Some(remote.to_string())
        };

        let mut runner = Self {
            gopls_path: go_cfg.gopls_path.trim().to_string(),
            remote,
            timeout: Duration::from_millis(go_cfg.timeout_ms.max(50)),
            root_dir: root_dir.to_path_buf(),
            resolver: GoWorkspaceResolver::new(root_dir),
            available: false,
        };
        runner.available = runner.probe_available().is_ok();
        runner
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn set_timeout_ms(&mut self, timeout_ms: u64) {
        self.timeout = Duration::from_millis(timeout_ms.max(50));
    }

    pub fn root_dir(&self) -> &Path {
        self.root_dir.as_path()
    }

    fn probe_available(&self) -> Result<()> {
        let out = Command::new(&self.gopls_path)
            .arg("version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output();
        match out {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => Err(anyhow!(
                "gopls version exited non-zero: {}",
                o.status.code().unwrap_or(-1)
            )),
            Err(err) => Err(anyhow!("spawn failed: {err}")),
        }
    }

    fn base_cmd(&self) -> Command {
        let mut cmd = Command::new(&self.gopls_path);
        if let Some(remote) = &self.remote {
            cmd.arg(format!("-remote={remote}"));
        }
        cmd
    }

    fn run_raw(&mut self, cwd: &Path, args: &[String]) -> Result<CmdOutput> {
        let out = self.run_raw_inner(cwd, args, self.timeout, true)?;
        if self.remote.is_some() && looks_like_remote_dial_error(&out) {
            let is_auto = self
                .remote
                .as_deref()
                .is_some_and(|r| r.trim().to_ascii_lowercase().starts_with("auto"));
            if is_auto {
                // If -remote=auto is broken/misconfigured, disable it for the rest of this sx run
                // and fall back to a local gopls invocation.
                self.remote = None;
                return self.run_raw_inner(cwd, args, self.timeout, false);
            }
            return Err(GoplsError::Failed {
                status: exit_status_string(&out.status),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            }
            .into());
        }
        Ok(out)
    }

    fn run_raw_inner(
        &self,
        cwd: &Path,
        args: &[String],
        timeout: Duration,
        allow_retry: bool,
    ) -> Result<CmdOutput> {
        let mut cmd = self.base_cmd();
        cmd.current_dir(cwd);
        cmd.args(args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|err| {
            let ctx = format!(
                "spawn gopls={} (cwd={}) failed: {err}",
                self.gopls_path,
                cwd.display()
            );
            anyhow!(GoplsError::Missing(ctx))
        })?;

        let mut stdout_pipe = child.stdout.take().context("capture gopls stdout")?;
        let mut stderr_pipe = child.stderr.take().context("capture gopls stderr")?;

        let out_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout_pipe.read_to_end(&mut buf);
            buf
        });
        let err_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr_pipe.read_to_end(&mut buf);
            buf
        });

        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = child.try_wait().context("poll gopls process")? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = out_handle.join().unwrap_or_default();
                let stderr = err_handle.join().unwrap_or_default();
                let timeout_ms = timeout.as_millis() as u64;

                // `-remote=auto` may be slow on first run while the daemon starts.
                if allow_retry
                    && self
                        .remote
                        .as_deref()
                        .is_some_and(|r| r.trim().eq_ignore_ascii_case("auto"))
                {
                    let cold_timeout = Duration::from_millis(15_000);
                    if timeout < cold_timeout {
                        return self.run_raw_inner(cwd, args, cold_timeout, false);
                    }
                }

                return Err(anyhow!(GoplsError::Timeout(timeout_ms))).context(format!(
                    "gopls timed out (stdout={} bytes stderr={} bytes)",
                    stdout.len(),
                    stderr.len()
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        };

        let stdout = out_handle.join().unwrap_or_default();
        let stderr = err_handle.join().unwrap_or_default();

        Ok(CmdOutput {
            status,
            stdout,
            stderr,
        })
    }

    pub fn definition(&mut self, abs_file: &Path, line: u32, col: u32) -> Result<DefinitionResult> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec![
            "definition".to_string(),
            "-json".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];

        let out = self.run_raw(&cwd, &args)?;
        if !out.status.success() {
            return Err(GoplsError::Failed {
                status: exit_status_string(&out.status),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            }
            .into());
        }

        let parsed: GoplsDefinitionJson =
            serde_json::from_slice(&out.stdout).context("parse gopls definition -json")?;
        let abs = uri_to_path(&parsed.span.uri)?;
        let path = normalize_path(self.root_dir(), abs.as_path())?;
        Ok(DefinitionResult {
            location: Location {
                path,
                range: Range {
                    start: Position {
                        line: parsed.span.start.line as u32,
                        column: parsed.span.start.column as u32,
                    },
                    end: Position {
                        line: parsed.span.end.line as u32,
                        column: parsed.span.end.column as u32,
                    },
                },
            },
            description: parsed.description,
        })
    }

    pub fn references(
        &mut self,
        abs_file: &Path,
        line: u32,
        col: u32,
        include_decl: bool,
    ) -> Result<Vec<Location>> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let mut args = vec![
            "references".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];
        if include_decl {
            args.insert(1, "-declaration".to_string());
        }
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_location_lines(self.root_dir(), &out.stdout)
    }

    pub fn implementation(&mut self, abs_file: &Path, line: u32, col: u32) -> Result<Vec<Location>> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec![
            "implementation".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_location_lines(self.root_dir(), &out.stdout)
    }

    pub fn highlight(&mut self, abs_file: &Path, line: u32, col: u32) -> Result<Vec<Location>> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec![
            "highlight".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_location_lines(self.root_dir(), &out.stdout)
    }

    pub fn signature(&mut self, abs_file: &Path, line: u32, col: u32) -> Result<String> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec![
            "signature".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    pub fn check(&mut self, abs_file: &Path) -> Result<Vec<Diagnostic>> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec!["check".to_string(), abs_file.display().to_string()];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_diagnostics(self.root_dir(), &out.stdout)
    }

    pub fn symbols(&mut self, abs_file: &Path) -> Result<Vec<DocumentSymbol>> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec!["symbols".to_string(), abs_file.display().to_string()];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_symbols(&out.stdout)
    }

    pub fn workspace_symbol(&mut self, query: &str, module_cwds: &[PathBuf]) -> Result<Vec<WorkspaceSymbol>> {
        let mut all = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for cwd in module_cwds {
            let args = vec!["workspace_symbol".to_string(), query.to_string()];
            let out = self.run_raw(cwd, &args)?;
            if !out.status.success() {
                continue;
            }
            for item in parse_workspace_symbols(self.root_dir(), &out.stdout)? {
                let key = format!(
                    "{}:{}:{}:{}",
                    item.location.path,
                    item.location.range.start.line,
                    item.location.range.start.column,
                    item.name
                );
                if seen.insert(key) {
                    all.push(item);
                }
            }
        }
        Ok(all)
    }

    pub fn call_hierarchy(
        &mut self,
        abs_file: &Path,
        line: u32,
        col: u32,
    ) -> Result<CallHierarchyResult> {
        let cwd = self.resolver.workspace_for_file(abs_file);
        let args = vec![
            "call_hierarchy".to_string(),
            format!("{}:{line}:{col}", abs_file.display()),
        ];
        let out = self.run_raw(&cwd, &args)?;
        ensure_success(&out)?;
        parse_call_hierarchy(self.root_dir(), &out.stdout)
    }

    pub fn module_cwds_for_workspace_symbol(&mut self) -> Vec<PathBuf> {
        if self.root_dir.join("go.work").exists() || self.root_dir.join("go.mod").exists() {
            return vec![self.resolver.workspace_root_default()];
        }

        // Best-effort: find go.mod files under root and search each module.
        let mut mods = Vec::new();
        let mut builder = ignore::WalkBuilder::new(&self.root_dir);
        builder.hidden(false);
        builder.git_ignore(true);
        builder.git_exclude(true);
        builder.git_global(false);

        for entry in builder.build() {
            let Ok(entry) = entry else { continue };
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                if entry.path().file_name().and_then(|s| s.to_str()) == Some("go.mod") {
                    if let Some(parent) = entry.path().parent() {
                        mods.push(parent.to_path_buf());
                    }
                }
            }
            if mods.len() >= 32 {
                break;
            }
        }
        mods.sort();
        mods.dedup();
        if mods.is_empty() {
            vec![self.root_dir.clone()]
        } else {
            mods
        }
    }
}

#[derive(Debug)]
struct CmdOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn ensure_success(out: &CmdOutput) -> Result<()> {
    if out.status.success() {
        return Ok(());
    }
    Err(GoplsError::Failed {
        status: exit_status_string(&out.status),
        stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
    }
    .into())
}

fn looks_like_remote_dial_error(out: &CmdOutput) -> bool {
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&out.stdout));
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    let low = combined.to_ascii_lowercase();
    low.contains("dialing remote")
}

fn exit_status_string(st: &ExitStatus) -> String {
    st.code()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".to_string())
}

fn uri_to_path(uri: &str) -> Result<PathBuf> {
    if let Some(stripped) = uri.strip_prefix("file://") {
        // file:///a/b/c.go
        let s = stripped.trim_start_matches('/');
        let path = format!("/{}", s);
        return Ok(PathBuf::from(path));
    }
    Err(anyhow!("unsupported URI {}", uri))
}

fn parse_location_lines(root_dir: &Path, stdout: &[u8]) -> Result<Vec<Location>> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((abs, range)) = parse_abs_location(line)? {
            let path = normalize_path(root_dir, abs.as_path())?;
            out.push(Location { path, range });
        }
    }
    Ok(out)
}

fn parse_abs_location(line: &str) -> Result<Option<(PathBuf, Range)>> {
    static RE_WITH_END_LINE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static RE_SAME_LINE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re1 = RE_WITH_END_LINE.get_or_init(|| {
        Regex::new(r"^(?P<path>.+):(?P<sl>\d+):(?P<sc>\d+)-(?P<el>\d+):(?P<ec>\d+)")
            .expect("re")
    });
    let re2 = RE_SAME_LINE.get_or_init(|| {
        Regex::new(r"^(?P<path>.+):(?P<sl>\d+):(?P<sc>\d+)-(?P<ec>\d+)")
            .expect("re")
    });

    if let Some(caps) = re1.captures(line) {
        let path = caps.name("path").unwrap().as_str();
        let sl: u32 = caps.name("sl").unwrap().as_str().parse().unwrap_or(1);
        let sc: u32 = caps.name("sc").unwrap().as_str().parse().unwrap_or(1);
        let el: u32 = caps.name("el").unwrap().as_str().parse().unwrap_or(sl);
        let ec: u32 = caps.name("ec").unwrap().as_str().parse().unwrap_or(sc);
        return Ok(Some((
            PathBuf::from(path),
            Range {
                start: Position {
                    line: sl,
                    column: sc,
                },
                end: Position {
                    line: el,
                    column: ec,
                },
            },
        )));
    }

    if let Some(caps) = re2.captures(line) {
        let path = caps.name("path").unwrap().as_str();
        let sl: u32 = caps.name("sl").unwrap().as_str().parse().unwrap_or(1);
        let sc: u32 = caps.name("sc").unwrap().as_str().parse().unwrap_or(1);
        let ec: u32 = caps.name("ec").unwrap().as_str().parse().unwrap_or(sc);
        return Ok(Some((
            PathBuf::from(path),
            Range {
                start: Position {
                    line: sl,
                    column: sc,
                },
                end: Position {
                    line: sl,
                    column: ec,
                },
            },
        )));
    }

    Ok(None)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbol {
    pub name: String,
    pub kind: String,
    pub range: Range,
}

fn parse_symbols(stdout: &[u8]) -> Result<Vec<DocumentSymbol>> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: "foo Function 3:6-3:9"
        let mut parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 3 {
            continue;
        }
        let range_str = parts.pop().unwrap();
        let kind = parts.pop().unwrap().to_string();
        let name = parts.join(" ");

        let range = parse_line_range(range_str)
            .ok_or_else(|| GoplsError::Parse(format!("bad symbol range: {range_str}")))?;
        out.push(DocumentSymbol { name, kind, range });
    }
    Ok(out)
}

fn parse_line_range(input: &str) -> Option<Range> {
    // "3:6-3:9"
    let (start, end) = input.split_once('-')?;
    let (sl, sc) = start.split_once(':')?;
    let (el, ec) = end.split_once(':')?;
    Some(Range {
        start: Position {
            line: sl.parse().ok()?,
            column: sc.parse().ok()?,
        },
        end: Position {
            line: el.parse().ok()?,
            column: ec.parse().ok()?,
        },
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSymbol {
    pub name: String,
    pub kind: String,
    pub location: Location,
}

fn parse_workspace_symbols(root_dir: &Path, stdout: &[u8]) -> Result<Vec<WorkspaceSymbol>> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: "<abs>:9:6-9 <name> <kind>"
        let mut parts = line.split_whitespace();
        let loc_str = parts.next().unwrap_or("");
        let kind = parts.next_back().unwrap_or("").to_string();
        let name = parts.collect::<Vec<_>>().join(" ");

        let Some((abs, range)) = parse_abs_location(loc_str)? else {
            continue;
        };
        let path = normalize_path(root_dir, abs.as_path())?;
        out.push(WorkspaceSymbol {
            name,
            kind,
            location: Location { path, range },
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub location: Location,
    pub message: String,
}

fn parse_diagnostics(root_dir: &Path, stdout: &[u8]) -> Result<Vec<Diagnostic>> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // "<abs>:4:2-9: message"
        let (loc_part, msg) = line.split_once(": ").unwrap_or((line, ""));
        let Some((abs, range)) = parse_abs_location(loc_part)? else {
            continue;
        };
        let path = normalize_path(root_dir, abs.as_path())?;
        out.push(Diagnostic {
            location: Location { path, range },
            message: msg.to_string(),
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefinitionResult {
    pub location: Location,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoplsDefinitionJson {
    span: GoplsSpan,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoplsSpan {
    uri: String,
    start: GoplsPos,
    end: GoplsPos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoplsPos {
    line: i64,
    column: i64,
    offset: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallHierarchyResult {
    pub identifier: Option<CallHierarchyIdentifier>,
    pub callers: Vec<CallHierarchyLink>,
    pub callees: Vec<CallHierarchyLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallHierarchyIdentifier {
    pub name: String,
    pub kind: String,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallHierarchyLink {
    pub direction: String,
    pub callsite: Location,
    pub target: CallHierarchyTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallHierarchyTarget {
    pub name: String,
    pub kind: String,
    pub location: Location,
}

fn parse_call_hierarchy(root_dir: &Path, stdout: &[u8]) -> Result<CallHierarchyResult> {
    let text = String::from_utf8_lossy(stdout);
    let mut identifier: Option<CallHierarchyIdentifier> = None;
    let mut callers = Vec::new();
    let mut callees = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(id) = parse_identifier_line(root_dir, line)? {
            identifier = Some(id);
            continue;
        }

        if let Some(link) = parse_call_link_line(root_dir, line, "caller")? {
            callers.push(link);
            continue;
        }
        if let Some(link) = parse_call_link_line(root_dir, line, "callee")? {
            callees.push(link);
            continue;
        }
    }

    Ok(CallHierarchyResult {
        identifier,
        callers,
        callees,
    })
}

fn parse_identifier_line(root_dir: &Path, line: &str) -> Result<Option<CallHierarchyIdentifier>> {
    // "identifier: function foo in /abs/main.go:3:6-9"
    let Some(rest) = line.strip_prefix("identifier: ") else {
        return Ok(None);
    };
    let (kind_name, loc_str) = rest
        .split_once(" in ")
        .ok_or_else(|| GoplsError::Parse(format!("bad identifier line: {line}")))?;
    let (kind, name) = kind_name
        .split_once(' ')
        .unwrap_or(("symbol", kind_name));
    let Some((abs, range)) = parse_abs_location(loc_str)? else {
        return Ok(None);
    };
    let path = normalize_path(root_dir, abs.as_path())?;
    Ok(Some(CallHierarchyIdentifier {
        name: name.to_string(),
        kind: kind.to_string(),
        location: Location { path, range },
    }))
}

fn parse_call_link_line(
    root_dir: &Path,
    line: &str,
    prefix: &str,
) -> Result<Option<CallHierarchyLink>> {
    // "callee[0]: ranges 4:2-5 in /abs/main.go from/to function bar in /abs/main.go:7:6-9"
    if !line.starts_with(prefix) {
        return Ok(None);
    }
    let Some((_, rest)) = line.split_once(": ranges ") else {
        return Ok(None);
    };
    let (range_part, rest) = rest
        .split_once(" in ")
        .ok_or_else(|| GoplsError::Parse(format!("bad call link line: {line}")))?;
    let callsite_range = parse_same_line_range(range_part).ok_or_else(|| {
        GoplsError::Parse(format!("bad callsite range in call link: {range_part}"))
    })?;

    let (path_part, rest) = rest
        .split_once(" from/to ")
        .ok_or_else(|| GoplsError::Parse(format!("bad call link line: {line}")))?;
    let abs_path = PathBuf::from(path_part);
    let callsite_path = normalize_path(root_dir, abs_path.as_path())?;
    let callsite = Location {
        path: callsite_path,
        range: callsite_range,
    };

    let rest = rest.trim_start_matches("function ");
    let (target_name, target_loc) = rest
        .split_once(" in ")
        .ok_or_else(|| GoplsError::Parse(format!("bad call target in line: {line}")))?;
    let Some((abs_def, def_range)) = parse_abs_location(target_loc)? else {
        return Ok(None);
    };
    let def_path = normalize_path(root_dir, abs_def.as_path())?;

    Ok(Some(CallHierarchyLink {
        direction: prefix.to_string(),
        callsite,
        target: CallHierarchyTarget {
            name: target_name.trim().to_string(),
            kind: "function".to_string(),
            location: Location {
                path: def_path,
                range: def_range,
            },
        },
    }))
}

fn parse_same_line_range(input: &str) -> Option<Range> {
    // "4:2-5"
    let (sl, rest) = input.split_once(':')?;
    let (sc, ec) = rest.split_once('-')?;
    let line: u32 = sl.parse().ok()?;
    Some(Range {
        start: Position {
            line,
            column: sc.parse().ok()?,
        },
        end: Position {
            line,
            column: ec.parse().ok()?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn parse_abs_location_same_line() {
        let (p, r) = parse_abs_location("/tmp/a.go:7:6-9").unwrap().unwrap();
        assert_eq!(p, PathBuf::from("/tmp/a.go"));
        assert_eq!(r.start.line, 7);
        assert_eq!(r.start.column, 6);
        assert_eq!(r.end.line, 7);
        assert_eq!(r.end.column, 9);
    }

    #[test]
    fn parse_abs_location_with_end_line() {
        let (p, r) = parse_abs_location("/tmp/a.go:7:6-8:9").unwrap().unwrap();
        assert_eq!(p, PathBuf::from("/tmp/a.go"));
        assert_eq!(r.start.line, 7);
        assert_eq!(r.start.column, 6);
        assert_eq!(r.end.line, 8);
        assert_eq!(r.end.column, 9);
    }

    #[test]
    fn parse_line_range_works() {
        let r = parse_line_range("3:6-3:9").unwrap();
        assert_eq!(r.start.line, 3);
        assert_eq!(r.start.column, 6);
        assert_eq!(r.end.line, 3);
        assert_eq!(r.end.column, 9);
    }

    #[test]
    fn parse_identifier_line_works() {
        let root_dir = PathBuf::from("/tmp");
        let id = parse_identifier_line(&root_dir, "identifier: function foo in /tmp/a.go:3:6-9")
            .unwrap()
            .unwrap();
        assert_eq!(id.name, "foo");
        assert_eq!(id.kind, "function");
        assert!(id.location.path.ends_with("a.go"));
        assert_eq!(id.location.range.start.line, 3);
    }

    #[test]
    fn parse_call_link_line_callee() {
        let root_dir = PathBuf::from("/tmp");
        let link = parse_call_link_line(
            &root_dir,
            "callee[0]: ranges 4:2-5 in /tmp/a.go from/to function bar in /tmp/b.go:7:6-9",
            "callee",
        )
        .unwrap()
        .unwrap();
        assert_eq!(link.direction, "callee");
        assert_eq!(link.callsite.range.start.line, 4);
        assert_eq!(link.target.name, "bar");
        assert!(link.target.location.path.ends_with("b.go"));
        assert_eq!(link.target.location.range.start.line, 7);
    }

    #[test]
    fn workspace_resolver_prefers_go_work() {
        let dir = TempDir::new().expect("tempdir");
        fs::write(dir.path().join("go.work"), "go 1.22\n").expect("write go.work");
        fs::create_dir_all(dir.path().join("a")).expect("mkdir a");
        let file = dir.path().join("a").join("main.go");
        fs::write(&file, "package main\n").expect("write main.go");

        let mut r = GoWorkspaceResolver::new(dir.path());
        let ws = r.workspace_for_file(&file);
        assert_eq!(ws, dir.path());
    }

    #[test]
    fn workspace_resolver_finds_nearest_go_mod() {
        let dir = TempDir::new().expect("tempdir");
        fs::create_dir_all(dir.path().join("a").join("sub")).expect("mkdir a/sub");
        fs::write(dir.path().join("a").join("go.mod"), "module example.com/a\n\ngo 1.22\n")
            .expect("write go.mod");
        let file = dir.path().join("a").join("sub").join("main.go");
        fs::write(&file, "package sub\n").expect("write main.go");

        let mut r = GoWorkspaceResolver::new(dir.path());
        let ws = r.workspace_for_file(&file);
        assert_eq!(ws, dir.path().join("a"));
    }

    #[test]
    fn workspace_resolver_falls_back_to_root() {
        let dir = TempDir::new().expect("tempdir");
        let file = dir.path().join("main.go");
        fs::write(&file, "package main\n").expect("write main.go");

        let mut r = GoWorkspaceResolver::new(dir.path());
        let ws = r.workspace_for_file(&file);
        assert_eq!(ws, dir.path());
    }
}
