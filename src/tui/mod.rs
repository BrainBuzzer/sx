use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use anyhow::{Context as _, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use rusqlite::Connection;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language as TsLanguage, Parser, Query, QueryCursor};

use crate::cli::{OpenArgs, QueryArgs};
use crate::index::scan::Language;
use crate::{actions, config, db, search, semantic, trace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Input,
    Results,
    Preview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryMode {
    Trace,
    Flat,
}

#[derive(Debug)]
struct DeepStageUpdate {
    query: String,
    stage: trace::types::TraceStageResult,
    status: String,
}

#[derive(Debug, Clone)]
struct TraceNode {
    trace_idx: usize,
    step_idx: Option<usize>,
    depth: usize,
    label: String,
    path: String,
    line: i64,
    evidence: Option<String>,
}

#[derive(Debug, Clone)]
struct TraceRootGroup {
    root_symbol_id: String,
    root_symbol: String,
    root_path: String,
    root_line: i64,
    best_score: f64,
    paths: Vec<trace::types::TracePath>,
}

#[derive(Debug, Clone)]
struct FlowStepNode {
    edge_kind: String,
    from_symbol: String,
    target: String,
    path: String,
    line: i64,
    confidence: f64,
    evidence: String,
    best_score: f64,
    count: usize,
    children: Vec<FlowStepNode>,
}

fn group_trace_paths(traces: Vec<trace::types::TracePath>) -> Vec<TraceRootGroup> {
    let mut groups: HashMap<String, TraceRootGroup> = HashMap::new();

    for trace_path in traces {
        let key = trace_path.root_symbol_id.clone();
        let entry = groups.entry(key.clone()).or_insert_with(|| TraceRootGroup {
            root_symbol_id: key,
            root_symbol: trace_path.root_symbol.clone(),
            root_path: trace_path.root_path.clone(),
            root_line: trace_path.root_line,
            best_score: trace_path.score,
            paths: Vec::new(),
        });

        entry.best_score = entry.best_score.max(trace_path.score);
        entry.paths.push(trace_path);
    }

    let mut out: Vec<TraceRootGroup> = groups.into_values().collect();
    out.sort_by(|a, b| {
        b.best_score
            .total_cmp(&a.best_score)
            .then_with(|| a.root_path.cmp(&b.root_path))
            .then_with(|| a.root_line.cmp(&b.root_line))
    });
    out
}

fn step_target(step: &trace::types::TraceStep) -> String {
    step.to_symbol
        .clone()
        .or(step.target_name.clone())
        .unwrap_or_else(|| "-".to_string())
}

fn insert_flow_steps(
    tree: &mut Vec<FlowStepNode>,
    steps: &[trace::types::TraceStep],
    path_score: f64,
) {
    let mut cur = tree;
    for step in steps {
        let target = step_target(step);
        if let Some(existing_idx) = cur.iter().position(|n| {
            n.edge_kind == step.edge_kind
                && n.from_symbol == step.from_symbol
                && n.target == target
                && n.path == step.path
                && n.line == step.line
        }) {
            let existing = &mut cur[existing_idx];
            existing.count = existing.count.saturating_add(1);
            existing.best_score = existing.best_score.max(path_score);
            existing.confidence = existing.confidence.max(step.confidence);
            if existing.evidence.trim().is_empty() && !step.evidence.trim().is_empty() {
                existing.evidence = step.evidence.clone();
            }
            cur = &mut existing.children;
            continue;
        }

        cur.push(FlowStepNode {
            edge_kind: step.edge_kind.clone(),
            from_symbol: step.from_symbol.clone(),
            target,
            path: step.path.clone(),
            line: step.line,
            confidence: step.confidence,
            evidence: step.evidence.clone(),
            best_score: path_score,
            count: 1,
            children: Vec::new(),
        });
        let last = cur.len().saturating_sub(1);
        cur = &mut cur[last].children;
    }
}

fn sort_flow_tree(nodes: &mut Vec<FlowStepNode>) {
    nodes.sort_by(|a, b| {
        b.best_score
            .total_cmp(&a.best_score)
            .then_with(|| b.confidence.total_cmp(&a.confidence))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.target.cmp(&b.target))
    });
    for n in nodes.iter_mut() {
        sort_flow_tree(&mut n.children);
    }
}

fn push_flow_tree_nodes(
    out: &mut Vec<TraceNode>,
    group_idx: usize,
    nodes: &[FlowStepNode],
    prefix: &str,
) {
    for (idx, node) in nodes.iter().enumerate() {
        let last = idx + 1 == nodes.len();
        let branch = if last { "└─" } else { "├─" };
        let next_prefix = if last {
            format!("{prefix}   ")
        } else {
            format!("{prefix}│  ")
        };

        let count_suffix = if node.count > 1 {
            format!(" x{}", node.count)
        } else {
            String::new()
        };
        out.push(TraceNode {
            trace_idx: group_idx,
            step_idx: Some(0),
            depth: 1,
            label: format!(
                "{prefix}{branch} [{}] {} -> {}  {}:{}{}",
                node.edge_kind, node.from_symbol, node.target, node.path, node.line, count_suffix
            ),
            path: node.path.clone(),
            line: node.line,
            evidence: if node.evidence.trim().is_empty() {
                None
            } else {
                Some(node.evidence.clone())
            },
        });

        if !node.children.is_empty() {
            push_flow_tree_nodes(out, group_idx, &node.children, &next_prefix);
        }
    }
}

pub fn run(conn: &Connection, root_dir: &Path, db_path: &Path, cfg: &config::Config) -> Result<()> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;

    let mut app = App::new();
    let res = app.run_loop(&mut terminal, conn, root_dir, db_path, cfg);

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    res
}

struct App {
    focus: Focus,
    mode: QueryMode,
    repo_root: Option<PathBuf>,
    input: String,
    status: String,

    results: Vec<search::SearchResult>,
    results_state: ListState,

    preview: Option<search::ChunkRecord>,
    trace_summary_text: Text<'static>,
    preview_text: Text<'static>,
    preview_scroll: u16,
    preview_highlighter: PreviewHighlighter,
    trace_response: Option<trace::types::TraceResponse>,
    trace_nodes: Vec<TraceNode>,
    trace_expanded: Vec<bool>,
    trace_group_keys: Vec<String>,
    show_all_roots: bool,
    deep_rx: Option<Receiver<DeepStageUpdate>>,
    show_diagnostics: bool,
    cfg_snapshot: Option<config::Config>,
}

impl App {
    fn new() -> Self {
        let mut results_state = ListState::default();
        results_state.select(Some(0));

        let mode = QueryMode::Trace;
        let status =
            "Mode: Trace · Enter search · Ctrl+T/F2 toggle · a roots · F1/Ctrl+G diagnostics · ←/→ collapse/expand · ↑/↓ select · Tab focus · o open · PgUp/PgDn scroll · q quit"
                .to_string();
        let trace_summary_text = Text::from("Run a query to build a flow tree.");
        let preview_text = Text::from("No selection.");

        Self {
            focus: Focus::Input,
            mode,
            repo_root: None,
            input: String::new(),
            status,
            results: Vec::new(),
            results_state,
            preview: None,
            trace_summary_text,
            preview_text,
            preview_scroll: 0,
            preview_highlighter: PreviewHighlighter::new(),
            trace_response: None,
            trace_nodes: Vec::new(),
            trace_expanded: Vec::new(),
            trace_group_keys: Vec::new(),
            show_all_roots: false,
            deep_rx: None,
            show_diagnostics: false,
            cfg_snapshot: None,
        }
    }

    fn run_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
        conn: &Connection,
        root_dir: &Path,
        db_path: &Path,
        cfg: &config::Config,
    ) -> Result<()> {
        self.repo_root = Some(root_dir.to_path_buf());
        self.cfg_snapshot = Some(cfg.clone());
        loop {
            self.poll_deep_updates(conn);
            terminal.draw(|f| self.render(f)).context("draw frame")?;

            if event::poll(Duration::from_millis(50)).context("poll events")? {
                match event::read().context("read event")? {
                    Event::Key(key) => {
                        if self.handle_key(key, conn, root_dir, db_path, cfg, terminal)? {
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn poll_deep_updates(&mut self, conn: &Connection) {
        if let Some(rx) = self.deep_rx.take() {
            let mut got_update = false;
            while let Ok(msg) = rx.try_recv() {
                got_update = true;
                if self.input.trim() == msg.query {
                    if let Some(resp) = &mut self.trace_response {
                        resp.deep = Some(msg.stage);
                        resp.deep_status = msg.status.clone();
                    }
                    if self.mode == QueryMode::Trace {
                        self.rebuild_trace_nodes();
                        self.preview_scroll = 0;
                        let _ = self.refresh_preview(conn);
                    }
                    self.status = format!("{}", {
                        let roots = self.trace_root_count();
                        if roots == 0 {
                            format!("0 roots for \"{}\" (deep ready)", msg.query)
                        } else if self.show_all_roots || roots == 1 {
                            format!("{roots} roots for \"{}\" (deep ready)", msg.query)
                        } else {
                            format!("best root for \"{}\" (1 of {roots}; deep ready)", msg.query)
                        }
                    });
                }
            }
            if !got_update {
                self.deep_rx = Some(rx);
            }
        }
    }

    fn handle_key(
        &mut self,
        key: KeyEvent,
        conn: &Connection,
        root_dir: &Path,
        db_path: &Path,
        cfg: &config::Config,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<bool> {
        if self.show_diagnostics {
            match (key.code, key.modifiers) {
                (KeyCode::Char('q'), _) => return Ok(true),
                (KeyCode::Esc, _) | (KeyCode::F(1), _) => {
                    self.show_diagnostics = false;
                    return Ok(false);
                }
                (KeyCode::Char('g'), m) if m.contains(KeyModifiers::CONTROL) => {
                    self.show_diagnostics = false;
                    return Ok(false);
                }
                (KeyCode::Char('?'), _) if self.focus != Focus::Input => {
                    self.show_diagnostics = false;
                    return Ok(false);
                }
                _ => return Ok(false),
            }
        }

        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => return Ok(true),
            (KeyCode::F(1), _) => {
                self.show_diagnostics = !self.show_diagnostics;
                return Ok(false);
            }
            (KeyCode::Char('g'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.show_diagnostics = !self.show_diagnostics;
                return Ok(false);
            }
            (KeyCode::Char('?'), _) if self.focus != Focus::Input => {
                self.show_diagnostics = !self.show_diagnostics;
                return Ok(false);
            }
            (KeyCode::Char('t'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.toggle_mode();
                return Ok(false);
            }
            (KeyCode::F(2), _) => {
                self.toggle_mode();
                return Ok(false);
            }
            (KeyCode::Tab, _) => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Results,
                    Focus::Results => Focus::Preview,
                    Focus::Preview => Focus::Input,
                };
                return Ok(false);
            }
            _ => {}
        }

        // fzf-like navigation: allow moving the result selection even when focus is on the input.
        if self.focus == Focus::Input {
            match (key.code, key.modifiers) {
                (KeyCode::Down, _) | (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                    self.select_next(conn)?;
                    return Ok(false);
                }
                (KeyCode::Up, _) | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                    self.select_prev(conn)?;
                    return Ok(false);
                }
                _ => {}
            }
        }

        match self.focus {
            Focus::Input => self.handle_input_key(key, conn, db_path, cfg)?,
            Focus::Results => {
                if self.handle_results_key(key, conn, root_dir, cfg, terminal)? {
                    return Ok(true);
                }
            }
            Focus::Preview => self.handle_preview_key(key)?,
        }

        Ok(false)
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            QueryMode::Trace => QueryMode::Flat,
            QueryMode::Flat => QueryMode::Trace,
        };

        self.results.clear();
        self.trace_nodes.clear();
        self.trace_response = None;
        self.deep_rx = None;
        self.show_diagnostics = false;
        self.preview = None;
        self.preview_scroll = 0;

        self.trace_summary_text = match self.mode {
            QueryMode::Trace => Text::from("Run a query to build a flow tree."),
            QueryMode::Flat => Text::from(""),
        };
        self.preview_text = Text::from("No selection.");
        self.results_state.select(Some(0));

        self.status = self.mode_hint_status();
    }

    fn mode_label(&self) -> &'static str {
        match self.mode {
            QueryMode::Trace => "Trace",
            QueryMode::Flat => "Flat",
        }
    }

    fn mode_hint_status(&self) -> String {
        match self.mode {
            QueryMode::Trace => {
                "Mode: Trace (Ctrl+T/F2 toggle · F1/Ctrl+G diagnostics)".to_string()
            }
            QueryMode::Flat => "Mode: Flat (Ctrl+T/F2 toggle · F1/Ctrl+G diagnostics)".to_string(),
        }
    }

    fn handle_input_key(
        &mut self,
        key: KeyEvent,
        conn: &Connection,
        db_path: &Path,
        cfg: &config::Config,
    ) -> Result<()> {
        match (key.code, key.modifiers) {
            (KeyCode::Enter, _) => self.run_search(conn, db_path, cfg),
            (KeyCode::Backspace, _) => {
                self.input.pop();
                Ok(())
            }
            (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.input.clear();
                Ok(())
            }
            (KeyCode::Char(c), m)
                if !m.contains(KeyModifiers::CONTROL) && !m.contains(KeyModifiers::ALT) =>
            {
                self.input.push(c);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn handle_results_key(
        &mut self,
        key: KeyEvent,
        conn: &Connection,
        root_dir: &Path,
        cfg: &config::Config,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<bool> {
        match (key.code, key.modifiers) {
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
                self.select_next(conn)?;
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
                self.select_prev(conn)?;
            }
            (KeyCode::Char('a'), _) if self.mode == QueryMode::Trace => {
                self.show_all_roots = !self.show_all_roots;
                self.rebuild_trace_nodes();
                self.preview_scroll = 0;
                self.refresh_preview(conn)?;
            }
            (KeyCode::Right, _) | (KeyCode::Char('l'), _) if self.mode == QueryMode::Trace => {
                self.expand_selected_trace(true);
                self.refresh_preview(conn)?;
            }
            (KeyCode::Left, _) | (KeyCode::Char('h'), _) if self.mode == QueryMode::Trace => {
                self.expand_selected_trace(false);
                self.refresh_preview(conn)?;
            }
            (KeyCode::Enter, _) | (KeyCode::Char('o'), _) => {
                let target = if self.mode == QueryMode::Trace {
                    self.selected_trace_node()
                        .map(|n| format!("{}:{}", n.path, n.line))
                } else {
                    let sel = self.results_state.selected().unwrap_or(0);
                    self.results.get(sel).map(|r| r.chunk_id.clone())
                };

                if let Some(target) = target {
                    // Restore terminal before spawning the editor; then exit TUI.
                    disable_raw_mode().ok();
                    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
                    terminal.show_cursor().ok();

                    let open_args = OpenArgs {
                        target,
                        dry_run: false,
                    };
                    actions::open(conn, root_dir, cfg, open_args)?;
                    return Ok(true);
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_preview_key(&mut self, key: KeyEvent) -> Result<()> {
        match (key.code, key.modifiers) {
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
                self.preview_scroll = self.preview_scroll.saturating_add(1);
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
                self.preview_scroll = self.preview_scroll.saturating_sub(1);
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                self.preview_scroll = self.preview_scroll.saturating_add(10);
            }
            (KeyCode::PageUp, _) | (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.preview_scroll = self.preview_scroll.saturating_sub(10);
            }
            _ => {}
        }
        Ok(())
    }

    fn run_search(
        &mut self,
        conn: &Connection,
        db_path: &Path,
        cfg: &config::Config,
    ) -> Result<()> {
        let q = self.input.trim().to_string();
        self.deep_rx = None;

        if q.is_empty() {
            self.status = "Empty query.".to_string();
            self.results.clear();
            self.trace_nodes.clear();
            self.preview = None;
            self.trace_summary_text = Text::from("Run a query to build a flow tree.");
            self.preview_text = Text::from("No selection.");
            self.trace_response = None;
            self.deep_rx = None;
            return Ok(());
        }

        match self.mode {
            QueryMode::Flat => {
                let args = QueryArgs {
                    query: q.clone(),
                    limit: 50,
                    bm25_limit: 200,
                    vec_limit: 200,
                    deep: true,
                    lang: Vec::new(),
                    path_prefix: Vec::new(),
                    output: crate::cli::OutputArgs::default(),
                };

                let results = semantic::query(conn, db_path, cfg, args)?;
                self.trace_response = None;
                self.trace_nodes.clear();
                self.status = format!("{} results for \"{}\" (flat+deep)", results.len(), q);
                self.results = results;
                self.results_state.select(Some(0));
                self.preview_scroll = 0;
                self.refresh_preview(conn)?;
            }
            QueryMode::Trace => {
                let (fast, fast_status) =
                    trace::run_fast_stage(conn, db_path, cfg, &q, 30, 4, &[], &[])?;
                self.trace_response = Some(trace::types::TraceResponse {
                    query: q.clone(),
                    fast,
                    deep: None,
                    deep_status: if fast_status == "timeout" {
                        "timeout".to_string()
                    } else {
                        "refining".to_string()
                    },
                });
                self.rebuild_trace_nodes();
                self.results_state.select(Some(0));
                self.preview_scroll = 0;
                self.refresh_preview(conn)?;
                self.status = {
                    let roots = self.trace_root_count();
                    if roots == 0 {
                        format!("0 roots for \"{}\" (fast ready; deep refining...)", q)
                    } else if self.show_all_roots || roots == 1 {
                        format!("{roots} roots for \"{}\" (fast ready; deep refining...)", q)
                    } else {
                        format!(
                            "best root for \"{}\" (1 of {roots}; fast ready; deep refining...)",
                            q
                        )
                    }
                };

                let (tx, rx) = mpsc::channel::<DeepStageUpdate>();
                self.deep_rx = Some(rx);
                let db_path = db_path.to_path_buf();
                let cfg = cfg.clone();
                let query = q.clone();
                std::thread::spawn(move || {
                    let conn = match db::open(&db_path) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    if db::migrate(&conn).is_err() {
                        return;
                    }
                    if let Ok((stage, status)) =
                        trace::run_deep_stage(&conn, &db_path, &cfg, &query, 30, 8, &[], &[])
                    {
                        let _ = tx.send(DeepStageUpdate {
                            query,
                            stage,
                            status,
                        });
                    }
                });
            }
        }
        Ok(())
    }

    fn select_next(&mut self, conn: &Connection) -> Result<()> {
        let len = self.current_list_len();
        if len == 0 {
            return Ok(());
        }
        let cur = self.results_state.selected().unwrap_or(0);
        let next = (cur + 1).min(len.saturating_sub(1));
        self.results_state.select(Some(next));
        self.preview_scroll = 0;
        self.refresh_preview(conn)?;
        Ok(())
    }

    fn select_prev(&mut self, conn: &Connection) -> Result<()> {
        if self.current_list_len() == 0 {
            return Ok(());
        }
        let cur = self.results_state.selected().unwrap_or(0);
        let prev = cur.saturating_sub(1);
        self.results_state.select(Some(prev));
        self.preview_scroll = 0;
        self.refresh_preview(conn)?;
        Ok(())
    }

    fn refresh_preview(&mut self, conn: &Connection) -> Result<()> {
        self.refresh_preview_inner(Some(conn))
    }

    fn refresh_preview_inner(&mut self, conn: Option<&Connection>) -> Result<()> {
        if self.mode == QueryMode::Trace {
            self.preview = None;
            let Some(stage) = self.displayed_trace_stage() else {
                self.trace_summary_text = Text::from("No trace data.");
                self.preview_text = Text::from("No trace selection.");
                return Ok(());
            };

            let selected = self.selected_trace_node().cloned();
            let mut summary_lines = Vec::new();
            summary_lines.push(Line::from(Span::styled(
                stage.summary.clone(),
                Style::default().fg(Color::LightCyan),
            )));
            summary_lines.push(Line::from(""));
            let root_count = {
                let mut roots: HashSet<&str> = HashSet::new();
                for t in &stage.traces {
                    roots.insert(t.root_symbol_id.as_str());
                }
                roots.len()
            };
            let best_root_paths = stage
                .traces
                .iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .map(|best| {
                    stage
                        .traces
                        .iter()
                        .filter(|t| t.root_symbol_id == best.root_symbol_id)
                        .count()
                })
                .unwrap_or(0);
            let shown_paths = if self.show_all_roots {
                stage.traces.len()
            } else {
                best_root_paths
            };
            let view = if self.show_all_roots { "all" } else { "best" };
            summary_lines.push(Line::from(format!(
                "view: {view} · roots: {root_count} · paths: {shown_paths}/{} · deep: {}",
                stage.traces.len(),
                self.trace_response
                    .as_ref()
                    .map(|r| r.deep_status.as_str())
                    .unwrap_or("n/a")
            )));
            summary_lines.push(Line::from(format!(
                "summary_source: {}",
                stage.summary_source
            )));
            if let Some(model) = &stage.summary_model {
                summary_lines.push(Line::from(format!("summary_model: {model}")));
            }
            if let Some(err) = &stage.summary_error {
                summary_lines.push(Line::from(Span::styled(
                    format!("summary_note: {err}"),
                    Style::default().fg(Color::LightRed),
                )));
            }
            if let Some(node) = &selected {
                summary_lines.push(Line::from(format!("selected: {}", node.label.trim())));
            }
            self.trace_summary_text = Text::from(summary_lines);
            self.preview_text = match selected {
                Some(node) => self.build_trace_code_preview(node),
                None => Text::from("No selection."),
            };
            return Ok(());
        }

        let Some(conn) = conn else {
            return Ok(());
        };
        let sel = self.results_state.selected().unwrap_or(0);
        let Some(r) = self.results.get(sel) else {
            self.preview = None;
            self.preview_text = Text::from("No selection.");
            return Ok(());
        };
        self.preview = search::get_chunk_by_id(conn, &r.chunk_id)?;
        self.preview_text = match &self.preview {
            Some(chunk) => self
                .preview_highlighter
                .highlight_for_path(&chunk.path, &chunk.content),
            None => Text::from("No selection."),
        };
        Ok(())
    }

    fn build_trace_code_preview(&self, node: TraceNode) -> Text<'static> {
        let mut lines = Vec::new();
        lines.push(Line::from(format!("{}:{}", node.path, node.line)));
        if let Some(ev) = node.evidence {
            if !ev.trim().is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("evidence: {}", ev),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
        lines.push(Line::from(""));

        let Some(repo_root) = &self.repo_root else {
            lines.push(Line::from("(repo root unavailable in current session)"));
            return Text::from(lines);
        };
        let abs = crate::root::from_repo_path(repo_root, &node.path);
        let body = match std::fs::read_to_string(&abs) {
            Ok(b) => b,
            Err(_) => {
                lines.push(Line::from(format!("Unable to read {}", abs.display())));
                return Text::from(lines);
            }
        };

        let file_lines: Vec<&str> = body.lines().collect();
        if file_lines.is_empty() {
            lines.push(Line::from("(empty file)"));
            return Text::from(lines);
        }

        let target = node.line.max(1).min(file_lines.len() as i64);
        let start = (target - 6).max(1);
        let end = (target + 6).min(file_lines.len() as i64);
        let start_idx = (start - 1) as usize;
        let end_idx = end as usize;
        let snippet = file_lines
            .get(start_idx..end_idx)
            .unwrap_or_default()
            .join("\n");
        let highlighted = self
            .preview_highlighter
            .highlight_for_path(&node.path, &snippet);

        let mut highlighted_lines = highlighted.lines.into_iter();
        let prefix_style = Style::default().fg(Color::DarkGray);
        for ln in start..=end {
            let code_line = highlighted_lines.next().unwrap_or_else(|| Line::from(""));
            let marker_style = if ln == target {
                Style::default().fg(Color::Yellow)
            } else {
                prefix_style
            };
            let mut spans = Vec::with_capacity(code_line.spans.len() + 2);
            spans.push(Span::styled(
                if ln == target { ">" } else { " " }.to_string(),
                marker_style,
            ));
            spans.push(Span::styled(format!("{ln:>6} | "), prefix_style));
            spans.extend(code_line.spans);
            lines.push(Line {
                style: code_line.style,
                alignment: code_line.alignment,
                spans,
            });
        }
        Text::from(lines)
    }

    fn displayed_trace_stage(&self) -> Option<&trace::types::TraceStageResult> {
        let resp = self.trace_response.as_ref()?;
        if let Some(deep) = &resp.deep {
            Some(deep)
        } else {
            Some(&resp.fast)
        }
    }

    fn rebuild_trace_nodes(&mut self) {
        let Some(stage) = self.displayed_trace_stage().cloned() else {
            self.trace_nodes.clear();
            return;
        };

        let mut groups = group_trace_paths(stage.traces);
        if !self.show_all_roots {
            groups.truncate(1);
        }

        let mut prev: HashMap<String, bool> = HashMap::new();
        for (idx, key) in self.trace_group_keys.iter().enumerate() {
            prev.insert(
                key.clone(),
                self.trace_expanded.get(idx).copied().unwrap_or(true),
            );
        }

        self.trace_group_keys = groups.iter().map(|g| g.root_symbol_id.clone()).collect();
        self.trace_expanded = groups
            .iter()
            .enumerate()
            .map(|(idx, g)| prev.get(&g.root_symbol_id).copied().unwrap_or(idx == 0))
            .collect();

        self.trace_nodes.clear();
        for (group_idx, g) in groups.iter().enumerate() {
            let expanded = self.trace_expanded.get(group_idx).copied().unwrap_or(true);
            let icon = if expanded { "▼" } else { "▶" };
            let max_hops = g.paths.iter().map(|p| p.steps.len()).max().unwrap_or(0);
            self.trace_nodes.push(TraceNode {
                trace_idx: group_idx,
                step_idx: None,
                depth: 0,
                label: format!(
                    "{icon} {}. {} ({:.3}, {} paths, ≤{} hops) {}:{}",
                    group_idx + 1,
                    g.root_symbol,
                    g.best_score,
                    g.paths.len(),
                    max_hops,
                    g.root_path,
                    g.root_line
                ),
                path: g.root_path.clone(),
                line: g.root_line,
                evidence: None,
            });

            if expanded {
                let mut tree = Vec::new();
                for p in &g.paths {
                    insert_flow_steps(&mut tree, &p.steps, p.score);
                }
                sort_flow_tree(&mut tree);
                push_flow_tree_nodes(&mut self.trace_nodes, group_idx, &tree, "  ");
            }
        }

        self.results.clear();
        if self.trace_nodes.is_empty() {
            self.results_state.select(None);
        } else {
            let cur = self.results_state.selected().unwrap_or(0);
            let bounded = cur.min(self.trace_nodes.len().saturating_sub(1));
            self.results_state.select(Some(bounded));
        }
    }

    fn selected_trace_node(&self) -> Option<&TraceNode> {
        let idx = self.results_state.selected().unwrap_or(0);
        self.trace_nodes.get(idx)
    }

    fn trace_root_count(&self) -> usize {
        let Some(stage) = self.displayed_trace_stage() else {
            return 0;
        };
        let mut roots: HashSet<&str> = HashSet::new();
        for t in &stage.traces {
            roots.insert(t.root_symbol_id.as_str());
        }
        roots.len()
    }

    fn current_list_len(&self) -> usize {
        if self.mode == QueryMode::Trace {
            self.trace_nodes.len()
        } else {
            self.results.len()
        }
    }

    fn expand_selected_trace(&mut self, expand: bool) {
        if self.mode != QueryMode::Trace {
            return;
        }
        let Some(node) = self.selected_trace_node().cloned() else {
            return;
        };
        if let Some(slot) = self.trace_expanded.get_mut(node.trace_idx) {
            *slot = if node.step_idx.is_some() && expand {
                *slot
            } else {
                expand
            };
        }
        self.rebuild_trace_nodes();
        if let Some(root_idx) = self
            .trace_nodes
            .iter()
            .position(|n| n.trace_idx == node.trace_idx && n.step_idx.is_none())
        {
            self.results_state.select(Some(root_idx));
        }
    }

    fn render(&mut self, f: &mut ratatui::Frame<'_>) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());

        self.render_input(f, outer[0]);
        self.render_body(f, outer[1]);
        if self.show_diagnostics {
            self.render_diagnostics_popup(f);
        }
    }

    fn render_input(&self, f: &mut ratatui::Frame<'_>, area: Rect) {
        let mode = self.mode_label();
        let title = match self.focus {
            Focus::Input => Span::styled(
                format!(" Query ({mode}) "),
                Style::default().fg(Color::Yellow),
            ),
            _ => Span::raw(format!(" Query ({mode}) ")),
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(if self.focus == Focus::Input {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        let text = Text::from(vec![
            Line::from(self.input.clone()),
            Line::from(Span::styled(
                self.status.clone(),
                Style::default().fg(Color::DarkGray),
            )),
        ]);

        let p = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
        f.render_widget(p, area);
    }

    fn render_body(&mut self, f: &mut ratatui::Frame<'_>, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);

        self.render_results(f, cols[0]);
        self.render_preview(f, cols[1]);
    }

    fn render_results(&mut self, f: &mut ratatui::Frame<'_>, area: Rect) {
        let title_name = if self.mode == QueryMode::Trace {
            " Flow Tree "
        } else {
            " Results "
        };
        let title = match self.focus {
            Focus::Results => Span::styled(title_name, Style::default().fg(Color::Yellow)),
            _ => Span::raw(title_name),
        };

        let items: Vec<ListItem> = if self.mode == QueryMode::Trace {
            self.trace_nodes
                .iter()
                .map(|n| {
                    let style = if n.depth == 0 {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    ListItem::new(Line::from(Span::styled(n.label.clone(), style)))
                })
                .collect()
        } else {
            self.results
                .iter()
                .map(|r| {
                    let sym = r.symbol.as_deref().unwrap_or("-");
                    ListItem::new(format!("{}:{} {} {}", r.path, r.start_line, r.kind, sym))
                })
                .collect()
        };

        let list = List::new(items)
            .block(
                Block::default()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_style(if self.focus == Focus::Results {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    }),
            )
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        f.render_stateful_widget(list, area, &mut self.results_state);
    }

    fn render_preview(&self, f: &mut ratatui::Frame<'_>, area: Rect) {
        if self.mode == QueryMode::Trace {
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(7), Constraint::Min(0)])
                .split(area);

            let summary_block = Block::default()
                .title(match self.focus {
                    Focus::Preview => Span::styled(" Summary ", Style::default().fg(Color::Yellow)),
                    _ => Span::raw(" Summary "),
                })
                .borders(Borders::ALL)
                .border_style(if self.focus == Focus::Preview {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                });

            let summary = Paragraph::new(self.trace_summary_text.clone())
                .block(summary_block)
                .wrap(Wrap { trim: false });
            f.render_widget(summary, rows[0]);

            let evidence_block = Block::default().title(" Evidence ").borders(Borders::ALL);
            let evidence = Paragraph::new(self.preview_text.clone())
                .block(evidence_block)
                .wrap(Wrap { trim: false })
                .scroll((self.preview_scroll, 0));
            f.render_widget(evidence, rows[1]);
            return;
        }

        let title = match self.focus {
            Focus::Preview => Span::styled(" Preview ", Style::default().fg(Color::Yellow)),
            _ => Span::raw(" Preview "),
        };
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(if self.focus == Focus::Preview {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });
        let p = Paragraph::new(self.preview_text.clone())
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((self.preview_scroll, 0));
        f.render_widget(p, area);
    }

    fn render_diagnostics_popup(&self, f: &mut ratatui::Frame<'_>) {
        let area = centered_rect(74, 76, f.area());
        let block = Block::default()
            .title(Span::styled(
                " Diagnostics (K9s-style) ",
                Style::default().fg(Color::Yellow),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));
        let text = self.build_diagnostics_text();
        let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
        f.render_widget(Clear, area);
        f.render_widget(para, area);
    }

    fn build_diagnostics_text(&self) -> Text<'static> {
        let mut lines = Vec::new();
        lines.push(Line::from("Runtime"));
        lines.push(Line::from(format!(
            "mode: {}",
            self.mode_label().to_ascii_lowercase()
        )));
        let q = self.input.trim();
        lines.push(Line::from(format!(
            "query: {}",
            if q.is_empty() { "(empty)" } else { q }
        )));
        if let Some(resp) = &self.trace_response {
            lines.push(Line::from(format!("deep_status: {}", resp.deep_status)));
            lines.push(Line::from(format!(
                "fast_summary_source: {}",
                resp.fast.summary_source
            )));
            if let Some(deep) = &resp.deep {
                lines.push(Line::from(format!(
                    "deep_summary_source: {}",
                    deep.summary_source
                )));
            }
            if let Some(stage) = self.displayed_trace_stage() {
                if let Some(model) = &stage.summary_model {
                    lines.push(Line::from(format!("active_summary_model: {model}")));
                }
                if let Some(err) = &stage.summary_error {
                    lines.push(Line::from(Span::styled(
                        format!("active_summary_note: {err}"),
                        Style::default().fg(Color::LightRed),
                    )));
                }
            }
        } else {
            lines.push(Line::from("deep_status: n/a"));
        }

        lines.push(Line::from(""));
        lines.push(Line::from("Config"));
        if let Some(cfg) = &self.cfg_snapshot {
            lines.push(Line::from(format!(
                "trace.llm_summary: {}",
                on_off(cfg.trace.llm_summary)
            )));
            lines.push(Line::from(format!("llm.provider: {}", cfg.llm.provider)));
            lines.push(Line::from(format!(
                "llm.model: {}",
                active_llm_model(cfg).unwrap_or_else(|| "(unset)".to_string())
            )));
            lines.push(Line::from(format!("llm.auth: {}", llm_auth_status(cfg))));
            lines.push(Line::from(format!(
                "embed.provider: {}",
                cfg.embed.provider
            )));
            lines.push(Line::from(format!("embed.model: {}", cfg.embed.model)));
            lines.push(Line::from(format!(
                "embed.auth: {}",
                embed_auth_status(cfg)
            )));
        } else {
            lines.push(Line::from("config: unavailable"));
        }

        lines.push(Line::from(""));
        lines.push(Line::from("Keys"));
        lines.push(Line::from("F1 / Ctrl+G: toggle diagnostics"));
        lines.push(Line::from("Esc / F1: close popup"));
        lines.push(Line::from("Ctrl+T / F2: Trace/Flat mode"));
        lines.push(Line::from("a: toggle best/all roots"));
        lines.push(Line::from("Tab: cycle focus · o: open evidence"));

        Text::from(lines)
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn on_off(v: bool) -> &'static str {
    if v { "on" } else { "off" }
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

fn llm_auth_status(cfg: &config::Config) -> String {
    let provider = cfg.llm.provider.trim().to_ascii_lowercase();
    match provider.as_str() {
        "openai" => {
            let env = cfg.providers.openai.api_key_env.trim();
            let ok = !env.is_empty()
                && std::env::var(env)
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
            if ok {
                format!("configured (env:{env})")
            } else {
                format!("missing (env:{env})")
            }
        }
        "zhipu" | "codex" => {
            let env = cfg.providers.zhipu.api_key_env.trim();
            let env_ok = !env.is_empty()
                && std::env::var(env)
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
            let global_ok = config::has_global_zhipu_api_key().unwrap_or(false);
            if env_ok || global_ok {
                format!(
                    "configured (env:{}, global:{})",
                    on_off(env_ok),
                    on_off(global_ok)
                )
            } else {
                format!("missing (env:{env}, global:off)")
            }
        }
        "ollama" => "n/a (local)".to_string(),
        "none" => "disabled".to_string(),
        _ => "unknown".to_string(),
    }
}

fn embed_auth_status(cfg: &config::Config) -> String {
    let provider = cfg.embed.provider.trim().to_ascii_lowercase();
    match provider.as_str() {
        "openai" => {
            let env = cfg.providers.openai.api_key_env.trim();
            let ok = !env.is_empty()
                && std::env::var(env)
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
            if ok {
                format!("configured (env:{env})")
            } else {
                format!("missing (env:{env})")
            }
        }
        "voyage" => {
            let env = cfg.providers.voyage.api_key_env.trim();
            let env_ok = !env.is_empty()
                && std::env::var(env)
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
            let global_ok = config::has_global_voyage_api_key().unwrap_or(false);
            if env_ok || global_ok {
                format!(
                    "configured (env:{}, global:{})",
                    on_off(env_ok),
                    on_off(global_ok)
                )
            } else {
                format!("missing (env:{env}, global:off)")
            }
        }
        "ollama" => "n/a (local)".to_string(),
        "none" => "disabled".to_string(),
        _ => "unknown".to_string(),
    }
}

#[derive(Debug)]
struct HighlightSpec {
    lang: TsLanguage,
    query: Query,
}

#[derive(Debug)]
struct PreviewHighlighter {
    rust: Option<HighlightSpec>,
    js: Option<HighlightSpec>,
    jsx: Option<HighlightSpec>,
    ts: Option<HighlightSpec>,
    tsx: Option<HighlightSpec>,
    python: Option<HighlightSpec>,
    go: Option<HighlightSpec>,
}

impl PreviewHighlighter {
    fn new() -> Self {
        let rust_lang: TsLanguage = tree_sitter_rust::LANGUAGE.into();
        let js_lang: TsLanguage = tree_sitter_javascript::LANGUAGE.into();
        let jsx_lang: TsLanguage = tree_sitter_javascript::LANGUAGE.into();
        let ts_lang: TsLanguage = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
        let tsx_lang: TsLanguage = tree_sitter_typescript::LANGUAGE_TSX.into();
        let py_lang: TsLanguage = tree_sitter_python::LANGUAGE.into();
        let go_lang: TsLanguage = tree_sitter_go::LANGUAGE.into();

        let jsx_query = format!(
            "{}\n{}",
            tree_sitter_javascript::HIGHLIGHT_QUERY,
            tree_sitter_javascript::JSX_HIGHLIGHT_QUERY
        );

        Self {
            rust: build_highlight_spec(rust_lang, tree_sitter_rust::HIGHLIGHTS_QUERY),
            js: build_highlight_spec(js_lang, tree_sitter_javascript::HIGHLIGHT_QUERY),
            jsx: build_highlight_spec(jsx_lang, &jsx_query),
            ts: build_highlight_spec(ts_lang, tree_sitter_typescript::HIGHLIGHTS_QUERY),
            tsx: build_highlight_spec(tsx_lang, tree_sitter_typescript::HIGHLIGHTS_QUERY),
            python: build_highlight_spec(py_lang, tree_sitter_python::HIGHLIGHTS_QUERY),
            go: build_highlight_spec(go_lang, tree_sitter_go::HIGHLIGHTS_QUERY),
        }
    }

    fn highlight_for_path(&self, path: &str, source: &str) -> Text<'static> {
        let source = expand_tabs(source, TAB_STOP);
        let lang = Language::from_path(Path::new(path));
        let Some(spec) = self.spec_for(lang) else {
            return Text::from(source);
        };

        highlight_source(&source, spec).unwrap_or_else(|| Text::from(source))
    }

    fn spec_for(&self, lang: Language) -> Option<&HighlightSpec> {
        match lang {
            Language::Rust => self.rust.as_ref(),
            Language::Js => self.js.as_ref(),
            Language::Jsx => self.jsx.as_ref(),
            Language::Ts => self.ts.as_ref(),
            Language::Tsx => self.tsx.as_ref(),
            Language::Python => self.python.as_ref(),
            Language::Go => self.go.as_ref(),
            Language::Markdown | Language::Unknown => None,
        }
    }
}

const TAB_STOP: usize = 4;

fn expand_tabs(text: &str, tab_stop: usize) -> String {
    let tab_stop = tab_stop.max(1);
    if !text.contains('\t') {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut col = 0usize;
    for ch in text.chars() {
        match ch {
            '\n' => {
                out.push('\n');
                col = 0;
            }
            '\t' => {
                let spaces = tab_stop - (col % tab_stop);
                out.extend(std::iter::repeat(' ').take(spaces));
                col += spaces;
            }
            _ => {
                out.push(ch);
                col = col.saturating_add(1);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{TAB_STOP, expand_tabs};

    #[test]
    fn expand_tabs_replaces_with_spaces() {
        assert_eq!(expand_tabs("\tfoo", TAB_STOP), "    foo");
        assert_eq!(expand_tabs("a\tb", TAB_STOP), "a   b");
        assert_eq!(expand_tabs("ab\n\tc", TAB_STOP), "ab\n    c");
        assert_eq!(expand_tabs("\t", 0), " ");
    }
}

fn build_highlight_spec(lang: TsLanguage, query: &str) -> Option<HighlightSpec> {
    let query = Query::new(&lang, query).ok()?;
    Some(HighlightSpec { lang, query })
}

#[derive(Debug)]
struct StyledRange {
    start: usize,
    end: usize,
    style: Style,
    priority: u8,
    order: usize,
}

fn highlight_source(source: &str, spec: &HighlightSpec) -> Option<Text<'static>> {
    let mut parser = Parser::new();
    parser.set_language(&spec.lang).ok()?;
    let tree = parser.parse(source, None)?;

    let mut cursor = QueryCursor::new();
    let capture_names = spec.query.capture_names();
    let mut ranges = Vec::new();

    let mut matches = cursor.matches(&spec.query, tree.root_node(), source.as_bytes());
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let cap_name = capture_names.get(cap.index as usize).copied().unwrap_or("");
            let Some((style, priority)) = style_for_capture(cap_name) else {
                continue;
            };

            let start = cap.node.start_byte();
            let end = cap.node.end_byte();
            if start >= end || end > source.len() {
                continue;
            }

            ranges.push(StyledRange {
                start,
                end,
                style,
                priority,
                order: ranges.len(),
            });
        }
    }

    Some(render_highlighted_text(source, &ranges))
}

fn render_highlighted_text(source: &str, ranges: &[StyledRange]) -> Text<'static> {
    if ranges.is_empty() {
        return Text::from(source.to_string());
    }

    let mut boundaries = Vec::with_capacity((ranges.len() * 2) + 2);
    boundaries.push(0);
    boundaries.push(source.len());
    for range in ranges {
        boundaries.push(range.start);
        boundaries.push(range.end);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut cur_spans: Vec<Span<'static>> = Vec::new();

    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        if start >= end || !source.is_char_boundary(start) || !source.is_char_boundary(end) {
            continue;
        }

        let style = best_style_for_segment(start, end, ranges).unwrap_or_else(Style::default);
        push_segment_lines(&mut lines, &mut cur_spans, &source[start..end], style);
    }

    lines.push(Line::from(cur_spans));
    Text::from(lines)
}

fn best_style_for_segment(start: usize, end: usize, ranges: &[StyledRange]) -> Option<Style> {
    let mut best_idx: Option<usize> = None;

    for (idx, range) in ranges.iter().enumerate() {
        if range.start <= start && range.end >= end {
            best_idx = match best_idx {
                Some(cur_idx) => {
                    let cur = &ranges[cur_idx];
                    let cur_len = cur.end.saturating_sub(cur.start);
                    let cand_len = range.end.saturating_sub(range.start);
                    if range.priority > cur.priority
                        || (range.priority == cur.priority && cand_len < cur_len)
                        || (range.priority == cur.priority
                            && cand_len == cur_len
                            && range.order > cur.order)
                    {
                        Some(idx)
                    } else {
                        Some(cur_idx)
                    }
                }
                None => Some(idx),
            };
        }
    }

    best_idx.map(|idx| ranges[idx].style)
}

fn push_segment_lines(
    lines: &mut Vec<Line<'static>>,
    cur_spans: &mut Vec<Span<'static>>,
    segment: &str,
    style: Style,
) {
    let mut rest = segment;
    loop {
        let Some(nl_idx) = rest.find('\n') else {
            push_styled(cur_spans, rest, style);
            break;
        };

        let (before, after) = rest.split_at(nl_idx);
        push_styled(cur_spans, before, style);
        lines.push(Line::from(std::mem::take(cur_spans)));
        rest = &after[1..];
    }
}

fn push_styled(spans: &mut Vec<Span<'static>>, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    if style == Style::default() {
        spans.push(Span::raw(text.to_string()));
    } else {
        spans.push(Span::styled(text.to_string(), style));
    }
}

fn style_for_capture(capture: &str) -> Option<(Style, u8)> {
    if capture.starts_with("comment") {
        return Some((
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
            90,
        ));
    }
    if capture.starts_with("keyword") {
        return Some((Style::default().fg(Color::Magenta), 85));
    }
    if capture.starts_with("string") {
        return Some((Style::default().fg(Color::Green), 80));
    }
    if capture.starts_with("number")
        || capture.starts_with("constant")
        || capture.starts_with("boolean")
    {
        return Some((Style::default().fg(Color::Cyan), 80));
    }
    if capture.starts_with("function")
        || capture.starts_with("method")
        || capture.starts_with("constructor")
    {
        return Some((Style::default().fg(Color::Blue), 78));
    }
    if capture.starts_with("type")
        || capture.starts_with("module")
        || capture.starts_with("namespace")
        || capture.starts_with("class")
    {
        return Some((Style::default().fg(Color::LightCyan), 76));
    }
    if capture.starts_with("variable.parameter") || capture.starts_with("parameter") {
        return Some((Style::default().fg(Color::Yellow), 75));
    }
    if capture.starts_with("property")
        || capture.starts_with("field")
        || capture.starts_with("attribute")
    {
        return Some((Style::default().fg(Color::LightBlue), 74));
    }
    if capture.starts_with("operator") {
        return Some((Style::default().fg(Color::Yellow), 70));
    }
    if capture.starts_with("tag") {
        return Some((Style::default().fg(Color::LightMagenta), 70));
    }
    if capture.starts_with("punctuation") {
        return Some((Style::default().fg(Color::DarkGray), 40));
    }
    None
}
