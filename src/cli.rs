use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "sx",
    version,
    about = "Semantic explorer for codebases (Phase 1: BM25 + chunks)"
)]
pub struct Cli {
    /// Override the repository root (defaults to nearest ancestor containing `.git`)
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Override the SQLite DB path (defaults to `<root>/.sx/index.sqlite`)
    #[arg(long)]
    pub db: Option<PathBuf>,

    /// Increase verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Bootstrap `.sx/`, migrate DB, index repository, and build embeddings
    Init,
    /// Interactive onboarding wizard for provider/model/auth configuration
    Onboard(OnboardArgs),
    /// Build/update the index (incremental)
    Index(IndexArgs),
    /// Search the index via SQLite FTS5 (BM25)
    Search(SearchArgs),
    /// Print the referenced chunk/snippet
    Get(GetArgs),
    /// Open the referenced location in $EDITOR (best-effort)
    Open(OpenArgs),
    /// Print a deterministic “reading trail” for a query
    Guide(GuideArgs),
    /// Print the best-matching directory (for `cd "$(sx cd …)"`)
    Cd(CdArgs),
    /// Generate/update embeddings for chunks and build the vector index (incremental)
    Embed(EmbedArgs),
    /// Vector-only semantic search (requires `sx embed` first)
    Vsearch(VsearchArgs),
    /// Hybrid search (BM25 + vector) with optional deep mode
    Query(QueryArgs),
    /// Graph-first semantic trace search (paths + summary)
    Trace(TraceArgs),
    /// Manage global provider credentials
    Auth(AuthArgs),
    /// Launch the interactive terminal UI (default when no subcommand is provided)
    Tui,
    /// Environment/config/DB checks
    Doctor,
}

#[derive(Args, Debug, Clone)]
pub struct IndexArgs {
    /// Drop and rebuild the index tables
    #[arg(long)]
    pub full: bool,

    /// Reserved for later parallelism; currently must be 1
    #[arg(long, default_value_t = 1)]
    pub jobs: usize,
}

#[derive(Args, Debug, Clone)]
pub struct OnboardArgs {
    /// Use current/default values without interactive prompts
    #[arg(long)]
    pub defaults: bool,

    /// Skip embedding connectivity preflight check
    #[arg(long)]
    pub skip_check: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Files,
    Md,
}

#[derive(Args, Debug, Clone, Default)]
pub struct OutputArgs {
    /// Output JSON (machine-readable)
    #[arg(long, conflicts_with_all = ["files", "md"])]
    pub json: bool,

    /// Output unique file paths only
    #[arg(long, conflicts_with_all = ["json", "md"])]
    pub files: bool,

    /// Output Markdown
    #[arg(long, conflicts_with_all = ["json", "files"])]
    pub md: bool,
}

impl OutputArgs {
    pub fn format(&self) -> OutputFormat {
        if self.json {
            OutputFormat::Json
        } else if self.files {
            OutputFormat::Files
        } else if self.md {
            OutputFormat::Md
        } else {
            OutputFormat::Text
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct SearchArgs {
    pub query: String,

    /// Max number of results to return
    #[arg(long, default_value_t = 50)]
    pub limit: usize,

    /// Escape FTS operators and treat the query as plain terms (default)
    #[arg(long, default_value_t = true)]
    pub literal: bool,

    /// Treat the query as raw FTS5 syntax
    #[arg(long)]
    pub fts: bool,

    /// Filter by language (repeatable)
    #[arg(long, value_name = "LANG")]
    pub lang: Vec<String>,

    /// Filter by path prefix (repeatable, repo-relative)
    #[arg(long = "path-prefix", value_name = "PREFIX")]
    pub path_prefix: Vec<String>,

    #[command(flatten)]
    pub output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
pub struct AuthArgs {
    #[command(subcommand)]
    pub command: AuthCommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum AuthCommand {
    /// Store/update Zhipu API key in ~/.sx/credentials.toml
    Zhipu(ZhipuAuthArgs),
    /// Store/update Voyage API key in ~/.sx/credentials.toml
    Voyage(VoyageAuthArgs),
    /// Show global auth status
    Status,
}

#[derive(Args, Debug, Clone)]
pub struct ZhipuAuthArgs {
    /// Provide key directly (avoid this on shared terminals/shell history)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Read key from current environment variable (defaults to ZAI_API_KEY)
    #[arg(long)]
    pub from_env: bool,

    /// Remove stored key from ~/.sx/credentials.toml
    #[arg(long)]
    pub clear: bool,
}

#[derive(Args, Debug, Clone)]
pub struct VoyageAuthArgs {
    /// Provide key directly (avoid this on shared terminals/shell history)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Read key from current environment variable (defaults to VOYAGE_API_KEY)
    #[arg(long)]
    pub from_env: bool,

    /// Remove stored key from ~/.sx/credentials.toml
    #[arg(long)]
    pub clear: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GetArgs {
    pub target: String,

    /// Include N lines of surrounding file context
    #[arg(long, default_value_t = 0)]
    pub context: usize,

    #[command(flatten)]
    pub output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
pub struct OpenArgs {
    pub target: String,

    /// Print the command that would run instead of spawning it
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GuideArgs {
    pub query: String,

    /// Max number of files in the trail
    #[arg(long, default_value_t = 8)]
    pub limit_files: usize,

    /// Max number of raw search results to consider before grouping
    #[arg(long, default_value_t = 200)]
    pub limit_results: usize,

    #[command(flatten)]
    pub output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
pub struct CdArgs {
    pub query: String,

    /// Max number of raw search results to consider
    #[arg(long, default_value_t = 200)]
    pub limit_results: usize,

    /// Print repo-relative directory instead of absolute path
    #[arg(long)]
    pub relative: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EmbedArgs {
    /// Drop and rebuild the vector collection for the selected provider/model
    #[arg(long)]
    pub full: bool,

    /// Override embedding provider for this run (ollama|openai|voyage)
    #[arg(long)]
    pub provider: Option<String>,

    /// Override embedding model name for this run
    #[arg(long)]
    pub model: Option<String>,

    /// Override embedding dimensions (OpenAI only)
    #[arg(long)]
    pub dimensions: Option<usize>,

    /// Override embedding batch size
    #[arg(long)]
    pub batch_size: Option<usize>,
}

#[derive(Args, Debug, Clone)]
pub struct VsearchArgs {
    pub query: String,

    /// Max number of results to return
    #[arg(long, default_value_t = 50)]
    pub limit: usize,

    /// Filter by language (repeatable)
    #[arg(long, value_name = "LANG")]
    pub lang: Vec<String>,

    /// Filter by path prefix (repeatable, repo-relative)
    #[arg(long = "path-prefix", value_name = "PREFIX")]
    pub path_prefix: Vec<String>,

    #[command(flatten)]
    pub output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
pub struct QueryArgs {
    pub query: String,

    /// Max number of results to return
    #[arg(long, default_value_t = 50)]
    pub limit: usize,

    /// BM25 candidate limit before fusion
    #[arg(long, default_value_t = 200)]
    pub bm25_limit: usize,

    /// Vector candidate limit before fusion
    #[arg(long, default_value_t = 200)]
    pub vec_limit: usize,

    /// Enable deep mode (PRF expansion + deterministic rerank; optional LLM expansion)
    #[arg(long)]
    pub deep: bool,

    /// Filter by language (repeatable)
    #[arg(long, value_name = "LANG")]
    pub lang: Vec<String>,

    /// Filter by path prefix (repeatable, repo-relative)
    #[arg(long = "path-prefix", value_name = "PREFIX")]
    pub path_prefix: Vec<String>,

    #[command(flatten)]
    pub output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
pub struct TraceArgs {
    pub query: String,

    /// Max number of trace paths to return
    #[arg(long, default_value_t = 5)]
    pub limit_traces: usize,

    /// Max hops during fast-stage path expansion
    #[arg(long, default_value_t = 4)]
    pub max_hops_fast: usize,

    /// Max hops during deep-stage path expansion
    #[arg(long, default_value_t = 8)]
    pub max_hops_deep: usize,

    /// Enable deep stage (default: enabled)
    #[arg(long, default_value_t = true)]
    pub deep: bool,

    /// Filter by language (repeatable)
    #[arg(long, value_name = "LANG")]
    pub lang: Vec<String>,

    /// Filter by path prefix (repeatable, repo-relative)
    #[arg(long = "path-prefix", value_name = "PREFIX")]
    pub path_prefix: Vec<String>,

    #[command(flatten)]
    pub output: OutputArgs,
}
