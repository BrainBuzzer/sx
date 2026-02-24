use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolRecord {
    pub symbol_id: String,
    pub chunk_id: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub language: String,
    pub kind: String,
    pub fq_name: String,
    pub short_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub edge_id: String,
    pub src_symbol_id: String,
    pub dst_symbol_id: Option<String>,
    pub dst_name: Option<String>,
    pub edge_kind: String,
    pub path: String,
    pub line: i64,
    pub confidence: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub path: String,
    pub line: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub edge_kind: String,
    pub from_symbol: String,
    pub to_symbol: Option<String>,
    pub target_name: Option<String>,
    pub path: String,
    pub line: i64,
    pub confidence: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracePath {
    pub score: f64,
    pub root_symbol_id: String,
    pub root_symbol: String,
    pub root_path: String,
    pub root_line: i64,
    pub steps: Vec<TraceStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStageResult {
    pub traces: Vec<TracePath>,
    pub summary: String,
    pub citations: Vec<Citation>,
    #[serde(default = "default_summary_source")]
    pub summary_source: String,
    #[serde(default)]
    pub summary_model: Option<String>,
    #[serde(default)]
    pub summary_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceResponse {
    pub query: String,
    pub fast: TraceStageResult,
    pub deep: Option<TraceStageResult>,
    pub deep_status: String,
}

#[derive(Debug, Clone)]
pub struct EdgeCandidate {
    pub src_symbol_id: String,
    pub dst_name: Option<String>,
    pub edge_kind: String,
    pub path: String,
    pub line: i64,
    pub confidence: f64,
    pub evidence: String,
}

fn default_summary_source() -> String {
    "deterministic".to_string()
}
