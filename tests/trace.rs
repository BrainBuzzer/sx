use std::fs;

use tempfile::TempDir;

fn setup_repo() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    fs::create_dir_all(dir.path().join(".git")).expect("create .git");
    dir
}

fn run_index(repo: &std::path::Path) {
    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(repo)
        .arg("index")
        .assert()
        .success();
}

fn run_trace_json(repo: &std::path::Path, query: &str) -> serde_json::Value {
    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(repo)
        .args(["trace", query, "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&out).expect("parse trace json")
}

#[test]
fn trace_schema_is_v3() {
    let dir = setup_repo();
    run_index(dir.path());

    let conn =
        rusqlite::Connection::open(dir.path().join(".sx/index.sqlite")).expect("open sqlite db");
    let schema: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |row| row.get(0),
        )
        .expect("read schema_version");
    assert_eq!(schema, "3");

    let tables = ["symbols", "trace_edges", "trace_query_cache"];
    for t in tables {
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [t],
                |row| row.get(0),
            )
            .expect("table exists query");
        assert_eq!(exists, 1, "missing table {t}");
    }
}

#[test]
fn trace_rust_multi_hop_and_summary() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/driver.rs"),
        r#"
pub fn show_missions() {
    store_driver();
}

pub fn store_driver() {
    let _q = "select * from driver_missions";
}
"#,
    )
    .expect("write rust file");

    run_index(dir.path());
    let json = run_trace_json(dir.path(), "show missions driver");
    let fast = json.get("fast").expect("fast");
    let traces = fast
        .get("traces")
        .and_then(|v| v.as_array())
        .expect("traces");
    assert!(!traces.is_empty());
    assert!(traces.iter().any(|t| {
        t.get("steps")
            .and_then(|s| s.as_array())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }));
    let summary = fast
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(!summary.is_empty());
    let source = fast
        .get("summary_source")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(matches!(source, "deterministic" | "llm"));
}

#[test]
fn trace_go_route_edges_present() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("go")).expect("create go dir");
    fs::write(
        dir.path().join("go/service.go"),
        r#"
package main

import "net/http"

func showMissions(w http.ResponseWriter, r *http.Request) {
    storeDriver()
}

func storeDriver() {
    q := "select * from driver_missions"
    _ = q
}

func registerRoutes() {
    http.HandleFunc("/missions", showMissions)
}
"#,
    )
    .expect("write go file");

    run_index(dir.path());
    let json = run_trace_json(dir.path(), "missions handler route");
    let traces = json["fast"]["traces"].as_array().expect("fast traces");
    assert!(!traces.is_empty());
    let has_route = traces.iter().any(|t| {
        t.get("steps")
            .and_then(|s| s.as_array())
            .map(|steps| {
                steps
                    .iter()
                    .any(|s| s.get("edge_kind").and_then(|k| k.as_str()) == Some("route"))
            })
            .unwrap_or(false)
    });
    assert!(has_route);
}

#[test]
fn trace_ts_and_python_supported() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("ts")).expect("create ts dir");
    fs::create_dir_all(dir.path().join("py")).expect("create py dir");

    fs::write(
        dir.path().join("ts/routes.ts"),
        r#"
function showMissions() {
  return storeDriver();
}
function storeDriver() {
  const q = "insert into driver_missions values (1)";
  return q;
}
function registerRoutes(router: any) {
  router.get("/missions", showMissions);
}
"#,
    )
    .expect("write ts file");

    fs::write(
        dir.path().join("py/routes.py"),
        r#"
def show_missions():
    return store_driver()

def store_driver():
    q = "select * from driver_missions"
    return q

@app.get("/missions")
def missions_handler():
    return show_missions()
"#,
    )
    .expect("write py file");

    run_index(dir.path());
    let json = run_trace_json(dir.path(), "missions route store");

    let paths: Vec<String> = json["fast"]["citations"]
        .as_array()
        .expect("citations")
        .iter()
        .filter_map(|c| {
            c.get("path")
                .and_then(|p| p.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    assert!(paths.iter().any(|p| p.ends_with("ts/routes.ts")));
    assert!(paths.iter().any(|p| p.ends_with("py/routes.py")));
}

#[test]
fn trace_works_with_llm_none() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        r#"
pub fn a() { b(); }
pub fn b() { let _q = "select * from x"; }
"#,
    )
    .expect("write rust");

    run_index(dir.path());
    let json = run_trace_json(dir.path(), "how does a call b");

    assert_eq!(
        json.get("deep_status")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        "ready"
    );
    let deep_summary = json
        .get("deep")
        .and_then(|v| v.get("summary"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(!deep_summary.is_empty());
    let deep_source = json
        .get("deep")
        .and_then(|v| v.get("summary_source"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(deep_source, "deterministic");
}
