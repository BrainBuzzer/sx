use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use predicates::prelude::*;
use tempfile::TempDir;

fn setup_repo() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(dir.path().join(".git")).expect("create .git");
    dir
}

#[test]
fn query_falls_back_without_vectors() {
    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 { 42 }\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["query", "foo", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Vec<serde_json::Value> =
        serde_json::from_slice(&out).expect("parse JSON query results");
    assert!(!results.is_empty());
    let paths: Vec<String> = results
        .iter()
        .filter_map(|v| {
            v.get("path")
                .and_then(|p| p.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    assert!(paths.iter().any(|p| p == "src/lib.rs"));
}

#[test]
fn embed_is_incremental_and_prunes() {
    let server = StubServer::start();

    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  // foo foo foo\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    write_config(
        dir.path(),
        &format!(
            r#"
version = 2

[providers.ollama]
base_url = "{base}"

[embed]
provider = "ollama"
model = "nomic-embed-text"
batch_size = 32
max_chars = 8000
"#,
            base = server.base_url()
        ),
    );

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out1 = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("embed")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let _ = String::from_utf8_lossy(&out1);

    let count_after_first = server.request_count();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("embed")
        .assert()
        .success();

    let count_after_second = server.request_count();
    assert_eq!(count_after_second, count_after_first);

    std::fs::remove_file(dir.path().join("src/lib.rs")).expect("remove lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out3 = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("embed")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s3 = String::from_utf8_lossy(&out3).to_string();

    let pruned = parse_kv_i64(&s3, "pruned").expect("pruned");
    assert!(pruned > 0);
}

#[test]
fn vsearch_returns_hits_after_embed() {
    let server = StubServer::start();

    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  // foo\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    write_config(
        dir.path(),
        &format!(
            r#"
version = 2

[providers.ollama]
base_url = "{base}"

[embed]
provider = "ollama"
model = "nomic-embed-text"
batch_size = 32
max_chars = 8000
"#,
            base = server.base_url()
        ),
    );

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("embed")
        .assert()
        .success();

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["vsearch", "foo", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Vec<serde_json::Value> =
        serde_json::from_slice(&out).expect("parse JSON vsearch results");
    assert!(!results.is_empty());
    let paths: Vec<String> = results
        .iter()
        .filter_map(|v| {
            v.get("path")
                .and_then(|p| p.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    assert!(paths.iter().any(|p| p == "src/lib.rs"));
}

#[test]
fn openai_embedder_works_with_stub() {
    let server = StubServer::start();

    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  // foo\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    write_config(
        dir.path(),
        &format!(
            r#"
version = 2

[providers.openai]
base_url = "{base}/v1"
api_key_env = "OPENAI_API_KEY"

[embed]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 8
batch_size = 32
max_chars = 8000
"#,
            base = server.base_url()
        ),
    );

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("OPENAI_API_KEY", "test")
        .arg("embed")
        .assert()
        .success()
        .stdout(predicate::str::contains("embedded_new="));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("OPENAI_API_KEY", "test")
        .args(["vsearch", "foo", "--json"])
        .assert()
        .success();
}

#[test]
fn voyage_embedder_works_with_stub() {
    let server = StubServer::start();

    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  // foo\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    write_config(
        dir.path(),
        &format!(
            r#"
version = 3

[providers.voyage]
base_url = "{base}/v1"
api_key_env = "VOYAGE_API_KEY"

[embed]
provider = "voyage"
model = "voyage-code-3"
batch_size = 32
max_chars = 8000
"#,
            base = server.base_url()
        ),
    );

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("VOYAGE_API_KEY", "test")
        .arg("embed")
        .assert()
        .success()
        .stdout(predicate::str::contains("embedded_new="));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("VOYAGE_API_KEY", "test")
        .args(["vsearch", "foo", "--json"])
        .assert()
        .success();
}

#[test]
fn deep_query_uses_voyage_rerank_model() {
    let server = StubServer::start();

    let dir = setup_repo();
    std::fs::create_dir_all(dir.path().join("src")).expect("create src");
    std::fs::write(
        dir.path().join("src/a.rs"),
        "pub fn alpha_main() {\n  // alpha alpha token token\n}\n",
    )
    .expect("write a.rs");
    std::fs::write(
        dir.path().join("src/b.rs"),
        "pub fn alpha_preferred() {\n  // alpha token preferred\n}\n",
    )
    .expect("write b.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success();

    write_config(
        dir.path(),
        &format!(
            r#"
version = 3

[providers.voyage]
base_url = "{base}/v1"
api_key_env = "VOYAGE_API_KEY"
rerank_model = "rerank-2.5"

[deep]
llm_expand = false
llm_rerank = true
rerank_top = 20
"#,
            base = server.base_url()
        ),
    );

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("VOYAGE_API_KEY", "test")
        .args(["query", "alpha token", "--deep", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    assert!(server.request_count() > 0, "expected rerank request");

    let results: Vec<serde_json::Value> =
        serde_json::from_slice(&out).expect("parse JSON deep query results");
    assert!(!results.is_empty());
    let first_path = results
        .first()
        .and_then(|v| v.get("path"))
        .and_then(|p| p.as_str())
        .expect("first result path");
    assert_eq!(first_path, "src/b.rs");
}

fn write_config(repo_root: &std::path::Path, body: &str) {
    let path = repo_root.join(".sx/config.toml");
    std::fs::write(&path, body.trim_start()).expect("write config.toml");
}

fn parse_kv_i64(line: &str, key: &str) -> Option<i64> {
    for part in line.split_whitespace() {
        let Some((k, v)) = part.split_once('=') else {
            continue;
        };
        if k == key {
            return v.trim().parse::<i64>().ok();
        }
    }
    None
}

struct StubServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    requests: Arc<AtomicUsize>,
    handle: Option<JoinHandle<()>>,
}

impl StubServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        listener.set_nonblocking(true).expect("nonblocking");

        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let stop2 = stop.clone();
        let req2 = requests.clone();

        let handle = std::thread::spawn(move || {
            loop {
                if stop2.load(Ordering::SeqCst) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        req2.fetch_add(1, Ordering::SeqCst);
                        let _ = handle_conn(stream);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            addr,
            stop,
            requests,
            handle: Some(handle),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock accept loop (best-effort).
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(100));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn handle_conn(mut stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;

    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];

    // Read until we have headers.
    let header_end;
    loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if buf.len() > 1024 * 1024 {
            return Ok(());
        }
    }

    let headers_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let (method, path) =
        parse_request_line(&headers_str).unwrap_or(("GET".to_string(), "/".to_string()));
    let headers = parse_headers(&headers_str);
    let content_len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    while buf.len() < header_end + content_len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }

    let body = if content_len > 0 && buf.len() >= header_end + content_len {
        &buf[header_end..header_end + content_len]
    } else {
        &[][..]
    };

    let (status, resp_body) = route(&method, &path, &headers, body);
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        resp_body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.write_all(&resp_body)?;
    Ok(())
}

fn route(
    method: &str,
    path: &str,
    headers: &std::collections::HashMap<String, String>,
    body: &[u8],
) -> (u16, Vec<u8>) {
    if method != "POST" {
        return (405, b"{\"error\":\"method not allowed\"}".to_vec());
    }

    if path == "/api/embed" {
        let v: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
        let inputs = extract_inputs(&v);
        let dim = 8;
        let embeddings: Vec<Vec<f32>> = inputs.iter().map(|s| embed_text(s, dim)).collect();
        let resp = serde_json::json!({ "embeddings": embeddings });
        return (200, serde_json::to_vec(&resp).unwrap());
    }

    if path == "/api/embeddings" {
        let v: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
        let prompt = v.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
        let dim = 8;
        let embedding = embed_text(prompt, dim);
        let resp = serde_json::json!({ "embedding": embedding });
        return (200, serde_json::to_vec(&resp).unwrap());
    }

    if path == "/v1/embeddings" {
        let auth_ok = headers
            .get("authorization")
            .map(|v| v.to_ascii_lowercase().starts_with("bearer "))
            .unwrap_or(false);
        if !auth_ok {
            return (401, b"{\"error\":\"missing auth\"}".to_vec());
        }
        let v: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
        let inputs = extract_inputs(&v);
        let dim = v.get("dimensions").and_then(|d| d.as_u64()).unwrap_or(8) as usize;
        let data: Vec<serde_json::Value> = inputs
            .iter()
            .map(|s| serde_json::json!({ "embedding": embed_text(s, dim) }))
            .collect();
        let resp = serde_json::json!({ "data": data });
        return (200, serde_json::to_vec(&resp).unwrap());
    }

    if path == "/v1/rerank" {
        let auth_ok = headers
            .get("authorization")
            .map(|v| v.to_ascii_lowercase().starts_with("bearer "))
            .unwrap_or(false);
        if !auth_ok {
            return (401, b"{\"error\":\"missing auth\"}".to_vec());
        }

        let v: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
        let docs: Vec<String> = v
            .get("documents")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        let mut data: Vec<(usize, f64)> = docs
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let score = if doc.to_ascii_lowercase().contains("preferred") {
                    1.0
                } else {
                    0.1
                };
                (idx, score)
            })
            .collect();
        data.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let resp = serde_json::json!({
            "data": data
                .into_iter()
                .map(|(idx, score)| serde_json::json!({"index": idx, "relevance_score": score}))
                .collect::<Vec<serde_json::Value>>()
        });
        return (200, serde_json::to_vec(&resp).unwrap());
    }

    (404, b"{\"error\":\"not found\"}".to_vec())
}

fn extract_inputs(v: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = v.get("input").and_then(|i| i.as_array()) {
        return arr
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect();
    }
    if let Some(s) = v.get("input").and_then(|i| i.as_str()) {
        return vec![s.to_string()];
    }
    vec![]
}

fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];
    for tok in tokenize(text) {
        let h = blake3::hash(tok.as_bytes());
        let b = h.as_bytes()[0] as usize;
        let idx = b % v.len();
        v[idx] += 1.0;
    }
    v
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let ch = ch.to_ascii_lowercase();
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_request_line(headers: &str) -> Option<(String, String)> {
    let line = headers.lines().next()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    Some((method, path))
}

fn parse_headers(headers: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for line in headers.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        out.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
    }
    out
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
