use std::fs;

use predicates::prelude::*;
use tempfile::TempDir;

fn setup_repo() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    fs::create_dir_all(dir.path().join(".git")).expect("create .git");
    dir
}

#[test]
fn init_creates_sx_dir_and_db() {
    let dir = setup_repo();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("init")
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialized:"));

    assert!(dir.path().join(".sx").is_dir());
    assert!(dir.path().join(".sx/config.toml").is_file());
    assert!(dir.path().join(".sx/index.sqlite").is_file());
}

#[test]
fn onboard_defaults_writes_config() {
    let dir = setup_repo();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["onboard", "--defaults", "--skip-check"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Saved config:"));

    assert!(dir.path().join(".sx/config.toml").is_file());
}

#[test]
fn index_is_incremental() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success()
        .stdout(predicate::str::contains("indexed=1"));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success()
        .stdout(predicate::str::contains("indexed=0"))
        .stdout(predicate::str::contains("skipped=1"));
}

#[test]
fn search_get_open_roundtrip() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let output = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["search", "foo", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Vec<serde_json::Value> =
        serde_json::from_slice(&output).expect("parse JSON search results");
    assert!(!results.is_empty());
    let chunk_id = results[0]["chunk_id"].as_str().expect("chunk_id string");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["get", chunk_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("fn foo").or(predicate::str::contains("pub fn foo")));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("EDITOR", "vim")
        .args(["open", "--dry-run", chunk_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("vim"))
        .stdout(predicate::str::contains("+"));
}

#[test]
fn cd_updates_frecency() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("auth")).expect("create auth");
    fs::create_dir_all(dir.path().join("other")).expect("create other");
    fs::write(
        dir.path().join("auth/mod.rs"),
        "pub fn auth_entry() {\n  // auth auth auth\n}\n",
    )
    .expect("write auth/mod.rs");
    fs::write(
        dir.path().join("other/mod.rs"),
        "pub fn other_entry() {\n  // auth\n}\n",
    )
    .expect("write other/mod.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out1 = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["cd", "auth", "--relative"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let chosen = String::from_utf8_lossy(&out1).trim().to_string();
    assert_eq!(chosen, "auth");

    let db_path = dir.path().join(".sx/index.sqlite");
    let conn = rusqlite::Connection::open(db_path).expect("open sqlite");
    let (rank1,): (f64,) = conn
        .query_row(
            "SELECT rank FROM dir_frecency WHERE path=?1",
            [&chosen],
            |row| Ok((row.get(0)?,)),
        )
        .expect("rank after first cd");
    assert!(rank1 >= 1.0);

    let out2 = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["cd", "auth", "--relative"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let chosen2 = String::from_utf8_lossy(&out2).trim().to_string();
    assert_eq!(chosen2, "auth");

    let (rank2,): (f64,) = conn
        .query_row(
            "SELECT rank FROM dir_frecency WHERE path=?1",
            [&chosen],
            |row| Ok((row.get(0)?,)),
        )
        .expect("rank after second cd");
    assert!(rank2 > rank1);
}

#[test]
fn get_by_path_line_works() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["get", "src/lib.rs:1"])
        .assert()
        .success()
        .stdout(predicate::str::contains("fn foo").or(predicate::str::contains("pub fn foo")));
}

#[test]
fn open_by_path_line_dry_run_renders_command() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .env("EDITOR", "vim")
        .args(["open", "--dry-run", "src/lib.rs:2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("vim"))
        .stdout(predicate::str::contains("+2"))
        .stdout(predicate::str::contains("src/lib.rs"));
}

#[test]
fn guide_json_returns_a_trail() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("auth")).expect("create auth");
    fs::create_dir_all(dir.path().join("docs")).expect("create docs");
    fs::write(
        dir.path().join("auth/mod.rs"),
        "pub fn auth_entry() {\n  // token: guided\n}\n",
    )
    .expect("write auth/mod.rs");
    fs::write(
        dir.path().join("docs/readme.md"),
        "# Auth\n\ntoken: guided\n",
    )
    .expect("write docs/readme.md");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["guide", "guided", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out).expect("parse guide JSON");
    assert!(!hits.is_empty());
    let paths: Vec<String> = hits
        .iter()
        .filter_map(|v| {
            v.get("path")
                .and_then(|p| p.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    assert!(paths.iter().any(|p| p == "auth/mod.rs"));
    assert!(paths.iter().any(|p| p == "docs/readme.md"));
}

#[test]
fn index_respects_gitignore() {
    let dir = setup_repo();
    fs::write(dir.path().join(".gitignore"), "ignored.rs\n").expect("write .gitignore");
    fs::write(
        dir.path().join("ignored.rs"),
        "pub fn ignored() { /* token: SHOULD_NOT_INDEX */ }\n",
    )
    .expect("write ignored");
    fs::write(
        dir.path().join("kept.rs"),
        "pub fn kept() { /* token: SHOULD_INDEX */ }\n",
    )
    .expect("write kept");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("index")
        .assert()
        .success();

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["search", "SHOULD_NOT_INDEX", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out).expect("parse JSON");
    assert!(hits.is_empty());

    let out2 = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["search", "SHOULD_INDEX", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let hits2: Vec<serde_json::Value> = serde_json::from_slice(&out2).expect("parse JSON");
    assert!(!hits2.is_empty());
}

#[test]
fn doctor_smoke() {
    let dir = setup_repo();
    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("fts5: ok"));
}

#[test]
fn auth_zhipu_stores_global_key_without_repo_root() {
    let work = TempDir::new().expect("work tempdir");
    let fake_home = TempDir::new().expect("home tempdir");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(work.path())
        .env("HOME", fake_home.path())
        .args(["auth", "zhipu", "--api-key", "zai_test_key_123"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stored Zhipu API key"));

    let creds_path = fake_home.path().join(".sx/credentials.toml");
    assert!(creds_path.is_file());
    let body = fs::read_to_string(&creds_path).expect("read credentials.toml");
    assert!(body.contains("zhipu_api_key"));
    assert!(body.contains("zai_test_key_123"));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(work.path())
        .env("HOME", fake_home.path())
        .args(["auth", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains("zhipu_api_key: configured"));
}

#[test]
fn auth_voyage_stores_global_key_without_repo_root() {
    let work = TempDir::new().expect("work tempdir");
    let fake_home = TempDir::new().expect("home tempdir");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(work.path())
        .env("HOME", fake_home.path())
        .args(["auth", "voyage", "--api-key", "voyage_test_key_123"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stored Voyage API key"));

    let creds_path = fake_home.path().join(".sx/credentials.toml");
    assert!(creds_path.is_file());
    let body = fs::read_to_string(&creds_path).expect("read credentials.toml");
    assert!(body.contains("voyage_api_key"));
    assert!(body.contains("voyage_test_key_123"));

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(work.path())
        .env("HOME", fake_home.path())
        .args(["auth", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains("voyage_api_key: configured"));
}

#[test]
fn root_discovery_works_from_nested_dir() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");
    fs::create_dir_all(dir.path().join("nested/inner")).expect("create nested");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path().join("nested/inner"))
        .arg("index")
        .assert()
        .success();

    assert!(dir.path().join(".sx/index.sqlite").is_file());

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path().join("nested/inner"))
        .args(["search", "foo", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out).expect("parse JSON");
    assert!(!hits.is_empty());
}

#[test]
fn db_override_allows_isolated_indexes() {
    let dir = setup_repo();
    fs::create_dir_all(dir.path().join("src")).expect("create src");
    fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn foo() -> i32 {\n  42\n}\n",
    )
    .expect("write lib.rs");

    let custom_db = dir.path().join("custom.sqlite");

    assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args(["--db", custom_db.to_str().unwrap(), "index"])
        .assert()
        .success();

    assert!(custom_db.is_file());

    let out = assert_cmd::cargo::cargo_bin_cmd!("sx")
        .current_dir(dir.path())
        .args([
            "--db",
            custom_db.to_str().unwrap(),
            "search",
            "foo",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let hits: Vec<serde_json::Value> = serde_json::from_slice(&out).expect("parse JSON");
    assert!(!hits.is_empty());
}
