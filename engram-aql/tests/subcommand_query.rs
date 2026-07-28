//! CLI query subcommand integration test.
//! Creates a temp .engram file, runs `engram-aql query`, asserts on stdout.

mod common;

use std::process::Command;

use engram_aql::vector::codec::encode_f32_le;
use rusqlite::params;
use tempfile::NamedTempFile;

fn build_test_db() -> NamedTempFile {
    let file = NamedTempFile::new().unwrap();
    let path = file.path();
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(common::SCHEMA_SQL).unwrap();
    conn.execute_batch(include_str!("fixtures/seed.sql"))
        .unwrap();
    drop(conn);
    file
}

/// Build a DB with three `world` chunks carrying known 4-dim embeddings so a
/// `--var q=[...]` array probe has a deterministic nearest-neighbour order.
fn build_vector_db() -> NamedTempFile {
    let file = NamedTempFile::new().unwrap();
    let conn = rusqlite::Connection::open(file.path()).unwrap();
    conn.execute_batch(common::SCHEMA_SQL).unwrap();
    let rows: [(&str, [f32; 4]); 3] = [
        ("v1", [1.0, 0.0, 0.0, 0.0]),
        ("v2", [0.0, 1.0, 0.0, 0.0]),
        ("v3", [0.9, 0.1, 0.0, 0.0]),
    ];
    for (id, emb) in rows {
        conn.execute(
            "INSERT INTO chunks (id, text, memory_type, embedding, trust_score, source_type, is_active) \
             VALUES (?1, ?2, 'world', ?3, 0.8, 'user_stated', 1)",
            params![id, format!("chunk {id}"), encode_f32_le(&emb)],
        )
        .unwrap();
    }
    drop(conn);
    file
}

#[test]
fn query_subcommand_returns_json() {
    let file = build_test_db();

    let output = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("query")
        .arg(file.path())
        .arg("RECALL FROM EPISODIC ALL LIMIT 2")
        .output()
        .expect("failed to run engram-aql");

    assert!(
        output.status.success(),
        "engram-aql exited non-zero\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert_eq!(json["success"], true);
    assert_eq!(json["statement"], "Recall");
    assert!(json["count"].as_u64().unwrap() >= 1);
}

#[test]
fn query_subcommand_var_array_probe_orders_by_similarity() {
    let file = build_vector_db();

    let output = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("query")
        .arg(file.path())
        .arg("RECALL FROM SEMANTIC LIKE $q")
        .arg("--var")
        .arg("q=[1,0,0,0]")
        .output()
        .expect("failed to run engram-aql");

    assert!(
        output.status.success(),
        "engram-aql exited non-zero\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert_eq!(json["success"], true);
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3, "all three world chunks should rank");
    // Probe [1,0,0,0]: v1 identical (dist 0) < v3 near < v2 orthogonal.
    assert_eq!(data[0]["id"], "v1", "nearest probe should rank v1 first");
    assert_eq!(data[1]["id"], "v3");
    assert_eq!(data[2]["id"], "v2");
}

#[test]
fn query_subcommand_malformed_var_exits_nonzero() {
    let file = build_vector_db();

    let output = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("query")
        .arg(file.path())
        .arg("RECALL FROM SEMANTIC ALL")
        .arg("--var")
        .arg("noequalsign") // missing '=' → parse error before opening DB
        .output()
        .expect("failed to run engram-aql");

    assert!(
        !output.status.success(),
        "a malformed --var (no '=') should exit non-zero"
    );
}

#[test]
fn query_subcommand_nonzero_exit_on_parse_error() {
    let file = build_test_db();

    let output = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("query")
        .arg(file.path())
        .arg("NOT A VALID QUERY")
        .output()
        .expect("failed to run engram-aql");

    // Parse errors produce a JSON error result AND a non-zero exit code
    assert!(!output.status.success(), "parse error should exit non-zero");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should still be valid JSON");
    assert_eq!(json["success"], false);
    assert!(json["error"].is_string());
}
