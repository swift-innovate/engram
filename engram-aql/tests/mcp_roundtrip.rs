//! MCP stdio server round-trip test.
//! Spawns `engram-aql mcp` against a temp DB, sends JSON-RPC requests,
//! asserts on the responses.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use engram_aql::vector::codec::encode_f32_le;
use rusqlite::params;
use tempfile::NamedTempFile;

fn setup_db() -> NamedTempFile {
    let file = NamedTempFile::new().unwrap();
    let conn = rusqlite::Connection::open(file.path()).unwrap();
    conn.execute_batch(common::SCHEMA_SQL).unwrap();
    conn.execute_batch(include_str!("fixtures/seed.sql"))
        .unwrap();
    drop(conn);
    file
}

/// DB with three `world` chunks carrying known 4-dim embeddings so an
/// array-bound `variables` probe yields a deterministic order (no bridge).
fn setup_vector_db() -> NamedTempFile {
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

fn send(stdin: &mut impl Write, msg: &str) {
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();
}

fn read_line(reader: &mut impl BufRead) -> String {
    let mut s = String::new();
    reader.read_line(&mut s).unwrap();
    s.trim().to_string()
}

#[test]
fn mcp_initialize_and_list_tools_and_call() {
    let file = setup_db();

    let mut child = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("mcp")
        .arg(file.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engram-aql mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // initialize
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value =
        serde_json::from_str(&resp).expect("invalid JSON in initialize response");
    assert_eq!(v["id"], 1);
    assert!(
        v["result"]["protocolVersion"].is_string(),
        "initialize result should include protocolVersion"
    );

    // notifications/initialized — no response expected
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    );

    // tools/list
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value =
        serde_json::from_str(&resp).expect("invalid JSON in tools/list response");
    assert_eq!(v["id"], 2);
    let tools = v["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "engram_aql");

    // tools/call with a simple RECALL
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"engram_aql","arguments":{"query":"RECALL FROM EPISODIC ALL LIMIT 2"}}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value =
        serde_json::from_str(&resp).expect("invalid JSON in tools/call response");
    assert_eq!(v["id"], 3);
    let content = &v["result"]["content"];
    assert!(content.is_array(), "result.content should be an array");
    let text = content[0]["text"].as_str().unwrap();
    assert!(
        text.contains("\"success\": true") || text.contains("\"success\":true"),
        "tool call result should contain success:true, got: {}",
        text
    );
    assert!(
        text.contains("\"statement\": \"Recall\"") || text.contains("\"statement\":\"Recall\""),
        "tool call result should contain statement:Recall, got: {}",
        text
    );

    // Close stdin to signal EOF
    drop(stdin);
    let _ = child.wait();
}

#[test]
fn mcp_tools_call_with_variables_runs_vector_search() {
    let file = setup_vector_db();

    let mut child = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("mcp")
        .arg(file.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engram-aql mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
    );
    let _ = read_line(&mut stdout);

    // tools/list should advertise the `variables` input property.
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(
        v["result"]["tools"][0]["inputSchema"]["properties"]["variables"].is_object(),
        "tools/list should expose a `variables` input property"
    );

    // tools/call with an array-bound probe (no engram-mcp bridge needed).
    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"engram_aql","arguments":{"query":"RECALL FROM SEMANTIC LIKE $q","variables":{"q":[1,0,0,0]}}}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 3);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let result: serde_json::Value = serde_json::from_str(text).expect("inner result JSON");
    assert_eq!(result["success"], true);
    let data = result["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3);
    assert_eq!(data[0]["id"], "v1", "probe [1,0,0,0] should rank v1 first");
    assert_eq!(data[2]["id"], "v2", "orthogonal vector should rank last");

    drop(stdin);
    let _ = child.wait();
}

#[test]
fn mcp_unknown_method_returns_error() {
    let file = setup_db();

    let mut child = Command::new(env!("CARGO_BIN_EXE_engram-aql"))
        .arg("mcp")
        .arg(file.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engram-aql mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        r#"{"jsonrpc":"2.0","id":42,"method":"unknown/method","params":{}}"#,
    );
    let resp = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&resp).expect("invalid JSON in error response");
    assert_eq!(v["id"], 42);
    assert!(v["error"].is_object(), "should return an error object");
    // Method not found is -32601
    assert_eq!(v["error"]["code"], -32601);

    drop(stdin);
    let _ = child.wait();
}
