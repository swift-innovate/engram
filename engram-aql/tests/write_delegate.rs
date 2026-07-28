//! Phase 2b write-delegation round-trips (GATED on engram-mcp).
//!
//! These prove that AQL writes go all the way through the bridge to the
//! canonical TS retain pipeline AND that the Rust read-only connection sees the
//! committed result on the SAME `Executor` (cross-process WAL visibility):
//!
//!   - STORE persists a chunk that a subsequent Rust RECALL reads.
//!   - The stored chunk is fully embedded (semantic RECALL ... LIKE finds it).
//!   - FORGET soft-deletes it (drops from active recall).
//!   - UPDATE supersedes it (old gone, new present).
//!
//! Skipped cleanly when `engram-mcp` is not discoverable (run locally with
//! `ENGRAM_MCP_CMD="node /abs/path/to/dist/mcp-server.js"`), mirroring the
//! other cross-process suites so CI without the built TS bin stays green.

mod common;

use std::collections::BTreeMap;

use engram_aql::bridge::child::resolve_engram_mcp_cmd;
use engram_aql::Executor;
use serde_json::{json, Value};
use tempfile::NamedTempFile;

/// True when no engram-mcp is discoverable; tests early-return with a notice.
fn bridge_unavailable(test: &str) -> bool {
    if resolve_engram_mcp_cmd(None).is_none() {
        eprintln!(
            "[{test}] SKIP — engram-mcp not discoverable. \
             Set ENGRAM_MCP_CMD=\"node /abs/path/to/dist/mcp-server.js\" to run."
        );
        true
    } else {
        false
    }
}

/// A fresh, schema-seeded `.engram` file the bridge child + Rust both open.
fn make_engram() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".engram").unwrap();
    let conn = rusqlite::Connection::open(file.path()).unwrap();
    conn.execute_batch(common::SCHEMA_SQL).unwrap();
    drop(conn);
    file
}

#[test]
fn store_then_recall_sees_new_chunk() {
    if bridge_unavailable("store_then_recall") {
        return;
    }
    let db = make_engram();
    let exec = Executor::open(db.path()).unwrap();

    let store = exec
        .query(r#"STORE INTO SEMANTIC (text = "terraform manages cloud infrastructure")"#)
        .unwrap();
    assert!(store.success, "STORE failed: {:?}", store.error);
    let chunk_id = store.data[0]["chunkId"].as_str();
    assert!(
        chunk_id.is_some(),
        "STORE should return a chunkId, got: {:?}",
        store.data
    );

    // The Rust RO connection must see the bridge child's committed write.
    let recall = exec.query("RECALL FROM SEMANTIC ALL LIMIT 50").unwrap();
    assert!(recall.success);
    let found = recall
        .data
        .iter()
        .any(|row| row["text"].as_str() == Some("terraform manages cloud infrastructure"));
    assert!(
        found,
        "RECALL should see the stored chunk; got {:?}",
        recall.data
    );
}

#[test]
fn stored_chunk_is_semantically_searchable() {
    if bridge_unavailable("semantic_search") {
        return;
    }
    let db = make_engram();
    let exec = Executor::open(db.path()).unwrap();

    exec.query(r#"STORE INTO SEMANTIC (text = "terraform provisions cloud infrastructure")"#)
        .unwrap();
    exec.query(r#"STORE INTO SEMANTIC (text = "a recipe for chocolate cake")"#)
        .unwrap();

    // A string-bound LIKE embeds the query through the bridge and ranks by
    // cosine distance — this only works if STORE embedded the chunk.
    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert(
        "q".to_string(),
        json!("infrastructure provisioning tooling"),
    );
    let recall = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q", vars)
        .unwrap();
    assert!(recall.success, "recall failed: {:?}", recall.error);
    assert!(recall.count >= 2, "expected both stored chunks ranked");
    assert!(
        recall.data[0]["text"]
            .as_str()
            .unwrap_or("")
            .contains("terraform"),
        "infrastructure query should rank the terraform chunk first; got {:?}",
        recall.data[0]["text"]
    );
}

#[test]
fn forget_drops_chunk_from_recall() {
    if bridge_unavailable("forget") {
        return;
    }
    let db = make_engram();
    let exec = Executor::open(db.path()).unwrap();

    let store = exec
        .query(r#"STORE INTO SEMANTIC (text = "ephemeral note to be forgotten")"#)
        .unwrap();
    let id = store.data[0]["chunkId"].as_str().unwrap().to_string();

    let forget = exec
        .query(&format!(r#"FORGET FROM SEMANTIC WHERE id = "{id}""#))
        .unwrap();
    assert!(forget.success, "FORGET failed: {:?}", forget.error);

    let recall = exec.query("RECALL FROM SEMANTIC ALL LIMIT 50").unwrap();
    let still_present = recall
        .data
        .iter()
        .any(|row| row["id"].as_str() == Some(&id));
    assert!(!still_present, "forgotten chunk must not appear in recall");
}

#[test]
fn update_supersedes_chunk() {
    if bridge_unavailable("update") {
        return;
    }
    let db = make_engram();
    let exec = Executor::open(db.path()).unwrap();

    let store = exec
        .query(r#"STORE INTO SEMANTIC (text = "old fact: the server runs on port 8080")"#)
        .unwrap();
    let id = store.data[0]["chunkId"].as_str().unwrap().to_string();

    let update = exec
        .query(&format!(
            r#"UPDATE INTO SEMANTIC WHERE id = "{id}" (text = "new fact: the server runs on port 9090")"#
        ))
        .unwrap();
    assert!(update.success, "UPDATE failed: {:?}", update.error);

    let recall = exec.query("RECALL FROM SEMANTIC ALL LIMIT 50").unwrap();
    let texts: Vec<&str> = recall
        .data
        .iter()
        .filter_map(|row| row["text"].as_str())
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("port 9090")),
        "superseding text should be present; got {texts:?}"
    );
    assert!(
        texts.iter().all(|t| !t.contains("port 8080")),
        "superseded (old) chunk should be soft-deleted; got {texts:?}"
    );
}
