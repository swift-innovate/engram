//! RECALL LIKE / PATTERN vector search tests.
//!
//! # Array-bound tests (no bridge, always run)
//!
//! Seed a fresh in-memory DB with chunks carrying known small embeddings
//! stored as LE-f32 BLOBs, then query with a pre-computed probe array via
//! `query_with_vars`. No `engram-mcp` child is needed.
//!
//! # String-bound test (gated)
//!
//! Requires `engram-mcp` to be discoverable (set
//! `ENGRAM_MCP_CMD="node /abs/path/to/dist/mcp-server.js"`). Embeds two
//! strings via the bridge, inserts chunks with those vectors, and asserts
//! that a `LIKE $q` query with a string variable returns the matching chunk
//! first.

mod common;

use std::collections::BTreeMap;

use engram_aql::vector::codec::encode_f32_le;
use engram_aql::Executor;
use rusqlite::Connection;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Seed the DB with chunks whose embeddings are tiny known vectors.
///
/// - e1 = [1, 0, 0, 0]   (unit vector along axis 0)
/// - e2 = [0, 1, 0, 0]   (unit vector along axis 1 — most distant from e1)
/// - e3 = [0.9, 0.1, 0, 0]  (close to e1, closer than e2)
///
/// Cosine distance to probe [1, 0, 0, 0]:
///   e1: 0.0   (identical)
///   e3: ≈ 0.005  (very close)
///   e2: 1.0   (orthogonal)
///
/// Expected RECALL order by distance ASC: e1, e3, e2.
fn seed_vector_chunks(conn: &Connection) {
    let e1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let e2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
    let e3: [f32; 4] = [0.9, 0.1, 0.0, 0.0];

    conn.execute(
        "INSERT INTO chunks (id, text, embedding, memory_type, source_type, is_active) \
         VALUES ('e1', 'axis-0 chunk', ?, 'world', 'user_stated', 1)",
        rusqlite::params![encode_f32_le(&e1)],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks (id, text, embedding, memory_type, source_type, is_active) \
         VALUES ('e2', 'axis-1 chunk', ?, 'world', 'user_stated', 1)",
        rusqlite::params![encode_f32_le(&e2)],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks (id, text, embedding, memory_type, source_type, is_active) \
         VALUES ('e3', 'near-axis-0 chunk', ?, 'world', 'user_stated', 1)",
        rusqlite::params![encode_f32_le(&e3)],
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Array-bound tests (always run — no bridge required)
// ---------------------------------------------------------------------------

/// `RECALL FROM SEMANTIC LIKE $q` with a precomputed array probe returns rows
/// ordered by cosine distance: e1 (exact match), e3 (near), e2 (orthogonal).
#[test]
fn like_array_probe_orders_by_distance() {
    let conn = common::fresh_db();
    seed_vector_chunks(&conn);
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, 0.0, 0.0, 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q", vars)
        .unwrap();

    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 3, "expected all 3 chunks");

    let ids: Vec<&str> = result
        .data
        .iter()
        .filter_map(|row| row.get("id").and_then(|v| v.as_str()))
        .collect();

    assert_eq!(
        ids[0], "e1",
        "e1 (exact match) must rank first; got {:?}",
        ids
    );
    assert_eq!(ids[1], "e3", "e3 (near) must rank second; got {:?}", ids);
    assert_eq!(
        ids[2], "e2",
        "e2 (orthogonal) must rank last; got {:?}",
        ids
    );
}

/// Distance column is present in vector search results.
#[test]
fn like_result_includes_distance_column() {
    let conn = common::fresh_db();
    seed_vector_chunks(&conn);
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, 0.0, 0.0, 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q LIMIT 1", vars)
        .unwrap();

    assert!(result.success, "error: {:?}", result.error);
    let row = &result.data[0];
    assert!(
        row.get("distance").is_some(),
        "vector result must include `distance` column; got {:?}",
        row
    );
    // e1 is an exact match — distance should be 0 (or very close due to f32
    // rounding through LE encoding/decoding).
    let dist = row["distance"].as_f64().unwrap();
    assert!(
        dist < 1e-4,
        "distance for exact match should be ~0; got {}",
        dist
    );
}

/// `PATTERN $q THRESHOLD 0.95` filters out e2 (distance ~1.0 from axis-0).
///
/// Threshold 0.95 = similarity floor → distance ceiling = 1.0 - 0.95 = 0.05.
/// e1: dist ≈ 0.0  → passes
/// e3: dist ≈ 0.005 → passes
/// e2: dist ≈ 1.0  → filtered out
#[test]
fn pattern_threshold_filters_distant_chunks() {
    let conn = common::fresh_db();
    seed_vector_chunks(&conn);
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, 0.0, 0.0, 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC PATTERN $q THRESHOLD 0.95", vars)
        .unwrap();

    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(
        result.count,
        2,
        "only e1 and e3 should pass THRESHOLD 0.95; got {:?}",
        result
            .data
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
    );

    let ids: Vec<&str> = result
        .data
        .iter()
        .filter_map(|row| row.get("id").and_then(|v| v.as_str()))
        .collect();

    // e2 must be absent
    assert!(
        !ids.contains(&"e2"),
        "e2 (orthogonal) must be filtered by THRESHOLD 0.95; got {:?}",
        ids
    );
}

/// Unbound variable produces a clear error (not a panic).
#[test]
fn like_unbound_variable_returns_error() {
    let conn = common::fresh_db();
    let exec = Executor::from_connection(conn).unwrap();

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $missing", BTreeMap::new())
        .unwrap();

    assert!(
        !result.success,
        "expected error result for unbound variable"
    );
    let err = result.error.unwrap_or_default();
    assert!(
        err.contains("$missing") || err.contains("missing"),
        "error should mention the variable name; got: {err}"
    );
}

/// Non-number array element produces a clear error.
#[test]
fn like_bad_array_element_returns_error() {
    let conn = common::fresh_db();
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, "bad", 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q", vars)
        .unwrap();

    assert!(
        !result.success,
        "expected error for non-number array element"
    );
    let err = result.error.unwrap_or_default();
    assert!(
        err.contains("not a number") || err.contains("number"),
        "error should mention number type constraint; got: {err}"
    );
}

/// Wrong variable type (object) produces a clear error.
#[test]
fn like_object_variable_returns_error() {
    let conn = common::fresh_db();
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!({"not": "a vector"}));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q", vars)
        .unwrap();

    assert!(!result.success, "expected error for object variable");
}

/// AGGREGATE with LIKE returns a warning and empty data (not an error/panic).
#[test]
fn aggregate_with_like_returns_warning_empty() {
    let conn = common::fresh_db();
    seed_vector_chunks(&conn);
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, 0.0, 0.0, 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q AGGREGATE COUNT(*) AS n", vars)
        .unwrap();

    // Should succeed (not error) but return empty data + a warning
    assert!(
        result.success,
        "AGGREGATE+LIKE should succeed (empty result); error: {:?}",
        result.error
    );
    assert_eq!(result.count, 0, "AGGREGATE+LIKE should return empty data");
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("AGGREGATE") || w.contains("vector")),
        "should warn about AGGREGATE+LIKE unsupported; warnings: {:?}",
        result.warnings
    );
}

/// `LIKE`/`PATTERN` on a non-chunks memory type (PROCEDURAL → observations,
/// which has no embeddings) must warn + return empty rather than silently
/// scanning the chunks table.
#[test]
fn like_on_non_chunks_memory_type_warns_empty() {
    let conn = common::fresh_db();
    seed_vector_chunks(&conn); // chunks exist, but PROCEDURAL must not return them
    let exec = Executor::from_connection(conn).unwrap();

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!([1.0, 0.0, 0.0, 0.0]));

    let result = exec
        .query_with_vars("RECALL FROM PROCEDURAL LIKE $q", vars)
        .unwrap();

    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(
        result.count, 0,
        "PROCEDURAL LIKE must not leak chunks rows; got {}",
        result.count
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("SEMANTIC/EPISODIC")
                || w.contains("no embeddings")
                || w.contains("stores none")),
        "should warn vector search is chunks-only; warnings: {:?}",
        result.warnings
    );
}

// ---------------------------------------------------------------------------
// String-bound test (gated — requires engram-mcp)
// ---------------------------------------------------------------------------

/// Verify that string-bound LIKE $q embeds the query text through the bridge
/// and returns the most similar chunk first.
///
/// **Gated:** skipped when `ENGRAM_MCP_CMD` is not set and `engram-mcp` is not
/// on PATH. Mirrors `tests/bridge_embed.rs`.
#[test]
fn like_string_probe_via_bridge_ranks_matching_chunk_first() {
    use engram_aql::bridge::child::resolve_engram_mcp_cmd;
    use engram_aql::bridge::Bridge;
    use tempfile::NamedTempFile;

    // ── gate ────────────────────────────────────────────────────────────────
    let cmd = resolve_engram_mcp_cmd(None);
    if cmd.is_none() {
        eprintln!(
            "[vector_search] SKIP string-bound test — engram-mcp not discoverable. \
             Set ENGRAM_MCP_CMD=\"node /abs/path/to/dist/mcp-server.js\" to run."
        );
        return;
    }
    let cmd = cmd.unwrap();

    // ── setup: temp .engram file ─────────────────────────────────────────────
    let db_file = NamedTempFile::with_suffix(".engram").unwrap();
    let db_path = db_file.path();

    // Seed schema + chunks with bridge-generated embeddings
    {
        let conn = Connection::open(db_path).unwrap();
        conn.execute_batch(common::SCHEMA_SQL).unwrap();

        let mut bridge = Bridge::new(cmd.clone(), db_path).expect("bridge should start");

        let alpha_vec = bridge.embed_query("alpha").expect("embed alpha");
        let zebra_vec = bridge.embed_query("zebra unrelated").expect("embed zebra");

        conn.execute(
            "INSERT INTO chunks (id, text, embedding, memory_type, source_type, is_active) \
             VALUES ('alpha-chunk', 'alpha content', ?, 'world', 'user_stated', 1)",
            rusqlite::params![encode_f32_le(&alpha_vec)],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO chunks (id, text, embedding, memory_type, source_type, is_active) \
             VALUES ('zebra-chunk', 'zebra content', ?, 'world', 'user_stated', 1)",
            rusqlite::params![encode_f32_le(&zebra_vec)],
        )
        .unwrap();
    }

    // ── query: open executor on the temp file and run LIKE $q ────────────────
    let exec = Executor::open(db_path).expect("executor should open");

    let mut vars: BTreeMap<String, Value> = BTreeMap::new();
    vars.insert("q".to_string(), json!("alpha"));

    let result = exec
        .query_with_vars("RECALL FROM SEMANTIC LIKE $q", vars)
        .expect("query_with_vars should succeed");

    assert!(
        result.success,
        "string-bound LIKE should succeed; error: {:?}",
        result.error
    );
    assert_eq!(result.count, 2, "should return both chunks");

    let first_id = result.data[0]
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("(none)");

    assert_eq!(
        first_id,
        "alpha-chunk",
        "alpha-chunk must rank first when querying for 'alpha'; got {:?}",
        result
            .data
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
    );
}
