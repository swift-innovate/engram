//! WITH LINKS and FOLLOW LINKS integration tests.

mod common;

use engram_aql::Executor;

#[test]
fn with_links_attaches_link_metadata() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    // e-001 is a deploy chunk linked via chunk_entities to ent-deploy,
    // which has a relation 'uses_pattern' to ent-bluegreen
    let result = exec
        .query(r#"RECALL FROM EPISODIC KEY id = "e-001" WITH LINKS ALL"#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 1);
    // Should have link metadata attached from the relations table
    assert!(
        result.links.is_some(),
        "expected links metadata to be attached"
    );
    let links = result.links.as_ref().unwrap();
    assert!(!links.is_empty(), "expected at least one link");
}

#[test]
fn with_links_type_filter_limits_to_matching_type() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"RECALL FROM EPISODIC KEY id = "e-001" WITH LINKS TYPE "uses_pattern""#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert!(
        result.links.is_some(),
        "WITH LINKS TYPE should populate links field (even if empty)"
    );
    let links = result.links.as_ref().unwrap();
    // All returned links should match the requested type
    for link in links {
        assert_eq!(link.link_type, "uses_pattern");
    }
}

#[test]
fn follow_links_expands_to_related_chunks() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    // e-001 (episodic) → ent-deploy → uses_pattern → ent-bluegreen → s-001 (semantic)
    let result = exec
        .query(r#"RECALL FROM EPISODIC KEY id = "e-001" FOLLOW LINKS TYPE "uses_pattern" DEPTH 1"#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    // Should return the base e-001 plus s-001 reached via graph traversal
    let ids: Vec<String> = result
        .data
        .iter()
        .filter_map(|row| row.get("id").and_then(|v| v.as_str()).map(String::from))
        .collect();
    assert!(
        ids.contains(&"e-001".to_string()),
        "base chunk e-001 missing from result: {:?}",
        ids
    );
    assert!(
        ids.contains(&"s-001".to_string()),
        "expected FOLLOW LINKS to reach s-001 via uses_pattern, got: {:?}",
        ids
    );
    // The implementer noted that FOLLOW LINKS intentionally crosses memory
    // types — s-001 is a "world" chunk but still appears when followed from
    // an episodic query. This is documented behavior per Task 10.
    assert_eq!(result.count, 2);
}

#[test]
fn follow_links_crosses_memory_types() {
    // Document the Task 10 behavior: FOLLOW LINKS traverses the entity graph
    // regardless of which memory type each reached chunk lives in. A query
    // that starts from EPISODIC can return SEMANTIC chunks when they share
    // entities via an active relation.
    //
    // If Phase 2 decides this is wrong, this test will fail and force a
    // conscious decision to change the behavior.
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"RECALL FROM EPISODIC KEY id = "e-001" FOLLOW LINKS TYPE "uses_pattern" DEPTH 1"#)
        .unwrap();
    assert!(result.success);

    // Find s-001 in the results and verify it's a "world" chunk
    let s001 = result
        .data
        .iter()
        .find(|row| row.get("id").and_then(|v| v.as_str()) == Some("s-001"))
        .expect("s-001 should be in the expanded results");
    assert_eq!(
        s001.get("memory_type").and_then(|v| v.as_str()),
        Some("world"),
        "s-001 should be a world chunk (semantic memory type)"
    );
}

#[test]
fn no_graph_modifiers_returns_no_links_field() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec.query("RECALL FROM EPISODIC ALL LIMIT 1").unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert!(
        result.links.is_none(),
        "plain RECALL should not populate links"
    );
}
