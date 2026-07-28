//! LOOKUP integration tests.

mod common;

use engram_aql::Executor;

#[test]
fn lookup_by_id() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"LOOKUP FROM EPISODIC KEY id = "e-001""#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 1);
}

#[test]
fn lookup_nonexistent_returns_empty() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"LOOKUP FROM EPISODIC KEY id = "nonexistent""#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 0);
}

#[test]
fn lookup_with_return_fields() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"LOOKUP FROM EPISODIC KEY id = "e-001" RETURN id, trust_score"#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 1);
    let row = &result.data[0];
    assert!(row.get("id").is_some());
    assert!(row.get("trust_score").is_some());
    assert!(
        row.get("text").is_none(),
        "text should be filtered out by RETURN"
    );
}
