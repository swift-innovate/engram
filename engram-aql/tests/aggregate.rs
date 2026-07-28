//! AGGREGATE + HAVING integration tests.

mod common;

use engram_aql::Executor;

#[test]
fn count_all_episodic() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query("RECALL FROM EPISODIC ALL AGGREGATE COUNT(*) AS total")
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let total = result.data[0]
        .get("total")
        .and_then(|v| v.as_i64())
        .unwrap();
    assert_eq!(total, 4); // 4 episodic records in seed
}

#[test]
fn count_filtered_by_context() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(r#"RECALL FROM EPISODIC WHERE context = "ops" AGGREGATE COUNT(*) AS total"#)
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let total = result.data[0]
        .get("total")
        .and_then(|v| v.as_i64())
        .unwrap();
    assert_eq!(total, 3);
}

#[test]
fn avg_trust_score() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query("RECALL FROM EPISODIC ALL AGGREGATE AVG(trust_score) AS avg_trust")
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let avg = result.data[0]
        .get("avg_trust")
        .and_then(|v| v.as_f64())
        .unwrap();
    // (0.9 + 0.7 + 0.8 + 0.85) / 4 = 0.8125
    assert!((avg - 0.8125).abs() < 0.0001, "avg was: {}", avg);
}

#[test]
fn min_max_trust_score() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(
            "RECALL FROM EPISODIC ALL AGGREGATE MIN(trust_score) AS min_t, MAX(trust_score) AS max_t",
        )
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let min = result.data[0]
        .get("min_t")
        .and_then(|v| v.as_f64())
        .unwrap();
    let max = result.data[0]
        .get("max_t")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!((min - 0.7).abs() < 0.001);
    assert!((max - 0.9).abs() < 0.001);
}

#[test]
fn having_filters_aggregate_results() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(
            "RECALL FROM EPISODIC WHERE context = \"ops\" \
             AGGREGATE AVG(trust_score) AS avg_t HAVING avg_t > 0.75",
        )
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 1);
}

#[test]
fn having_excludes_non_matching() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query(
            "RECALL FROM EPISODIC WHERE context = \"ops\" \
             AGGREGATE AVG(trust_score) AS avg_t HAVING avg_t > 0.99",
        )
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    assert_eq!(result.count, 0);
}

#[test]
fn sum_trust_score() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    let result = exec
        .query("RECALL FROM EPISODIC ALL AGGREGATE SUM(trust_score) AS total_trust")
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let sum = result.data[0]
        .get("total_trust")
        .and_then(|v| v.as_f64())
        .unwrap();
    // 0.9 + 0.7 + 0.8 + 0.85 = 3.25
    assert!((sum - 3.25).abs() < 0.0001, "sum was: {}", sum);
}

#[test]
fn count_on_named_field() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    // COUNT on a non-* field. Returns count of non-null values.
    let result = exec
        .query("RECALL FROM EPISODIC ALL AGGREGATE COUNT(trust_score) AS n")
        .unwrap();
    assert!(result.success, "error: {:?}", result.error);
    let n = result.data[0].get("n").and_then(|v| v.as_i64()).unwrap();
    assert_eq!(n, 4);
}

#[test]
fn having_with_unsupported_operator_errors() {
    let conn = common::seeded_db();
    let exec = Executor::from_connection(conn).unwrap();
    // HAVING with CONTAINS should return an error result (not silently pass)
    let result = exec
        .query("RECALL FROM EPISODIC ALL AGGREGATE COUNT(*) AS total HAVING total CONTAINS \"2\"")
        .unwrap();
    assert!(!result.success);
    let err = result.error.unwrap();
    assert!(
        err.contains("HAVING") || err.contains("operator"),
        "unexpected error: {}",
        err
    );
}
