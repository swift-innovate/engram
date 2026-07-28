//! SQL builder unit tests.

mod common;

use aql_parser::ast::{Condition, LogicalOp, Operator, Value};
use engram_aql::memory_map::EngramTable;
use engram_aql::sql::conditions::condition_to_sql;
use engram_aql::sql::fields::{resolve_field, FieldRef};
use rusqlite::types::Value as RusqValue;

#[test]
fn simple_eq_string() {
    let cond = Condition::Simple {
        field: "context".into(),
        operator: Operator::Eq,
        value: Value::String("ops".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "context = ?");
    assert_eq!(params.len(), 1);
    assert!(matches!(params[0], RusqValue::Text(ref s) if s == "ops"));
}

#[test]
fn simple_gt_float() {
    let cond = Condition::Simple {
        field: "trust_score".into(),
        operator: Operator::Gt,
        value: Value::Float(0.75),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "trust_score > ?");
}

#[test]
fn json_field_uses_json_extract() {
    // "outcome" is not a direct column on chunks; should translate to json_extract
    let cond = Condition::Simple {
        field: "outcome".into(),
        operator: Operator::Eq,
        value: Value::String("success".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "json_extract(text, '$.outcome') = ?");
}

#[test]
fn contains_becomes_like_wrapped() {
    let cond = Condition::Simple {
        field: "text".into(),
        operator: Operator::Contains,
        value: Value::String("deploy".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "text LIKE ? ESCAPE '\\'");
    assert!(matches!(params[0], RusqValue::Text(ref s) if s == "%deploy%"));
}

#[test]
fn starts_with_anchors_at_start() {
    let cond = Condition::Simple {
        field: "name".into(),
        operator: Operator::StartsWith,
        value: Value::String("k8s".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Tools, &mut params);
    assert_eq!(sql, "name LIKE ? ESCAPE '\\'");
    assert!(matches!(params[0], RusqValue::Text(ref s) if s == "k8s%"));
}

#[test]
fn in_operator_expands_array() {
    let cond = Condition::Simple {
        field: "status".into(),
        operator: Operator::In,
        value: Value::Array(vec![
            Value::String("success".into()),
            Value::String("partial".into()),
        ]),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "json_extract(text, '$.status') IN (?, ?)");
    assert_eq!(params.len(), 2);
}

#[test]
fn group_with_and() {
    // (context = "ops" AND trust_score > 0.8)
    let inner1 = Condition::Simple {
        field: "context".into(),
        operator: Operator::Eq,
        value: Value::String("ops".into()),
        logical_op: None,
    };
    let inner2 = Condition::Simple {
        field: "trust_score".into(),
        operator: Operator::Gt,
        value: Value::Float(0.8),
        logical_op: Some(LogicalOp::And),
    };
    let group = Condition::Group {
        conditions: vec![inner1, inner2],
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&group, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "(context = ? AND trust_score > ?)");
    assert_eq!(params.len(), 2);
}

#[test]
fn group_with_or() {
    let inner1 = Condition::Simple {
        field: "context".into(),
        operator: Operator::Eq,
        value: Value::String("ops".into()),
        logical_op: None,
    };
    let inner2 = Condition::Simple {
        field: "context".into(),
        operator: Operator::Eq,
        value: Value::String("ci".into()),
        logical_op: Some(LogicalOp::Or),
    };
    let group = Condition::Group {
        conditions: vec![inner1, inner2],
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&group, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "(context = ? OR context = ?)");
}

#[test]
fn ends_with_anchors_at_end() {
    let cond = Condition::Simple {
        field: "name".into(),
        operator: Operator::EndsWith,
        value: Value::String("_prod".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Tools, &mut params);
    assert_eq!(sql, "name LIKE ? ESCAPE '\\'");
    assert!(matches!(params[0], RusqValue::Text(ref s) if s == r"%\_prod"));
}

#[test]
fn unsafe_json_path_field_is_unresolvable() {
    // A field name containing characters outside [A-Za-z0-9_.-] must NOT be
    // interpolated into a json_extract expression. `resolve_field` returns
    // `Unresolvable`, which `to_sql()` renders as the literal `NULL`, so a
    // query using such a field returns zero rows instead of injecting SQL.
    for poisoned in &[
        "outcome'; DROP TABLE chunks; --",
        "outcome OR 1=1",
        "outcome\"",
        "outcome;",
        "",
    ] {
        let field = resolve_field(poisoned, EngramTable::Chunks);
        assert!(
            matches!(field, FieldRef::Unresolvable { .. }),
            "expected Unresolvable for poisoned field {:?}, got safe variant",
            poisoned
        );
        assert_eq!(field.to_sql(), "NULL");
    }
}

#[test]
fn unsafe_json_path_in_query_returns_zero_rows() {
    // End-to-end guard: building a WHERE clause from a poisoned field name
    // and executing it against the seeded database must produce zero rows
    // (NULL = 'anything' is NULL, never TRUE) — not a SQL error, and not an
    // injected result. The seeded database has rows where outcome = 'success',
    // so if the guard failed and the field leaked through, we would see > 0.
    let conn = common::seeded_db();

    let cond = Condition::Simple {
        field: "outcome'; DROP TABLE chunks; --".into(),
        operator: Operator::Eq,
        value: Value::String("success".into()),
        logical_op: None,
    };
    let mut params: Vec<RusqValue> = Vec::new();
    let where_sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);

    // Sanity: the rendered WHERE clause must not contain the malicious payload.
    assert!(
        !where_sql.contains("DROP"),
        "malicious field leaked into SQL: {}",
        where_sql
    );

    let sql = format!("SELECT COUNT(*) FROM chunks WHERE {}", where_sql);
    let count: i64 = conn
        .query_row(&sql, rusqlite::params_from_iter(params.iter()), |row| {
            row.get(0)
        })
        .expect("query must execute cleanly, not SQL-error");
    assert_eq!(count, 0, "poisoned field must match zero rows");

    // And confirm the table still exists (i.e. no injection fired).
    let chunks_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();
    assert!(chunks_rows > 0, "chunks table should be intact");
}

#[test]
fn contains_escapes_percent_and_underscore() {
    let cond = Condition::Simple {
        field: "text".into(),
        operator: Operator::Contains,
        value: Value::String("100% progress_bar".into()),
        logical_op: None,
    };
    let mut params = Vec::new();
    let sql = condition_to_sql(&cond, EngramTable::Chunks, &mut params);
    assert_eq!(sql, "text LIKE ? ESCAPE '\\'");
    // The param should have escaped % and _
    assert!(
        matches!(params[0], RusqValue::Text(ref s) if s == r"%100\% progress\_bar%"),
        "unexpected param: {:?}",
        params[0]
    );
}
