//! SCAN statement handler. Reads from working_memory (the only memory type SCAN supports).

use aql_parser::ast::{ScanStmt, Window};
use rusqlite::Connection;
use serde_json::{Map, Value as JsonValue};

use crate::error::AqlResult;
use crate::result::QueryResult;
use crate::sql::serialize::rusqlite_to_json;

pub fn execute(conn: &Connection, stmt: &ScanStmt) -> AqlResult<QueryResult> {
    let mut limit: usize = 10;
    let mut after_datetime: Option<String> = None;

    if let Some(w) = &stmt.window {
        match w {
            Window::LastN { count } => {
                limit = (*count).min(1000);
            }
            Window::LastDuration { duration } => {
                let secs = duration.as_secs();
                after_datetime = Some(format!("datetime('now', '-{} seconds')", secs));
            }
            Window::TopBy { count, field: _ } => {
                limit = (*count).min(1000);
            }
            Window::Since { .. } => {
                // SINCE with condition — Phase 1 treats as default active-sessions filter.
                // A future phase could translate the condition to a SQL WHERE predicate.
            }
        }
    }
    if let Some(n) = stmt.modifiers.limit {
        limit = n.min(1000);
    }

    let mut where_parts: Vec<String> =
        vec!["(expires_at IS NULL OR expires_at > datetime('now'))".into()];
    if let Some(clause) = &after_datetime {
        where_parts.push(format!("updated_at > {}", clause));
    }

    let sql = format!(
        "SELECT * FROM working_memory WHERE {} ORDER BY updated_at DESC LIMIT {}",
        where_parts.join(" AND "),
        limit
    );

    let mut prepared = conn.prepare(&sql)?;
    let column_names: Vec<String> = prepared
        .column_names()
        .into_iter()
        .map(String::from)
        .collect();

    let rows = prepared.query_map([], |row| {
        let mut map = Map::new();
        for (i, name) in column_names.iter().enumerate() {
            let v = row.get::<_, rusqlite::types::Value>(i)?;
            map.insert(name.clone(), rusqlite_to_json(v));
        }
        Ok(JsonValue::Object(map))
    })?;

    let mut data = Vec::new();
    for r in rows {
        data.push(r?);
    }

    Ok(QueryResult::success("Scan", data))
}
