//! Write-statement delegation (Phase 2b).
//!
//! Rust never writes the `.engram` file itself. Each AQL write statement is
//! translated into a call to the canonical TypeScript retain pipeline over the
//! `engram-mcp` bridge:
//!
//! | AQL      | TS tool             | Notes                                        |
//! |----------|---------------------|----------------------------------------------|
//! | `STORE`  | `engram_retain`     | payload fields → retain options (pass-through provenance) |
//! | `UPDATE` | `engram_supersede`  | resolve target ids via a RO read, supersede each |
//! | `FORGET` | `engram_forget`     | resolve target ids via a RO read, forget each |
//! | `REFLECT`| `engram_reflect`    | global cycle; source/THEN filters not yet honored |
//!
//! `LINK` stays rejected (no canonical TS manual-relation surface — see the
//! spec's Open Questions). Writes target chunk-backed memory only
//! (`SEMANTIC`/`EPISODIC`); other memory types are rejected with a clear error.

use std::collections::BTreeMap;

use aql_parser::ast::{
    Condition, ForgetStmt, MemoryType, ReflectStmt, StoreStmt, UpdateStmt, Value as AqlValue,
};
use rusqlite::types::Value as RusqValue;
use serde_json::{json, Map, Value as JsonValue};

use crate::error::{AqlError, AqlResult};
use crate::exec_ctx::ExecCtx;
use crate::memory_map::EngramTable;
use crate::result::QueryResult;
use crate::sql::conditions::condition_to_sql;

/// `STORE INTO <type> field = value, ...` → `engram_retain`.
pub fn store(ctx: &ExecCtx<'_>, stmt: &StoreStmt) -> AqlResult<QueryResult> {
    let memory_type = chunk_write_type(stmt.memory_type)?;

    let mut args = Map::new();
    args.insert("memoryType".to_string(), json!(memory_type));
    let mut warnings: Vec<String> = Vec::new();
    let mut has_text = false;

    for fa in &stmt.payload {
        let value = aql_value_to_json(&fa.value, ctx.vars)?;
        match fa.field.as_str() {
            "text" | "content" => {
                args.insert("text".to_string(), value);
                has_text = true;
            }
            other => match retain_option_name(other) {
                Some(opt) => {
                    args.insert(opt.to_string(), value);
                }
                None => warnings.push(format!("STORE: ignored unknown field `{other}`")),
            },
        }
    }

    if !has_text {
        return Err(AqlError::InvalidQuery(
            "STORE requires a `text` field, e.g. STORE INTO SEMANTIC text = \"...\"".to_string(),
        ));
    }

    // Provenance is pass-through: unspecified sourceType/trustScore inherit
    // engram_retain's own defaults (inferred / 0.5).
    let payload = ctx
        .bridge
        .call_tool("engram_retain", JsonValue::Object(args))?;
    let mut result = QueryResult::success("Store", vec![payload]);
    result.warnings = warnings;
    Ok(result)
}

/// `UPDATE <type> WHERE ... SET text = ...` → `engram_supersede` per matched id.
pub fn update(ctx: &ExecCtx<'_>, stmt: &UpdateStmt) -> AqlResult<QueryResult> {
    // Extract the new text (required) and any optional provenance overrides.
    let mut new_text: Option<JsonValue> = None;
    let mut extra = Map::new();
    let mut warnings: Vec<String> = Vec::new();
    for fa in &stmt.payload {
        let value = aql_value_to_json(&fa.value, ctx.vars)?;
        match fa.field.as_str() {
            "text" | "content" => new_text = Some(value),
            other => match retain_option_name(other) {
                Some(opt) => {
                    extra.insert(opt.to_string(), value);
                }
                None => warnings.push(format!("UPDATE: ignored unknown field `{other}`")),
            },
        }
    }
    let new_text = new_text.ok_or_else(|| {
        AqlError::InvalidQuery(
            "UPDATE requires a `text` field to supersede with, e.g. UPDATE SEMANTIC WHERE ... SET text = \"...\""
                .to_string(),
        )
    })?;

    let (ids, truncated) = resolve_chunk_ids(ctx, stmt.memory_type, &stmt.conditions)?;
    let mut data = Vec::with_capacity(ids.len());
    for id in &ids {
        let mut args = extra.clone();
        args.insert("oldChunkId".to_string(), json!(id));
        args.insert("newText".to_string(), new_text.clone());
        data.push(
            ctx.bridge
                .call_tool("engram_supersede", JsonValue::Object(args))?,
        );
    }

    let mut result = QueryResult::success("Update", data);
    if ids.is_empty() {
        warnings.push("UPDATE matched no active chunks; nothing superseded".to_string());
    } else if truncated {
        warnings.push(truncation_warning("UPDATE", "superseded"));
    }
    result.warnings = warnings;
    Ok(result)
}

/// `FORGET FROM <type> WHERE ...` → `engram_forget` per matched id.
pub fn forget(ctx: &ExecCtx<'_>, stmt: &ForgetStmt) -> AqlResult<QueryResult> {
    let (ids, truncated) = resolve_chunk_ids(ctx, stmt.memory_type, &stmt.conditions)?;
    let mut data = Vec::with_capacity(ids.len());
    for id in &ids {
        data.push(
            ctx.bridge
                .call_tool("engram_forget", json!({ "chunkId": id }))?,
        );
    }

    let mut result = QueryResult::success("Forget", data);
    if ids.is_empty() {
        result
            .warnings
            .push("FORGET matched no active chunks; nothing forgotten".to_string());
    } else if truncated {
        result
            .warnings
            .push(truncation_warning("FORGET", "forgotten"));
    }
    Ok(result)
}

/// `REFLECT ...` → `engram_reflect` (global cycle).
pub fn reflect(ctx: &ExecCtx<'_>, stmt: &ReflectStmt) -> AqlResult<QueryResult> {
    let payload = ctx.bridge.call_tool("engram_reflect", json!({}))?;
    let mut result = QueryResult::success("Reflect", vec![payload]);
    if !stmt.sources.is_empty() || stmt.then_clause.is_some() {
        result.warnings.push(
            "REFLECT runs a global reflection cycle over all unreflected memories; \
             source filters and THEN clauses are not yet honored."
                .to_string(),
        );
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an AQL memory type to the `chunks.memory_type` value used for writes.
/// Writes go through `engram_retain`/`_supersede`/`_forget`, which all operate
/// on the chunks table, so only chunk-backed types are valid.
fn chunk_write_type(mt: MemoryType) -> AqlResult<&'static str> {
    match mt {
        MemoryType::Semantic => Ok("world"),
        MemoryType::Episodic => Ok("experience"),
        other => Err(AqlError::InvalidQuery(format!(
            "AQL writes target chunk-backed memory only (SEMANTIC or EPISODIC); \
             {other:?} is not writable via engram-aql. Use the TypeScript tools for observations/working memory."
        ))),
    }
}

/// Map an AQL payload field name (snake_case) to its engram tool option name
/// (camelCase). Returns `None` for the text field and unknown fields.
fn retain_option_name(field: &str) -> Option<&'static str> {
    match field {
        "source" => Some("source"),
        "context" => Some("context"),
        "source_type" | "sourceType" => Some("sourceType"),
        "trust_score" | "trustScore" => Some("trustScore"),
        "event_time" | "eventTime" => Some("eventTime"),
        "temporal_label" | "temporalLabel" => Some("temporalLabel"),
        "memory_type" | "memoryType" => Some("memoryType"),
        _ => None,
    }
}

/// Upper bound on how many chunks a single `UPDATE`/`FORGET` will target.
/// Matches > this many are truncated (ordered by `id` for determinism) and
/// the caller attaches a warning rather than silently dropping the rest.
const WRITE_TARGET_LIMIT: usize = 1000;

/// Resolve the active chunk ids matching `conditions` for a write target.
/// Read-only — the resolved ids are then handed to the TS tools to mutate.
///
/// Returns the (possibly truncated) id list plus whether truncation occurred,
/// so callers can warn instead of reporting silent partial completion.
fn resolve_chunk_ids(
    ctx: &ExecCtx<'_>,
    memory_type: MemoryType,
    conditions: &[Condition],
) -> AqlResult<(Vec<String>, bool)> {
    let chunk_type = chunk_write_type(memory_type)?;

    let mut params: Vec<RusqValue> = vec![RusqValue::Text(chunk_type.to_string())];
    let mut where_parts: Vec<String> =
        vec!["is_active = 1".to_string(), "memory_type = ?".to_string()];
    for cond in conditions {
        where_parts.push(condition_to_sql(cond, EngramTable::Chunks, &mut params));
    }

    // Fetch one row past the cap so we can detect (rather than silently
    // produce) truncation; ORDER BY id keeps which rows get dropped stable.
    let sql = format!(
        "SELECT id FROM chunks WHERE {} ORDER BY id LIMIT {}",
        where_parts.join(" AND "),
        WRITE_TARGET_LIMIT + 1
    );
    let mut prepared = ctx.conn.prepare(&sql)?;
    let rows = prepared.query_map(rusqlite::params_from_iter(params.iter()), |row| {
        row.get::<_, String>(0)
    })?;

    let mut ids = Vec::new();
    for r in rows {
        ids.push(r?);
    }

    let truncated = ids.len() > WRITE_TARGET_LIMIT;
    ids.truncate(WRITE_TARGET_LIMIT);
    Ok((ids, truncated))
}

/// Build the warning attached when a write matched more than
/// `WRITE_TARGET_LIMIT` chunks and only the first batch was processed.
fn truncation_warning(stmt: &str, verb: &str) -> String {
    format!(
        "{stmt} matched more than {WRITE_TARGET_LIMIT} active chunks; only the first \
         {WRITE_TARGET_LIMIT} (ordered by id) were {verb}. Narrow the WHERE clause to \
         cover the rest."
    )
}

/// Convert an AQL literal/variable into a JSON value for a tool argument.
/// Variables resolve from `ctx.vars`; an unbound variable is an error.
fn aql_value_to_json(v: &AqlValue, vars: &BTreeMap<String, JsonValue>) -> AqlResult<JsonValue> {
    Ok(match v {
        AqlValue::Null => JsonValue::Null,
        AqlValue::Bool(b) => JsonValue::Bool(*b),
        AqlValue::Int(n) => JsonValue::Number((*n).into()),
        AqlValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        AqlValue::String(s) => JsonValue::String(s.clone()),
        AqlValue::Variable(name) => vars.get(name).cloned().ok_or_else(|| {
            AqlError::InvalidQuery(format!(
                "variable ${name} is not bound; pass it in `variables`"
            ))
        })?,
        AqlValue::Array(items) => JsonValue::Array(
            items
                .iter()
                .map(|i| aql_value_to_json(i, vars))
                .collect::<AqlResult<Vec<_>>>()?,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_write_type_maps_semantic_episodic() {
        assert_eq!(chunk_write_type(MemoryType::Semantic).unwrap(), "world");
        assert_eq!(
            chunk_write_type(MemoryType::Episodic).unwrap(),
            "experience"
        );
        assert!(chunk_write_type(MemoryType::Procedural).is_err());
        assert!(chunk_write_type(MemoryType::Working).is_err());
    }

    #[test]
    fn retain_option_name_maps_snake_to_camel() {
        assert_eq!(retain_option_name("trust_score"), Some("trustScore"));
        assert_eq!(retain_option_name("source_type"), Some("sourceType"));
        assert_eq!(retain_option_name("context"), Some("context"));
        assert_eq!(retain_option_name("bogus"), None);
    }

    #[test]
    fn aql_value_to_json_resolves_variable() {
        let mut vars = BTreeMap::new();
        vars.insert("q".to_string(), json!("hello"));
        let v = aql_value_to_json(&AqlValue::Variable("q".to_string()), &vars).unwrap();
        assert_eq!(v, json!("hello"));
        assert!(aql_value_to_json(&AqlValue::Variable("missing".to_string()), &vars).is_err());
    }

    /// Build a minimal in-memory `chunks` table (only the columns
    /// `resolve_chunk_ids` reads) seeded with `count` active "world" rows.
    fn seeded_ctx(count: usize) -> (rusqlite::Connection, BTreeMap<String, JsonValue>) {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, is_active INTEGER, memory_type TEXT);",
        )
        .unwrap();
        for i in 0..count {
            conn.execute(
                "INSERT INTO chunks (id, is_active, memory_type) VALUES (?1, 1, 'world')",
                rusqlite::params![format!("c{i:05}")],
            )
            .unwrap();
        }
        (conn, BTreeMap::new())
    }

    #[test]
    fn resolve_chunk_ids_under_cap_is_not_truncated() {
        let (conn, vars) = seeded_ctx(5);
        let bridge = crate::exec_ctx::BridgeHandle::new(None, std::path::Path::new("test.db"));
        let ctx = ExecCtx {
            conn: &conn,
            vars: &vars,
            bridge: &bridge,
        };
        let (ids, truncated) = resolve_chunk_ids(&ctx, MemoryType::Semantic, &[]).unwrap();
        assert_eq!(ids.len(), 5);
        assert!(!truncated);
    }

    #[test]
    fn resolve_chunk_ids_over_cap_is_truncated_and_flagged() {
        let (conn, vars) = seeded_ctx(WRITE_TARGET_LIMIT + 5);
        let bridge = crate::exec_ctx::BridgeHandle::new(None, std::path::Path::new("test.db"));
        let ctx = ExecCtx {
            conn: &conn,
            vars: &vars,
            bridge: &bridge,
        };
        let (ids, truncated) = resolve_chunk_ids(&ctx, MemoryType::Semantic, &[]).unwrap();
        assert_eq!(ids.len(), WRITE_TARGET_LIMIT);
        assert!(truncated);
    }

    #[test]
    fn truncation_warning_names_statement_and_verb() {
        let msg = truncation_warning("UPDATE", "superseded");
        assert!(msg.contains("UPDATE"));
        assert!(msg.contains("superseded"));
        assert!(msg.contains(&WRITE_TARGET_LIMIT.to_string()));
    }
}
