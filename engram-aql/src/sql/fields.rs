//! Field resolution: AQL field names → SQL column expressions.
//!
//! Fields that match a known column on the target table are used directly.
//! Fields that don't match are assumed to be JSON-stored inside the table's
//! text/data_json column and resolved via json_extract.

use crate::memory_map::EngramTable;

pub enum FieldRef {
    /// Direct column reference
    Column(&'static str),
    /// json_extract(<column>, '$.<path>')
    JsonPath { column: &'static str, path: String },
    /// Field cannot be resolved against this table's columns AND there is
    /// no JSON bag column to fall back to. Produces a SQL expression that
    /// always evaluates to NULL so queries fail cleanly (no rows matched)
    /// instead of silently returning wrong data.
    Unresolvable { table: EngramTable, field: String },
}

impl FieldRef {
    pub fn to_sql(&self) -> String {
        match self {
            FieldRef::Column(name) => (*name).to_string(),
            FieldRef::JsonPath { column, path } => {
                format!("json_extract({}, '$.{}')", column, path)
            }
            FieldRef::Unresolvable { .. } => {
                // Expression that compares as NULL against any value,
                // ensuring the query returns no rows rather than a silent
                // bogus match.
                "NULL".to_string()
            }
        }
    }
}

const CHUNK_COLUMNS: &[&str] = &[
    "id",
    "text",
    "memory_type",
    "source",
    "source_uri",
    "context",
    "source_type",
    "trust_score",
    "verified_by_user",
    "event_time",
    "event_time_end",
    "temporal_label",
    "text_hash",
    "created_at",
    "updated_at",
    "reflected_at",
    "superseded_by",
    "is_active",
];

const TOOLS_COLUMNS: &[&str] = &[
    "id",
    "name",
    "description",
    "api_url",
    "ranking",
    "tags",
    "namespace",
    "scope",
    "created_at",
    "updated_at",
    "is_active",
];

const OBSERVATION_COLUMNS: &[&str] = &[
    "id",
    "summary",
    "source_chunks",
    "source_entities",
    "domain",
    "topic",
    "synthesized_at",
    "last_refreshed",
    "refresh_count",
    "is_active",
];

const WORKING_MEMORY_COLUMNS: &[&str] = &[
    "id",
    "task_id",
    "scope",
    "data_json",
    "seed_query",
    "topic_embedding",
    "updated_at",
    "expires_at",
];

/// A JSON-path field name is safe to interpolate into the `$.<path>` segment
/// of a `json_extract(...)` expression only if it matches the character set
/// the AQL grammar (`aql.pest`) allows: `[A-Za-z0-9_.-]`. Callers outside the
/// parser (tests, other crates) can bypass the grammar, so this runtime guard
/// is the real security boundary. `debug_assert!` was previously used here —
/// it compiles out in release builds, so it was not sufficient.
fn is_safe_json_path(path: &str) -> bool {
    !path.is_empty()
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

pub fn resolve_field(field: &str, table: EngramTable) -> FieldRef {
    let known: &[&str] = match table {
        EngramTable::Chunks | EngramTable::All => CHUNK_COLUMNS,
        EngramTable::Tools => TOOLS_COLUMNS,
        EngramTable::Observations => OBSERVATION_COLUMNS,
        EngramTable::WorkingMemory => WORKING_MEMORY_COLUMNS,
    };

    for col in known {
        if *col == field {
            return FieldRef::Column(col);
        }
    }

    // Fallback: json_extract on the "JSON bag" column when one exists.
    // Tables without a JSON bag column (Observations, Tools) return Unresolvable
    // so queries fail cleanly instead of silently matching NULL.
    // Note: tools.tags is a JSON array (not a bag), so it is also Unresolvable.
    //
    // Before constructing a JsonPath variant we validate the field name. An
    // unsafe name falls through to Unresolvable (which renders as `NULL` in
    // `to_sql()`), so a malicious field name yields a zero-row query instead
    // of injectable SQL. This keeps `to_sql()` infallible.
    match table {
        EngramTable::Chunks | EngramTable::All if is_safe_json_path(field) => FieldRef::JsonPath {
            column: "text",
            path: field.to_string(),
        },
        EngramTable::WorkingMemory if is_safe_json_path(field) => FieldRef::JsonPath {
            column: "data_json",
            path: field.to_string(),
        },
        _ => FieldRef::Unresolvable {
            table,
            field: field.to_string(),
        },
    }
}
