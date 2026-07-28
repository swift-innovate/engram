//! AQL MemoryType ↔ Engram table mapping.
//!
//! AQL defines 6 memory types (Working, Tools, Procedural, Semantic, Episodic, All).
//! Engram stores data across 4 tables:
//!   - chunks (world + experience + observation + opinion via memory_type column)
//!   - observations (synthesized patterns — separate table)
//!   - working_memory (session state — separate table)
//!   - tools (tool registry — separate table)
//!
//! This module provides the forward mapping from AQL types to Engram tables,
//! and the chunks-specific mapping to the `memory_type` column value for types
//! that live inside the chunks table (Semantic → "world", Episodic → "experience").

use aql_parser::ast::MemoryType;

/// Engram tables that AQL queries can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngramTable {
    Chunks,
    Observations,
    WorkingMemory,
    Tools,
    /// "ALL" — cross-table query. Currently resolved to Chunks for Phase 1;
    /// future phases may federate across multiple tables.
    All,
}

impl EngramTable {
    /// Return the SQL table name for use in prepared queries.
    pub fn as_sql_name(self) -> &'static str {
        match self {
            EngramTable::Chunks => "chunks",
            EngramTable::Observations => "observations",
            EngramTable::WorkingMemory => "working_memory",
            EngramTable::Tools => "tools",
            // Phase 1: ALL queries the chunks table only; observations and
            // working_memory are not federated yet. Callers that accept a
            // MemoryType::All query must warn the user about this limitation.
            EngramTable::All => "chunks",
        }
    }
}

/// Map an AQL memory type to the Engram table it queries.
pub fn aql_to_table(aql: MemoryType) -> EngramTable {
    match aql {
        MemoryType::Episodic | MemoryType::Semantic => EngramTable::Chunks,
        MemoryType::Procedural => EngramTable::Observations,
        MemoryType::Working => EngramTable::WorkingMemory,
        MemoryType::Tools => EngramTable::Tools,
        MemoryType::All => EngramTable::All,
    }
}

/// For AQL types that live in the `chunks` table, return the `memory_type`
/// column value Engram uses. Returns `None` for types that use a different
/// table entirely (Procedural → observations, Working → working_memory,
/// Tools → tools, All → cross-table).
pub fn aql_to_chunk_memory_type(aql: MemoryType) -> Option<&'static str> {
    match aql {
        MemoryType::Episodic => Some("experience"),
        MemoryType::Semantic => Some("world"),
        MemoryType::Procedural | MemoryType::Working | MemoryType::Tools | MemoryType::All => None,
    }
}
