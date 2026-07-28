//! Memory type mapping tests.

use aql_parser::ast::MemoryType;
use engram_aql::memory_map::{aql_to_chunk_memory_type, aql_to_table, EngramTable};

#[test]
fn semantic_maps_to_chunks_world() {
    assert_eq!(aql_to_table(MemoryType::Semantic), EngramTable::Chunks);
    assert_eq!(
        aql_to_chunk_memory_type(MemoryType::Semantic),
        Some("world")
    );
}

#[test]
fn episodic_maps_to_chunks_experience() {
    assert_eq!(aql_to_table(MemoryType::Episodic), EngramTable::Chunks);
    assert_eq!(
        aql_to_chunk_memory_type(MemoryType::Episodic),
        Some("experience")
    );
}

#[test]
fn procedural_maps_to_observations() {
    assert_eq!(
        aql_to_table(MemoryType::Procedural),
        EngramTable::Observations
    );
    assert_eq!(aql_to_chunk_memory_type(MemoryType::Procedural), None);
}

#[test]
fn working_maps_to_working_memory() {
    assert_eq!(
        aql_to_table(MemoryType::Working),
        EngramTable::WorkingMemory
    );
    assert_eq!(aql_to_chunk_memory_type(MemoryType::Working), None);
}

#[test]
fn tools_maps_to_tools_table() {
    assert_eq!(aql_to_table(MemoryType::Tools), EngramTable::Tools);
    assert_eq!(aql_to_chunk_memory_type(MemoryType::Tools), None);
}

#[test]
fn all_is_recognized() {
    assert_eq!(aql_to_table(MemoryType::All), EngramTable::All);
    assert_eq!(aql_to_chunk_memory_type(MemoryType::All), None);
}

#[test]
fn engram_table_as_sql_name_matches_schema() {
    assert_eq!(EngramTable::Chunks.as_sql_name(), "chunks");
    assert_eq!(EngramTable::Tools.as_sql_name(), "tools");
    assert_eq!(EngramTable::Observations.as_sql_name(), "observations");
    assert_eq!(EngramTable::WorkingMemory.as_sql_name(), "working_memory");
    // ALL degrades to chunks in Phase 1
    assert_eq!(EngramTable::All.as_sql_name(), "chunks");
}
