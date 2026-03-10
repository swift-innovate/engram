-- =============================================================================
-- Engram Schema
-- Memory traces for AI agents that strengthen with reinforcement
-- 
-- Combines:
--   - memory-ragkg trust/provenance layer (our design)
--   - Hindsight's four-network cognitive architecture  
--   - SQLite + sqlite-vec for zero-infrastructure deployment
--
-- One .engram file per agent: ~/.valor/memory/<bankId>.engram
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- =============================================================================
-- BANK CONFIGURATION
-- Per-agent persona and behavioral settings for retain/recall/reflect
-- Inspired by Hindsight's bank-level mission/disposition config
-- =============================================================================

CREATE TABLE IF NOT EXISTS bank_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default configuration entries (inserted on bank creation)
-- retain_mission:   What to prioritize when storing memories
-- reflect_mission:  How to reason during reflection cycles
-- disposition:      JSON behavioral parameters for reflect
-- recall_budget:    Max tokens to return from recall (default 4096)
-- reflect_schedule: Cron expression or 'manual' (default 'manual')

-- =============================================================================
-- CHUNKS (Memory Units)
-- Core storage for all memory content with embeddings
-- Extended from memory-ragkg with memory_type classification
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    
    -- Content
    text TEXT NOT NULL,
    embedding BLOB,                    -- sqlite-vec compatible float32 array
    
    -- Classification (Hindsight's four networks)
    memory_type TEXT NOT NULL DEFAULT 'world'
        CHECK (memory_type IN ('world', 'experience', 'observation', 'opinion')),
    
    -- Source & context
    source TEXT,                        -- filename, conversation_id, tool_name, etc.
    source_uri TEXT,                    -- full URI for traceability
    context TEXT,                       -- freeform context tag (e.g. 'career update', 'mission:VALOR-042')
    
    -- Trust/Provenance (from memory-ragkg)
    source_type TEXT NOT NULL DEFAULT 'inferred'
        CHECK (source_type IN ('user_stated', 'inferred', 'external_doc', 'tool_result', 'agent_generated')),
    trust_score REAL NOT NULL DEFAULT 0.5
        CHECK (trust_score >= 0.0 AND trust_score <= 1.0),
    verified_by_user BOOLEAN DEFAULT FALSE,
    
    -- Temporal anchoring (from Hindsight's temporal retrieval)
    event_time TIMESTAMP,              -- when the event/fact occurred (may differ from created_at)
    event_time_end TIMESTAMP,          -- for facts with temporal ranges ("worked there 2020-2023")
    temporal_label TEXT,               -- human-readable: 'last spring', 'Q4 2025', etc.
    
    -- Lifecycle
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reflected_at TIMESTAMP,            -- NULL = not yet processed by reflect cycle
    superseded_by TEXT REFERENCES chunks(id),  -- for corrected/updated facts
    is_active BOOLEAN DEFAULT TRUE     -- soft delete
);

-- Full-text search index (BM25 via FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    source,
    context,
    temporal_label,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, source, context, temporal_label)
    VALUES (new.rowid, new.text, new.source, new.context, new.temporal_label);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, source, context, temporal_label)
    VALUES ('delete', old.rowid, old.text, old.source, old.context, old.temporal_label);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, source, context, temporal_label)
    VALUES ('delete', old.rowid, old.text, old.source, old.context, old.temporal_label);
    INSERT INTO chunks_fts(rowid, text, source, context, temporal_label)
    VALUES (new.rowid, new.text, new.source, new.context, new.temporal_label);
END;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunks_memory_type ON chunks(memory_type) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_event_time ON chunks(event_time) WHERE event_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_reflected ON chunks(reflected_at) WHERE reflected_at IS NULL AND is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_trust ON chunks(trust_score);
CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at);

-- =============================================================================
-- ENTITIES
-- Canonical entities extracted from chunks
-- From memory-ragkg design with enhancements for entity resolution
-- =============================================================================

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,                 -- display name
    canonical_name TEXT NOT NULL,       -- normalized for matching (lowercase, trimmed)
    entity_type TEXT NOT NULL           -- 'person', 'project', 'organization', 'technology',
        CHECK (entity_type IN (         -- 'location', 'concept', 'event', 'tool'
            'person', 'project', 'organization', 'technology',
            'location', 'concept', 'event', 'tool'
        )),
    aliases TEXT DEFAULT '[]',          -- JSON array of alternate names
    description TEXT,                   -- short entity summary, updated by reflect
    
    -- Trust (inherited from highest-trust source chunk)
    trust_score REAL DEFAULT 0.5,
    source_type TEXT DEFAULT 'inferred',
    
    -- Lifecycle
    mention_count INTEGER DEFAULT 1,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_mentions ON entities(mention_count DESC);

-- =============================================================================
-- RELATIONS
-- Directed edges between entities forming the knowledge graph
-- From memory-ragkg with confidence scoring
-- =============================================================================

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL REFERENCES entities(id),
    target_entity_id TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,        -- 'works_on', 'knows', 'prefers', 'owns',
                                        -- 'depends_on', 'located_in', 'part_of',
                                        -- 'decided', 'caused', 'related_to'
    description TEXT,                   -- freeform edge label
    
    -- Provenance
    source_chunk_id TEXT REFERENCES chunks(id),  -- which chunk established this relation
    confidence REAL DEFAULT 0.5,
    trust_score REAL DEFAULT 0.5,
    
    -- Temporal (when was this relation active?)
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,              -- NULL = still active
    
    -- Lifecycle
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Prevent duplicate edges
    UNIQUE(source_entity_id, target_entity_id, relation_type) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(source_chunk_id);

-- =============================================================================
-- ENTITY-CHUNK JUNCTION
-- Maps which chunks mention which entities (many-to-many)
-- Enables graph-seeded retrieval: entity → related chunks
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    entity_id TEXT NOT NULL REFERENCES entities(id),
    mention_type TEXT DEFAULT 'reference'  -- 'subject', 'reference', 'context'
        CHECK (mention_type IN ('subject', 'reference', 'context')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);

-- =============================================================================
-- OPINIONS (Hindsight's belief network, lightweight)
-- Confidence-scored beliefs that evolve with evidence
-- =============================================================================

CREATE TABLE IF NOT EXISTS opinions (
    id TEXT PRIMARY KEY,
    belief TEXT NOT NULL,               -- the opinion statement
    confidence REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Evidence tracking
    supporting_chunks TEXT DEFAULT '[]', -- JSON array of chunk IDs that support this
    contradicting_chunks TEXT DEFAULT '[]', -- JSON array of chunk IDs that contradict
    evidence_count INTEGER DEFAULT 0,
    
    -- Categorization
    domain TEXT,                        -- 'architecture', 'preferences', 'workflow', etc.
    related_entities TEXT DEFAULT '[]', -- JSON array of entity IDs
    
    -- Lifecycle
    formed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMP,
    last_challenged TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_opinions_domain ON opinions(domain) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_opinions_confidence ON opinions(confidence DESC) WHERE is_active = TRUE;

-- =============================================================================
-- OBSERVATIONS (Synthesized knowledge from reflect cycles)
-- Higher-order understanding consolidated from multiple facts
-- =============================================================================

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,              -- the synthesized observation
    
    -- Source tracking
    source_chunks TEXT DEFAULT '[]',    -- JSON array of chunk IDs that informed this
    source_entities TEXT DEFAULT '[]',  -- JSON array of entity IDs involved
    
    -- Categorization
    domain TEXT,                        -- matches opinion domains
    topic TEXT,                         -- specific topic within domain
    
    -- Lifecycle
    synthesized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_refreshed TIMESTAMP,           -- when reflect last updated this
    refresh_count INTEGER DEFAULT 0,    -- how many times it's been refined
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_observations_domain ON observations(domain) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_observations_topic ON observations(topic) WHERE is_active = TRUE;

-- =============================================================================
-- REFLECT LOG
-- Audit trail for reflection cycles
-- =============================================================================

CREATE TABLE IF NOT EXISTS reflect_log (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Scope
    facts_processed INTEGER DEFAULT 0,
    observations_created INTEGER DEFAULT 0,
    observations_updated INTEGER DEFAULT 0,
    opinions_formed INTEGER DEFAULT 0,
    opinions_reinforced INTEGER DEFAULT 0,
    opinions_challenged INTEGER DEFAULT 0,
    
    -- LLM details
    model_used TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    
    -- Status
    status TEXT DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    error TEXT
);

-- =============================================================================
-- EXTRACTION QUEUE
-- Deferred entity extraction for the fast-write/slow-extract pattern
-- =============================================================================

CREATE TABLE IF NOT EXISTS extraction_queue (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER DEFAULT 0,
    last_attempt TIMESTAMP,
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_extraction_queue_status ON extraction_queue(status, queued_at);

-- =============================================================================
-- VIEWS
-- Convenience views for common query patterns
-- =============================================================================

-- Active world facts with trust context
CREATE VIEW IF NOT EXISTS v_world_facts AS
SELECT c.id, c.text, c.trust_score, c.source_type, c.event_time, c.created_at,
       c.verified_by_user, c.context
FROM chunks c
WHERE c.memory_type = 'world' AND c.is_active = TRUE AND c.superseded_by IS NULL
ORDER BY c.trust_score DESC, c.created_at DESC;

-- Active experiences (agent's own history)
CREATE VIEW IF NOT EXISTS v_experiences AS
SELECT c.id, c.text, c.source, c.context, c.event_time, c.created_at
FROM chunks c
WHERE c.memory_type = 'experience' AND c.is_active = TRUE
ORDER BY c.event_time DESC, c.created_at DESC;

-- High-confidence opinions
CREATE VIEW IF NOT EXISTS v_strong_opinions AS
SELECT o.id, o.belief, o.confidence, o.domain, o.evidence_count,
       o.formed_at, o.last_reinforced
FROM opinions o
WHERE o.is_active = TRUE AND o.confidence >= 0.7
ORDER BY o.confidence DESC;

-- Entity relationship graph (for graph traversal queries)
CREATE VIEW IF NOT EXISTS v_entity_graph AS
SELECT 
    r.id AS relation_id,
    se.name AS source_name,
    se.entity_type AS source_type,
    r.relation_type,
    te.name AS target_name,
    te.entity_type AS target_type,
    r.confidence,
    r.description
FROM relations r
JOIN entities se ON r.source_entity_id = se.id AND se.is_active = TRUE
JOIN entities te ON r.target_entity_id = te.id AND te.is_active = TRUE
WHERE r.is_active = TRUE
ORDER BY r.confidence DESC;

-- Unreflected facts (input for next reflect cycle)
CREATE VIEW IF NOT EXISTS v_unreflected AS
SELECT c.id, c.text, c.memory_type, c.source, c.context, c.event_time, c.created_at
FROM chunks c
WHERE c.reflected_at IS NULL 
  AND c.is_active = TRUE
  AND c.memory_type IN ('world', 'experience')
ORDER BY c.created_at ASC;

-- Pending entity extractions
CREATE VIEW IF NOT EXISTS v_pending_extractions AS
SELECT eq.chunk_id, c.text, c.source, eq.queued_at, eq.attempts
FROM extraction_queue eq
JOIN chunks c ON eq.chunk_id = c.id
WHERE eq.status = 'pending'
ORDER BY eq.queued_at ASC;
