// =============================================================================
// recall.ts - Memory Retrieval (Multi-Pathway Access)
//
// Mirrors biological recall — multiple access pathways converge on the
// same memory traces. Semantic similarity is pattern matching, keyword is
// direct access, graph traversal is associative recall, temporal is episodic.
//
// Four parallel retrieval strategies fused via Reciprocal Rank Fusion:
//   1. Semantic search (sqlite-vec cosine similarity)
//   2. Keyword search (FTS5 BM25)
//   3. Entity graph traversal (SQL recursive CTE)
//   4. Temporal filtering (date range queries)
//
// Results merged via Reciprocal Rank Fusion, then trust-weighted.
// =============================================================================

import Database from 'better-sqlite3';
import type { EmbeddingProvider } from './retain.js';

// =============================================================================
// Types
// =============================================================================

export interface RecallOptions {
  /** Max results to return */
  topK?: number;
  /** Max characters per snippet */
  snippetChars?: number;
  /** Which retrieval strategies to use (default: all) */
  strategies?: Array<'semantic' | 'keyword' | 'graph' | 'temporal'>;
  /** Memory types to include (default: all) */
  memoryTypes?: Array<'world' | 'experience' | 'observation' | 'opinion'>;
  /** Temporal filter: only facts after this date */
  after?: string;
  /** Temporal filter: only facts before this date */
  before?: string;
  /** Minimum trust score (default: 0.0) */
  minTrust?: number;
  /** Include opinions in results */
  includeOpinions?: boolean;
  /** Include observations in results */
  includeObservations?: boolean;
  /** RRF constant (default: 60) */
  rrfK?: number;
}

export interface RecallResult {
  id: string;
  text: string;
  memoryType: string;
  source: string | null;
  trustScore: number;
  sourceType: string;
  eventTime: string | null;
  score: number;           // final fused score
  strategies: string[];    // which strategies found this
}

export interface RecallResponse {
  results: RecallResult[];
  opinions: Array<{ belief: string; confidence: number; domain: string | null }>;
  observations: Array<{ summary: string; domain: string | null; topic: string | null }>;
  totalCandidates: number;
  strategiesUsed: string[];
}

// Internal types for per-strategy results
interface ScoredChunk {
  id: string;
  text: string;
  memory_type: string;
  source: string | null;
  trust_score: number;
  source_type: string;
  event_time: string | null;
  rank: number;
  strategy: string;
}

// =============================================================================
// Retrieval Strategies
// =============================================================================

/**
 * Strategy 1: Semantic search via sqlite-vec
 */
function semanticSearch(
  db: Database.Database,
  queryEmbedding: Float32Array,
  limit: number,
  filters: { memoryTypes?: string[]; minTrust?: number }
): ScoredChunk[] {
  // sqlite-vec uses vec_distance_cosine for similarity
  // We need to filter after the vector search since sqlite-vec
  // doesn't support WHERE clauses in the same query
  const embeddingBuffer = Buffer.from(queryEmbedding.buffer);

  try {
    // Use sqlite-vec's virtual table for KNN search
    const rows = db.prepare(`
      SELECT c.id, c.text, c.memory_type, c.source, c.trust_score,
             c.source_type, c.event_time,
             vec_distance_cosine(c.embedding, ?) AS distance
      FROM chunks c
      WHERE c.is_active = TRUE
        AND c.embedding IS NOT NULL
        AND c.trust_score >= ?
      ORDER BY distance ASC
      LIMIT ?
    `).all(embeddingBuffer, filters.minTrust || 0, limit * 2) as any[];

    return rows
      .filter(r => !filters.memoryTypes || filters.memoryTypes.includes(r.memory_type))
      .slice(0, limit)
      .map((r, i) => ({
        ...r,
        rank: i + 1,
        strategy: 'semantic',
      }));
  } catch {
    // sqlite-vec may not be loaded — fallback gracefully
    return [];
  }
}

/**
 * Strategy 2: Keyword search via FTS5 BM25
 */
function keywordSearch(
  db: Database.Database,
  query: string,
  limit: number,
  filters: { memoryTypes?: string[]; minTrust?: number }
): ScoredChunk[] {
  // FTS5 with BM25 ranking
  const typeFilter = filters.memoryTypes
    ? `AND c.memory_type IN (${filters.memoryTypes.map(t => `'${t}'`).join(',')})`
    : '';

  try {
    const rows = db.prepare(`
      SELECT c.id, c.text, c.memory_type, c.source, c.trust_score,
             c.source_type, c.event_time,
             rank AS bm25_rank
      FROM chunks_fts fts
      JOIN chunks c ON c.rowid = fts.rowid
      WHERE chunks_fts MATCH ?
        AND c.is_active = TRUE
        AND c.trust_score >= ?
        ${typeFilter}
      ORDER BY rank
      LIMIT ?
    `).all(query, filters.minTrust || 0, limit) as any[];

    return rows.map((r, i) => ({
      ...r,
      rank: i + 1,
      strategy: 'keyword',
    }));
  } catch {
    return [];
  }
}

/**
 * Strategy 3: Entity graph traversal
 * Finds entities mentioned in the query, then retrieves chunks
 * connected to those entities (1-hop and 2-hop)
 */
function graphSearch(
  db: Database.Database,
  query: string,
  limit: number,
  filters: { memoryTypes?: string[]; minTrust?: number }
): ScoredChunk[] {
  // Simple approach: tokenize query, match against entity names/aliases
  const queryTokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  
  if (queryTokens.length === 0) return [];

  try {
    // Find matching entities
    const likeClauses = queryTokens
      .map(() => `(e.canonical_name LIKE ? OR e.aliases LIKE ?)`)
      .join(' OR ');
    const likeParams = queryTokens.flatMap(t => [`%${t}%`, `%${t}%`]);

    const matchedEntities = db.prepare(`
      SELECT e.id, e.canonical_name, e.mention_count
      FROM entities e
      WHERE e.is_active = TRUE AND (${likeClauses})
      ORDER BY e.mention_count DESC
      LIMIT 10
    `).all(...likeParams) as Array<{ id: string; canonical_name: string; mention_count: number }>;

    if (matchedEntities.length === 0) return [];

    const entityIds = matchedEntities.map(e => e.id);

    // 1-hop: chunks directly mentioning these entities
    const placeholders = entityIds.map(() => '?').join(',');
    const typeFilter = filters.memoryTypes
      ? `AND c.memory_type IN (${filters.memoryTypes.map(t => `'${t}'`).join(',')})`
      : '';

    const directChunks = db.prepare(`
      SELECT DISTINCT c.id, c.text, c.memory_type, c.source, c.trust_score,
             c.source_type, c.event_time
      FROM chunk_entities ce
      JOIN chunks c ON ce.chunk_id = c.id
      WHERE ce.entity_id IN (${placeholders})
        AND c.is_active = TRUE
        AND c.trust_score >= ?
        ${typeFilter}
      ORDER BY c.trust_score DESC, c.created_at DESC
      LIMIT ?
    `).all(...entityIds, filters.minTrust || 0, limit) as any[];

    // 2-hop: chunks mentioning entities related to matched entities
    const relatedChunks = db.prepare(`
      SELECT DISTINCT c.id, c.text, c.memory_type, c.source, c.trust_score,
             c.source_type, c.event_time
      FROM relations r
      JOIN chunk_entities ce ON (
        ce.entity_id = r.target_entity_id OR ce.entity_id = r.source_entity_id
      )
      JOIN chunks c ON ce.chunk_id = c.id
      WHERE (r.source_entity_id IN (${placeholders}) OR r.target_entity_id IN (${placeholders}))
        AND r.is_active = TRUE
        AND c.is_active = TRUE
        AND c.trust_score >= ?
        AND c.id NOT IN (${directChunks.map(() => '?').join(',') || "''"})
        ${typeFilter}
      ORDER BY r.confidence DESC, c.trust_score DESC
      LIMIT ?
    `).all(
      ...entityIds, ...entityIds,
      filters.minTrust || 0,
      ...directChunks.map((c: any) => c.id),
      Math.max(0, limit - directChunks.length)
    ) as any[];

    const combined = [...directChunks, ...relatedChunks];
    return combined.map((r, i) => ({
      ...r,
      rank: i + 1,
      strategy: 'graph',
    }));
  } catch {
    return [];
  }
}

/**
 * Strategy 4: Temporal search
 * Filters by time range, prioritizes recent memories
 */
function temporalSearch(
  db: Database.Database,
  limit: number,
  filters: {
    after?: string;
    before?: string;
    memoryTypes?: string[];
    minTrust?: number;
  }
): ScoredChunk[] {
  if (!filters.after && !filters.before) return [];

  const conditions: string[] = [
    'c.is_active = TRUE',
    `c.trust_score >= ${filters.minTrust || 0}`,
  ];
  const params: any[] = [];

  if (filters.after) {
    conditions.push('(c.event_time >= ? OR (c.event_time IS NULL AND c.created_at >= ?))');
    params.push(filters.after, filters.after);
  }
  if (filters.before) {
    conditions.push('(c.event_time <= ? OR (c.event_time IS NULL AND c.created_at <= ?))');
    params.push(filters.before, filters.before);
  }
  if (filters.memoryTypes) {
    conditions.push(`c.memory_type IN (${filters.memoryTypes.map(t => `'${t}'`).join(',')})`);
  }

  try {
    const rows = db.prepare(`
      SELECT c.id, c.text, c.memory_type, c.source, c.trust_score,
             c.source_type, c.event_time
      FROM chunks c
      WHERE ${conditions.join(' AND ')}
      ORDER BY COALESCE(c.event_time, c.created_at) DESC
      LIMIT ?
    `).all(...params, limit) as any[];

    return rows.map((r, i) => ({
      ...r,
      rank: i + 1,
      strategy: 'temporal',
    }));
  } catch {
    return [];
  }
}

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

function reciprocalRankFusion(
  strategyResults: ScoredChunk[][],
  k: number = 60
): Map<string, { score: number; strategies: string[]; chunk: ScoredChunk }> {
  const fused = new Map<string, { score: number; strategies: string[]; chunk: ScoredChunk }>();

  for (const results of strategyResults) {
    for (const chunk of results) {
      const existing = fused.get(chunk.id);
      const rrfScore = 1 / (k + chunk.rank);

      if (existing) {
        existing.score += rrfScore;
        if (!existing.strategies.includes(chunk.strategy)) {
          existing.strategies.push(chunk.strategy);
        }
      } else {
        fused.set(chunk.id, {
          score: rrfScore,
          strategies: [chunk.strategy],
          chunk,
        });
      }
    }
  }

  return fused;
}

/**
 * Apply trust weighting to fused scores.
 * Higher trust = score boosted, lower trust = score penalized.
 * verified_by_user gets maximum boost.
 */
function applyTrustWeighting(
  fused: Map<string, { score: number; strategies: string[]; chunk: ScoredChunk }>
): void {
  for (const [, entry] of fused) {
    // Trust multiplier: 0.5 trust = 0.8x, 1.0 trust = 1.2x
    const trustMultiplier = 0.6 + (entry.chunk.trust_score * 0.6);
    
    // Boost for multi-strategy hits (found by multiple strategies = more relevant)
    const strategyBoost = 1 + (entry.strategies.length - 1) * 0.1;
    
    entry.score *= trustMultiplier * strategyBoost;
  }
}

// =============================================================================
// Main Recall Function
// =============================================================================

export async function recall(
  db: Database.Database,
  query: string,
  embedder: EmbeddingProvider,
  options: RecallOptions = {}
): Promise<RecallResponse> {
  const {
    topK = 10,
    snippetChars = 500,
    strategies = ['semantic', 'keyword', 'graph', 'temporal'],
    memoryTypes,
    after,
    before,
    minTrust = 0.0,
    includeOpinions = true,
    includeObservations = true,
    rrfK = 60,
  } = options;

  const perStrategyLimit = topK * 3; // Oversample per strategy, then fuse
  const filters = { memoryTypes, minTrust };
  const strategyResults: ScoredChunk[][] = [];
  const strategiesUsed: string[] = [];

  // Run strategies
  if (strategies.includes('semantic')) {
    const queryEmbedding = await embedder.embed(query);
    const results = semanticSearch(db, queryEmbedding, perStrategyLimit, filters);
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('semantic');
    }
  }

  if (strategies.includes('keyword')) {
    const results = keywordSearch(db, query, perStrategyLimit, filters);
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('keyword');
    }
  }

  if (strategies.includes('graph')) {
    const results = graphSearch(db, query, perStrategyLimit, filters);
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('graph');
    }
  }

  if (strategies.includes('temporal') && (after || before)) {
    const results = temporalSearch(db, perStrategyLimit, { after, before, ...filters });
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('temporal');
    }
  }

  // Fuse results
  const fused = reciprocalRankFusion(strategyResults, rrfK);
  applyTrustWeighting(fused);

  // Sort by fused score and take top K
  const sorted = [...fused.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  const results: RecallResult[] = sorted.map(entry => ({
    id: entry.chunk.id,
    text: entry.chunk.text.length > snippetChars
      ? entry.chunk.text.substring(0, snippetChars) + '...'
      : entry.chunk.text,
    memoryType: entry.chunk.memory_type,
    source: entry.chunk.source,
    trustScore: entry.chunk.trust_score,
    sourceType: entry.chunk.source_type,
    eventTime: entry.chunk.event_time,
    score: entry.score,
    strategies: entry.strategies,
  }));

  // Gather relevant opinions
  const opinions = includeOpinions
    ? (db.prepare(`
        SELECT belief, confidence, domain
        FROM opinions
        WHERE is_active = TRUE AND confidence >= 0.5
        ORDER BY confidence DESC
        LIMIT 5
      `).all() as Array<{ belief: string; confidence: number; domain: string | null }>)
    : [];

  // Gather relevant observations
  const observations = includeObservations
    ? (db.prepare(`
        SELECT summary, domain, topic
        FROM observations
        WHERE is_active = TRUE
        ORDER BY last_refreshed DESC, synthesized_at DESC
        LIMIT 5
      `).all() as Array<{ summary: string; domain: string | null; topic: string | null }>)
    : [];

  return {
    results,
    opinions,
    observations,
    totalCandidates: fused.size,
    strategiesUsed,
  };
}
