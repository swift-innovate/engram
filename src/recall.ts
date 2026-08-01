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
import { embeddingToBuffer, type EmbeddingProvider } from './retain.js';
import { bufferToFloat32Array, cosineSimilarity } from './insight-shared.js';
import { parseTemporalQuery } from './temporal-parser.js';

// =============================================================================
// Constants
// =============================================================================

/** Stop words filtered from graph search tokenization to avoid false positives */
const GRAPH_STOP_WORDS = new Set([
  'what',
  'who',
  'where',
  'when',
  'why',
  'how',
  'which',
  'the',
  'and',
  'for',
  'with',
  'not',
  'but',
  'this',
  'that',
  'are',
  'was',
  'were',
  'been',
  'has',
  'have',
  'had',
  'does',
  'did',
  'can',
  'could',
  'would',
  'should',
  'will',
  'about',
  'from',
  'into',
  'than',
  'then',
  'them',
  'they',
  'its',
  'our',
  'your',
  'use',
  'used',
  'using',
]);

// =============================================================================
// Source tiers
// =============================================================================

/**
 * Source-type → ranking tier. Final recall ordering is lexicographic
 * (tier ascending, fused score descending), so a lower-tier result can
 * NEVER outrank a higher-tier one regardless of trust score or relevance.
 * This is what enforces the trust-layer rule that external content
 * (tool results, external docs) cannot override user-stated directives.
 *
 * Tier 0: user_stated — directives and facts from the user.
 * Tier 1: inferred, agent_generated — the agent's own conclusions.
 * Tier 2: tool_result, external_doc — content from outside the trust boundary.
 */
export const DEFAULT_SOURCE_TIERS: Record<string, number> = {
  user_stated: 0,
  inferred: 1,
  agent_generated: 1,
  tool_result: 2,
  external_doc: 2,
};

/** Unknown source types rank in the least-privileged tier. */
const UNKNOWN_SOURCE_TIER = 2;

/**
 * Memory-type → rank. Used as a SOFT multiplicative weight in applyWeighting
 * (via memoryTypeWeightFromRank below), not a hard lexicographic floor.
 *
 * It used to be a hard floor — sorted before score, so no relevance match
 * could ever let an 'experience' chunk outrank a weakly-related 'observation'
 * or 'world' chunk. That's backwards for "recall this specific memory"
 * queries (personal narrative is exactly memory_type='experience') even
 * though it may have been reasonable for "give me a synthesized answer"
 * queries. A 2023 chunk ranked #1 by raw semantic distance out of ~4,700
 * chunks still failed to appear in the results at all, purely because it
 * was 'experience' competing against unrelated 'world'/'observation'
 * chunks that structurally outranked it regardless of relevance.
 *
 * Kept as a gentle preference instead: world/observation get a modest boost,
 * opinion a modest penalty (still consistent with the 0.85 opinion-confidence
 * cap in formatForPrompt), experience is the neutral baseline — but a strong
 * match in a "lower" type can still beat a weak match in a "higher" one.
 */
export const DEFAULT_MEMORY_TYPE_RANK: Record<string, number> = {
  world: 0,
  observation: 1,
  experience: 2,
  opinion: 3,
};

/** rank 0 (world) -> 1.15x, rank 3 (opinion) -> 0.925x. Linear in between. */
function memoryTypeWeightFromRank(rank: number): number {
  return 1.15 - rank * 0.075;
}

/** Unknown/missing memory types sort last within their tier. */
const UNKNOWN_MEMORY_TYPE_RANK = 99;

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
  /** Filter results to chunks whose source contains this string (substring match) */
  sourceFilter?: string;
  /** Filter results to chunks whose context contains this string (substring match) */
  contextFilter?: string;
  /**
   * Which scope(s) to search. Default: `['durable']` — the four long-term
   * memory types, matching recall()'s historical behavior exactly. Pass
   * `['task']` (or `['durable', 'task']`) to include ContextStore artifacts.
   * Expired 'task' rows (past `expires_at`) are always excluded regardless
   * of this option — TTL is enforced lazily at query time.
   */
  scope?: Array<'durable' | 'task'>;
  /**
   * Restrict 'task'-scope results to artifacts chained under this parent
   * ContextRef id (chunks.parent_ref). Ignored for 'durable' rows. Used by
   * ContextStore.query() to scope a search to one task's committed artifacts.
   */
  parentRef?: string;
  /** Boost RRF score for chunks whose source contains this pattern (soft preference) */
  sourceBoost?: { pattern: string; multiplier: number };
  /** Boost RRF score for chunks whose context contains this pattern (soft preference) */
  contextBoost?: { pattern: string; multiplier: number };
  /** Trust decay half-life in days (default: 180). Set to 0 to disable decay. */
  decayHalfLifeDays?: number;
  /**
   * Source-type → ranking tier overrides, merged over DEFAULT_SOURCE_TIERS.
   * Lower tier = structurally higher rank (lexicographic sort, see
   * DEFAULT_SOURCE_TIERS). Unknown source types fall to the lowest tier.
   */
  sourceTiers?: Record<string, number>;
  /**
   * Memory-type → rank overrides, merged over DEFAULT_MEMORY_TYPE_RANK.
   * Orders results within a source tier; unknown types sort last.
   */
  memoryTypeRank?: Record<string, number>;
  /**
   * Drop fused results whose final weighted `score` (after applyWeighting —
   * trust/decay/strategy-boost/memory-type) falls below this threshold.
   * Applied post-weighting, so the cutoff is on the exact same `score` field
   * returned to callers, and results are dropped before the tier-major sort
   * and topK truncation. Nominal range [0, 1] like other 0-1 options
   * (clamped at the MCP/CLI boundary the same way minTrust is); recall()
   * itself does not re-clamp. Default: undefined — no filtering, zero
   * behavior change for existing callers. Since D6, `score` is cosine
   * similarity (×0.94-0.99 trust tiebreak, then the usual decay/boost/
   * memory-type multipliers) for chunks the semantic strategy touched, so
   * this doubles as a semantic-relevance gate — a `minScore` around 0.4-0.45
   * is a reasonable floor for dropping weak semantic noise while keeping
   * genuine matches; RRF-fallback (non-semantic) chunks are on a different
   * scale and unaffected by that guidance.
   */
  minScore?: number;
  /**
   * Cosine-similarity floor for the opinions/observations attached to a
   * response (default 0.45). Insights are ranked semantically against the
   * query when they carry embeddings; anything below this floor is dropped
   * rather than padded in, so an off-topic query returns NO beliefs instead
   * of the highest-confidence ones it happens to hold.
   *
   * This gate applies ONLY to the embedded path. Rows without an embedding
   * (formed before the column existed, or without an embedder) fall back to
   * the legacy keyword match, which has no relevance score to gate on.
   * Set to 0 to keep every embedded insight ranked but unfiltered.
   */
  insightMinScore?: number;
  /**
   * When true, each result gains a `strategyScores` breakdown: the
   * per-strategy rank/RRF-contribution that fed fusion for this chunk, the
   * pre-weighting fused score, and the individual weighting multipliers
   * applied by applyWeighting(). Default false — the field is omitted
   * entirely (not present, not null) when off, so the payload stays lean.
   */
  explainScores?: boolean;
}

/** One strategy's contribution to a chunk's fused RRF score. */
export interface StrategyScoreBreakdown {
  strategy: string;
  /** This chunk's 1-indexed rank within that strategy's candidate list. */
  rank: number;
  /** 1 / (rrfK + rank) — this strategy's contribution to the fused score. */
  rrfScore: number;
}

/** Score observability payload attached to a result when explainScores is true. */
export interface ScoreExplanation {
  /** One entry per strategy that returned this chunk. */
  perStrategy: StrategyScoreBreakdown[];
  /**
   * Base relevance score before trust/decay/boost/memory-type weighting.
   * Cosine similarity when the chunk was a semantic hit (cosine-primary,
   * see D6); the RRF-summed fusion score otherwise.
   */
  rawFusedScore: number;
  /** Multiplicative weighting factors applied by applyWeighting(). */
  weighting: {
    trust: number;
    strategyBoost: number;
    decay: number;
    sourceBoost: number;
    contextBoost: number;
    memoryType: number;
  };
}

export interface RecallResult {
  id: string;
  text: string;
  memoryType: string;
  source: string | null;
  trustScore: number;
  sourceType: string;
  eventTime: string | null;
  /** When the chunk was written (SQLite CURRENT_TIMESTAMP, UTC). Powers the
   * staleness bracket in formatForPrompt's showProvenance. */
  createdAt: string;
  score: number; // final fused score
  strategies: string[]; // which strategies found this
  /** Present only when RecallOptions.explainScores is true. */
  strategyScores?: ScoreExplanation;
}

export interface RecallResponse {
  /**
   * Tier-major order, NOT pure relevance order: sorted by (source tier,
   * memory-type rank, trust-weighted score). results[0] is the best match
   * in the highest-present tier, not necessarily the highest-relevance
   * match overall.
   */
  results: RecallResult[];
  opinions: RecallOpinion[];
  observations: RecallObservation[];
  totalCandidates: number;
  strategiesUsed: string[];
}

/** A belief attached to a recall response. */
export interface RecallOpinion {
  belief: string;
  confidence: number;
  domain: string | null;
}

/** A synthesized observation attached to a recall response. */
export interface RecallObservation {
  summary: string;
  domain: string | null;
  topic: string | null;
}

// Internal types for per-strategy results
interface ScoredChunk {
  id: string;
  text: string;
  memory_type: string;
  source: string | null;
  context: string | null;
  trust_score: number;
  source_type: string;
  event_time: string | null;
  created_at: string;
  rank: number;
  strategy: string;
  /**
   * Raw cosine similarity (1 - vec_distance_cosine), set only by
   * semanticSearch. Undefined for keyword/graph/temporal hits — those
   * strategies have no relevance magnitude, only ordinal rank. See D6.
   */
  cosine?: number;
}

interface QueryFilters {
  memoryTypes?: string[];
  minTrust?: number;
  sourceFilter?: string;
  contextFilter?: string;
  after?: string;
  before?: string;
  /** Restrict to these source types (used by the tier-0 truncation reserve) */
  sourceTypeIn?: string[];
  /** Which chunks.scope values to include. Always set — recall() defaults it to ['durable']. */
  scope: string[];
  /** Restrict 'task'-scope rows to this parent_ref (ignored for 'durable' rows) */
  parentRef?: string;
}

// =============================================================================
// Helpers
// =============================================================================

function inClausePlaceholders(values: string[]): string {
  return values.map(() => '?').join(',');
}

/**
 * Strip characters that break FTS5 MATCH syntax and graph tokenization.
 * Preserves words and whitespace only. Used by the graph strategy, which
 * tokenizes on whitespace and has no concept of quoted phrases.
 */
function sanitizeQuery(query: string): string {
  return query
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Same punctuation stripping as sanitizeQuery, but preserves balanced
 * double-quoted segments as genuine FTS5 phrase terms instead of flattening
 * them into independently-ANDed words — `"blue green deployment"` becomes
 * an exact-phrase match instead of three unordered terms. Only used by the
 * keyword strategy, which is the only consumer of FTS5 MATCH syntax.
 *
 * Unquoted text (including FTS5 operator keywords like NEAR/AND/OR/NOT and
 * symbols like * ^ - : ( )) is stripped exactly as before, so unquoted
 * queries behave identically to sanitizeQuery. An unpaired quote has no
 * partner to form a phrase with, so the regex below never matches it — it
 * falls through to the plain strip path along with the rest of the string,
 * degrading gracefully instead of producing invalid FTS5 syntax.
 */
function sanitizeQueryForFts(query: string): string {
  const stripNonWord = (s: string): string =>
    s
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  const quoteRegex = /"([^"]*)"/g;
  const parts: string[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = quoteRegex.exec(query)) !== null) {
    const before = stripNonWord(query.slice(lastIndex, match.index));
    if (before) parts.push(before);

    const phraseContent = stripNonWord(match[1]);
    if (phraseContent) {
      // Double any internal quote per FTS5 phrase-escaping rules. Defensive
      // only — the capture group is quote-delimited, so it can't itself
      // contain a literal '"'.
      parts.push(`"${phraseContent.replace(/"/g, '""')}"`);
    }

    lastIndex = quoteRegex.lastIndex;
  }

  const rest = stripNonWord(query.slice(lastIndex));
  if (rest) parts.push(rest);

  return parts.join(' ').replace(/\s+/g, ' ').trim();
}

/**
 * Build SQL conditions and params for temporal filtering.
 * Uses event_time when set, falls back to created_at.
 * Same logic as temporalSearch — shared so all strategies filter consistently.
 */
function buildTemporalFilter(
  filters: QueryFilters,
  tableAlias: string = 'c',
): { conditions: string[]; params: any[] } {
  const conditions: string[] = [];
  const params: any[] = [];

  if (filters.after) {
    conditions.push(
      `(datetime(${tableAlias}.event_time) >= datetime(?) OR (${tableAlias}.event_time IS NULL AND datetime(${tableAlias}.created_at) >= datetime(?)))`,
    );
    params.push(filters.after, filters.after);
  }
  if (filters.before) {
    conditions.push(
      `(datetime(${tableAlias}.event_time) <= datetime(?) OR (${tableAlias}.event_time IS NULL AND datetime(${tableAlias}.created_at) <= datetime(?)))`,
    );
    params.push(filters.before, filters.before);
  }

  return { conditions, params };
}

/**
 * Build SQL fragment and params restricting results to specific source types.
 * Used by the tier-0 truncation reserve in recall().
 */
function buildSourceTypeFilter(filters: QueryFilters): {
  sql: string;
  params: string[];
} {
  if (!filters.sourceTypeIn?.length) return { sql: '', params: [] };
  return {
    sql: `AND c.source_type IN (${inClausePlaceholders(filters.sourceTypeIn)})`,
    params: filters.sourceTypeIn,
  };
}

/**
 * Build the scope/parent/TTL SQL fragment shared by all four strategies.
 * Always restricts to filters.scope (recall() defaults this to ['durable'],
 * so pre-ContextStore callers see identical results). The expires_at check
 * is unconditional — 'durable' rows never set it, so it's a no-op for them,
 * and it's what makes TTL expiry "lazy, checked at read time" rather than
 * requiring a background reaper.
 */
function buildScopeFilter(filters: QueryFilters): {
  sql: string;
  params: string[];
} {
  const params: string[] = [...filters.scope];
  // datetime(c.expires_at) normalizes storage format (ISO 8601 w/ 'T'/'Z',
  // as written by JS's Date#toISOString()) before comparing against
  // datetime('now')'s "YYYY-MM-DD HH:MM:SS" — a raw string compare would
  // spuriously order 'T' (0x54) after ' ' (0x20) and never expire anything.
  let sql = `AND c.scope IN (${inClausePlaceholders(filters.scope)}) AND (c.expires_at IS NULL OR datetime(c.expires_at) > datetime('now'))`;
  if (filters.parentRef) {
    sql += ' AND c.parent_ref = ?';
    params.push(filters.parentRef);
  }
  return { sql, params };
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
  filters: QueryFilters,
): ScoredChunk[] {
  const { memoryTypes, minTrust = 0, sourceFilter, contextFilter } = filters;
  const embeddingBuffer = embeddingToBuffer(queryEmbedding);

  // Build optional memory_type filter
  const typeFilter = memoryTypes?.length
    ? `AND c.memory_type IN (${inClausePlaceholders(memoryTypes)})`
    : '';

  // Build optional temporal filter
  const temporal = buildTemporalFilter(filters);
  const temporalSQL = temporal.conditions.length
    ? 'AND ' + temporal.conditions.join(' AND ')
    : '';
  const sourceType = buildSourceTypeFilter(filters);
  const scopeF = buildScopeFilter(filters);

  try {
    const rows = db
      .prepare(
        `
      SELECT c.id, c.text, c.memory_type, c.source, c.context, c.trust_score,
             c.source_type, c.event_time, c.created_at,
             vec_distance_cosine(c.embedding, ?) AS distance
      FROM chunks c
      -- distance is cosine distance in [0, 2]; converted to similarity below
      WHERE c.is_active = TRUE
        AND c.embedding IS NOT NULL
        AND c.trust_score >= ?
        AND (? IS NULL OR c.source LIKE '%' || ? || '%')
        AND (? IS NULL OR c.context LIKE '%' || ? || '%')
        ${typeFilter}
        ${sourceType.sql}
        ${scopeF.sql}
        ${temporalSQL}
      ORDER BY distance ASC
      LIMIT ?
    `,
      )
      .all(
        embeddingBuffer,
        minTrust,
        sourceFilter ?? null,
        sourceFilter ?? null,
        contextFilter ?? null,
        contextFilter ?? null,
        ...(memoryTypes ?? []),
        ...sourceType.params,
        ...scopeF.params,
        ...temporal.params,
        limit,
      ) as any[];

    return rows.map((r, i) => ({
      ...r,
      rank: i + 1,
      strategy: 'semantic',
      // cosine similarity from cosine distance — see vec_distance_cosine's
      // native counterpart in engram-aql/src/vector/cosine.rs for the same
      // `1 - dot/(‖a‖·‖b‖)` formula this is the inverse of.
      cosine: 1 - r.distance,
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
  filters: QueryFilters,
): ScoredChunk[] {
  const { memoryTypes, minTrust = 0, sourceFilter, contextFilter } = filters;

  // FTS5 with BM25 ranking
  const typeFilter = memoryTypes?.length
    ? `AND c.memory_type IN (${inClausePlaceholders(memoryTypes)})`
    : '';

  // Build optional temporal filter
  const temporal = buildTemporalFilter(filters);
  const temporalSQL = temporal.conditions.length
    ? 'AND ' + temporal.conditions.join(' AND ')
    : '';
  const sourceType = buildSourceTypeFilter(filters);
  const scopeF = buildScopeFilter(filters);

  try {
    const rows = db
      .prepare(
        `
      SELECT c.id, c.text, c.memory_type, c.source, c.context, c.trust_score,
             c.source_type, c.event_time, c.created_at,
             rank AS bm25_rank
      FROM chunks_fts fts
      JOIN chunks c ON c.rowid = fts.rowid
      WHERE chunks_fts MATCH ?
        AND c.is_active = TRUE
        AND c.trust_score >= ?
        AND (? IS NULL OR c.source LIKE '%' || ? || '%')
        AND (? IS NULL OR c.context LIKE '%' || ? || '%')
        ${typeFilter}
        ${sourceType.sql}
        ${scopeF.sql}
        ${temporalSQL}
      ORDER BY rank
      LIMIT ?
    `,
      )
      .all(
        query,
        minTrust,
        sourceFilter ?? null,
        sourceFilter ?? null,
        contextFilter ?? null,
        contextFilter ?? null,
        ...(memoryTypes ?? []),
        ...sourceType.params,
        ...scopeF.params,
        ...temporal.params,
        limit,
      ) as any[];

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
  filters: QueryFilters,
): ScoredChunk[] {
  const { memoryTypes, minTrust = 0, sourceFilter, contextFilter } = filters;

  // Tokenize query, filtering stop words that cause false positives
  const queryTokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2 && !GRAPH_STOP_WORDS.has(t));

  if (queryTokens.length === 0) return [];

  try {
    // Expand tokens via known entity aliases (e.g., "IaC" → "infrastructure-as-code")
    const expandedTokens = [...queryTokens];
    for (const token of queryTokens) {
      const aliasMatches = db
        .prepare(
          `SELECT DISTINCT e.canonical_name
           FROM entities e, json_each(e.aliases) AS a
           WHERE e.is_active = TRUE AND LOWER(a.value) = ?`,
        )
        .all(token) as Array<{ canonical_name: string }>;
      for (const m of aliasMatches) {
        const name = m.canonical_name.toLowerCase();
        if (!expandedTokens.includes(name)) {
          expandedTokens.push(name);
        }
      }
    }

    // Find matching entities
    const likeClauses = expandedTokens
      .map(() => `(e.canonical_name LIKE ? OR e.aliases LIKE ?)`)
      .join(' OR ');
    const likeParams = expandedTokens.flatMap((t) => [`%${t}%`, `%${t}%`]);

    const matchedEntities = db
      .prepare(
        `
      SELECT e.id, e.canonical_name, e.mention_count
      FROM entities e
      WHERE e.is_active = TRUE AND (${likeClauses})
      ORDER BY e.mention_count DESC
      LIMIT 10
    `,
      )
      .all(...likeParams) as Array<{
      id: string;
      canonical_name: string;
      mention_count: number;
    }>;

    if (matchedEntities.length === 0) return [];

    const entityIds = matchedEntities.map((e) => e.id);

    // 1-hop: chunks directly mentioning these entities
    const placeholders = entityIds.map(() => '?').join(',');
    const typeFilter = memoryTypes?.length
      ? `AND c.memory_type IN (${inClausePlaceholders(memoryTypes)})`
      : '';

    // Build optional temporal filter
    const temporal = buildTemporalFilter(filters);
    const temporalSQL = temporal.conditions.length
      ? 'AND ' + temporal.conditions.join(' AND ')
      : '';
    const sourceType = buildSourceTypeFilter(filters);
    const scopeF = buildScopeFilter(filters);

    const directChunks = db
      .prepare(
        `
      SELECT DISTINCT c.id, c.text, c.memory_type, c.source, c.context, c.trust_score,
             c.source_type, c.event_time, c.created_at
      FROM chunk_entities ce
      JOIN chunks c ON ce.chunk_id = c.id
      WHERE ce.entity_id IN (${placeholders})
        AND c.is_active = TRUE
        AND c.trust_score >= ?
        AND (? IS NULL OR c.source LIKE '%' || ? || '%')
        AND (? IS NULL OR c.context LIKE '%' || ? || '%')
        ${typeFilter}
        ${sourceType.sql}
        ${scopeF.sql}
        ${temporalSQL}
      ORDER BY c.trust_score DESC, c.created_at DESC
      LIMIT ?
    `,
      )
      .all(
        ...entityIds,
        minTrust,
        sourceFilter ?? null,
        sourceFilter ?? null,
        contextFilter ?? null,
        contextFilter ?? null,
        ...(memoryTypes ?? []),
        ...sourceType.params,
        ...scopeF.params,
        ...temporal.params,
        limit,
      ) as any[];

    // 2-hop: chunks mentioning entities related to matched entities
    const relatedChunks = db
      .prepare(
        `
      SELECT DISTINCT c.id, c.text, c.memory_type, c.source, c.context, c.trust_score,
             c.source_type, c.event_time, c.created_at
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
        AND (? IS NULL OR c.source LIKE '%' || ? || '%')
        AND (? IS NULL OR c.context LIKE '%' || ? || '%')
        ${typeFilter}
        ${sourceType.sql}
        ${scopeF.sql}
        ${temporalSQL}
      ORDER BY r.confidence DESC, c.trust_score DESC
      LIMIT ?
    `,
      )
      .all(
        ...entityIds,
        ...entityIds,
        minTrust,
        ...directChunks.map((c: any) => c.id),
        sourceFilter ?? null,
        sourceFilter ?? null,
        contextFilter ?? null,
        contextFilter ?? null,
        ...(memoryTypes ?? []),
        ...sourceType.params,
        ...scopeF.params,
        ...temporal.params,
        Math.max(0, limit - directChunks.length),
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
  filters: QueryFilters,
): ScoredChunk[] {
  if (!filters.after && !filters.before) return [];

  const { sourceFilter, contextFilter } = filters;
  const minTrustVal =
    typeof filters.minTrust === 'number' ? filters.minTrust : 0;

  const conditions: string[] = ['c.is_active = TRUE', 'c.trust_score >= ?'];
  const params: any[] = [minTrustVal];

  if (filters.after) {
    conditions.push(
      '(datetime(c.event_time) >= datetime(?) OR (c.event_time IS NULL AND datetime(c.created_at) >= datetime(?)))',
    );
    params.push(filters.after, filters.after);
  }
  if (filters.before) {
    conditions.push(
      '(datetime(c.event_time) <= datetime(?) OR (c.event_time IS NULL AND datetime(c.created_at) <= datetime(?)))',
    );
    params.push(filters.before, filters.before);
  }
  if (filters.memoryTypes?.length) {
    conditions.push(
      `c.memory_type IN (${inClausePlaceholders(filters.memoryTypes)})`,
    );
    params.push(...filters.memoryTypes);
  }
  if (filters.sourceTypeIn?.length) {
    conditions.push(
      `c.source_type IN (${inClausePlaceholders(filters.sourceTypeIn)})`,
    );
    params.push(...filters.sourceTypeIn);
  }

  conditions.push(`(? IS NULL OR c.source LIKE '%' || ? || '%')`);
  params.push(sourceFilter ?? null, sourceFilter ?? null);

  conditions.push(`(? IS NULL OR c.context LIKE '%' || ? || '%')`);
  params.push(contextFilter ?? null, contextFilter ?? null);

  conditions.push(
    `c.scope IN (${inClausePlaceholders(filters.scope)}) AND (c.expires_at IS NULL OR datetime(c.expires_at) > datetime('now'))`,
  );
  params.push(...filters.scope);
  if (filters.parentRef) {
    conditions.push('c.parent_ref = ?');
    params.push(filters.parentRef);
  }

  try {
    const rows = db
      .prepare(
        `
      SELECT c.id, c.text, c.memory_type, c.source, c.context, c.trust_score,
             c.source_type, c.event_time, c.created_at
      FROM chunks c
      WHERE ${conditions.join(' AND ')}
      ORDER BY COALESCE(c.event_time, c.created_at) DESC
      LIMIT ?
    `,
      )
      .all(...params, limit) as any[];

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

/** Fused per-chunk accumulator. `perStrategy`/`rawScore`/`weighting` are only
 * populated when explainScores is requested — kept optional so the common
 * (non-explain) path allocates nothing extra. */
interface FusedEntry {
  score: number;
  strategies: string[];
  chunk: ScoredChunk;
  perStrategy?: StrategyScoreBreakdown[];
  rawScore?: number;
  weighting?: ScoreExplanation['weighting'];
  /**
   * Raw cosine similarity, carried over from the semantic strategy's
   * ScoredChunk when present. Drives cosine-primary within-tier scoring in
   * applyWeighting — see D6. Undefined when the chunk was never a semantic
   * hit, in which case the RRF-summed `score` above is the relevance signal.
   */
  cosine?: number;
}

function reciprocalRankFusion(
  strategyResults: ScoredChunk[][],
  k: number = 60,
  collectPerStrategy: boolean = false,
): Map<string, FusedEntry> {
  const fused = new Map<string, FusedEntry>();

  for (const results of strategyResults) {
    for (const chunk of results) {
      const existing = fused.get(chunk.id);
      const rrfScore = 1 / (k + chunk.rank);

      if (existing) {
        existing.score += rrfScore;
        if (chunk.cosine !== undefined) existing.cosine = chunk.cosine;
        if (!existing.strategies.includes(chunk.strategy)) {
          existing.strategies.push(chunk.strategy);
        }
        existing.perStrategy?.push({
          strategy: chunk.strategy,
          rank: chunk.rank,
          rrfScore,
        });
      } else {
        fused.set(chunk.id, {
          score: rrfScore,
          strategies: [chunk.strategy],
          chunk,
          cosine: chunk.cosine,
          perStrategy: collectPerStrategy
            ? [{ strategy: chunk.strategy, rank: chunk.rank, rrfScore }]
            : undefined,
        });
      }
    }
  }

  return fused;
}

function temporalDecayMultiplier(
  createdAt: string,
  halfLifeDays: number,
): number {
  if (halfLifeDays === 0) return 1.0;
  const ageMs = Date.now() - new Date(createdAt).getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  // Exponential decay: 1.0 at age=0, 0.5 at age=halfLife
  return Math.pow(2, -ageDays / halfLifeDays);
}

function applyWeighting(
  fused: Map<string, FusedEntry>,
  options: {
    decayHalfLifeDays?: number;
    sourceBoost?: { pattern: string; multiplier: number };
    contextBoost?: { pattern: string; multiplier: number };
    memoryRankOf?: (memoryType: string) => number;
    collectExplain?: boolean;
  } = {},
): void {
  const {
    decayHalfLifeDays = 180,
    sourceBoost,
    contextBoost,
    memoryRankOf,
    collectExplain = false,
  } = options;

  for (const [, entry] of fused) {
    // D6: within-tier relevance is cosine-primary when a semantic hit
    // contributed raw similarity magnitude — RRF's ordinal rank alone let a
    // strongly-relevant chunk lose to weakly-relevant ones because only
    // rank survived fusion, not the actual distance. Chunks never touched
    // by the semantic strategy have no cosine signal and keep the
    // RRF-summed score from fusion unchanged.
    const cosinePrimary = entry.cosine !== undefined;
    if (cosinePrimary) {
      entry.score = entry.cosine!;
    }
    // Trust: gentle tiebreak (0.94x-0.99x) when cosine is already carrying
    // the primary relevance signal — the old 0.6x-1.2x swing would swamp
    // it. Non-cosine (RRF-fallback) entries keep the original, coarser
    // trust weighting since RRF rank alone is a weaker relevance signal.
    const trustMultiplier = cosinePrimary
      ? 0.94 + entry.chunk.trust_score * 0.05
      : 0.6 + entry.chunk.trust_score * 0.6;
    // Multi-strategy bonus
    const strategyBoost = 1 + (entry.strategies.length - 1) * 0.1;
    // Temporal decay
    const decay = temporalDecayMultiplier(
      entry.chunk.created_at,
      decayHalfLifeDays,
    );
    // Source boost
    const srcMultiplier =
      sourceBoost && entry.chunk.source?.includes(sourceBoost.pattern)
        ? sourceBoost.multiplier
        : 1.0;
    // Context boost
    const ctxMultiplier =
      contextBoost && entry.chunk.context?.includes(contextBoost.pattern)
        ? contextBoost.multiplier
        : 1.0;
    // Memory-type weight — a gentle preference (world/observation slightly
    // up, opinion slightly down), not the hard floor this used to be. See
    // the comment on DEFAULT_MEMORY_TYPE_RANK.
    const memoryTypeMultiplier = memoryRankOf
      ? memoryTypeWeightFromRank(memoryRankOf(entry.chunk.memory_type))
      : 1.0;

    if (collectExplain) {
      entry.rawScore = entry.score;
      entry.weighting = {
        trust: trustMultiplier,
        strategyBoost,
        decay,
        sourceBoost: srcMultiplier,
        contextBoost: ctxMultiplier,
        memoryType: memoryTypeMultiplier,
      };
    }

    entry.score *=
      trustMultiplier *
      strategyBoost *
      decay *
      srcMultiplier *
      ctxMultiplier *
      memoryTypeMultiplier;
  }
}

// =============================================================================
// Insight (opinion / observation) selection
// =============================================================================

/** Max opinions or observations attached to one recall response. */
const INSIGHT_LIMIT = 5;

interface OpinionInsightRow {
  belief: string;
  confidence: number;
  domain: string | null;
  embedding?: Buffer | null;
}

interface ObservationInsightRow {
  summary: string;
  domain: string | null;
  topic: string | null;
  embedding?: Buffer | null;
}

interface SelectInsightsArgs<TRow, TOut> {
  db: Database.Database;
  /**
   * Normalized query terms (lowercased, punctuation-stripped, longer than 3
   * characters) used by the lexical path. Empty means the query carried no
   * usable term, in which case the lexical path matches nothing.
   */
  queryTerms: string[];
  table: 'opinions' | 'observations';
  /** Column holding the human-readable text (belief / summary). */
  textColumn: 'belief' | 'summary';
  /** Projection list for the SELECT, excluding `embedding`. */
  columns: string;
  /** Extra always-ANDed predicate beyond `is_active = TRUE`, if any. */
  extraWhere?: string;
  /** ORDER BY used on the lexical path only. */
  lexicalOrder: string;
  getQueryEmbedding: () => Promise<Float32Array>;
  insightMinScore: number;
  project: (row: TRow) => TOut;
}

/**
 * Select the insights attached to a recall response.
 *
 * Synthesized context is query-scoped: opinions and observations are generated
 * material, so they must never enter a prompt merely because the store holds
 * them. Both paths below can legitimately return nothing, and an empty list is
 * the correct answer for a query the store has no belief about.
 *
 * Two paths coexist:
 *
 * 1. **Semantic** — rows carrying an embedding are cosine-ranked against the
 *    query and gated by `insightMinScore`. This is the intended path.
 * 2. **Lexical** — `LIKE '%term%'` over the text column, ordered by
 *    confidence/recency. It ranks nothing by relevance: ordinary query words
 *    match a large fraction of beliefs, so its top-N is close to constant
 *    across unrelated questions. It survives only for rows with no embedding.
 *
 * Which runs depends on the store, so the improvement is monotonic and no
 * existing deployment regresses: a bank with nothing embedded behaves exactly
 * as before, a fully-embedded bank is purely semantic, and the mixed state a
 * backfill passes through puts semantic hits first and fills any remaining
 * slots lexically from the un-embedded rows.
 */
async function selectInsights<TRow extends { embedding?: Buffer | null }, TOut>(
  args: SelectInsightsArgs<TRow, TOut>,
): Promise<TOut[]> {
  const {
    db,
    queryTerms,
    table,
    textColumn,
    columns,
    extraWhere,
    lexicalOrder,
    getQueryEmbedding,
    insightMinScore,
    project,
  } = args;

  const baseWhere = `is_active = TRUE${extraWhere ? ` AND ${extraWhere}` : ''}`;

  const embeddedRows = db
    .prepare(
      `SELECT ${columns}, embedding FROM ${table}
       WHERE ${baseWhere} AND embedding IS NOT NULL`,
    )
    .all() as TRow[];

  const selected: TOut[] = [];
  const takenText = new Set<string>();

  if (embeddedRows.length > 0) {
    const queryEmbedding = await getQueryEmbedding();
    const scored: Array<{ row: TRow; score: number }> = [];
    for (const row of embeddedRows) {
      if (!row.embedding) continue;
      const score = cosineSimilarity(
        queryEmbedding,
        bufferToFloat32Array(row.embedding),
      );
      if (score >= insightMinScore) scored.push({ row, score });
    }
    scored.sort((a, b) => b.score - a.score);
    for (const { row } of scored.slice(0, INSIGHT_LIMIT)) {
      takenText.add(String((row as Record<string, unknown>)[textColumn]));
      selected.push(project(row));
    }
    if (selected.length >= INSIGHT_LIMIT) return selected;
  }

  // Lexical path — for the un-embedded remainder (or the whole table, on a
  // store with no embeddings at all).
  if (queryTerms.length === 0) return selected;
  const embeddingClause =
    embeddedRows.length > 0 ? ' AND embedding IS NULL' : '';
  const conditions = queryTerms
    .map(() => `LOWER(${textColumn}) LIKE ?`)
    .join(' OR ');
  const rows = db
    .prepare(
      `SELECT ${columns} FROM ${table}
       WHERE ${baseWhere}${embeddingClause} AND (${conditions})
       ORDER BY ${lexicalOrder}
       LIMIT ${INSIGHT_LIMIT - selected.length}`,
    )
    .all(...queryTerms.map((t) => `%${t}%`)) as TRow[];
  for (const row of rows) {
    const text = String((row as Record<string, unknown>)[textColumn]);
    if (takenText.has(text)) continue;
    takenText.add(text);
    selected.push(project(row));
  }
  return selected;
}

// =============================================================================
// Main Recall Function
// =============================================================================

export async function recall(
  db: Database.Database,
  query: string,
  embedder: EmbeddingProvider,
  options: RecallOptions = {},
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
    sourceFilter,
    contextFilter,
    sourceBoost,
    contextBoost,
    decayHalfLifeDays = 180,
    sourceTiers,
    memoryTypeRank,
    scope = ['durable'],
    parentRef,
    minScore,
    insightMinScore = 0.45,
    explainScores = false,
  } = options;

  // The query embedding is needed by the semantic strategy AND by insight
  // (opinion/observation) ranking, which are independently switchable — so it
  // is computed lazily and memoized rather than up front. A caller that
  // disables the semantic strategy and both insight kinds still pays nothing.
  let queryEmbeddingPromise: Promise<Float32Array> | null = null;
  const getQueryEmbedding = (): Promise<Float32Array> => {
    queryEmbeddingPromise ??=
      'embedQuery' in embedder
        ? (
            embedder as EmbeddingProvider & {
              embedQuery: (t: string) => Promise<Float32Array>;
            }
          ).embedQuery(query)
        : embedder.embed(query);
    return queryEmbeddingPromise;
  };

  const tiers = { ...DEFAULT_SOURCE_TIERS, ...sourceTiers };
  const tierOf = (sourceType: string): number =>
    tiers[sourceType] ?? UNKNOWN_SOURCE_TIER;
  const tierZeroTypes = Object.keys(tiers).filter((t) => tiers[t] === 0);

  const memoryRanks = { ...DEFAULT_MEMORY_TYPE_RANK, ...memoryTypeRank };
  const memoryRankOf = (memoryType: string): number =>
    memoryRanks[memoryType] ?? UNKNOWN_MEMORY_TYPE_RANK;

  // Auto-derive temporal bounds from natural language when not explicitly provided
  const parsed = !after && !before ? parseTemporalQuery(query) : null;
  const effectiveAfter = after ?? parsed?.after;
  const effectiveBefore = before ?? parsed?.before;

  const perStrategyLimit = topK * 3; // Oversample per strategy, then fuse
  const filters: QueryFilters = {
    memoryTypes,
    minTrust,
    sourceFilter,
    contextFilter,
    after: effectiveAfter,
    before: effectiveBefore,
    scope,
    parentRef,
  };
  const strategyResults: ScoredChunk[][] = [];
  const strategiesUsed: string[] = [];

  // Truncation guard: when a strategy's candidate list is full (hit its SQL
  // LIMIT), sheer volume of lower-tier content may have crowded tier-0 rows
  // out of the candidate set before the tier floor could protect them.
  // Re-run the strategy restricted to tier-0 source types and append any
  // rows not already present. The appended rank only orders results within
  // their tier — the lexicographic final sort guarantees tier-0 survival.
  const withTierZeroReserve = (
    run: (f: QueryFilters, limit: number) => ScoredChunk[],
  ): ScoredChunk[] => {
    const results = run(filters, perStrategyLimit);
    if (tierZeroTypes.length === 0 || results.length < perStrategyLimit) {
      return results;
    }
    const have = new Set(results.map((r) => r.id));
    const reserve = run(
      { ...filters, sourceTypeIn: tierZeroTypes },
      topK,
    ).filter((r) => !have.has(r.id));
    return [
      ...results,
      ...reserve.map((r, i) => ({ ...r, rank: results.length + i + 1 })),
    ];
  };

  // Run strategies
  if (strategies.includes('semantic')) {
    const queryEmbedding = await getQueryEmbedding();
    const results = withTierZeroReserve((f, limit) =>
      semanticSearch(db, queryEmbedding, limit, f),
    );
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('semantic');
    }
  }

  // Sanitize for strategies that use raw text (FTS5 MATCH and graph tokenizer
  // choke on punctuation like '?', '*', '"' etc.). Keyword search gets the
  // phrase-aware variant (quoted segments become FTS5 phrase terms); graph
  // search keeps the plain strip-everything variant — it has no concept of
  // FTS5 phrase syntax and a stray quote character would corrupt its
  // whitespace tokenization.
  const sanitized = sanitizeQuery(query);
  const ftsQuery = sanitizeQueryForFts(query);

  if (strategies.includes('keyword') && ftsQuery) {
    const results = withTierZeroReserve((f, limit) =>
      keywordSearch(db, ftsQuery, limit, f),
    );
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('keyword');
    }
  }

  if (strategies.includes('graph') && sanitized) {
    const results = withTierZeroReserve((f, limit) =>
      graphSearch(db, sanitized, limit, f),
    );
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('graph');
    }
  }

  if (strategies.includes('temporal') && (effectiveAfter || effectiveBefore)) {
    const results = withTierZeroReserve((f, limit) =>
      temporalSearch(db, limit, f),
    );
    if (results.length > 0) {
      strategyResults.push(results);
      strategiesUsed.push('temporal');
    }
  }

  // Fuse results
  const fused = reciprocalRankFusion(strategyResults, rrfK, explainScores);
  applyWeighting(fused, {
    decayHalfLifeDays,
    sourceBoost,
    contextBoost,
    memoryRankOf,
    collectExplain: explainScores,
  });

  // minScore filters the fused set post-weighting — same `score` field
  // callers see — before the tier-major sort and topK truncation.
  const candidates =
    minScore !== undefined
      ? [...fused.values()].filter((entry) => entry.score >= minScore)
      : [...fused.values()];

  // Sort by (source tier, fused score) and take top K. The tier is a
  // structural floor: no amount of trust or relevance lets a lower-tier
  // chunk outrank a higher-tier one — provenance (was this actually stated,
  // vs. inferred, vs. a raw tool result) is the one thing worth an absolute
  // floor. Memory-type is NOT a floor anymore — it's a gentle multiplier
  // inside the score itself (see applyWeighting / memoryTypeWeightFromRank),
  // so a strong 'experience' match can still beat a weak 'observation' or
  // 'world' match instead of losing to it categorically. Sorting the FULL
  // fused set before truncation means tier-0 matches always survive the
  // topK cut.
  const sorted = candidates
    .sort((a, b) => {
      const tierDiff =
        tierOf(a.chunk.source_type) - tierOf(b.chunk.source_type);
      if (tierDiff !== 0) return tierDiff;
      return b.score - a.score;
    })
    .slice(0, topK);

  const results: RecallResult[] = sorted.map((entry) => {
    const result: RecallResult = {
      id: entry.chunk.id,
      text:
        entry.chunk.text.length > snippetChars
          ? entry.chunk.text.substring(0, snippetChars) + '...'
          : entry.chunk.text,
      memoryType: entry.chunk.memory_type,
      source: entry.chunk.source,
      trustScore: entry.chunk.trust_score,
      sourceType: entry.chunk.source_type,
      eventTime: entry.chunk.event_time,
      createdAt: entry.chunk.created_at,
      score: entry.score,
      strategies: entry.strategies,
    };
    if (explainScores && entry.perStrategy && entry.weighting) {
      result.strategyScores = {
        perStrategy: entry.perStrategy,
        rawFusedScore: entry.rawScore ?? entry.score,
        weighting: entry.weighting,
      };
    }
    return result;
  });

  // Synthesized context is query-scoped: unlike raw chunks, opinions and
  // observations are generated material and must never enter a prompt merely
  // because no lexical match was found. Normalize before matching so ordinary
  // punctuation ("Terraform?", "Rust's") cannot suppress a valid hit.
  const synthesizedQueryTerms = [
    ...new Set(
      query
        .toLocaleLowerCase()
        .replace(/[^\p{L}\p{N}]+/gu, ' ')
        .split(/\s+/)
        .filter((term) => term.length > 3),
    ),
  ];

  // Gather relevant opinions.
  //
  // Semantic-primary when the store carries belief embeddings, lexical when it
  // doesn't — see selectInsights().
  const opinions = includeOpinions
    ? await selectInsights<OpinionInsightRow, RecallOpinion>({
        db,
        queryTerms: synthesizedQueryTerms,
        table: 'opinions',
        textColumn: 'belief',
        columns: 'belief, confidence, domain',
        // Weakly-held beliefs are gated out of recall (introspect() is the
        // no-floor surface); unchanged from the pre-embedding behavior.
        extraWhere: 'confidence >= 0.5',
        lexicalOrder: 'confidence DESC',
        getQueryEmbedding,
        insightMinScore,
        project: ({ belief, confidence, domain }) => ({
          belief,
          confidence,
          domain,
        }),
      })
    : [];

  // Gather relevant observations (same two-path selection as opinions).
  const observations = includeObservations
    ? await selectInsights<ObservationInsightRow, RecallObservation>({
        db,
        queryTerms: synthesizedQueryTerms,
        table: 'observations',
        textColumn: 'summary',
        columns: 'summary, domain, topic',
        lexicalOrder: 'last_refreshed DESC, synthesized_at DESC',
        getQueryEmbedding,
        insightMinScore,
        project: ({ summary, domain, topic }) => ({ summary, domain, topic }),
      })
    : [];

  return {
    results,
    opinions,
    observations,
    totalCandidates: fused.size,
    strategiesUsed,
  };
}

// =============================================================================
// Prompt Formatting Helper
// =============================================================================

export interface FormatForPromptOptions {
  /** Max characters for the entire block (default: 2000) */
  maxChars?: number;
  /** Include trust scores inline (default: false) */
  showTrust?: boolean;
  /** Include source attribution (default: true) */
  showSource?: boolean;
  /**
   * Append a compact provenance bracket to each result line — memory type,
   * source type, trust, and created date, e.g.
   * `[world/user_stated, trust 0.85, 2026-03-14]`. Makes the trust-tier
   * guarantee and staleness visible to any consumer of this formatter, not
   * just ones that inspect RecallResult fields directly. Default: false —
   * existing output is byte-identical when omitted.
   */
  showProvenance?: boolean;
  /**
   * Append a terse `  why: ...` line under each result derived from its
   * `strategyScores` (only present when the recall() call passed
   * `explainScores: true`): the semantic strategy's contribution renders as
   * its cosine score, every other strategy as its rank ordinal, plus the
   * source tier the result sorted under, e.g.
   * `  why: semantic 0.82 · keyword r3 · tier 0`. Silent no-op — no line, no
   * warning — when a result has no `strategyScores`. Default: false.
   */
  showWhy?: boolean;
  /** Header text (default: "## Relevant Memory Context") */
  header?: string;
}

/** YYYY-MM-DD from a stored TIMESTAMP. SQLite's CURRENT_TIMESTAMP format
 * ("YYYY-MM-DD HH:MM:SS") and JS's Date#toISOString() ("YYYY-MM-DDTHH:...")
 * both start with the date, so a plain slice works for either. */
function formatProvenanceDate(createdAt: string): string {
  return createdAt.slice(0, 10);
}

/**
 * Terse per-result "why" line from strategyScores. The semantic strategy's
 * entry renders as `rawFusedScore` (cosine-primary once a chunk is a
 * semantic hit — see D6); every other strategy has no relevance magnitude,
 * only rank, so it renders as `r{rank}`. Tier uses DEFAULT_SOURCE_TIERS only
 * — formatForPrompt has no access to a per-call `sourceTiers` override
 * passed to recall(), so this is illustrative, not authoritative.
 */
export function formatWhyLine(result: RecallResult): string | null {
  const scores = result.strategyScores;
  if (!scores) return null;
  const parts = scores.perStrategy.map((s) =>
    s.strategy === 'semantic'
      ? `semantic ${scores.rawFusedScore.toFixed(2)}`
      : `${s.strategy} r${s.rank}`,
  );
  const tier = DEFAULT_SOURCE_TIERS[result.sourceType] ?? UNKNOWN_SOURCE_TIER;
  parts.push(`tier ${tier}`);
  return `  why: ${parts.join(' · ')}`;
}

/**
 * Format a RecallResponse into a string suitable for system prompt injection.
 * Handles token budgeting, prioritization, and clean formatting.
 *
 * Priority order: opinions (highest signal) → observations → memory results.
 * Stops adding content once maxChars would be exceeded, with a truncation notice.
 */
export function formatForPrompt(
  response: RecallResponse,
  options: FormatForPromptOptions = {},
): string {
  const {
    maxChars = 2000,
    showTrust = false,
    showSource = true,
    showProvenance = false,
    showWhy = false,
    header = '## Relevant Memory Context',
  } = options;

  const lines: string[] = [];
  let charCount = 0;

  const tryAdd = (line: string): boolean => {
    const cost = line.length + 1; // +1 for newline
    if (charCount + cost > maxChars) return false;
    lines.push(line);
    charCount += cost;
    return true;
  };

  tryAdd(header);
  tryAdd('');

  // 1. Opinions — highest signal, most condensed
  // Disclaimer + confidence cap prevent opinion feedback loops:
  // without these, opinions injected into LLM prompts get reinforced to max confidence
  if (response.opinions.length > 0) {
    if (!tryAdd('### Beliefs (agent-synthesized, not ground truth)'))
      return lines.join('\n');
    tryAdd('');
    let omitted = 0;
    for (const o of response.opinions) {
      const displayConf = Math.min(o.confidence, 0.85);
      const conf = `${(displayConf * 100).toFixed(0)}%`;
      const domain = o.domain ? ` (${o.domain})` : '';
      if (!tryAdd(`- [${conf}] ${o.belief}${domain}`)) {
        omitted++;
      }
    }
    if (omitted > 0)
      tryAdd(`(${omitted} belief${omitted > 1 ? 's' : ''} omitted)`);
    tryAdd('');
  }

  // 2. Observations
  if (response.observations.length > 0) {
    if (!tryAdd('### Observations')) return lines.join('\n');
    tryAdd('');
    let omitted = 0;
    for (const o of response.observations) {
      const topic = o.topic ? ` (${o.topic})` : '';
      if (!tryAdd(`- ${o.summary}${topic}`)) {
        omitted++;
      }
    }
    if (omitted > 0)
      tryAdd(`(${omitted} observation${omitted > 1 ? 's' : ''} omitted)`);
    tryAdd('');
  }

  // 3. Memory results
  if (response.results.length > 0) {
    if (!tryAdd('### Memory')) return lines.join('\n');
    tryAdd('');
    let omitted = 0;
    for (const r of response.results) {
      const trust = showTrust ? `[trust ${r.trustScore.toFixed(2)}] ` : '';
      const src = showSource && r.source ? ` [${r.source}]` : '';
      const provenance = showProvenance
        ? ` [${r.memoryType}/${r.sourceType}, trust ${r.trustScore.toFixed(2)}, ${formatProvenanceDate(r.createdAt)}]`
        : '';
      if (!tryAdd(`- ${trust}${r.text}${src}${provenance}`)) {
        omitted++;
        continue;
      }
      if (showWhy) {
        const whyLine = formatWhyLine(r);
        if (whyLine) tryAdd(whyLine);
      }
    }
    if (omitted > 0)
      tryAdd(`(${omitted} result${omitted > 1 ? 's' : ''} omitted)`);
  }

  return lines.join('\n');
}
