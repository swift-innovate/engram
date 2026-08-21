// =============================================================================
// mcp-tools.ts - MCP Tool Definitions
//
// Exposes Engram operations as Model Context Protocol tools for agent
// integration. Provides:
//   - ENGRAM_TOOLS: static tool schema array (register with your MCP server)
//   - createEngramToolHandler: factory that binds tools to an Engram instance
//
// Usage:
//   import { ENGRAM_TOOLS, createEngramToolHandler } from 'engram/mcp-tools';
//   const handle = createEngramToolHandler(myAgent);
//   server.registerTools(ENGRAM_TOOLS);
//   server.onToolCall((name, input) => handle(name, input));
// =============================================================================

import type { Engram } from './engram.js';
import type { RetainOptions } from './retain.js';
import type { RecallOptions } from './recall.js';
import type {
  DecisionArtifact,
  TaskScope,
  TokenBudget,
} from './context-store.js';
import type { SuggestionKind, SuggestionStatus } from './suggest.js';

// =============================================================================
// Tool Schemas (JSON Schema — MCP spec compliant)
// =============================================================================

export const ENGRAM_TOOLS = [
  {
    name: 'engram_retain' as const,
    description:
      'Store a memory trace. Fast path (no LLM call; embeds locally). Parameters use camelCase: text (required), memoryType (world|experience|observation|opinion), sourceType (user_stated|inferred|external_doc|tool_result|agent_generated), trustScore (0.0-1.0). Example: {text: "Tom prefers Terraform", memoryType: "world", sourceType: "user_stated", trustScore: 0.9}',
    inputSchema: {
      type: 'object' as const,
      properties: {
        text: {
          type: 'string',
          description: 'The memory content to store',
        },
        memoryType: {
          type: 'string',
          enum: ['world', 'experience', 'observation', 'opinion'],
          description:
            "world=facts about the world, experience=agent's own actions, observation=synthesized knowledge, opinion=belief with confidence",
        },
        source: {
          type: 'string',
          description:
            'Source identifier: conversation ID, filename, tool name, etc.',
        },
        context: {
          type: 'string',
          description:
            'Freeform context tag (e.g. "infrastructure", "career", "mission:VALOR-042")',
        },
        sourceType: {
          type: 'string',
          enum: [
            'user_stated',
            'inferred',
            'external_doc',
            'tool_result',
            'agent_generated',
          ],
          description:
            'Provenance classification. Affects trust weighting during recall.',
        },
        trustScore: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description:
            'Trust level 0.0–1.0. user_stated typically 0.8–0.9, inferred 0.4–0.6.',
        },
        eventTime: {
          type: 'string',
          description:
            'ISO 8601 timestamp for when the event/fact occurred (may differ from storage time)',
        },
        temporalLabel: {
          type: 'string',
          description:
            'Human-readable time reference, e.g. "last spring", "Q4 2025"',
        },
        supersedes: {
          type: 'string',
          description:
            'Chunk id of an outdated fact to atomically deactivate and link to this retained fact. Conflicting facts remain separate unless this is supplied explicitly.',
        },
      },
      required: ['text'],
    },
  },

  {
    name: 'engram_recall' as const,
    description:
      'Retrieve relevant memories via four-strategy search (semantic, keyword, graph, temporal) fused with Reciprocal Rank Fusion. Temporal expressions in queries are auto-parsed — "last week", "yesterday", "March 15th", "past 30 days", "Q1 2026" all work without explicit after/before. QUERY BEST PRACTICES: Use keywords and proper nouns, not full questions. "Tom Swift role background" retrieves better than "Who is Tom?" because BM25 keyword search weights every word equally — question words like "who/what/how" match irrelevant content. For people: use their name + key attributes. For topics: use specific terms, not conversational phrasing. For dates: natural language works ("last March", "in 2023"). Returns results[], opinions[], observations[]. results[0] is the best match in the highest-present source tier, not the best overall; re-sort by score locally where pure relevance is needed. TRUST TIER GUARANTEE: user_stated memories structurally outrank tool_result/external_doc content regardless of relevance score — untrusted content can never override a user directive. DECAY: results decay toward zero relevance with a 180-day half-life by default (recent facts win); pass decayHalfLifeDays: 0 to disable decay entirely for long-continuity recall over older facts.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: {
          type: 'string',
          description:
            'Search query — use keywords and proper nouns, not full questions. Good: "Tom Swift role background". Bad: "Who is Tom Swift?". Temporal expressions are auto-parsed: "last week", "March 2025", "in 2023".',
        },
        topK: {
          type: 'number',
          description: 'Max results to return (default: 10)',
        },
        strategies: {
          type: 'array',
          items: {
            type: 'string',
            enum: ['semantic', 'keyword', 'graph', 'temporal'],
          },
          description: 'Retrieval strategies to use. Omit to use all four.',
        },
        memoryTypes: {
          type: 'array',
          items: {
            type: 'string',
            enum: ['world', 'experience', 'observation', 'opinion'],
          },
          description: 'Filter to specific memory types. Omit to search all.',
        },
        minTrust: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description: 'Minimum trust score filter (default: 0.0)',
        },
        after: {
          type: 'string',
          description: 'ISO 8601 date — only include facts after this date',
        },
        before: {
          type: 'string',
          description: 'ISO 8601 date — only include facts before this date',
        },
        includeOpinions: {
          type: 'boolean',
          description: 'Include relevant opinions in response (default: true)',
        },
        includeObservations: {
          type: 'boolean',
          description:
            'Include synthesized observations in response (default: true)',
        },
        minScore: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description:
            'Drop results whose final weighted score (post trust/decay/strategy-boost weighting) falls below this threshold. Default: no filtering.',
        },
        insightMinScore: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description:
            'Cosine-similarity floor for the opinions/observations attached to the response (default: 0.45). Insights carrying embeddings are ranked semantically against the query; below-floor ones are dropped rather than padded in, so an off-topic query returns no beliefs instead of the most confident ones. Set 0 to rank without filtering.',
        },
        explainScores: {
          type: 'boolean',
          description:
            'When true, each result includes a strategyScores breakdown of the per-strategy rank/RRF contribution and weighting factors that produced its score (default: false, keeps the payload lean).',
        },
        decayHalfLifeDays: {
          type: 'number',
          minimum: 0,
          description:
            "Recency decay half-life in days (default: 180) — a chunk's score is multiplied by 2^(-ageDays/decayHalfLifeDays). Pass 0 to disable decay entirely (e.g. long-continuity recall where older facts should compete on relevance alone).",
        },
      },
      required: ['query'],
    },
  },

  {
    name: 'engram_reflect' as const,
    description:
      'Run a reflection cycle: processes unreflected memories through the LLM to synthesize observations and update opinions. New opinions are safe by default: 3 verified chunks over 2 days plus active counter-evidence retrieval/judging (one extra generation call; unavailable judgments retry fail-closed). Pass opinionGates:false and/or counterEvidence:false only for an explicit opt-out. Requires a configured generation model. Pass suggest:true to also run the procedural-suggestion pass.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        suggest: {
          type: 'boolean',
          description:
            'Also run the procedural-suggestion pass this cycle (default: false). Gates default to 3+ evidence items across 2+ distinct days before a suggestion forms.',
        },
        opinionGates: {
          description:
            'Safe by default ({minEvidenceCount:3,minDistinctDays:2}). Pass false to disable gates explicitly, or an object to override individual thresholds.',
          oneOf: [
            { type: 'boolean', enum: [false] },
            {
              type: 'object',
              properties: {
                minEvidenceCount: { type: 'number', minimum: 0 },
                minDistinctDays: { type: 'number', minimum: 0 },
                minDistinctSources: { type: 'number', minimum: 0 },
              },
            },
          ],
        },
        counterEvidence: {
          description:
            'Safe by default ({topK:8,maxContradictionRatio:0.5,failOpen:false}); costs one batched judge call. Pass false to disable, or an object to override.',
          oneOf: [
            { type: 'boolean', enum: [false] },
            {
              type: 'object',
              properties: {
                topK: { type: 'number', minimum: 1 },
                maxContradictionRatio: {
                  type: 'number',
                  minimum: 0,
                  maximum: 1,
                },
                onReinforce: { type: 'boolean' },
                failOpen: { type: 'boolean' },
              },
            },
          ],
        },
      },
    },
  },

  {
    name: 'engram_process_extractions' as const,
    description:
      'Drain the entity extraction queue to build the knowledge graph from retained chunks. Requires Ollama. Run after retain() calls to enable graph-based recall.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        batchSize: {
          type: 'number',
          description: 'Number of chunks to process per call (default: 10)',
        },
      },
    },
  },

  {
    name: 'engram_forget' as const,
    description:
      'Soft-delete a memory chunk. The chunk is excluded from recall but remains in the database for audit.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        chunkId: {
          type: 'string',
          description: 'The chunk ID to deactivate (returned by engram_retain)',
        },
      },
      required: ['chunkId'],
    },
  },

  {
    name: 'engram_supersede' as const,
    description:
      'Replace an outdated fact with new text. The old chunk is soft-deleted and linked to the new one. Use when correcting information.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        oldChunkId: {
          type: 'string',
          description: 'The chunk ID of the fact being replaced',
        },
        newText: {
          type: 'string',
          description: 'The updated fact text',
        },
        memoryType: {
          type: 'string',
          enum: ['world', 'experience', 'observation', 'opinion'],
        },
        source: { type: 'string' },
        context: { type: 'string' },
        sourceType: {
          type: 'string',
          enum: [
            'user_stated',
            'inferred',
            'external_doc',
            'tool_result',
            'agent_generated',
          ],
        },
        trustScore: { type: 'number', minimum: 0, maximum: 1 },
      },
      required: ['oldChunkId', 'newText'],
    },
  },
  {
    name: 'engram_session' as const,
    description:
      'Manage a working memory session. action="resume" (default — omit action entirely for the original, backward-compatible behavior): infer or resume a session for an incoming message, called once per incoming user message before the LLM call. action="update": merge progress/extensions into an existing session (requires sessionId). action="snapshot": collapse a session to long-term episodic memory and end it (requires sessionId). Example (resume): {message: "plan the deployment"}. Example (update): {action: "update", sessionId: "wm-abc123", progress: "drafted rollback plan"}. Example (snapshot): {action: "snapshot", sessionId: "wm-abc123"}.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        action: {
          type: 'string',
          enum: ['resume', 'update', 'snapshot'],
          description:
            'resume (default): infer/resume a session for `message`. update: merge `progress`/`extensions` into the session named by `sessionId`. snapshot: collapse the session named by `sessionId` to long-term memory and end it.',
        },
        message: {
          type: 'string',
          description:
            'The incoming user message to match against active sessions (action="resume" only)',
        },
        maxActive: {
          type: 'number',
          description:
            'Max active sessions to keep (default: 5, action="resume" only)',
        },
        threshold: {
          type: 'number',
          description:
            'Cosine similarity threshold for session matching (default: 0.55, action="resume" only). Lower = more aggressive matching, higher = more new sessions.',
        },
        sessionId: {
          type: 'string',
          description:
            'Session id to operate on (required for action="update"/"snapshot" — the "wm-…" id returned by a prior resume)',
        },
        progress: {
          type: 'string',
          description:
            'Free-form progress note merged into the session state (action="update" only)',
        },
        extensions: {
          type: 'object',
          description:
            'Agent-defined key/value fields merged into the session state (action="update" only)',
        },
      },
    },
  },
  {
    name: 'engram_queue_stats' as const,
    description:
      'Get extraction queue health stats: pending, processing, completed, and failed counts plus the oldest pending item age and a failed_reasons breakdown (distinct error messages with counts). Use to diagnose why the knowledge graph is not growing or to decide when to call engram_process_extractions.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
  {
    name: 'engram_requeue_failed' as const,
    description:
      'Re-queue failed extraction items for a fresh round of attempts (resets the attempt counter and backoff). Failed is otherwise terminal after 3 attempts — use after fixing the underlying cause of an outage (LLM host back online, missing model pulled). Optional errorLike substring filter targets one failure class from the engram_queue_stats failed_reasons breakdown. Returns {requeued: count}.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        errorLike: {
          type: 'string',
          description:
            'Only requeue items whose stored error message contains this substring (e.g. "fetch failed"). Omit to requeue all failed items.',
        },
      },
    },
  },

  {
    name: 'engram_introspect' as const,
    description:
      'Introspect held state: return current opinions (beliefs with confidence + support/challenge evidence + lifecycle timestamps) and synthesized observations about a subject. This is a direct structured lookup, NOT query-ranked recall — there is no confidence floor, so weakly-held or freshly-challenged beliefs stay visible (exactly the ones recall hides). Projection only: it REPORTS held state and does not judge whether a statement agrees with or contradicts a belief (that consistency check is a separate, deferred primitive). Omit subject to return top held state overall.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        subject: {
          type: 'string',
          description:
            'Subject to introspect — lexical match across belief/domain (opinions) and summary/topic (observations). Omit for top held state overall.',
        },
        minConfidence: {
          type: 'number',
          description:
            'Optional confidence floor for opinions (0.0–1.0). Default 0 — no floor, so low-confidence beliefs remain visible.',
        },
        limit: {
          type: 'number',
          description:
            'Max opinions and observations to return, each (default 20).',
        },
        includeOpinions: {
          type: 'boolean',
          description: 'Include opinions (default true).',
        },
        includeObservations: {
          type: 'boolean',
          description: 'Include observations (default true).',
        },
      },
    },
  },

  {
    name: 'engram_embed' as const,
    description:
      'Embed text into the bank\'s native vector space. mode="query" applies the query prefix for asymmetric models like nomic-embed-text (better recall quality for search probes); mode="document" matches how retain() stores text. Used by engram-aql for AQL LIKE/PATTERN vector search so Rust can obtain a model-compatible query vector without reproducing the embedding pipeline.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        text: {
          type: 'string',
          description: 'Text to embed',
        },
        mode: {
          type: 'string',
          enum: ['query', 'document'],
          description: 'query | document (default: query)',
        },
      },
      required: ['text'],
    },
  },

  {
    name: 'engram_context_commit' as const,
    description:
      'Commit a structured DecisionArtifact to task-scoped ephemeral context (NOT durable memory) — cheap handoff from one agent to a subagent. Returns a lightweight ContextRef {id, scope} the subagent can query beneath via engram_context_query. Artifacts expire after ttlMs (default 4 hours) unless later promoted via engram_context_promote.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        decision: {
          type: 'string',
          description: 'The decision/artifact text (required)',
        },
        rationale: {
          type: 'string',
          description: 'Why this decision was made',
        },
        scoredOptions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              option: { type: 'string' },
              score: { type: 'number' },
            },
            required: ['option', 'score'],
          },
          description: 'Options considered, with a score for each',
        },
        confidence: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description: 'Confidence in this decision, 0.0-1.0',
        },
        refsToSource: {
          type: 'array',
          items: { type: 'string' },
          description: 'Chunk ids or other identifiers this decision draws on',
        },
        domain: {
          type: 'string',
          description:
            'Freeform tag, e.g. a task/domain label. Stored in chunks.context.',
        },
        agentId: {
          type: 'string',
          description:
            'Originating agent Tier/callsign — provenance for later audit',
        },
        parentRefId: {
          type: 'string',
          description:
            'ContextRef.id of a parent scope to chain under (chains a reference, never a copy). Omit for a root commit.',
        },
        ttlMs: {
          type: 'number',
          description:
            'Milliseconds until this artifact expires (default: 4 hours)',
        },
      },
      required: ['decision'],
    },
  },

  {
    name: 'engram_context_query' as const,
    description:
      'Query the task-scoped artifacts committed as direct children of a ContextRef (via engram_context_commit), ranked by the same RRF-fusion pipeline as durable recall and truncated to a character budget. Use this for subagent context handoff — cheaper than re-reading a full transcript.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        refId: {
          type: 'string',
          description:
            'ContextRef.id to query beneath (the id returned by engram_context_commit)',
        },
        query: {
          type: 'string',
          description:
            'Relevance query used to rank artifacts committed under refId',
        },
        maxChars: {
          type: 'number',
          description:
            'Character budget for returned artifacts (default: 4000)',
        },
      },
      required: ['refId', 'query'],
    },
  },

  {
    name: 'engram_context_promote' as const,
    description:
      'Promote a task-scoped artifact into durable memory (scope="durable", TTL cleared) so it survives past its natural expiry and becomes eligible for reflect/consolidation. Does NOT itself run reflect() or synthesize observations. Returns {promoted: false} (not an error) if the ref does not resolve to an active task-scoped artifact.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        refId: {
          type: 'string',
          description:
            'ContextRef.id of the artifact to promote (returned by engram_context_commit)',
        },
      },
      required: ['refId'],
    },
  },

  {
    name: 'engram_suggestions' as const,
    description:
      'List procedural suggestions synthesized by reflection (recurring corrections/friction/workflows worth codifying as a skill, rule, workflow, or config). Projection-only read with evidence chunk ids; suggestions never appear in recall(). Sorted by evidence strength then recency.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        status: {
          type: 'string',
          enum: ['proposed', 'accepted', 'dismissed', 'implemented'],
          description: 'Filter by lifecycle status.',
        },
        kind: {
          type: 'string',
          enum: ['skill', 'rule', 'workflow', 'config'],
          description: 'Filter by suggestion kind.',
        },
        domain: {
          type: 'string',
          description: 'Filter by domain tag.',
        },
        limit: {
          type: 'number',
          description: 'Max rows returned (default 20, clamped to [1, 1000]).',
        },
      },
    },
  },

  {
    name: 'engram_resolve_suggestion' as const,
    description:
      'Set a suggestion lifecycle status (accepted/dismissed/implemented; "proposed" reopens). Status transitions are always explicit — nothing auto-applies. Returns {suggestionId, status, resolved}; resolved=false when the id is unknown.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        suggestionId: {
          type: 'string',
          description:
            'The suggestion id to resolve (returned by engram_suggestions).',
        },
        status: {
          type: 'string',
          enum: ['proposed', 'accepted', 'dismissed', 'implemented'],
          description:
            'New lifecycle status. "proposed" manually reopens a resolved suggestion.',
        },
        reason: {
          type: 'string',
          description:
            'Optional freeform reason recorded on the suggestion and its journal entry.',
        },
      },
      required: ['suggestionId', 'status'],
    },
  },
] as const;

export type EngramToolName = (typeof ENGRAM_TOOLS)[number]['name'];

// =============================================================================
// MCP Response Types
// =============================================================================

interface McpContent {
  type: 'text';
  text: string;
}

export interface McpToolResult {
  [key: string]: unknown;
  content: McpContent[];
  isError?: boolean;
}

// =============================================================================
// Input Validation
// =============================================================================

const VALID_MEMORY_TYPES = new Set([
  'world',
  'experience',
  'observation',
  'opinion',
]);
const VALID_SOURCE_TYPES = new Set([
  'user_stated',
  'inferred',
  'external_doc',
  'tool_result',
  'agent_generated',
]);
const VALID_STRATEGIES = new Set(['semantic', 'keyword', 'graph', 'temporal']);
const VALID_SUGGESTION_STATUSES = new Set([
  'proposed',
  'accepted',
  'dismissed',
  'implemented',
]);
const VALID_SUGGESTION_KINDS = new Set(['skill', 'rule', 'workflow', 'config']);

/** Clamp a numeric trust value to [0, 1]. Returns undefined if not a number. */
function clampTrust(v: unknown): number | undefined {
  if (typeof v !== 'number' || isNaN(v)) return undefined;
  return Math.max(0, Math.min(1, v));
}

/** Clamp a numeric value to a minimum of 0 (no upper bound). Returns undefined if not a number. */
function clampNonNegative(v: unknown): number | undefined {
  if (typeof v !== 'number' || isNaN(v)) return undefined;
  return Math.max(0, v);
}

/** Filter an array to only values present in a valid set. Returns undefined if result is empty or input is not an array. */
function filterEnums<T extends string>(
  arr: unknown,
  valid: Set<string>,
): T[] | undefined {
  if (!Array.isArray(arr)) return undefined;
  const filtered = arr.filter(
    (v) => typeof v === 'string' && valid.has(v),
  ) as T[];
  return filtered.length > 0 ? filtered : undefined;
}

/** Filter an array to only its string values. Returns undefined if result is empty or input is not an array. */
function filterStrings(arr: unknown): string[] | undefined {
  if (!Array.isArray(arr)) return undefined;
  const filtered = arr.filter((v) => typeof v === 'string') as string[];
  return filtered.length > 0 ? filtered : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Assert a required string field is a non-empty string. Returns error result if invalid. */
function requireString(
  v: unknown,
  fieldName: string,
): { error: McpToolResult } | { value: string } {
  if (typeof v !== 'string' || v.trim() === '') {
    return {
      error: {
        content: [
          {
            type: 'text',
            text: `engram tool error: ${fieldName} must be a non-empty string`,
          },
        ],
        isError: true,
      },
    };
  }
  return { value: v };
}

// =============================================================================
// Handler Factory
// =============================================================================

/**
 * Bind the ENGRAM_TOOLS to a live Engram instance.
 * Returns a handler function suitable for passing to your MCP server's onToolCall.
 */
export function createEngramToolHandler(engram: Engram) {
  return async function handleTool(
    name: EngramToolName,
    input: Record<string, unknown>,
  ): Promise<McpToolResult> {
    try {
      switch (name) {
        case 'engram_retain': {
          const textCheck = requireString(input.text, 'text');
          if ('error' in textCheck) return textCheck.error;
          const opts: RetainOptions = {
            memoryType: VALID_MEMORY_TYPES.has(input.memoryType as string)
              ? (input.memoryType as RetainOptions['memoryType'])
              : undefined,
            sourceType: VALID_SOURCE_TYPES.has(input.sourceType as string)
              ? (input.sourceType as RetainOptions['sourceType'])
              : undefined,
            trustScore: clampTrust(input.trustScore),
            source: typeof input.source === 'string' ? input.source : undefined,
            context:
              typeof input.context === 'string' ? input.context : undefined,
            eventTime:
              typeof input.eventTime === 'string' ? input.eventTime : undefined,
            temporalLabel:
              typeof input.temporalLabel === 'string'
                ? input.temporalLabel
                : undefined,
            supersedes:
              typeof input.supersedes === 'string'
                ? input.supersedes
                : undefined,
          };
          const result = await engram.retain(textCheck.value, opts);
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }

        case 'engram_recall': {
          const queryCheck = requireString(input.query, 'query');
          if ('error' in queryCheck) return queryCheck.error;
          const opts: RecallOptions = {
            topK:
              typeof input.topK === 'number'
                ? Math.max(1, Math.floor(input.topK))
                : undefined,
            strategies: filterEnums<
              'semantic' | 'keyword' | 'graph' | 'temporal'
            >(input.strategies, VALID_STRATEGIES),
            memoryTypes: filterEnums<
              'world' | 'experience' | 'observation' | 'opinion'
            >(input.memoryTypes, VALID_MEMORY_TYPES),
            minTrust: clampTrust(input.minTrust),
            after: typeof input.after === 'string' ? input.after : undefined,
            before: typeof input.before === 'string' ? input.before : undefined,
            includeOpinions:
              typeof input.includeOpinions === 'boolean'
                ? input.includeOpinions
                : undefined,
            includeObservations:
              typeof input.includeObservations === 'boolean'
                ? input.includeObservations
                : undefined,
            minScore: clampTrust(input.minScore),
            insightMinScore: clampTrust(input.insightMinScore),
            explainScores:
              typeof input.explainScores === 'boolean'
                ? input.explainScores
                : undefined,
            decayHalfLifeDays: clampNonNegative(input.decayHalfLifeDays),
          };
          const result = await engram.recall(queryCheck.value, opts);
          return {
            content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
          };
        }

        case 'engram_reflect': {
          const gates = isRecord(input.opinionGates)
            ? {
                minEvidenceCount: clampNonNegative(
                  input.opinionGates.minEvidenceCount,
                ),
                minDistinctDays: clampNonNegative(
                  input.opinionGates.minDistinctDays,
                ),
                minDistinctSources: clampNonNegative(
                  input.opinionGates.minDistinctSources,
                ),
              }
            : input.opinionGates === false
              ? false
              : undefined;
          const counterEvidence = isRecord(input.counterEvidence)
            ? {
                topK:
                  typeof input.counterEvidence.topK === 'number'
                    ? Math.max(1, Math.floor(input.counterEvidence.topK))
                    : undefined,
                maxContradictionRatio: clampTrust(
                  input.counterEvidence.maxContradictionRatio,
                ),
                onReinforce:
                  typeof input.counterEvidence.onReinforce === 'boolean'
                    ? input.counterEvidence.onReinforce
                    : undefined,
                failOpen:
                  typeof input.counterEvidence.failOpen === 'boolean'
                    ? input.counterEvidence.failOpen
                    : undefined,
              }
            : input.counterEvidence === false
              ? false
              : undefined;
          const result = await engram.reflect({
            ...(input.suggest === true ? { suggestions: {} } : {}),
            ...(gates !== undefined ? { opinionGates: gates } : {}),
            ...(counterEvidence !== undefined ? { counterEvidence } : {}),
          });
          return {
            content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
          };
        }

        case 'engram_process_extractions': {
          const batchSize =
            typeof input.batchSize === 'number' ? input.batchSize : 10;
          const result = await engram.processExtractions(batchSize);
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }

        case 'engram_forget': {
          const chunkIdCheck = requireString(input.chunkId, 'chunkId');
          if ('error' in chunkIdCheck) return chunkIdCheck.error;
          const forgotten = await engram.forget(chunkIdCheck.value);
          return {
            content: [{ type: 'text', text: JSON.stringify({ forgotten }) }],
          };
        }

        case 'engram_supersede': {
          const oldChunkIdCheck = requireString(input.oldChunkId, 'oldChunkId');
          if ('error' in oldChunkIdCheck) return oldChunkIdCheck.error;
          const newTextCheck = requireString(input.newText, 'newText');
          if ('error' in newTextCheck) return newTextCheck.error;
          const opts: RetainOptions = {
            memoryType: VALID_MEMORY_TYPES.has(input.memoryType as string)
              ? (input.memoryType as RetainOptions['memoryType'])
              : undefined,
            sourceType: VALID_SOURCE_TYPES.has(input.sourceType as string)
              ? (input.sourceType as RetainOptions['sourceType'])
              : undefined,
            trustScore: clampTrust(input.trustScore),
            source: typeof input.source === 'string' ? input.source : undefined,
            context:
              typeof input.context === 'string' ? input.context : undefined,
          };
          const result = await engram.supersede(
            oldChunkIdCheck.value,
            newTextCheck.value,
            opts,
          );
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }

        case 'engram_session': {
          const action =
            input.action === 'update' || input.action === 'snapshot'
              ? input.action
              : 'resume';

          if (action === 'resume') {
            const msgCheck = requireString(input.message, 'message');
            if ('error' in msgCheck) return msgCheck.error;
            const opts = {
              maxActive:
                typeof input.maxActive === 'number'
                  ? input.maxActive
                  : undefined,
              threshold:
                typeof input.threshold === 'number'
                  ? input.threshold
                  : undefined,
            };
            const result = await engram.inferWorkingSession(
              msgCheck.value,
              opts,
            );
            return {
              content: [
                { type: 'text', text: JSON.stringify(result, null, 2) },
              ],
            };
          }

          const sessionIdCheck = requireString(input.sessionId, 'sessionId');
          if ('error' in sessionIdCheck) return sessionIdCheck.error;
          const existing = engram.getWorkingSession(sessionIdCheck.value);
          if (!existing) {
            return {
              content: [
                {
                  type: 'text',
                  text: `engram tool error: working memory session not found: ${sessionIdCheck.value}`,
                },
              ],
              isError: true,
            };
          }

          if (action === 'update') {
            const updates: Record<string, unknown> =
              input.extensions &&
              typeof input.extensions === 'object' &&
              !Array.isArray(input.extensions)
                ? { ...(input.extensions as Record<string, unknown>) }
                : {};
            if (typeof input.progress === 'string') {
              updates.progress = input.progress;
            }
            await engram.updateWorkingSession(sessionIdCheck.value, updates);
            const state = engram.getWorkingSession(sessionIdCheck.value);
            return {
              content: [{ type: 'text', text: JSON.stringify(state, null, 2) }],
            };
          }

          // action === 'snapshot'
          const snapResult = await engram.snapshotWorkingSession(
            sessionIdCheck.value,
          );
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify({
                  sessionId: sessionIdCheck.value,
                  ...snapResult,
                }),
              },
            ],
          };
        }

        case 'engram_queue_stats': {
          const stats = engram.getQueueStats();
          return {
            content: [{ type: 'text', text: JSON.stringify(stats, null, 2) }],
          };
        }

        case 'engram_requeue_failed': {
          const result = engram.requeueFailedExtractions({
            errorLike:
              typeof input.errorLike === 'string' ? input.errorLike : undefined,
          });
          return {
            content: [{ type: 'text', text: JSON.stringify(result) }],
          };
        }

        case 'engram_introspect': {
          const subject =
            typeof input.subject === 'string' ? input.subject : undefined;
          const result = engram.introspect(subject, {
            minConfidence:
              typeof input.minConfidence === 'number'
                ? input.minConfidence
                : undefined,
            limit:
              typeof input.limit === 'number'
                ? Math.max(1, Math.floor(input.limit))
                : undefined,
            includeOpinions:
              typeof input.includeOpinions === 'boolean'
                ? input.includeOpinions
                : undefined,
            includeObservations:
              typeof input.includeObservations === 'boolean'
                ? input.includeObservations
                : undefined,
          });
          return {
            content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
          };
        }

        case 'engram_embed': {
          const textCheck = requireString(input.text, 'text');
          if ('error' in textCheck) return textCheck.error;
          const mode = (input.mode === 'document' ? 'document' : 'query') as
            | 'query'
            | 'document';
          const vec = await engram.embedForMode(textCheck.value, mode);
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify({
                  embedding: Array.from(vec),
                  dimensions: vec.length,
                }),
              },
            ],
          };
        }

        case 'engram_context_commit': {
          const decisionCheck = requireString(input.decision, 'decision');
          if ('error' in decisionCheck) return decisionCheck.error;
          const artifact: DecisionArtifact = {
            decision: decisionCheck.value,
            rationale:
              typeof input.rationale === 'string' ? input.rationale : undefined,
            scoredOptions: Array.isArray(input.scoredOptions)
              ? (input.scoredOptions as DecisionArtifact['scoredOptions'])
              : undefined,
            confidence: clampTrust(input.confidence),
            refsToSource: filterStrings(input.refsToSource),
            domain: typeof input.domain === 'string' ? input.domain : undefined,
            agentId:
              typeof input.agentId === 'string' ? input.agentId : undefined,
          };
          const scope: TaskScope = {};
          if (
            typeof input.parentRefId === 'string' &&
            input.parentRefId.trim() !== ''
          ) {
            scope.parent = { id: input.parentRefId, scope: 'task' };
          }
          if (typeof input.ttlMs === 'number' && !isNaN(input.ttlMs)) {
            scope.ttlMs = input.ttlMs;
          }
          const ref = await engram.commitContext(artifact, scope);
          return { content: [{ type: 'text', text: JSON.stringify(ref) }] };
        }

        case 'engram_context_query': {
          const refIdCheck = requireString(input.refId, 'refId');
          if ('error' in refIdCheck) return refIdCheck.error;
          const queryCheck = requireString(input.query, 'query');
          if ('error' in queryCheck) return queryCheck.error;
          const budget: TokenBudget | undefined =
            typeof input.maxChars === 'number' && input.maxChars > 0
              ? { maxChars: Math.floor(input.maxChars) }
              : undefined;
          const slice = await engram.queryContext(
            { id: refIdCheck.value, scope: 'task' },
            queryCheck.value,
            budget,
          );
          return {
            content: [{ type: 'text', text: JSON.stringify(slice, null, 2) }],
          };
        }

        case 'engram_context_promote': {
          const refIdCheck = requireString(input.refId, 'refId');
          if ('error' in refIdCheck) return refIdCheck.error;
          try {
            await engram.promoteContext({
              id: refIdCheck.value,
              scope: 'task',
            });
            return {
              content: [
                { type: 'text', text: JSON.stringify({ promoted: true }) },
              ],
            };
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            if (message.includes('not found or already expired')) {
              return {
                content: [
                  { type: 'text', text: JSON.stringify({ promoted: false }) },
                ],
              };
            }
            throw err;
          }
        }

        case 'engram_suggestions': {
          const status = VALID_SUGGESTION_STATUSES.has(input.status as string)
            ? (input.status as SuggestionStatus)
            : undefined;
          const kind = VALID_SUGGESTION_KINDS.has(input.kind as string)
            ? (input.kind as SuggestionKind)
            : undefined;
          const result = engram.suggestions({
            status,
            kind,
            domain: typeof input.domain === 'string' ? input.domain : undefined,
            limit:
              typeof input.limit === 'number'
                ? Math.max(1, Math.floor(input.limit))
                : undefined,
          });
          return {
            content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
          };
        }

        case 'engram_resolve_suggestion': {
          const idCheck = requireString(input.suggestionId, 'suggestionId');
          if ('error' in idCheck) return idCheck.error;
          if (!VALID_SUGGESTION_STATUSES.has(input.status as string)) {
            return {
              content: [
                {
                  type: 'text',
                  text: 'engram tool error: status must be one of proposed|accepted|dismissed|implemented',
                },
              ],
              isError: true,
            };
          }
          const status = input.status as SuggestionStatus;
          const reason =
            typeof input.reason === 'string' ? input.reason : undefined;
          const resolved = engram.resolveSuggestion(
            idCheck.value,
            status,
            reason,
          );
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify({
                  suggestionId: idCheck.value,
                  status,
                  resolved,
                }),
              },
            ],
          };
        }
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        content: [{ type: 'text', text: `engram tool error: ${message}` }],
        isError: true,
      };
    }
  };
}
