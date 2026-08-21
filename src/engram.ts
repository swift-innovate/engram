// =============================================================================
// engram.ts - Unified Engram Class
//
// Public API surface for the Engram library. Wraps retain/recall/reflect into
// a single class with a clean lifecycle:
//
//   const myAgent = await Engram.create('./myAgent.engram', { ... });
//   await myAgent.retain('Tom prefers Terraform', { memoryType: 'world', ... });
//   const context = await myAgent.recall('IaC tools?', { topK: 5 });
//   await myAgent.reflect();
//   myAgent.close();
//
// Schema bootstrap:
//   Reads schema.sql from the same directory as this compiled file (dist/).
//   The build step copies src/schema.sql → dist/schema.sql.
//   All CREATE TABLE statements are IF NOT EXISTS — safe to run on every open.
//
// Connection model:
//   Engram holds one persistent connection for retain/recall (low-latency).
//   reflect() opens a separate connection internally (reflect.ts pattern).
//   SQLite WAL mode allows concurrent readers alongside a single writer.
//   If both connections write simultaneously, the second blocks until the
//   first commits — writes serialize, they do not run in parallel.
//   reflect() is safe to call while retain/recall are active, but its
//   write transaction will wait for any in-progress retain() to finish.
// =============================================================================

import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';
import { hostname } from 'os';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import {
  retain,
  retainBatch,
  processExtractionQueue,
  recoverStalledExtractions,
  requeueFailedExtractions,
  getQueueStats,
  embeddingToBuffer,
  OllamaEmbeddings,
  LocalEmbedder,
  shouldRetain,
  chunkText,
  computeTextHash,
  type RetainOptions,
  type ChunkOptions,
  type RetainResult,
  type EmbeddingProvider,
  type QueueStats,
} from './retain.js';

import {
  recall,
  formatForPrompt,
  formatWhyLine,
  DEFAULT_SOURCE_TIERS,
  DEFAULT_MEMORY_TYPE_RANK,
  type RecallOptions,
  type RecallResponse,
  type RecallResult,
  type FormatForPromptOptions,
} from './recall.js';
import { parseTemporalQuery, type TemporalRange } from './temporal-parser.js';
import { entityId as buildEntityId } from './extract-cpu.js';

import {
  introspect,
  type IntrospectOptions,
  type IntrospectResult,
  type OpinionView,
  type ObservationView,
} from './introspect.js';

import { ReadonlyEngram } from './readonly-engram.js';

import {
  groundSubagent,
  taskContext,
  metabolizeReport,
  GROUNDING_TYPES,
  type GroundingType,
  type GroundingScope,
  type Grounding,
  type SubagentReport,
  type OrchestratorWriter,
  type MetabolizeOptions,
  type MetabolizeResult,
} from './grounding.js';

import {
  reflect,
  reflectCatchUp,
  backfillInsightEmbeddings,
  ReflectScheduler,
  getBeliefJournal,
  DEFAULT_OPINION_GATES,
  DEFAULT_COUNTER_EVIDENCE,
  type ReflectConfig,
  type ReflectResult,
  type CatchUpConfig,
  type CatchUpResult,
  type OpinionGates,
  type CounterEvidenceConfig,
  type BeliefJournalAction,
  type BeliefJournalEntry,
  type BeliefJournalQuery,
  type BackfillInsightEmbeddingsResult,
} from './reflect.js';

import {
  getSuggestions,
  resolveSuggestion,
  getSuggestionJournal,
  type SuggestionConfig,
  type SuggestionGates,
  type SuggestionKind,
  type SuggestionStatus,
  type SuggestionJournalAction,
  type SuggestionView,
  type SuggestionQuery,
  type SuggestionJournalEntry,
  type SuggestionJournalQuery,
} from './suggest.js';

import {
  OllamaGeneration,
  OpenAICompatibleGeneration,
  AnthropicGeneration,
  UnconfiguredGeneration,
  DEFAULT_OLLAMA_URL,
  type GenerationProvider,
  type GenerationOptions,
} from './generation.js';

import type {
  WorkingMemoryState,
  WorkingMemoryOptions,
  WorkingSessionResult,
  SessionCandidate,
} from './working-memory-types.js';

import {
  commitContext,
  queryContext,
  expireContext,
  promoteToDurable,
  type ContextRef,
  type TaskScope,
  type DecisionArtifact,
  type TokenBudget,
  type ContextSlice,
  type CommittedArtifact,
} from './context-store.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// =============================================================================
// Public API Types
// =============================================================================

// Re-export types so library consumers import only from 'engram'
export type {
  RetainOptions,
  RetainResult,
  EmbeddingProvider,
  RecallOptions,
  RecallResponse,
  RecallResult,
  ReflectConfig,
  ReflectResult,
  CatchUpConfig,
  CatchUpResult,
  OpinionGates,
  CounterEvidenceConfig,
  BeliefJournalAction,
  BeliefJournalEntry,
  BeliefJournalQuery,
  BackfillInsightEmbeddingsResult,
  SuggestionConfig,
  SuggestionGates,
  SuggestionKind,
  SuggestionStatus,
  SuggestionJournalAction,
  SuggestionView,
  SuggestionQuery,
  SuggestionJournalEntry,
  SuggestionJournalQuery,
  FormatForPromptOptions,
  WorkingMemoryState,
  WorkingMemoryOptions,
  WorkingSessionResult,
  SessionCandidate,
  GenerationProvider,
  GenerationOptions,
  ContextRef,
  TaskScope,
  DecisionArtifact,
  TokenBudget,
  ContextSlice,
  CommittedArtifact,
  IntrospectOptions,
  IntrospectResult,
  OpinionView,
  ObservationView,
};

export { DEFAULT_OPINION_GATES, DEFAULT_COUNTER_EVIDENCE };
export {
  OllamaEmbeddings,
  LocalEmbedder,
  reflectCatchUp,
  backfillInsightEmbeddings,
  ReflectScheduler,
  getBeliefJournal,
  getSuggestions,
  resolveSuggestion,
  getSuggestionJournal,
  shouldRetain,
  chunkText,
  formatForPrompt,
  formatWhyLine,
  OllamaGeneration,
  OpenAICompatibleGeneration,
  AnthropicGeneration,
  UnconfiguredGeneration,
  parseTemporalQuery,
  DEFAULT_OLLAMA_URL,
  DEFAULT_SOURCE_TIERS,
  DEFAULT_MEMORY_TYPE_RANK,
  ReadonlyEngram,
  groundSubagent,
  taskContext,
  metabolizeReport,
  GROUNDING_TYPES,
};
export type {
  GroundingType,
  GroundingScope,
  Grounding,
  SubagentReport,
  OrchestratorWriter,
  MetabolizeOptions,
  MetabolizeResult,
};
export type { TemporalRange, ChunkOptions };

// Model selection + preflight — the single surface for choosing a generation
// model. Re-exported so harness adapters (Pi, OpenClaw) resolve the same way.
export {
  resolveModelSpec,
  resolveModelSpecOrNull,
  preflightModel,
  formatPreflightFailure,
  isModelServed,
} from './model-resolver.js';
export type {
  ModelRole,
  ModelSpec,
  ResolveModelInput,
  PreflightResult,
} from './model-resolver.js';

export interface EngramOptions {
  /** Mission for the reflection engine: what to focus on during synthesis */
  reflectMission?: string;
  /** Mission for retain: what to prioritize when storing memories */
  retainMission?: string;
  /** Behavioral disposition for reflection (skepticism/literalism/empathy 0-1) */
  disposition?: {
    skepticism?: number;
    literalism?: number;
    empathy?: number;
  };
  /** Ollama endpoint (default: http://localhost:11434) */
  ollamaUrl?: string;
  /** Embedding model (default: 'nomic-ai/nomic-embed-text-v1.5' local, 'nomic-embed-text' Ollama) */
  embedModel?: string;
  /** Embedding dimensions (default: 768 for nomic-embed-text) */
  embedDimensions?: number;
  /**
   * LLM for extraction + reflection. NO default — if omitted (and no
   * `generator`/`anthropicGeneration`/`generationEndpoint` is given), the
   * engram opens with a fail-loud `UnconfiguredGeneration`: retain/recall work,
   * but the first reflect()/extract() throws. Resolve via model-resolver.ts.
   */
  reflectModel?: string;
  /** Override the embedding provider — useful for testing without Ollama */
  embedder?: EmbeddingProvider;
  /** Use Ollama for embeddings instead of local Transformers.js (default: false) */
  useOllamaEmbeddings?: boolean;
  /** Override the generation provider for extraction + reflection */
  generator?: GenerationProvider;
  /** Shorthand: use OpenAI-compatible endpoint for generation */
  generationEndpoint?: {
    baseUrl: string;
    model: string;
    apiKey?: string;
  };
  /** Shorthand: use Anthropic API for generation. Model is required (no default). */
  anthropicGeneration?: {
    apiKey: string;
    model: string;
  };
}

// =============================================================================
// Engram Class
// =============================================================================

export class Engram {
  private readonly db: Database.Database;
  private readonly dbPath: string;
  private readonly embedder: EmbeddingProvider;
  private readonly generator: GenerationProvider;
  /**
   * This instance's stable node-origin — the identity stamped onto every chunk
   * this instance authors (Provenance groundwork). Read once from bank_config
   * in init() and held here so the write path never re-queries per retain.
   */
  private readonly nodeOrigin: string;

  private constructor(
    db: Database.Database,
    dbPath: string,
    embedder: EmbeddingProvider,
    generator: GenerationProvider,
    nodeOrigin: string,
  ) {
    this.db = db;
    this.dbPath = dbPath;
    this.embedder = embedder;
    this.generator = generator;
    this.nodeOrigin = nodeOrigin;
  }

  // ---------------------------------------------------------------------------
  // Internal factory — shared by create() and open()
  // ---------------------------------------------------------------------------

  private static async init(
    path: string,
    options: EngramOptions,
  ): Promise<Engram> {
    const {
      ollamaUrl = DEFAULT_OLLAMA_URL,
      embedModel,
      embedDimensions,
      reflectModel,
      embedder: injectedEmbedder,
      useOllamaEmbeddings = false,
      generator: injectedGenerator,
    } = options;

    const db = new Database(path);

    // Enable WAL mode for concurrent read-while-write access from the
    // engram-aql Rust binary (or any second process). The TypeScript write
    // path runs one transaction at a time; WAL lets readers see the
    // committed state without blocking on the writer.
    //
    // synchronous = NORMAL is the canonical WAL pair: on OS-level crash
    // the last transaction may roll back, but process crashes and power
    // loss remain safe. For agent memory the trade is obvious — losing
    // one retain on a kernel panic just means the agent re-learns that
    // fact next time it matters.
    //
    // busy_timeout gives SQLite a 5-second window to wait for a held
    // lock before returning SQLITE_BUSY — enough for short writes but
    // short enough to surface genuine deadlocks quickly. The Rust
    // engram-aql binary uses the same 5-second window for symmetry.
    //
    // Note: journal_mode = WAL is persistent in the database header.
    // First open upgrades any pre-WAL `.engram` file on disk; subsequent
    // opens are no-ops. Sidecar files `.engram-wal` and `.engram-shm`
    // appear alongside the main file at runtime — use engram.backup()
    // for single-file portability instead of raw `cp`.
    db.pragma('journal_mode = WAL');
    db.pragma('synchronous = NORMAL');
    db.pragma('busy_timeout = 5000');

    // Load sqlite-vec for vector search. Graceful fallback: semantic strategy
    // is simply skipped (recall.ts catches the error) if the extension is absent.
    try {
      const mod = (await import('sqlite-vec')) as unknown as {
        load: (db: Database.Database) => void;
      };
      mod.load(db);
    } catch {
      // sqlite-vec not installed — semantic search disabled
    }

    // Bootstrap schema — idempotent, safe to run on every open
    const schemaPath = join(__dirname, 'schema.sql');
    const schema = readFileSync(schemaPath, 'utf8');
    db.exec(schema);

    // Migration: add text_hash column to existing .engram files (pre-F4)
    const columns = db.pragma('table_info(chunks)') as Array<{ name: string }>;
    if (!columns.some((c) => c.name === 'text_hash')) {
      db.exec('ALTER TABLE chunks ADD COLUMN text_hash TEXT');
      db.exec(
        'CREATE INDEX IF NOT EXISTS idx_chunks_text_hash ON chunks(text_hash) WHERE is_active = TRUE',
      );
    }

    // Backfill text_hash for any existing chunks that predate this migration
    const missing = db
      .prepare(
        `SELECT COUNT(*) as c FROM chunks WHERE text_hash IS NULL AND is_active = TRUE`,
      )
      .get() as { c: number };
    if (missing.c > 0) {
      const rows = db
        .prepare(
          `SELECT id, text FROM chunks WHERE text_hash IS NULL AND is_active = TRUE`,
        )
        .all() as Array<{ id: string; text: string }>;
      const updateHash = db.prepare(
        `UPDATE chunks SET text_hash = ? WHERE id = ?`,
      );
      const backfill = db.transaction(() => {
        for (const row of rows) {
          updateHash.run(computeTextHash(row.text), row.id);
        }
      });
      backfill();
    }

    // Migration: add next_retry_after column to existing .engram files
    const eqColumns = db.pragma('table_info(extraction_queue)') as Array<{
      name: string;
    }>;
    if (!eqColumns.some((c) => c.name === 'next_retry_after')) {
      db.exec(
        'ALTER TABLE extraction_queue ADD COLUMN next_retry_after TIMESTAMP',
      );
    }

    // Migration: add ContextStore scope columns to existing .engram files.
    // Columns are added first (guarded — only if missing), then indexes are
    // created unconditionally afterward: by that point the columns exist
    // either way (freshly created above via schema.sql, or just ALTERed in),
    // so "CREATE INDEX IF NOT EXISTS" never runs against a missing column.
    const scopeColumns = db.pragma('table_info(chunks)') as Array<{
      name: string;
    }>;
    if (!scopeColumns.some((c) => c.name === 'scope')) {
      db.exec(
        `ALTER TABLE chunks ADD COLUMN scope TEXT NOT NULL DEFAULT 'durable' CHECK (scope IN ('durable', 'task'))`,
      );
    }
    if (!scopeColumns.some((c) => c.name === 'expires_at')) {
      db.exec('ALTER TABLE chunks ADD COLUMN expires_at TIMESTAMP');
    }
    if (!scopeColumns.some((c) => c.name === 'parent_ref')) {
      db.exec(
        'ALTER TABLE chunks ADD COLUMN parent_ref TEXT REFERENCES chunks(id)',
      );
    }
    if (!scopeColumns.some((c) => c.name === 'agent_id')) {
      db.exec('ALTER TABLE chunks ADD COLUMN agent_id TEXT');
    }
    if (!scopeColumns.some((c) => c.name === 'artifact_json')) {
      db.exec('ALTER TABLE chunks ADD COLUMN artifact_json TEXT');
    }
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope) WHERE is_active = TRUE',
    );
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_chunks_parent_ref ON chunks(parent_ref) WHERE parent_ref IS NOT NULL',
    );
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_chunks_expires_at ON chunks(expires_at) WHERE expires_at IS NOT NULL',
    );

    // Migration: add node_origin provenance columns to existing .engram files.
    // Additive only — records which Engram instance authored a chunk/opinion/
    // observation, so a future sync/merge has provenance without backfill.
    // Same guarded-ALTER-then-unconditional-index pattern as the scope columns
    // above: pre-existing rows stay NULL ("origin unknown / pre-distribution"),
    // never falsely claimed as authored by this instance. Backfilling would
    // lie about who first recorded memories that predate origin tracking.
    const chunkOriginCols = db.pragma('table_info(chunks)') as Array<{
      name: string;
    }>;
    if (!chunkOriginCols.some((c) => c.name === 'node_origin')) {
      db.exec('ALTER TABLE chunks ADD COLUMN node_origin TEXT');
    }
    const opinionOriginCols = db.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    if (!opinionOriginCols.some((c) => c.name === 'node_origin')) {
      db.exec('ALTER TABLE opinions ADD COLUMN node_origin TEXT');
    }
    const obsOriginCols = db.pragma('table_info(observations)') as Array<{
      name: string;
    }>;
    if (!obsOriginCols.some((c) => c.name === 'node_origin')) {
      db.exec('ALTER TABLE observations ADD COLUMN node_origin TEXT');
    }
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_chunks_node_origin ON chunks(node_origin) WHERE node_origin IS NOT NULL',
    );
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_opinions_node_origin ON opinions(node_origin) WHERE node_origin IS NOT NULL',
    );
    db.exec(
      'CREATE INDEX IF NOT EXISTS idx_observations_node_origin ON observations(node_origin) WHERE node_origin IS NOT NULL',
    );

    // Migration: add the falsifier column to existing .engram files (issue
    // #38 item 3). Pre-existing opinions stay NULL ("never stated") — a
    // falsifier is the model's own statement at formation, not backfillable.
    const opinionFalsifierCols = db.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    if (!opinionFalsifierCols.some((c) => c.name === 'would_change_this')) {
      db.exec('ALTER TABLE opinions ADD COLUMN would_change_this TEXT');
    }

    // Migration: add insight embeddings to existing .engram files. Opinions
    // and observations were retrieved at recall by SQL LIKE over their text
    // with no relevance ordering (top-N by confidence/recency among whatever
    // matched), so any query containing ordinary words returned a near-constant
    // set of beliefs — the synthesized layer was the only part of the store not
    // reachable semantically. Pre-existing rows stay NULL and keep the keyword
    // path until backfilled (see backfillInsightEmbeddings), so this migration
    // never changes behavior on its own.
    const opinionEmbedCols = db.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    if (!opinionEmbedCols.some((c) => c.name === 'embedding')) {
      db.exec('ALTER TABLE opinions ADD COLUMN embedding BLOB');
    }
    const obsEmbedCols = db.pragma('table_info(observations)') as Array<{
      name: string;
    }>;
    if (!obsEmbedCols.some((c) => c.name === 'embedding')) {
      db.exec('ALTER TABLE observations ADD COLUMN embedding BLOB');
    }

    // Establish this bank's stable node-origin identity. Generated exactly once,
    // on first open of a bank that lacks it (ON CONFLICT DO NOTHING never
    // regenerates — the value must survive restarts). Lives in bank_config,
    // exactly like embed_dimensions / entity_id_v2_migrated: one .engram = one
    // origin, written at birth.
    const hostSlug =
      hostname()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'host';
    const generatedOrigin = `node-${hostSlug}-${randomUUID().slice(0, 8)}`;
    db.prepare(
      `INSERT INTO bank_config (key, value) VALUES ('node_origin', ?)
       ON CONFLICT(key) DO NOTHING`,
    ).run(generatedOrigin);
    const nodeOrigin = (
      db
        .prepare(`SELECT value FROM bank_config WHERE key = 'node_origin'`)
        .get() as { value: string }
    ).value;

    // Migration: re-key entity IDs to collision-resistant format (slug + hash).
    // Disables FK checks during the transaction because child rows (chunk_entities,
    // relations) reference entities.id — updating the parent first would violate
    // FK constraints. All rows are updated atomically within the transaction.
    const entityIdV2 = db
      .prepare(
        `SELECT value FROM bank_config WHERE key = 'entity_id_v2_migrated'`,
      )
      .get() as { value: string } | undefined;
    if (!entityIdV2) {
      const entities = db
        .prepare(`SELECT id, canonical_name FROM entities`)
        .all() as Array<{ id: string; canonical_name: string }>;
      if (entities.length > 0) {
        db.pragma('foreign_keys = OFF');
        const updateEntity = db.prepare(
          `UPDATE entities SET id = ? WHERE id = ?`,
        );
        const updateChunkEntity = db.prepare(
          `UPDATE chunk_entities SET entity_id = ? WHERE entity_id = ?`,
        );
        const updateRelSrc = db.prepare(
          `UPDATE relations SET source_entity_id = ? WHERE source_entity_id = ?`,
        );
        const updateRelTgt = db.prepare(
          `UPDATE relations SET target_entity_id = ? WHERE target_entity_id = ?`,
        );
        const migrate = db.transaction(() => {
          for (const ent of entities) {
            const newId = buildEntityId(ent.canonical_name);
            if (newId !== ent.id) {
              updateChunkEntity.run(newId, ent.id);
              updateRelSrc.run(newId, ent.id);
              updateRelTgt.run(newId, ent.id);
              updateEntity.run(newId, ent.id);
            }
          }
        });
        migrate();
        db.pragma('foreign_keys = ON');
      }
      db.prepare(
        `INSERT OR REPLACE INTO bank_config (key, value) VALUES ('entity_id_v2_migrated', 'true')`,
      ).run();
    }

    // Embedding provider selection (priority order):
    // 1. Injected embedder — caller controls everything (used in tests)
    // 2. Ollama embeddings — opt-in via useOllamaEmbeddings flag
    // 3. Local Transformers.js — default, zero network dependency
    let embedder: EmbeddingProvider;
    try {
      if (injectedEmbedder) {
        embedder = injectedEmbedder;
      } else if (useOllamaEmbeddings) {
        embedder = new OllamaEmbeddings(
          ollamaUrl,
          embedModel ?? 'nomic-embed-text',
          embedDimensions ?? 768,
        );
      } else {
        const localModel = embedModel ?? 'nomic-ai/nomic-embed-text-v1.5';
        const local = new LocalEmbedder(localModel);
        await local.init();
        embedder = local;
      }
    } catch (err) {
      db.close();
      throw err;
    }

    // Generation provider selection (priority order):
    // 1. Injected generator — caller controls everything (tests, custom providers)
    // 2. Anthropic generation — direct API
    // 3. OpenAI-compatible endpoint — OpenRouter, Herd, vLLM, etc.
    // 4. Ollama generation — ONLY when a model is configured (no default)
    // 5. Unconfigured — fail-loud placeholder; retain/recall still work, but
    //    reflect()/extract() throw. No silent fallback to a default model.
    let generator: GenerationProvider;
    if (injectedGenerator) {
      generator = injectedGenerator;
    } else if (options.anthropicGeneration) {
      generator = new AnthropicGeneration(
        options.anthropicGeneration.apiKey,
        options.anthropicGeneration.model,
      );
    } else if (options.generationEndpoint) {
      generator = new OpenAICompatibleGeneration(
        options.generationEndpoint.baseUrl,
        options.generationEndpoint.model,
        options.generationEndpoint.apiKey,
      );
    } else if (reflectModel && reflectModel.trim()) {
      generator = new OllamaGeneration({ url: ollamaUrl, model: reflectModel });
    } else {
      generator = new UnconfiguredGeneration();
    }

    // Validate embedding dimensions against existing data.
    // Legacy .engram files won't have embed_dimensions in bank_config,
    // so we derive it from existing embeddings before recording.
    const currentDim = embedder.dimensions;
    const storedDim = db
      .prepare(`SELECT value FROM bank_config WHERE key = 'embed_dimensions'`)
      .get() as { value: string } | undefined;

    let knownDim: number | null = storedDim
      ? parseInt(storedDim.value, 10)
      : null;

    // Legacy migration: no stored dimension — derive from existing embeddings
    if (knownDim === null) {
      const sample = db
        .prepare(
          `SELECT length(embedding) as len FROM chunks WHERE embedding IS NOT NULL LIMIT 1`,
        )
        .get() as { len: number } | undefined;
      if (sample && sample.len > 0) {
        // Float32 = 4 bytes per dimension
        knownDim = sample.len / 4;
      }
    }

    if (knownDim !== null && knownDim !== currentDim) {
      const hasEmbeddings = (
        db
          .prepare(
            `SELECT COUNT(*) as cnt FROM chunks WHERE embedding IS NOT NULL`,
          )
          .get() as { cnt: number }
      ).cnt;
      if (hasEmbeddings > 0) {
        db.close();
        throw new Error(
          `Embedding dimension mismatch: database contains ${hasEmbeddings} chunks with ${knownDim}d embeddings, ` +
            `but the current embedder produces ${currentDim}d vectors. ` +
            `Cosine similarity will be invalid. Use the same model or re-embed existing chunks.`,
        );
      }
    }

    // Record (or update) the current dimension
    db.prepare(
      `INSERT INTO bank_config (key, value) VALUES ('embed_dimensions', ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP`,
    ).run(String(currentDim));

    return new Engram(db, path, embedder, generator, nodeOrigin);
  }

  private upsertBankConfig(options: EngramOptions): void {
    const upsert = this.db.prepare(`
      INSERT INTO bank_config (key, value) VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
    `);
    const tx = this.db.transaction(() => {
      if (options.reflectMission !== undefined)
        upsert.run('reflect_mission', options.reflectMission);
      if (options.retainMission !== undefined)
        upsert.run('retain_mission', options.retainMission);
      if (options.disposition !== undefined)
        upsert.run('disposition', JSON.stringify(options.disposition));
    });
    tx();
  }

  // ---------------------------------------------------------------------------
  // Public factory methods
  // ---------------------------------------------------------------------------

  /**
   * Create or reinitialize an engram at the given path.
   * All provided options are written to bank_config on every call.
   */
  static async create(
    path: string,
    options: EngramOptions = {},
  ): Promise<Engram> {
    const engram = await Engram.init(path, options);
    engram.upsertBankConfig(options);
    return engram;
  }

  /**
   * Open an existing engram. Schema is bootstrapped if missing (safe on new files too).
   * Bank config is updated only for options that are explicitly provided.
   */
  static async open(
    path: string,
    options: EngramOptions = {},
  ): Promise<Engram> {
    const engram = await Engram.init(path, options);
    const configKeys: Array<keyof EngramOptions> = [
      'reflectMission',
      'retainMission',
      'disposition',
    ];
    if (configKeys.some((k) => options[k] !== undefined)) {
      engram.upsertBankConfig(options);
    }
    return engram;
  }

  // ---------------------------------------------------------------------------
  // Core operations
  // ---------------------------------------------------------------------------

  /** Store a memory trace. Fast path — no LLM call. The SQLite write is ~5ms;
   *  total latency is dominated by the local embedding (tens of ms on CPU). */
  async retain(text: string, options?: RetainOptions): Promise<RetainResult> {
    return retain(this.db, text, this.embedder, options, this.nodeOrigin);
  }

  /**
   * Bulk store. Queues all entity extractions at the end rather than one-by-one.
   * Efficient for importing conversations or documents.
   */
  async retainBatch(
    items: Array<{ text: string; options?: RetainOptions }>,
    onProgress?: (current: number, total: number) => void,
  ): Promise<RetainResult[]> {
    return retainBatch(
      this.db,
      items,
      this.embedder,
      onProgress,
      undefined,
      this.nodeOrigin,
    );
  }

  /**
   * Retrieve relevant memories via 4-way retrieval (semantic + keyword + graph + temporal),
   * fused via Reciprocal Rank Fusion and weighted by trust score.
   */
  async recall(
    query: string,
    options?: RecallOptions,
  ): Promise<RecallResponse> {
    return recall(this.db, query, this.embedder, options);
  }

  /**
   * Run a reflection cycle: processes unreflected facts through the LLM,
   * synthesizes observations, and reinforces/challenges opinions.
   *
   * Opens a separate DB connection internally (WAL mode: safe alongside
   * idle retain/recall connections).
   *
   * @remarks Only chunks with memoryType 'world' or 'experience' are processed.
   * Observations and opinions are outputs of reflection, not inputs.
   */
  async reflect(
    options?: Pick<
      ReflectConfig,
      | 'batchSize'
      | 'minFactsThreshold'
      | 'sourceTypes'
      | 'existingContextCharBudget'
      | 'opinionGates'
      | 'counterEvidence'
      | 'suggestions'
    >,
  ): Promise<ReflectResult> {
    return reflect({
      dbPath: this.dbPath,
      generator: this.generator,
      embedder: this.embedder,
      ...options,
    });
  }

  /**
   * Run a catch-up pass: many reflect() batches in one invocation to drain a
   * reflection backlog a single cycle can't keep up with (D5). Meant for
   * off-peak use, where a burst of metered-model calls is acceptable.
   *
   * Leave `batchSize` unset so the adaptive-shrink hint self-heals a mid-pass
   * context overrun; bound the pass with `maxBatches` / `maxFacts` /
   * `maxDurationMs`. See {@link reflectCatchUp}.
   */
  async reflectCatchUp(
    options?: Pick<
      CatchUpConfig,
      | 'batchSize'
      | 'minFactsThreshold'
      | 'maxBatches'
      | 'maxFacts'
      | 'maxDurationMs'
      | 'maxStalls'
      | 'sourceTypes'
      | 'existingContextCharBudget'
      | 'opinionGates'
      | 'counterEvidence'
      | 'suggestions'
    >,
  ): Promise<CatchUpResult> {
    return reflectCatchUp({
      dbPath: this.dbPath,
      generator: this.generator,
      embedder: this.embedder,
      ...options,
    });
  }

  /**
   * Give opinions/observations formed before the `embedding` column existed a
   * vector, so recall ranks them semantically rather than by keyword match.
   *
   * Normally unnecessary: `reflect()` runs a bounded backfill each cycle
   * whenever an embedder is present. Call this directly to drain a large
   * legacy backlog in one go (pass a bigger `limit`) instead of waiting for
   * cycles to chip away at it. Bounded, resumable, and never rewrites an
   * embedding that already exists. Library-only surface — not an MCP tool.
   */
  async backfillInsightEmbeddings(options?: {
    limit?: number;
  }): Promise<BackfillInsightEmbeddingsResult> {
    return backfillInsightEmbeddings(this.db, this.embedder, options);
  }

  /**
   * Query the per-belief audit trail (issue #38): one row per opinion
   * decision reflection made — formed/reinforced/challenged — or declined to
   * make (`rejected`, including gate rejections with per-gate measurements).
   * Newest first. Projection-only; no LLM call. Library-only surface — not
   * exposed as an MCP tool.
   */
  beliefJournal(query?: BeliefJournalQuery): BeliefJournalEntry[] {
    return getBeliefJournal(this.db, query);
  }

  /**
   * List procedural suggestions (issue #39) — "this recurring pattern would
   * benefit from being codified as a skill/rule/workflow/config." Newest/
   * most-evidenced first. Projection-only; no LLM call. Exposed as the
   * engram_suggestions MCP tool / `suggestions` CLI subcommand. Suggestions
   * never enter recall() or groundSubagent().
   */
  suggestions(query?: SuggestionQuery): SuggestionView[] {
    return getSuggestions(this.db, query);
  }

  /**
   * Resolve a suggestion's status (accept/dismiss/implement, or reopen back
   * to 'proposed'). Returns false when the id doesn't exist.
   */
  resolveSuggestion(
    suggestionId: string,
    status: SuggestionStatus,
    reason?: string,
  ): boolean {
    return resolveSuggestion(this.db, suggestionId, status, reason);
  }

  /**
   * Query the suggestion audit trail (issue #39), newest first.
   * Projection-only; no LLM call. Library-only surface.
   */
  suggestionJournal(query?: SuggestionJournalQuery): SuggestionJournalEntry[] {
    return getSuggestionJournal(this.db, query);
  }

  /**
   * Drain the entity extraction queue. Calls Ollama to extract entities and
   * relations from retained chunks, building out the knowledge graph.
   */
  async processExtractions(
    batchSize: number = 10,
  ): Promise<{ processed: number; failed: number }> {
    return processExtractionQueue(this.db, this.generator, batchSize);
  }

  /**
   * Recover extraction queue items stuck in 'processing' state after a crash.
   * Called automatically by processExtractions(), but can be invoked manually.
   *
   * @param stallTimeoutMinutes — items processing longer than this are reset (default: 5)
   * @returns number of recovered items
   */
  recoverExtractions(stallTimeoutMinutes?: number): number {
    return recoverStalledExtractions(this.db, stallTimeoutMinutes);
  }

  /**
   * Re-queue failed extraction items for a fresh round of attempts — the
   * recovery path after a transient outage (LLM host down, model missing).
   * Failed is otherwise terminal (3 attempts exhausted). Optionally filter
   * by an error-message substring to target one failure class.
   */
  requeueFailedExtractions(options?: { errorLike?: string }): {
    requeued: number;
  } {
    return { requeued: requeueFailedExtractions(this.db, options) };
  }

  /** Get extraction queue health stats (pending, completed, failed counts). */
  getQueueStats(): QueueStats {
    return getQueueStats(this.db);
  }

  /**
   * Introspect held state: current opinions (beliefs with confidence + full
   * support/challenge evidence + lifecycle) and synthesized observations about
   * a subject. A direct structured lookup — NOT query-ranked recall: there is
   * no confidence floor, so weakly-held or freshly-challenged beliefs stay
   * visible. Pure read, no LLM/embedding.
   *
   * Projection only: reports held state, does not judge whether a candidate
   * statement is consistent with it (the consistency check is a deferred,
   * separate primitive).
   *
   * @param subject — lexical subject to introspect; omit for top held state overall.
   */
  introspect(subject?: string, options?: IntrospectOptions): IntrospectResult {
    return introspect(this.db, subject, options);
  }

  /** Close the database connection. Call when the agent shuts down. */
  close(): void {
    this.db.close();
  }

  /**
   * Copy this engram to a portable single-file destination.
   *
   * Uses SQLite's online backup API — safe to call while retain/recall are
   * running on this or any other process. Readers and writers on the source
   * are not blocked; the backup sees a consistent snapshot.
   *
   * The destination file is a complete standalone `.engram` with no WAL
   * sidecar files, suitable for `git add`, rsync, cloud upload, or archival.
   * Prefer this over `cp` — raw file copy can miss uncommitted WAL frames
   * and leave the backup missing recent writes.
   *
   * @param destPath - Destination path for the backup file. Overwrites if it exists.
   * @returns Total pages copied (one page is typically 4KB).
   */
  async backup(destPath: string): Promise<number> {
    const metadata = await this.db.backup(destPath);
    return metadata.totalPages;
  }

  /**
   * Open a capability-restricted, read-only view over this engram — the
   * read-only guarantee for the Subagent Grounding Layer. The returned
   * ReadonlyEngram exposes ONLY recall / queryContext / introspect (and, via
   * the grounding module, groundSubagent / taskContext), over a second
   * `{ readonly: true }` connection: a stateless subagent handed this view is
   * structurally incapable of mutating the store — no write method exists on
   * the surface, and a raw-SQL escape fails at the driver.
   *
   * Safe to call while retain/recall run on this instance (WAL: concurrent
   * readers never block the writer). The view holds its own connection; call
   * `view.close()` when the subagent is done — it does not affect this handle.
   */
  async readonlyView(): Promise<ReadonlyEngram> {
    return ReadonlyEngram.open(this.dbPath, this.embedder);
  }

  // ---------------------------------------------------------------------------
  // Memory lifecycle — forget / supersede
  // ---------------------------------------------------------------------------

  /**
   * Soft-delete a memory chunk. Sets is_active = FALSE.
   * The chunk remains in the database for audit but is excluded from recall.
   * Returns true if the chunk was found and deactivated.
   */
  async forget(chunkId: string): Promise<boolean> {
    const result = this.db.transaction(() => {
      const chunkResult = this.db
        .prepare(
          `UPDATE chunks SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND is_active = TRUE`,
        )
        .run(chunkId);

      if (chunkResult.changes > 0) {
        this.db
          .prepare(
            `
          UPDATE extraction_queue
          SET status = 'completed', error = 'chunk deactivated'
          WHERE chunk_id = ? AND status IN ('pending', 'processing')
        `,
          )
          .run(chunkId);
      }

      return chunkResult;
    })();

    return result.changes > 0;
  }

  /**
   * Supersede an old fact with new text. The old chunk is soft-deleted and
   * linked to the new one via superseded_by. Use when correcting information:
   * supersede("Tom prefers Terraform" → "Tom switched to Pulumi").
   *
   * Thin wrapper: threads oldChunkId into retain()'s `supersedes` option so
   * the "mark old chunk superseded" write lands inside retain()'s own
   * synchronous transaction — atomic with the new chunk's insert (or dedup
   * update), not a separate statement after the fact. See markSuperseded()
   * in retain.ts.
   */
  async supersede(
    oldChunkId: string,
    newText: string,
    options?: RetainOptions,
  ): Promise<RetainResult> {
    return this.retain(newText, { ...options, supersedes: oldChunkId });
  }

  /**
   * Soft-delete all chunks whose source contains the given pattern.
   * Returns the count of deactivated chunks.
   * Useful for clearing out an entire conversation or document import.
   */
  async forgetBySource(sourcePattern: string): Promise<number> {
    const result = this.db.transaction(() => {
      const chunkIds = this.db
        .prepare(
          `
        SELECT id
        FROM chunks
        WHERE source LIKE ? AND is_active = TRUE
      `,
        )
        .all(`%${sourcePattern}%`) as Array<{ id: string }>;

      const chunkResult = this.db
        .prepare(
          `UPDATE chunks SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE source LIKE ? AND is_active = TRUE`,
        )
        .run(`%${sourcePattern}%`);

      if (chunkIds.length > 0) {
        const updateQueue = this.db.prepare(`
          UPDATE extraction_queue
          SET status = 'completed', error = 'chunk deactivated'
          WHERE chunk_id = ? AND status IN ('pending', 'processing')
        `);
        for (const chunk of chunkIds) {
          updateQueue.run(chunk.id);
        }
      }

      return chunkResult;
    })();

    return result.changes;
  }

  // ---------------------------------------------------------------------------
  // ContextStore — task-scoped ephemeral context (commit/query/expire)
  //
  // Not to be confused with the working_memory session table below: these are
  // immutable, multi-artifact-per-task, RRF-ranked records for agent-to-
  // subagent handoff. See context-store.ts's module doc for the distinction.
  // ---------------------------------------------------------------------------

  /** Commit a DecisionArtifact under a TaskScope. Returns a lightweight ContextRef. */
  async commitContext(
    artifact: DecisionArtifact,
    scope?: TaskScope,
  ): Promise<ContextRef> {
    return commitContext(this.db, this.embedder, artifact, scope);
  }

  /**
   * Query the artifacts committed as children of `ref`, ranked via the same
   * RRF-fusion recall() pipeline as durable memory, truncated to `budget`.
   */
  async queryContext(
    ref: ContextRef,
    relevanceQuery: string,
    budget?: TokenBudget,
  ): Promise<ContextSlice> {
    return queryContext(this.db, this.embedder, ref, relevanceQuery, budget);
  }

  /** Explicitly expire a committed artifact ahead of its natural TTL. */
  async expireContext(ref: ContextRef): Promise<void> {
    return expireContext(this.db, ref);
  }

  /** Promotion seam: moves an artifact into durable scope (no reflect/consolidation wiring yet). */
  async promoteContext(ref: ContextRef): Promise<void> {
    return promoteToDurable(this.db, ref);
  }

  // ---------------------------------------------------------------------------
  // Working Memory — primitives
  // ---------------------------------------------------------------------------

  /**
   * Generate an embedding for arbitrary text using the configured provider.
   * Useful for agents that need embeddings for session matching without
   * going through retain() or recall().
   */
  async embedText(text: string): Promise<Float32Array> {
    return this.embedder.embed(text);
  }

  /**
   * Embed text for a given retrieval mode.
   *
   * `query` applies the query prefix for asymmetric models like nomic-embed-text
   * (better recall quality for search probes). `document` matches how retain()
   * stores text. Falls back to embed() when embedQuery is not implemented by the
   * configured provider (e.g. OllamaEmbeddings, MockEmbedder).
   *
   * Used by the engram_embed MCP tool so engram-aql can obtain model-compatible
   * query vectors for AQL LIKE/PATTERN vector search without reproducing the
   * embedding pipeline in Rust.
   */
  async embedForMode(
    text: string,
    mode: 'query' | 'document',
  ): Promise<Float32Array> {
    if (mode === 'query' && typeof this.embedder.embedQuery === 'function') {
      return this.embedder.embedQuery(text);
    }
    return this.embedder.embed(text);
  }

  /**
   * Find active working memory sessions similar to the given embedding.
   * Returns candidates sorted by similarity (highest first).
   *
   * This is the low-level primitive — it does NOT make a match/new decision.
   * The agent's adapter layer uses these results to implement its own policy.
   */
  findSimilarSessions(
    embedding: Float32Array,
    limit: number = 3,
  ): SessionCandidate[] {
    const embeddingBuffer = embeddingToBuffer(embedding);
    try {
      const rows = this.db
        .prepare(
          `
        SELECT id, data_json,
               vec_distance_cosine(topic_embedding, ?) AS distance
        FROM working_memory
        WHERE (expires_at IS NULL OR expires_at > datetime('now'))
          AND topic_embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
      `,
        )
        .all(embeddingBuffer, limit) as Array<{
        id: string;
        data_json: string;
        distance: number;
      }>;

      return rows.map((c) => ({
        id: c.id,
        state: JSON.parse(c.data_json) as WorkingMemoryState,
        similarity: 1 - (c.distance ?? 1),
      }));
    } catch {
      // sqlite-vec not loaded — no vector matching available
      return [];
    }
  }

  // ---------------------------------------------------------------------------
  // Working Memory — session management
  // ---------------------------------------------------------------------------

  /**
   * Infer which working memory session an incoming message belongs to.
   * Embeds the message, cosine-matches against active sessions, and either
   * resumes the best match or creates a new session. Then loads related
   * long-term context via recall().
   */
  async inferWorkingSession(
    message: string,
    options: WorkingMemoryOptions = {},
  ): Promise<WorkingSessionResult> {
    const maxActive = Math.max(1, options.maxActive ?? 5);
    const threshold = Math.max(0, Math.min(options.threshold ?? 0.55, 1));

    // 1. Embed the incoming message
    const msgEmbedding = await this.embedText(message);
    const embeddingBuffer = embeddingToBuffer(msgEmbedding);

    // 2. Find active sessions and score by similarity
    const candidates = this.findSimilarSessions(msgEmbedding, 3);

    // 3. Pick best match or create new session
    let session: WorkingMemoryState;
    let confidence: number;
    let reason: 'match' | 'new';

    const best = candidates[0];
    if (best && best.similarity >= threshold) {
      // Resume existing session
      session = best.state;
      confidence = best.similarity;
      reason = 'match';

      // Touch the session timestamp
      this.db
        .prepare(
          `UPDATE working_memory SET updated_at = CURRENT_TIMESTAMP WHERE id = ?`,
        )
        .run(best.id);
    } else {
      // Create new session
      session = await this.createWorkingSession(message, embeddingBuffer);
      confidence = 1.0;
      reason = 'new';

      // Enforce maxActive — snapshot oldest if over the cap
      await this.enforceSessionCap(maxActive);
    }

    // 4. Load related long-term context using session goal as seed
    const recallResponse = await this.recall(session.goal, {
      topK: 8,
      snippetChars: 800,
    });
    const relatedContext = formatForPrompt(recallResponse, { maxChars: 1200 });

    return {
      session,
      relatedContext,
      confidence,
      diagnostics: {
        sessionId: session.id,
        reason,
        candidatesEvaluated: candidates.length,
      },
    };
  }

  /**
   * Create a new working memory session from a message and pre-computed embedding.
   * Called internally by inferWorkingSession, but exposed for agents that
   * implement custom session resolution using the primitives.
   */
  async createWorkingSession(
    message: string,
    embeddingBuffer: Buffer,
  ): Promise<WorkingMemoryState> {
    const id = `wm-${randomUUID().substring(0, 12)}`;
    const now = new Date().toISOString();

    const state: WorkingMemoryState = {
      id,
      goal: message.slice(0, 200),
      updated_at: now,
    };

    this.db
      .prepare(
        `
      INSERT INTO working_memory (id, data_json, seed_query, topic_embedding)
      VALUES (?, ?, ?, ?)
    `,
      )
      .run(id, JSON.stringify(state), message.slice(0, 200), embeddingBuffer);

    return state;
  }

  /**
   * Update an existing working memory session with new state.
   * Merges partial updates into the existing data_json and re-embeds
   * the seed query for future similarity matching.
   */
  async updateWorkingSession(
    sessionId: string,
    updates: Partial<WorkingMemoryState>,
  ): Promise<void> {
    const row = this.db
      .prepare(
        `SELECT data_json FROM working_memory WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`,
      )
      .get(sessionId) as { data_json: string } | undefined;

    if (!row)
      throw new Error(
        `Working memory session ${sessionId} not found or expired`,
      );

    const existing = JSON.parse(row.data_json) as WorkingMemoryState;
    const merged: WorkingMemoryState = {
      ...existing,
      ...updates,
      id: sessionId, // never overwrite ID
      updated_at: new Date().toISOString(),
    };

    const seedQuery = `${merged.goal}`.trim();
    const embedding = await this.embedder.embed(seedQuery);
    const embeddingBuffer = embeddingToBuffer(embedding);

    this.db
      .prepare(
        `
      UPDATE working_memory
      SET data_json = ?, seed_query = ?, topic_embedding = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
      )
      .run(JSON.stringify(merged), seedQuery, embeddingBuffer, sessionId);
  }

  /**
   * Get the current state of a working memory session.
   * Returns null if the session doesn't exist or has expired.
   */
  getWorkingSession(sessionId: string): WorkingMemoryState | null {
    const row = this.db
      .prepare(
        `SELECT data_json FROM working_memory WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`,
      )
      .get(sessionId) as { data_json: string } | undefined;

    return row ? (JSON.parse(row.data_json) as WorkingMemoryState) : null;
  }

  /**
   * List all active (non-expired) working memory sessions.
   */
  listWorkingSessions(): WorkingMemoryState[] {
    const rows = this.db
      .prepare(
        `
      SELECT data_json FROM working_memory
      WHERE expires_at IS NULL OR expires_at > datetime('now')
      ORDER BY updated_at DESC
    `,
      )
      .all() as Array<{ data_json: string }>;

    return rows.map((r) => JSON.parse(r.data_json) as WorkingMemoryState);
  }

  /**
   * Snapshot a working memory session to long-term episodic memory,
   * then mark it as expired.
   */
  async snapshotWorkingSession(sessionId: string): Promise<RetainResult> {
    const row = this.db
      .prepare(`SELECT data_json, seed_query FROM working_memory WHERE id = ?`)
      .get(sessionId) as
      | { data_json: string; seed_query: string | null }
      | undefined;

    if (!row) throw new Error(`Working memory session ${sessionId} not found`);

    const state = JSON.parse(row.data_json) as WorkingMemoryState;
    const progressNotes = (state.progress as string | undefined) ?? '';
    const summary = [
      `Session goal: ${state.goal}`,
      progressNotes ? `Progress: ${progressNotes}` : '',
    ]
      .filter(Boolean)
      .join(' ');

    const result = await this.retain(summary, {
      memoryType: 'experience',
      source: `working_memory:${sessionId}`,
      sourceType: 'agent_generated',
      trustScore: 0.6,
      skipExtraction: false,
    });

    // Mark as expired
    this.db
      .prepare(
        `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`,
      )
      .run(sessionId);

    return result;
  }

  /**
   * Expire and snapshot all sessions that haven't been updated within
   * the given threshold. Call this from a background maintenance tick.
   */
  async expireStaleWorkingSessions(maxAgeHours: number = 48): Promise<number> {
    const stale = this.db
      .prepare(
        `
      SELECT id FROM working_memory
      WHERE updated_at <= datetime('now', '-' || ? || ' hours')
        AND (expires_at IS NULL OR expires_at > datetime('now'))
    `,
      )
      .all(String(maxAgeHours)) as Array<{ id: string }>;

    for (const { id } of stale) {
      try {
        await this.snapshotWorkingSession(id);
      } catch {
        // If snapshot fails, still expire it
        this.db
          .prepare(
            `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`,
          )
          .run(id);
      }
    }

    return stale.length;
  }

  /**
   * Clear a specific working memory session without snapshotting.
   */
  clearWorkingSession(sessionId: string): boolean {
    const result = this.db
      .prepare(
        `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`,
      )
      .run(sessionId);
    return result.changes > 0;
  }

  /**
   * Enforce the maximum active session cap.
   * Snapshots the oldest sessions that exceed the cap.
   */
  private async enforceSessionCap(maxActive: number): Promise<void> {
    const active = this.db
      .prepare(
        `
      SELECT id FROM working_memory
      WHERE expires_at IS NULL OR expires_at > datetime('now')
      ORDER BY updated_at ASC
    `,
      )
      .all() as Array<{ id: string }>;

    const excess = active.length - maxActive;
    if (excess <= 0) return;

    for (let i = 0; i < excess; i++) {
      try {
        await this.snapshotWorkingSession(active[i].id);
      } catch {
        this.db
          .prepare(
            `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`,
          )
          .run(active[i].id);
      }
    }
  }
}
