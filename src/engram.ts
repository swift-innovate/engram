// =============================================================================
// engram.ts - Unified Engram Class
//
// Public API surface for the Engram library. Wraps retain/recall/reflect into
// a single class with a clean lifecycle:
//
//   const mira = await Engram.create('./mira.engram', { ... });
//   await mira.retain('Tom prefers Terraform', { memoryType: 'world', ... });
//   const context = await mira.recall('IaC tools?', { topK: 5 });
//   await mira.reflect();
//   mira.close();
//
// Schema bootstrap:
//   Reads schema.sql from the same directory as this compiled file (dist/).
//   The build step copies src/schema.sql → dist/schema.sql.
//   All CREATE TABLE statements are IF NOT EXISTS — safe to run on every open.
//
// Connection model:
//   Engram holds one persistent connection for retain/recall (low-latency).
//   reflect() opens a separate connection internally (reflect.ts pattern).
//   SQLite WAL mode supports concurrent readers + one writer — both are safe.
// =============================================================================

import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import {
  retain,
  retainBatch,
  processExtractionQueue,
  OllamaEmbeddings,
  LocalEmbedder,
  shouldRetain,
  type RetainOptions,
  type RetainResult,
  type EmbeddingProvider,
} from './retain.js';

import { recall, formatForPrompt, type RecallOptions, type RecallResponse, type RecallResult, type FormatForPromptOptions } from './recall.js';

import {
  reflect,
  ReflectScheduler,
  type ReflectConfig,
  type ReflectResult,
} from './reflect.js';

import type {
  WorkingMemoryState,
  WorkingMemoryOptions,
  WorkingSessionResult,
  SessionCandidate,
} from './working-memory-types.js';

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
  FormatForPromptOptions,
  WorkingMemoryState,
  WorkingMemoryOptions,
  WorkingSessionResult,
  SessionCandidate,
};
export { OllamaEmbeddings, LocalEmbedder, ReflectScheduler, shouldRetain, formatForPrompt };

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
  /** Ollama endpoint (default: http://starbase:40114) */
  ollamaUrl?: string;
  /** Embedding model (default: nomic-embed-text) */
  embedModel?: string;
  /** Embedding dimensions (default: 768 for nomic-embed-text) */
  embedDimensions?: number;
  /** LLM for extraction + reflection (default: llama3.1:8b) */
  reflectModel?: string;
  /** Override the embedding provider — useful for testing without Ollama */
  embedder?: EmbeddingProvider;
  /** Use Ollama for embeddings instead of local Transformers.js (default: false) */
  useOllamaEmbeddings?: boolean;
}

// =============================================================================
// Engram Class
// =============================================================================

export class Engram {
  private readonly db: Database.Database;
  private readonly dbPath: string;
  private readonly embedder: EmbeddingProvider;
  private readonly ollamaUrl: string;
  private readonly reflectModel: string;

  private constructor(
    db: Database.Database,
    dbPath: string,
    embedder: EmbeddingProvider,
    ollamaUrl: string,
    reflectModel: string
  ) {
    this.db = db;
    this.dbPath = dbPath;
    this.embedder = embedder;
    this.ollamaUrl = ollamaUrl;
    this.reflectModel = reflectModel;
  }

  // ---------------------------------------------------------------------------
  // Internal factory — shared by create() and open()
  // ---------------------------------------------------------------------------

  private static async init(path: string, options: EngramOptions): Promise<Engram> {
    const {
      ollamaUrl = 'http://starbase:40114',
      embedModel,
      embedDimensions,
      reflectModel = 'llama3.1:8b',
      embedder: injectedEmbedder,
      useOllamaEmbeddings = false,
    } = options;

    const db = new Database(path);

    // Load sqlite-vec for vector search. Graceful fallback: semantic strategy
    // is simply skipped (recall.ts catches the error) if the extension is absent.
    try {
      const mod = await import('sqlite-vec') as unknown as { load: (db: Database.Database) => void };
      mod.load(db);
    } catch {
      // sqlite-vec not installed — semantic search disabled
    }

    // Bootstrap schema — idempotent, safe to run on every open
    const schemaPath = join(__dirname, 'schema.sql');
    const schema = readFileSync(schemaPath, 'utf8');
    db.exec(schema);

    // Embedding provider selection (priority order):
    // 1. Injected embedder — caller controls everything (used in tests)
    // 2. Ollama embeddings — opt-in via useOllamaEmbeddings flag
    // 3. Local Transformers.js — default, zero network dependency
    let embedder: EmbeddingProvider;
    if (injectedEmbedder) {
      embedder = injectedEmbedder;
    } else if (useOllamaEmbeddings) {
      embedder = new OllamaEmbeddings(
        ollamaUrl,
        embedModel ?? 'nomic-embed-text',
        embedDimensions ?? 768,
      );
    } else {
      const localModel = embedModel ?? 'Xenova/nomic-embed-text-v1.5';
      const local = new LocalEmbedder(localModel);
      await local.init();
      embedder = local;
    }

    return new Engram(db, path, embedder, ollamaUrl, reflectModel);
  }

  private upsertBankConfig(options: EngramOptions): void {
    const upsert = this.db.prepare(`
      INSERT INTO bank_config (key, value) VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
    `);
    const tx = this.db.transaction(() => {
      if (options.reflectMission !== undefined) upsert.run('reflect_mission', options.reflectMission);
      if (options.retainMission !== undefined)  upsert.run('retain_mission',  options.retainMission);
      if (options.disposition  !== undefined)  upsert.run('disposition',     JSON.stringify(options.disposition));
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
  static async create(path: string, options: EngramOptions = {}): Promise<Engram> {
    const engram = await Engram.init(path, options);
    engram.upsertBankConfig(options);
    return engram;
  }

  /**
   * Open an existing engram. Schema is bootstrapped if missing (safe on new files too).
   * Bank config is updated only for options that are explicitly provided.
   */
  static async open(path: string, options: EngramOptions = {}): Promise<Engram> {
    const engram = await Engram.init(path, options);
    const configKeys: Array<keyof EngramOptions> = ['reflectMission', 'retainMission', 'disposition'];
    if (configKeys.some(k => options[k] !== undefined)) {
      engram.upsertBankConfig(options);
    }
    return engram;
  }

  // ---------------------------------------------------------------------------
  // Core operations
  // ---------------------------------------------------------------------------

  /** Store a memory trace. Fast path — embeds locally, no LLM call. ~5ms. */
  async retain(text: string, options?: RetainOptions): Promise<RetainResult> {
    return retain(this.db, text, this.embedder, options);
  }

  /**
   * Bulk store. Queues all entity extractions at the end rather than one-by-one.
   * Efficient for importing conversations or documents.
   */
  async retainBatch(
    items: Array<{ text: string; options?: RetainOptions }>,
    onProgress?: (current: number, total: number) => void
  ): Promise<RetainResult[]> {
    return retainBatch(this.db, items, this.embedder, onProgress);
  }

  /**
   * Retrieve relevant memories via 4-way retrieval (semantic + keyword + graph + temporal),
   * fused via Reciprocal Rank Fusion and weighted by trust score.
   */
  async recall(query: string, options?: RecallOptions): Promise<RecallResponse> {
    return recall(this.db, query, this.embedder, options);
  }

  /**
   * Run a reflection cycle: processes unreflected facts through the LLM,
   * synthesizes observations, and reinforces/challenges opinions.
   *
   * Opens a separate DB connection internally (WAL mode: safe alongside
   * idle retain/recall connections).
   */
  async reflect(): Promise<ReflectResult> {
    return reflect({
      dbPath: this.dbPath,
      ollamaUrl: this.ollamaUrl,
      reflectModel: this.reflectModel,
    });
  }

  /**
   * Drain the entity extraction queue. Calls Ollama to extract entities and
   * relations from retained chunks, building out the knowledge graph.
   */
  async processExtractions(batchSize: number = 10): Promise<{ processed: number; failed: number }> {
    return processExtractionQueue(this.db, this.ollamaUrl, this.reflectModel, batchSize);
  }

  /** Close the database connection. Call when the agent shuts down. */
  close(): void {
    this.db.close();
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
      const chunkResult = this.db.prepare(
        `UPDATE chunks SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND is_active = TRUE`
      ).run(chunkId);

      if (chunkResult.changes > 0) {
        this.db.prepare(`
          UPDATE extraction_queue
          SET status = 'completed', error = 'chunk deactivated'
          WHERE chunk_id = ? AND status IN ('pending', 'processing')
        `).run(chunkId);
      }

      return chunkResult;
    })();

    return result.changes > 0;
  }

  /**
   * Supersede an old fact with new text. The old chunk is soft-deleted and
   * linked to the new one via superseded_by. Use when correcting information:
   * supersede("Tom prefers Terraform" → "Tom switched to Pulumi").
   */
  async supersede(oldChunkId: string, newText: string, options?: RetainOptions): Promise<RetainResult> {
    const newResult = await this.retain(newText, options);
    this.db.prepare(
      `UPDATE chunks SET is_active = FALSE, superseded_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?`
    ).run(newResult.chunkId, oldChunkId);
    return newResult;
  }

  /**
   * Soft-delete all chunks whose source contains the given pattern.
   * Returns the count of deactivated chunks.
   * Useful for clearing out an entire conversation or document import.
   */
  async forgetBySource(sourcePattern: string): Promise<number> {
    const result = this.db.transaction(() => {
      const chunkIds = this.db.prepare(`
        SELECT id
        FROM chunks
        WHERE source LIKE ? AND is_active = TRUE
      `).all(`%${sourcePattern}%`) as Array<{ id: string }>;

      const chunkResult = this.db.prepare(
        `UPDATE chunks SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE source LIKE ? AND is_active = TRUE`
      ).run(`%${sourcePattern}%`);

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
   * Find active working memory sessions similar to the given embedding.
   * Returns candidates sorted by similarity (highest first).
   *
   * This is the low-level primitive — it does NOT make a match/new decision.
   * The agent's adapter layer uses these results to implement its own policy.
   */
  findSimilarSessions(embedding: Float32Array, limit: number = 3): SessionCandidate[] {
    const embeddingBuffer = Buffer.from(embedding.buffer);
    try {
      const rows = this.db.prepare(`
        SELECT id, data_json,
               vec_distance_cosine(topic_embedding, ?) AS distance
        FROM working_memory
        WHERE (expires_at IS NULL OR expires_at > datetime('now'))
          AND topic_embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
      `).all(embeddingBuffer, limit) as Array<{ id: string; data_json: string; distance: number }>;

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
    options: WorkingMemoryOptions = {}
  ): Promise<WorkingSessionResult> {
    const maxActive = Math.max(1, options.maxActive ?? 5);
    const threshold = Math.max(0, Math.min(options.threshold ?? 0.55, 1));

    // 1. Embed the incoming message
    const msgEmbedding = await this.embedText(message);
    const embeddingBuffer = Buffer.from(msgEmbedding.buffer);

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
      this.db.prepare(
        `UPDATE working_memory SET updated_at = CURRENT_TIMESTAMP WHERE id = ?`
      ).run(best.id);
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
    embeddingBuffer: Buffer
  ): Promise<WorkingMemoryState> {
    const id = `wm-${randomUUID().substring(0, 12)}`;
    const now = new Date().toISOString();

    const state: WorkingMemoryState = {
      id,
      goal: message.slice(0, 200),
      updated_at: now,
    };

    this.db.prepare(`
      INSERT INTO working_memory (id, data_json, seed_query, topic_embedding)
      VALUES (?, ?, ?, ?)
    `).run(id, JSON.stringify(state), message.slice(0, 200), embeddingBuffer);

    return state;
  }

  /**
   * Update an existing working memory session with new state.
   * Merges partial updates into the existing data_json and re-embeds
   * the seed query for future similarity matching.
   */
  async updateWorkingSession(
    sessionId: string,
    updates: Partial<WorkingMemoryState>
  ): Promise<void> {
    const row = this.db.prepare(
      `SELECT data_json FROM working_memory WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`
    ).get(sessionId) as { data_json: string } | undefined;

    if (!row) throw new Error(`Working memory session ${sessionId} not found or expired`);

    const existing = JSON.parse(row.data_json) as WorkingMemoryState;
    const merged: WorkingMemoryState = {
      ...existing,
      ...updates,
      id: sessionId, // never overwrite ID
      updated_at: new Date().toISOString(),
    };

    const seedQuery = `${merged.goal}`.trim();
    const embedding = await this.embedder.embed(seedQuery);
    const embeddingBuffer = Buffer.from(embedding.buffer);

    this.db.prepare(`
      UPDATE working_memory
      SET data_json = ?, seed_query = ?, topic_embedding = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `).run(JSON.stringify(merged), seedQuery, embeddingBuffer, sessionId);
  }

  /**
   * Get the current state of a working memory session.
   * Returns null if the session doesn't exist or has expired.
   */
  getWorkingSession(sessionId: string): WorkingMemoryState | null {
    const row = this.db.prepare(
      `SELECT data_json FROM working_memory WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`
    ).get(sessionId) as { data_json: string } | undefined;

    return row ? JSON.parse(row.data_json) as WorkingMemoryState : null;
  }

  /**
   * List all active (non-expired) working memory sessions.
   */
  listWorkingSessions(): WorkingMemoryState[] {
    const rows = this.db.prepare(`
      SELECT data_json FROM working_memory
      WHERE expires_at IS NULL OR expires_at > datetime('now')
      ORDER BY updated_at DESC
    `).all() as Array<{ data_json: string }>;

    return rows.map(r => JSON.parse(r.data_json) as WorkingMemoryState);
  }

  /**
   * Snapshot a working memory session to long-term episodic memory,
   * then mark it as expired.
   */
  async snapshotWorkingSession(sessionId: string): Promise<RetainResult> {
    const row = this.db.prepare(
      `SELECT data_json, seed_query FROM working_memory WHERE id = ?`
    ).get(sessionId) as { data_json: string; seed_query: string | null } | undefined;

    if (!row) throw new Error(`Working memory session ${sessionId} not found`);

    const state = JSON.parse(row.data_json) as WorkingMemoryState;
    const progressNotes = (state.progress as string | undefined) ?? '';
    const summary = [
      `Session goal: ${state.goal}`,
      progressNotes ? `Progress: ${progressNotes}` : '',
    ].filter(Boolean).join(' ');

    const result = await this.retain(summary, {
      memoryType: 'experience',
      source: `working_memory:${sessionId}`,
      sourceType: 'agent_generated',
      trustScore: 0.6,
      skipExtraction: false,
    });

    // Mark as expired
    this.db.prepare(
      `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`
    ).run(sessionId);

    return result;
  }

  /**
   * Expire and snapshot all sessions that haven't been updated within
   * the given threshold. Call this from a background maintenance tick.
   */
  async expireStaleWorkingSessions(maxAgeHours: number = 48): Promise<number> {
    const stale = this.db.prepare(`
      SELECT id FROM working_memory
      WHERE updated_at <= datetime('now', '-' || ? || ' hours')
        AND (expires_at IS NULL OR expires_at > datetime('now'))
    `).all(String(maxAgeHours)) as Array<{ id: string }>;

    for (const { id } of stale) {
      try {
        await this.snapshotWorkingSession(id);
      } catch {
        // If snapshot fails, still expire it
        this.db.prepare(
          `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`
        ).run(id);
      }
    }

    return stale.length;
  }

  /**
   * Clear a specific working memory session without snapshotting.
   */
  clearWorkingSession(sessionId: string): boolean {
    const result = this.db.prepare(
      `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`
    ).run(sessionId);
    return result.changes > 0;
  }

  /**
   * Enforce the maximum active session cap.
   * Snapshots the oldest sessions that exceed the cap.
   */
  private async enforceSessionCap(maxActive: number): Promise<void> {
    const active = this.db.prepare(`
      SELECT id FROM working_memory
      WHERE expires_at IS NULL OR expires_at > datetime('now')
      ORDER BY updated_at ASC
    `).all() as Array<{ id: string }>;

    const excess = active.length - maxActive;
    if (excess <= 0) return;

    for (let i = 0; i < excess; i++) {
      try {
        await this.snapshotWorkingSession(active[i].id);
      } catch {
        this.db.prepare(
          `UPDATE working_memory SET expires_at = datetime('now') WHERE id = ?`
        ).run(active[i].id);
      }
    }
  }
}
