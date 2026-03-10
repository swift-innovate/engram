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
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import {
  retain,
  retainBatch,
  processExtractionQueue,
  OllamaEmbeddings,
  type RetainOptions,
  type RetainResult,
  type EmbeddingProvider,
} from './retain.js';

import { recall, type RecallOptions, type RecallResponse, type RecallResult } from './recall.js';

import {
  reflect,
  ReflectScheduler,
  type ReflectConfig,
  type ReflectResult,
} from './reflect.js';

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
};
export { OllamaEmbeddings, ReflectScheduler };

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
  /** Embedding model (default: nomic-embed-text) */
  embedModel?: string;
  /** Embedding dimensions (default: 768 for nomic-embed-text) */
  embedDimensions?: number;
  /** LLM for extraction + reflection (default: llama3.1:8b) */
  reflectModel?: string;
  /** Override the embedding provider — useful for testing without Ollama */
  embedder?: EmbeddingProvider;
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
      ollamaUrl = 'http://localhost:11434',
      embedModel = 'nomic-embed-text',
      embedDimensions = 768,
      reflectModel = 'llama3.1:8b',
      embedder: injectedEmbedder,
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

    const embedder = injectedEmbedder ?? new OllamaEmbeddings(ollamaUrl, embedModel, embedDimensions);
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
}
