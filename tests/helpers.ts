// =============================================================================
// Test helpers
// Shared utilities for all test suites. Designed to avoid Ollama dependency:
//   - MockEmbedder: deterministic vectors, no HTTP
//   - mockOllamaFetch: canned responses for extraction + reflection
//   - createTestDb: in-memory SQLite with schema bootstrapped
// =============================================================================

import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';
import type { EmbeddingProvider } from '../src/retain.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

export function loadSchema(db: Database.Database): void {
  const schema = readFileSync(join(__dirname, '../src/schema.sql'), 'utf8');
  db.exec(schema);
}

/** In-memory SQLite with schema + sqlite-vec loaded (if available). */
export function createTestDb(): Database.Database {
  const db = new Database(':memory:');
  try {
    const vec = require('sqlite-vec') as { load: (db: Database.Database) => void };
    vec.load(db);
  } catch {
    // sqlite-vec unavailable — semantic search tests will be skipped
  }
  loadSchema(db);
  return db;
}

/** Temp file path for tests that need a real file (reflect, Engram). */
export function tmpDbPath(): string {
  return join(tmpdir(), `engram-test-${randomUUID()}.sqlite`);
}

/** Remove a temp db and its WAL/SHM sidecar files. */
export function cleanupDb(path: string): void {
  for (const suffix of ['', '-wal', '-shm']) {
    try {
      const { unlinkSync } = require('fs');
      unlinkSync(path + suffix);
    } catch {
      // file may not exist
    }
  }
}

// ---------------------------------------------------------------------------
// MockEmbedder
// ---------------------------------------------------------------------------

/**
 * Deterministic, Ollama-free embedder for testing.
 * Produces L2-normalized float32 vectors by distributing character codes
 * across dimensions — same text always produces the same vector.
 *
 * Default 8 dimensions keeps the buffer small while still producing valid
 * cosine-distance comparisons when sqlite-vec is loaded.
 */
export class MockEmbedder implements EmbeddingProvider {
  readonly dimensions: number;

  constructor(dimensions: number = 8) {
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<Float32Array> {
    const vec = new Float32Array(this.dimensions);
    for (let i = 0; i < text.length; i++) {
      vec[i % this.dimensions] += text.charCodeAt(i) / 256;
    }
    const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return mag > 0 ? new Float32Array(vec.map(v => v / mag)) : vec;
  }
}

// ---------------------------------------------------------------------------
// Ollama mock responses
// ---------------------------------------------------------------------------

/** Wrap a JSON string as a fake Ollama /api/generate response. */
export function mockOllamaFetch(responseJson: string): typeof globalThis.fetch {
  return async (_url: string | URL | Request, _init?: RequestInit): Promise<Response> =>
    ({
      ok: true,
      status: 200,
      json: async () => ({ response: responseJson }),
      text: async () => responseJson,
    }) as unknown as Response;
}

/** Canned entity extraction: Alice → Rust, prefers relation. */
export const EXTRACTION_RESPONSE = JSON.stringify({
  entities: [
    { name: 'Alice', canonical_name: 'alice', entity_type: 'person', aliases: [] },
    { name: 'Rust', canonical_name: 'rust', entity_type: 'technology', aliases: ['rs'] },
  ],
  relations: [
    { source: 'alice', target: 'rust', relation_type: 'prefers', description: 'Alice prefers Rust' },
  ],
});

/** Canned reflect output: one observation, one new opinion. */
export const REFLECT_RESPONSE = JSON.stringify({
  observations: [
    {
      summary: 'Alice consistently chooses Rust for systems programming tasks',
      domain: 'preferences',
      topic: 'programming languages',
      source_chunk_ids: [],
      entity_names: ['Alice'],
    },
  ],
  opinion_updates: [
    {
      belief: 'Alice strongly prefers Rust over other systems languages',
      direction: 'new',
      confidence_delta: 0.2,
      domain: 'preferences',
      evidence_chunk_ids: [],
      entity_names: ['Alice'],
    },
  ],
  observation_refreshes: [],
});
