import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain, retainBatch, processExtractionQueue } from '../src/retain.js';
import { createTestDb, MockEmbedder, mockOllamaFetch, EXTRACTION_RESPONSE } from './helpers.js';

// ---------------------------------------------------------------------------
// retain()
// ---------------------------------------------------------------------------

describe('retain()', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('stores chunk with correct fields', async () => {
    const result = await retain(db, 'Tom prefers Terraform', embedder, {
      memoryType: 'world',
      sourceType: 'user_stated',
      trustScore: 0.9,
      source: 'conversation:test',
      context: 'infrastructure',
    });

    expect(result.chunkId).toMatch(/^chk-/);
    const chunk = db.prepare('SELECT * FROM chunks WHERE id = ?').get(result.chunkId) as any;
    expect(chunk.text).toBe('Tom prefers Terraform');
    expect(chunk.memory_type).toBe('world');
    expect(chunk.source_type).toBe('user_stated');
    expect(chunk.trust_score).toBe(0.9);
    expect(chunk.source).toBe('conversation:test');
    expect(chunk.context).toBe('infrastructure');
    expect(chunk.is_active).toBe(1);
    expect(chunk.reflected_at).toBeNull();
  });

  it('stores embedding as a buffer of the correct size', async () => {
    const result = await retain(db, 'test text', embedder);
    const chunk = db.prepare('SELECT embedding FROM chunks WHERE id = ?').get(result.chunkId) as any;
    // MockEmbedder dimensions=8, each float32 = 4 bytes → 32 bytes total
    expect(chunk.embedding).toBeInstanceOf(Buffer);
    expect(chunk.embedding.byteLength).toBe(embedder.dimensions * 4);
  });

  it('applies default options', async () => {
    const result = await retain(db, 'plain text', embedder);
    const chunk = db.prepare('SELECT * FROM chunks WHERE id = ?').get(result.chunkId) as any;
    expect(chunk.memory_type).toBe('world');
    expect(chunk.source_type).toBe('inferred');
    expect(chunk.trust_score).toBe(0.5);
  });

  it('stores temporal fields', async () => {
    const result = await retain(db, 'event', embedder, {
      eventTime: '2025-01-15T00:00:00Z',
      temporalLabel: 'mid-January 2025',
    });
    const chunk = db.prepare('SELECT * FROM chunks WHERE id = ?').get(result.chunkId) as any;
    expect(chunk.event_time).toBe('2025-01-15T00:00:00Z');
    expect(chunk.temporal_label).toBe('mid-January 2025');
  });

  it('queues extraction for world type', async () => {
    const result = await retain(db, 'hello world', embedder, { memoryType: 'world' });
    expect(result.queued).toBe(true);
    const row = db.prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?').get(result.chunkId);
    expect(row).toBeTruthy();
  });

  it('queues extraction for experience type', async () => {
    const result = await retain(db, 'I did a thing', embedder, { memoryType: 'experience' });
    expect(result.queued).toBe(true);
  });

  it('does not queue for observation type', async () => {
    const result = await retain(db, 'observed pattern', embedder, { memoryType: 'observation' });
    expect(result.queued).toBe(false);
    // better-sqlite3 .get() returns undefined (not null) when no row exists
    const row = db.prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?').get(result.chunkId);
    expect(row).toBeUndefined();
  });

  it('does not queue for opinion type', async () => {
    const result = await retain(db, 'I believe this', embedder, { memoryType: 'opinion' });
    expect(result.queued).toBe(false);
  });

  it('skips queueing when skipExtraction is true', async () => {
    const result = await retain(db, 'hello', embedder, { memoryType: 'world', skipExtraction: true });
    expect(result.queued).toBe(false);
    const row = db.prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?').get(result.chunkId);
    expect(row).toBeUndefined();
  });

  it('generates unique chunk IDs across multiple retains', async () => {
    const r1 = await retain(db, 'first', embedder);
    const r2 = await retain(db, 'second', embedder);
    expect(r1.chunkId).not.toBe(r2.chunkId);
  });
});

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

describe('retain() — deduplication', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('exact dedup: retaining identical text returns existing chunk', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder);
    const r2 = await retain(db, 'Tom prefers Terraform', embedder);

    expect(r2.chunkId).toBe(r1.chunkId);
    expect(r2.deduplicated).toBe(true);

    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as { n: number };
    expect(count.n).toBe(1);
  });

  it('exact dedup: reinforces trust score with the higher value', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder, { trustScore: 0.5 });
    await retain(db, 'Tom prefers Terraform', embedder, { trustScore: 0.9 });

    const chunk = db.prepare('SELECT trust_score FROM chunks WHERE id = ?').get(r1.chunkId) as any;
    expect(chunk.trust_score).toBe(0.9);
  });

  it('dedupMode none: always creates new chunks', async () => {
    await retain(db, 'Tom prefers Terraform', embedder, { dedupMode: 'none' });
    const r2 = await retain(db, 'Tom prefers Terraform', embedder, { dedupMode: 'none' });

    expect(r2.deduplicated).toBeUndefined();
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as { n: number };
    expect(count.n).toBe(2);
  });

  it('dedupMode normalized: deduplicates despite case/whitespace differences', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder, { dedupMode: 'normalized' });
    const r2 = await retain(db, '  tom prefers terraform  ', embedder, { dedupMode: 'normalized' });

    expect(r2.chunkId).toBe(r1.chunkId);
    expect(r2.deduplicated).toBe(true);
  });

  it('exact dedup: different text creates new chunk', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder);
    const r2 = await retain(db, 'Tom prefers Pulumi', embedder);

    expect(r1.chunkId).not.toBe(r2.chunkId);
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as { n: number };
    expect(count.n).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// retainBatch()
// ---------------------------------------------------------------------------

describe('retainBatch()', () => {
  it('stores all items and returns all results', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();

    const results = await retainBatch(db, [
      { text: 'first', options: { memoryType: 'world' } },
      { text: 'second', options: { memoryType: 'world' } },
      { text: 'third', options: { memoryType: 'world' } },
    ], embedder);

    expect(results).toHaveLength(3);
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as { n: number };
    expect(count.n).toBe(3);
    db.close();
  });

  it('reports progress for each item', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const progress: Array<[number, number]> = [];

    await retainBatch(db, [
      { text: 'a' }, { text: 'b' }, { text: 'c' },
    ], embedder, (current, total) => progress.push([current, total]));

    expect(progress).toEqual([[1, 3], [2, 3], [3, 3]]);
    db.close();
  });

  it('queues world and experience items but not observation/opinion', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();

    await retainBatch(db, [
      { text: 'world fact', options: { memoryType: 'world' } },
      { text: 'my experience', options: { memoryType: 'experience' } },
      { text: 'observed pattern', options: { memoryType: 'observation' } },
    ], embedder);

    const count = db.prepare('SELECT count(*) as n FROM extraction_queue').get() as { n: number };
    expect(count.n).toBe(2);
    db.close();
  });
});

// ---------------------------------------------------------------------------
// processExtractionQueue()
// ---------------------------------------------------------------------------

describe('processExtractionQueue()', () => {
  it('extracts entities and relations from queued chunks', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    vi.stubGlobal('fetch', mockOllamaFetch(EXTRACTION_RESPONSE));

    await retain(db, 'Alice prefers Rust for systems programming', embedder, {
      memoryType: 'world',
    });

    const result = await processExtractionQueue(db);
    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);

    // Entities created
    const entities = db.prepare('SELECT * FROM entities').all() as any[];
    expect(entities.some(e => e.canonical_name === 'alice')).toBe(true);
    expect(entities.some(e => e.canonical_name === 'rust')).toBe(true);

    // Relation created
    const relations = db.prepare('SELECT * FROM relations').all() as any[];
    expect(relations.length).toBeGreaterThan(0);
    expect(relations[0].relation_type).toBe('prefers');

    // chunk_entities linked
    const links = db.prepare('SELECT * FROM chunk_entities').all() as any[];
    expect(links.length).toBeGreaterThan(0);

    // Queue item marked complete
    const queued = db.prepare("SELECT status FROM extraction_queue LIMIT 1").get() as any;
    expect(queued.status).toBe('completed');

    db.close();
  });

  it('marks queue item as failed on Ollama error', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 500, text: async () => 'error' } as unknown as Response));

    await retain(db, 'some text', embedder, { memoryType: 'world' });
    const result = await processExtractionQueue(db);

    expect(result.failed).toBe(1);
    const queued = db.prepare("SELECT status FROM extraction_queue LIMIT 1").get() as any;
    expect(queued.status).toBe('failed');
    db.close();
  });

  it('returns zero when queue is empty', async () => {
    const db = createTestDb();
    const result = await processExtractionQueue(db);
    expect(result.processed).toBe(0);
    expect(result.failed).toBe(0);
    db.close();
  });
});
