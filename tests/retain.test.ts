import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import {
  retain,
  retainBatch,
  processExtractionQueue,
  recoverStalledExtractions,
  requeueFailedExtractions,
  getQueueStats,
  chunkText,
} from '../src/retain.js';
import {
  createTestDb,
  MockEmbedder,
  MockGenerator,
  EXTRACTION_RESPONSE,
  loadSchema,
  tmpDbPath,
  cleanupDb,
} from './helpers.js';

// ---------------------------------------------------------------------------
// retain()
// ---------------------------------------------------------------------------

describe('retain()', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
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
    const chunk = db
      .prepare('SELECT * FROM chunks WHERE id = ?')
      .get(result.chunkId) as any;
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
    const chunk = db
      .prepare('SELECT embedding FROM chunks WHERE id = ?')
      .get(result.chunkId) as any;
    // MockEmbedder dimensions=8, each float32 = 4 bytes → 32 bytes total
    expect(chunk.embedding).toBeInstanceOf(Buffer);
    expect(chunk.embedding.byteLength).toBe(embedder.dimensions * 4);
  });

  it('stores only the active slice of a Float32Array embedding', async () => {
    class SlicedEmbedder {
      readonly dimensions = 2;
      private readonly backing = new Float32Array([1, 2, 3, 4]);

      async embed(_text: string): Promise<Float32Array> {
        return new Float32Array(this.backing.buffer, 8, 2);
      }
    }

    const result = await retain(db, 'slice test', new SlicedEmbedder() as any);
    const chunk = db
      .prepare('SELECT embedding FROM chunks WHERE id = ?')
      .get(result.chunkId) as any;
    expect(chunk.embedding.byteLength).toBe(2 * 4);
  });

  it('applies default options', async () => {
    const result = await retain(db, 'plain text', embedder);
    const chunk = db
      .prepare('SELECT * FROM chunks WHERE id = ?')
      .get(result.chunkId) as any;
    expect(chunk.memory_type).toBe('world');
    expect(chunk.source_type).toBe('inferred');
    expect(chunk.trust_score).toBe(0.5);
  });

  it('stores temporal fields', async () => {
    const result = await retain(db, 'event', embedder, {
      eventTime: '2025-01-15T00:00:00Z',
      temporalLabel: 'mid-January 2025',
    });
    const chunk = db
      .prepare('SELECT * FROM chunks WHERE id = ?')
      .get(result.chunkId) as any;
    expect(chunk.event_time).toBe('2025-01-15T00:00:00Z');
    expect(chunk.temporal_label).toBe('mid-January 2025');
  });

  it('queues extraction for world type', async () => {
    const result = await retain(db, 'hello world', embedder, {
      memoryType: 'world',
    });
    expect(result.queued).toBe(true);
    const row = db
      .prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?')
      .get(result.chunkId);
    expect(row).toBeTruthy();
  });

  it('queues extraction for experience type', async () => {
    const result = await retain(db, 'I did a thing', embedder, {
      memoryType: 'experience',
    });
    expect(result.queued).toBe(true);
  });

  it('does not queue for observation type', async () => {
    const result = await retain(db, 'observed pattern', embedder, {
      memoryType: 'observation',
    });
    expect(result.queued).toBe(false);
    // better-sqlite3 .get() returns undefined (not null) when no row exists
    const row = db
      .prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?')
      .get(result.chunkId);
    expect(row).toBeUndefined();
  });

  it('does not queue for opinion type', async () => {
    const result = await retain(db, 'I believe this', embedder, {
      memoryType: 'opinion',
    });
    expect(result.queued).toBe(false);
  });

  it('skips queueing when skipExtraction is true', async () => {
    const result = await retain(db, 'hello', embedder, {
      memoryType: 'world',
      skipExtraction: true,
    });
    expect(result.queued).toBe(false);
    const row = db
      .prepare('SELECT * FROM extraction_queue WHERE chunk_id = ?')
      .get(result.chunkId);
    expect(row).toBeUndefined();
  });

  it('generates unique chunk IDs across multiple retains', async () => {
    const r1 = await retain(db, 'first', embedder);
    const r2 = await retain(db, 'second', embedder);
    expect(r1.chunkId).not.toBe(r2.chunkId);
  });
});

describe('retain() — concurrent deduplication', () => {
  async function retainConcurrently(
    firstText: string,
    secondText: string,
    dedupMode: 'normalized' | 'exact' | 'none',
  ) {
    const path = tmpDbPath();
    const bootstrap = new Database(path);
    loadSchema(bootstrap);
    bootstrap.close();
    const first = new Database(path);
    const second = new Database(path);
    let started = 0;
    let release!: () => void;
    const bothEmbeddingsStarted = new Promise<void>((resolve) => {
      release = resolve;
    });
    const racingEmbedder = {
      dimensions: 8,
      async embed(): Promise<Float32Array> {
        started++;
        if (started === 2) release();
        await bothEmbeddingsStarted;
        return new Float32Array(8);
      },
    };
    try {
      return await Promise.all([
        retain(first, firstText, racingEmbedder, { dedupMode }),
        retain(second, secondText, racingEmbedder, { dedupMode }),
      ]);
    } finally {
      first.close();
      second.close();
      cleanupDb(path);
    }
  }

  it.each([
    ['normalized', 'Tom prefers Terraform', ' tom  prefers terraform '],
    ['exact', 'Tom prefers Terraform', 'Tom prefers Terraform'],
  ] as const)(
    'serializes concurrent %s duplicate writes',
    async (mode, a, b) => {
      const [first, second] = await retainConcurrently(a, b, mode);
      expect(first.chunkId).toBe(second.chunkId);
      expect([first.deduplicated, second.deduplicated]).toContain(true);
    },
  );

  it('keeps concurrent duplicates when dedupMode is none', async () => {
    const [first, second] = await retainConcurrently(
      'Tom prefers Terraform',
      'Tom prefers Terraform',
      'none',
    );
    expect(first.chunkId).not.toBe(second.chunkId);
    expect(first.deduplicated).toBeUndefined();
    expect(second.deduplicated).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

describe('retain() — deduplication', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('exact dedup: retaining identical text returns existing chunk', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder);
    const r2 = await retain(db, 'Tom prefers Terraform', embedder);

    expect(r2.chunkId).toBe(r1.chunkId);
    expect(r2.deduplicated).toBe(true);

    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as {
      n: number;
    };
    expect(count.n).toBe(1);
  });

  it('exact dedup: reinforces trust score with the higher value', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder, {
      trustScore: 0.5,
    });
    await retain(db, 'Tom prefers Terraform', embedder, { trustScore: 0.9 });

    const chunk = db
      .prepare('SELECT trust_score FROM chunks WHERE id = ?')
      .get(r1.chunkId) as any;
    expect(chunk.trust_score).toBe(0.9);
  });

  it('exact dedup: promotes provenance when a higher-trust source retains the same text', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder, {
      sourceType: 'external_doc',
      trustScore: 0.2,
      source: 'doc:v1',
      context: 'imported-doc',
    });

    const r2 = await retain(db, 'Tom prefers Terraform', embedder, {
      sourceType: 'user_stated',
      trustScore: 0.95,
      source: 'conversation:pi',
      context: 'interactive',
    });

    expect(r2.chunkId).toBe(r1.chunkId);
    const chunk = db
      .prepare(
        'SELECT source_type, trust_score, source, context FROM chunks WHERE id = ?',
      )
      .get(r1.chunkId) as any;
    expect(chunk.source_type).toBe('user_stated');
    expect(chunk.trust_score).toBe(0.95);
    expect(chunk.source).toBe('conversation:pi');
    expect(chunk.context).toBe('interactive');
  });

  it('dedupMode none: always creates new chunks', async () => {
    await retain(db, 'Tom prefers Terraform', embedder, { dedupMode: 'none' });
    const r2 = await retain(db, 'Tom prefers Terraform', embedder, {
      dedupMode: 'none',
    });

    expect(r2.deduplicated).toBeUndefined();
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as {
      n: number;
    };
    expect(count.n).toBe(2);
  });

  it('dedupMode normalized: deduplicates despite case/whitespace differences', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder, {
      dedupMode: 'normalized',
    });
    const r2 = await retain(db, '  tom prefers terraform  ', embedder, {
      dedupMode: 'normalized',
    });

    expect(r2.chunkId).toBe(r1.chunkId);
    expect(r2.deduplicated).toBe(true);
  });

  it('dedupMode normalized: deduplicates despite internal whitespace differences', async () => {
    const r1 = await retain(db, 'Tom  prefers   Terraform', embedder, {
      dedupMode: 'normalized',
    });
    const r2 = await retain(db, 'tom prefers terraform', embedder, {
      dedupMode: 'normalized',
    });

    expect(r2.chunkId).toBe(r1.chunkId);
    expect(r2.deduplicated).toBe(true);
  });

  it('exact dedup: different text creates new chunk', async () => {
    const r1 = await retain(db, 'Tom prefers Terraform', embedder);
    const r2 = await retain(db, 'Tom prefers Pulumi', embedder);

    expect(r1.chunkId).not.toBe(r2.chunkId);
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as {
      n: number;
    };
    expect(count.n).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// retain() — supersedes option (atomic-supersede seam; see Engram.supersede())
// ---------------------------------------------------------------------------

describe('retain() — supersedes option', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('marks the old chunk superseded in the same write as a fresh insert', async () => {
    const old = await retain(db, 'Tom prefers Terraform', embedder);
    const next = await retain(db, 'Tom switched to Pulumi', embedder, {
      supersedes: old.chunkId,
    });

    expect(next.chunkId).not.toBe(old.chunkId);
    const oldRow = db
      .prepare('SELECT is_active, superseded_by FROM chunks WHERE id = ?')
      .get(old.chunkId) as any;
    expect(oldRow.is_active).toBe(0);
    expect(oldRow.superseded_by).toBe(next.chunkId);
  });

  it('marks the old chunk superseded when the new text dedups to an existing chunk', async () => {
    const existing = await retain(db, 'Tom uses Kubernetes', embedder);
    const old = await retain(db, 'Tom uses Docker Swarm', embedder);

    // New text is a normalized duplicate of `existing` — retain() resolves
    // to existing.chunkId via the dedup path, not a fresh insert.
    const result = await retain(db, '  tom uses kubernetes  ', embedder, {
      supersedes: old.chunkId,
    });

    expect(result.chunkId).toBe(existing.chunkId);
    expect(result.deduplicated).toBe(true);

    const oldRow = db
      .prepare('SELECT is_active, superseded_by FROM chunks WHERE id = ?')
      .get(old.chunkId) as any;
    expect(oldRow.is_active).toBe(0);
    expect(oldRow.superseded_by).toBe(existing.chunkId);
  });

  it('self-supersede: dedup resolving back onto the chunk being superseded does not deactivate it', async () => {
    const chunk = await retain(db, 'Tom uses Kubernetes', embedder);

    // supersedes points at the SAME chunk the dedup hit resolves to.
    const result = await retain(db, 'tom uses kubernetes', embedder, {
      supersedes: chunk.chunkId,
    });

    expect(result.chunkId).toBe(chunk.chunkId);
    const row = db
      .prepare('SELECT is_active, superseded_by FROM chunks WHERE id = ?')
      .get(chunk.chunkId) as any;
    expect(row.is_active).toBe(1);
    expect(row.superseded_by).toBeNull();
  });

  it('supersedes pointing at a non-existent chunk id does not throw', async () => {
    const result = await retain(db, 'brand new fact', embedder, {
      supersedes: 'chk-doesnotexist',
    });
    expect(result.chunkId).toMatch(/^chk-/);
    expect(result.deduplicated).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// retainBatch()
// ---------------------------------------------------------------------------

describe('retainBatch()', () => {
  it('stores all items and returns all results', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();

    const results = await retainBatch(
      db,
      [
        { text: 'first', options: { memoryType: 'world' } },
        { text: 'second', options: { memoryType: 'world' } },
        { text: 'third', options: { memoryType: 'world' } },
      ],
      embedder,
    );

    expect(results).toHaveLength(3);
    const count = db.prepare('SELECT count(*) as n FROM chunks').get() as {
      n: number;
    };
    expect(count.n).toBe(3);
    db.close();
  });

  it('reports progress per concurrency chunk', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const progress: Array<[number, number]> = [];

    // With concurrency=1, reports per-item like the old sequential behavior
    await retainBatch(
      db,
      [{ text: 'a' }, { text: 'b' }, { text: 'c' }],
      embedder,
      (current, total) => progress.push([current, total]),
      1, // concurrency=1 forces sequential
    );

    expect(progress).toEqual([
      [1, 3],
      [2, 3],
      [3, 3],
    ]);
    db.close();
  });

  it('queues world and experience items but not observation/opinion', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();

    await retainBatch(
      db,
      [
        { text: 'world fact', options: { memoryType: 'world' } },
        { text: 'my experience', options: { memoryType: 'experience' } },
        { text: 'observed pattern', options: { memoryType: 'observation' } },
      ],
      embedder,
    );

    const count = db
      .prepare('SELECT count(*) as n FROM extraction_queue')
      .get() as { n: number };
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
    const generator = new MockGenerator(EXTRACTION_RESPONSE);

    await retain(db, 'Alice prefers Rust for systems programming', embedder, {
      memoryType: 'world',
    });

    const result = await processExtractionQueue(db, generator);
    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);

    // Entities created (by Tier 1 CPU + Tier 2 LLM)
    const entities = db.prepare('SELECT * FROM entities').all() as any[];
    expect(entities.some((e) => e.canonical_name === 'alice')).toBe(true);
    expect(entities.some((e) => e.canonical_name === 'rust')).toBe(true);

    // Relation created
    const relations = db.prepare('SELECT * FROM relations').all() as any[];
    expect(relations.length).toBeGreaterThan(0);

    // chunk_entities linked
    const links = db.prepare('SELECT * FROM chunk_entities').all() as any[];
    expect(links.length).toBeGreaterThan(0);

    // Queue item marked complete
    const queued = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(queued.status).toBe('completed');

    db.close();
  });

  it('clamps off-list entity_type and skips malformed entities instead of failing extraction', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    // Real-world LLM drift: an entity_type outside the schema CHECK list
    // ("company") and an entity missing canonical_name. Pre-fix, either one
    // aborted the whole chunk's extraction transaction and burned all 3
    // retries on the same output.
    const driftResponse = JSON.stringify({
      entities: [
        {
          name: 'Acme',
          canonical_name: 'acme',
          entity_type: 'company',
          aliases: [],
        },
        { name: 'Ghost', entity_type: 'person', aliases: [] },
        {
          name: 'Bob',
          canonical_name: 'bob',
          entity_type: 'Person',
          aliases: [],
        },
      ],
      relations: [],
    });
    const generator = new MockGenerator(driftResponse);

    // All-lowercase text so the Tier 1 CPU extractor creates no entities —
    // otherwise it pre-creates 'acme'/'bob' as 'concept' and the LLM upsert's
    // ON CONFLICT path (which doesn't touch entity_type) would mask the clamp.
    await retain(db, 'the vendor hired someone new', embedder, {
      memoryType: 'world',
    });
    const result = await processExtractionQueue(db, generator);

    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);

    const acme = db
      .prepare(`SELECT entity_type FROM entities WHERE canonical_name = 'acme'`)
      .get() as any;
    expect(acme.entity_type).toBe('concept'); // off-list → neutral fallback

    const bob = db
      .prepare(`SELECT entity_type FROM entities WHERE canonical_name = 'bob'`)
      .get() as any;
    expect(bob.entity_type).toBe('person'); // case-normalized, kept

    // Malformed entity skipped, not fatal
    const ghost = db
      .prepare(`SELECT count(*) as n FROM entities WHERE name = 'Ghost'`)
      .get() as any;
    expect(ghost.n).toBe(0);

    const queued = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(queued.status).toBe('completed');
    db.close();
  });

  it('sets status to pending (not failed) after first extraction failure', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const failGen: import('../src/generation.js').GenerationProvider = {
      name: 'mock/fail',
      generate: async () => {
        throw new Error('generation error');
      },
    };

    await retain(db, 'some text', embedder, { memoryType: 'world' });
    const result = await processExtractionQueue(db, failGen);

    expect(result.failed).toBe(1);
    const queued = db
      .prepare('SELECT status, attempts FROM extraction_queue LIMIT 1')
      .get() as any;
    // First failure: status back to 'pending' so it can be retried
    expect(queued.status).toBe('pending');
    expect(queued.attempts).toBe(1);
    db.close();
  });

  it('sets status to failed after 3 extraction failures', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const failGen: import('../src/generation.js').GenerationProvider = {
      name: 'mock/fail',
      generate: async () => {
        throw new Error('generation error');
      },
    };

    await retain(db, 'retry exhaustion text', embedder, {
      memoryType: 'world',
    });

    // Run 3 times — clear backoff window between each to simulate time passing
    await processExtractionQueue(db, failGen);
    db.prepare(`UPDATE extraction_queue SET next_retry_after = NULL`).run();
    await processExtractionQueue(db, failGen);
    db.prepare(`UPDATE extraction_queue SET next_retry_after = NULL`).run();
    const thirdResult = await processExtractionQueue(db, failGen);

    expect(thirdResult.failed).toBe(1);
    const queued = db
      .prepare('SELECT status, attempts FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(queued.attempts).toBe(3);
    expect(queued.status).toBe('failed');
    db.close();
  });

  it('respects backoff window — does not retry immediately after failure', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const failGen: import('../src/generation.js').GenerationProvider = {
      name: 'mock/fail',
      generate: async () => {
        throw new Error('generation error');
      },
    };

    await retain(db, 'backoff test text', embedder, { memoryType: 'world' });

    // First call fails and sets a backoff window
    await processExtractionQueue(db, failGen);
    const queued = db
      .prepare('SELECT status, next_retry_after FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(queued.status).toBe('pending');
    expect(queued.next_retry_after).toBeTruthy();

    // Second call immediately after — item should be skipped due to backoff
    const secondResult = await processExtractionQueue(db, failGen);
    expect(secondResult.processed).toBe(0);
    expect(secondResult.failed).toBe(0);

    db.close();
  });

  it('returns zero when queue is empty', async () => {
    const db = createTestDb();
    const generator = new MockGenerator();
    const result = await processExtractionQueue(db, generator);
    expect(result.processed).toBe(0);
    expect(result.failed).toBe(0);
    db.close();
  });

  it('skips inactive chunks even if they remain in the extraction queue', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const generator = new MockGenerator(EXTRACTION_RESPONSE);

    const retained = await retain(db, 'Alice prefers Rust', embedder, {
      memoryType: 'world',
    });
    db.prepare(`UPDATE chunks SET is_active = FALSE WHERE id = ?`).run(
      retained.chunkId,
    );

    const result = await processExtractionQueue(db, generator);
    expect(result.processed).toBe(0);
    expect(result.failed).toBe(0);

    // Tier 1 CPU extraction created entities during retain(), so count > 0
    // But Tier 2 should NOT have processed the inactive chunk
    db.close();
  });

  it('getQueueStats returns correct counts per status', async () => {
    const db = createTestDb();
    const embedder = new MockEmbedder();
    const generator = new MockGenerator(EXTRACTION_RESPONSE);

    // One completed
    await retain(db, 'fact one', embedder, { memoryType: 'world' });
    await processExtractionQueue(db, generator);

    // One pending
    await retain(db, 'fact two', embedder, { memoryType: 'world' });

    const stats = getQueueStats(db);
    expect(stats.completed).toBe(1);
    expect(stats.pending).toBe(1);
    expect(stats.failed).toBe(0);
    expect(stats.oldest_pending).toBeTruthy();

    db.close();
  });

  it('getQueueStats returns zeros on empty queue', () => {
    const db = createTestDb();
    const stats = getQueueStats(db);
    expect(stats.pending).toBe(0);
    expect(stats.processing).toBe(0);
    expect(stats.completed).toBe(0);
    expect(stats.failed).toBe(0);
    expect(stats.oldest_pending).toBeNull();
    db.close();
  });
});

// ---------------------------------------------------------------------------
// recoverStalledExtractions
// ---------------------------------------------------------------------------

describe('recoverStalledExtractions()', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('recovers items stuck in processing longer than stallTimeout', async () => {
    await retain(db, 'stalled extraction test', embedder, {
      memoryType: 'world',
    });
    // Simulate a crash: manually set status to processing with old last_attempt
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', last_attempt = datetime('now', '-10 minutes')`,
    ).run();

    const recovered = recoverStalledExtractions(db, 5);
    expect(recovered).toBe(1);

    const row = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(row.status).toBe('pending');
  });

  it('does not recover recently-started processing items', async () => {
    await retain(db, 'recent processing test', embedder, {
      memoryType: 'world',
    });
    // Set status to processing with recent last_attempt
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', last_attempt = datetime('now', '-1 minutes')`,
    ).run();

    const recovered = recoverStalledExtractions(db, 5);
    expect(recovered).toBe(0);

    const row = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(row.status).toBe('processing');
  });

  it('moves stalled items with attempts >= 3 to failed instead of pending', async () => {
    await retain(db, 'max attempts test', embedder, {
      memoryType: 'world',
    });
    // Simulate crash on 3rd attempt
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', attempts = 3, last_attempt = datetime('now', '-10 minutes')`,
    ).run();

    const recovered = recoverStalledExtractions(db, 5);
    expect(recovered).toBe(1);

    const row = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(row.status).toBe('failed');
  });

  it('recovers processing items with NULL last_attempt (retryable branch)', async () => {
    await retain(db, 'null last_attempt test', embedder, {
      memoryType: 'world',
    });
    // A row can enter 'processing' without last_attempt via older code paths
    // or manual SQL. NULL < datetime(...) is NULL, so pre-fix this row
    // matched neither recovery branch and was stuck forever.
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', last_attempt = NULL`,
    ).run();

    const recovered = recoverStalledExtractions(db, 5);
    expect(recovered).toBe(1);

    const row = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(row.status).toBe('pending');
  });

  it('moves NULL-last_attempt items with attempts >= 3 to failed', async () => {
    await retain(db, 'null last_attempt max attempts test', embedder, {
      memoryType: 'world',
    });
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', attempts = 3, last_attempt = NULL`,
    ).run();

    const recovered = recoverStalledExtractions(db, 5);
    expect(recovered).toBe(1);

    const row = db
      .prepare('SELECT status FROM extraction_queue LIMIT 1')
      .get() as any;
    expect(row.status).toBe('failed');
  });

  it('claim uses compare-and-set — skips already-claimed items', async () => {
    await retain(db, 'cas test item', embedder, { memoryType: 'world' });

    // Manually set to processing (simulating another worker claimed it)
    db.prepare(
      `UPDATE extraction_queue SET status = 'processing', last_attempt = CURRENT_TIMESTAMP`,
    ).run();

    const generator = new MockGenerator(EXTRACTION_RESPONSE);
    const result = await processExtractionQueue(db, generator);
    // Should skip the item because CAS fails (status is not 'pending')
    expect(result.processed).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// requeueFailedExtractions + failed_reasons breakdown
// ---------------------------------------------------------------------------

describe('requeueFailedExtractions()', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  /** Seed one retained chunk whose queue item is in terminal 'failed' state. */
  async function seedFailed(text: string, error: string): Promise<string> {
    const r = await retain(db, text, embedder, { memoryType: 'world' });
    db.prepare(
      `UPDATE extraction_queue
       SET status = 'failed', attempts = 3, error = ?,
           next_retry_after = datetime('now', '+1 hour')
       WHERE chunk_id = ?`,
    ).run(error, r.chunkId);
    return r.chunkId;
  }

  it('resets failed items to pending with fresh attempts and no backoff', async () => {
    await seedFailed('outage casualty one', 'fetch failed');
    await seedFailed('outage casualty two', 'fetch failed');

    const requeued = requeueFailedExtractions(db);
    expect(requeued).toBe(2);

    const rows = db
      .prepare(
        `SELECT status, attempts, next_retry_after FROM extraction_queue`,
      )
      .all() as any[];
    for (const row of rows) {
      expect(row.status).toBe('pending');
      expect(row.attempts).toBe(0);
      expect(row.next_retry_after).toBeNull();
    }
  });

  it('errorLike filter requeues only the matching failure class', async () => {
    await seedFailed('network failure', 'fetch failed: ECONNREFUSED');
    await seedFailed('missing model', 'model "nope" not found');

    const requeued = requeueFailedExtractions(db, {
      errorLike: 'fetch failed',
    });
    expect(requeued).toBe(1);

    const stillFailed = db
      .prepare(`SELECT error FROM extraction_queue WHERE status = 'failed'`)
      .all() as any[];
    expect(stillFailed.length).toBe(1);
    expect(stillFailed[0].error).toContain('not found');
  });

  it('skips failed items whose chunk has been deactivated', async () => {
    const chunkId = await seedFailed('forgotten fact', 'fetch failed');
    db.prepare(`UPDATE chunks SET is_active = FALSE WHERE id = ?`).run(chunkId);

    const requeued = requeueFailedExtractions(db);
    expect(requeued).toBe(0);
  });

  it('requeued items get processed on the next queue drain', async () => {
    await seedFailed('second chance fact', 'fetch failed');
    requeueFailedExtractions(db);

    const generator = new MockGenerator(EXTRACTION_RESPONSE);
    const result = await processExtractionQueue(db, generator);
    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);
  });

  it('getQueueStats reports a failed_reasons breakdown, most common first', async () => {
    await seedFailed('one', 'fetch failed');
    await seedFailed('two', 'fetch failed');
    await seedFailed('three', 'timeout after 30s');

    const stats = getQueueStats(db);
    expect(stats.failed).toBe(3);
    expect(stats.failed_reasons).toEqual([
      { error: 'fetch failed', count: 2 },
      { error: 'timeout after 30s', count: 1 },
    ]);
  });

  it('getQueueStats failed_reasons is empty when nothing has failed', () => {
    const stats = getQueueStats(db);
    expect(stats.failed_reasons).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// retainBatch dedup
// ---------------------------------------------------------------------------

describe('retainBatch() intra-batch dedup', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('deduplicates identical items within a batch', async () => {
    const items = [
      { text: 'Tom uses Terraform' },
      { text: 'Tom uses Terraform' },
      { text: 'Tom uses Terraform' },
    ];
    const results = await retainBatch(db, items, embedder);

    // All should resolve to the same chunkId
    expect(results[0].chunkId).toBe(results[1].chunkId);
    expect(results[0].chunkId).toBe(results[2].chunkId);

    // Only one chunk should exist in the DB
    const count = db.prepare(`SELECT COUNT(*) as cnt FROM chunks`).get() as any;
    expect(count.cnt).toBe(1);
  });

  it('deduplicates case/whitespace-normalized duplicates', async () => {
    const items = [
      { text: 'Tom uses Terraform' },
      { text: 'tom  uses  terraform' },
    ];
    const results = await retainBatch(db, items, embedder);

    expect(results[1].deduplicated).toBe(true);
    expect(results[0].chunkId).toBe(results[1].chunkId);
  });
});

// ---------------------------------------------------------------------------
// chunkText()
// ---------------------------------------------------------------------------

describe('chunkText()', () => {
  it('returns single chunk for short text', () => {
    const result = chunkText('Short text here.', { maxChunkChars: 1000 });
    expect(result).toEqual(['Short text here.']);
  });

  it('returns empty array for empty text', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   ')).toEqual([]);
  });

  it('splits long text at paragraph boundaries', () => {
    const text =
      'Paragraph one about Terraform.\n\nParagraph two about Kubernetes.\n\nParagraph three about SQLite.';
    const chunks = chunkText(text, { maxChunkChars: 50, overlapChars: 0 });
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0]).toContain('Terraform');
  });

  it('no chunk exceeds maxChunkChars', () => {
    const paragraphs = Array.from(
      { length: 20 },
      (_, i) =>
        `This is paragraph ${i} with enough words to be meaningful content.`,
    ).join('\n\n');
    const chunks = chunkText(paragraphs, {
      maxChunkChars: 200,
      overlapChars: 0,
    });
    const withinLimit = chunks.filter((c) => c.length <= 200);
    expect(withinLimit.length).toBe(chunks.length);
  });

  it('applies overlap from previous chunk', () => {
    const text =
      'AAAA first section content.\n\nBBBB second section content.\n\nCCCC third section content.';
    const chunks = chunkText(text, { maxChunkChars: 50, overlapChars: 10 });
    if (chunks.length >= 2) {
      const tail = chunks[0].slice(-10).trim();
      expect(chunks[1].startsWith(tail)).toBe(true);
    }
  });
});
