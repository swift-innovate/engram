import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { recall } from '../src/recall.js';
import { createTestDb, MockEmbedder } from './helpers.js';

// ---------------------------------------------------------------------------
// Keyword search
// ---------------------------------------------------------------------------

describe('recall() — keyword search', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('finds chunks matching query terms', async () => {
    await retain(db, 'Tom uses Terraform for infrastructure', embedder, { trustScore: 0.9 });
    await retain(db, 'Mira is an AI assistant', embedder, { trustScore: 0.9 });

    const result = await recall(db, 'Terraform infrastructure', embedder, {
      strategies: ['keyword'],
      topK: 5,
    });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].text).toContain('Terraform');
    expect(result.strategiesUsed).toContain('keyword');
  });

  it('returns empty for an unmatched query', async () => {
    await retain(db, 'completely unrelated content', embedder);
    const result = await recall(db, 'zzzyyyxxx', embedder, { strategies: ['keyword'] });
    expect(result.results).toHaveLength(0);
  });

  it('respects minTrust filter', async () => {
    await retain(db, 'low trust fact about widgets', embedder, { trustScore: 0.2 });
    await retain(db, 'high trust fact about widgets', embedder, { trustScore: 0.9 });

    const result = await recall(db, 'widgets', embedder, {
      strategies: ['keyword'],
      minTrust: 0.5,
    });

    expect(result.results.every(r => r.trustScore >= 0.5)).toBe(true);
    expect(result.results.some(r => r.text.includes('high trust'))).toBe(true);
    expect(result.results.some(r => r.text.includes('low trust'))).toBe(false);
  });

  it('respects memoryType filter', async () => {
    await retain(db, 'world fact about coding', embedder, { memoryType: 'world' });
    await retain(db, 'experience with coding today', embedder, { memoryType: 'experience' });

    const result = await recall(db, 'coding', embedder, {
      strategies: ['keyword'],
      memoryTypes: ['world'],
    });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results.every(r => r.memoryType === 'world')).toBe(true);
  });

  it('result includes source and trustScore fields', async () => {
    await retain(db, 'tagged content for recall', embedder, {
      trustScore: 0.75,
      source: 'conversation:xyz',
      sourceType: 'user_stated',
    });

    const result = await recall(db, 'tagged content', embedder, { strategies: ['keyword'] });
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].trustScore).toBe(0.75);
    expect(result.results[0].source).toBe('conversation:xyz');
    expect(result.results[0].sourceType).toBe('user_stated');
  });
});

// ---------------------------------------------------------------------------
// Temporal search
// ---------------------------------------------------------------------------

describe('recall() — temporal search', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('returns nothing without a date filter', async () => {
    await retain(db, 'some fact', embedder);
    const result = await recall(db, 'fact', embedder, { strategies: ['temporal'] });
    expect(result.results).toHaveLength(0);
  });

  it('filters by after date using event_time', async () => {
    await retain(db, 'old fact', embedder, { eventTime: '2020-06-01T00:00:00Z' });
    await retain(db, 'new fact', embedder, { eventTime: '2025-06-01T00:00:00Z' });

    const result = await recall(db, 'fact', embedder, {
      strategies: ['temporal'],
      after: '2024-01-01T00:00:00Z',
    });

    const texts = result.results.map(r => r.text);
    expect(texts).toContain('new fact');
    expect(texts).not.toContain('old fact');
  });

  it('filters by before date using event_time', async () => {
    await retain(db, 'old fact', embedder, { eventTime: '2020-06-01T00:00:00Z' });
    await retain(db, 'new fact', embedder, { eventTime: '2025-06-01T00:00:00Z' });

    const result = await recall(db, 'fact', embedder, {
      strategies: ['temporal'],
      before: '2023-01-01T00:00:00Z',
    });

    const texts = result.results.map(r => r.text);
    expect(texts).toContain('old fact');
    expect(texts).not.toContain('new fact');
  });
});

// ---------------------------------------------------------------------------
// Graph search (entity wiring done directly to avoid needing Ollama)
// ---------------------------------------------------------------------------

describe('recall() — graph search', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(async () => {
    db = createTestDb();

    // Add a chunk and manually link entities — simulates post-extraction state
    const { chunkId } = await retain(db, 'Alice uses Rust for embedded systems', embedder, {
      trustScore: 0.9,
    });

    db.prepare(`INSERT INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-alice', 'Alice', 'alice', 'person')`).run();
    db.prepare(`INSERT INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-rust', 'Rust', 'rust', 'technology')`).run();
    db.prepare(`INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-alice')`).run(chunkId);
    db.prepare(`INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-rust')`).run(chunkId);
    db.prepare(`INSERT INTO relations (id, source_entity_id, target_entity_id, relation_type)
      VALUES ('rel-1', 'ent-alice', 'ent-rust', 'prefers')`).run();
  });

  afterEach(() => db.close());

  it('finds chunks directly connected to a matched entity', async () => {
    const result = await recall(db, 'alice', embedder, { strategies: ['graph'], topK: 5 });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].text).toContain('Alice');
    expect(result.strategiesUsed).toContain('graph');
  });

  it('returns empty for a query that matches no entities', async () => {
    const result = await recall(db, 'xyzunknownentity', embedder, { strategies: ['graph'] });
    expect(result.results).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Multi-strategy fusion + scoring
// ---------------------------------------------------------------------------

describe('recall() — multi-strategy fusion', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('reports strategiesUsed', async () => {
    await retain(db, 'keyword searchable content', embedder);
    const result = await recall(db, 'keyword searchable', embedder, { strategies: ['keyword'] });
    expect(result.strategiesUsed).toContain('keyword');
  });

  it('reports totalCandidates', async () => {
    await retain(db, 'content for recall', embedder);
    const result = await recall(db, 'content recall', embedder, { strategies: ['keyword'] });
    expect(result.totalCandidates).toBeGreaterThan(0);
  });

  it('promotes chunks found by multiple strategies', async () => {
    // This chunk will match keyword AND graph
    const { chunkId } = await retain(db, 'Tom prefers Terraform', embedder, { trustScore: 0.9 });

    db.prepare(`INSERT INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-tom', 'Tom', 'tom', 'person')`).run();
    db.prepare(`INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-tom')`).run(chunkId);

    const result = await recall(db, 'Tom Terraform', embedder, {
      strategies: ['keyword', 'graph'],
      topK: 5,
    });

    const hit = result.results.find(r => r.text.includes('Tom'));
    expect(hit).toBeDefined();
    expect(hit!.strategies).toContain('keyword');
    expect(hit!.strategies).toContain('graph');
  });

  it('high-trust chunks outscore low-trust chunks with same text relevance', async () => {
    // Two chunks that match equally on keyword — trust should break the tie
    await retain(db, 'widget documentation reference', embedder, { trustScore: 0.9 });
    await retain(db, 'widget documentation reference', embedder, { trustScore: 0.1 });

    const result = await recall(db, 'widget documentation', embedder, {
      strategies: ['keyword'],
      topK: 10,
    });

    // The high-trust chunk should rank first
    expect(result.results[0].trustScore).toBe(0.9);
  });
});

// ---------------------------------------------------------------------------
// Opinions and observations in recall response
// ---------------------------------------------------------------------------

describe('recall() — opinions and observations', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('returns opinions with confidence >= 0.5', async () => {
    db.prepare(`INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-1', 'Tom prefers SQLite over Postgres', 0.8, 'architecture')`).run();
    db.prepare(`INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-2', 'Tom might like MongoDB', 0.3, 'architecture')`).run();

    await retain(db, 'something about databases', embedder);
    const result = await recall(db, 'databases', embedder, { strategies: ['keyword'] });

    // Only confidence >= 0.5 returned
    expect(result.opinions.every(o => o.confidence >= 0.5)).toBe(true);
    expect(result.opinions.some(o => o.belief.includes('SQLite'))).toBe(true);
  });

  it('omits opinions when includeOpinions is false', async () => {
    db.prepare(`INSERT INTO opinions (id, belief, confidence) VALUES ('op-1', 'A belief', 0.9)`).run();
    await retain(db, 'test content', embedder);

    const result = await recall(db, 'content', embedder, {
      strategies: ['keyword'],
      includeOpinions: false,
    });
    expect(result.opinions).toHaveLength(0);
  });

  it('returns observations', async () => {
    db.prepare(`INSERT INTO observations (id, summary, domain, topic)
      VALUES ('obs-1', 'Tom consistently chooses minimal tooling', 'architecture', 'tooling')`).run();

    await retain(db, 'something about Tom', embedder);
    const result = await recall(db, 'Tom', embedder, { strategies: ['keyword'] });

    expect(result.observations.some(o => o.summary.includes('Tom'))).toBe(true);
  });

  it('omits observations when includeObservations is false', async () => {
    db.prepare(`INSERT INTO observations (id, summary) VALUES ('obs-1', 'An observation')`).run();

    const result = await recall(db, 'anything', embedder, {
      strategies: ['keyword'],
      includeObservations: false,
    });
    expect(result.observations).toHaveLength(0);
  });
});
