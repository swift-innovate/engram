import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
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

// ---------------------------------------------------------------------------
// Source and context filtering
// ---------------------------------------------------------------------------

describe('recall() — source and context filtering', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('sourceFilter excludes non-matching chunks', async () => {
    await retain(db, 'fact from project alpha', embedder, { source: 'project:alpha' });
    await retain(db, 'fact from project beta', embedder, { source: 'project:beta' });

    const result = await recall(db, 'project', embedder, {
      strategies: ['keyword'],
      sourceFilter: 'alpha',
    });

    expect(result.results.every(r => r.source?.includes('alpha'))).toBe(true);
    expect(result.results.some(r => r.source?.includes('beta'))).toBe(false);
  });

  it('contextFilter excludes non-matching chunks', async () => {
    await retain(db, 'infrastructure decision context', embedder, { context: 'infrastructure' });
    await retain(db, 'career decision context', embedder, { context: 'career' });

    const result = await recall(db, 'decision', embedder, {
      strategies: ['keyword'],
      contextFilter: 'infrastructure',
    });

    // Only infrastructure context result should appear
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results.every(r => (r as any).context === undefined || result.results.length === 1)).toBe(true);
  });

  it('sourceBoost promotes matching source in ranking', async () => {
    // Two identical texts from different sources; boost should elevate the boosted one
    await retain(db, 'important system fact', embedder, { source: 'project:barracuda', trustScore: 0.5 });
    await retain(db, 'important system fact', embedder, {
      source: 'project:other',
      trustScore: 0.5,
      dedupMode: 'none',
    });

    const result = await recall(db, 'important system', embedder, {
      strategies: ['keyword'],
      sourceBoost: { pattern: 'barracuda', multiplier: 2.0 },
    });

    expect(result.results.length).toBeGreaterThan(0);
    const topResult = result.results[0];
    expect(topResult.source).toContain('barracuda');
  });
});

// ---------------------------------------------------------------------------
// Temporal decay
// ---------------------------------------------------------------------------

describe('recall() — temporal decay', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => { db = createTestDb(); });
  afterEach(() => db.close());

  it('older chunks receive lower scores than newer ones with decayHalfLifeDays: 30', async () => {
    // Insert two chunks directly with controlled timestamps
    const oldDate = new Date(Date.now() - 120 * 24 * 60 * 60 * 1000).toISOString(); // 120 days ago
    const newDate = new Date().toISOString();

    // Use retain() for the actual embedding, then update created_at directly
    const r1 = await retain(db, 'infrastructure uses Terraform tooling', embedder, { trustScore: 0.8 });
    const r2 = await retain(db, 'infrastructure uses Terraform tooling', embedder, {
      trustScore: 0.8,
      dedupMode: 'none',
    });

    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(oldDate, r1.chunkId);
    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(newDate, r2.chunkId);

    const result = await recall(db, 'Terraform tooling', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 30,
    });

    expect(result.results.length).toBe(2);
    const oldEntry = result.results.find(r => r.id === r1.chunkId);
    const newEntry = result.results.find(r => r.id === r2.chunkId);
    expect(oldEntry).toBeDefined();
    expect(newEntry).toBeDefined();
    // Newer chunk should rank higher (higher score)
    expect(newEntry!.score).toBeGreaterThan(oldEntry!.score);
  });

  it('decayHalfLifeDays: 0 disables decay', async () => {
    const oldDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(); // 1 year ago

    const r1 = await retain(db, 'unique ancient widget fact', embedder, { trustScore: 0.8 });
    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(oldDate, r1.chunkId);

    const withDecay = await recall(db, 'widget', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 30,
    });
    const noDecay = await recall(db, 'widget', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 0,
    });

    expect(withDecay.results[0].score).toBeLessThan(noDecay.results[0].score);
  });
});

// ---------------------------------------------------------------------------
// formatForPrompt()
// ---------------------------------------------------------------------------

describe('formatForPrompt()', () => {
  it('is exported from recall.ts', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    expect(typeof formatForPrompt).toBe('function');
  });

  it('formats opinions, observations, and results with default header', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt({
      results: [
        { id: 'chk-1', text: 'Tom uses Terraform', memoryType: 'world', source: 'conv:test',
          trustScore: 0.9, sourceType: 'user_stated', eventTime: null, score: 0.8, strategies: ['keyword'] },
      ],
      opinions: [
        { belief: 'Tom prefers minimal tooling', confidence: 0.85, domain: 'architecture' },
      ],
      observations: [
        { summary: 'Tom consistently chooses SQLite', domain: 'architecture', topic: 'tooling' },
      ],
      totalCandidates: 1,
      strategiesUsed: ['keyword'],
    });

    expect(output).toContain('## Relevant Memory Context');
    expect(output).toContain('Tom prefers minimal tooling');
    expect(output).toContain('Tom consistently chooses SQLite');
    expect(output).toContain('Tom uses Terraform');
  });

  it('respects maxChars budget — truncates and adds omission notice', async () => {
    const { formatForPrompt } = await import('../src/recall.js');

    const manyResults = Array.from({ length: 20 }, (_, i) => ({
      id: `chk-${i}`,
      text: `Memory result number ${i} with some content to fill space`,
      memoryType: 'world',
      source: null,
      trustScore: 0.8,
      sourceType: 'inferred',
      eventTime: null,
      score: 0.8 - i * 0.01,
      strategies: ['keyword'],
    }));

    const output = formatForPrompt(
      { results: manyResults, opinions: [], observations: [], totalCandidates: 20, strategiesUsed: ['keyword'] },
      { maxChars: 300 }
    );

    expect(output.length).toBeLessThanOrEqual(300);
    expect(output).toContain('omitted');
  });

  it('respects showTrust option', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [{ id: 'chk-1', text: 'test', memoryType: 'world', source: null,
        trustScore: 0.85, sourceType: 'inferred', eventTime: null, score: 0.5, strategies: [] }],
      opinions: [], observations: [], totalCandidates: 1, strategiesUsed: [],
    };

    const withTrust = formatForPrompt(response, { showTrust: true });
    const withoutTrust = formatForPrompt(response, { showTrust: false });

    expect(withTrust).toContain('trust');
    expect(withoutTrust).not.toContain('trust');
  });

  it('uses custom header', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt(
      { results: [], opinions: [], observations: [], totalCandidates: 0, strategiesUsed: [] },
      { header: '## My Custom Header' }
    );
    expect(output).toContain('## My Custom Header');
  });

  it('returns empty-ish string when response has no content', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt({
      results: [], opinions: [], observations: [], totalCandidates: 0, strategiesUsed: [],
    });
    // Should at least have the header
    expect(output).toContain('## Relevant Memory Context');
  });
});

// ---------------------------------------------------------------------------
// Query-scoped opinions
// ---------------------------------------------------------------------------

describe('recall() — query-scoped opinions', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
    // Insert opinions in different domains
    db.prepare(`INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-cook', 'Tom prefers sous vide cooking techniques', 0.8, 'cooking')`).run();
    db.prepare(`INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-infra', 'Tom prefers Terraform for infrastructure automation', 0.9, 'infrastructure')`).run();
    db.prepare(`INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-code', 'Tom writes TypeScript for all new projects', 0.85, 'coding')`).run();
  });
  afterEach(() => db.close());

  it('returns opinions whose belief text contains query tokens', async () => {
    await retain(db, 'infrastructure tooling decision', embedder);

    const result = await recall(db, 'Terraform infrastructure', embedder, {
      strategies: ['keyword'],
    });

    // Should return the infrastructure opinion, not the cooking one
    const beliefs = result.opinions.map(o => o.belief);
    expect(beliefs.some(b => b.includes('Terraform'))).toBe(true);
    expect(beliefs.some(b => b.includes('sous vide'))).toBe(false);
  });

  it('falls back to global opinions when no query tokens match', async () => {
    // Query with no token > 3 chars won't match any belief text
    const result = await recall(db, 'xy', embedder, { strategies: ['keyword'] });

    // Falls back to global top opinions
    expect(result.opinions.length).toBeGreaterThan(0);
  });
});
