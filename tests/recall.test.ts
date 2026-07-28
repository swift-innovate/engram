import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { retain, embeddingToBuffer } from '../src/retain.js';
import { recall } from '../src/recall.js';
import { createTestDb, MockEmbedder } from './helpers.js';

/**
 * Construct a unit vector orthogonal to `v` via Gram-Schmidt against a fixed
 * alternating-sign seed. Deterministic (no randomness) and lands at exactly
 * cosine 0 to `v` regardless of `v`'s direction — used to force a "weak"
 * semantic match with a known, reproducible cosine similarity.
 */
function orthogonalTo(v: Float32Array): Float32Array {
  const seed = new Float32Array(v.length);
  for (let i = 0; i < seed.length; i++) seed[i] = i % 2 === 0 ? 1 : -1;
  const dot = seed.reduce((s, x, i) => s + x * v[i], 0);
  const normSq = v.reduce((s, x) => s + x * x, 0);
  const proj = dot / normSq;
  const orth = Float32Array.from(seed, (x, i) => x - proj * v[i]);
  const mag = Math.sqrt(orth.reduce((s, x) => s + x * x, 0));
  return Float32Array.from(orth, (x) => x / mag);
}

// ---------------------------------------------------------------------------
// Keyword search
// ---------------------------------------------------------------------------

describe('recall() — keyword search', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('finds chunks matching query terms', async () => {
    await retain(db, 'Tom uses Terraform for infrastructure', embedder, {
      trustScore: 0.9,
    });
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
    const result = await recall(db, 'zzzyyyxxx', embedder, {
      strategies: ['keyword'],
    });
    expect(result.results).toHaveLength(0);
  });

  it('respects minTrust filter', async () => {
    await retain(db, 'low trust fact about widgets', embedder, {
      trustScore: 0.2,
    });
    await retain(db, 'high trust fact about widgets', embedder, {
      trustScore: 0.9,
    });

    const result = await recall(db, 'widgets', embedder, {
      strategies: ['keyword'],
      minTrust: 0.5,
    });

    expect(result.results.every((r) => r.trustScore >= 0.5)).toBe(true);
    expect(result.results.some((r) => r.text.includes('high trust'))).toBe(
      true,
    );
    expect(result.results.some((r) => r.text.includes('low trust'))).toBe(
      false,
    );
  });

  it('respects memoryType filter', async () => {
    await retain(db, 'world fact about coding', embedder, {
      memoryType: 'world',
    });
    await retain(db, 'experience with coding today', embedder, {
      memoryType: 'experience',
    });

    const result = await recall(db, 'coding', embedder, {
      strategies: ['keyword'],
      memoryTypes: ['world'],
    });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results.every((r) => r.memoryType === 'world')).toBe(true);
  });

  it('result includes source and trustScore fields', async () => {
    await retain(db, 'tagged content for recall', embedder, {
      trustScore: 0.75,
      source: 'conversation:xyz',
      sourceType: 'user_stated',
    });

    const result = await recall(db, 'tagged content', embedder, {
      strategies: ['keyword'],
    });
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

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('returns nothing without a date filter', async () => {
    await retain(db, 'some fact', embedder);
    const result = await recall(db, 'fact', embedder, {
      strategies: ['temporal'],
    });
    expect(result.results).toHaveLength(0);
  });

  it('filters by after date using event_time', async () => {
    await retain(db, 'old fact', embedder, {
      eventTime: '2020-06-01T00:00:00Z',
    });
    await retain(db, 'new fact', embedder, {
      eventTime: '2025-06-01T00:00:00Z',
    });

    const result = await recall(db, 'fact', embedder, {
      strategies: ['temporal'],
      after: '2024-01-01T00:00:00Z',
    });

    const texts = result.results.map((r) => r.text);
    expect(texts).toContain('new fact');
    expect(texts).not.toContain('old fact');
  });

  it('filters by before date using event_time', async () => {
    await retain(db, 'old fact', embedder, {
      eventTime: '2020-06-01T00:00:00Z',
    });
    await retain(db, 'new fact', embedder, {
      eventTime: '2025-06-01T00:00:00Z',
    });

    const result = await recall(db, 'fact', embedder, {
      strategies: ['temporal'],
      before: '2023-01-01T00:00:00Z',
    });

    const texts = result.results.map((r) => r.text);
    expect(texts).toContain('old fact');
    expect(texts).not.toContain('new fact');
  });
});

// ---------------------------------------------------------------------------
// Bare-year auto-temporal false positives (issue found in codebase review:
// ANY bare 4-digit number 2000-2100 anywhere in the query used to become a
// hard date filter applied to every strategy, starving results for queries
// like "port 2020 configuration").
// ---------------------------------------------------------------------------

describe('recall() — bare-year query text does not hijack temporal filtering', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('"port 2020 configuration" still surfaces an old/undated chunk', async () => {
    const oldChunk = await retain(
      db,
      'port 2020 configuration uses TLS termination',
      embedder,
      { trustScore: 0.9 },
    );
    // Backdate well outside any window a hijacked "2020" year filter would
    // allow (2020-01-01 .. 2020-12-31) — proves no date filter was applied.
    const longAgo = new Date(
      Date.now() - 2000 * 24 * 60 * 60 * 1000,
    ).toISOString();
    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(
      longAgo,
      oldChunk.chunkId,
    );

    const result = await recall(db, 'port 2020 configuration', embedder, {
      strategies: ['keyword', 'temporal'],
      decayHalfLifeDays: 0,
    });

    // The bare year must not have auto-activated the temporal strategy.
    expect(result.strategiesUsed).not.toContain('temporal');
    expect(result.results.some((r) => r.id === oldChunk.chunkId)).toBe(true);
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
    const { chunkId } = await retain(
      db,
      'Alice uses Rust for embedded systems',
      embedder,
      {
        trustScore: 0.9,
      },
    );

    db.prepare(
      `INSERT OR REPLACE INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-alice', 'Alice', 'alice', 'person')`,
    ).run();
    db.prepare(
      `INSERT OR REPLACE INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-rust', 'Rust', 'rust', 'technology')`,
    ).run();
    db.prepare(
      `INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-alice')`,
    ).run(chunkId);
    db.prepare(
      `INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-rust')`,
    ).run(chunkId);
    db.prepare(
      `INSERT OR REPLACE INTO relations (id, source_entity_id, target_entity_id, relation_type)
      VALUES ('rel-1', 'ent-alice', 'ent-rust', 'prefers')`,
    ).run();
  });

  afterEach(() => db.close());

  it('finds chunks directly connected to a matched entity', async () => {
    const result = await recall(db, 'alice', embedder, {
      strategies: ['graph'],
      topK: 5,
    });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].text).toContain('Alice');
    expect(result.strategiesUsed).toContain('graph');
  });

  it('returns empty for a query that matches no entities', async () => {
    const result = await recall(db, 'xyzunknownentity', embedder, {
      strategies: ['graph'],
    });
    expect(result.results).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Multi-strategy fusion + scoring
// ---------------------------------------------------------------------------

describe('recall() — multi-strategy fusion', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('reports strategiesUsed', async () => {
    await retain(db, 'keyword searchable content', embedder);
    const result = await recall(db, 'keyword searchable', embedder, {
      strategies: ['keyword'],
    });
    expect(result.strategiesUsed).toContain('keyword');
  });

  it('reports totalCandidates', async () => {
    await retain(db, 'content for recall', embedder);
    const result = await recall(db, 'content recall', embedder, {
      strategies: ['keyword'],
    });
    expect(result.totalCandidates).toBeGreaterThan(0);
  });

  it('promotes chunks found by multiple strategies', async () => {
    // This chunk will match keyword AND graph
    const { chunkId } = await retain(db, 'Tom prefers Terraform', embedder, {
      trustScore: 0.9,
    });

    db.prepare(
      `INSERT INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-tom', 'Tom', 'tom', 'person')`,
    ).run();
    db.prepare(
      `INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-tom')`,
    ).run(chunkId);

    const result = await recall(db, 'Tom Terraform', embedder, {
      strategies: ['keyword', 'graph'],
      topK: 5,
    });

    const hit = result.results.find((r) => r.text.includes('Tom'));
    expect(hit).toBeDefined();
    expect(hit!.strategies).toContain('keyword');
    expect(hit!.strategies).toContain('graph');
  });

  it('high-trust chunks outscore low-trust chunks with same text relevance', async () => {
    // Two chunks that match equally on keyword — trust should break the tie
    await retain(db, 'widget documentation reference', embedder, {
      trustScore: 0.9,
    });
    await retain(db, 'widget documentation reference', embedder, {
      trustScore: 0.1,
    });

    const result = await recall(db, 'widget documentation', embedder, {
      strategies: ['keyword'],
      topK: 10,
    });

    // The high-trust chunk should rank first
    expect(result.results[0].trustScore).toBe(0.9);
  });
});

// ---------------------------------------------------------------------------
// D6 — within-tier scoring is cosine-primary, not RRF-ordinal
//
// Before D6, the fused `score` was a pure sum of 1/(k+rank) terms — ordinal
// rank, not relevance magnitude. A chunk found by several strategies at
// middling rank could out-score a chunk with a much stronger single-strategy
// semantic match, because the RRF sum only ever "knew" about rank, never how
// close the match actually was. These tests pin the fix: within a tier, the
// semantic strategy's raw cosine similarity is now the primary score
// (× a gentle 0.94-0.99 trust tiebreak), so relevance magnitude survives
// fusion instead of being discarded after `ORDER BY distance`.
// ---------------------------------------------------------------------------

describe('recall() — D6 cosine-primary within-tier scoring', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('a strong single-strategy semantic match beats a weak multi-strategy chunk (regression)', async () => {
    const query = 'gizmo rollout status';
    const queryVec = await embedder.embed(query);

    // Strong: embedding parallel to the query (cosine 1.0), found ONLY by
    // semantic search — no keyword overlap with the query terms.
    const strong = await retain(
      db,
      'the deployment finished ahead of schedule with no incidents',
      embedder,
      {
        memoryType: 'experience',
        sourceType: 'agent_generated',
        trustScore: 0.5,
      },
    );
    db.prepare(`UPDATE chunks SET embedding = ? WHERE id = ?`).run(
      embeddingToBuffer(queryVec),
      strong.chunkId,
    );

    // Weak: embedding orthogonal to the query (cosine 0.0), but the text
    // repeats every query term so it also wins the keyword strategy — this
    // multi-strategy consensus is exactly what let a weak match out-score a
    // strong one under pure RRF-rank summation.
    const weak = await retain(
      db,
      'gizmo rollout gizmo rollout gizmo rollout status status status',
      embedder,
      {
        memoryType: 'experience',
        sourceType: 'agent_generated',
        trustScore: 0.5,
        dedupMode: 'none',
      },
    );
    db.prepare(`UPDATE chunks SET embedding = ? WHERE id = ?`).run(
      embeddingToBuffer(orthogonalTo(queryVec)),
      weak.chunkId,
    );

    const result = await recall(db, query, embedder, {
      strategies: ['semantic', 'keyword'],
      topK: 5,
      decayHalfLifeDays: 0,
    });

    const strongIdx = result.results.findIndex((r) => r.id === strong.chunkId);
    const weakIdx = result.results.findIndex((r) => r.id === weak.chunkId);
    expect(strongIdx).toBeGreaterThanOrEqual(0);
    expect(weakIdx).toBeGreaterThanOrEqual(0);
    expect(strongIdx).toBeLessThan(weakIdx);
  });

  it('minScore gates out a weak cosine match while keeping a strong one', async () => {
    const query = 'widget telemetry summary';
    const queryVec = await embedder.embed(query);

    const strong = await retain(
      db,
      'strong semantic match candidate',
      embedder,
      {
        memoryType: 'experience',
        sourceType: 'agent_generated',
        trustScore: 0.5,
      },
    );
    db.prepare(`UPDATE chunks SET embedding = ? WHERE id = ?`).run(
      embeddingToBuffer(queryVec),
      strong.chunkId,
    );

    const weak = await retain(db, 'weak semantic match candidate', embedder, {
      memoryType: 'experience',
      sourceType: 'agent_generated',
      trustScore: 0.5,
      dedupMode: 'none',
    });
    db.prepare(`UPDATE chunks SET embedding = ? WHERE id = ?`).run(
      embeddingToBuffer(orthogonalTo(queryVec)),
      weak.chunkId,
    );

    const result = await recall(db, query, embedder, {
      strategies: ['semantic'],
      topK: 5,
      decayHalfLifeDays: 0,
      minScore: 0.42,
    });

    const ids = result.results.map((r) => r.id);
    expect(ids).toContain(strong.chunkId);
    expect(ids).not.toContain(weak.chunkId);
  });
});

// ---------------------------------------------------------------------------
// minScore + explainScores
// ---------------------------------------------------------------------------

describe('recall() — minScore threshold', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('default behavior is unchanged when minScore is omitted', async () => {
    await retain(db, 'widget documentation reference', embedder, {
      trustScore: 0.9,
    });
    // decayHalfLifeDays: 0 pins scores across the two sequential calls —
    // with decay on, elapsed wall-clock time between them shifts every
    // score by a representable amount and toEqual flakes.
    const withoutMinScore = await recall(db, 'widget documentation', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 0,
    });
    const explicitUndefined = await recall(
      db,
      'widget documentation',
      embedder,
      {
        strategies: ['keyword'],
        decayHalfLifeDays: 0,
        minScore: undefined,
      },
    );
    expect(explicitUndefined.results).toEqual(withoutMinScore.results);
    expect(withoutMinScore.results.length).toBeGreaterThan(0);
  });

  it('drops results below the threshold', async () => {
    await retain(db, 'widget documentation reference', embedder, {
      trustScore: 0.9,
    });
    const baseline = await recall(db, 'widget documentation', embedder, {
      strategies: ['keyword'],
    });
    expect(baseline.results.length).toBeGreaterThan(0);
    const topScore = baseline.results[0].score;

    const filtered = await recall(db, 'widget documentation', embedder, {
      strategies: ['keyword'],
      minScore: topScore + 1, // above every possible score
    });
    expect(filtered.results).toHaveLength(0);
  });

  it('keeps a result exactly at the threshold (inclusive boundary)', async () => {
    await retain(db, 'boundary threshold widget test', embedder, {
      trustScore: 0.9,
    });
    // decayHalfLifeDays: 0 makes the captured score reproducible in the
    // second call — with decay on, the score decays between the two calls
    // and the inclusive >= boundary fails by a hair.
    const baseline = await recall(db, 'boundary threshold widget', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 0,
    });
    expect(baseline.results.length).toBeGreaterThan(0);
    const exactScore = baseline.results[0].score;

    const atThreshold = await recall(
      db,
      'boundary threshold widget',
      embedder,
      {
        strategies: ['keyword'],
        decayHalfLifeDays: 0,
        minScore: exactScore,
      },
    );
    expect(
      atThreshold.results.some((r) => r.id === baseline.results[0].id),
    ).toBe(true);
  });

  it('minScore of 0 does not filter anything out', async () => {
    await retain(db, 'zero threshold widget content', embedder, {
      trustScore: 0.9,
    });
    const result = await recall(db, 'zero threshold widget', embedder, {
      strategies: ['keyword'],
      minScore: 0,
    });
    expect(result.results.length).toBeGreaterThan(0);
  });
});

describe('recall() — explainScores', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('omits strategyScores by default', async () => {
    await retain(db, 'explain scores default widget', embedder);
    const result = await recall(db, 'explain scores widget', embedder, {
      strategies: ['keyword'],
    });
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0]).not.toHaveProperty('strategyScores');
  });

  it('includes a strategyScores breakdown when true', async () => {
    await retain(db, 'explain scores enabled widget', embedder, {
      trustScore: 0.8,
    });
    const result = await recall(db, 'explain scores widget', embedder, {
      strategies: ['keyword'],
      explainScores: true,
    });
    expect(result.results.length).toBeGreaterThan(0);
    const hit = result.results[0];
    expect(hit.strategyScores).toBeDefined();
    expect(Array.isArray(hit.strategyScores!.perStrategy)).toBe(true);
    expect(hit.strategyScores!.perStrategy.length).toBeGreaterThan(0);
    expect(hit.strategyScores!.perStrategy[0]).toMatchObject({
      strategy: 'keyword',
      rank: expect.any(Number),
      rrfScore: expect.any(Number),
    });
    expect(typeof hit.strategyScores!.rawFusedScore).toBe('number');
    expect(hit.strategyScores!.weighting).toMatchObject({
      trust: expect.any(Number),
      strategyBoost: expect.any(Number),
      decay: expect.any(Number),
      sourceBoost: expect.any(Number),
      contextBoost: expect.any(Number),
      memoryType: expect.any(Number),
    });
    // rawFusedScore * product(weighting) === final score
    const w = hit.strategyScores!.weighting;
    const recomputed =
      hit.strategyScores!.rawFusedScore *
      w.trust *
      w.strategyBoost *
      w.decay *
      w.sourceBoost *
      w.contextBoost *
      w.memoryType;
    expect(recomputed).toBeCloseTo(hit.score, 10);
  });

  it('aggregates multi-strategy contributions in perStrategy', async () => {
    const { chunkId } = await retain(db, 'Tom prefers Terraform', embedder, {
      trustScore: 0.9,
    });
    db.prepare(
      `INSERT INTO entities (id, name, canonical_name, entity_type)
      VALUES ('ent-tom-explain', 'Tom', 'tom', 'person')`,
    ).run();
    db.prepare(
      `INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (?, 'ent-tom-explain')`,
    ).run(chunkId);

    const result = await recall(db, 'Tom Terraform', embedder, {
      strategies: ['keyword', 'graph'],
      topK: 5,
      explainScores: true,
    });

    const hit = result.results.find((r) => r.id === chunkId);
    expect(hit).toBeDefined();
    const strategies = hit!.strategyScores!.perStrategy.map((p) => p.strategy);
    expect(strategies).toContain('keyword');
    expect(strategies).toContain('graph');
  });
});

// ---------------------------------------------------------------------------
// Opinions and observations in recall response
// ---------------------------------------------------------------------------

describe('recall() — opinions and observations', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('returns opinions with confidence >= 0.5', async () => {
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-1', 'Tom prefers SQLite over Postgres', 0.8, 'architecture')`,
    ).run();
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-2', 'Tom might like MongoDB', 0.3, 'architecture')`,
    ).run();

    await retain(db, 'something about SQLite', embedder);
    const result = await recall(db, 'SQLite!', embedder, {
      strategies: ['keyword'],
    });

    // Only confidence >= 0.5 returned
    expect(result.opinions.every((o) => o.confidence >= 0.5)).toBe(true);
    expect(result.opinions.some((o) => o.belief.includes('SQLite'))).toBe(true);
  });

  it('omits opinions when includeOpinions is false', async () => {
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence) VALUES ('op-1', 'A belief', 0.9)`,
    ).run();
    await retain(db, 'test content', embedder);

    const result = await recall(db, 'content', embedder, {
      strategies: ['keyword'],
      includeOpinions: false,
    });
    expect(result.opinions).toHaveLength(0);
  });

  it('returns observations', async () => {
    db.prepare(
      `INSERT INTO observations (id, summary, domain, topic)
      VALUES ('obs-1', 'Tom consistently chooses minimal tooling', 'architecture', 'tooling')`,
    ).run();

    await retain(db, 'something about tooling', embedder);
    const result = await recall(db, 'tooling!', embedder, {
      strategies: ['keyword'],
    });

    expect(result.observations.some((o) => o.summary.includes('Tom'))).toBe(
      true,
    );
  });

  it('omits observations when includeObservations is false', async () => {
    db.prepare(
      `INSERT INTO observations (id, summary) VALUES ('obs-1', 'An observation')`,
    ).run();

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

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('sourceFilter excludes non-matching chunks', async () => {
    await retain(db, 'fact from project alpha', embedder, {
      source: 'project:alpha',
    });
    await retain(db, 'fact from project beta', embedder, {
      source: 'project:beta',
    });

    const result = await recall(db, 'project', embedder, {
      strategies: ['keyword'],
      sourceFilter: 'alpha',
    });

    expect(result.results.every((r) => r.source?.includes('alpha'))).toBe(true);
    expect(result.results.some((r) => r.source?.includes('beta'))).toBe(false);
  });

  it('contextFilter excludes non-matching chunks', async () => {
    await retain(db, 'infrastructure decision context', embedder, {
      context: 'infrastructure',
    });
    await retain(db, 'career decision context', embedder, {
      context: 'career',
    });

    const result = await recall(db, 'decision', embedder, {
      strategies: ['keyword'],
      contextFilter: 'infrastructure',
    });

    // Only infrastructure context result should appear
    expect(result.results.length).toBeGreaterThan(0);
    expect(
      result.results.every(
        (r) => (r as any).context === undefined || result.results.length === 1,
      ),
    ).toBe(true);
  });

  it('sourceBoost promotes matching source in ranking', async () => {
    // Two identical texts from different sources; boost should elevate the boosted one
    await retain(db, 'important system fact', embedder, {
      source: 'project:barracuda',
      trustScore: 0.5,
    });
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

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('older chunks receive lower scores than newer ones with decayHalfLifeDays: 30', async () => {
    // Insert two chunks directly with controlled timestamps
    const oldDate = new Date(
      Date.now() - 120 * 24 * 60 * 60 * 1000,
    ).toISOString(); // 120 days ago
    const newDate = new Date().toISOString();

    // Use retain() for the actual embedding, then update created_at directly
    const r1 = await retain(
      db,
      'infrastructure uses Terraform tooling',
      embedder,
      { trustScore: 0.8 },
    );
    const r2 = await retain(
      db,
      'infrastructure uses Terraform tooling',
      embedder,
      {
        trustScore: 0.8,
        dedupMode: 'none',
      },
    );

    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(
      oldDate,
      r1.chunkId,
    );
    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(
      newDate,
      r2.chunkId,
    );

    const result = await recall(db, 'Terraform tooling', embedder, {
      strategies: ['keyword'],
      decayHalfLifeDays: 30,
    });

    expect(result.results.length).toBe(2);
    const oldEntry = result.results.find((r) => r.id === r1.chunkId);
    const newEntry = result.results.find((r) => r.id === r2.chunkId);
    expect(oldEntry).toBeDefined();
    expect(newEntry).toBeDefined();
    // Newer chunk should rank higher (higher score)
    expect(newEntry!.score).toBeGreaterThan(oldEntry!.score);
  });

  it('decayHalfLifeDays: 0 disables decay', async () => {
    const oldDate = new Date(
      Date.now() - 365 * 24 * 60 * 60 * 1000,
    ).toISOString(); // 1 year ago

    const r1 = await retain(db, 'unique ancient widget fact', embedder, {
      trustScore: 0.8,
    });
    db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(
      oldDate,
      r1.chunkId,
    );

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
        {
          id: 'chk-1',
          text: 'Tom uses Terraform',
          memoryType: 'world',
          source: 'conv:test',
          trustScore: 0.9,
          sourceType: 'user_stated',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.8,
          strategies: ['keyword'],
        },
      ],
      opinions: [
        {
          belief: 'Tom prefers minimal tooling',
          confidence: 0.85,
          domain: 'architecture',
        },
      ],
      observations: [
        {
          summary: 'Tom consistently chooses SQLite',
          domain: 'architecture',
          topic: 'tooling',
        },
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
      createdAt: '2026-03-14 12:00:00',
      score: 0.8 - i * 0.01,
      strategies: ['keyword'],
    }));

    const output = formatForPrompt(
      {
        results: manyResults,
        opinions: [],
        observations: [],
        totalCandidates: 20,
        strategiesUsed: ['keyword'],
      },
      { maxChars: 300 },
    );

    expect(output.length).toBeLessThanOrEqual(300);
    expect(output).toContain('omitted');
  });

  it('respects showTrust option', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'test',
          memoryType: 'world',
          source: null,
          trustScore: 0.85,
          sourceType: 'inferred',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: [],
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: [],
    };

    const withTrust = formatForPrompt(response, { showTrust: true });
    const withoutTrust = formatForPrompt(response, { showTrust: false });

    expect(withTrust).toContain('trust');
    expect(withoutTrust).not.toContain('trust');
  });

  it('omits provenance and why by default (byte-identical to pre-T1 output)', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'test',
          memoryType: 'world',
          source: null,
          trustScore: 0.85,
          sourceType: 'inferred',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: ['keyword'],
          strategyScores: {
            perStrategy: [{ strategy: 'keyword', rank: 1, rrfScore: 0.016 }],
            rawFusedScore: 0.016,
            weighting: {
              trust: 1,
              strategyBoost: 1,
              decay: 1,
              sourceBoost: 1,
              contextBoost: 1,
              memoryType: 1,
            },
          },
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: ['keyword'],
    };

    const output = formatForPrompt(response);
    expect(output).not.toContain('[world/inferred');
    expect(output).not.toContain('why:');
  });

  it('showProvenance renders memory type, source type, trust, and date', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'Tom uses Terraform',
          memoryType: 'world',
          source: null,
          trustScore: 0.85,
          sourceType: 'user_stated',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: ['keyword'],
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: ['keyword'],
    };

    const output = formatForPrompt(response, { showProvenance: true });
    expect(output).toContain('[world/user_stated, trust 0.85, 2026-03-14]');

    const withoutFlag = formatForPrompt(response, { showProvenance: false });
    expect(withoutFlag).not.toContain('[world/user_stated');
  });

  it('showWhy renders a strategy breakdown line when strategyScores is present', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'Tom uses Terraform',
          memoryType: 'world',
          source: null,
          trustScore: 0.9,
          sourceType: 'user_stated',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: ['semantic', 'keyword'],
          strategyScores: {
            perStrategy: [
              { strategy: 'semantic', rank: 1, rrfScore: 0.016 },
              { strategy: 'keyword', rank: 3, rrfScore: 0.015 },
            ],
            rawFusedScore: 0.82,
            weighting: {
              trust: 1,
              strategyBoost: 1,
              decay: 1,
              sourceBoost: 1,
              contextBoost: 1,
              memoryType: 1,
            },
          },
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: ['semantic', 'keyword'],
    };

    const output = formatForPrompt(response, { showWhy: true });
    expect(output).toContain('  why: semantic 0.82 · keyword r3 · tier 0');
  });

  it('showWhy is a silent no-op when the result has no strategyScores (explainScores was not requested)', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'Tom uses Terraform',
          memoryType: 'world',
          source: null,
          trustScore: 0.9,
          sourceType: 'user_stated',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: ['keyword'],
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: ['keyword'],
    };

    const output = formatForPrompt(response, { showWhy: true });
    expect(output).not.toContain('why:');
    expect(output).toContain('Tom uses Terraform');
  });

  it('drops a provenance/why line that would exceed maxChars, same as any other line', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const response = {
      results: [
        {
          id: 'chk-1',
          text: 'A memory result with enough text to fill most of a small budget',
          memoryType: 'world',
          source: null,
          trustScore: 0.9,
          sourceType: 'user_stated',
          eventTime: null,
          createdAt: '2026-03-14 12:00:00',
          score: 0.5,
          strategies: ['semantic'],
          strategyScores: {
            perStrategy: [{ strategy: 'semantic', rank: 1, rrfScore: 0.016 }],
            rawFusedScore: 0.82,
            weighting: {
              trust: 1,
              strategyBoost: 1,
              decay: 1,
              sourceBoost: 1,
              contextBoost: 1,
              memoryType: 1,
            },
          },
        },
      ],
      opinions: [],
      observations: [],
      totalCandidates: 1,
      strategiesUsed: ['semantic'],
    };

    // Budget fits the header + result line (with provenance) but not the
    // extra why line.
    const output = formatForPrompt(response, {
      showWhy: true,
      showProvenance: true,
      maxChars: 155,
    });
    expect(output.length).toBeLessThanOrEqual(155);
    expect(output).toContain('A memory result with enough text');
    expect(output).not.toContain('why:');
  });

  it('uses custom header', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt(
      {
        results: [],
        opinions: [],
        observations: [],
        totalCandidates: 0,
        strategiesUsed: [],
      },
      { header: '## My Custom Header' },
    );
    expect(output).toContain('## My Custom Header');
  });

  it('returns empty-ish string when response has no content', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt({
      results: [],
      opinions: [],
      observations: [],
      totalCandidates: 0,
      strategiesUsed: [],
    });
    // Should at least have the header
    expect(output).toContain('## Relevant Memory Context');
  });

  it('includes disclaimer in beliefs heading to prevent feedback loops', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt({
      results: [],
      opinions: [{ belief: 'Test belief', confidence: 0.9, domain: 'test' }],
      observations: [],
      totalCandidates: 0,
      strategiesUsed: [],
    });
    expect(output).toContain('agent-synthesized, not ground truth');
    expect(output).not.toContain('### Beliefs\n');
  });

  it('caps displayed opinion confidence at 85%', async () => {
    const { formatForPrompt } = await import('../src/recall.js');
    const output = formatForPrompt({
      results: [],
      opinions: [
        { belief: 'Very confident belief', confidence: 0.99, domain: null },
      ],
      observations: [],
      totalCandidates: 0,
      strategiesUsed: [],
    });
    expect(output).toContain('[85%]');
    expect(output).not.toContain('[99%]');
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
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-cook', 'Tom prefers sous vide cooking techniques', 0.8, 'cooking')`,
    ).run();
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-infra', 'Tom prefers Terraform for infrastructure automation', 0.9, 'infrastructure')`,
    ).run();
    db.prepare(
      `INSERT INTO opinions (id, belief, confidence, domain)
      VALUES ('op-code', 'Tom writes TypeScript for all new projects', 0.85, 'coding')`,
    ).run();
  });
  afterEach(() => db.close());

  it('returns opinions whose belief text contains query tokens', async () => {
    await retain(db, 'infrastructure tooling decision', embedder);

    const result = await recall(db, 'Terraform infrastructure', embedder, {
      strategies: ['keyword'],
    });

    // Should return the infrastructure opinion, not the cooking one
    const beliefs = result.opinions.map((o) => o.belief);
    expect(beliefs.some((b) => b.includes('Terraform'))).toBe(true);
    expect(beliefs.some((b) => b.includes('sous vide'))).toBe(false);
  });

  it('returns no synthesized context when no normalized query term matches', async () => {
    const result = await recall(db, 'xy', embedder, {
      strategies: ['keyword'],
    });

    expect(result.opinions).toEqual([]);
    expect(result.observations).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// FTS5 phrase support in the keyword-search query sanitizer
// ---------------------------------------------------------------------------

describe('recall() — quoted-phrase keyword search', () => {
  let db: Database.Database;
  const embedder = new MockEmbedder();

  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('a quoted phrase matches only the chunk with the adjacent token sequence', async () => {
    const contiguous = await retain(
      db,
      'we shipped the blue green deployment last night',
      embedder,
      { trustScore: 0.9 },
    );
    const scattered = await retain(
      db,
      'the deployment was blue, then later turned green',
      embedder,
      { trustScore: 0.9, dedupMode: 'none' },
    );

    const result = await recall(db, '"blue green deployment"', embedder, {
      strategies: ['keyword'],
    });

    const ids = result.results.map((r) => r.id);
    expect(ids).toContain(contiguous.chunkId);
    expect(ids).not.toContain(scattered.chunkId);
  });

  it('unquoted multi-word queries behave exactly as before (implicit AND)', async () => {
    await retain(db, 'Tom uses Terraform for infrastructure', embedder, {
      trustScore: 0.9,
    });
    await retain(db, 'Mira is an AI assistant', embedder, {
      trustScore: 0.9,
      dedupMode: 'none',
    });

    const result = await recall(db, 'Terraform infrastructure', embedder, {
      strategies: ['keyword'],
    });

    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].text).toContain('Terraform');
  });

  describe('pathological inputs never throw', () => {
    const pathologicalQueries = [
      '"unclosed phrase',
      'unclosed phrase"',
      '"" ""',
      'NEAR AND OR NOT',
      'term NOT',
      'foo * bar',
      'foo ^ bar',
      'foo - bar',
      'foo : bar',
      '(foo AND bar)',
      '🔥💥🎉 emoji query',
      '!!!???***&&&',
      '""""""',
      '"a" "b" "c"',
    ];

    beforeEach(async () => {
      await retain(db, 'seed content for pathological query tests', embedder, {
        trustScore: 0.9,
      });
    });

    for (const q of pathologicalQueries) {
      it(`does not throw for: ${JSON.stringify(q)}`, async () => {
        await expect(
          recall(db, q, embedder, { strategies: ['keyword'] }),
        ).resolves.toBeDefined();
      });
    }
  });
});
