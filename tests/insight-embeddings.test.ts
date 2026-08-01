// =============================================================================
// Insight embeddings — semantic ranking of opinions/observations at recall.
//
// Before this, both were retrieved by `LOWER(text) LIKE '%term%'` over query
// terms >3 chars, ordered by confidence/recency. Ordinary query words ("what",
// "like", "does") match a large fraction of beliefs, so the attached insights
// were close to constant regardless of the question.
//
// These tests pin the three regimes the new selection has to satisfy:
// fully-embedded (semantic), un-embedded (lexical, unchanged), and the mixed
// state a backfill passes through. Synthesized context stays query-scoped in
// all three: returning nothing is a valid answer, never a reason to pad.
// =============================================================================

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { recall } from '../src/recall.js';
import {
  backfillInsightEmbeddings,
  DEFAULT_INSIGHT_BACKFILL_LIMIT,
} from '../src/reflect.js';
import { embeddingToBuffer, type EmbeddingProvider } from '../src/retain.js';
import { Engram } from '../src/engram.js';
import {
  createTestDb,
  loadSchema,
  tmpDbPath,
  cleanupDb,
  MockEmbedder,
  MockGenerator,
  REFLECT_RESPONSE,
} from './helpers.js';

const DIMS = 8;

/** Unit vector with 1.0 in slot `i` — cosine 1 to itself, 0 to any other slot. */
function axis(i: number): Float32Array {
  const v = new Float32Array(DIMS);
  v[i] = 1;
  return v;
}

const ON_TOPIC = axis(0);
const OFF_TOPIC = axis(1);
const THIRD = axis(2);

/**
 * Embedder with hand-assigned vectors, so a test states relevance directly
 * rather than hoping a hashing mock lands where it needs to. Unmapped text
 * embeds to a fourth axis — orthogonal to every vector a test names, i.e.
 * reliably "irrelevant".
 */
class StubEmbedder implements EmbeddingProvider {
  readonly dimensions = DIMS;
  embedCalls = 0;
  constructor(private readonly map: Record<string, Float32Array> = {}) {}
  async embed(text: string): Promise<Float32Array> {
    this.embedCalls++;
    return this.map[text] ?? axis(3);
  }
}

interface SeedOpinion {
  belief: string;
  confidence: number;
  embedding?: Float32Array;
}

function seedOpinions(db: Database.Database, rows: SeedOpinion[]): void {
  const stmt = db.prepare(
    `INSERT INTO opinions (id, belief, confidence, domain, embedding, is_active)
     VALUES (?, ?, ?, 'test', ?, TRUE)`,
  );
  rows.forEach((r, i) => {
    stmt.run(
      `op-${i}`,
      r.belief,
      r.confidence,
      r.embedding ? embeddingToBuffer(r.embedding) : null,
    );
  });
}

interface SeedObservation {
  summary: string;
  embedding?: Float32Array;
}

function seedObservations(
  db: Database.Database,
  rows: SeedObservation[],
): void {
  const stmt = db.prepare(
    `INSERT INTO observations (id, summary, domain, topic, embedding, is_active)
     VALUES (?, ?, 'test', 'test', ?, TRUE)`,
  );
  rows.forEach((r, i) => {
    stmt.run(
      `obs-${i}`,
      r.summary,
      r.embedding ? embeddingToBuffer(r.embedding) : null,
    );
  });
}

// ---------------------------------------------------------------------------
// Embedded store — semantic ranking
// ---------------------------------------------------------------------------

describe('insight selection — embedded store', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('ranks a relevant low-confidence belief above an irrelevant high-confidence one', async () => {
    seedOpinions(db, [
      {
        belief: 'off topic but certain',
        confidence: 0.99,
        embedding: OFF_TOPIC,
      },
      { belief: 'on topic', confidence: 0.55, embedding: ON_TOPIC },
    ]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder);

    expect(res.opinions.map((o) => o.belief)).toEqual(['on topic']);
  });

  it('returns NO opinions for an off-topic query rather than padding with top-confidence ones', async () => {
    seedOpinions(db, [
      {
        belief: 'certain but unrelated',
        confidence: 1.0,
        embedding: OFF_TOPIC,
      },
      { belief: 'also unrelated', confidence: 0.9, embedding: THIRD },
    ]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder);

    // The pre-fix behavior returned both, ordered by confidence.
    expect(res.opinions).toEqual([]);
  });

  it('honors insightMinScore — 0 keeps every embedded belief ranked but unfiltered', async () => {
    seedOpinions(db, [
      { belief: 'unrelated', confidence: 0.9, embedding: OFF_TOPIC },
      { belief: 'relevant', confidence: 0.6, embedding: ON_TOPIC },
    ]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder, { insightMinScore: 0 });

    expect(res.opinions.map((o) => o.belief)).toEqual([
      'relevant',
      'unrelated',
    ]);
  });

  it('still applies the 0.5 confidence floor to embedded opinions', async () => {
    seedOpinions(db, [
      {
        belief: 'relevant but weakly held',
        confidence: 0.3,
        embedding: ON_TOPIC,
      },
    ]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder);

    expect(res.opinions).toEqual([]);
  });

  it('caps attached opinions at 5', async () => {
    seedOpinions(
      db,
      Array.from({ length: 9 }, (_, i) => ({
        belief: `belief ${i}`,
        confidence: 0.6,
        embedding: ON_TOPIC,
      })),
    );
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder);

    expect(res.opinions).toHaveLength(5);
  });

  it('ranks observations semantically too', async () => {
    seedObservations(db, [
      { summary: 'unrelated observation', embedding: OFF_TOPIC },
      { summary: 'relevant observation', embedding: ON_TOPIC },
    ]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    const res = await recall(db, 'the query', embedder);

    expect(res.observations.map((o) => o.summary)).toEqual([
      'relevant observation',
    ]);
  });

  it('embeds the query once even though chunks and both insight kinds use it', async () => {
    seedOpinions(db, [
      { belief: 'a belief', confidence: 0.6, embedding: ON_TOPIC },
    ]);
    seedObservations(db, [{ summary: 'an observation', embedding: ON_TOPIC }]);
    const embedder = new StubEmbedder({ 'the query': ON_TOPIC });

    await recall(db, 'the query', embedder);

    expect(embedder.embedCalls).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Un-embedded store — legacy behavior must be byte-compatible
// ---------------------------------------------------------------------------

describe('insight selection — un-embedded (legacy) store', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('keeps the lexical match when no row is embedded', async () => {
    seedOpinions(db, [
      { belief: 'Tom prefers Terraform for infrastructure', confidence: 0.6 },
      { belief: 'something else entirely', confidence: 0.9 },
    ]);
    const embedder = new StubEmbedder();

    const res = await recall(db, 'Terraform choices', embedder);

    expect(res.opinions.map((o) => o.belief)).toEqual([
      'Tom prefers Terraform for infrastructure',
    ]);
  });

  it('returns nothing rather than padding when no term matches', async () => {
    seedOpinions(db, [
      { belief: 'alpha belief', confidence: 0.6 },
      { belief: 'beta belief', confidence: 0.95 },
    ]);
    const embedder = new StubEmbedder();

    // No term >3 chars matches either belief. Synthesized context is
    // query-scoped, so the confident beliefs must NOT be attached.
    const res = await recall(db, 'zzzz yyyy', embedder);

    expect(res.opinions).toEqual([]);
  });

  it('matches lexically despite punctuation in the query', async () => {
    seedOpinions(db, [
      { belief: 'Tom prefers Terraform for infrastructure', confidence: 0.6 },
    ]);
    const embedder = new StubEmbedder();

    const res = await recall(db, 'Terraform?', embedder);

    expect(res.opinions.map((o) => o.belief)).toEqual([
      'Tom prefers Terraform for infrastructure',
    ]);
  });

  it('does not embed the query when nothing needs a vector', async () => {
    seedOpinions(db, [{ belief: 'alpha belief', confidence: 0.6 }]);
    const embedder = new StubEmbedder();

    await recall(db, 'zzzz yyyy', embedder, { strategies: ['keyword'] });

    expect(embedder.embedCalls).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Mixed store — the transient state during backfill
// ---------------------------------------------------------------------------

describe('insight selection — partially embedded store', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  it('puts semantic hits first and fills remaining slots lexically', async () => {
    seedOpinions(db, [
      { belief: 'legacy row mentioning Terraform', confidence: 0.9 },
      { belief: 'embedded and relevant', confidence: 0.6, embedding: ON_TOPIC },
    ]);
    const embedder = new StubEmbedder({ 'Terraform please': ON_TOPIC });

    const res = await recall(db, 'Terraform please', embedder);

    expect(res.opinions.map((o) => o.belief)).toEqual([
      'embedded and relevant',
      'legacy row mentioning Terraform',
    ]);
  });

  it('returns nothing when neither path matches', async () => {
    seedOpinions(db, [
      { belief: 'legacy high confidence', confidence: 0.99 },
      {
        belief: 'embedded but unrelated',
        confidence: 0.6,
        embedding: OFF_TOPIC,
      },
    ]);
    const embedder = new StubEmbedder({ 'zzzz yyyy': ON_TOPIC });

    const res = await recall(db, 'zzzz yyyy', embedder);

    // Nothing is semantically relevant and nothing matches lexically, so the
    // honest answer is none — not "here are the most confident beliefs".
    expect(res.opinions).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Backfill
// ---------------------------------------------------------------------------

describe('backfillInsightEmbeddings()', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = createTestDb();
  });
  afterEach(() => db.close());

  const embeddedCount = (table: 'opinions' | 'observations'): number =>
    (
      db
        .prepare(
          `SELECT COUNT(*) AS n FROM ${table} WHERE embedding IS NOT NULL`,
        )
        .get() as { n: number }
    ).n;

  it('fills NULL embeddings across both tables and reports the remainder', async () => {
    seedOpinions(db, [
      { belief: 'first', confidence: 0.6 },
      { belief: 'second', confidence: 0.6 },
    ]);
    seedObservations(db, [{ summary: 'an observation' }]);

    const out = await backfillInsightEmbeddings(db, new StubEmbedder());

    expect(out).toEqual({
      opinionsEmbedded: 2,
      observationsEmbedded: 1,
      remaining: 0,
    });
    expect(embeddedCount('opinions')).toBe(2);
    expect(embeddedCount('observations')).toBe(1);
  });

  it('is bounded by limit and resumable across calls', async () => {
    seedOpinions(
      db,
      Array.from({ length: 5 }, (_, i) => ({
        belief: `belief ${i}`,
        confidence: 0.6,
      })),
    );

    const first = await backfillInsightEmbeddings(db, new StubEmbedder(), {
      limit: 2,
    });
    expect(first.opinionsEmbedded).toBe(2);
    expect(first.remaining).toBe(3);

    const second = await backfillInsightEmbeddings(db, new StubEmbedder(), {
      limit: 10,
    });
    expect(second.opinionsEmbedded).toBe(3);
    expect(second.remaining).toBe(0);
  });

  it('spends its budget on opinions first, then observations', async () => {
    seedOpinions(db, [
      { belief: 'first', confidence: 0.6 },
      { belief: 'second', confidence: 0.6 },
    ]);
    seedObservations(db, [{ summary: 'an observation' }]);

    const out = await backfillInsightEmbeddings(db, new StubEmbedder(), {
      limit: 2,
    });

    expect(out.opinionsEmbedded).toBe(2);
    expect(out.observationsEmbedded).toBe(0);
    expect(out.remaining).toBe(1);
  });

  it('never rewrites an embedding that already exists', async () => {
    seedOpinions(db, [
      { belief: 'already embedded', confidence: 0.6, embedding: ON_TOPIC },
    ]);
    const before = db
      .prepare(`SELECT embedding FROM opinions WHERE id = 'op-0'`)
      .get() as { embedding: Buffer };

    const out = await backfillInsightEmbeddings(db, new StubEmbedder());

    expect(out.opinionsEmbedded).toBe(0);
    const after = db
      .prepare(`SELECT embedding FROM opinions WHERE id = 'op-0'`)
      .get() as { embedding: Buffer };
    expect(after.embedding.equals(before.embedding)).toBe(true);
  });

  it('leaves a row NULL when its embedding call throws, and retries it later', async () => {
    seedOpinions(db, [{ belief: 'explodes', confidence: 0.6 }]);
    const failing: EmbeddingProvider = {
      dimensions: DIMS,
      embed: async () => {
        throw new Error('embedder offline');
      },
    };

    const out = await backfillInsightEmbeddings(db, failing);

    expect(out.opinionsEmbedded).toBe(0);
    expect(out.remaining).toBe(1);
    expect(embeddedCount('opinions')).toBe(0);

    // A later pass with a working embedder picks it up.
    const retry = await backfillInsightEmbeddings(db, new StubEmbedder());
    expect(retry.opinionsEmbedded).toBe(1);
    expect(retry.remaining).toBe(0);
  });

  it('skips inactive rows', async () => {
    seedOpinions(db, [{ belief: 'retired', confidence: 0.6 }]);
    db.prepare(`UPDATE opinions SET is_active = FALSE`).run();

    const out = await backfillInsightEmbeddings(db, new StubEmbedder());

    expect(out).toEqual({
      opinionsEmbedded: 0,
      observationsEmbedded: 0,
      remaining: 0,
    });
  });

  it('exposes a positive default limit', () => {
    expect(DEFAULT_INSIGHT_BACKFILL_LIMIT).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Migration + reflect integration (real files, via the Engram class)
// ---------------------------------------------------------------------------

describe('insight embeddings — migration and reflect', () => {
  let dbPath: string;
  let engram: Engram | undefined;

  afterEach(() => {
    engram?.close();
    engram = undefined;
    cleanupDb(dbPath);
  });

  it('adds the embedding columns to an .engram that predates them', async () => {
    dbPath = tmpDbPath();

    // Reproduce a genuinely pre-migration file: current schema minus the two
    // new columns, with a row already in each table.
    const raw = new Database(dbPath);
    loadSchema(raw);
    raw
      .prepare(
        `INSERT INTO opinions (id, belief, confidence, domain) VALUES ('op-legacy', 'a legacy belief', 0.6, 'test')`,
      )
      .run();
    raw
      .prepare(
        `INSERT INTO observations (id, summary, domain, topic) VALUES ('obs-legacy', 'a legacy observation', 'test', 'test')`,
      )
      .run();
    raw.exec('ALTER TABLE opinions DROP COLUMN embedding');
    raw.exec('ALTER TABLE observations DROP COLUMN embedding');
    raw.close();

    engram = await Engram.open(dbPath, { embedder: new MockEmbedder() });

    const check = new Database(dbPath, { readonly: true });
    const opCols = check.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    const obsCols = check.pragma('table_info(observations)') as Array<{
      name: string;
    }>;
    // Legacy rows survive the migration with a NULL vector — they keep the
    // keyword path until a backfill reaches them.
    const legacy = check
      .prepare(`SELECT embedding FROM opinions WHERE id = 'op-legacy'`)
      .get() as { embedding: Buffer | null };
    check.close();

    expect(opCols.some((c) => c.name === 'embedding')).toBe(true);
    expect(obsCols.some((c) => c.name === 'embedding')).toBe(true);
    expect(legacy.embedding).toBeNull();
  });

  it('stores embeddings for opinions and observations a reflect cycle forms', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, {
      embedder: new MockEmbedder(),
      generator: new MockGenerator(REFLECT_RESPONSE),
    });

    for (let i = 0; i < 5; i++) {
      await engram.retain(`Alice prefers Rust — fact ${i}`, {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      });
    }
    // Gates/counter-evidence off: this test is about the embedding write, and
    // the default gates need multi-day, multi-source evidence to form at all.
    const result = await engram.reflect({
      counterEvidence: false,
      opinionGates: false,
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.observationsCreated).toBe(1);

    const check = new Database(dbPath, { readonly: true });
    const op = check.prepare(`SELECT embedding FROM opinions`).get() as {
      embedding: Buffer | null;
    };
    const obs = check.prepare(`SELECT embedding FROM observations`).get() as {
      embedding: Buffer | null;
    };
    check.close();

    expect(op.embedding).toBeInstanceOf(Buffer);
    expect(obs.embedding).toBeInstanceOf(Buffer);
  });

  it('backfills legacy insights during a reflect cycle', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, {
      embedder: new MockEmbedder(),
      generator: new MockGenerator(REFLECT_RESPONSE),
    });

    // A belief that predates the column, sitting in an otherwise-idle store:
    // reflect returns early on min-facts, and the backfill must still run.
    const seed = new Database(dbPath);
    seed
      .prepare(
        `INSERT INTO opinions (id, belief, confidence, domain) VALUES ('op-legacy', 'a legacy belief', 0.6, 'test')`,
      )
      .run();
    seed.close();

    await engram.reflect();

    const check = new Database(dbPath, { readonly: true });
    const op = check
      .prepare(`SELECT embedding FROM opinions WHERE id = 'op-legacy'`)
      .get() as { embedding: Buffer | null };
    check.close();
    expect(op.embedding).toBeInstanceOf(Buffer);
  });
});
