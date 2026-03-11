import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect } from '../src/reflect.js';
import {
  MockEmbedder,
  loadSchema,
  tmpDbPath,
  cleanupDb,
  mockOllamaFetch,
  REFLECT_RESPONSE,
} from './helpers.js';

// Helpers
const embedder = new MockEmbedder();

/** Create a temp file db, insert n facts, close it. */
async function setupDb(path: string, factCount: number): Promise<void> {
  const db = new Database(path);
  loadSchema(db);
  for (let i = 0; i < factCount; i++) {
    await retain(db, `Alice prefers Rust — fact ${i}`, embedder, {
      memoryType: 'world',
      sourceType: 'user_stated',
      trustScore: 0.8,
    });
  }
  db.close();
}

// ---------------------------------------------------------------------------
// reflect()
// ---------------------------------------------------------------------------

describe('reflect()', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('skips when fewer than minFactsThreshold facts exist', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 2); // below default threshold of 5

    // fetch should never be called — we set a mock that would fail the test if invoked
    vi.stubGlobal('fetch', async () => {
      throw new Error('fetch should not be called when threshold not met');
    });

    const result = await reflect({ dbPath });
    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBe(0);
  });

  it('processes facts when threshold is met', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({ dbPath });
    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBe(5);
  });

  it('creates observations from LLM output', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({ dbPath });
    expect(result.observationsCreated).toBe(1);

    const db = new Database(dbPath);
    const obs = db.prepare('SELECT * FROM observations').all() as any[];
    db.close();
    expect(obs).toHaveLength(1);
    expect(obs[0].summary).toContain('Alice');
  });

  it('forms new opinions from LLM output', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({ dbPath });
    expect(result.opinionsFormed).toBe(1);

    const db = new Database(dbPath);
    const ops = db.prepare('SELECT * FROM opinions').all() as any[];
    db.close();
    expect(ops).toHaveLength(1);
    expect(ops[0].belief).toContain('Alice');
    expect(ops[0].confidence).toBeGreaterThan(0);
    expect(ops[0].confidence).toBeLessThanOrEqual(0.7);
  });

  it('marks all processed facts as reflected', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    await reflect({ dbPath });

    const db = new Database(dbPath);
    const unreflected = db.prepare(
      `SELECT count(*) as n FROM chunks WHERE reflected_at IS NULL`
    ).get() as { n: number };
    db.close();
    expect(unreflected.n).toBe(0);
  });

  it('writes a completed reflect_log entry', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({ dbPath });

    const db = new Database(dbPath);
    const log = db.prepare('SELECT * FROM reflect_log WHERE id = ?').get(result.logId) as any;
    db.close();

    expect(log.status).toBe('completed');
    expect(log.facts_processed).toBe(5);
    expect(log.observations_created).toBe(1);
    expect(log.opinions_formed).toBe(1);
    expect(log.completed_at).toBeTruthy();
  });

  it('sets status to failed and logs error when LLM returns unparseable output', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

    const result = await reflect({ dbPath });
    expect(result.status).toBe('failed');
    expect(result.error).toBeDefined();

    const db = new Database(dbPath);
    const log = db.prepare('SELECT * FROM reflect_log WHERE id = ?').get(result.logId) as any;
    db.close();
    expect(log.status).toBe('failed');
    expect(log.error).toBeTruthy();
  });

  it('sets status to failed when Ollama is unreachable', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 503, text: async () => 'Service Unavailable' } as unknown as Response));

    const result = await reflect({ dbPath });
    expect(result.status).toBe('failed');
  });

  it('records durationMs in result', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({ dbPath });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('reinforces the exact matching opinion instead of a same-prefix sibling', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(`
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-1', 'Tom prefers Terraform for all infrastructure work', 0.7, 'infrastructure', '[]', '[]')
    `).run();
    db.prepare(`
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-2', 'Tom prefers Terraform for all production deployments', 0.7, 'infrastructure', '[]', '[]')
    `).run();
    db.close();

    vi.stubGlobal('fetch', mockOllamaFetch(JSON.stringify({
      observations: [],
      opinion_updates: [
        {
          belief: 'Tom prefers Terraform for all production deployments',
          direction: 'reinforce',
          confidence_delta: 0.1,
          domain: 'infrastructure',
          evidence_chunk_ids: ['chk-1'],
          entity_names: [],
        },
      ],
      observation_refreshes: [],
    })));

    const result = await reflect({ dbPath });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const opinions = verify.prepare(`
      SELECT id, belief, confidence
      FROM opinions
      WHERE id IN ('op-1', 'op-2')
      ORDER BY id
    `).all() as Array<{ id: string; belief: string; confidence: number }>;
    verify.close();

    expect(opinions.find(op => op.id === 'op-1')?.confidence).toBe(0.7);
    expect(opinions.find(op => op.id === 'op-2')?.confidence).toBeGreaterThan(0.7);
  });
});
