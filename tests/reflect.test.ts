import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect, reflectCatchUp, ReflectScheduler } from '../src/reflect.js';
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

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBe(0);
  });

  it('processes facts when threshold is met', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBe(5);
  });

  it('creates observations from LLM output', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
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

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
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

    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const db = new Database(dbPath);
    const unreflected = db
      .prepare(`SELECT count(*) as n FROM chunks WHERE reflected_at IS NULL`)
      .get() as { n: number };
    db.close();
    expect(unreflected.n).toBe(0);
  });

  it('writes a completed reflect_log entry', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const db = new Database(dbPath);
    const log = db
      .prepare('SELECT * FROM reflect_log WHERE id = ?')
      .get(result.logId) as any;
    db.close();

    expect(log.status).toBe('completed');
    expect(log.facts_processed).toBe(5);
    expect(log.observations_created).toBe(1);
    expect(log.opinions_formed).toBe(1);
    expect(log.completed_at).toBeTruthy();
  });

  it('completes with zero observations when LLM returns unparseable output', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    // Graceful recovery: unparseable JSON → empty arrays, not a hard failure.
    // But this IS a silent-failure cycle (0 insights despite chunks meeting
    // the threshold) — status is 'partial', not 'completed', so it's
    // distinguishable from a genuine no-data-yet quiet cycle (issue #17).
    expect(result.status).toBe('partial');
    expect(result.observationsCreated).toBe(0);
    expect(result.opinionsFormed).toBe(0);
  });

  it('does not mark facts as reflected when parse produces no insights', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 3);
    vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    // Facts should still be unreflected so the next cycle can retry them
    const db = new Database(dbPath);
    const unreflected = db
      .prepare(`SELECT COUNT(*) as cnt FROM chunks WHERE reflected_at IS NULL`)
      .get() as any;
    expect(unreflected.cnt).toBe(3);
    db.close();
  });

  it('sets status to failed when Ollama is unreachable', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      async () =>
        ({
          ok: false,
          status: 503,
          text: async () => 'Service Unavailable',
        }) as unknown as Response,
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.status).toBe('failed');
  });

  it('records durationMs in result', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('reinforces only the opinion in the matching domain, not a same-belief in a different domain', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    // Same belief text, different domains
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-infra', 'Kubernetes is the preferred orchestration platform', 0.7, 'infrastructure', '[]', '[]')
    `,
    ).run();
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-dev', 'Kubernetes is the preferred orchestration platform', 0.7, 'development', '[]', '[]')
    `,
    ).run();
    db.close();

    // Reinforce only the 'infrastructure' domain opinion
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief: 'Kubernetes is the preferred orchestration platform',
              direction: 'reinforce',
              confidence_delta: 0.1,
              domain: 'infrastructure',
              evidence_chunk_ids: ['chk-x'],
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const opinions = verify
      .prepare(
        `
      SELECT id, confidence FROM opinions WHERE id IN ('op-infra', 'op-dev')
    `,
      )
      .all() as Array<{ id: string; confidence: number }>;
    verify.close();

    const infra = opinions.find((op) => op.id === 'op-infra')!;
    const dev = opinions.find((op) => op.id === 'op-dev')!;
    // Only the infrastructure opinion should have increased confidence
    expect(infra.confidence).toBeGreaterThan(0.7);
    expect(dev.confidence).toBe(0.7);
  });

  it('reinforces the exact matching opinion instead of a same-prefix sibling', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-1', 'Tom prefers Terraform for all infrastructure work', 0.7, 'infrastructure', '[]', '[]')
    `,
    ).run();
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-2', 'Tom prefers Terraform for all production deployments', 0.7, 'infrastructure', '[]', '[]')
    `,
    ).run();
    db.close();

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
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
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const opinions = verify
      .prepare(
        `
      SELECT id, belief, confidence
      FROM opinions
      WHERE id IN ('op-1', 'op-2')
      ORDER BY id
    `,
      )
      .all() as Array<{ id: string; belief: string; confidence: number }>;
    verify.close();

    expect(opinions.find((op) => op.id === 'op-1')?.confidence).toBe(0.7);
    expect(opinions.find((op) => op.id === 'op-2')?.confidence).toBeGreaterThan(
      0.7,
    );
  });

  it('reinforces via fuzzy match when belief text differs slightly', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    // Stored belief has slightly different wording than what the LLM will output
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-fuzzy', 'Alice strongly prefers Rust over other systems languages for performance work', 0.6, 'preferences', '[]', '[]')
    `,
    ).run();
    db.close();

    // LLM outputs a belief that is similar but not identical (triggers beliefSimilarity >= 0.85)
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief:
                'Alice strongly prefers Rust over other systems languages for performance tasks',
              direction: 'reinforce',
              confidence_delta: 0.1,
              domain: 'preferences',
              evidence_chunk_ids: ['chk-fuzzy'],
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE id = 'op-fuzzy'`)
      .get() as { confidence: number };
    verify.close();

    // With row-ID-based UPDATE, fuzzy match successfully updates the opinion
    expect(op.confidence).toBeGreaterThan(0.6);
  });

  it('decays stale opinions not reinforced in 30+ days', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    // Insert an opinion that hasn't been reinforced in 60 days and not updated in 8 days
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities,
        last_reinforced, updated_at)
      VALUES ('op-stale', 'Stale opinion', 0.8, 'test', '[]', '[]',
        datetime('now', '-60 days'), datetime('now', '-8 days'))
    `,
    ).run();
    db.close();

    // Reflect will apply decay before gathering facts
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [],
          observation_refreshes: [],
        }),
      ),
    );

    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE id = 'op-stale'`)
      .get() as { confidence: number };
    verify.close();

    expect(op.confidence).toBeCloseTo(0.78, 2);
  });

  it('clamps reinforce delta exceeding +0.15 to 0.15', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-clamp', 'Clampable belief', 0.5, 'test', '[]', '[]')
    `,
    ).run();
    db.close();

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief: 'Clampable belief',
              direction: 'reinforce',
              confidence_delta: 0.5, // way over 0.15
              domain: 'test',
              evidence_chunk_ids: ['chk-x'],
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE id = 'op-clamp'`)
      .get() as { confidence: number };
    verify.close();

    // Delta clamped to +0.15 → 0.5 + 0.15 = 0.65
    expect(op.confidence).toBeCloseTo(0.65, 2);
  });

  it('clamps challenge delta exceeding -0.15 to -0.15', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, contradicting_chunks, related_entities)
      VALUES ('op-challenge', 'Challengeable belief', 0.8, 'test', '[]', '[]', '[]')
    `,
    ).run();
    db.close();

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief: 'Challengeable belief',
              direction: 'challenge',
              confidence_delta: -0.5, // way under -0.15
              domain: 'test',
              evidence_chunk_ids: ['chk-y'],
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsChallenged).toBe(1);

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE id = 'op-challenge'`)
      .get() as { confidence: number };
    verify.close();

    // Delta clamped to -0.15 → 0.8 - 0.15 = 0.65
    expect(op.confidence).toBeCloseTo(0.65, 2);
  });

  it('clamps new opinion confidence to [0.3, 0.7] even with extreme delta', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief: 'Extreme confidence new opinion',
              direction: 'new',
              confidence_delta: 5.0, // absurdly high
              domain: 'test',
              evidence_chunk_ids: ['chk-z'],
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsFormed).toBe(1);

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE belief LIKE '%Extreme%'`)
      .get() as { confidence: number };
    verify.close();

    // New opinion: min(0.7, max(0.3, 0.5 + 5.0)) = min(0.7, 5.5) = 0.7
    expect(op.confidence).toBe(0.7);
  });

  it('dedups a repeated "new" verdict into a reinforcement — no duplicate opinion row, confidence increases', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    // Same belief, re-emitted as "new" both cycles — this is what a model
    // does when it re-derives a belief it already formed instead of
    // recognizing it should reinforce. Without dedup, this accumulates a
    // duplicate opinion row every cycle.
    const sameNewOpinionResponse = JSON.stringify({
      observations: [],
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

    vi.stubGlobal('fetch', mockOllamaFetch(sameNewOpinionResponse));

    // Cycle 1: no matching opinion yet — genuinely new, inserts.
    const result1 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result1.opinionsFormed).toBe(1);
    expect(result1.opinionsReinforced).toBe(0);

    const afterCycle1 = new Database(dbPath);
    const opsAfter1 = afterCycle1
      .prepare('SELECT * FROM opinions')
      .all() as any[];
    afterCycle1.close();
    expect(opsAfter1).toHaveLength(1);

    // Cycle 2: fresh unreflected facts (distinct text — setupDb's literal
    // text would otherwise dedup against the already-reflected chunks from
    // cycle 1 and never register as new unreflected facts), same "new"
    // verdict re-emitted.
    const db2 = new Database(dbPath);
    for (let i = 0; i < 5; i++) {
      await retain(db2, `Alice prefers Rust — round 2 fact ${i}`, embedder, {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      });
    }
    db2.close();
    vi.unstubAllGlobals();
    vi.stubGlobal('fetch', mockOllamaFetch(sameNewOpinionResponse));

    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result2.opinionsFormed).toBe(0);
    expect(result2.opinionsReinforced).toBe(1);

    const afterCycle2 = new Database(dbPath);
    const opsAfter2 = afterCycle2
      .prepare('SELECT * FROM opinions')
      .all() as any[];
    afterCycle2.close();

    // Still exactly one opinion row — converted to a reinforcement, not inserted again.
    expect(opsAfter2).toHaveLength(1);
    expect(opsAfter2[0].id).toBe(opsAfter1[0].id);
    // Confidence increased rather than being reset by a fresh insert.
    expect(opsAfter2[0].confidence).toBeGreaterThan(opsAfter1[0].confidence);
  });

  it('a "new" verdict with no matching existing opinion still inserts (not swallowed by dedup)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-unrelated', 'Tom prefers Terraform for infrastructure', 0.6, 'infrastructure', '[]', '[]')
    `,
    ).run();
    db.close();

    // REFLECT_RESPONSE's belief/domain don't match the pre-existing opinion.
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsReinforced).toBe(0);

    const verify = new Database(dbPath);
    const ops = verify.prepare('SELECT * FROM opinions').all() as any[];
    verify.close();
    expect(ops).toHaveLength(2); // pre-existing unrelated opinion + the new one
  });

  // ---------------------------------------------------------------------------
  // D2 — observation dedup-on-synthesis
  // ---------------------------------------------------------------------------

  it('dedups a repeated "new" observation into a refresh — no duplicate observation row, sources merge (D2)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const cycle1Response = JSON.stringify({
      observations: [
        {
          summary:
            'Alice consistently chooses Rust for systems programming tasks',
          domain: 'preferences',
          topic: 'programming languages',
          source_chunk_ids: ['chk-a'],
          entity_names: [],
        },
      ],
      opinion_updates: [],
      observation_refreshes: [],
    });

    vi.stubGlobal('fetch', mockOllamaFetch(cycle1Response));

    // Cycle 1: no matching observation yet — genuinely new, inserts.
    const result1 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result1.observationsCreated).toBe(1);
    expect(result1.observationsUpdated).toBe(0);

    const afterCycle1 = new Database(dbPath);
    const obsAfter1 = afterCycle1
      .prepare('SELECT * FROM observations')
      .all() as any[];
    afterCycle1.close();
    expect(obsAfter1).toHaveLength(1);

    // Cycle 2: fresh unreflected facts (distinct text — setupDb's literal
    // text would otherwise dedup against the already-reflected chunks from
    // cycle 1 and never register as new unreflected facts), the same
    // insight re-derived as "new" with a different source chunk.
    const db2 = new Database(dbPath);
    for (let i = 0; i < 5; i++) {
      await retain(db2, `Alice prefers Rust — round 2 fact ${i}`, embedder, {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      });
    }
    db2.close();
    vi.unstubAllGlobals();
    const cycle2Response = JSON.stringify({
      observations: [
        {
          summary:
            'Alice consistently chooses Rust for systems programming tasks',
          domain: 'preferences',
          topic: 'programming languages',
          source_chunk_ids: ['chk-b'],
          entity_names: [],
        },
      ],
      opinion_updates: [],
      observation_refreshes: [],
    });
    vi.stubGlobal('fetch', mockOllamaFetch(cycle2Response));

    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result2.observationsCreated).toBe(0);
    expect(result2.observationsUpdated).toBe(1);

    const afterCycle2 = new Database(dbPath);
    const obsAfter2 = afterCycle2
      .prepare('SELECT * FROM observations')
      .all() as any[];
    afterCycle2.close();

    // Still exactly one observation row — refreshed, not inserted again.
    expect(obsAfter2).toHaveLength(1);
    expect(obsAfter2[0].id).toBe(obsAfter1[0].id);
    expect(obsAfter2[0].refresh_count).toBe(1);
    const sources = JSON.parse(obsAfter2[0].source_chunks);
    expect(sources).toContain('chk-a');
    expect(sources).toContain('chk-b');
  });

  it('a "new" observation with no matching existing observation still inserts (not swallowed by dedup)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at)
      VALUES ('obs-unrelated', 'Tom deploys with Terraform', '[]', '[]', 'infrastructure', 'iac', datetime('now'))
    `,
    ).run();
    db.close();

    // REFLECT_RESPONSE's summary/domain/topic don't match the pre-existing observation.
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.observationsCreated).toBe(1);
    expect(result.observationsUpdated).toBe(0);

    const verify = new Database(dbPath);
    const obs = verify.prepare('SELECT * FROM observations').all() as any[];
    verify.close();
    expect(obs).toHaveLength(2); // pre-existing unrelated observation + the new one
  });

  it('does not dedup observations that match on summary but differ in topic (D2 scoping)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    // Same domain, same summary text, but a different topic — findMatchingObservation
    // scopes by domain+topic (tighter than opinions' domain-only scope), so this
    // must NOT match.
    db.prepare(
      `
      INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at)
      VALUES ('obs-other-topic', 'Alice consistently chooses Rust for systems programming tasks', '[]', '[]', 'preferences', 'tooling', datetime('now'))
    `,
    ).run();
    db.close();

    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.observationsCreated).toBe(1);
    expect(result.observationsUpdated).toBe(0);

    const verify = new Database(dbPath);
    const obs = verify.prepare('SELECT * FROM observations').all() as any[];
    verify.close();
    expect(obs).toHaveLength(2);
  });

  it('refreshes an existing observation with new source chunks', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    db.prepare(
      `
      INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at, refresh_count)
      VALUES ('obs-refresh', 'Alice uses Rust', '["chk-old"]', '[]', 'preferences', 'languages', datetime('now'), 0)
    `,
    ).run();
    db.close();

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [],
          observation_refreshes: [
            {
              existing_observation_id: 'obs-refresh',
              updated_summary: 'Alice uses Rust for all systems and CLI work',
              new_source_chunk_ids: ['chk-new1', 'chk-new2'],
            },
          ],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.observationsUpdated).toBe(1);

    const verify = new Database(dbPath);
    const obs = verify
      .prepare(
        `SELECT summary, source_chunks, refresh_count FROM observations WHERE id = 'obs-refresh'`,
      )
      .get() as any;
    verify.close();

    expect(obs.summary).toContain('CLI work');
    expect(obs.refresh_count).toBe(1);
    const sources = JSON.parse(obs.source_chunks);
    expect(sources).toContain('chk-old');
    expect(sources).toContain('chk-new1');
    expect(sources).toContain('chk-new2');
  });

  it('skips observation refresh when existing_observation_id is not found', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [
            {
              summary: 'A real observation for marking reflected',
              domain: 'test',
              topic: 'test',
              source_chunk_ids: [],
              entity_names: [],
            },
          ],
          opinion_updates: [],
          observation_refreshes: [
            {
              existing_observation_id: 'obs-nonexistent',
              updated_summary: 'This should be skipped',
              new_source_chunk_ids: ['chk-x'],
            },
          ],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    // The refresh is silently skipped (observation not found), but the new observation counts
    expect(result.observationsUpdated).toBe(0);
    expect(result.observationsCreated).toBe(1);
  });

  it('leaves facts unreflected when LLM returns valid JSON with empty arrays', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    // 0 insights with a full batch (>= minFactsThreshold) is a silent-failure
    // signal for monitoring purposes, even though the JSON parsed cleanly —
    // status is 'partial', not 'completed' (issue #17).
    expect(result.status).toBe('partial');
    expect(result.factsProcessed).toBe(0);

    // Facts stay unreflected for retry
    const db = new Database(dbPath);
    const unreflected = db
      .prepare(`SELECT COUNT(*) as cnt FROM chunks WHERE reflected_at IS NULL`)
      .get() as any;
    db.close();
    expect(unreflected.cnt).toBe(5);
  });

  it('dampens reinforcement when evidence is agent-generated', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 5);

    const db = new Database(dbPath);
    // Insert an existing opinion to be reinforced
    db.prepare(
      `
      INSERT INTO opinions (id, belief, confidence, domain, supporting_chunks, related_entities)
      VALUES ('op-agent', 'Agent-reinforced belief', 0.6, 'test', '[]', '[]')
    `,
    ).run();

    // Insert agent-generated chunks as evidence
    const chunkIds = ['chk-ag1', 'chk-ag2'];
    for (const id of chunkIds) {
      db.prepare(
        `
        INSERT INTO chunks (id, text, embedding, memory_type, source_type, trust_score)
        VALUES (?, 'agent output', zeroblob(32), 'world', 'agent_generated', 0.5)
      `,
      ).run(id);
    }
    db.close();

    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        JSON.stringify({
          observations: [],
          opinion_updates: [
            {
              belief: 'Agent-reinforced belief',
              direction: 'reinforce',
              confidence_delta: 0.1,
              domain: 'test',
              evidence_chunk_ids: chunkIds,
              entity_names: [],
            },
          ],
          observation_refreshes: [],
        }),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsReinforced).toBe(1);

    const verify = new Database(dbPath);
    const op = verify
      .prepare(`SELECT confidence FROM opinions WHERE id = 'op-agent'`)
      .get() as { confidence: number };
    verify.close();

    // With dampening (0.5x), delta of 0.10 becomes 0.05, so 0.6 + 0.05 = 0.65
    expect(op.confidence).toBeCloseTo(0.65, 2);
  });

  // ---------------------------------------------------------------------------
  // source_type filtering — reflect the high-value slice ahead of tool noise
  // ---------------------------------------------------------------------------

  describe('sourceTypes filter', () => {
    /** Seed a backlog shaped like a real auto-retain deployment: mostly tool noise. */
    async function setupMixedDb(path: string): Promise<void> {
      const db = new Database(path);
      loadSchema(db);
      for (let i = 0; i < 8; i++) {
        await retain(db, `bash output line ${i}`, embedder, {
          memoryType: 'experience',
          sourceType: 'tool_result',
          trustScore: 0.3,
        });
      }
      // Retained LAST, so `created_at ASC` puts them behind the noise — the
      // whole point of the filter is that they still get reflected first.
      for (let i = 0; i < 3; i++) {
        await retain(db, `Alice prefers Rust — stated ${i}`, embedder, {
          memoryType: 'world',
          sourceType: 'user_stated',
          trustScore: 0.9,
        });
      }
      db.close();
    }

    it('reflects only facts matching the filter, ignoring older non-matching ones', async () => {
      dbPath = tmpDbPath();
      await setupMixedDb(dbPath);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 50,
        minFactsThreshold: 1,
        sourceTypes: ['user_stated'],
      });

      // 3 user_stated facts consumed; the 8 older tool_result rows untouched.
      expect(result.factsProcessed).toBe(3);

      const db = new Database(dbPath);
      const stillPending = db
        .prepare(
          `SELECT source_type AS st, COUNT(*) AS n FROM chunks
            WHERE reflected_at IS NULL GROUP BY st`,
        )
        .all() as Array<{ st: string; n: number }>;
      db.close();

      expect(stillPending).toEqual([{ st: 'tool_result', n: 8 }]);
    });

    it('accepts multiple source types', async () => {
      dbPath = tmpDbPath();
      await setupMixedDb(dbPath);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 50,
        minFactsThreshold: 1,
        sourceTypes: ['user_stated', 'tool_result'],
      });

      expect(result.factsProcessed).toBe(11);
    });

    it('treats an omitted filter as "every source type" (unchanged behaviour)', async () => {
      dbPath = tmpDbPath();
      await setupMixedDb(dbPath);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 50,
        minFactsThreshold: 1,
      });

      expect(result.factsProcessed).toBe(11);
    });

    it('treats an empty filter array as "no filter", never "match nothing"', async () => {
      dbPath = tmpDbPath();
      await setupMixedDb(dbPath);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 50,
        minFactsThreshold: 1,
        sourceTypes: [],
      });

      expect(result.factsProcessed).toBe(11);
    });

    it('reports a fully-consumed filtered slice as drained, not stalled', async () => {
      // Regression guard: remainingBacklog/stall detection must count only the
      // slice the filter can consume. Counting the whole backlog would leave the
      // 8 tool_result rows visible, so catch-up would keep looping on an empty
      // slice and stop with 'stalled' despite having drained everything it could.
      dbPath = tmpDbPath();
      await setupMixedDb(dbPath);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflectCatchUp({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 50,
        minFactsThreshold: 1,
        maxBatches: 5,
        sourceTypes: ['user_stated'],
      });

      expect(result.factsProcessed).toBe(3);
      expect(result.remainingBacklog).toBe(0);
      expect(result.status).toBe('drained');
    });
  });

  // ---------------------------------------------------------------------------
  // issue #17 — adaptive batch sizing, char-budget truncation, status distinction
  // ---------------------------------------------------------------------------

  describe('adaptive batch sizing (issue #17)', () => {
    it('persists a shrunk reflect_batch_hint after a 0-insight cycle with a full batch', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 10);
      vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 10,
      });

      const db = new Database(dbPath);
      const hint = db
        .prepare(
          `SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`,
        )
        .get() as { value: string } | undefined;
      db.close();

      expect(hint).toBeDefined();
      // floor(10 / 2) = 5, and >= minFactsThreshold (5)
      expect(Number(hint!.value)).toBe(5);
    });

    it('applies the persisted batch hint to shrink the next cycle, when batchSize is not explicitly passed', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 20);

      const db = new Database(dbPath);
      db.prepare(
        `INSERT INTO bank_config (key, value) VALUES ('reflect_batch_hint', '7')`,
      ).run();
      db.close();

      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      }); // no explicit batchSize — should honor the hint

      const verify = new Database(dbPath);
      const reflectedCount = verify
        .prepare(
          `SELECT COUNT(*) as cnt FROM chunks WHERE reflected_at IS NOT NULL`,
        )
        .get() as { cnt: number };
      verify.close();

      // Hint caps the batch at 7, even though 20 facts and the configured
      // default (50) would otherwise allow more.
      expect(reflectedCount.cnt).toBe(7);
    });

    it('does not apply the batch hint when the caller passes an explicit batchSize', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 20);

      const db = new Database(dbPath);
      db.prepare(
        `INSERT INTO bank_config (key, value) VALUES ('reflect_batch_hint', '5')`,
      ).run();
      db.close();

      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      // Explicit override of 12 must win over the persisted hint of 5
      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 12,
      });

      const verify = new Database(dbPath);
      const reflectedCount = verify
        .prepare(
          `SELECT COUNT(*) as cnt FROM chunks WHERE reflected_at IS NOT NULL`,
        )
        .get() as { cnt: number };
      verify.close();

      expect(reflectedCount.cnt).toBe(12);
    });

    it('resets the batch hint back to default after a cycle that produces insights', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      const db = new Database(dbPath);
      db.prepare(
        `INSERT INTO bank_config (key, value) VALUES ('reflect_batch_hint', '5')`,
      ).run();
      db.close();

      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });
      expect(result.observationsCreated).toBeGreaterThan(0);

      const verify = new Database(dbPath);
      const hint = verify
        .prepare(
          `SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`,
        )
        .get();
      verify.close();

      expect(hint).toBeUndefined();
    });

    it('floors the shrunk batch hint at minFactsThreshold', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 6);
      vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

      // floor(6/2) = 3, which is below minFactsThreshold (5) → clamp to 5
      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        batchSize: 6,
        minFactsThreshold: 5,
      });

      const db = new Database(dbPath);
      const hint = db
        .prepare(
          `SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`,
        )
        .get() as { value: string };
      db.close();

      expect(Number(hint.value)).toBe(5);
    });
  });

  describe('existing-context character-budget cap (issue #17)', () => {
    it('truncates existing observations included in the prompt once the char budget is exceeded', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      const db = new Database(dbPath);
      // Insert observations with long summaries so a small budget truncates
      // most of them. Ordered by last_refreshed/synthesized_at DESC, so the
      // most recently inserted (highest N) sort first.
      const longSummary = 'X'.repeat(500);
      for (let i = 0; i < 10; i++) {
        db.prepare(
          `
          INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at)
          VALUES (?, ?, '[]', '[]', 'test', 'topic', datetime('now', ?))
        `,
        ).run(`obs-${i}`, `${longSummary} (${i})`, `+${i} seconds`);
      }
      db.close();

      let capturedPrompt = '';
      vi.stubGlobal('fetch', async (_url: unknown, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body ?? '{}'));
        capturedPrompt = body.prompt ?? '';
        return {
          ok: true,
          status: 200,
          json: async () => ({ response: REFLECT_RESPONSE }),
          text: async () => REFLECT_RESPONSE,
        } as unknown as Response;
      });

      // Budget small enough that only a couple of the 500-char summaries fit
      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        existingContextCharBudget: 1200,
      });

      // The most-recently-synthesized observation (obs-9) should always be
      // present under a most-recent-first ordering; an old one inserted
      // first (obs-0) should have been truncated out of the 10-observation
      // set given the tight budget.
      expect(capturedPrompt).toContain('(9)');
      expect(capturedPrompt).not.toContain('(0)');
    });

    it('always includes at least one observation even if it alone exceeds the char budget', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      const db = new Database(dbPath);
      db.prepare(
        `
        INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at)
        VALUES ('obs-huge', ?, '[]', '[]', 'test', 'topic', datetime('now'))
      `,
      ).run('Y'.repeat(5000));
      db.close();

      let capturedPrompt = '';
      vi.stubGlobal('fetch', async (_url: unknown, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body ?? '{}'));
        capturedPrompt = body.prompt ?? '';
        return {
          ok: true,
          status: 200,
          json: async () => ({ response: REFLECT_RESPONSE }),
          text: async () => REFLECT_RESPONSE,
        } as unknown as Response;
      });

      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
        existingContextCharBudget: 100,
      });

      expect(capturedPrompt).toContain('obs-huge');
    });
  });

  describe('reflect_log status distinction (issue #17)', () => {
    it('logs status "partial" (not "completed") when a full batch produces 0 insights', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);
      vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      const db = new Database(dbPath);
      const log = db
        .prepare('SELECT status FROM reflect_log WHERE id = ?')
        .get(result.logId) as { status: string };
      db.close();

      expect(result.status).toBe('partial');
      expect(log.status).toBe('partial');
    });

    it('keeps status "completed" when there are too few unreflected facts to trigger reflection', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 2); // below default minFactsThreshold of 5
      vi.stubGlobal('fetch', async () => {
        throw new Error('fetch should not be called when threshold not met');
      });

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      const db = new Database(dbPath);
      const log = db
        .prepare('SELECT status FROM reflect_log WHERE id = ?')
        .get(result.logId) as { status: string };
      db.close();

      // Not enough data yet is a genuinely healthy quiet cycle, not a
      // silent failure — must stay 'completed'.
      expect(result.status).toBe('completed');
      expect(log.status).toBe('completed');
    });

    it('keeps status "completed" when the cycle actually produces insights', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);
      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      const db = new Database(dbPath);
      const log = db
        .prepare('SELECT status FROM reflect_log WHERE id = ?')
        .get(result.logId) as { status: string };
      db.close();

      expect(result.status).toBe('completed');
      expect(log.status).toBe('completed');
    });
  });

  // ---------------------------------------------------------------------------
  // Empty generation response (transient cloud/endpoint failure)
  // ---------------------------------------------------------------------------
  //
  // An empty completion (0-char body on an HTTP 200, e.g. a flaky ':cloud'
  // model) is a *transient generation failure*, NOT the oversized-prompt/parse
  // failure the issue-#17 auto-shrink path is built for. It must be classified
  // 'failed' (honest, greppable) and must NOT shrink the batch hint — otherwise
  // a momentary blip throttles throughput for a problem that isn't size-related.
  // Contrast with the 'partial'+shrink tests above, which use *non-empty*
  // garbage: that path is intentionally left intact.
  describe('empty generation response', () => {
    it('records status "failed" (not "partial") on an empty response', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5); // full batch, >= minFactsThreshold
      vi.stubGlobal('fetch', mockOllamaFetch('')); // 200 OK, empty body

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      const db = new Database(dbPath);
      const log = db
        .prepare('SELECT status FROM reflect_log WHERE id = ?')
        .get(result.logId) as { status: string };
      db.close();

      expect(result.status).toBe('failed');
      expect(log.status).toBe('failed');
      expect(result.error).toMatch(/empty response/i);
    });

    it('does NOT shrink the batch hint on an empty response', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 10);
      vi.stubGlobal('fetch', mockOllamaFetch(''));

      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      const db = new Database(dbPath);
      const hint = db
        .prepare(
          `SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`,
        )
        .get() as { value: string } | undefined;
      db.close();

      // The shrink block is never reached (the throw short-circuits before it),
      // so no hint is written — unlike the non-empty-garbage case above.
      expect(hint).toBeUndefined();
    });

    it('leaves facts unreflected for the next scheduled cycle', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);
      vi.stubGlobal('fetch', mockOllamaFetch(''));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });
      expect(result.factsProcessed).toBe(0);

      const db = new Database(dbPath);
      const unreflected = db
        .prepare(
          `SELECT COUNT(*) AS n FROM chunks WHERE reflected_at IS NULL AND is_active = TRUE`,
        )
        .get() as { n: number };
      db.close();

      expect(unreflected.n).toBe(5);
    });
  });

  // ---------------------------------------------------------------------------
  // D4 — durability prompt + attribution guard
  // ---------------------------------------------------------------------------

  describe('durability rubric in prompt (D4)', () => {
    it('includes a durability rubric instructing rejection of transient operational state', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      let capturedPrompt = '';
      vi.stubGlobal('fetch', async (_url: unknown, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body ?? '{}'));
        capturedPrompt = body.prompt ?? '';
        return {
          ok: true,
          status: 200,
          json: async () => ({ response: REFLECT_RESPONSE }),
          text: async () => REFLECT_RESPONSE,
        } as unknown as Response;
      });

      await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });

      expect(capturedPrompt).toContain('Durability Rule');
      expect(capturedPrompt.toLowerCase()).toContain('expiring');
      expect(capturedPrompt.toLowerCase()).toContain('transient');
    });
  });

  describe('entity attribution sanity check (D4)', () => {
    it('drops an opinion entity attribution that does not appear in the belief text', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      vi.stubGlobal(
        'fetch',
        mockOllamaFetch(
          JSON.stringify({
            observations: [],
            opinion_updates: [
              {
                belief: 'Rust is well-suited to systems programming',
                direction: 'new',
                confidence_delta: 0.2,
                domain: 'preferences',
                evidence_chunk_ids: [],
                // 'Zephyr' never appears in the belief text — an implausible
                // attribution the guard should drop.
                entity_names: ['Zephyr'],
              },
            ],
            observation_refreshes: [],
          }),
        ),
      );

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });
      expect(result.opinionsFormed).toBe(1);

      const verify = new Database(dbPath);
      const op = verify
        .prepare(
          `SELECT related_entities FROM opinions WHERE belief LIKE '%Rust is well-suited%'`,
        )
        .get() as { related_entities: string };
      verify.close();

      expect(JSON.parse(op.related_entities)).toEqual([]);
    });

    it('keeps an entity attribution that plausibly appears in the belief text', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });
      expect(result.opinionsFormed).toBe(1);

      const verify = new Database(dbPath);
      const op = verify
        .prepare(`SELECT related_entities FROM opinions`)
        .get() as { related_entities: string };
      verify.close();

      // 'Alice' appears in REFLECT_RESPONSE's belief text — attribution kept
      // (falls back to the raw name since no entity has been extracted yet).
      expect(JSON.parse(op.related_entities)).toEqual(['Alice']);
    });

    it('drops an observation entity attribution that does not appear in the summary text', async () => {
      dbPath = tmpDbPath();
      await setupDb(dbPath, 5);

      vi.stubGlobal(
        'fetch',
        mockOllamaFetch(
          JSON.stringify({
            observations: [
              {
                summary: 'Rust is chosen for systems programming tasks',
                domain: 'preferences',
                topic: 'programming languages',
                source_chunk_ids: [],
                // 'Zephyr' never appears in the summary text.
                entity_names: ['Zephyr'],
              },
            ],
            opinion_updates: [],
            observation_refreshes: [],
          }),
        ),
      );

      const result = await reflect({
        counterEvidence: false,
        opinionGates: false,
        dbPath,
        reflectModel: 'llama-test',
      });
      expect(result.observationsCreated).toBe(1);

      const verify = new Database(dbPath);
      const obs = verify
        .prepare(`SELECT source_entities FROM observations`)
        .get() as { source_entities: string };
      verify.close();

      expect(JSON.parse(obs.source_entities)).toEqual([]);
    });
  });
});

describe('ReflectScheduler', () => {
  it('requires an embedder when the default counter-evidence audit is enabled', () => {
    expect(
      () =>
        new ReflectScheduler({
          dbPath: './agent.engram',
          reflectModel: 'llama-test',
        }),
    ).toThrow('requires an embedder');
  });

  it('allows an explicit counter-evidence opt-out without an embedder', () => {
    expect(
      () =>
        new ReflectScheduler({
          dbPath: './agent.engram',
          reflectModel: 'llama-test',
          counterEvidence: false,
        }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// reflectCatchUp() — D5: drain a reflection backlog over many batches
// ---------------------------------------------------------------------------

describe('reflectCatchUp()', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('drains a backlog across multiple batches until below threshold', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 12);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      batchSize: 5,
      minFactsThreshold: 1,
    });

    // 12 facts / batch 5 → 5 + 5 + 2 = 3 batches, then backlog < threshold.
    expect(result.batches).toBe(3);
    expect(result.factsProcessed).toBe(12);
    expect(result.remainingBacklog).toBe(0);
    expect(result.status).toBe('drained');
    expect(result.batchResults).toHaveLength(3);
  });

  it('is a no-op when the backlog is already below threshold', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 3); // below the default threshold of 5
    // fetch must never be called — no batch should run
    vi.stubGlobal('fetch', async () => {
      throw new Error('fetch should not be called when already caught up');
    });

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    expect(result.batches).toBe(0);
    expect(result.factsProcessed).toBe(0);
    expect(result.remainingBacklog).toBe(3);
    expect(result.status).toBe('drained');
  });

  it('stops at maxBatches with backlog remaining (capped)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 30);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      batchSize: 5,
      minFactsThreshold: 1,
      maxBatches: 2,
    });

    expect(result.batches).toBe(2);
    expect(result.factsProcessed).toBe(10);
    expect(result.remainingBacklog).toBe(20);
    expect(result.status).toBe('capped');
  });

  it('stops once maxFacts have been reflected (capped)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 20);
    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      batchSize: 5,
      minFactsThreshold: 1,
      maxFacts: 8,
    });

    // Checked between batches: 5 (<8, continue), 10 (>=8, stop before batch 3).
    expect(result.factsProcessed).toBe(10);
    expect(result.status).toBe('capped');
    expect(result.remainingBacklog).toBe(10);
  });

  it('stops after maxStalls consecutive zero-progress batches (stalled)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 12);
    // Unparseable output → 0 insights → facts left unreflected → no progress.
    vi.stubGlobal('fetch', mockOllamaFetch('this is not json at all'));

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      minFactsThreshold: 1,
      maxStalls: 2,
    });

    expect(result.batches).toBe(2);
    expect(result.factsProcessed).toBe(0);
    expect(result.status).toBe('stalled');
    // Nothing was reflected, so the whole backlog is still outstanding.
    expect(result.remainingBacklog).toBe(12);
  });

  it('stops immediately when a batch fails (failed)', async () => {
    dbPath = tmpDbPath();
    await setupDb(dbPath, 12);
    // A connection-class error surfaces as reflect status 'failed'.
    vi.stubGlobal('fetch', async () => {
      throw new Error('connection refused');
    });

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      minFactsThreshold: 1,
    });

    expect(result.batches).toBe(1);
    expect(result.status).toBe('failed');
    expect(result.error).toBeTruthy();
    expect(result.remainingBacklog).toBe(12);
  });
});
