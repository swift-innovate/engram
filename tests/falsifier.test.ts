// =============================================================================
// falsifier.test.ts — issue #38 item 3: the would_change_this falsifier field
// and contradiction-driven ("weakened") decay.
//
// Covers: falsifier stored at formation (clamped), omitted → NULL, surfaced
// in the next cycle's reflect prompt, backfilled on reinforcement (never
// overwritten), shown to the counter-evidence judge as "Stated falsifier",
// unanswered-contradiction decay journaled `weakened`, reinforcement-answers-
// challenge stops decay, plain idle decay stays unjournaled, the 7-day
// throttle, introspect projection, and the old-file column migration.
// =============================================================================

import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect } from '../src/reflect.js';
import { introspect } from '../src/introspect.js';
import { Engram } from '../src/engram.js';
import { MockEmbedder, loadSchema, tmpDbPath, cleanupDb } from './helpers.js';

const embedder = new MockEmbedder();

/** Sequenced fetch mock with prompt capture (same shape as counter-evidence tests). */
function mockFetchSequence(responses: string[]): {
  fetchFn: typeof globalThis.fetch;
  prompts: string[];
} {
  const prompts: string[] = [];
  let call = 0;
  const fetchFn = (async (
    _url: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    try {
      const body = JSON.parse((init?.body as string) ?? '{}');
      prompts.push(body.prompt ?? '');
    } catch {
      prompts.push('');
    }
    const responseJson = responses[Math.min(call, responses.length - 1)];
    call++;
    return {
      ok: true,
      status: 200,
      json: async () => ({ response: responseJson }),
      text: async () => responseJson,
    } as unknown as Response;
  }) as typeof globalThis.fetch;
  return { fetchFn, prompts };
}

async function seedFacts(
  path: string,
  texts: string[],
  fresh = true,
): Promise<string[]> {
  const db = new Database(path);
  if (fresh) loadSchema(db);
  const ids: string[] = [];
  for (const text of texts) {
    const result = await retain(db, text, embedder, {
      memoryType: 'world',
      sourceType: 'user_stated',
      trustScore: 0.8,
    });
    ids.push(result.chunkId);
  }
  db.close();
  return ids;
}

const FACTS = [
  'Tom deployed the gateway service on the Proxmox homelab',
  'Tom moved CI runners onto the homelab cluster',
  'Tom said self-hosting keeps recurring costs near zero',
  'Tom migrated the artifact cache off S3 to local MinIO',
  'Tom benchmarked the homelab NAS for build caching',
];
const FALSIFIER =
  'Tom adopts a managed cloud service for a core production workload';

function reflectResponse(updates: Array<Record<string, unknown>>): string {
  return JSON.stringify({
    observations: [],
    opinion_updates: updates,
    observation_refreshes: [],
  });
}

function newOpinion(
  evidenceIds: string[],
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    belief: 'Tom prefers self-hosted infrastructure over managed cloud',
    direction: 'new',
    confidence_delta: 0.1,
    domain: 'infrastructure',
    evidence_chunk_ids: evidenceIds,
    entity_names: ['Tom'],
    rationale: 'Repeated self-hosting decisions',
    would_change_this: FALSIFIER,
    ...overrides,
  };
}

function getOpinions(path: string): any[] {
  const db = new Database(path);
  const rows = db.prepare('SELECT * FROM opinions').all() as any[];
  db.close();
  return rows;
}

function getJournal(path: string, action?: string): any[] {
  const db = new Database(path);
  const rows = (
    action
      ? db
          .prepare(
            'SELECT * FROM belief_journal WHERE action = ? ORDER BY rowid ASC',
          )
          .all(action)
      : db.prepare('SELECT * FROM belief_journal ORDER BY rowid ASC').all()
  ) as any[];
  db.close();
  return rows;
}

/** Insert an opinion row directly, with SQL-expression timestamps. */
function seedOpinion(
  path: string,
  overrides: Partial<Record<string, string>> = {},
): string {
  const db = new Database(path);
  const id = `op-seed-${Math.random().toString(36).slice(2, 8)}`;
  const cols = {
    contradicting_chunks: "'[]'",
    last_challenged: 'NULL',
    last_reinforced: 'NULL',
    updated_at: "datetime('now', '-8 days')",
    confidence: '0.7',
    ...overrides,
  };
  db.exec(`
    INSERT INTO opinions (id, belief, confidence, supporting_chunks, contradicting_chunks,
                          domain, formed_at, last_reinforced, last_challenged, updated_at)
    VALUES ('${id}', 'Tom prefers self-hosted infrastructure', ${cols.confidence}, '["c1"]',
            ${cols.contradicting_chunks}, 'infrastructure', datetime('now', '-60 days'),
            ${cols.last_reinforced}, ${cols.last_challenged}, ${cols.updated_at})
  `);
  db.close();
  return id;
}

/** Run a reflect cycle against an empty backlog — only step-0 decay executes. */
async function decayOnlyCycle(dbPath: string) {
  const { fetchFn } = mockFetchSequence([reflectResponse([])]);
  vi.stubGlobal('fetch', fetchFn);
  return reflect({
    counterEvidence: false,
    opinionGates: false,
    dbPath,
    reflectModel: 'llama-test',
  });
}

describe('falsifier field (would_change_this)', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('stores the stated falsifier on the formed opinion and projects it via introspect', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, FACTS);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsFormed).toBe(1);

    const [opinion] = getOpinions(dbPath);
    expect(opinion.would_change_this).toBe(FALSIFIER);

    const db = new Database(dbPath);
    const view = introspect(db, 'self-hosted');
    db.close();
    expect(view.opinions[0].wouldChangeThis).toBe(FALSIFIER);
  });

  it('stores NULL when the model omits the field', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, FACTS);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0]], { would_change_this: undefined })]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    const [opinion] = getOpinions(dbPath);
    expect(opinion.would_change_this).toBeNull();
  });

  it("surfaces the falsifier in the next cycle's reflect prompt", async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, FACTS);
    const first = mockFetchSequence([reflectResponse([newOpinion([ids[0]])])]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    await seedFacts(
      dbPath,
      [
        'Tom compared Hetzner and homelab costs',
        'Tom updated the Proxmox cluster firmware',
        'Tom reviewed the NAS backup strategy',
        'Tom tested a cloud burst configuration',
        'Tom documented the homelab network layout',
      ],
      false,
    );
    const second = mockFetchSequence([reflectResponse([])]);
    vi.stubGlobal('fetch', second.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    expect(second.prompts[0]).toContain(`(would change if: ${FALSIFIER})`);
  });

  it('backfills a missing falsifier on reinforcement but never overwrites one', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, FACTS);
    // Cycle 1: formation WITHOUT a falsifier.
    const first = mockFetchSequence([
      reflectResponse([newOpinion([ids[0]], { would_change_this: undefined })]),
    ]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(getOpinions(dbPath)[0].would_change_this).toBeNull();

    // Cycle 2: reinforcement states one → backfilled.
    const more = await seedFacts(
      dbPath,
      [
        'Tom kept the new service on the homelab',
        'Tom expanded local storage instead of using S3',
        'Tom wrote a self-hosting cost analysis',
        'Tom declined the managed database quote',
        'Tom automated homelab provisioning',
      ],
      false,
    );
    const second = mockFetchSequence([
      reflectResponse([
        newOpinion([more[0]], {
          direction: 'reinforce',
          would_change_this: FALSIFIER,
        }),
      ]),
    ]);
    vi.stubGlobal('fetch', second.fetchFn);
    const r2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(r2.opinionsReinforced).toBe(1);
    expect(getOpinions(dbPath)[0].would_change_this).toBe(FALSIFIER);

    // Cycle 3: a different statement does NOT overwrite the recorded one.
    const evenMore = await seedFacts(
      dbPath,
      [
        'Tom renewed the homelab UPS batteries',
        'Tom scripted MinIO bucket replication',
        'Tom moved monitoring onto the cluster',
        'Tom benchmarked local inference hosts',
        'Tom rejected a SaaS observability pitch',
      ],
      false,
    );
    const third = mockFetchSequence([
      reflectResponse([
        newOpinion([evenMore[0]], {
          direction: 'reinforce',
          would_change_this: 'A completely different falsifier',
        }),
      ]),
    ]);
    vi.stubGlobal('fetch', third.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(getOpinions(dbPath)[0].would_change_this).toBe(FALSIFIER);
  });

  it('shows the stated falsifier to the counter-evidence judge for reinforcement candidates', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, FACTS);
    const first = mockFetchSequence([reflectResponse([newOpinion([ids[0]])])]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const more = await seedFacts(
      dbPath,
      [
        'Tom provisioned another homelab node',
        'Tom moved the queue broker off a managed plan',
        'Tom praised local-first inference latency',
        'Tom hardened the reverse proxy config',
        'Tom archived the cloud migration proposal',
      ],
      false,
    );
    const second = mockFetchSequence([
      reflectResponse([newOpinion([more[0]], { direction: 'reinforce' })]),
      JSON.stringify({ verdicts: [] }),
    ]);
    vi.stubGlobal('fetch', second.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      counterEvidence: { onReinforce: true },
    });

    expect(second.prompts).toHaveLength(2);
    expect(second.prompts[1]).toContain(`Stated falsifier: ${FALSIFIER}`);
  });
});

describe('weakened decay (unanswered contradictions)', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('decays a contradicted, unanswered opinion without waiting out the idle window, journaled weakened', async () => {
    dbPath = tmpDbPath();
    const db = new Database(dbPath);
    loadSchema(db);
    db.close();
    // Reinforced 10 days ago (inside the 30-day idle window — plain idle
    // decay would NOT fire), challenged 8 days ago, never re-reinforced.
    const opId = seedOpinion(dbPath, {
      contradicting_chunks: `'["contra-1", "contra-2"]'`,
      last_reinforced: "datetime('now', '-10 days')",
      last_challenged: "datetime('now', '-8 days')",
    });

    const result = await decayOnlyCycle(dbPath);
    expect(result.opinionsWeakened).toBe(1);

    const [opinion] = getOpinions(dbPath);
    expect(opinion.confidence).toBeCloseTo(0.68, 5);

    const [row] = getJournal(dbPath, 'weakened');
    expect(row.opinion_id).toBe(opId);
    expect(row.reflect_run_id).toBe(result.logId);
    expect(JSON.parse(row.contradicting_chunks)).toEqual([
      'contra-1',
      'contra-2',
    ]);
    const gateResults = JSON.parse(row.gate_results);
    expect(gateResults.reason).toBe('unanswered_contradictions');
    expect(gateResults.contradicting_count).toBe(2);
    expect(row.rationale).toContain('2 recorded contradiction(s)');
  });

  it('stops decaying once a reinforcement answers the challenge', async () => {
    dbPath = tmpDbPath();
    const db = new Database(dbPath);
    loadSchema(db);
    db.close();
    seedOpinion(dbPath, {
      contradicting_chunks: `'["contra-1"]'`,
      last_challenged: "datetime('now', '-20 days')",
      last_reinforced: "datetime('now', '-2 days')", // answered AFTER the challenge
    });

    const result = await decayOnlyCycle(dbPath);
    expect(result.opinionsWeakened).toBe(0);
    expect(getOpinions(dbPath)[0].confidence).toBeCloseTo(0.7, 5);
    expect(getJournal(dbPath, 'weakened')).toHaveLength(0);
  });

  it('journals nothing for plain idle decay', async () => {
    dbPath = tmpDbPath();
    const db = new Database(dbPath);
    loadSchema(db);
    db.close();
    // No contradictions; idle 40+ days → arm (a) decays it, unjournaled.
    seedOpinion(dbPath, {
      last_reinforced: "datetime('now', '-40 days')",
    });

    const result = await decayOnlyCycle(dbPath);
    expect(result.opinionsWeakened).toBe(0);
    expect(getOpinions(dbPath)[0].confidence).toBeCloseTo(0.68, 5);
    expect(getJournal(dbPath)).toHaveLength(0);
  });

  it('respects the 7-day throttle for contradicted opinions', async () => {
    dbPath = tmpDbPath();
    const db = new Database(dbPath);
    loadSchema(db);
    db.close();
    seedOpinion(dbPath, {
      contradicting_chunks: `'["contra-1"]'`,
      last_challenged: "datetime('now', '-1 days')",
      updated_at: "datetime('now', '-1 days')", // touched yesterday
    });

    const result = await decayOnlyCycle(dbPath);
    expect(result.opinionsWeakened).toBe(0);
    expect(getOpinions(dbPath)[0].confidence).toBeCloseTo(0.7, 5);
  });
});

describe('migration', () => {
  let dbPath: string;

  afterEach(() => {
    cleanupDb(dbPath);
  });

  it('adds would_change_this to a pre-existing .engram file on open', async () => {
    dbPath = tmpDbPath();
    const db = new Database(dbPath);
    loadSchema(db);
    // Simulate a pre-item-3 file.
    db.exec('ALTER TABLE opinions DROP COLUMN would_change_this');
    const before = db.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    expect(before.some((c) => c.name === 'would_change_this')).toBe(false);
    db.close();

    const engram = await Engram.open(dbPath, {
      embedder: new MockEmbedder(),
    });
    const after = engram.db.pragma('table_info(opinions)') as Array<{
      name: string;
    }>;
    expect(after.some((c) => c.name === 'would_change_this')).toBe(true);
    engram.close();
  });
});
