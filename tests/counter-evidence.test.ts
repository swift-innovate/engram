// =============================================================================
// counter-evidence.test.ts — issue #38 item 2: the active counter-evidence
// pass. Before forming (and optionally reinforcing) an opinion, related
// chunks are retrieved from the whole store and one batched judge call
// classifies contradictions.
//
// Covers: off-by-default (no judge call), sub-threshold contradictions
// recorded at formation, ratio-based formation blocking, record-only mode,
// onReinforce evidence recording, hallucinated-judge-id filtering, fail-open
// judge failure, cited-evidence exclusion from the judge pool, the
// contradicted-count line in the next cycle's prompt, and the
// missing-embedder skip.
// =============================================================================

import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect } from '../src/reflect.js';
import { MockEmbedder, loadSchema, tmpDbPath, cleanupDb } from './helpers.js';

const embedder = new MockEmbedder();

/**
 * Sequenced fetch mock: call N gets responses[N] (last response repeats).
 * Captures each request's prompt so tests can assert on judge-prompt content.
 */
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

const SUPPORT_TEXTS = [
  'Alice praised Rust for systems programming reliability',
  'Alice shipped the parser rewrite in Rust and was happy with it',
  'Alice recommended Rust to the platform team',
  'Alice said Rust tooling has been solid for her',
];
const CONTRA_TEXT =
  'Alice said she is fed up with Rust compile times and is moving new services to Go';

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
    belief: 'Alice strongly prefers Rust over other systems languages',
    direction: 'new',
    confidence_delta: 0.1,
    domain: 'preferences',
    evidence_chunk_ids: evidenceIds,
    entity_names: ['Alice'],
    rationale: 'Repeated positive statements about Rust',
    ...overrides,
  };
}

function judgeResponse(
  verdicts: Array<{ index: number; ids: string[]; reason?: string }>,
): string {
  return JSON.stringify({
    verdicts: verdicts.map((v) => ({
      candidate_index: v.index,
      contradicting_chunk_ids: v.ids,
      reason: v.reason ?? 'Directly cuts against the stated preference',
    })),
  });
}

function getOpinions(path: string): any[] {
  const db = new Database(path);
  const rows = db.prepare('SELECT * FROM opinions').all() as any[];
  db.close();
  return rows;
}

function getJournal(path: string): any[] {
  const db = new Database(path);
  const rows = db
    .prepare('SELECT * FROM belief_journal ORDER BY rowid ASC')
    .all() as any[];
  db.close();
  return rows;
}

describe('counter-evidence pass', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('runs the default audit but rejects one-day evidence through safe gates', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn, prompts } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      judgeResponse([{ index: 0, ids: [] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
    });
    expect(result.opinionsFormed).toBe(0);
    expect(result.opinionsRejected).toBe(1);
    expect(result.counterEvidenceChecked).toBe(1);
    expect(prompts).toHaveLength(2);

    const journal = getJournal(dbPath);
    expect(JSON.parse(journal[0].gate_results).gates).toBeDefined();
  });

  it('records sub-threshold contradictions on the formed opinion and in the journal', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const contraId = ids[4];
    const { fetchFn, prompts } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      judgeResponse([{ index: 0, ids: [contraId] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });
    // ratio 1/(2+1) = 0.33 < 0.5 → forms, contradictions ride along
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsRejected).toBe(0);
    expect(result.counterEvidenceChecked).toBe(1);
    expect(prompts).toHaveLength(2);

    const [opinion] = getOpinions(dbPath);
    expect(JSON.parse(opinion.contradicting_chunks)).toEqual([contraId]);
    expect(opinion.last_challenged).not.toBeNull();

    const [row] = getJournal(dbPath);
    expect(row.action).toBe('formed');
    expect(JSON.parse(row.contradicting_chunks)).toEqual([contraId]);
    const ce = JSON.parse(row.gate_results).counter_evidence;
    expect(ce.checked).toBe(true);
    expect(ce.contradicting_count).toBe(1);
    expect(ce.reason).toContain('cuts against');
  });

  it('blocks formation when contradictions outweigh support', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0]])]),
      judgeResponse([{ index: 0, ids: [ids[4], ids[3]] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });
    // ratio 2/(1+2) = 0.67 > 0.5 → blocked
    expect(result.opinionsFormed).toBe(0);
    expect(result.opinionsRejected).toBe(1);
    expect(result.status).toBe('completed');
    expect(getOpinions(dbPath)).toHaveLength(0);

    const [row] = getJournal(dbPath);
    expect(row.action).toBe('rejected');
    const gateResults = JSON.parse(row.gate_results);
    expect(gateResults.reason).toBe('counter_evidence');
    expect(gateResults.counter_evidence.ratio).toBeCloseTo(0.667, 2);
    expect(gateResults.counter_evidence.threshold).toBe(0.5);
    expect(JSON.parse(row.contradicting_chunks)).toEqual([ids[4], ids[3]]);

    // Rejection is an outcome, not a failure: batch consumed.
    const db = new Database(dbPath);
    const unreflected = (
      db
        .prepare('SELECT COUNT(*) AS n FROM chunks WHERE reflected_at IS NULL')
        .get() as { n: number }
    ).n;
    db.close();
    expect(unreflected).toBe(0);
  });

  it('record-only mode (maxContradictionRatio: 1) never blocks', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0]])]),
      judgeResponse([{ index: 0, ids: [ids[4], ids[3]] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: { maxContradictionRatio: 1 },
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsRejected).toBe(0);
    const [opinion] = getOpinions(dbPath);
    expect(JSON.parse(opinion.contradicting_chunks)).toHaveLength(2);
  });

  it('onReinforce records contradictions on the reinforced opinion without blocking', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    // Cycle 1: form the opinion, counter-evidence off.
    const first = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
    ]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: false,
    });
    const confidenceBefore = getOpinions(dbPath)[0].confidence;

    // Cycle 2: reinforce; judge finds a contradiction.
    const moreIds = await seedFacts(
      dbPath,
      [
        'Alice endorsed Rust again in the retro',
        'Alice merged another Rust service',
        'Alice mentored a junior on Rust idioms',
        'Alice defaulted to Rust for the new daemon',
        'Alice wrote the RFC in favor of Rust adoption',
      ],
      false,
    );
    const second = mockFetchSequence([
      reflectResponse([
        newOpinion([moreIds[0]], {
          direction: 'reinforce',
          confidence_delta: 0.1,
        }),
      ]),
      judgeResponse([{ index: 0, ids: [ids[4]] }]),
    ]);
    vi.stubGlobal('fetch', second.fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: { onReinforce: true },
    });
    expect(result.opinionsReinforced).toBe(1);
    expect(result.counterEvidenceChecked).toBe(1);

    const [opinion] = getOpinions(dbPath);
    expect(opinion.confidence).toBeGreaterThan(confidenceBefore); // reinforcement applied
    expect(JSON.parse(opinion.contradicting_chunks)).toEqual([ids[4]]);
    expect(opinion.last_challenged).not.toBeNull();

    const journal = getJournal(dbPath);
    const reinforcedRow = journal[journal.length - 1];
    expect(reinforcedRow.action).toBe('reinforced');
    expect(JSON.parse(reinforcedRow.contradicting_chunks)).toEqual([ids[4]]);
    expect(
      JSON.parse(reinforcedRow.gate_results).counter_evidence.checked,
    ).toBe(true);
  });

  it('drops judge-cited ids that were not in the retrieved pool', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      judgeResponse([{ index: 0, ids: ['hallucinated-chunk-id'] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });
    expect(result.opinionsFormed).toBe(1);
    const [opinion] = getOpinions(dbPath);
    expect(JSON.parse(opinion.contradicting_chunks)).toEqual([]);
    expect(opinion.last_challenged).toBeNull();
  });

  it('fails open only when explicitly requested, journaling the candidate as unchecked', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      'this is not json at all',
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: { failOpen: true },
    });
    expect(result.opinionsFormed).toBe(1); // insights not lost
    expect(result.counterEvidenceChecked).toBe(0);

    const [row] = getJournal(dbPath);
    expect(row.action).toBe('formed');
    expect(JSON.parse(row.gate_results).counter_evidence.checked).toBe(false);
  });

  it('fails closed by default when the judge output is invalid and retries its facts', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      'this is not json at all',
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });
    expect(result.status).toBe('partial');
    expect(result.opinionsFormed).toBe(0);
    expect(result.error).toContain('judge');
    const [row] = getJournal(dbPath);
    expect(JSON.parse(row.gate_results).reason).toBe(
      'counter_evidence_unavailable',
    );

    const db = new Database(dbPath);
    const unreflected = (
      db
        .prepare(
          'SELECT COUNT(*) AS count FROM chunks WHERE reflected_at IS NULL',
        )
        .get() as { count: number }
    ).count;
    db.close();
    expect(unreflected).toBe(ids.length);
  });

  it("excludes the candidate's own cited evidence from the judge pool", async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const citedId = ids[0];
    const { fetchFn, prompts } = mockFetchSequence([
      reflectResponse([newOpinion([citedId])]),
      judgeResponse([{ index: 0, ids: [] }]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });
    expect(prompts).toHaveLength(2);
    const judgePrompt = prompts[1];
    expect(judgePrompt).toContain('counter-evidence auditor');
    expect(judgePrompt).not.toContain(citedId); // own evidence excluded
    expect(judgePrompt).toContain(ids[4]); // other chunks are in the pool
  });

  it("surfaces an opinion's contradiction count in the next cycle's reflect prompt", async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const first = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
      judgeResponse([{ index: 0, ids: [ids[4]] }]),
    ]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      opinionGates: false,
      counterEvidence: {},
    });

    await seedFacts(
      dbPath,
      [
        'Alice tried Zig on a side project',
        'Alice benchmarked Go against Rust',
        'Alice paired with Bob on the Rust service',
        'Alice reviewed the Go migration RFC',
        'Alice updated the Rust style guide',
      ],
      false,
    );
    const second = mockFetchSequence([reflectResponse([])]);
    vi.stubGlobal('fetch', second.fetchFn);
    await reflect({ dbPath, reflectModel: 'llama-test', embedder });

    expect(second.prompts[0]).toContain('contradicted by 1 chunk(s)');
  });

  it('fails clearly without an embedder unless counter-evidence is explicitly disabled', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, [...SUPPORT_TEXTS, CONTRA_TEXT]);
    const { fetchFn, prompts } = mockFetchSequence([
      reflectResponse([newOpinion([ids[0], ids[1]])]),
    ]);
    vi.stubGlobal('fetch', fetchFn);
    await expect(
      reflect({ dbPath, reflectModel: 'llama-test', opinionGates: false }),
    ).rejects.toThrow('requires an embedder');
    expect(prompts).toHaveLength(0);

    const result = await reflect({
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: false,
      counterEvidence: false,
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.counterEvidenceChecked).toBe(0);
    expect(prompts).toHaveLength(1);

    const [row] = getJournal(dbPath);
    expect(row.gate_results).toBeNull();
  });
});
