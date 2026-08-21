// =============================================================================
// belief-journal.test.ts — issue #38 items 1+4: opinion formation gates +
// per-belief audit journal.
//
// Covers: journaling parity when no gates are configured (formed/reinforced/
// challenged), gate rejection on each threshold (count / distinct days /
// distinct sources), hallucinated-evidence verification, prior-rejection
// evidence merging across cycles, the rejected-only-cycle semantics (facts
// marked reflected, no shrink hint, not a silent failure), unmatched
// reinforce/challenge journaling, the read API, and old-file migration.
// =============================================================================

import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect, getBeliefJournal } from '../src/reflect.js';
import { Engram } from '../src/engram.js';
import {
  MockEmbedder,
  loadSchema,
  tmpDbPath,
  cleanupDb,
  mockOllamaFetch,
} from './helpers.js';

const embedder = new MockEmbedder();

/** Retain `factCount` distinct facts; returns their chunk ids in order. */
async function seedFacts(
  path: string,
  factCount: number,
  opts: { source?: string } = {},
): Promise<string[]> {
  const db = new Database(path);
  loadSchema(db);
  const ids: string[] = [];
  for (let i = 0; i < factCount; i++) {
    const result = await retain(
      db,
      `Alice prefers Rust for systems work — fact ${i} ${Math.random().toString(36).slice(2)}`,
      embedder,
      {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
        ...(opts.source ? { source: opts.source } : {}),
      },
    );
    ids.push(result.chunkId);
  }
  db.close();
  return ids;
}

/** Additional facts into an existing db (fresh unreflected batch for cycle 2). */
async function seedMoreFacts(
  path: string,
  factCount: number,
): Promise<string[]> {
  const db = new Database(path);
  const ids: string[] = [];
  for (let i = 0; i < factCount; i++) {
    const result = await retain(
      db,
      `Alice keeps choosing Rust — later fact ${i} ${Math.random().toString(36).slice(2)}`,
      embedder,
      { memoryType: 'world', sourceType: 'user_stated', trustScore: 0.8 },
    );
    ids.push(result.chunkId);
  }
  db.close();
  return ids;
}

function opinionUpdateResponse(
  updates: Array<Record<string, unknown>>,
): string {
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
    rationale: 'Multiple stated preferences point the same way',
    ...overrides,
  };
}

function allJournalRows(path: string): any[] {
  const db = new Database(path);
  const rows = db
    .prepare('SELECT * FROM belief_journal ORDER BY rowid ASC')
    .all() as any[];
  db.close();
  return rows;
}

describe('belief journal — journaling without gates', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('journals a formed opinion with rationale and opinion_id', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsRejected).toBe(0);

    const rows = allJournalRows(dbPath);
    expect(rows).toHaveLength(1);
    expect(rows[0].action).toBe('formed');
    expect(rows[0].reflect_run_id).toBe(result.logId);
    expect(rows[0].candidate_belief).toContain('Alice');
    expect(rows[0].domain).toBe('preferences');
    expect(rows[0].rationale).toContain('stated preferences');
    expect(JSON.parse(rows[0].supporting_chunks)).toEqual([ids[0], ids[1]]);
    expect(rows[0].gate_results).toBeNull(); // no gates configured

    const db = new Database(dbPath);
    const opinion = db.prepare('SELECT id FROM opinions').get() as {
      id: string;
    };
    db.close();
    expect(rows[0].opinion_id).toBe(opinion.id);
  });

  it('journals reinforcement of an existing opinion (including new-dedup-to-reinforce)', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0]])])),
    );
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    // Cycle 2: the model re-derives the same belief as "new" — dedups into a
    // reinforcement, which must journal as 'reinforced', not 'formed'.
    const moreIds = await seedMoreFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([moreIds[0]])])),
    );
    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result2.opinionsReinforced).toBe(1);

    const rows = allJournalRows(dbPath);
    expect(rows.map((r) => r.action)).toEqual(['formed', 'reinforced']);
    expect(rows[1].opinion_id).toBe(rows[0].opinion_id);
    expect(rows[1].reflect_run_id).toBe(result2.logId);
  });

  it('journals a challenge with the contradicting evidence', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0]])])),
    );
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const moreIds = await seedMoreFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        opinionUpdateResponse([
          newOpinion([moreIds[0]], {
            direction: 'challenge',
            confidence_delta: -0.1,
            rationale: 'Recent evidence cuts against this',
          }),
        ]),
      ),
    );
    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result2.opinionsChallenged).toBe(1);

    const rows = allJournalRows(dbPath);
    expect(rows[1].action).toBe('challenged');
    expect(JSON.parse(rows[1].contradicting_chunks)).toEqual([moreIds[0]]);
    expect(JSON.parse(rows[1].supporting_chunks)).toEqual([]);
  });

  it('adopts an unmatched reinforce as a formation candidate, and journals an unmatched challenge as no_matching_opinion', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        opinionUpdateResponse([
          newOpinion([ids[0]], { direction: 'reinforce' }),
          newOpinion([ids[1]], {
            belief: 'Alice distrusts garbage-collected languages',
            direction: 'challenge',
            confidence_delta: -0.1,
          }),
        ]),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });
    // No matching opinion exists for either verdict, but they are not
    // symmetric. A `reinforce` asserts a proposition, so it is adopted as a
    // formation candidate rather than discarded; a `challenge` only names a
    // belief to weaken, and with no such belief there is nothing to adopt.
    expect(result.opinionsReinforced).toBe(0);
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsChallenged).toBe(0);
    // Not gate rejections: the gate counter stays 0.
    expect(result.opinionsRejected).toBe(0);

    const rows = allJournalRows(dbPath);
    expect(rows).toHaveLength(2);
    expect(rows[0].action).toBe('formed');
    expect(rows[0].opinion_id).not.toBeNull();
    expect(JSON.parse(rows[0].supporting_chunks)).toEqual([ids[0]]);

    expect(rows[1].action).toBe('rejected');
    expect(rows[1].opinion_id).toBeNull();
    expect(JSON.parse(rows[1].gate_results).reason).toBe('no_matching_opinion');
    // Challenge evidence lands on the contradicting side.
    expect(JSON.parse(rows[1].contradicting_chunks)).toEqual([ids[1]]);

    // The model engaged (parseable verdicts) — this is not a context-size
    // failure: the batch is consumed, no #17 shrink hint, status completed.
    // Otherwise the same verdicts would re-journal as duplicates every cycle.
    expect(result.status).toBe('completed');
    const db = new Database(dbPath);
    const unreflected = (
      db
        .prepare('SELECT COUNT(*) AS n FROM chunks WHERE reflected_at IS NULL')
        .get() as { n: number }
    ).n;
    const shrinkHint = db
      .prepare(`SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`)
      .get();
    db.close();
    expect(unreflected).toBe(0);
    expect(shrinkHint).toBeUndefined();
  });

  it('holds an adopted unmatched reinforce to the same formation gates', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        opinionUpdateResponse([
          newOpinion([ids[0]], { direction: 'reinforce' }),
        ]),
      ),
    );

    // Adoption is not a bypass: one same-day chunk fails the default-shaped
    // gates exactly as an equivalent `direction: 'new'` candidate would.
    const result = await reflect({
      counterEvidence: false,
      opinionGates: { minEvidenceCount: 3, minDistinctDays: 2 },
      dbPath,
      reflectModel: 'llama-test',
    });
    expect(result.opinionsFormed).toBe(0);
    expect(result.opinionsRejected).toBe(1);

    const rows = allJournalRows(dbPath);
    expect(rows).toHaveLength(1);
    expect(rows[0].action).toBe('rejected');
    const gateResults = JSON.parse(rows[0].gate_results);
    expect(gateResults.reason).toBe('insufficient_evidence');
    expect(gateResults.gates.min_evidence_count.pass).toBe(false);
  });
});

describe('belief journal — formation gates', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('rejects below minEvidenceCount, journals measurements, and still completes the cycle', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0]])])),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(result.opinionsFormed).toBe(0);
    expect(result.opinionsRejected).toBe(1);
    expect(result.status).toBe('completed'); // NOT 'partial' — the model engaged

    const db = new Database(dbPath);
    const opinionCount = (
      db.prepare('SELECT COUNT(*) AS n FROM opinions').get() as { n: number }
    ).n;
    const unreflected = (
      db
        .prepare('SELECT COUNT(*) AS n FROM chunks WHERE reflected_at IS NULL')
        .get() as { n: number }
    ).n;
    const shrinkHint = db
      .prepare(`SELECT value FROM bank_config WHERE key = 'reflect_batch_hint'`)
      .get();
    db.close();
    expect(opinionCount).toBe(0);
    expect(unreflected).toBe(0); // batch consumed — rejection is an outcome, not a failure
    expect(shrinkHint).toBeUndefined(); // no #17 shrink for a gate-rejected cycle

    const rows = allJournalRows(dbPath);
    expect(rows).toHaveLength(1);
    expect(rows[0].action).toBe('rejected');
    const gateResults = JSON.parse(rows[0].gate_results);
    expect(gateResults.reason).toBe('insufficient_evidence');
    expect(gateResults.gates.min_evidence_count).toEqual({
      required: 2,
      measured: 1,
      pass: false,
    });
  });

  it('does not let hallucinated evidence ids pass a gate', async () => {
    dbPath = tmpDbPath();
    await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        opinionUpdateResponse([
          newOpinion(['made-up-1', 'made-up-2', 'made-up-3']),
        ]),
      ),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(result.opinionsRejected).toBe(1);

    const rows = allJournalRows(dbPath);
    const gateResults = JSON.parse(rows[0].gate_results);
    expect(gateResults.gates.min_evidence_count.measured).toBe(0);
  });

  it('forms when gates pass, carrying gate measurements in the journal', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsRejected).toBe(0);

    const rows = allJournalRows(dbPath);
    expect(rows[0].action).toBe('formed');
    const gateResults = JSON.parse(rows[0].gate_results);
    expect(gateResults.reason).toBeUndefined();
    expect(gateResults.gates.min_evidence_count.pass).toBe(true);
    expect(gateResults.merged_prior_rejection).toBeNull();
  });

  it('enforces minDistinctDays on date(COALESCE(event_time, created_at))', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);

    // All facts retained "today": two cited chunks → 1 distinct day → reject.
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );
    const rejected = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 0, minDistinctDays: 2 },
    });
    expect(rejected.opinionsRejected).toBe(1);
    const rejectedRow = allJournalRows(dbPath)[0];
    expect(
      JSON.parse(rejectedRow.gate_results).gates.min_distinct_days,
    ).toEqual({ required: 2, measured: 1, pass: false });

    // Backdate one cited chunk, seed a fresh batch, re-derive: 2 distinct days → form.
    const db = new Database(dbPath);
    db.prepare(
      `UPDATE chunks SET created_at = '2026-07-01 10:00:00' WHERE id = ?`,
    ).run(ids[0]);
    db.close();
    await seedMoreFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );
    const formed = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 0, minDistinctDays: 2 },
    });
    expect(formed.opinionsFormed).toBe(1);
  });

  it('enforces minDistinctSources, bucketing NULL sources as one source', async () => {
    dbPath = tmpDbPath();
    // No source option → chunks carry NULL source → one bucket.
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );
    const rejected = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: {
        minEvidenceCount: 0,
        minDistinctDays: 0,
        minDistinctSources: 2,
      },
    });
    expect(rejected.opinionsRejected).toBe(1);
    expect(
      JSON.parse(allJournalRows(dbPath)[0].gate_results).gates
        .min_distinct_sources,
    ).toEqual({ required: 2, measured: 1, pass: false });

    // Give one cited chunk a real source: NULL-bucket + 'conversation:a' → 2.
    const db = new Database(dbPath);
    db.prepare(`UPDATE chunks SET source = 'conversation:a' WHERE id = ?`).run(
      ids[0],
    );
    db.close();
    await seedMoreFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );
    const formed = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: {
        minEvidenceCount: 0,
        minDistinctDays: 0,
        minDistinctSources: 2,
      },
    });
    expect(formed.opinionsFormed).toBe(1);
  });

  it('merges evidence from a prior rejection so slow-accumulating beliefs eventually form', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);

    // Cycle 1: one evidence chunk — rejected under minEvidenceCount 2.
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0]])])),
    );
    const cycle1 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(cycle1.opinionsRejected).toBe(1);
    const rejectionId = allJournalRows(dbPath)[0].id;

    // Cycle 2: ONE new evidence chunk for the same belief. Alone it would be
    // rejected again; unioned with the journaled prior evidence it passes.
    const moreIds = await seedMoreFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([moreIds[0]])])),
    );
    const cycle2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(cycle2.opinionsFormed).toBe(1);
    expect(cycle2.opinionsRejected).toBe(0);

    const db = new Database(dbPath);
    const opinion = db
      .prepare('SELECT supporting_chunks, evidence_count FROM opinions')
      .get() as { supporting_chunks: string; evidence_count: number };
    db.close();
    const supporting = JSON.parse(opinion.supporting_chunks);
    // The formed opinion carries the union, not just this cycle's citation.
    expect(supporting).toEqual(expect.arrayContaining([ids[0], moreIds[0]]));
    expect(opinion.evidence_count).toBe(2);

    const formedRow = allJournalRows(dbPath)[1];
    expect(formedRow.action).toBe('formed');
    expect(JSON.parse(formedRow.gate_results).merged_prior_rejection).toBe(
      rejectionId,
    );
  });
});

describe('belief journal — read API', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('filters by action, run id, and opinion id; parses JSON fields; respects limit', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(
        opinionUpdateResponse([
          newOpinion([ids[0], ids[1]]),
          newOpinion([ids[2]], {
            belief: 'Alice avoids dynamic typing for production systems',
            rationale: 'Only one weak signal so far',
          }),
        ]),
      ),
    );
    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      opinionGates: { minEvidenceCount: 2, minDistinctDays: 0 },
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.opinionsRejected).toBe(1);

    const db = new Database(dbPath);
    const all = getBeliefJournal(db);
    expect(all).toHaveLength(2);

    const rejectedOnly = getBeliefJournal(db, { action: 'rejected' });
    expect(rejectedOnly).toHaveLength(1);
    expect(rejectedOnly[0].candidateBelief).toContain('dynamic typing');
    expect(rejectedOnly[0].opinionId).toBeNull();
    expect(rejectedOnly[0].gateResults?.reason).toBe('insufficient_evidence');
    expect(rejectedOnly[0].supportingChunks).toEqual([ids[2]]);

    const byRun = getBeliefJournal(db, { reflectRunId: result.logId });
    expect(byRun).toHaveLength(2);
    expect(getBeliefJournal(db, { reflectRunId: 'nonexistent' })).toHaveLength(
      0,
    );

    const formed = getBeliefJournal(db, { action: 'formed' });
    const byOpinion = getBeliefJournal(db, {
      opinionId: formed[0].opinionId!,
    });
    expect(byOpinion).toHaveLength(1);
    expect(byOpinion[0].action).toBe('formed');

    expect(getBeliefJournal(db, { limit: 1 })).toHaveLength(1);
    db.close();
  });

  it('is reachable from the Engram class', async () => {
    dbPath = tmpDbPath();
    const ids = await seedFacts(dbPath, 5);
    vi.stubGlobal(
      'fetch',
      mockOllamaFetch(opinionUpdateResponse([newOpinion([ids[0], ids[1]])])),
    );
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
    });

    const engram = await Engram.open(dbPath, { embedder: new MockEmbedder() });
    const rows = engram.beliefJournal({ action: 'formed' });
    engram.close();
    expect(rows).toHaveLength(1);
    expect(rows[0].candidateBelief).toContain('Alice');
  });
});

describe('belief journal — migration', () => {
  let dbPath: string;

  afterEach(() => {
    cleanupDb(dbPath);
  });

  it('Engram.open adds belief_journal to a pre-existing file that lacks it', async () => {
    dbPath = tmpDbPath();
    // Simulate a pre-#38 .engram: full schema, then drop the new table.
    const db = new Database(dbPath);
    loadSchema(db);
    db.exec('DROP INDEX IF EXISTS idx_belief_journal_run');
    db.exec('DROP INDEX IF EXISTS idx_belief_journal_opinion');
    db.exec('DROP INDEX IF EXISTS idx_belief_journal_action');
    db.exec('DROP TABLE IF EXISTS belief_journal');
    db.close();

    const engram = await Engram.open(dbPath, { embedder: new MockEmbedder() });
    expect(engram.beliefJournal()).toEqual([]); // table exists again, empty
    engram.close();
  });
});
