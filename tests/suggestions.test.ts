// =============================================================================
// suggestions.test.ts — issue #39: the procedural-suggestion pass. A third
// insight kind alongside observations/opinions: "this recurring pattern
// (corrections, tool friction, repeated workflows) would benefit from being
// codified as a skill/rule/workflow/config."
//
// Covers: omission is byte-identical, formation/gate/dedup/merge-forward/
// dismissed-reopen mechanics (mirroring issue #38's belief_journal template),
// watermark fail-open semantics, signal-scan correctness, and reflectCatchUp
// aggregation. MCP/CLI exposure is Slice 2 — this suite is library-only.
// =============================================================================

import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { retain } from '../src/retain.js';
import { reflect, reflectCatchUp } from '../src/reflect.js';
import { Engram } from '../src/engram.js';
import { groundSubagent } from '../src/grounding.js';
import {
  MockEmbedder,
  loadSchema,
  tmpDbPath,
  cleanupDb,
  REFLECT_RESPONSE,
} from './helpers.js';

const embedder = new MockEmbedder();

/**
 * Sequenced fetch mock: call N gets responses[N] (last response repeats).
 * Captures each request's prompt so tests can assert on prompt content and
 * on how many generate() calls actually happened. Mirrors
 * tests/counter-evidence.test.ts's helper of the same name.
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

function suggestResponse(suggestions: Array<Record<string, unknown>>): string {
  return JSON.stringify({ suggestions });
}

function newSuggestion(
  evidenceIds: string[],
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    kind: 'rule',
    summary: 'Always double-check deploy script paths after a correction',
    rationale:
      'The agent was corrected about the deploy script location multiple times',
    domain: 'workflow',
    evidence_chunk_ids: evidenceIds,
    ...overrides,
  };
}

function reflectResponse(updates: Array<Record<string, unknown>> = []): string {
  return JSON.stringify({
    observations: [],
    opinion_updates: updates,
    observation_refreshes: [],
  });
}

// ---------------------------------------------------------------------------
// Seeding helpers
// ---------------------------------------------------------------------------

async function seedWorldFacts(
  path: string,
  n: number,
  fresh = false,
): Promise<string[]> {
  const db = new Database(path);
  if (fresh) loadSchema(db);
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    const r = await retain(
      db,
      `Fact ${i}: the team prefers ${Math.random().toString(36).slice(2, 8)} for this task`,
      embedder,
      { memoryType: 'world', sourceType: 'user_stated', trustScore: 0.8 },
    );
    ids.push(r.chunkId);
  }
  db.close();
  return ids;
}

/**
 * Seed N correction events (a superseded "old" chunk + its replacement),
 * each old-chunk's `updated_at` pinned to `updatedAt` — deterministic
 * ordering across multi-cycle tests instead of relying on wall-clock
 * resolution to separate batches. The replacement is stamped `reflected_at`
 * immediately so these fixtures don't also pull the main opinion/observation
 * batch in (keeps each test to exactly the generate() calls it asserts on).
 */
async function seedCorrections(
  path: string,
  n: number,
  updatedAt: string,
  fresh = false,
): Promise<Array<{ oldId: string; newId: string }>> {
  const db = new Database(path);
  if (fresh) loadSchema(db);
  const out: Array<{ oldId: string; newId: string }> = [];
  for (let i = 0; i < n; i++) {
    const rand = Math.random().toString(36).slice(2, 8);
    const oldResult = await retain(
      db,
      `The deploy script lives at scripts/deploy-${rand}.sh`,
      embedder,
      { memoryType: 'world', sourceType: 'agent_generated', trustScore: 0.6 },
    );
    const newResult = await retain(
      db,
      `Correction: the deploy script actually lives at tools/deploy-${rand}.sh`,
      embedder,
      {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.9,
        supersedes: oldResult.chunkId,
      },
    );
    db.prepare(`UPDATE chunks SET updated_at = ? WHERE id = ?`).run(
      updatedAt,
      oldResult.chunkId,
    );
    db.prepare(
      `UPDATE chunks SET reflected_at = CURRENT_TIMESTAMP WHERE id = ?`,
    ).run(newResult.chunkId);
    out.push({ oldId: oldResult.chunkId, newId: newResult.chunkId });
  }
  db.close();
  return out;
}

async function seedFriction(
  path: string,
  text: string,
  fresh = false,
): Promise<string> {
  const db = new Database(path);
  if (fresh) loadSchema(db);
  const r = await retain(db, text, embedder, {
    memoryType: 'experience',
    sourceType: 'tool_result',
    trustScore: 0.5,
  });
  db.close();
  return r.chunkId;
}

async function seedWorkflow(
  path: string,
  text: string,
  fresh = false,
): Promise<string> {
  const db = new Database(path);
  if (fresh) loadSchema(db);
  const r = await retain(db, text, embedder, {
    memoryType: 'experience',
    sourceType: 'agent_generated',
    trustScore: 0.6,
  });
  db.close();
  return r.chunkId;
}

function backdateCreatedAt(
  path: string,
  chunkId: string,
  createdAt: string,
): void {
  const db = new Database(path);
  db.prepare(`UPDATE chunks SET created_at = ? WHERE id = ?`).run(
    createdAt,
    chunkId,
  );
  db.close();
}

function insertTaskScopeChunk(path: string, text: string): string {
  const db = new Database(path);
  const id = `chk-task-${Math.random().toString(36).slice(2, 10)}`;
  db.prepare(
    `INSERT INTO chunks (id, text, memory_type, source_type, scope, is_active) VALUES (?, ?, 'experience', 'agent_generated', 'task', 1)`,
  ).run(id, text);
  db.close();
  return id;
}

function markDismissed(path: string, suggestionId: string): void {
  const db = new Database(path);
  db.prepare(`UPDATE suggestions SET status = 'dismissed' WHERE id = ?`).run(
    suggestionId,
  );
  db.close();
}

function getSuggestionRows(path: string): any[] {
  const db = new Database(path);
  const rows = db
    .prepare('SELECT * FROM suggestions ORDER BY rowid ASC')
    .all() as any[];
  db.close();
  return rows;
}

function getSuggestionJournalRows(path: string): any[] {
  const db = new Database(path);
  const rows = db
    .prepare('SELECT * FROM suggestion_journal ORDER BY rowid ASC')
    .all() as any[];
  db.close();
  return rows;
}

function getBankConfigValue(path: string, key: string): string | undefined {
  const db = new Database(path);
  const row = db
    .prepare('SELECT value FROM bank_config WHERE key = ?')
    .get(key) as { value: string } | undefined;
  db.close();
  return row?.value;
}

// =============================================================================
// Tests
// =============================================================================

describe('procedural suggestions (issue #39)', () => {
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanupDb(dbPath);
  });

  it('1. omission is byte-identical: reflect() without `suggestions` never touches the new tables', async () => {
    dbPath = tmpDbPath();
    await seedWorldFacts(dbPath, 5, true);
    const { fetchFn, prompts } = mockFetchSequence([reflectResponse()]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
    });

    expect(prompts).toHaveLength(1); // main reflect call only
    expect(result.suggestionsProposed).toBe(0);
    expect(result.suggestionsReinforced).toBe(0);
    expect(result.suggestionsRejected).toBe(0);
    expect(getSuggestionRows(dbPath)).toHaveLength(0);
    expect(getSuggestionJournalRows(dbPath)).toHaveLength(0);
    expect(getBankConfigValue(dbPath, 'suggest_watermark')).toBeUndefined();
  });

  it('2. forms a new suggestion from evidence that clears the gate, fully stamped', async () => {
    dbPath = tmpDbPath();
    const corr = await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const candidate = newSuggestion([
      corr[0].oldId,
      corr[1].oldId,
      corr[2].oldId,
    ]);
    const { fetchFn, prompts } = mockFetchSequence([
      suggestResponse([candidate]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });

    expect(prompts).toHaveLength(1); // suggestion pass only — main batch stays empty
    expect(result.suggestionsProposed).toBe(1);
    expect(result.suggestionsReinforced).toBe(0);
    expect(result.suggestionsRejected).toBe(0);

    const [row] = getSuggestionRows(dbPath);
    expect(row.kind).toBe('rule');
    expect(row.summary).toBe(candidate.summary);
    expect(row.rationale).toBe(candidate.rationale);
    expect(JSON.parse(row.supporting_chunks).sort()).toEqual(
      [corr[0].oldId, corr[1].oldId, corr[2].oldId].sort(),
    );
    expect(row.evidence_count).toBe(3);
    // These fixtures seed via raw Database + loadSchema (not Engram.open()),
    // which is the only thing that mints node_origin — NULL is correct here,
    // not a gap (see CLAUDE.md's node-origin bullet: NULL = origin unknown).
    expect(row.node_origin).toBeNull();
    expect(row.status).toBe('proposed');

    const [journalRow] = getSuggestionJournalRows(dbPath);
    expect(journalRow.action).toBe('proposed');
    expect(journalRow.suggestion_id).toBe(row.id);
    expect(journalRow.reflect_run_id).toBe(result.logId);
  });

  it('3. gates reject thin evidence, journaling the shortfall with default gate keys present', async () => {
    dbPath = tmpDbPath();
    const corr = await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const candidate = newSuggestion([corr[0].oldId, corr[1].oldId]); // only 2, default requires 3
    const { fetchFn, prompts } = mockFetchSequence([
      suggestResponse([candidate]),
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {}, // default gates: { minEvidenceCount: 3, minDistinctDays: 2 }
    });

    expect(prompts).toHaveLength(1);
    expect(result.suggestionsRejected).toBe(1);
    expect(result.suggestionsProposed).toBe(0);
    expect(getSuggestionRows(dbPath)).toHaveLength(0);

    const [journalRow] = getSuggestionJournalRows(dbPath);
    expect(journalRow.action).toBe('rejected');
    expect(journalRow.suggestion_id).toBeNull();
    const gateResults = JSON.parse(journalRow.gate_results);
    expect(gateResults.reason).toBe('insufficient_evidence');
    expect(gateResults.gates.min_evidence_count).toEqual({
      required: 3,
      measured: 2,
      pass: false,
    });
    expect(gateResults.gates.min_distinct_days).toBeDefined(); // default gates check both
  });

  it('4. hallucinated evidence ids do not count toward the gate', async () => {
    dbPath = tmpDbPath();
    const corr = await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const candidate = newSuggestion([
      corr[0].oldId,
      corr[1].oldId,
      'chunk-does-not-exist',
    ]);
    const { fetchFn } = mockFetchSequence([suggestResponse([candidate])]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });

    expect(result.suggestionsRejected).toBe(1);
    expect(result.suggestionsProposed).toBe(0);

    const [journalRow] = getSuggestionJournalRows(dbPath);
    expect(
      JSON.parse(journalRow.gate_results).gates.min_evidence_count.measured,
    ).toBe(2);
  });

  it('5. inactive (superseded) evidence still counts toward the gate — requireActive:false', async () => {
    dbPath = tmpDbPath();
    const corr = await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const citedIds = [corr[0].oldId, corr[1].oldId, corr[2].oldId];
    const candidate = newSuggestion(citedIds);
    const { fetchFn } = mockFetchSequence([suggestResponse([candidate])]);
    vi.stubGlobal('fetch', fetchFn);

    // Confirm the cited evidence really is inactive before asserting it still counts.
    const checkDb = new Database(dbPath);
    const activeFlags = checkDb
      .prepare(
        `SELECT is_active FROM chunks WHERE id IN (${citedIds.map(() => '?').join(',')})`,
      )
      .all(...citedIds) as Array<{ is_active: number }>;
    checkDb.close();
    expect(activeFlags.every((r) => r.is_active === 0)).toBe(true);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });

    expect(result.suggestionsProposed).toBe(1);
    const [journalRow] = getSuggestionJournalRows(dbPath);
    expect(journalRow.gate_results).toBeNull(); // formed — no rejection journaled
  });

  it("6. a rejected candidate's evidence merges forward into a later cycle that re-derives it", async () => {
    dbPath = tmpDbPath();
    const batchA = await seedCorrections(
      dbPath,
      5,
      '2026-07-17 10:00:00',
      true,
    );
    const candidateA = newSuggestion([batchA[0].oldId, batchA[1].oldId]); // 2 < 3, rejected
    const first = mockFetchSequence([suggestResponse([candidateA])]);
    vi.stubGlobal('fetch', first.fetchFn);

    const resultA = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(resultA.suggestionsRejected).toBe(1);

    const batchB = await seedCorrections(
      dbPath,
      5,
      '2026-07-18 10:00:00',
      false,
    );
    // Same summary/domain as candidateA — findPriorSuggestionRejection must match.
    const candidateB = newSuggestion([batchB[0].oldId]);
    const second = mockFetchSequence([suggestResponse([candidateB])]);
    vi.stubGlobal('fetch', second.fetchFn);

    const resultB = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });

    expect(resultB.suggestionsProposed).toBe(1);
    expect(resultB.suggestionsRejected).toBe(0);
    const [row] = getSuggestionRows(dbPath);
    expect(row.evidence_count).toBe(3);
    expect(JSON.parse(row.supporting_chunks).sort()).toEqual(
      [batchA[0].oldId, batchA[1].oldId, batchB[0].oldId].sort(),
    );
  });

  it('7. cosine-matching candidate reinforces the existing suggestion instead of duplicating', async () => {
    dbPath = tmpDbPath();
    const batchA = await seedCorrections(
      dbPath,
      5,
      '2026-07-17 10:00:00',
      true,
    );
    const candidateA = newSuggestion([
      batchA[0].oldId,
      batchA[1].oldId,
      batchA[2].oldId,
    ]);
    const first = mockFetchSequence([suggestResponse([candidateA])]);
    vi.stubGlobal('fetch', first.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    const [formed] = getSuggestionRows(dbPath);
    expect(formed.evidence_count).toBe(3);
    expect(formed.embedding).not.toBeNull();
    expect(formed.last_reinforced).toBeNull();

    const batchB = await seedCorrections(
      dbPath,
      5,
      '2026-07-18 10:00:00',
      false,
    );
    const candidateB = newSuggestion([
      batchB[0].oldId,
      batchB[1].oldId,
      batchB[2].oldId,
    ]); // same summary text — cosine 1.0 with the MockEmbedder
    const second = mockFetchSequence([suggestResponse([candidateB])]);
    vi.stubGlobal('fetch', second.fetchFn);
    const resultB = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });

    expect(resultB.suggestionsReinforced).toBe(1);
    expect(resultB.suggestionsProposed).toBe(0);
    expect(getSuggestionRows(dbPath)).toHaveLength(1);
    const [reinforced] = getSuggestionRows(dbPath);
    expect(reinforced.evidence_count).toBe(6);
    expect(reinforced.last_reinforced).not.toBeNull();
    expect(reinforced.status).toBe('proposed');

    const journal = getSuggestionJournalRows(dbPath);
    expect(journal[journal.length - 1].action).toBe('reinforced');
  });

  it('8. without an embedder, dedup falls back to lexical similarity and warns', async () => {
    dbPath = tmpDbPath();
    const batchA = await seedCorrections(
      dbPath,
      5,
      '2026-07-17 10:00:00',
      true,
    );
    const candidateA = newSuggestion(
      [batchA[0].oldId, batchA[1].oldId, batchA[2].oldId],
      { summary: 'Always run the linter before committing any change' },
    );
    const first = mockFetchSequence([suggestResponse([candidateA])]);
    vi.stubGlobal('fetch', first.fetchFn);
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const resultA = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      // no embedder
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(resultA.suggestionsProposed).toBe(1);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('No embedder'),
    );
    const [formed] = getSuggestionRows(dbPath);
    expect(formed.embedding).toBeNull();
    warnSpy.mockClear();

    const batchB = await seedCorrections(
      dbPath,
      5,
      '2026-07-18 10:00:00',
      false,
    );
    const candidateB = newSuggestion(
      [batchB[0].oldId, batchB[1].oldId, batchB[2].oldId],
      { summary: 'Always run the linter before committing any new change' }, // near-identical
    );
    const second = mockFetchSequence([suggestResponse([candidateB])]);
    vi.stubGlobal('fetch', second.fetchFn);

    const resultB = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(resultB.suggestionsReinforced).toBe(1);
    expect(resultB.suggestionsProposed).toBe(0);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('9. a dismissed suggestion is reopened only when materially new evidence arrives', async () => {
    dbPath = tmpDbPath();
    const batchA = await seedCorrections(
      dbPath,
      5,
      '2026-07-16 10:00:00',
      true,
    );
    const candidateA = newSuggestion([
      batchA[0].oldId,
      batchA[1].oldId,
      batchA[2].oldId,
    ]);
    const fetchA = mockFetchSequence([suggestResponse([candidateA])]);
    vi.stubGlobal('fetch', fetchA.fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    const [formed] = getSuggestionRows(dbPath);
    markDismissed(dbPath, formed.id);

    // Cycle B: 2 known + 1 new — not enough net-new evidence to reopen.
    const batchB = await seedCorrections(
      dbPath,
      5,
      '2026-07-17 10:00:00',
      false,
    );
    const candidateB = newSuggestion([
      batchA[0].oldId,
      batchA[1].oldId,
      batchB[0].oldId,
    ]);
    const fetchB = mockFetchSequence([suggestResponse([candidateB])]);
    vi.stubGlobal('fetch', fetchB.fetchFn);
    const resultB = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(resultB.suggestionsRejected).toBe(1);
    expect(resultB.suggestionsProposed).toBe(0);
    let [row] = getSuggestionRows(dbPath);
    expect(row.status).toBe('dismissed');
    expect(row.evidence_count).toBe(3); // unchanged

    const journalB = getSuggestionJournalRows(dbPath);
    const lastB = journalB[journalB.length - 1];
    expect(lastB.action).toBe('rejected');
    const gateResultsB = JSON.parse(lastB.gate_results);
    expect(gateResultsB.reason).toBe('previously_dismissed');
    expect(gateResultsB.newEvidence).toBe(1);
    expect(gateResultsB.knownEvidence).toBe(3);

    // Cycle C: 3 brand-new ids — enough net-new evidence to reopen.
    const batchC = await seedCorrections(
      dbPath,
      5,
      '2026-07-18 10:00:00',
      false,
    );
    const candidateC = newSuggestion([
      batchC[0].oldId,
      batchC[1].oldId,
      batchC[2].oldId,
    ]);
    const fetchC = mockFetchSequence([suggestResponse([candidateC])]);
    vi.stubGlobal('fetch', fetchC.fetchFn);
    const resultC = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(resultC.suggestionsProposed).toBe(1);
    expect(resultC.suggestionsRejected).toBe(0);
    [row] = getSuggestionRows(dbPath);
    expect(row.status).toBe('proposed');
    expect(row.status_reason).toContain('reopened');
    expect(row.evidence_count).toBe(6);

    const journalC = getSuggestionJournalRows(dbPath);
    expect(journalC[journalC.length - 1].action).toBe('reopened');
  });

  it('10a. a parsed-empty suggestion output advances the watermark (next cycle scans nothing new)', async () => {
    dbPath = tmpDbPath();
    await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const { fetchFn, prompts } = mockFetchSequence([suggestResponse([])]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {},
    });
    expect(prompts).toHaveLength(1);
    expect(result.suggestionsProposed).toBe(0);
    expect(result.status).toBe('completed');
    expect(getBankConfigValue(dbPath, 'suggest_watermark')).toBe(
      '2026-07-19 10:00:00',
    );

    // Cycle 2: the same 5 corrections are now before the advanced watermark —
    // nothing new to scan, so the pass makes no LLM call at all.
    const second = mockFetchSequence(['SHOULD NOT BE CALLED']);
    vi.stubGlobal('fetch', second.fetchFn);
    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {},
    });
    expect(second.prompts).toHaveLength(0);
    expect(result2.suggestionsProposed).toBe(0);
  });

  it('10b. malformed suggestion output leaves the watermark untouched and does not block opinion/observation insights', async () => {
    dbPath = tmpDbPath();
    await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    await seedWorldFacts(dbPath, 5, false);

    const first = mockFetchSequence([
      'not valid json at all',
      REFLECT_RESPONSE,
    ]);
    vi.stubGlobal('fetch', first.fetchFn);
    const result1 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {},
    });
    expect(first.prompts).toHaveLength(2); // failed suggestion attempt + main reflect call
    expect(result1.suggestionsProposed).toBe(0);
    expect(result1.suggestionsRejected).toBe(0);
    expect(result1.observationsCreated).toBe(1); // main insight still applied — fail-open
    const watermark1 = getBankConfigValue(dbPath, 'suggest_watermark');
    expect(watermark1).toBeDefined();

    // Cycle 2: main batch is drained; the suggestion pass fires again over
    // the SAME 5 (still-unconsumed) corrections and fails to parse again —
    // watermark must still not move.
    const second = mockFetchSequence(['still not json']);
    vi.stubGlobal('fetch', second.fetchFn);
    const result2 = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {},
    });
    expect(second.prompts).toHaveLength(1); // suggestion pass only
    expect(result2.suggestionsProposed).toBe(0);
    expect(getBankConfigValue(dbPath, 'suggest_watermark')).toBe(watermark1);
  });

  it('11. scans corrections/forgets/friction/workflow since the watermark, excludes task-scope and pre-watermark rows', async () => {
    dbPath = tmpDbPath();
    const boot = new Database(dbPath);
    loadSchema(boot);
    boot.close();

    const frictionId = await seedFriction(
      dbPath,
      'ripgrep exited 1 with no matches for the search pattern',
      false,
    );
    const workflowId = await seedWorkflow(
      dbPath,
      'Ran the full test suite before opening the PR, as usual',
      false,
    );
    const corr = await seedCorrections(dbPath, 3, '2026-07-19 10:00:00', false);
    const [forgottenId] = await seedWorldFacts(dbPath, 1, false);
    const forgetDb = new Database(dbPath);
    forgetDb
      .prepare(
        `UPDATE chunks SET is_active = 0, updated_at = '2026-07-19 10:00:00' WHERE id = ?`,
      )
      .run(forgottenId);
    forgetDb.close();

    const taskId = insertTaskScopeChunk(
      dbPath,
      'Should never be scanned — task scope',
    );

    const preWatermarkId = await seedFriction(
      dbPath,
      'This tool result predates the suggestion watermark',
      false,
    );
    backdateCreatedAt(dbPath, preWatermarkId, '2026-06-01 10:00:00');

    const { fetchFn, prompts } = mockFetchSequence([suggestResponse([])]);
    vi.stubGlobal('fetch', fetchFn);
    await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: {},
    });

    expect(prompts).toHaveLength(1);
    const prompt = prompts[0];
    expect(prompt).toContain(
      `[${frictionId}] (friction, experience/tool_result)`,
    );
    expect(prompt).toContain(
      `[${workflowId}] (workflow, experience/agent_generated)`,
    );
    expect(prompt).toContain(
      `[${corr[0].oldId}] (correction, world/agent_generated)`,
    );
    expect(prompt).toContain(`→ CORRECTED TO [${corr[0].newId}]`);
    expect(prompt).toContain(
      `[${forgottenId}] (correction, world/user_stated)`,
    );
    expect(prompt).toContain('→ FORGOTTEN');
    expect(prompt).not.toContain(taskId);
    expect(prompt).not.toContain(preWatermarkId);
  });

  it('12. suggestions never surface via recall() or groundSubagent() — structural isolation', async () => {
    dbPath = tmpDbPath();
    const corr = await seedCorrections(dbPath, 5, '2026-07-19 10:00:00', true);
    const distinctToken = 'ZarquonFlangeProtocol';
    const candidate = newSuggestion(
      [corr[0].oldId, corr[1].oldId, corr[2].oldId],
      { summary: `Always apply the ${distinctToken} before deploying` },
    );
    const { fetchFn } = mockFetchSequence([suggestResponse([candidate])]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflect({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      suggestions: { gates: { minEvidenceCount: 3 } },
    });
    expect(result.suggestionsProposed).toBe(1);
    const [row] = getSuggestionRows(dbPath);
    expect(row.summary).toContain(distinctToken);

    const engram = await Engram.open(dbPath, { embedder });
    try {
      const recallResponse = await engram.recall(distinctToken);
      for (const r of recallResponse.results) {
        expect(r.text).not.toContain(distinctToken);
      }
      for (const o of recallResponse.observations) {
        expect(o.summary).not.toContain(distinctToken);
      }

      const readonly = await engram.readonlyView();
      const grounding = await groundSubagent(readonly, {
        task: distinctToken,
      });
      expect(grounding.prompt).not.toContain(distinctToken);
      expect(
        grounding.facts.every((f) => !f.text.includes(distinctToken)),
      ).toBe(true);
    } finally {
      engram.close();
    }
  });

  it('13. reflectCatchUp sums suggestion counters across multiple inner reflect() batches', async () => {
    dbPath = tmpDbPath();
    await seedWorldFacts(dbPath, 12, true);
    const batch1 = await seedCorrections(
      dbPath,
      5,
      '2026-07-17 10:00:00',
      false,
    );
    const batch2 = await seedCorrections(
      dbPath,
      5,
      '2026-07-18 10:00:00',
      false,
    );

    const candidateA = newSuggestion([batch1[0].oldId, batch1[1].oldId], {
      summary: 'Rule A: always double-check script paths after a correction',
      domain: 'workflow',
    });
    const candidateB = newSuggestion([batch2[0].oldId, batch2[1].oldId], {
      summary: 'Rule B: verify config values before deploying',
      domain: 'infrastructure',
    });

    // Main-batch responses must produce SOME insight (not an empty
    // reflectResponse([])) — a genuinely empty cycle trips the issue-#17
    // zero-insight guard, which leaves that batch's facts unreflected and
    // would register as a reflectCatchUp stall, unrelated to what this test
    // is actually verifying (suggestion counter aggregation).
    const { fetchFn, prompts } = mockFetchSequence([
      suggestResponse([candidateA]),
      REFLECT_RESPONSE,
      suggestResponse([candidateB]),
      REFLECT_RESPONSE,
    ]);
    vi.stubGlobal('fetch', fetchFn);

    const result = await reflectCatchUp({
      counterEvidence: false,
      opinionGates: false,
      dbPath,
      reflectModel: 'llama-test',
      embedder,
      batchSize: 5,
      minFactsThreshold: 5,
      maxBatches: 5,
      suggestions: {
        batchSize: 5,
        minSignalThreshold: 5,
        gates: { minEvidenceCount: 2, minDistinctDays: 1 },
        // This test is about counter aggregation, not dedup precision (tests
        // 7/8 cover that) — force both candidates to form as new suggestions
        // regardless of the crude MockEmbedder's cosine output for two
        // similarly-shaped English sentences (score can never reach >1).
        dedupThreshold: 1.01,
      },
    });

    expect(result.batches).toBe(2);
    expect(prompts).toHaveLength(4);
    expect(result.suggestionsProposed).toBe(2);
    expect(result.suggestionsReinforced).toBe(0);
    expect(result.suggestionsRejected).toBe(0);
    expect(result.status).toBe('drained');
  });
});
