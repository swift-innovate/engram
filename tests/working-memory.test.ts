import { describe, it, expect, afterEach } from 'vitest';
import { Engram, type SessionCandidate } from '../src/engram.js';
import { MockEmbedder, tmpDbPath, cleanupDb } from './helpers.js';

describe('Working Memory', () => {
  let engram: Engram | undefined;
  let dbPath: string;

  afterEach(() => {
    try { engram?.close(); } catch { /* already closed */ }
    engram = undefined;
    cleanupDb(dbPath);
  });

  // ── Session Creation ───────────────────────────────────────────────────

  it('creates a new session from a message when no sessions exist', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('check my email for updates');

    expect(result.session.id).toMatch(/^wm-/);
    expect(result.session.goal).toContain('check my email');
    expect(result.confidence).toBe(1.0);
    expect(result.diagnostics.reason).toBe('new');
  });

  // ── Session Resumption ────────────────────────────────────────────────

  it('resumes an existing session when topic matches', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const first = await engram.inferWorkingSession('check my email for updates');
    const second = await engram.inferWorkingSession('any new emails today');

    expect(second.diagnostics.reason).toBe('match');
    expect(second.session.id).toBe(first.session.id);
  });

  // ── Topic Isolation ────────────────────────────────────────────────────

  it('creates isolated sessions for different topics', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const email = await engram.inferWorkingSession('check my email for updates from the office');
    const garden = await engram.inferWorkingSession('ZZZZZ plant roses in the garden ZZZZZ', { threshold: 0.999 });

    expect(email.session.id).not.toBe(garden.session.id);

    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBe(2);
  });

  // ── Session Update ─────────────────────────────────────────────────────

  it('updates session state with merged data', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('plan the deployment');

    await engram.updateWorkingSession(result.session.id, {
      goal: 'plan the production deployment for Friday',
      status: 'in_progress',
    });

    const updated = engram.getWorkingSession(result.session.id);
    expect(updated).not.toBeNull();
    expect(updated!.goal).toBe('plan the production deployment for Friday');
    expect((updated as any).status).toBe('in_progress');
  });

  it('throws when updating a non-existent session', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await expect(
      engram.updateWorkingSession('wm-doesnotexist', { goal: 'test' })
    ).rejects.toThrow('not found');
  });

  // ── Session Retrieval ──────────────────────────────────────────────────

  it('getWorkingSession returns null for expired sessions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('temporary task');
    engram.clearWorkingSession(result.session.id);

    const retrieved = engram.getWorkingSession(result.session.id);
    expect(retrieved).toBeNull();
  });

  it('listWorkingSessions returns only active sessions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await engram.inferWorkingSession('task one about alpha');
    const second = await engram.inferWorkingSession('ZZZZ completely different topic ZZZZ', { threshold: 0.999 });
    engram.clearWorkingSession(second.session.id);

    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBe(1);
    expect(sessions[0].goal).toContain('task one');
  });

  // ── Snapshot ───────────────────────────────────────────────────────────

  it('snapshotWorkingSession creates an experience chunk and expires the session', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('design the API schema');
    const snapshot = await engram.snapshotWorkingSession(result.session.id);

    expect(snapshot.chunkId).toMatch(/^chk-/);

    // Session should now be expired
    const retrieved = engram.getWorkingSession(result.session.id);
    expect(retrieved).toBeNull();

    // The snapshot chunk should be findable via recall
    const recallResult = await engram.recall('API schema', { strategies: ['keyword'] });
    expect(recallResult.results.some(r => r.source?.includes('working_memory'))).toBe(true);
  });

  it('snapshotWorkingSession includes progress in the chunk when set', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('debug the auth bug');
    await engram.updateWorkingSession(result.session.id, {
      progress: 'Identified root cause in JWT validation. Applied fix to middleware.',
    });

    await engram.snapshotWorkingSession(result.session.id);

    // Snapshot should include progress notes
    const recallResult = await engram.recall('JWT validation fix', { strategies: ['keyword'] });
    expect(recallResult.results.some(r => r.source?.includes('working_memory'))).toBe(true);
  });

  // ── Clear ──────────────────────────────────────────────────────────────

  it('clearWorkingSession expires without snapshotting', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('throwaway task');
    const cleared = engram.clearWorkingSession(result.session.id);

    expect(cleared).toBe(true);
    expect(engram.getWorkingSession(result.session.id)).toBeNull();

    // No experience chunk should exist from this session
    const recallResult = await engram.recall('throwaway', { strategies: ['keyword'] });
    expect(recallResult.results.some(r => r.source?.includes('working_memory'))).toBe(false);
  });

  it('clearWorkingSession returns false for already-expired session', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('temp');
    engram.clearWorkingSession(result.session.id);

    const secondClear = engram.clearWorkingSession(result.session.id);
    expect(secondClear).toBe(false);
  });

  // ── Session Cap ────────────────────────────────────────────────────────

  it('enforces maxActive by snapshotting oldest sessions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await engram.inferWorkingSession('AAAA first topic about alpha');
    await engram.inferWorkingSession('BBBB second topic about beta');
    await engram.inferWorkingSession('CCCC third topic about gamma');

    // 4th with maxActive: 3 — should snapshot the oldest
    await engram.inferWorkingSession('DDDD fourth topic about delta', { maxActive: 3 });

    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBeLessThanOrEqual(3);
  });

  it('clamps maxActive=0 to 1, keeping at least one session active', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    // Create a session first
    await engram.inferWorkingSession('AAAA first session');

    // Pass maxActive: 0 — should be clamped to 1, not snapshot the existing session
    await engram.inferWorkingSession('ZZZZ completely different session ZZZZ', { maxActive: 0 });

    // Should have at most 1 active (clamped), not 0
    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBeGreaterThanOrEqual(1);
  });

  // ── Stale Expiry ───────────────────────────────────────────────────────

  it('expireStaleWorkingSessions snapshots sessions older than maxAgeHours', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('old task to expire');

    // Use expireStaleWorkingSessions(0) — "anything older than 0 hours" = everything
    // SQLite datetime('now', '-0 hours') == now, and updated_at was set moments ago,
    // so updated_at < now should be true (sub-second difference from INSERT time).
    // Small sleep to ensure the timestamp inequality holds.
    await new Promise(r => setTimeout(r, 50));
    const expired = await engram.expireStaleWorkingSessions(0);

    expect(expired).toBeGreaterThanOrEqual(1);
    expect(engram.getWorkingSession(result.session.id)).toBeNull();
  });

  // ── Primitives: embedText + findSimilarSessions ──────────────────────

  it('embedText() returns a Float32Array of correct dimensions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const embedding = await engram.embedText('test embedding text');

    expect(embedding).toBeInstanceOf(Float32Array);
    expect(embedding.length).toBeGreaterThan(0);
  });

  it('findSimilarSessions() returns empty array when no sessions exist', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const embedding = await engram.embedText('anything');
    const candidates = engram.findSimilarSessions(embedding);

    expect(candidates).toEqual([]);
  });

  it('findSimilarSessions() returns candidates sorted by similarity descending', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    // Create two sessions with different topics
    await engram.inferWorkingSession('check my email for updates');
    await engram.inferWorkingSession('ZZZZ plant roses in the garden ZZZZ', { threshold: 0.999 });

    const embedding = await engram.embedText('check my email for updates');
    const candidates = engram.findSimilarSessions(embedding);

    expect(candidates.length).toBe(2);
    expect(candidates[0].similarity).toBeGreaterThanOrEqual(candidates[1].similarity);
    // Each candidate has parsed state, not raw JSON
    expect(candidates[0].state).toHaveProperty('id');
    expect(candidates[0].state).toHaveProperty('goal');
  });

  it('findSimilarSessions() excludes expired sessions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.inferWorkingSession('session to expire');
    engram.clearWorkingSession(result.session.id);

    const embedding = await engram.embedText('session to expire');
    const candidates = engram.findSimilarSessions(embedding);

    expect(candidates).toEqual([]);
  });

  it('findSimilarSessions() respects limit parameter', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await engram.inferWorkingSession('AAAA first topic alpha');
    await engram.inferWorkingSession('BBBB second topic beta', { threshold: 0.999 });
    await engram.inferWorkingSession('CCCC third topic gamma', { threshold: 0.999 });

    const embedding = await engram.embedText('first topic');
    const candidates = engram.findSimilarSessions(embedding, 1);

    expect(candidates.length).toBe(1);
  });

  it('createWorkingSession() creates a session accessible via findSimilarSessions', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const embedding = await engram.embedText('custom session topic');
    const embeddingBuffer = Buffer.from(embedding.buffer);
    const session = await engram.createWorkingSession('custom session topic', embeddingBuffer);

    expect(session.id).toMatch(/^wm-/);
    expect(session.goal).toBe('custom session topic');

    const candidates = engram.findSimilarSessions(embedding);
    expect(candidates.length).toBe(1);
    expect(candidates[0].id).toBe(session.id);
  });

  // ── Related Context ────────────────────────────────────────────────────

  it('returns related long-term context alongside the session', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await engram.retain('Tom uses Terraform for all IaC deployments', {
      memoryType: 'world',
      trustScore: 0.9,
    });

    const result = await engram.inferWorkingSession('plan the Terraform deployment');

    expect(result.relatedContext).toBeTruthy();
    expect(typeof result.relatedContext).toBe('string');
    expect(result.relatedContext.length).toBeGreaterThan(0);
  });
});
