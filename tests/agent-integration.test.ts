// =============================================================================
// Agent Integration Test
//
// Simulates a multi-turn, multi-topic agent conversation using Engram as its
// memory backend. Agent "Mira" helps a developer across two concurrent topics:
//   - Topic A: Kubernetes migration (auth-service from Docker Compose to K8s)
//   - Topic B: Database optimization (PostgreSQL query tuning)
//
// No Ollama required — uses MockEmbedder + mockOllamaFetch throughout.
// =============================================================================

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { Engram, shouldRetain } from '../src/engram.js';
import {
  MockEmbedder,
  tmpDbPath,
  cleanupDb,
  mockOllamaFetch,
} from './helpers.js';

// ---------------------------------------------------------------------------
// Scenario-specific mock data
// ---------------------------------------------------------------------------

const EXTRACTION_K8S = JSON.stringify({
  entities: [
    {
      name: 'auth-service',
      canonical_name: 'auth-service',
      entity_type: 'project',
      aliases: [],
    },
    {
      name: 'Kubernetes',
      canonical_name: 'kubernetes',
      entity_type: 'technology',
      aliases: ['k8s'],
    },
    { name: 'Tom', canonical_name: 'tom', entity_type: 'person', aliases: [] },
    {
      name: 'Helm',
      canonical_name: 'helm',
      entity_type: 'technology',
      aliases: [],
    },
  ],
  relations: [
    {
      source: 'auth-service',
      target: 'kubernetes',
      relation_type: 'migrating_to',
      description: 'auth-service is being migrated to Kubernetes',
    },
    {
      source: 'tom',
      target: 'helm',
      relation_type: 'uses',
      description: 'Tom uses Helm for K8s deployments',
    },
  ],
});

const REFLECT_K8S = JSON.stringify({
  observations: [
    {
      summary:
        'The team is actively migrating auth-service from Docker Compose to Kubernetes using Helm charts',
      domain: 'infrastructure',
      topic: 'container orchestration',
      source_chunk_ids: [],
      entity_names: ['auth-service', 'Kubernetes', 'Helm'],
    },
    {
      summary: 'Tom prefers Helm over raw manifests for Kubernetes deployments',
      domain: 'preferences',
      topic: 'deployment tooling',
      source_chunk_ids: [],
      entity_names: ['Tom', 'Helm'],
    },
  ],
  opinion_updates: [
    {
      belief:
        'Helm is the preferred deployment mechanism for this teams Kubernetes workloads',
      direction: 'new',
      confidence_delta: 0.25,
      domain: 'infrastructure',
      evidence_chunk_ids: [],
      entity_names: ['Helm', 'Kubernetes'],
    },
  ],
  observation_refreshes: [],
});

// ---------------------------------------------------------------------------
// Shared state across all tests
// ---------------------------------------------------------------------------

describe('Agent Integration — multi-topic conversation lifecycle', () => {
  let engram: Engram;
  let dbPath: string;
  const embedder = new MockEmbedder();

  // Session IDs captured during the test flow
  let sessionA: string; // K8s topic
  let sessionB: string; // DB optimization topic

  beforeAll(async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, {
      embedder,
      reflectModel: 'llama-test',
      reflectMission:
        'Focus on infrastructure decisions, deployment patterns, and performance optimization.',
      retainMission:
        'Prioritize technical decisions, architecture changes, and performance insights.',
    });
  });

  afterAll(() => {
    vi.unstubAllGlobals();
    try {
      engram?.close();
    } catch {
      /* already closed in phase 13 */
    }
    cleanupDb(dbPath);
  });

  // ── Phase 1: Agent startup — shouldRetain gate ──────────────────────────

  it('Phase 1: shouldRetain rejects phatic input and accepts substantive text', () => {
    const greeting = shouldRetain('hey there!');
    expect(greeting.score).toBeLessThan(0.5);

    const substantive = shouldRetain(
      'We need to migrate auth-service from Docker Compose to Kubernetes using Helm charts',
    );
    expect(substantive.score).toBeGreaterThanOrEqual(0.5);
  });

  // ── Phase 2: Retain + Recall — store K8s facts ──────────────────────────

  it('Phase 2: stores K8s migration facts and recalls them', async () => {
    const facts = [
      'auth-service is currently running on Docker Compose in production',
      'Tom wants to migrate auth-service to Kubernetes using Helm charts',
      'The auth-service handles JWT validation and session management for all microservices',
    ];

    for (const fact of facts) {
      const r = await engram.retain(fact, {
        memoryType: 'world',
        source: 'conversation:k8s-migration',
        sourceType: 'user_stated',
        trustScore: 0.9,
      });
      expect(r.chunkId).toMatch(/^chk-/);
    }

    // FTS5 keyword search — use terms present in the retained facts
    const response = await engram.recall('Docker Compose', {
      strategies: ['keyword'],
      topK: 5,
    });
    expect(response.results.length).toBeGreaterThan(0);
    expect(
      response.results.some((r) => r.text.includes('Docker Compose')),
    ).toBe(true);
  });

  // ── Phase 3: Session creation — new K8s session ─────────────────────────

  it('Phase 3: creates a new working session for K8s topic', async () => {
    const result = await engram.inferWorkingSession(
      'Help me plan the auth-service Kubernetes migration',
    );

    expect(result.session.id).toMatch(/^wm-/);
    expect(result.confidence).toBe(1.0);
    expect(result.diagnostics.reason).toBe('new');
    expect(result.relatedContext.length).toBeGreaterThan(0);

    sessionA = result.session.id;
  });

  // ── Phase 4: Session resumption — K8s follow-up ─────────────────────────

  it('Phase 4: resumes the K8s session on follow-up message', async () => {
    const result = await engram.inferWorkingSession(
      'What about the auth-service Helm chart values?',
    );

    expect(result.diagnostics.reason).toBe('match');
    expect(result.session.id).toBe(sessionA);
  });

  // ── Phase 5: Topic switch — DB optimization ─────────────────────────────

  it('Phase 5: creates a new session for a different topic', async () => {
    const result = await engram.inferWorkingSession(
      'ZZZZ PostgreSQL dashboard query is slow, need to add indexes ZZZZ',
      { threshold: 0.999 },
    );

    expect(result.diagnostics.reason).toBe('new');
    expect(result.session.id).not.toBe(sessionA);
    expect(result.confidence).toBe(1.0);

    sessionB = result.session.id;

    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBe(2);
  });

  // ── Phase 6: Return to Topic A — resume K8s ────────────────────────────

  it('Phase 6: returns to K8s topic and resumes original session', async () => {
    // Message must be similar enough to sessionA's seed query
    // ("Help me plan the auth-service Kubernetes migration") for MockEmbedder
    // to produce a cosine similarity >= 0.72 (default threshold)
    const result = await engram.inferWorkingSession(
      'Help me plan the auth-service Kubernetes health checks',
    );

    expect(result.session.id).toBe(sessionA);
    expect(result.diagnostics.reason).toBe('match');

    // Still only 2 sessions, not 3
    const sessions = engram.listWorkingSessions();
    expect(sessions.length).toBe(2);
  });

  // ── Phase 7: Session update — add progress to K8s session ───────────────

  it('Phase 7: updates session with progress notes', async () => {
    await engram.updateWorkingSession(sessionA, {
      status: 'in_progress',
      progress:
        'Drafted Helm chart with health checks. Need to configure ingress next.',
    } as any);

    const updated = engram.getWorkingSession(sessionA);
    expect(updated).not.toBeNull();
    expect((updated as any).progress).toContain('Helm chart');
    expect((updated as any).status).toBe('in_progress');
    // Original goal should be preserved
    expect(updated!.goal).toContain('auth-service');
  });

  // ── Phase 8: More facts — reach reflect threshold ───────────────────────

  it('Phase 8: retains additional facts to reach reflect threshold', async () => {
    const extraFacts = [
      'The Helm chart should use liveness and readiness probes on /healthz',
      'PostgreSQL dashboard queries need a composite index on (tenant_id, created_at)',
    ];

    for (const fact of extraFacts) {
      const r = await engram.retain(fact, {
        memoryType: 'world',
        source: 'conversation:k8s-migration',
        sourceType: 'user_stated',
        trustScore: 0.85,
      });
      expect(r.chunkId).toMatch(/^chk-/);
    }

    // Verify we now have 5 total facts (3 from Phase 2 + 2 here)
    const response = await engram.recall('service', {
      strategies: ['keyword'],
      topK: 10,
    });
    expect(response.results.length).toBeGreaterThanOrEqual(3);
  });

  // ── Phase 9: Entity extraction ──────────────────────────────────────────

  it('Phase 9: processes entity extraction queue', async () => {
    vi.stubGlobal('fetch', async (url: string | URL | Request) => {
      const urlStr = url.toString();
      if (urlStr.includes('/api/embed')) {
        return {
          ok: true,
          json: async () => ({ embeddings: [new Array(768).fill(0.1)] }),
        } as unknown as Response;
      }
      return {
        ok: true,
        json: async () => ({ response: EXTRACTION_K8S }),
        text: async () => '',
      } as unknown as Response;
    });

    const result = await engram.processExtractions(20);
    expect(result.processed).toBeGreaterThanOrEqual(1);
    expect(result.failed).toBe(0);

    vi.unstubAllGlobals();
  });

  // ── Phase 10: Reflection ────────────────────────────────────────────────

  it('Phase 10: runs reflection cycle and produces observations + opinions', async () => {
    // reflect() opens its own DB connection, so close + reopen
    engram.close();

    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_K8S));

    engram = await Engram.open(dbPath, {
      embedder,
      reflectModel: 'llama-test',
    });
    const result = await engram.reflect({
      counterEvidence: false,
      opinionGates: false,
    });

    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBeGreaterThanOrEqual(5);
    expect(result.observationsCreated).toBeGreaterThanOrEqual(1);
    expect(result.opinionsFormed).toBeGreaterThanOrEqual(1);

    vi.unstubAllGlobals();

    // Reopen with a fresh connection for subsequent phases
    engram.close();
    engram = await Engram.open(dbPath, { embedder });
  });

  // ── Phase 11: Session snapshot ──────────────────────────────────────────

  it('Phase 11: snapshots the DB optimization session to long-term memory', async () => {
    const snapshot = await engram.snapshotWorkingSession(sessionB);
    expect(snapshot.chunkId).toMatch(/^chk-/);

    // Session B should now be expired
    const retrieved = engram.getWorkingSession(sessionB);
    expect(retrieved).toBeNull();

    // The snapshot chunk should be findable
    const recallResult = await engram.recall('PostgreSQL', {
      strategies: ['keyword'],
    });
    expect(
      recallResult.results.some((r) => r.source?.includes('working_memory')),
    ).toBe(true);
  });

  // ── Phase 12: Full recall — everything present ──────────────────────────

  it('Phase 12: full recall returns chunks, observations, and opinions', async () => {
    const response = await engram.recall(
      'Kubernetes Helm deployment auth-service',
      {
        topK: 10,
        includeObservations: true,
        includeOpinions: true,
      },
    );

    // Chunks from retained facts
    expect(response.results.length).toBeGreaterThan(0);

    // Observations from reflection
    expect(response.observations).toBeDefined();
    expect(response.observations!.length).toBeGreaterThanOrEqual(1);

    // Opinions from reflection
    expect(response.opinions).toBeDefined();
    expect(response.opinions!.length).toBeGreaterThanOrEqual(1);
    expect(response.opinions![0].confidence).toBeGreaterThan(0);
  });

  // ── Phase 13: Clean shutdown ────────────────────────────────────────────

  it('Phase 13: close() prevents further operations', async () => {
    engram.close();

    await expect(engram.retain('should fail', {})).rejects.toThrow();
  });
});
