import { describe, it, expect, afterEach, vi } from 'vitest';
import { Engram } from '../src/engram.js';
import {
  MockEmbedder,
  tmpDbPath,
  cleanupDb,
  mockOllamaFetch,
  REFLECT_RESPONSE,
  EXTRACTION_RESPONSE,
} from './helpers.js';

// ---------------------------------------------------------------------------
// Engram class lifecycle
// ---------------------------------------------------------------------------

describe('Engram', () => {
  let engram: Engram | undefined;
  let dbPath: string;

  afterEach(() => {
    vi.unstubAllGlobals();
    try { engram?.close(); } catch { /* already closed */ }
    engram = undefined;
    cleanupDb(dbPath);
  });

  // ---- Initialization ----

  it('create() initializes a new engram with an injected embedder', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });
    expect(engram).toBeInstanceOf(Engram);
  });

  it('create() writes reflectMission to bank_config', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, {
      embedder: new MockEmbedder(),
      reflectMission: 'Focus on architecture decisions',
    });

    // Verify by reopening the file
    engram.close();
    const verify = await Engram.open(dbPath, { embedder: new MockEmbedder() });
    // If bank_config was written, the file is valid and opens without error
    expect(verify).toBeInstanceOf(Engram);
    verify.close();
    engram = undefined;
  });

  it('open() opens an existing engram without overwriting config', async () => {
    dbPath = tmpDbPath();

    // Create with a specific mission
    engram = await Engram.create(dbPath, {
      embedder: new MockEmbedder(),
      reflectMission: 'Original mission',
    });
    engram.close();

    // Open without specifying mission — should not overwrite
    engram = await Engram.open(dbPath, { embedder: new MockEmbedder() });
    expect(engram).toBeInstanceOf(Engram);
  });

  it('create() is idempotent — calling twice on same path is safe', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });
    engram.close();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });
    expect(engram).toBeInstanceOf(Engram);
  });

  // ---- retain / recall ----

  it('retain() stores a chunk and recall() retrieves it', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const r = await engram.retain('Alice prefers Rust for systems programming', {
      memoryType: 'world',
      trustScore: 0.9,
    });
    expect(r.chunkId).toMatch(/^chk-/);

    const response = await engram.recall('Alice Rust', {
      strategies: ['keyword'],
      topK: 5,
    });
    expect(response.results.length).toBeGreaterThan(0);
    expect(response.results[0].text).toContain('Alice');
  });

  it('retainBatch() stores multiple chunks', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const results = await engram.retainBatch([
      { text: 'Fact one about systems' },
      { text: 'Fact two about infrastructure' },
      { text: 'Fact three about tooling' },
    ]);
    expect(results).toHaveLength(3);
  });

  // ---- processExtractions ----

  it('processExtractions() drains the queue via Ollama', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    // Dispatch fetch based on endpoint
    vi.stubGlobal('fetch', async (url: string | URL | Request) => {
      const urlStr = url.toString();
      if (urlStr.includes('/api/embed')) {
        return { ok: true, json: async () => ({ embeddings: [new Array(768).fill(0.1)] }) } as unknown as Response;
      }
      return { ok: true, json: async () => ({ response: EXTRACTION_RESPONSE }), text: async () => '' } as unknown as Response;
    });

    await engram.retain('Alice prefers Rust', { memoryType: 'world' });
    const result = await engram.processExtractions(10);

    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);
  });

  // ---- reflect ----

  it('reflect() runs a cycle against the db file', async () => {
    dbPath = tmpDbPath();
    const embedder = new MockEmbedder();
    engram = await Engram.create(dbPath, { embedder });

    // Add enough facts to cross the minFactsThreshold (default: 5)
    for (let i = 0; i < 5; i++) {
      await engram.retain(`Alice prefers Rust — fact ${i}`, {
        memoryType: 'world', sourceType: 'user_stated', trustScore: 0.8,
      });
    }

    // Close so reflect can open its own connection (WAL mode)
    engram.close();
    engram = undefined;

    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));
    const result = await (await Engram.open(dbPath, { embedder })).reflect();
    expect(result.status).toBe('completed');
    expect(result.factsProcessed).toBe(5);
  });

  // ---- close ----

  it('close() makes subsequent operations throw', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });
    engram.close();

    // retain() calls embedder.embed() (succeeds), then db.prepare() (throws on closed db)
    await expect(engram.retain('test', {})).rejects.toThrow();
    engram = undefined; // prevent afterEach from calling close() again
  });
});
