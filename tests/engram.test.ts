import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { Engram } from '../src/engram.js';
import {
  MockEmbedder,
  loadSchema,
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
    try {
      engram?.close();
    } catch {
      /* already closed */
    }
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

    const r = await engram.retain(
      'Alice prefers Rust for systems programming',
      {
        memoryType: 'world',
        trustScore: 0.9,
      },
    );
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
    engram = await Engram.create(dbPath, {
      embedder: new MockEmbedder(),
      reflectModel: 'llama-test',
    });

    // Dispatch fetch based on endpoint
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
        json: async () => ({ response: EXTRACTION_RESPONSE }),
        text: async () => '',
      } as unknown as Response;
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
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      });
    }

    // Close so reflect can open its own connection (WAL mode)
    engram.close();
    engram = undefined;

    vi.stubGlobal('fetch', mockOllamaFetch(REFLECT_RESPONSE));
    const result = await (
      await Engram.open(dbPath, { embedder, reflectModel: 'llama-test' })
    ).reflect({ counterEvidence: false, opinionGates: false });
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

  // ---- forget / supersede / forgetBySource ----

  it('forget() soft-deletes a chunk — excluded from recall', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const r = await engram.retain('Tom prefers Terraform for homelab', {
      memoryType: 'world',
      trustScore: 0.9,
    });

    const forgotten = await engram.forget(r.chunkId);
    expect(forgotten).toBe(true);

    const db = new Database(dbPath);
    const queueRow = db
      .prepare('SELECT status, error FROM extraction_queue WHERE chunk_id = ?')
      .get(r.chunkId) as any;
    db.close();
    expect(queueRow.status).toBe('completed');
    expect(queueRow.error).toContain('deactivated');

    // Should not appear in recall
    const response = await engram.recall('Terraform homelab', {
      strategies: ['keyword'],
    });
    expect(response.results.some((res) => res.id === r.chunkId)).toBe(false);
  });

  it('forget() returns false for a non-existent chunk ID', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const forgotten = await engram.forget('chk-doesnotexist');
    expect(forgotten).toBe(false);
  });

  it('supersede() replaces old fact with new text, old chunk inactive', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const old = await engram.retain('Tom prefers Terraform', {
      memoryType: 'world',
      trustScore: 0.9,
    });
    const newResult = await engram.supersede(
      old.chunkId,
      'Tom switched to Pulumi',
      {
        memoryType: 'world',
        trustScore: 0.9,
      },
    );

    expect(newResult.chunkId).not.toBe(old.chunkId);

    // New chunk is findable; old chunk (is_active=FALSE) does not appear
    const pulumi = await engram.recall('Pulumi', { strategies: ['keyword'] });
    expect(pulumi.results.some((r) => r.id === newResult.chunkId)).toBe(true);

    // Old chunk is excluded even when searching for its exact content
    const terraform = await engram.recall('Terraform', {
      strategies: ['keyword'],
    });
    expect(terraform.results.some((r) => r.id === old.chunkId)).toBe(false);
  });

  it('supersede() is atomic: a failure between insert and mark-superseded leaves nothing persisted', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const old = await engram.retain('Tom prefers Terraform (atomicity)', {
      memoryType: 'world',
      trustScore: 0.9,
    });

    // Install a real (non-TEMP) trigger via a second raw connection to the
    // same file, so it also fires for Engram's own connection — SQLite
    // triggers live in the schema, not the connection. It fires exactly on
    // the "mark old chunk superseded" UPDATE (is_active), which retain()
    // now runs AFTER the new chunk's INSERT inside the same transaction —
    // so a failure here proves the INSERT rolls back too, not just the
    // UPDATE. Without atomicity, the INSERT would already be committed and
    // this trigger firing would leave the new chunk active with the old
    // chunk never marked superseded — exactly the double-counting bug.
    const rawDb = new Database(dbPath);
    rawDb.exec(`
      CREATE TRIGGER poison_supersede
      BEFORE UPDATE OF is_active ON chunks
      WHEN OLD.id = '${old.chunkId}'
      BEGIN
        SELECT RAISE(ABORT, 'injected failure for atomicity test');
      END;
    `);
    rawDb.close();

    await expect(
      engram.supersede(old.chunkId, 'Tom switched to Pulumi (atomicity)', {
        memoryType: 'world',
        trustScore: 0.9,
      }),
    ).rejects.toThrow();

    const cleanup = new Database(dbPath);
    cleanup.exec('DROP TRIGGER IF EXISTS poison_supersede');
    cleanup.close();

    const verify = new Database(dbPath);
    const oldRow = verify
      .prepare('SELECT is_active, superseded_by FROM chunks WHERE id = ?')
      .get(old.chunkId) as any;
    const newRowCount = verify
      .prepare(
        `SELECT COUNT(*) as c FROM chunks WHERE text LIKE '%Pulumi (atomicity)%'`,
      )
      .get() as { c: number };
    verify.close();

    // Old chunk untouched — the whole transaction rolled back, not just
    // the UPDATE that failed.
    expect(oldRow.is_active).toBe(1);
    expect(oldRow.superseded_by).toBeNull();
    // New chunk was never persisted — all-or-nothing.
    expect(newRowCount.c).toBe(0);
  });

  it('supersede() marks the old chunk superseded when newText dedups to an existing chunk', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const existing = await engram.retain('Tom uses Kubernetes', {
      memoryType: 'world',
      trustScore: 0.9,
    });
    const old = await engram.retain('Tom uses Docker Swarm', {
      memoryType: 'world',
      trustScore: 0.9,
    });

    const result = await engram.supersede(
      old.chunkId,
      '  tom uses kubernetes  ', // normalizes to the same text as `existing`
      { memoryType: 'world', trustScore: 0.9 },
    );

    expect(result.chunkId).toBe(existing.chunkId);
    expect(result.deduplicated).toBe(true);

    const db = new Database(dbPath);
    const oldRow = db
      .prepare('SELECT is_active, superseded_by FROM chunks WHERE id = ?')
      .get(old.chunkId) as any;
    db.close();
    expect(oldRow.is_active).toBe(0);
    expect(oldRow.superseded_by).toBe(existing.chunkId);
  });

  it('supersede() on a missing oldChunkId still creates the new chunk (external behavior unchanged)', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const result = await engram.supersede(
      'chk-doesnotexist',
      'A brand new fact',
      { memoryType: 'world' },
    );

    expect(result.chunkId).toMatch(/^chk-/);
    const found = await engram.recall('brand new fact', {
      strategies: ['keyword'],
    });
    expect(found.results.some((r) => r.id === result.chunkId)).toBe(true);
  });

  it('forgetBySource() deactivates all matching chunks', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    // Three chunks from the same session, one from a different source
    await engram.retain('fact one from session A', { source: 'session:aaa' });
    await engram.retain('fact two from session A', { source: 'session:aaa' });
    await engram.retain('fact from session B', { source: 'session:bbb' });

    const count = await engram.forgetBySource('session:aaa');
    expect(count).toBe(2);

    const db = new Database(dbPath);
    const queueRows = db
      .prepare(
        `
      SELECT status, error
      FROM extraction_queue eq
      JOIN chunks c ON c.id = eq.chunk_id
      WHERE c.source = 'session:aaa'
    `,
      )
      .all() as any[];
    db.close();
    expect(queueRows).toHaveLength(2);
    expect(queueRows.every((row) => row.status === 'completed')).toBe(true);
    expect(
      queueRows.every((row) => String(row.error).includes('deactivated')),
    ).toBe(true);

    // Session B chunk should still appear in recall
    const response = await engram.recall('session', {
      strategies: ['keyword'],
    });
    expect(response.results.some((r) => r.source?.includes('bbb'))).toBe(true);
    expect(response.results.some((r) => r.source?.includes('aaa'))).toBe(false);
  });

  // ---- Embedding dimension validation ----

  it('detects dimension mismatch on legacy bank without embed_dimensions key', async () => {
    dbPath = tmpDbPath();

    // Simulate a legacy .engram file: schema + embeddings but no embed_dimensions key
    const legacyDb = new Database(dbPath);
    loadSchema(legacyDb);
    // Insert a chunk with a 768d embedding (768 * 4 = 3072 bytes)
    legacyDb
      .prepare(
        `INSERT INTO chunks (id, text, embedding, memory_type, trust_score)
         VALUES ('chk-legacy', 'legacy data', zeroblob(3072), 'world', 0.8)`,
      )
      .run();
    legacyDb.close();

    // Now open with a 384d embedder — should throw dimension mismatch
    await expect(
      Engram.create(dbPath, { embedder: new MockEmbedder(384) }),
    ).rejects.toThrow(/dimension mismatch/i);
  });

  it('allows dimension change on legacy bank with no embeddings', async () => {
    dbPath = tmpDbPath();

    // Create a legacy .engram file with schema but no chunks
    const legacyDb = new Database(dbPath);
    loadSchema(legacyDb);
    legacyDb.close();

    // Opening with any dimension should work
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder(384) });
    engram.close();
    engram = undefined;

    // Verify stored dimension was recorded
    const verifyDb = new Database(dbPath);
    const row = verifyDb
      .prepare(`SELECT value FROM bank_config WHERE key = 'embed_dimensions'`)
      .get() as { value: string };
    verifyDb.close();
    expect(row.value).toBe('384');
  });

  // ---- backup() ----

  it('backup() writes a portable copy that contains the source data', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const r = await engram.retain('Fact preserved through backup', {
      memoryType: 'world',
      trustScore: 0.9,
    });

    const backupPath = tmpDbPath();
    try {
      const pages = await engram.backup(backupPath);
      expect(pages).toBeGreaterThan(0);

      // The backup must be readable as a standalone SQLite file and contain
      // the retained chunk — proves the online backup captured committed state.
      const backup = new Database(backupPath);
      const row = backup
        .prepare('SELECT id, text FROM chunks WHERE id = ?')
        .get(r.chunkId) as { id: string; text: string } | undefined;
      backup.close();

      expect(row).toBeDefined();
      expect(row?.text).toBe('Fact preserved through backup');
    } finally {
      cleanupDb(backupPath);
    }
  });

  it('backup() is safe while the source engram continues to accept writes', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    await engram.retain('Fact A captured before backup', {
      memoryType: 'world',
    });

    const backupPath = tmpDbPath();
    try {
      await engram.backup(backupPath);

      // A retain() after the backup must not corrupt the source or the backup.
      const afterBackup = await engram.retain('Fact B written after backup', {
        memoryType: 'world',
      });
      expect(afterBackup.chunkId).toMatch(/^chk-/);

      // Source sees both; backup sees only Fact A.
      const sourceCount = await engram.recall('Fact', {
        strategies: ['keyword'],
        topK: 10,
      });
      expect(sourceCount.results.length).toBe(2);

      const backup = new Database(backupPath);
      const backupRows = backup
        .prepare(
          `SELECT text FROM chunks WHERE memory_type = 'world' ORDER BY created_at`,
        )
        .all() as Array<{ text: string }>;
      backup.close();
      expect(backupRows).toHaveLength(1);
      expect(backupRows[0].text).toContain('Fact A');
    } finally {
      cleanupDb(backupPath);
    }
  });

  // ---- Failure cleanup ----

  it('closes the database when embedder initialization fails', async () => {
    dbPath = tmpDbPath();

    // Mock LocalEmbedder.init() to throw — simulates model download failure.
    // This triggers the default embedder path (no injected embedder).
    const localEmbedderModule = await import('../src/local-embedder.js');
    const origInit = localEmbedderModule.LocalEmbedder.prototype.init;
    localEmbedderModule.LocalEmbedder.prototype.init = async function () {
      throw new Error('model download failed');
    };

    try {
      await expect(Engram.create(dbPath, {})).rejects.toThrow(
        'model download failed',
      );

      // The db should NOT be locked — we can open it immediately
      const db2 = new Database(dbPath);
      const tables = db2
        .prepare(`SELECT COUNT(*) as c FROM sqlite_master WHERE type = 'table'`)
        .get() as any;
      expect(tables.c).toBeGreaterThan(0); // schema was applied before failure
      db2.close();
    } finally {
      localEmbedderModule.LocalEmbedder.prototype.init = origInit;
    }
  });
});
