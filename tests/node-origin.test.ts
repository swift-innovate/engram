// =============================================================================
// node-origin.test.ts — Node-Origin Provenance groundwork
//
// Verifies that every durable authored trace records which Engram instance
// wrote it (chunks on retain; opinions + observations on reflect), that the
// origin is stable per-bank, that a pre-distribution .engram upgrades cleanly
// with existing rows left NULL, and that dedup never rewrites the first author.
// =============================================================================

import { describe, it, expect, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { Engram } from '../src/engram.js';
import { retain } from '../src/retain.js';
import {
  MockEmbedder,
  MockGenerator,
  loadSchema,
  tmpDbPath,
  cleanupDb,
  REFLECT_RESPONSE,
} from './helpers.js';

const embedder = new MockEmbedder();

/** Read the bank's configured node_origin directly from bank_config. */
function readBankOrigin(dbPath: string): string | null {
  const db = new Database(dbPath, { readonly: true });
  const row = db
    .prepare(`SELECT value FROM bank_config WHERE key = 'node_origin'`)
    .get() as { value: string } | undefined;
  db.close();
  return row?.value ?? null;
}

describe('node-origin provenance', () => {
  let dbPath: string;
  let engram: Engram | undefined;

  afterEach(() => {
    vi.unstubAllGlobals();
    engram?.close();
    engram = undefined;
    cleanupDb(dbPath);
  });

  it('mints a stable node_origin on first open and never regenerates it', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder });
    const first = readBankOrigin(dbPath);
    expect(first).toBeTruthy();
    // Format: node-<hostslug>-<8 hex>
    expect(first).toMatch(/^node-.+-[0-9a-f]{8}$/);
    engram.close();
    engram = undefined;

    // Reopening the SAME file must read back the identical origin, not mint a new one.
    engram = await Engram.open(dbPath, { embedder });
    const second = readBankOrigin(dbPath);
    expect(second).toBe(first);
  });

  it('stamps a retained chunk with the instance node_origin', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder });
    const origin = readBankOrigin(dbPath);

    const { chunkId } = await engram.retain(
      'Alice prefers Rust for systems work',
      {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      },
    );

    const db = new Database(dbPath, { readonly: true });
    const row = db
      .prepare(`SELECT node_origin FROM chunks WHERE id = ?`)
      .get(chunkId) as { node_origin: string | null };
    db.close();
    expect(row.node_origin).toBe(origin);
  });

  it('stamps opinions and observations formed by a reflect cycle', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, {
      embedder,
      generator: new MockGenerator(REFLECT_RESPONSE),
    });
    const origin = readBankOrigin(dbPath);

    // Enough world facts to clear the default min-facts threshold.
    for (let i = 0; i < 5; i++) {
      await engram.retain(`Alice prefers Rust — fact ${i}`, {
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.8,
      });
    }

    const result = await engram.reflect({
      counterEvidence: false,
      opinionGates: false,
    });
    expect(result.opinionsFormed).toBe(1);
    expect(result.observationsCreated).toBe(1);

    const db = new Database(dbPath, { readonly: true });
    const op = db.prepare(`SELECT node_origin FROM opinions`).get() as {
      node_origin: string | null;
    };
    const obs = db.prepare(`SELECT node_origin FROM observations`).get() as {
      node_origin: string | null;
    };
    db.close();
    expect(op.node_origin).toBe(origin);
    expect(obs.node_origin).toBe(origin);
  });

  it('upgrades a pre-distribution .engram cleanly, leaving legacy rows NULL', async () => {
    dbPath = tmpDbPath();

    // Simulate an .engram authored before origin tracking: current schema, but
    // with the node_origin columns removed and no bank_config origin key.
    const raw = new Database(dbPath);
    loadSchema(raw);
    // Write a legacy chunk while the column still exists (retain's INSERT names it),
    // then drop the columns to reproduce a genuinely pre-migration file on disk.
    await retain(
      raw,
      'A fact recorded before origin tracking existed',
      embedder,
      {
        memoryType: 'world',
      },
    );
    raw.exec('ALTER TABLE chunks DROP COLUMN node_origin');
    raw.exec('ALTER TABLE opinions DROP COLUMN node_origin');
    raw.exec('ALTER TABLE observations DROP COLUMN node_origin');
    raw.close();

    // Opening via Engram must run the guarded migration without crashing.
    engram = await Engram.open(dbPath, { embedder });
    const origin = readBankOrigin(dbPath);
    expect(origin).toBeTruthy();

    const check = new Database(dbPath, { readonly: true });
    // Legacy row: origin unknown → NULL (never falsely claimed as this instance's).
    const legacy = check
      .prepare(
        `SELECT node_origin FROM chunks WHERE text LIKE 'A fact recorded%'`,
      )
      .get() as { node_origin: string | null };
    expect(legacy.node_origin).toBeNull();
    check.close();

    // A write AFTER migration is stamped with the freshly-minted origin.
    const { chunkId } = await engram.retain(
      'A fact recorded after the upgrade',
      {
        memoryType: 'world',
      },
    );
    const check2 = new Database(dbPath, { readonly: true });
    const fresh = check2
      .prepare(`SELECT node_origin FROM chunks WHERE id = ?`)
      .get(chunkId) as { node_origin: string | null };
    check2.close();
    expect(fresh.node_origin).toBe(origin);
  });

  it('does not rewrite node_origin on a dedup hit (first author wins)', async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder });

    const { chunkId } = await engram.retain('Alice prefers Rust', {
      memoryType: 'world',
    });

    // Pretend a DIFFERENT node first authored this chunk.
    const raw = new Database(dbPath);
    raw
      .prepare(`UPDATE chunks SET node_origin = ? WHERE id = ?`)
      .run('node-other-deadbeef', chunkId);
    raw.close();

    // Re-retaining the same normalized text on THIS instance is a dedup hit;
    // the dedup UPDATE must leave the original author's origin intact.
    const again = await engram.retain('alice   prefers   rust', {
      memoryType: 'world',
    });
    expect(again.deduplicated).toBe(true);
    expect(again.chunkId).toBe(chunkId);

    const check = new Database(dbPath, { readonly: true });
    const row = check
      .prepare(`SELECT node_origin FROM chunks WHERE id = ?`)
      .get(chunkId) as { node_origin: string | null };
    check.close();
    expect(row.node_origin).toBe('node-other-deadbeef');
  });
});
