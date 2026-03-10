// =============================================================================
// retain.ts - Memory Encoding (Fast Write Path)
//
// Mirrors biological encoding — sensory input is rapidly stored as a raw
// trace, with consolidation happening in the background.
//
// Implements the two-speed retain strategy:
//   FAST PATH: content → chunk + embedding → SQLite (instant, no LLM)
//   SLOW PATH: queued chunks → Ollama entity extraction → entities/relations (batch)
//
// This avoids the latency problem we hit with mem0+Ollama in Hearthmind
// while still building the knowledge graph over time.
// =============================================================================

import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';

// =============================================================================
// Types
// =============================================================================

export interface RetainOptions {
  /** Memory type classification */
  memoryType?: 'world' | 'experience' | 'observation' | 'opinion';
  /** Source identifier (filename, conversation ID, tool name) */
  source?: string;
  /** Full URI for traceability */
  sourceUri?: string;
  /** Freeform context tag */
  context?: string;
  /** Trust classification */
  sourceType?: 'user_stated' | 'inferred' | 'external_doc' | 'tool_result' | 'agent_generated';
  /** Trust score override (0.0 - 1.0) */
  trustScore?: number;
  /** When the event/fact occurred */
  eventTime?: string;
  /** End of temporal range */
  eventTimeEnd?: string;
  /** Human-readable temporal label */
  temporalLabel?: string;
  /** Skip entity extraction queue (for bulk imports) */
  skipExtraction?: boolean;
}

export interface RetainResult {
  chunkId: string;
  queued: boolean;
}

export interface EmbeddingProvider {
  embed(text: string): Promise<Float32Array>;
  readonly dimensions: number;
}

// =============================================================================
// Ollama Embedding Provider
// =============================================================================

export class OllamaEmbeddings implements EmbeddingProvider {
  private url: string;
  private model: string;
  public readonly dimensions: number;

  constructor(
    url: string = 'http://localhost:11434',
    model: string = 'nomic-embed-text',
    dimensions: number = 768
  ) {
    this.url = url;
    this.model = model;
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<Float32Array> {
    const response = await fetch(`${this.url}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: text }),
    });

    if (!response.ok) {
      throw new Error(`Embedding failed: ${response.status} ${await response.text()}`);
    }

    const data = await response.json();
    // Ollama returns { embeddings: [[...]] } for single input
    const vec = data.embeddings?.[0] || data.embedding;
    return new Float32Array(vec);
  }
}

// =============================================================================
// Fast Retain (no LLM, just embed + store)
// =============================================================================

export async function retain(
  db: Database.Database,
  text: string,
  embedder: EmbeddingProvider,
  options: RetainOptions = {}
): Promise<RetainResult> {
  const {
    memoryType = 'world',
    source = null,
    sourceUri = null,
    context = null,
    sourceType = 'inferred',
    trustScore = 0.5,
    eventTime = null,
    eventTimeEnd = null,
    temporalLabel = null,
    skipExtraction = false,
  } = options;

  const chunkId = `chk-${randomUUID().substring(0, 12)}`;

  // Generate embedding (this is the only async step)
  const embedding = await embedder.embed(text);
  const embeddingBuffer = Buffer.from(embedding.buffer);

  // Write chunk + queue extraction in a single transaction
  const insertTransaction = db.transaction(() => {
    // Insert chunk
    db.prepare(`
      INSERT INTO chunks (
        id, text, embedding, memory_type,
        source, source_uri, context,
        source_type, trust_score,
        event_time, event_time_end, temporal_label
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      chunkId, text, embeddingBuffer, memoryType,
      source, sourceUri, context,
      sourceType, trustScore,
      eventTime, eventTimeEnd, temporalLabel
    );

    // Queue for entity extraction (unless skipped)
    if (!skipExtraction && (memoryType === 'world' || memoryType === 'experience')) {
      db.prepare(`
        INSERT OR IGNORE INTO extraction_queue (chunk_id)
        VALUES (?)
      `).run(chunkId);
    }
  });

  insertTransaction();

  return {
    chunkId,
    queued: !skipExtraction && (memoryType === 'world' || memoryType === 'experience'),
  };
}

// =============================================================================
// Batch Retain (for bulk imports — conversations, documents)
// =============================================================================

export async function retainBatch(
  db: Database.Database,
  items: Array<{ text: string; options?: RetainOptions }>,
  embedder: EmbeddingProvider,
  onProgress?: (current: number, total: number) => void
): Promise<RetainResult[]> {
  const results: RetainResult[] = [];

  for (let i = 0; i < items.length; i++) {
    const { text, options } = items[i];
    const result = await retain(db, text, embedder, {
      ...options,
      skipExtraction: options?.skipExtraction ?? true, // Default skip for batch
    });
    results.push(result);
    onProgress?.(i + 1, items.length);
  }

  // Queue world/experience items that weren't already queued during retain.
  // Cross-reference with original items to check memory type — RetainResult
  // doesn't carry it, but the skip logic in retain() only omits world/experience.
  const toQueue: string[] = [];
  for (let i = 0; i < items.length; i++) {
    if (results[i].queued) continue; // already handled
    const opts = items[i].options;
    if (opts?.skipExtraction) continue; // caller explicitly opted out
    const memType = opts?.memoryType ?? 'world';
    if (memType === 'world' || memType === 'experience') {
      toQueue.push(results[i].chunkId);
    }
  }

  if (toQueue.length > 0) {
    const queueInsert = db.prepare(`
      INSERT OR IGNORE INTO extraction_queue (chunk_id) VALUES (?)
    `);
    const queueTransaction = db.transaction(() => {
      for (const chunkId of toQueue) {
        queueInsert.run(chunkId);
      }
    });
    queueTransaction();
  }

  return results;
}

// =============================================================================
// Entity Extraction (Slow Path - runs in background)
// =============================================================================

const ENTITY_EXTRACTION_PROMPT = `You are an entity extraction engine. Given the text below, extract:

1. **Entities**: People, projects, organizations, technologies, locations, concepts, events, tools
2. **Relations**: Directed relationships between entities

## Text to Analyze
{TEXT}

## Response Format
Respond with ONLY a JSON object (no markdown, no backticks):

{
  "entities": [
    {
      "name": "Display Name",
      "canonical_name": "lowercase normalized",
      "entity_type": "person|project|organization|technology|location|concept|event|tool",
      "aliases": ["alt name 1"]
    }
  ],
  "relations": [
    {
      "source": "canonical_name_1",
      "target": "canonical_name_2",
      "relation_type": "works_on|knows|prefers|owns|depends_on|located_in|part_of|decided|caused|related_to",
      "description": "brief description of the relationship"
    }
  ]
}

If there are no entities or relations, return empty arrays. Be conservative — only extract entities that are clearly identifiable.`;

interface ExtractedEntity {
  name: string;
  canonical_name: string;
  entity_type: string;
  aliases: string[];
}

interface ExtractedRelation {
  source: string;
  target: string;
  relation_type: string;
  description: string;
}

interface ExtractionOutput {
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
}

async function extractEntities(
  text: string,
  ollamaUrl: string,
  model: string
): Promise<ExtractionOutput> {
  const prompt = ENTITY_EXTRACTION_PROMPT.replace('{TEXT}', text);

  const response = await fetch(`${ollamaUrl}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      options: { temperature: 0.1, num_predict: 2048 },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama extraction error: ${response.status}`);
  }

  const data = await response.json();
  const cleaned = data.response
    .replace(/```json\s*/g, '')
    .replace(/```\s*/g, '')
    .trim();

  try {
    const parsed = JSON.parse(cleaned);
    return {
      entities: parsed.entities || [],
      relations: parsed.relations || [],
    };
  } catch {
    return { entities: [], relations: [] };
  }
}

/**
 * Process the extraction queue — call this from a background worker or cron
 */
export async function processExtractionQueue(
  db: Database.Database,
  ollamaUrl: string = 'http://localhost:11434',
  model: string = 'llama3.1:8b',
  batchSize: number = 10
): Promise<{ processed: number; failed: number }> {
  let processed = 0;
  let failed = 0;

  // Grab pending items
  const pending = db.prepare(`
    SELECT eq.chunk_id, c.text
    FROM extraction_queue eq
    JOIN chunks c ON eq.chunk_id = c.id
    WHERE eq.status = 'pending' AND eq.attempts < 3
    ORDER BY eq.queued_at ASC
    LIMIT ?
  `).all(batchSize) as Array<{ chunk_id: string; text: string }>;

  for (const item of pending) {
    // Mark as processing
    db.prepare(`
      UPDATE extraction_queue
      SET status = 'processing', last_attempt = CURRENT_TIMESTAMP, attempts = attempts + 1
      WHERE chunk_id = ?
    `).run(item.chunk_id);

    try {
      const extracted = await extractEntities(item.text, ollamaUrl, model);

      const applyExtraction = db.transaction(() => {
        const now = new Date().toISOString();

        // Upsert entities
        const upsertEntity = db.prepare(`
          INSERT INTO entities (id, name, canonical_name, entity_type, aliases, first_seen_at, last_seen_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(id) DO UPDATE SET
            mention_count = mention_count + 1,
            last_seen_at = excluded.last_seen_at,
            updated_at = CURRENT_TIMESTAMP
        `);

        const linkChunkEntity = db.prepare(`
          INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id, mention_type)
          VALUES (?, ?, 'reference')
        `);

        const entityIdMap: Record<string, string> = {};

        for (const ent of extracted.entities) {
          // Use canonical_name as the stable entity ID basis
          const entityId = `ent-${ent.canonical_name.replace(/[^a-z0-9]/g, '-').substring(0, 30)}`;
          entityIdMap[ent.canonical_name] = entityId;

          upsertEntity.run(
            entityId, ent.name, ent.canonical_name, ent.entity_type,
            JSON.stringify(ent.aliases || []),
            now, now
          );

          linkChunkEntity.run(item.chunk_id, entityId);
        }

        // Insert relations
        const insertRelation = db.prepare(`
          INSERT INTO relations (id, source_entity_id, target_entity_id, relation_type, description, source_chunk_id, confidence)
          VALUES (?, ?, ?, ?, ?, ?, 0.5)
        `);

        for (const rel of extracted.relations) {
          const sourceId = entityIdMap[rel.source];
          const targetId = entityIdMap[rel.target];
          if (sourceId && targetId) {
            const relId = `rel-${randomUUID().substring(0, 8)}`;
            try {
              insertRelation.run(relId, sourceId, targetId, rel.relation_type, rel.description, item.chunk_id);
            } catch {
              // Duplicate edge — ON CONFLICT REPLACE handles this
            }
          }
        }

        // Mark extraction complete
        db.prepare(`
          UPDATE extraction_queue SET status = 'completed' WHERE chunk_id = ?
        `).run(item.chunk_id);
      });

      applyExtraction();
      processed++;

    } catch (error: any) {
      db.prepare(`
        UPDATE extraction_queue SET status = 'failed', error = ? WHERE chunk_id = ?
      `).run(error.message, item.chunk_id);
      failed++;
    }
  }

  return { processed, failed };
}
