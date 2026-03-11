// =============================================================================
// reflect.ts - The Learning Engine (Reconsolidation)
// 
// Implements the "reflect" operation for Engram
// Mirrors biological memory reconsolidation — periodic review of
// accumulated traces produces higher-order understanding.
// Runs on a schedule or on-demand to:
//   1. Process unreflected facts and experiences
//   2. Synthesize observations (consolidated knowledge)
//   3. Form or update opinions (beliefs with confidence)
//   4. Log everything for auditability
//
// Dependencies: Ollama (local LLM), SQLite (via better-sqlite3)
// =============================================================================

import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';
import { pathToFileURL } from 'url';

// =============================================================================
// Types
// =============================================================================

export interface ReflectConfig {
  /** Path to the agent's SQLite memory file */
  dbPath: string;
  /** Ollama endpoint (default: http://starbase:40114) */
  ollamaUrl?: string;
  /** Model for reflection (default: llama3.1:8b — fast, good enough for synthesis) */
  reflectModel?: string;
  /** Max unreflected facts to process per cycle (default: 50) */
  batchSize?: number;
  /** Min facts needed to trigger reflection (default: 5) */
  minFactsThreshold?: number;
}

interface Chunk {
  id: string;
  text: string;
  memory_type: string;
  source: string | null;
  context: string | null;
  event_time: string | null;
  created_at: string;
}

interface Observation {
  id: string;
  summary: string;
  source_chunks: string[];
  source_entities: string[];
  domain: string | null;
  topic: string | null;
}

interface Opinion {
  id: string;
  belief: string;
  confidence: number;
  supporting_chunks: string[];
  domain: string | null;
  related_entities: string[];
}

interface ExistingOpinion extends Opinion {
  contradicting_chunks: string[];
  evidence_count: number;
}

export interface ReflectResult {
  logId: string;
  factsProcessed: number;
  observationsCreated: number;
  observationsUpdated: number;
  opinionsFormed: number;
  opinionsReinforced: number;
  opinionsChallenged: number;
  status: 'completed' | 'failed' | 'partial';
  durationMs: number;
  error?: string;
}

interface LLMReflectOutput {
  observations: Array<{
    summary: string;
    domain: string;
    topic: string;
    source_chunk_ids: string[];
    entity_names: string[];
  }>;
  opinion_updates: Array<{
    belief: string;
    direction: 'reinforce' | 'challenge' | 'new';
    confidence_delta: number;  // +/- adjustment
    domain: string;
    evidence_chunk_ids: string[];
    entity_names: string[];
  }>;
  observation_refreshes: Array<{
    existing_observation_id: string;
    updated_summary: string;
    new_source_chunk_ids: string[];
  }>;
}

// =============================================================================
// Prompt Templates
// =============================================================================

function buildReflectPrompt(
  unreflectedFacts: Chunk[],
  existingObservations: Observation[],
  existingOpinions: ExistingOpinion[],
  bankConfig: Record<string, string>
): string {
  const reflectMission = bankConfig['reflect_mission'] || 
    'Identify patterns, consolidate related facts, and form beliefs about preferences and approaches.';
  
  const disposition = bankConfig['disposition'] 
    ? JSON.parse(bankConfig['disposition']) 
    : { skepticism: 0.5, literalism: 0.5, empathy: 0.5 };

  const factsBlock = unreflectedFacts.map(f => {
    const timeStr = f.event_time ? ` [${f.event_time}]` : '';
    const ctxStr = f.context ? ` (context: ${f.context})` : '';
    const srcStr = f.source ? ` [source: ${f.source}]` : '';
    return `  - [${f.id}] (${f.memory_type})${timeStr}${ctxStr}${srcStr}: ${f.text}`;
  }).join('\n');

  const obsBlock = existingObservations.length > 0
    ? existingObservations.map(o => 
        `  - [${o.id}] (${o.domain}/${o.topic}): ${o.summary}`
      ).join('\n')
    : '  (none yet)';

  const opBlock = existingOpinions.length > 0
    ? existingOpinions.map(o =>
        `  - [${o.id}] (confidence: ${o.confidence.toFixed(2)}, domain: ${o.domain}): ${o.belief}`
      ).join('\n')
    : '  (none yet)';

  return `You are the reflection engine for an AI agent's memory system. Your job is to analyze recent memories and produce structured insights.

## Your Mission
${reflectMission}

## Your Disposition
- Skepticism: ${disposition.skepticism} (0=trusting, 1=highly skeptical)
- Literalism: ${disposition.literalism} (0=creative interpretation, 1=strict facts only)
- Empathy: ${disposition.empathy} (0=purely analytical, 1=highly empathetic)

## Recent Unreflected Memories
These are facts and experiences that have not yet been analyzed:

${factsBlock}

## Existing Observations
These are previously synthesized observations. You may update them if new facts are relevant:

${obsBlock}

## Existing Opinions
These are current beliefs with confidence scores. You may reinforce or challenge them:

${opBlock}

## Instructions

Analyze the unreflected memories and produce:

1. **New Observations**: Consolidate related facts into higher-order understanding. An observation synthesizes multiple facts into a pattern or summary that would be useful for future reasoning. Only create observations when you see genuine patterns across 2+ facts — do not simply rephrase individual facts.

2. **Observation Refreshes**: If new facts add to or modify an existing observation, provide an updated summary. Reference the existing observation ID.

3. **Opinion Updates**: 
   - "new": Form a new belief when evidence suggests a preference, pattern, or approach. Set initial confidence between 0.3-0.7 depending on evidence strength.
   - "reinforce": Increase confidence in an existing opinion when new evidence supports it. Provide a positive confidence_delta (max +0.15 per cycle).
   - "challenge": Decrease confidence when evidence contradicts an existing opinion. Provide a negative confidence_delta (max -0.15 per cycle).

## Response Format

Respond with ONLY a JSON object (no markdown, no backticks, no preamble):

{
  "observations": [
    {
      "summary": "Concise synthesized observation",
      "domain": "architecture|preferences|workflow|people|projects|infrastructure|creative|general",
      "topic": "specific topic within domain",
      "source_chunk_ids": ["chunk-id-1", "chunk-id-2"],
      "entity_names": ["EntityName1", "EntityName2"]
    }
  ],
  "opinion_updates": [
    {
      "belief": "Clear statement of the belief",
      "direction": "new|reinforce|challenge",
      "confidence_delta": 0.1,
      "domain": "same domain list as observations",
      "evidence_chunk_ids": ["chunk-id"],
      "entity_names": ["EntityName"]
    }
  ],
  "observation_refreshes": [
    {
      "existing_observation_id": "obs-id",
      "updated_summary": "Updated observation text incorporating new evidence",
      "new_source_chunk_ids": ["chunk-id"]
    }
  ]
}

If you find no meaningful patterns, return empty arrays. Do not force observations or opinions — quality over quantity.`;
}

// =============================================================================
// Ollama Client (minimal, no external deps beyond fetch)
// =============================================================================

async function ollamaGenerate(
  url: string, 
  model: string, 
  prompt: string
): Promise<string> {
  const response = await fetch(`${url}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      options: {
        temperature: 0.3,     // Low temp for analytical work
        num_predict: 4096,    // Enough for structured output
      }
    })
  });

  if (!response.ok) {
    throw new Error(`Ollama error ${response.status}: ${await response.text()}`);
  }

  const data = await response.json();
  return data.response;
}

function parseReflectOutput(raw: string): LLMReflectOutput {
  // Strip any markdown fencing the model might add despite instructions
  const cleaned = raw
    .replace(/```json\s*/g, '')
    .replace(/```\s*/g, '')
    .trim();
  
  try {
    const parsed = JSON.parse(cleaned);
    return {
      observations: parsed.observations || [],
      opinion_updates: parsed.opinion_updates || [],
      observation_refreshes: parsed.observation_refreshes || [],
    };
  } catch (e) {
    throw new Error(`Failed to parse reflect output: ${e}\nRaw: ${cleaned.substring(0, 500)}`);
  }
}

// =============================================================================
// Database Helpers
// =============================================================================

function getUnreflectedFacts(db: Database.Database, limit: number): Chunk[] {
  return db.prepare(`
    SELECT id, text, memory_type, source, context, event_time, created_at
    FROM chunks
    WHERE reflected_at IS NULL
      AND is_active = TRUE
      AND memory_type IN ('world', 'experience')
    ORDER BY created_at ASC
    LIMIT ?
  `).all(limit) as Chunk[];
}

function getExistingObservations(db: Database.Database, limit: number = 20): Observation[] {
  const rows = db.prepare(`
    SELECT id, summary, source_chunks, source_entities, domain, topic
    FROM observations
    WHERE is_active = TRUE
    ORDER BY last_refreshed DESC, synthesized_at DESC
    LIMIT ?
  `).all(limit) as any[];

  return rows.map(r => ({
    ...r,
    source_chunks: JSON.parse(r.source_chunks || '[]'),
    source_entities: JSON.parse(r.source_entities || '[]'),
  }));
}

function getExistingOpinions(db: Database.Database, limit: number = 20): ExistingOpinion[] {
  const rows = db.prepare(`
    SELECT id, belief, confidence, supporting_chunks, contradicting_chunks,
           evidence_count, domain, related_entities
    FROM opinions
    WHERE is_active = TRUE
    ORDER BY confidence DESC, evidence_count DESC
    LIMIT ?
  `).all(limit) as any[];

  return rows.map(r => ({
    ...r,
    supporting_chunks: JSON.parse(r.supporting_chunks || '[]'),
    contradicting_chunks: JSON.parse(r.contradicting_chunks || '[]'),
    related_entities: JSON.parse(r.related_entities || '[]'),
  }));
}

function getBankConfig(db: Database.Database): Record<string, string> {
  const rows = db.prepare('SELECT key, value FROM bank_config').all() as { key: string; value: string }[];
  const config: Record<string, string> = {};
  for (const row of rows) {
    config[row.key] = row.value;
  }
  return config;
}

function normalizeBelief(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function beliefSimilarity(a: string, b: string): number {
  const aNorm = normalizeBelief(a);
  const bNorm = normalizeBelief(b);

  if (!aNorm || !bNorm) return 0;
  if (aNorm === bNorm) return 1;

  const aTokens = new Set(aNorm.split(' ').filter(Boolean));
  const bTokens = new Set(bNorm.split(' ').filter(Boolean));
  const intersection = [...aTokens].filter(token => bTokens.has(token)).length;
  return intersection / Math.max(aTokens.size, bTokens.size, 1);
}

function findMatchingOpinion(
  existingOpinions: ExistingOpinion[],
  belief: string,
  domain: string
): ExistingOpinion | undefined {
  const exact = existingOpinions.find(op =>
    op.domain === domain && normalizeBelief(op.belief) === normalizeBelief(belief)
  );
  if (exact) return exact;

  let best: { opinion: ExistingOpinion; score: number } | undefined;
  for (const opinion of existingOpinions) {
    if (opinion.domain !== domain) continue;
    const score = beliefSimilarity(opinion.belief, belief);
    if (score < 0.85) continue;
    if (!best || score > best.score) {
      best = { opinion, score };
    }
  }

  return best?.opinion;
}

// =============================================================================
// Core Reflect Operation
// =============================================================================

export async function reflect(config: ReflectConfig): Promise<ReflectResult> {
  const {
    dbPath,
    ollamaUrl = 'http://starbase:40114',
    reflectModel = 'llama3.1:8b',
    batchSize = 50,
    minFactsThreshold = 5,
  } = config;

  const db = new Database(dbPath);
  const startTime = Date.now();
  const logId = randomUUID();

  // Start the reflect log entry
  db.prepare(`
    INSERT INTO reflect_log (id, status, model_used)
    VALUES (?, 'running', ?)
  `).run(logId, reflectModel);

  const result: ReflectResult = {
    logId,
    factsProcessed: 0,
    observationsCreated: 0,
    observationsUpdated: 0,
    opinionsFormed: 0,
    opinionsReinforced: 0,
    opinionsChallenged: 0,
    status: 'completed',
    durationMs: 0,
  };

  try {
    // 1. Gather unreflected facts
    const unreflected = getUnreflectedFacts(db, batchSize);
    
    if (unreflected.length < minFactsThreshold) {
      result.status = 'completed';
      result.durationMs = Date.now() - startTime;
      db.prepare(`
        UPDATE reflect_log 
        SET completed_at = CURRENT_TIMESTAMP, status = 'completed',
            facts_processed = 0
        WHERE id = ?
      `).run(logId);
      db.close();
      return result;
    }

    // 2. Load existing context
    const existingObs = getExistingObservations(db);
    const existingOps = getExistingOpinions(db);
    const bankConfig = getBankConfig(db);

    // 3. Build prompt and call LLM
    const prompt = buildReflectPrompt(unreflected, existingObs, existingOps, bankConfig);
    const rawResponse = await ollamaGenerate(ollamaUrl, reflectModel, prompt);
    const output = parseReflectOutput(rawResponse);

    // 4. Apply results in a transaction
    const applyTransaction = db.transaction(() => {
      const now = new Date().toISOString();

      // -- New Observations --
      const insertObs = db.prepare(`
        INSERT INTO observations (id, summary, source_chunks, source_entities, domain, topic, synthesized_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `);
      for (const obs of output.observations) {
        const obsId = `obs-${randomUUID().substring(0, 8)}`;
        insertObs.run(
          obsId,
          obs.summary,
          JSON.stringify(obs.source_chunk_ids),
          JSON.stringify(obs.entity_names),
          obs.domain,
          obs.topic,
          now
        );
        result.observationsCreated++;
      }

      // -- Observation Refreshes --
      const updateObsSimple = db.prepare(`
        UPDATE observations
        SET summary = ?,
            source_chunks = ?,
            last_refreshed = ?,
            refresh_count = refresh_count + 1
        WHERE id = ?
      `);
      for (const refresh of output.observation_refreshes) {
        const existing = existingObs.find(o => o.id === refresh.existing_observation_id);
        if (existing) {
          const mergedSources = [...new Set([
            ...existing.source_chunks,
            ...refresh.new_source_chunk_ids
          ])];
          updateObsSimple.run(
            refresh.updated_summary,
            JSON.stringify(mergedSources),
            now,
            refresh.existing_observation_id
          );
          result.observationsUpdated++;
        }
      }

      // -- Opinion Updates --
      const insertOpinion = db.prepare(`
        INSERT INTO opinions (id, belief, confidence, supporting_chunks, domain, related_entities, formed_at, evidence_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `);
      const reinforceOpinion = db.prepare(`
        UPDATE opinions
        SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
            supporting_chunks = ?,
            evidence_count = evidence_count + 1,
            last_reinforced = ?,
            updated_at = ?
        WHERE belief = ? AND is_active = TRUE
      `);
      const challengeOpinion = db.prepare(`
        UPDATE opinions
        SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
            contradicting_chunks = ?,
            evidence_count = evidence_count + 1,
            last_challenged = ?,
            updated_at = ?
        WHERE belief = ? AND is_active = TRUE
      `);

      for (const opUpdate of output.opinion_updates) {
        if (opUpdate.direction === 'new') {
          const initialConfidence = Math.min(0.7, Math.max(0.3, 
            0.5 + opUpdate.confidence_delta
          ));
          insertOpinion.run(
            `op-${randomUUID().substring(0, 8)}`,
            opUpdate.belief,
            initialConfidence,
            JSON.stringify(opUpdate.evidence_chunk_ids),
            opUpdate.domain,
            JSON.stringify(opUpdate.entity_names),
            now,
            opUpdate.evidence_chunk_ids.length
          );
          result.opinionsFormed++;

        } else if (opUpdate.direction === 'reinforce') {
          const existing = findMatchingOpinion(existingOps, opUpdate.belief, opUpdate.domain);
          if (existing) {
            const clampedDelta = Math.min(0.15, Math.max(0, opUpdate.confidence_delta));
            const mergedSupporting = [...new Set([
              ...existing.supporting_chunks,
              ...opUpdate.evidence_chunk_ids
            ])];
            reinforceOpinion.run(
              clampedDelta,
              JSON.stringify(mergedSupporting),
              now, now,
              existing.belief
            );
            result.opinionsReinforced++;
          }

        } else if (opUpdate.direction === 'challenge') {
          const existing = findMatchingOpinion(existingOps, opUpdate.belief, opUpdate.domain);
          if (existing) {
            const clampedDelta = Math.max(-0.15, Math.min(0, opUpdate.confidence_delta));
            const mergedContradicting = [...new Set([
              ...existing.contradicting_chunks,
              ...opUpdate.evidence_chunk_ids
            ])];
            challengeOpinion.run(
              clampedDelta,
              JSON.stringify(mergedContradicting),
              now, now,
              existing.belief
            );
            result.opinionsChallenged++;
          }
        }
      }

      // -- Mark facts as reflected --
      const markReflected = db.prepare(`
        UPDATE chunks SET reflected_at = ? WHERE id = ?
      `);
      for (const fact of unreflected) {
        markReflected.run(now, fact.id);
      }
      result.factsProcessed = unreflected.length;
    });

    applyTransaction();

    // 5. Update reflect log
    result.durationMs = Date.now() - startTime;
    db.prepare(`
      UPDATE reflect_log
      SET completed_at = CURRENT_TIMESTAMP,
          status = 'completed',
          facts_processed = ?,
          observations_created = ?,
          observations_updated = ?,
          opinions_formed = ?,
          opinions_reinforced = ?,
          opinions_challenged = ?
      WHERE id = ?
    `).run(
      result.factsProcessed,
      result.observationsCreated,
      result.observationsUpdated,
      result.opinionsFormed,
      result.opinionsReinforced,
      result.opinionsChallenged,
      logId
    );

  } catch (error: any) {
    result.status = 'failed';
    result.error = error.message;
    result.durationMs = Date.now() - startTime;
    
    db.prepare(`
      UPDATE reflect_log
      SET completed_at = CURRENT_TIMESTAMP, status = 'failed', error = ?
      WHERE id = ?
    `).run(error.message, logId);
    
  } finally {
    db.close();
  }

  return result;
}

// =============================================================================
// Scheduled Reflect Runner
// =============================================================================

export class ReflectScheduler {
  private config: ReflectConfig;
  private intervalHandle: ReturnType<typeof setInterval> | null = null;
  private isRunning = false;

  constructor(config: ReflectConfig) {
    this.config = config;
  }

  /**
   * Start periodic reflection cycles.
   * @param intervalMs - How often to run (default: 6 hours)
   */
  start(intervalMs: number = 6 * 60 * 60 * 1000): void {
    if (this.intervalHandle) {
      console.warn('[Reflect] Scheduler already running');
      return;
    }

    console.log(`[Reflect] Starting scheduler (every ${intervalMs / 1000}s) for ${this.config.dbPath}`);
    
    this.intervalHandle = setInterval(async () => {
      if (this.isRunning) {
        console.log('[Reflect] Previous cycle still running, skipping');
        return;
      }
      await this.runOnce();
    }, intervalMs);

    // Run once immediately on start
    this.runOnce();
  }

  stop(): void {
    if (this.intervalHandle) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
      console.log('[Reflect] Scheduler stopped');
    }
  }

  async runOnce(): Promise<ReflectResult | null> {
    if (this.isRunning) return null;
    
    this.isRunning = true;
    try {
      console.log(`[Reflect] Starting cycle for ${this.config.dbPath}`);
      const result = await reflect(this.config);
      console.log(
        `[Reflect] Cycle complete: ${result.factsProcessed} facts → ` +
        `${result.observationsCreated} new obs, ${result.observationsUpdated} updated obs, ` +
        `${result.opinionsFormed} new opinions, ${result.opinionsReinforced} reinforced, ` +
        `${result.opinionsChallenged} challenged (${result.durationMs}ms)`
      );
      return result;
    } catch (error) {
      console.error('[Reflect] Cycle failed:', error);
      return null;
    } finally {
      this.isRunning = false;
    }
  }
}

// =============================================================================
// CLI Entry Point (for manual runs)
// =============================================================================

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const dbPath = process.argv[2];
  if (!dbPath) {
    console.error('Usage: npx tsx reflect.ts <path-to-memory.sqlite>');
    process.exit(1);
  }

  const ollamaUrl = process.env.OLLAMA_URL || 'http://starbase:40114';
  const model = process.env.REFLECT_MODEL || 'llama3.1:8b';

  console.log(`[Reflect] Manual run: ${dbPath} via ${model} @ ${ollamaUrl}`);
  
  reflect({ dbPath, ollamaUrl, reflectModel: model })
    .then(result => {
      console.log('[Reflect] Result:', JSON.stringify(result, null, 2));
      process.exit(result.status === 'completed' ? 0 : 1);
    })
    .catch(err => {
      console.error('[Reflect] Fatal:', err);
      process.exit(1);
    });
}
