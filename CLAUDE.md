# Engram — Memory Traces for AI Agents

> *A memory trace physically encoded in neural tissue that strengthens with reinforcement.*

## What This Is

Engram is a lightweight, zero-infrastructure memory system for AI agents that **learns** over time. Each agent gets its own SQLite file — an engram — containing everything it knows, has experienced, believes, and has learned. Raw facts are retained, entities and relationships are extracted, and periodic reflection cycles synthesize higher-order observations and confidence-scored beliefs.

Agents running on Engram don't just remember. They form opinions, refine them with evidence, and get smarter the longer they run.

## Repo

`swift-innovate/engram` — Apache 2.0 (or BSL 1.1 — TBD)

## Design Heritage

Engram synthesizes ideas from four prior efforts:

- **Hindsight** (vectorize-io): Four memory networks (world/experience/observation/opinion), retain/recall/reflect operations, bank-level configuration, confidence-scored beliefs. We took the cognitive architecture and dropped the Postgres dependency.
- **memory-ragkg** (our OpenClaw plugin design): SQLite-based knowledge graph with entities/relations tables, trust/provenance layer, hybrid query router. We carried the trust layer forward — neither Hindsight nor Memvid has it.
- **Hearthmind/Mira-Memory**: Lessons learned from mem0 integration pain — Ollama connectivity issues, write latency blocking agents, metadata restrictions. The fast-write/slow-extract pattern exists because of this.
- **Memvid**: Single-file portability inspiration. One file per agent, git-committable, zero infrastructure.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                           ENGRAM                                  │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RETAIN (fast write)              RECALL (four-way retrieval)     │
│  ┌─────────────────┐              ┌──────────────────────────┐    │
│  │ text → embed     │              │ Semantic (sqlite-vec)    │    │
│  │ → SQLite chunk   │              │ Keyword  (FTS5 BM25)    │    │
│  │ → queue extract  │              │ Graph    (entity walk)   │    │
│  └────────┬────────┘              │ Temporal (date filter)   │    │
│           │                       └────────────┬─────────────┘    │
│           ▼                                    ▼                  │
│  EXTRACT (background)             RRF + Trust Weighting           │
│  ┌─────────────────┐              ┌──────────────────────────┐    │
│  │ Ollama → entities│              │ Ranked results +         │    │
│  │ → relations      │              │ opinions + observations  │    │
│  │ → chunk_entities │              └──────────────────────────┘    │
│  └─────────────────┘                                              │
│                                                                   │
│  REFLECT (scheduled learning)                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ unreflected facts → Ollama → observations + opinions     │     │
│  │ (batch, not real-time — avoids latency on every write)   │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │               <agentId>.engram (SQLite)                   │     │
│  │  chunks | entities | relations | chunk_entities           │     │
│  │  opinions | observations | reflect_log | extraction_queue │     │
│  │  bank_config | chunks_fts (FTS5)                          │     │
│  └──────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

## The Engram Pipeline (Biological Metaphor)

The system mirrors how biological memory works:

1. **Encoding** (retain) — Sensory input is rapidly stored as a raw trace. Fast, no deliberation. Entity extraction happens in the background, the way the hippocampus processes memories during idle time.

2. **Consolidation** (extract) — Background process strengthens the trace by linking it to existing knowledge structures. Entities are resolved, relationships mapped, the knowledge graph densifies.

3. **Reconsolidation** (reflect) — Periodic review of accumulated traces produces higher-order understanding. Observations emerge from patterns. Beliefs form and update with confidence. Old observations get refined as new evidence arrives. This is the learning loop.

4. **Retrieval** (recall) — Multi-pathway access to stored traces. Semantic similarity (pattern matching), keyword (direct access), graph traversal (associative recall), temporal (episodic memory). Results are fused and weighted by trace strength (trust score).

## Key Design Decisions

### Fast Write / Slow Extract
The #1 lesson from Hearthmind: never block writes on LLM calls. `retain()` embeds and stores in ~5ms. Entity extraction runs in a background queue via Ollama. The knowledge graph builds up over time without impacting agent responsiveness.

### Trust Layer (Unique to Engram)
Neither Memvid nor Hindsight tracks provenance. Every chunk carries `source_type` (user_stated, inferred, external_doc, tool_result, agent_generated) and `trust_score` (0.0-1.0). The recall pipeline weights results by trust. Security rule: external docs and tool results can NEVER override core agent directives regardless of trust score.

### SQLite Over Postgres
Hindsight uses embedded Postgres. Engram uses SQLite because:
- Proven in our OpenClaw memory-ragkg work
- Single file per agent = portable, git-committable, backup-friendly
- sqlite-vec for vector search, FTS5 for keyword search
- Zero Docker containers required
- Runs on the homelab without additional services

### Batch Reflect Over Real-Time
Hindsight reflects on every query. Engram reflects on a schedule (default: manual trigger, configurable to hourly/daily). This keeps Ollama usage bounded while still building observations and opinions over time.

### Four Memory Types
Inspired by Hindsight's four networks, mapped to biological memory:
- **World** = semantic memory (facts about the world)
- **Experience** = episodic memory (what the agent itself did)
- **Observation** = consolidated knowledge (patterns across facts)
- **Opinion** = belief with confidence (strengthens or weakens with evidence)

## File Extension

Agent memory files use the `.engram` extension (they're SQLite files):
```
~/.valor/memory/mira.engram
~/.valor/memory/sit-agent.engram
~/.valor/memory/shared.engram
```

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| better-sqlite3 | SQLite driver for Node.js | Yes |
| sqlite-vec | Vector similarity search extension | Yes (for semantic recall) |
| @xenova/transformers | In-process embeddings (retain + recall) | Yes (default embedding path) |
| Ollama | Local LLM for extraction + reflection | Yes (for extract + reflect) |
| llama3.1:8b | Fast model for extraction + reflection | Recommended |
| nomic-embed-text (Ollama) | Ollama embedding model | Only when `useOllamaEmbeddings: true` |

**Herd alternative:** swift-innovate/herd exposes the same Ollama HTTP API on port `40114`. Use `ollamaUrl: 'http://localhost:40114'` to point Engram at Herd instead.

## File Structure

```
engram/
├── CLAUDE.md           ← you are here
├── README.md           ← public-facing docs
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── src/
│   ├── schema.sql           ← full database schema
│   ├── retain.ts            ← fast write + dedup + batch import + extraction queue
│   ├── recall.ts            ← four-way retrieval + RRF + trust/decay weighting + formatForPrompt
│   ├── reflect.ts           ← scheduled learning engine + prompt templates
│   ├── local-embedder.ts    ← in-process embeddings via @xenova/transformers
│   ├── engram.ts            ← unified Engram class + public API exports
│   └── mcp-tools.ts         ← MCP tool definitions (retain/recall/reflect/forget/supersede)
├── tests/
│   ├── helpers.ts
│   ├── retain.test.ts
│   ├── retain-gate.test.ts
│   ├── recall.test.ts
│   ├── reflect.test.ts
│   └── engram.test.ts
└── examples/
    └── basic-usage.ts
```

## Usage

### Initialize an engram
```typescript
import { Engram } from 'engram';

const mira = await Engram.create('./mira.engram', {
  reflectMission: 'Focus on architecture preferences, project patterns, and infrastructure decisions.',
  retainMission: 'Prioritize technical decisions, code patterns, and project context. Ignore greetings.',
  ollamaUrl: 'http://starbase:40114',
});
```

### Store a memory
```typescript
await mira.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',
  source: 'conversation:valor-engine-planning',
  sourceType: 'user_stated',
  trustScore: 0.9,
  context: 'infrastructure',
});
```

### Recall memories
```typescript
const response = await mira.recall('What IaC tools does Tom use?', { topK: 5 });
// response.results   → ranked chunks from four retrieval strategies
// response.opinions  → relevant beliefs with confidence scores
// response.observations → synthesized knowledge
```

### Run reflection
```bash
# CLI
npx engram reflect ./mira.engram

# Or programmatically
const result = await mira.reflect();
// result: { observationsCreated: 3, opinionsFormed: 1, opinionsReinforced: 2, ... }
```

### Process entity extraction queue
```typescript
await mira.processExtractions();
// Runs Ollama against queued chunks, builds out entity graph
```

## Integration with valor-engine

Engram is a standalone library. valor-engine consumes it as a dependency:

```typescript
// valor-engine/engine/memory/index.ts
import { Engram } from 'engram';

// One engram per operative
const miraMemory = await Engram.open('./mira.engram');
const sitMemory = await Engram.open('./sit-agent.engram');

// Auto-retain conversation turns
gateway.on('message', async (msg) => {
  await miraMemory.retain(msg.text, {
    memoryType: 'experience',
    source: `conversation:${msg.conversationId}`,
    sourceType: 'user_stated',
    trustScore: 0.9,
  });
});

// Inject context before LLM calls
const context = await miraMemory.recall(userMessage, { topK: 10 });
const systemPrompt = buildPromptWithMemory(basePrompt, context);
```

## Git

```bash
git init && git branch -M main
```

## Decisions

- [x] **License**: Apache 2.0. Set in `package.json`. Maximizes adoption for a utility library. Revisit to BSL 1.1 only if commercial protection becomes a requirement.

- [x] **sqlite-vec loading**: Prebuilt binaries via npm (`sqlite-vec` package). No compile-from-source step. The `Engram` class loads it with a `try/catch` — graceful fallback to 3-strategy recall if absent.

- [x] **Shared engram**: Naming convention. Multiple agents call `Engram.open('./shared.engram')`. SQLite WAL mode supports concurrent readers + one writer — concurrent `retain()` from multiple agents is safe as long as calls aren't simultaneous. No separate API needed.

- [x] **MCP tool granularity**: All four operations exposed (`engram_retain`, `engram_recall`, `engram_reflect`, `engram_process_extractions`). Agents that only need recall+retain simply ignore the other two tools.

- [x] **Reflect schedule**: Library default is manual (call `engram.reflect()` or the CLI). `ReflectScheduler` class ships for timer-based use. Recommendation for valor-engine: `ReflectScheduler` with a 6-hour default, configurable per operative.

- [x] **Embedding model**: `Xenova/nomic-embed-text-v1.5` (768d) runs in-process via `@xenova/transformers` as default — no Ollama required for retain/recall. Override via `embedModel` option. `Xenova/all-MiniLM-L6-v2` (384d) is a valid alternative for lower disk/memory use. Opt into Ollama embeddings via `useOllamaEmbeddings: true` (e.g., for GPU acceleration). Existing `.engram` files with Ollama-generated vectors are fully compatible — same model weights, same 768-dim space.

- [ ] **`.engram` MIME type**: Deferred. Extension is established; OS MIME registration is future work if IDE/tooling support becomes valuable.
