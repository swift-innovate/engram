# Engram

A lightweight, zero-infrastructure memory system for AI agents that learns over time.

Each agent gets its own SQLite file — an engram — containing everything it knows, has experienced, believes, and has learned. Raw facts are retained, entities and relationships are extracted, and periodic reflection cycles synthesize higher-order observations and confidence-scored beliefs.

Agents running on Engram don't just remember. They form opinions, refine them with evidence, and get smarter the longer they run.

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

## The Pipeline

Engram mirrors how biological memory works:

1. **Encoding** (retain) — Sensory input is rapidly stored as a raw trace. Fast, no deliberation. Entity extraction happens in the background, the way the hippocampus processes memories during idle time.

2. **Consolidation** (extract) — Background process strengthens the trace by linking it to existing knowledge structures. Entities are resolved, relationships mapped, the knowledge graph densifies.

3. **Reconsolidation** (reflect) — Periodic review of accumulated traces produces higher-order understanding. Observations emerge from patterns. Beliefs form and update with confidence. Old observations get refined as new evidence arrives.

4. **Retrieval** (recall) — Multi-pathway access to stored traces. Semantic similarity (pattern matching), keyword (direct access), graph traversal (associative recall), temporal (episodic memory). Results are fused and weighted by trace strength (trust score).

## Prerequisites

Ollama must be running with the required models before using extraction or reflection:

```bash
ollama pull nomic-embed-text   # embedding model (retain + recall)
ollama pull llama3.1:8b        # extraction + reflection
```

If you run models via **Herd** (swift-innovate/herd), use port `40114` instead of Ollama's default:

```typescript
const mira = await Engram.create('./mira.engram', {
  ollamaUrl: 'http://localhost:40114',  // Herd
});
```

Herd exposes the same Ollama HTTP API (`/api/embed`, `/api/generate`), so no other changes are needed.

## Quick Start

```bash
npm install engram
```

```typescript
import { Engram } from 'engram';

const mira = await Engram.create('./mira.engram', {
  ollamaUrl: 'http://localhost:11434',
  reflectMission: 'Focus on architecture preferences and infrastructure decisions.',
  retainMission: 'Prioritize technical decisions and project context. Ignore greetings.',
});

await mira.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',
  sourceType: 'user_stated',
  trustScore: 0.9,
});

const response = await mira.recall('What IaC tools does Tom use?', { topK: 5 });
console.log(response.results);

mira.close();
```

## API

### `Engram.create(path, options)`

Create a new engram file at the given path. Initializes the schema and applies configuration.

```typescript
const mira = await Engram.create('./mira.engram', {
  ollamaUrl: 'http://localhost:11434',       // Ollama base URL
  reflectMission: string,                    // guides what to synthesize during reflection
  retainMission: string,                     // guides what to prioritize during retention
  embedModel?: string,                       // default: 'nomic-embed-text'
  reflectModel?: string,                     // default: 'llama3.1:8b'
});
```

### `Engram.open(path, options?)`

Open an existing engram file. Use this when resuming a session with an agent that already has memory.

```typescript
const mira = await Engram.open('./mira.engram');
```

### `retain(text, options?)`

Store a memory. Embeds the text and writes to SQLite in ~5ms. Entity extraction is queued for background processing — no LLM call on this path.

```typescript
await mira.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',             // 'world' | 'experience' | 'observation' | 'opinion'
  source?: string,                  // e.g. 'conversation:session-123'
  sourceType?: string,              // 'user_stated' | 'inferred' | 'external_doc' | 'tool_result' | 'agent_generated'
  trustScore?: number,              // 0.0–1.0, default: 0.7
  context?: string,                 // freeform context tag
});
```

**Memory types:**
- `world` — semantic memory: facts about the world
- `experience` — episodic memory: what the agent itself did or observed
- `observation` — consolidated knowledge: patterns across facts (typically written by `reflect`)
- `opinion` — belief with confidence: strengthens or weakens with evidence

### `recall(query, options?)`

Retrieve memories using four parallel strategies: semantic (vector similarity via sqlite-vec), keyword (FTS5 BM25), graph (entity walk), and temporal (date filter). Results are fused using Reciprocal Rank Fusion and weighted by trust score.

```typescript
const response = await mira.recall('What IaC tools does Tom use?', {
  topK?: number,          // default: 10
  memoryType?: string,    // filter by memory type
  minTrust?: number,      // filter by minimum trust score
  since?: Date,           // temporal lower bound
});

// response shape:
// {
//   results: Array<{ text, score, sourceType, trustScore, memoryType, createdAt }>,
//   opinions: Array<{ statement, confidence, evidence }>,
//   observations: Array<{ text, createdAt }>,
// }
```

### `processExtractions()`

Run the background entity extraction queue. Calls Ollama against queued chunks to extract entities and relationships, building out the knowledge graph. Run this periodically or after bulk ingestion.

```typescript
await mira.processExtractions();
```

### `reflect()`

Run a reflection cycle. Reads unreflected chunks, calls Ollama to synthesize observations and update opinion confidence scores. Returns a summary of what was produced.

```typescript
const result = await mira.reflect();
// {
//   observationsCreated: number,
//   opinionsFormed: number,
//   opinionsReinforced: number,
//   chunksProcessed: number,
// }
```

### `close()`

Close the database connection. Call this when the agent session ends.

```typescript
mira.close();
```

## MCP Integration

Expose Engram operations as MCP tools for agent frameworks that support the Model Context Protocol.

```typescript
import { ENGRAM_TOOLS, createEngramToolHandler } from 'engram/mcp-tools';

const handle = createEngramToolHandler(mira);

// Register with your MCP server
server.registerTools(ENGRAM_TOOLS);
server.onToolCall((name, input) => handle(name, input));
```

Tools exposed:

| Tool | Description |
|------|-------------|
| `engram_retain` | Store a memory with source and trust metadata |
| `engram_recall` | Retrieve relevant memories via four-way retrieval |
| `engram_reflect` | Trigger a reflection cycle to synthesize observations and opinions |
| `engram_process_extractions` | Process the entity extraction queue |

## Scheduled Reflection

Run reflection on a timer rather than triggering it manually.

```typescript
import { ReflectScheduler } from 'engram';

const scheduler = new ReflectScheduler({
  dbPath: './mira.engram',
  ollamaUrl: 'http://localhost:11434',
  reflectModel: 'llama3.1:8b',
});

scheduler.start(6 * 60 * 60 * 1000); // every 6 hours
// scheduler.stop();
```

## CLI

Trigger a reflection cycle directly from the command line:

```bash
# Manual reflection
npx tsx src/reflect.ts ./mira.engram

# Custom Ollama endpoint and model
OLLAMA_URL=http://my-server:11434 REFLECT_MODEL=llama3.2:3b npx tsx src/reflect.ts ./mira.engram
```

## Design Decisions

**Fast write / slow extract.** `retain()` never blocks on an LLM call. Embedding and SQLite write take ~5ms. Entity extraction runs asynchronously via Ollama in a background queue. This keeps agents responsive regardless of Ollama latency.

**Trust layer.** Every chunk carries `source_type` and `trust_score`. The recall pipeline weights results by trust. External documents and tool results cannot override core agent directives regardless of their trust score.

**SQLite over Postgres.** One file per agent — portable, git-committable, backup-friendly. sqlite-vec handles vector search; FTS5 handles keyword search. No Docker containers, no infrastructure dependencies.

**Batch reflect over real-time.** Reflection runs on a schedule, not on every query. This keeps Ollama usage predictable while still building observations and opinions over time.

## File Format

Agent memory files use the `.engram` extension. They are standard SQLite databases and can be inspected with any SQLite tooling.

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
| Ollama | Local LLM for extraction + reflection | Yes (for extract + reflect) |
| nomic-embed-text | Embedding model via Ollama | Yes (for retain + recall) |
| llama3.1:8b | Fast model for extraction + reflection | Recommended |

## Integration with valor-engine

```typescript
// valor-engine/engine/memory/index.ts
import { Engram } from 'engram';

const miraMemory = await Engram.open('./mira.engram');

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

## Development

```bash
npm install
npm run build        # compile TypeScript → dist/
npm test             # run test suite (52 tests)
npm run typecheck    # TypeScript check without emit
npm run example      # run examples/basic-usage.ts (requires Ollama)
```

## License

Apache 2.0
