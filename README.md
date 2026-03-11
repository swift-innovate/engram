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

Ollama is required for entity extraction and reflection (LLM tasks). Embeddings run locally via Transformers.js — no Ollama needed for `retain()` or `recall()`.

```bash
ollama pull llama3.1:8b        # required for processExtractions() and reflect()
```

The embedding model (`Xenova/nomic-embed-text-v1.5`) downloads automatically on first use to `~/.cache/xenova/` (~30MB). No manual pull needed.

If you run models via **Herd** (swift-innovate/herd), use port `40114` instead of Ollama's default:

```typescript
const mira = await Engram.create('./mira.engram', {
  ollamaUrl: 'http://localhost:40114',  // Herd
});
```

Herd exposes the same Ollama HTTP API (`/api/embed`, `/api/generate`), so no other changes are needed.

### Verifying Your Setup

Before integrating Engram into an agent, verify the embedding model is reachable. A common failure mode is Engram opening successfully (`Engram.open()` only touches SQLite) while the embedding pipeline silently fails on every `retain()` call because the Ollama model isn't available.

**Quick verification (default local embeddings path):**

```typescript
// Embeddings are local — just verify the library loads correctly
import { LocalEmbedder } from 'engram';
const e = new LocalEmbedder();
await e.init(); // downloads model on first run, loads from cache after
const v = await e.embed('test');
console.log(v.length); // 768
```

**If using `useOllamaEmbeddings: true`:**

```bash
# Check if nomic-embed-text is available
curl http://localhost:11434/api/tags | grep nomic-embed-text

# Test an embedding directly
curl http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"test"}'
```

**Programmatic health check:**

```typescript
const engram = await Engram.open('./agent.engram', { ollamaUrl: 'http://localhost:11434' });

// Verify the embedding pipeline works end-to-end
try {
  const result = await engram.retain('__health_check__', {
    memoryType: 'world',
    source: 'health_check',
    trustScore: 0.0,
    skipExtraction: true,
  });
  await engram.forget(result.chunkId);
  console.log('✓ Engram healthy — embedding model reachable');
} catch (err) {
  console.error('✗ Engram broken — embedding model unreachable:', err.message);
  // Disable Engram for this session rather than running with silent failures
}
```

**Why this matters:** `Engram.open()` succeeds even when the embedding model doesn't exist — it only fails on the first actual `retain()` call. Without a health check, your agent can run for hours with "Engram: active" in its status while silently storing nothing.

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
  ollamaUrl?: string,           // Ollama base URL (default: 'http://localhost:11434')
  reflectMission?: string,      // guides what to synthesize during reflection
  retainMission?: string,       // guides what to prioritize during retention
  embedModel?: string,          // default: 'Xenova/nomic-embed-text-v1.5' (local) or 'nomic-embed-text' (Ollama)
  embedDimensions?: number,     // only relevant when useOllamaEmbeddings: true
  reflectModel?: string,        // default: 'llama3.1:8b'
  useOllamaEmbeddings?: boolean, // use Ollama for embeddings instead of local (default: false)
  disposition?: {               // behavioral tuning for reflection
    skepticism?: number,        // 0–1
    literalism?: number,        // 0–1
    empathy?: number,           // 0–1
  },
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
  memoryType?: string,    // 'world' | 'experience' | 'observation' | 'opinion' (default: 'world')
  source?: string,        // e.g. 'conversation:session-123'
  sourceType?: string,    // 'user_stated' | 'inferred' | 'external_doc' | 'tool_result' | 'agent_generated'
  trustScore?: number,    // 0.0–1.0 (default: 0.7)
  context?: string,       // freeform context tag (e.g. 'infrastructure', 'project:valor')
  dedupMode?: string,     // 'exact' | 'normalized' | 'none' (default: 'normalized')
});
```

**Memory types:**
- `world` — semantic memory: facts about the world
- `experience` — episodic memory: what the agent itself did or observed
- `observation` — consolidated knowledge: patterns across facts (typically written by `reflect`)
- `opinion` — belief with confidence: strengthens or weakens with evidence

**Deduplication:** By default (`normalized`), text that is identical after case-folding and whitespace normalization is not stored twice — the existing chunk's trust score is reinforced instead. Set `dedupMode: 'none'` to disable.

### `retainBatch(items, onProgress?)`

Bulk store. More efficient than calling `retain()` in a loop — all entity extractions are queued at the end rather than one-by-one.

```typescript
const results = await mira.retainBatch([
  { text: 'Tom prefers Terraform', options: { memoryType: 'world', trustScore: 0.9 } },
  { text: 'Vault stores PKI certs', options: { memoryType: 'world' } },
], (current, total) => console.log(`${current}/${total}`));
```

### `recall(query, options?)`

Retrieve memories using four parallel strategies: semantic (vector similarity via sqlite-vec), keyword (FTS5 BM25), graph (entity walk), and temporal (date filter). Results are fused using Reciprocal Rank Fusion and weighted by trust score.

```typescript
const response = await mira.recall('What IaC tools does Tom use?', {
  topK?: number,          // max results (default: 10)
  memoryTypes?: string[], // filter by memory type: ['world', 'experience', ...]
  minTrust?: number,      // minimum trust score (default: 0.0)
  after?: string,         // ISO date string — only facts after this date
  before?: string,        // ISO date string — only facts before this date
  strategies?: string[],  // which strategies to run (default: all four)
  includeOpinions?: boolean,      // default: true
  includeObservations?: boolean,  // default: true
  sourceFilter?: string,          // hard filter: only chunks whose source contains this
  contextFilter?: string,         // hard filter: only chunks whose context contains this
  sourceBoost?: { pattern: string; multiplier: number },   // soft preference by source
  contextBoost?: { pattern: string; multiplier: number },  // soft preference by context
  decayHalfLifeDays?: number,     // trust decay half-life (default: 180, set 0 to disable)
  snippetChars?: number,          // max chars per result snippet
  rrfK?: number,                  // RRF constant (default: 60)
});

// response shape:
// {
//   results: Array<{
//     id: string,
//     text: string,
//     memoryType: string,
//     source: string | null,
//     sourceType: string,
//     trustScore: number,
//     eventTime: string | null,
//     score: number,           // final fused + weighted score
//     strategies: string[],    // which strategies found this chunk
//   }>,
//   opinions: Array<{ belief: string; confidence: number; domain: string | null }>,
//   observations: Array<{ summary: string; domain: string | null; topic: string | null }>,
//   totalCandidates: number,
//   strategiesUsed: string[],
// }
```

### `processExtractions(batchSize?)`

Run the background entity extraction queue. Calls Ollama against queued chunks to extract entities and relationships, building out the knowledge graph. Run this periodically or after bulk ingestion.

```typescript
const { processed, failed } = await mira.processExtractions(10); // batchSize default: 10
```

### `forget(chunkId)`

Soft-delete a memory chunk by ID. Sets `is_active = FALSE` so the chunk is excluded from all recall queries while remaining in the database for audit purposes.

```typescript
const wasFound = await mira.forget('chunk-uuid-here'); // returns false if ID not found
```

### `supersede(oldChunkId, newText, options?)`

Replace an outdated fact with corrected information. The old chunk is soft-deleted and linked to the new one via `superseded_by`. Use when a fact has changed:

```typescript
const newResult = await mira.supersede(
  oldChunkId,
  'Tom switched from Terraform to Pulumi for all new projects',
  { memoryType: 'world', trustScore: 0.9 }
);
```

### `forgetBySource(sourcePattern)`

Soft-delete all chunks whose `source` field contains the given substring. Returns the count of deactivated chunks. Useful for clearing an entire conversation or document import:

```typescript
const count = await mira.forgetBySource('conversation:session-123');
// count = number of chunks deactivated
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

## Utility Functions

These are exported directly from the `engram` package — no `Engram` instance required.

### `shouldRetain(text)`

Heuristic gate that scores text on how worth storing it is (0.0–1.0). Useful for filtering out phatic expressions, trivial acknowledgements, and bare questions before calling `retain()`.

```typescript
import { shouldRetain } from 'engram';

const { score, reason } = shouldRetain('Tom decided to use Pulumi for all new IaC');
// score: ~0.8   reason: 'decision language, technical terms'

const { score } = shouldRetain('ok cool');
// score: ~0.1   — phatic expression, skip storing

if (score >= 0.5) {
  await mira.retain(text, { ... });
}
```

Signals that raise the score: decision language, technical terms (camelCase/paths), temporal markers, proper nouns, substantive length. Signals that lower it: phatic expressions, pure interrogatives, very short text.

### `formatForPrompt(response, options?)`

Format a `RecallResponse` into a string suitable for injecting into a system prompt. Handles token budgeting — stops adding content once `maxChars` would be exceeded.

```typescript
import { formatForPrompt } from 'engram';

const memory = await mira.recall(userMessage, { topK: 10 });
const memoryBlock = formatForPrompt(memory, {
  maxChars?: number,       // default: 2000
  showTrust?: boolean,     // include trust percentages inline (default: false)
  showSource?: boolean,    // include source attribution (default: true)
  header?: string,         // section heading (default: '## Relevant Memory Context')
});

const systemPrompt = `${basePrompt}\n\n${memoryBlock}`;
```

Priority order in output: opinions (highest signal) → observations → memory results.

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
| `engram_forget` | Soft-delete a memory chunk by ID |
| `engram_supersede` | Replace an outdated fact with corrected text |

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
| @xenova/transformers | In-process embeddings (retain + recall) | Yes (default embedding path) |
| Ollama | Local LLM for extraction + reflection | Yes (for extract + reflect) |
| llama3.1:8b | Fast model for extraction + reflection | Recommended |
| nomic-embed-text (Ollama) | Ollama embedding model | Only when `useOllamaEmbeddings: true` |

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

## Integration with Custom Frameworks

If you're building your own agent framework (not using Operative or valor-engine), there are three integration patterns to be aware of.

### Pattern 1: Direct Integration (Simplest)

Call Engram methods directly from your agent loop. This works but couples your framework to the Engram API.

```typescript
import { Engram, shouldRetain, formatForPrompt } from 'engram';

const memory = await Engram.open('./agent.engram', { ollamaUrl: '...' });

// In your conversation loop:
async function agentLoop(userInput: string) {
  // 1. Recall before LLM call
  const context = await memory.recall(userInput, { topK: 10 });
  const memoryBlock = formatForPrompt(context, { maxChars: 2000 });

  // 2. Call LLM with memory context injected into system prompt
  const response = await callLLM(userInput, memoryBlock);

  // 3. Retain after LLM call (with gate)
  const userScore = shouldRetain(userInput);
  if (userScore.score >= 0.3) {
    await memory.retain(userInput, {
      memoryType: 'experience',
      sourceType: 'user_stated',
      trustScore: 0.85,
    });
  }

  const assistantScore = shouldRetain(response);
  if (assistantScore.score >= 0.3) {
    await memory.retain(response, {
      memoryType: 'experience',
      sourceType: 'agent_generated',
      trustScore: 0.6,
    });
  }

  return response;
}

// 4. Background ticks
setInterval(() => memory.processExtractions(10), 5 * 60 * 1000);  // every 5 min
setInterval(() => memory.reflect(), 6 * 60 * 60 * 1000);           // every 6 hours
```

### Pattern 2: Adapter Layer (Recommended)

Wrap Engram behind a thin adapter so your framework never imports `engram` directly. This makes Engram an optional dependency.

**Critical: pass `shouldRetain` and `formatForPrompt` alongside the instance.**

These are standalone functions exported from the `engram` package. If your framework tries to dynamically `import('engram')` at runtime to get these functions, Node's module resolution may fail depending on how your framework is installed (e.g., as a `file:` dependency with its own `node_modules`). The safe pattern is to import at the agent level and pass the references in:

```typescript
// In your AGENT code (not framework code):
import { Engram, shouldRetain, formatForPrompt } from 'engram';

const instance = await Engram.open('./agent.engram', { ... });

// Pass all three to your framework
framework.init({
  engram: {
    instance,
    shouldRetain,      // pass the function reference
    formatForPrompt,   // pass the function reference
  },
});
```

```typescript
// In your FRAMEWORK code:
async function recallContext(query: string, engram: EngramLike, formatFn: Function) {
  const response = await engram.recall(query, { topK: 10 });
  return formatFn(response, { maxChars: 2000 });
}

async function retainTurn(input: string, response: string, engram: EngramLike, shouldRetainFn: Function) {
  if (shouldRetainFn(input).score >= threshold) {
    await engram.retain(input, { ... });
  }
  if (shouldRetainFn(response).score >= threshold) {
    await engram.retain(response, { ... });
  }
}
```

**Why not dynamic import?** If your framework package has its own `node_modules` directory (common with `file:` dependencies or npm workspaces), `import('engram')` resolves from the framework's module context — not the agent's. The `engram` package may not be in the framework's dependency tree, causing a silent failure. Passing the functions directly at the agent level avoids this entirely.

### Pattern 3: MCP Tools

If your agent supports the Model Context Protocol, use the MCP tools export. See the [MCP Integration](#mcp-integration) section above.

### Background Processing

Engram's extraction and reflection are designed to run in the background, not inline with conversation turns:

- **Extraction** (`processExtractions`): Run every 3–5 minutes. Processes queued chunks through Ollama to extract entities and relationships. Each batch processes up to N chunks (default 10).
- **Reflection** (`reflect`): Run every 2–6 hours. Synthesizes unreflected chunks into observations and opinions. This is the expensive operation — it reads all unreflected facts and calls Ollama to produce higher-order understanding.

Both are safe to run concurrently with `retain()` and `recall()` — SQLite WAL mode handles concurrent readers and one writer.

If your framework has no scheduler, use `setInterval`:

```typescript
const extractionTimer = setInterval(() => memory.processExtractions(10), 5 * 60 * 1000);
const reflectionTimer = setInterval(() => memory.reflect(), 6 * 60 * 60 * 1000);

// On shutdown:
clearInterval(extractionTimer);
clearInterval(reflectionTimer);
memory.close();
```

### Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Engram active" but zero chunks | Embedding model failed to load | Default path: verify `@xenova/transformers` is installed and `~/.cache/xenova/` is writable. Ollama path: run `ollama pull nomic-embed-text` |
| Health check passes but retain still empty | `shouldRetain`/`formatForPrompt` imported via dynamic `import('engram')` fails silently in framework context | Pass the function references from the agent, not via dynamic import in the framework |
| Reflection runs but produces nothing | Zero chunks in the database | Fix retain first; reflection works on accumulated chunks |
| Extraction queue stays at 0 | Chunks retained with `skipExtraction: true` or non-world/experience types | Only `world` and `experience` types are queued for extraction by default |
| Entities/relations empty despite chunks | Extraction tick not running, or reflect model not available | Verify `processExtractions()` is being called periodically and `llama3.1:8b` is pulled |
| Recall returns empty despite chunks | `sqlite-vec` extension not loaded | Check that the `sqlite-vec` npm package is installed; without it, semantic recall is disabled |
| Trust scores not affecting results | All chunks have the same trust score | Differentiate: user statements (0.85), agent-generated (0.6), inferred (0.5), tool results (0.4) |

## Development

```bash
npm install
npm run build        # compile TypeScript → dist/
npm test             # run test suite (87 tests)
npm run typecheck    # TypeScript check without emit
npm run example      # run examples/basic-usage.ts (requires Ollama)
```

## License

Apache 2.0
