<div align="center">

# 🧠 Engram

### Memory traces for AI agents that strengthen with reinforcement.

*A lightweight, zero-infrastructure memory system where agents don't just remember — they form opinions, refine them with evidence, and get smarter the longer they run.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Node.js >= 20](https://img.shields.io/badge/Node.js-%3E%3D%2020-green.svg)](https://nodejs.org/)
[![CI](https://github.com/swift-innovate/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/swift-innovate/engram/actions/workflows/ci.yml)
[![SQLite](https://img.shields.io/badge/Storage-SQLite-003B57.svg)](https://sqlite.org)

</div>

---

Each agent gets its own SQLite file — an **engram** — containing everything it knows, has experienced, believes, and has learned. Raw facts are retained with no LLM call — a fast in-process embed plus a ~5ms SQLite write. Entities and relationships are extracted in the background. Periodic reflection cycles synthesize higher-order observations and confidence-scored beliefs.

One file. Zero infrastructure. Full cognitive architecture.

```
npm install engram
```

## Why Engram?

Most agent memory systems are either too simple (append-only logs) or too heavy (Postgres + Redis + vector DB). Engram sits in the sweet spot:

- **Fast writes** — no LLM in the hot path; the SQLite write is ~5ms, and total latency is dominated by the local embedding (tens of ms on CPU, faster with Ollama GPU embeddings)
- **Four-way retrieval** — semantic vectors, BM25 keyword, knowledge graph traversal, and temporal filtering fused via Reciprocal Rank Fusion
- **Trust layer** — every memory carries provenance and a confidence score that weights recall
- **Learns over time** — scheduled reflection synthesizes observations and forms beliefs that strengthen or weaken with evidence
- **Single file per agent** — portable, git-committable, inspectable with any SQLite tool
- **Working memory** — auto-switching session contexts for multi-topic conversations
- **Zero cloud dependency** — embeddings run locally via Transformers.js; extraction and reflection use any local LLM via Ollama or pluggable providers

### Where Engram fits

Engram's core value — trust-tiered provenance, fast-write/slow-extract, session continuity, cross-agent decision handoff (see [ContextStore](#task-scoped-context-contextstore)) — is agent-shape-agnostic. It's equally useful to a workflow/orchestration agent handing off a decision to a subagent, a general assistant remembering user preferences across sessions, or a coding agent remembering *why* a project made an architectural choice.

Where it's **not** a complete answer is code-symbol retrieval. RRF fusion over semantic embeddings + FTS5 keyword search + graph traversal is built for natural-language recall ("what did we decide about auth last week"), not for finding `handleAuthMiddleware` by name — `nomic-embed-text` embeddings and FTS5 tokenization aren't symbol-aware (no camelCase/snake_case splitting), and there's no invalidation signal when the underlying code changes out from under a stored memory. For a coding agent, pair Engram with proper code search (grep, ctags, an AST-aware index) for "find this symbol" — use Engram for the durable project memory layer (decisions, conventions, session state) that code search doesn't cover, not as a replacement for it. Workflow and assistant agents, whose primary retrieval need *is* natural-language recall, get the more complete fit out of the box.

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
│  │ → Tier 1 extract │              │ Graph    (entity walk)   │    │
│  │ → queue Tier 2   │              │ Temporal (date filter)   │    │
│  └────────┬────────┘              └────────────┬─────────────┘    │
│           │                                    ▼                  │
│           ▼                       RRF + Trust + Temporal Decay    │
│  EXTRACT (two-tier)               ┌──────────────────────────┐    │
│  ┌─────────────────┐              │ Ranked results +         │    │
│  │ Tier 1: CPU      │              │ opinions + observations  │    │
│  │  (inline, instant)│              └──────────────────────────┘    │
│  │ Tier 2: LLM      │                                             │
│  │  (background)     │                                             │
│  └─────────────────┘                                              │
│                                                                   │
│  REFLECT (scheduled learning)                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ unreflected facts → LLM → observations + opinions        │     │
│  │ (batch, not real-time — avoids latency on every write)   │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  WORKING MEMORY (short-term sessions)                             │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ auto-switching topic sessions → snapshot to episodic      │     │
│  │ embedding similarity matching → session cap enforcement   │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │               <agentId>.engram (SQLite)                   │     │
│  │  chunks | entities | relations | chunk_entities           │     │
│  │  opinions | observations | reflect_log | extraction_queue │     │
│  │  bank_config | chunks_fts (FTS5) | working_memory         │     │
│  └──────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

## The Biological Metaphor

Engram's pipeline mirrors how biological memory actually works:

1. **Encoding** (`retain`) — Sensory input is rapidly stored as a raw trace. Fast, no deliberation. Tier 1 CPU extraction links entities inline; Tier 2 LLM extraction queues for background processing — the way the hippocampus processes memories during idle time.

2. **Consolidation** (`extract`) — Background processing strengthens the trace by linking it to existing knowledge structures. Entities are resolved, relationships mapped, the knowledge graph densifies. Two tiers: CPU pattern matching runs inline (~2ms); LLM extraction runs asynchronously via Ollama.

3. **Reconsolidation** (`reflect`) — Periodic review of accumulated traces produces higher-order understanding. Observations emerge from patterns. Beliefs form and update with confidence. Old observations get refined as new evidence arrives.

4. **Retrieval** (`recall`) — Multi-pathway access to stored traces. Semantic similarity (pattern matching), keyword (direct access), graph traversal (associative recall), temporal (episodic memory). Results are fused via RRF, weighted by trust score with temporal decay, and ranked by source tier so external content never outranks user-stated directives.

## Prerequisites

**Engram has zero mandatory external dependencies.** Out of the box, `npm install engram` gives you:

- **Embeddings** via Transformers.js — runs in-process, no server needed. The model (`nomic-ai/nomic-embed-text-v1.5`, ~30MB) downloads automatically on first use.
- **Tier 1 entity extraction** via CPU pattern matching — runs inline with every `retain()` call, building the knowledge graph immediately.
- **Four-way recall** — semantic, keyword, graph, and temporal retrieval all work with zero external services.

That means `retain()`, `recall()`, and basic knowledge graph building work with nothing but Node.js and SQLite.

**Optional: LLM for deeper extraction and reflection.** Tier 2 entity extraction (`processExtractions()`) and reflection (`reflect()`) call a generation model to find complex relationships and synthesize observations/opinions. These are purely additive — if you never run them, the core pipeline still works. Ollama is supported out of the box, but Engram intentionally has no implicit generation-model default: configure a model before running extraction or reflection.

```bash
ollama pull llama3.1:8b
```

If you run models via [Herd](https://github.com/swift-innovate/herd) (Rust-based LLM gateway), use port `40114`:

```typescript
const agent = await Engram.create('./agent.engram', {
  ollamaUrl: 'http://localhost:40114',
});
```

Engram also supports **pluggable generation providers** — OpenAI-compatible APIs, Anthropic, or any custom backend. See [Generation Providers](#generation-providers).

## Quick Start

```typescript
import { Engram } from 'engram';

// Create an engram file for your agent
const agent = await Engram.create('./myAgent.engram', {
  ollamaUrl: 'http://localhost:11434',
  reflectModel: 'llama3.1:8b',
  reflectMission: 'Focus on architecture preferences and infrastructure decisions.',
  retainMission: 'Prioritize technical decisions and project context. Ignore greetings.',
});

// Store a fact (fast path, no LLM call)
await agent.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',
  sourceType: 'user_stated',
  trustScore: 0.9,
});

// Recall with four-way retrieval + trust weighting
const response = await agent.recall('What IaC tools does Tom use?', { topK: 5 });
console.log(response.results);      // ranked chunks
console.log(response.opinions);     // beliefs with confidence
console.log(response.observations); // synthesized knowledge

// Build knowledge graph (runs Ollama in background)
await agent.processExtractions(10);

// Synthesize observations and form opinions
await agent.reflect();

agent.close();
```

## Core API

### Creating and Opening

```typescript
// Create a new engram file
const agent = await Engram.create('./agent.engram', {
  ollamaUrl?: string,              // default: 'http://localhost:11434'
  reflectMission?: string,         // guides reflection synthesis
  retainMission?: string,          // guides retention prioritization
  embedModel?: string,             // default: 'nomic-ai/nomic-embed-text-v1.5'
  reflectModel?: string,           // no default; required for Ollama extraction/reflection
  useOllamaEmbeddings?: boolean,   // use Ollama instead of local (default: false)
  disposition?: {                  // behavioral tuning for reflection
    skepticism?: number,           // 0–1
    literalism?: number,           // 0–1
    empathy?: number,              // 0–1
  },
});

// Open an existing engram file
const agent = await Engram.open('./agent.engram');
```

### `retain(text, options?)`

Store a memory. Embeds in-process and writes to SQLite (the write is ~5ms; the embedding dominates total latency). Tier 1 CPU extraction links entities inline. Tier 2 LLM extraction is queued for background processing. No LLM call blocks this path.

```typescript
await agent.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType?: 'world' | 'experience' | 'observation' | 'opinion',  // default: 'world'
  source?: string,         // e.g. 'conversation:session-123'
  sourceType?: string,     // 'user_stated' | 'inferred' | 'external_doc' | 'tool_result' | 'agent_generated'
  trustScore?: number,     // 0.0–1.0 (default: 0.5)
  context?: string,        // freeform tag (e.g. 'infrastructure', 'project:valor')
  dedupMode?: string,      // 'exact' | 'normalized' | 'none' (default: 'normalized')
  supersedes?: string,     // explicit stale chunk id to deactivate and link
});
```

**Memory types** map to biological memory:

| Type | Biological analog | Written by |
|------|------------------|------------|
| `world` | Semantic memory — facts about the world | Agent or user |
| `experience` | Episodic memory — what the agent did | Agent |
| `observation` | Consolidated knowledge — patterns across facts | `reflect()` |
| `opinion` | Belief with confidence — strengthens/weakens with evidence | `reflect()` |

**Deduplication:** By default (`normalized`), identical text after case-folding and whitespace normalization is not stored twice — the existing chunk's trust score is reinforced instead.

### `retainBatch(items, onProgress?)`

Bulk store. More efficient than calling `retain()` in a loop.

```typescript
const results = await agent.retainBatch([
  { text: 'Tom prefers Terraform', options: { memoryType: 'world', trustScore: 0.9 } },
  { text: 'Vault stores PKI certs', options: { memoryType: 'world' } },
], (current, total) => console.log(`${current}/${total}`));
```

### `recall(query, options?)`

Retrieve memories using four parallel strategies fused via Reciprocal Rank Fusion, weighted by trust score with temporal decay, then ranked lexicographically by source tier (user-stated content structurally outranks external content — see Design Decisions).

Temporal expressions are auto-parsed from the query — `"last week"`, `"yesterday"`, `"March 15th"`, `"past 30 days"`, `"Q1 2026"` all activate the temporal strategy without explicit parameters.

**Query best practices:** Use keywords and proper nouns, not full questions. BM25 keyword search weights every word equally — question words like "who/what/how" dilute the signal and match irrelevant content. For people: `"Tom Swift role background"` not `"Who is Tom?"`. For topics: `"Herd router deployment"` not `"What is the Herd router?"`. Temporal expressions work naturally: `"projects in 2023"`.

```typescript
const response = await agent.recall('What IaC tools does Tom use?', {
  topK?: number,                    // max results (default: 10)
  memoryTypes?: string[],           // filter by type
  minTrust?: number,                // minimum trust score
  after?: string,                   // ISO date — overrides auto-parsed dates
  before?: string,                  // ISO date — overrides auto-parsed dates
  strategies?: string[],            // subset of ['semantic','keyword','graph','temporal']
  includeOpinions?: boolean,        // default: true
  includeObservations?: boolean,    // default: true
  sourceFilter?: string,            // hard filter by source substring
  contextFilter?: string,           // hard filter by context substring
  sourceBoost?: { pattern: string; multiplier: number },
  contextBoost?: { pattern: string; multiplier: number },
  decayHalfLifeDays?: number,       // trust decay half-life (default: 180, 0 to disable)
  sourceTiers?: Record<string, number>, // source_type → tier overrides (see DEFAULT_SOURCE_TIERS)
  memoryTypeRank?: Record<string, number>, // memory_type → within-tier rank overrides (see DEFAULT_MEMORY_TYPE_RANK)
  snippetChars?: number,            // max chars per result
  rrfK?: number,                    // RRF constant (default: 60)
  minScore?: number,                // drop fused results below this weighted score
  explainScores?: boolean,          // attach a per-result strategyScores breakdown (default: false)
});

// response.results       → ranked chunks with scores, strategies, and metadata
// response.opinions      → relevant beliefs with confidence scores
// response.observations  → synthesized knowledge
// response.totalCandidates
// response.strategiesUsed
```

Synthesized `opinions` and `observations` are query-scoped: Engram
case-folds and strips punctuation from the query, then includes only entries
with a matching normalized term. If no term matches, those arrays are empty;
there is no global fallback context.

**Result ordering.** `recall` returns results in **tier-major order, not pure relevance order**: sorted by source tier first (0 `user_stated`, 1 `inferred`/`agent_generated`, 2 `tool_result`/`external_doc`), then by trust-weighted relevance within each tier. This enforces the trust-layer guarantee — external content cannot outrank user directives regardless of relevance or trust score. Memory type (`world`/`observation`/`experience`/`opinion`) is **not** part of this ordering floor — it's a soft multiplicative weight folded into the score itself (world/observation get a gentle boost, opinion a gentle penalty), so a strong match in one type can still beat a weak match in another instead of losing to it categorically; only source tier is an absolute floor. Integrators: `results[0]` is the best match in the highest-present tier, **not** necessarily the highest-relevance match overall; do not assume score-descending order across the full list (re-sort by `score` locally where you genuinely need relevance order). Tier mapping is configurable via `RecallOptions.sourceTiers`; the memory-type weight curve via `RecallOptions.memoryTypeRank` (defaults exported as `DEFAULT_SOURCE_TIERS` / `DEFAULT_MEMORY_TYPE_RANK`). `minScore` filters the fused, weighted set — anything below the threshold is dropped before results are returned. `explainScores: true` attaches a `strategyScores` breakdown (per-strategy rank/score that fed RRF) to each result, off by default to keep the payload lean.

**Recency decay tradeoff.** `decayHalfLifeDays` defaults to 180 — a chunk's score is multiplied by `2^(-ageDays/180)`, so content much older than ~18 months becomes functionally unrecallable regardless of relevance, with no warning. That's a sensible default for a short-lived coding-session context, but a real trap for a personal-assistant/journal-style consumer meant to have continuity across years — pass `decayHalfLifeDays: 0` (or a much longer half-life) explicitly if that's your use case; don't rely on the default.

### `processExtractions(batchSize?)`

Run the Tier 2 LLM entity extraction queue. Builds out the knowledge graph.

```typescript
const { processed, failed } = await agent.processExtractions(10);
```

### `reflect()`

Synthesize observations and form/update opinions from accumulated facts.

```typescript
const result = await agent.reflect();
// { observationsCreated, opinionsFormed, opinionsReinforced, chunksProcessed }
```

New opinions are safe by default: they need three verified chunks spanning two
days, and reflection retrieves and judges counter-evidence before forming them.
That audit costs one extra batched generation call per cycle. Object settings
merge over the defaults; `false` is the explicit opt-out. A standalone
`reflect()` call needs an embedder while the audit is enabled.

```typescript
await agent.reflect({
  opinionGates: { minEvidenceCount: 4 },
  counterEvidence: { topK: 12 },
});

// Legacy/permissive behavior is an explicit choice:
await agent.reflect({ opinionGates: false, counterEvidence: false });
```

### `forget(chunkId)` / `supersede(oldChunkId, newText)` / `forgetBySource(pattern)`

Manage memory lifecycle:

```typescript
await agent.forget('chunk-uuid');                          // soft-delete
await agent.supersede(oldId, 'corrected fact', options);   // replace with link
const count = await agent.forgetBySource('session-123');   // bulk soft-delete
```

Conflicting facts are intentionally retained as separate active records until
the caller explicitly supersedes the stale chunk. Recall's human CLI output
includes chunk IDs so a returned fact can be corrected immediately:

```bash
engram retain "Tom switched to Kubernetes" --supersedes chk-abc123
```

### `close()`

Close the database connection when the agent session ends.

## Working Memory

Short-term session state for agents handling multiple concurrent topics. Sessions auto-match incoming messages via embedding similarity — the agent doesn't need to track which session it's in.

```typescript
async function handleMessage(userInput: string) {
  // 1. Infer or resume session + load related long-term context
  const { session, relatedContext } = await agent.inferWorkingSession(userInput);

  // 2. Build prompt with session state + memory
  const systemPrompt = `${basePrompt}

## Current Task
Goal: ${session.goal}
${session.progress ? `Progress: ${session.progress}` : ''}

## Memory Context
${relatedContext}`;

  // 3. Call LLM
  const response = await callLLM(userInput, systemPrompt);

  // 4. Update session progress
  await agent.updateWorkingSession(session.id, {
    progress: extractProgress(response),
  });

  return response;
}

// Background maintenance
setInterval(() => agent.expireStaleWorkingSessions(48), 60 * 60 * 1000);
```

**Session API:**

| Method | Description |
|--------|-------------|
| `inferWorkingSession(message, options?)` | Match/create session, load related context |
| `updateWorkingSession(sessionId, updates)` | Merge state updates, re-embed for matching |
| `getWorkingSession(sessionId)` | Get session state or `null` if expired |
| `listWorkingSessions()` | All active sessions, newest first |
| `snapshotWorkingSession(sessionId)` | Save to episodic memory, then expire |
| `clearWorkingSession(sessionId)` | Expire without snapshot |
| `expireStaleWorkingSessions(maxAgeHours?)` | Batch-expire old sessions (default: 48h) |

## Task-Scoped Context (ContextStore)

A second, easily-confused-with-Working-Memory concept: cheap agent-to-subagent
handoff of small structured decisions, not conversation session state.

|                          | Working Memory (above)                          | ContextStore (this section)                        |
|--------------------------|--------------------------------------------------|------------------------------------------------------|
| Shape                    | One mutable JSON blob per session                | Many immutable `DecisionArtifact`s per task          |
| Matched by                | Its own cosine similarity over `topic_embedding` | The same RRF-fusion `recall()` pipeline as long-term memory |
| Storage                  | `working_memory` table                           | `chunks` table, `scope = 'task'`                     |
| Typical use              | "Which conversation topic is this?"              | "Lead committed a decision; subagent needs the relevant slice of it." |

A lead agent commits a small structured artifact instead of handing a
subagent the whole transcript; the subagent queries for only what's
relevant, ranked through the exact same trust-tiered RRF recall as
everything else. Artifacts are short-lived by default (TTL, not days),
excluded from `reflect()`/consolidation, and never appear in a default
`recall()` call — `scope` defaults to `'durable'`, so existing recall
behavior is unchanged unless a caller opts in.

```typescript
import { commitContext, queryContext } from 'engram';

// Lead agent commits two sibling decisions under a shared parent scope
const parent = await commitContext(db, embedder, { decision: 'decompose the migration task' });
const dbDecision = await commitContext(db, embedder, {
  decision: 'use Postgres for the new service',
  rationale: 'team already runs Postgres in production',
  agentId: 'lead-tier1',
}, { parent });
await commitContext(db, embedder, {
  decision: 'use React for the dashboard frontend',
  rationale: 'matches the existing component library',
}, { parent });

// Subagent, handed only `parent`, pulls just the relevant slice under a token budget
const slice = await queryContext(db, embedder, parent, 'database choice', { maxChars: 2000 });
// slice.artifacts -> [{ ref, artifact, parentRef, createdAt, expiresAt }, ...]
// slice.truncated -> true if relevant results didn't fit the budget
```

TTL is enforced lazily (checked at query time via `expires_at`, no background
reaper) rather than swept eagerly. `expireContext(ref)` hard-expires an
artifact early; `promoteToDurable(ref)` is a mechanical seam that moves an
artifact into durable scope (clears TTL/parent, becomes reflect-eligible) —
it does **not** run `reflect()` itself; wiring that up is future work.

**Task-scoped context over MCP/CLI.** The same primitives are reachable
without a direct TypeScript import via `engram_context_commit` /
`engram_context_query` / `engram_context_promote` (MCP) or `context-commit` /
`context-query` / `context-promote` (CLI) — see [Tools Exposed](#tools-exposed)
and the [CLI](#cli) section. One gotcha to know before using either: querying
a ref returns the **children** committed under it (`parentRefId` pointing at
that ref), not the artifact stored at the ref itself.

## Grounding a Subagent

"Grounding in, report out, nothing written by the subagent." When an
orchestrator spawns a **stateless** subagent, it situates it with scoped,
**belief-free** context and gets a plain report back — every durable write
stays on the orchestrator's side.

```typescript
import { groundSubagent, taskContext, metabolizeReport } from 'engram';

// Orchestrator holds a read/write Engram. Hand the subagent a READ-ONLY view:
// it exposes recall/queryContext/introspect only, over a readonly connection —
// a subagent physically cannot write to the store (no method, and the driver
// rejects raw SQL).
const ro = await orchestrator.readonlyView();

// Situate the subagent from durable world/experience/observation facts.
// Opinions are NEVER injected — a stateless subagent has no revision loop to
// tell a sound belief from a stale one, so beliefs stay behind the orchestrator.
const grounding = await groundSubagent(ro, { task: 'design the retry policy', maxChars: 2000 });
// grounding.prompt -> ready-to-inject, headered, budgeted block (no "Beliefs" section)
// grounding.facts / grounding.observations -> structured form for custom prompts

// To also show specific parent-task decisions, the orchestrator selects them
// EXPLICITLY (never auto-inherited) and concatenates them into the prompt:
const slice = await taskContext(ro, parentRef, 'retry/backoff decisions');

// The subagent returns a plain object; the ORCHESTRATOR metabolizes it (single
// writer). Candidate experiences land as agent_generated (tier 1) — challengeable
// by the next reflect cycle, never able to outrank a user-stated fact.
await metabolizeReport(orchestrator, subagentReport, {
  keepExperiences: (c) => c.context === 'perf-spike',
});
ro.close();
```

Grounding from long-lived `world` facts? Recall's recency decay
(`decayHalfLifeDays`, default 180) applies here too — pass `decayHalfLifeDays: 0`
via a custom `recall()` if year-old facts must stay recallable. This is a
library-only capability (no MCP tool); see `docs/GROUNDING-LAYER-SPEC.md`.

## Utility Functions

Exported directly from `engram` — no instance required.

### `shouldRetain(text)`

Heuristic gate scoring text on retention value (0.0–1.0). Filters phatic expressions, trivial acknowledgements, and bare questions.

```typescript
import { shouldRetain } from 'engram';

const { score, reason } = shouldRetain('Tom decided to use Pulumi for all new IaC');
// score: ~0.8, reason: 'decision language, technical terms'

if (score >= 0.5) await agent.retain(text, { ... });
```

### `formatForPrompt(response, options?)`

Format a `RecallResponse` into a string for system prompt injection with token budgeting.

```typescript
import { formatForPrompt } from 'engram';

const memory = await agent.recall(userMessage, { topK: 10 });
const block = formatForPrompt(memory, { maxChars: 2000, showTrust: false });
const systemPrompt = `${basePrompt}\n\n${block}`;
```

## Generation Providers

Entity extraction and reflection use a pluggable `GenerationProvider` interface. Three implementations ship out of the box:

```typescript
// Ollama (default — no config needed)
await Engram.create('./agent.engram', { ollamaUrl: 'http://localhost:11434' });

// OpenAI-compatible (OpenRouter, Herd Pro, vLLM, LiteLLM, etc.)
await Engram.create('./agent.engram', {
  generationEndpoint: {
    baseUrl: 'https://openrouter.ai/api',
    model: 'anthropic/claude-haiku-4.5',
    apiKey: process.env.OPENROUTER_API_KEY,
  },
});

// Anthropic direct
await Engram.create('./agent.engram', {
  anthropicGeneration: {
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: 'claude-haiku-4-5-20251001',
  },
});

// Custom — inject any GenerationProvider implementation
await Engram.create('./agent.engram', { generator: myCustomProvider });
```

## MCP Integration

Expose Engram operations as MCP tools for agent frameworks that support the Model Context Protocol.

### Programmatic

```typescript
import { ENGRAM_TOOLS, createEngramToolHandler } from 'engram/mcp-tools';

const handle = createEngramToolHandler(agent);
server.registerTools(ENGRAM_TOOLS);
server.onToolCall((name, input) => handle(name, input));
```

### Stdio Server

Ships a standalone MCP server for Claude Code, Claude Desktop, Cursor, and any MCP-compatible client:

```bash
npx engram-mcp ./agent.engram
npx engram-mcp ./agent.engram --use-ollama-embeddings --ollama-url http://localhost:11434
npx engram-mcp ./agent.engram --generation-endpoint http://localhost:8080 --generation-model gpt-4
npx engram-mcp ./agent.engram --anthropic-api-key sk-ant-... --anthropic-model claude-haiku-4-5-20251001
```

**Claude Code / Claude Desktop config:**

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["/path/to/engram/dist/mcp-server.js", "./agent.engram"],
      "transport": "stdio"
    }
  }
}
```

### Tools Exposed

| Tool | Description |
|------|-------------|
| `engram_retain` | Store a memory with source and trust metadata |
| `engram_recall` | Four-way retrieval with auto-parsed temporal expressions |
| `engram_reflect` | Trigger reflection cycle |
| `engram_process_extractions` | Process entity extraction queue |
| `engram_forget` | Soft-delete a memory chunk |
| `engram_supersede` | Replace an outdated fact with corrected text |
| `engram_session` | Infer/resume, update, or snapshot a working memory session (`action: resume\|update\|snapshot`, default `resume`) |
| `engram_queue_stats` | Extraction queue depth, processing metrics, and failure-reason breakdown |
| `engram_requeue_failed` | Re-queue failed extractions after an outage (optional error filter) |
| `engram_introspect` | Read held state — opinions (belief + confidence + support/challenge evidence + lifecycle) and observations for a subject; direct lookup, no confidence floor |
| `engram_embed` | Embed text in the bank's stored vector space (query or document mode) |
| `engram_context_commit` | Commit a task-scoped decision artifact (ContextStore) |
| `engram_context_query` | Query the children committed under a ContextStore ref |
| `engram_context_promote` | Promote a task-scoped artifact into durable memory |
| `engram_suggestions` | List procedural suggestions (recurring corrections/friction/workflows worth codifying as a skill/rule/workflow/config) |
| `engram_resolve_suggestion` | Set a suggestion's lifecycle status (`accepted`/`dismissed`/`implemented`; `proposed` reopens) |

`engram_session` omitting `action` behaves exactly as before (backward compatible); `action: 'update'`/`'snapshot'` require `sessionId` and let an MCP-only agent drive the full session lifecycle without a direct API call. `engram_retain` accepts `supersedes` to atomically replace a known stale chunk. `engram_reflect` takes an optional `suggest: boolean` (default `false`) to also run the procedural-suggestion pass that cycle; suggestions never appear in `engram_recall`, only via `engram_suggestions`. Reflection is safe by default: `opinionGates` defaults to three verified chunks across two days and `counterEvidence` defaults to a fail-closed, extra batched judge call (`topK: 8`, contradiction ratio `0.5`). Pass either field as `false` only for an explicit opt-out; object values override individual safe defaults.

## CLI

A third transport over the same Engram core — one subcommand per MCP tool,
kebab-cased. Ideal as a coding-agent skill (e.g. Pi): the agent shells out, pipes
context in on stdin, and parses `--json` on stdout. Ships as the `engram` bin.

```bash
npm install -g engram      # or: npx engram <command>
```

Point it at a database with `--db <path>` on each call, or set `ENGRAM_DB` once
(`--db` takes precedence; if neither is set the command exits 1):

```bash
export ENGRAM_DB=./agent.engram
```

Every command accepts `--json`, which emits the raw method return as JSON to
**stdout and nothing else** (diagnostics go to stderr). Without it you get
human-readable output. The same generation/embedding flags as `engram-mcp` are
accepted: `--ollama-url`, `--use-ollama-embeddings`, `--reflect-model`,
`--generation-endpoint`/`--generation-model`/`--generation-api-key`,
`--anthropic-api-key`/`--anthropic-model`.

**Exit codes:** `0` success · `2` not-found (`forget`/`supersede` on a missing
chunk) · `1` error (bad/missing argument, no DB path, operation failure).

The sixteen commands:

```bash
# Store a fact
engram retain "Tom prefers Pulumi over Terraform" \
  --memory-type world --source-type user_stated --trust-score 0.9 --json

# Search (keywords/proper nouns beat questions; temporal phrases auto-parse)
engram recall "Terraform IaC provider" --top-k 5 --json
engram recall "decisions last week" --strategies semantic,temporal --json
engram recall "IaC tools" --min-score 0.3 --explain-scores --json

# Infer/resume a working-memory session (once per incoming message)
engram session "Help me plan the deployment" --json
# Update progress, or snapshot to long-term memory and end it (both need --session-id)
engram session --action update --session-id sess-abc123 --progress "step 2 done" --json
engram session --action snapshot --session-id sess-abc123 --json

# Correct an outdated fact (old chunk soft-deleted + linked to the new one)
engram supersede chk-abc123 "Tom switched to Kubernetes" --json

# Soft-delete a chunk
engram forget chk-abc123 --json

# Background maintenance (need an LLM)
engram reflect --json
engram reflect --suggest --json   # also propose procedural suggestions
# Explicit safety overrides when needed
engram reflect --opinion-min-evidence 4 --counter-evidence-top-k 12 --json
engram reflect --unsafe-opinions --no-counter-evidence --json  # permissive opt-out
engram process-extractions --batch-size 10 --json

# Queue health (includes a failed_reasons breakdown)
engram queue-stats --json

# Re-drive failed extractions after an outage (all, or one failure class)
engram requeue-failed --json
engram requeue-failed --error-like "fetch failed" --json

# Introspect held state about a subject (opinions + observations, no confidence floor)
engram introspect "kubernetes" --json
engram introspect --json                 # top held state overall

# Embed text in the bank's vector space (query mode by default)
engram embed "deploy pipeline" --json
engram embed "stored document text" --mode document --json

# Task-scoped context handoff (ContextStore) — see below
engram context-commit '{"decision":"use Postgres"}' --json
engram context-query <refId> "database choice" --json
engram context-promote <refId> --json

# Read/resolve procedural suggestions proposed by `reflect --suggest`
engram suggestions --status proposed --json
engram resolve-suggestion <suggestionId> accepted --json
```

The primary text argument for `retain`, `recall`, `session`, `embed`, and the
`newText` of `supersede` is read from **stdin** when the positional is omitted,
so an agent can pipe context straight in:

```bash
echo "long pasted context to remember" | engram retain --db ./agent.engram --json
cat error.log | engram recall --json
```

The `--json` shape of each command is its method's return value verbatim — see
`skills/cli-memory/SKILL.md` for the per-command schemas and an agent-facing
decision guide (when to recall vs. retain vs. supersede vs. session).

## engram-aql — Native Rust Query Binary (Optional)

For declarative AQL (Agent Query Language) queries over your `.engram` file,
this repo also contains a companion Rust binary at `engram-aql/`.

It runs alongside TypeScript Engram as a second process, sharing the same
`.engram` SQLite file in WAL mode. Phase 1 is read-only — writes still go
through `engram.retain()` on the TS side — but reads support the full AQL
spec including RECALL, SCAN, LOOKUP, LOAD, AGGREGATE, ORDER BY, WITH LINKS,
FOLLOW LINKS, and PIPELINE.

Three subcommands:

- `engram-aql query <path> "<aql>"` — one-shot CLI query, JSON output
- `engram-aql repl <path>` — interactive REPL with pretty tables
- `engram-aql mcp <path>` — stdio MCP server exposing an `engram_aql` tool

Build and install:

```bash
cd engram-aql
cargo install --path .
```

See `engram-aql/README.md` for details and
`docs/superpowers/specs/2026-04-12-engram-aql-rust-binary-design.md`
for the architecture.

## Two-Tier Entity Extraction

Engram builds a knowledge graph without blocking writes:

**Tier 1 (CPU, inline with `retain()`):** Pure pattern matching — graph lookup against existing entities, proper noun detection, technical term detection, and relation template matching (`"X uses Y"`, `"X prefers Y"`, etc.). Runs in ~2ms, zero LLM, zero model loads. Every `retain()` enriches the graph immediately.

**Tier 2 (LLM, background via `processExtractions()`):** Ollama-based extraction catches complex relationships and reclassifies provisional entities. Purely additive — if Ollama never runs, the graph still works from Tier 1.

The knowledge graph compounds automatically: each entity discovered by Tier 2 makes Tier 1 smarter — future mentions get linked instantly.

## Harness Integrations

Engram is a memory substrate; agent harnesses consume it via different integration models. See **[integrations/README.md](integrations/README.md)** for the index.

### OpenClaw

Replaces OpenClaw's built-in flat-file FTS with semantic four-strategy retrieval via the `memory-engram` plugin and mcporter (out-of-process). Production-verified with the Tracer agent. See **[docs/OPENCLAW-INTEGRATION.md](docs/OPENCLAW-INTEGRATION.md)**.

A migration CLI ships in `tools/openclaw-import/` for importing existing OpenClaw `memory/` directories into `.engram` files (deterministic classification, no LLM during import):

```bash
cd tools/openclaw-import && npm install
npx tsx src/index.ts -i /path/to/memory -o ./agent.engram --dry-run  # preview
npx tsx src/index.ts -i /path/to/memory -o ./agent.engram            # import
```

See **[tools/openclaw-import/README.md](tools/openclaw-import/README.md)** for the category mapping and CLI options.

### Pi.dev (`pi-mono`)

In-process Pi extension exposing slash commands (`/remember`, `/recall`, `/memory`, `/forget`) and LLM tools (`engram_remember`, `engram_recall`, `engram_memory_stats`, `engram_forget`). Loaded directly via Node.js + `jiti` — millisecond-latency operations against a project-local `.engram/pi.db`.

```bash
cd integrations/pi && npm install && npm run build
ln -s "$PWD" ~/.pi/agent/extensions/engram-pi   # or: pi -e ./dist/index.js
```

See **[docs/PI-INTEGRATION.md](docs/PI-INTEGRATION.md)** for full setup, lifecycle behavior, and the OpenClaw-vs-Pi adapter comparison.

## Agent Skills

Portable skill files for agents using Engram via mcporter:

- **[skills/engram.md](skills/engram.md)** — Complete tool reference with all 16 MCP tools, usage patterns, and common mistakes
- **[skills/engram-session.md](skills/engram-session.md)** — Working memory session lifecycle and tuning guide
- **[skills/cli-memory/SKILL.md](skills/cli-memory/SKILL.md)** — `engram` CLI contract for coding agents (e.g. Pi): per-command `--json` schemas, exit codes, and when to recall vs. retain vs. supersede vs. session

## Integration Patterns

### Direct Integration

```typescript
import { Engram, shouldRetain, formatForPrompt } from 'engram';

const memory = await Engram.open('./agent.engram', {
  reflectModel: 'llama3.1:8b', // required for the extraction/reflect timers below
});

async function agentLoop(userInput: string) {
  const context = await memory.recall(userInput, { topK: 10 });
  const block = formatForPrompt(context, { maxChars: 2000 });

  const response = await callLLM(userInput, block);

  if (shouldRetain(userInput).score >= 0.3) {
    await memory.retain(userInput, { memoryType: 'experience', sourceType: 'user_stated', trustScore: 0.85 });
  }
  if (shouldRetain(response).score >= 0.3) {
    await memory.retain(response, { memoryType: 'experience', sourceType: 'agent_generated', trustScore: 0.6 });
  }

  return response;
}

// Background ticks
setInterval(() => memory.processExtractions(10), 5 * 60 * 1000);
setInterval(() => memory.reflect(), 6 * 60 * 60 * 1000);
```

### Adapter Layer (Recommended)

Pass `shouldRetain` and `formatForPrompt` as function references from the agent level — don't rely on dynamic `import('engram')` in framework code (Node module resolution across `file:` dependencies can fail silently):

```typescript
// Agent code:
import { Engram, shouldRetain, formatForPrompt } from 'engram';
const instance = await Engram.open('./agent.engram');
framework.init({ engram: { instance, shouldRetain, formatForPrompt } });

// Framework code:
async function recallContext(query, engram, formatFn) {
  const response = await engram.recall(query, { topK: 10 });
  return formatFn(response, { maxChars: 2000 });
}
```

### Scheduled Reflection

```typescript
import { LocalEmbedder, ReflectScheduler } from 'engram';

const scheduler = new ReflectScheduler({
  dbPath: './agent.engram',
  ollamaUrl: 'http://localhost:11434',
  reflectModel: 'llama3.1:8b',
  // Required by the default counter-evidence audit; initialization is lazy.
  embedder: new LocalEmbedder(),
});
scheduler.start(6 * 60 * 60 * 1000); // every 6 hours
```

### CLI Reflection

```bash
REFLECT_MODEL=llama3.1:8b npx tsx src/reflect.ts ./agent.engram
OLLAMA_URL=http://my-server:11434 REFLECT_MODEL=llama3.2:3b npx tsx src/reflect.ts ./agent.engram
```

## Verifying Your Setup

A common failure mode is `Engram.open()` completing successfully while the embedding pipeline silently fails. The database opens fine — but embeddings load lazily on the first `retain()` call.

```typescript
// Health check — verify end-to-end before trusting the pipeline
const engram = await Engram.open('./agent.engram');
try {
  const result = await engram.retain('__health_check__', {
    memoryType: 'world', source: 'health_check', trustScore: 0.0, skipExtraction: true,
  });
  await engram.forget(result.chunkId);
  console.log('✓ Engram healthy');
} catch (err) {
  console.error('✗ Embedding model unreachable:', err.message);
}
```

## Design Decisions

**Fast write / slow extract.** `retain()` never blocks on an LLM call. This exists because of lessons learned from mem0 integration — Ollama connectivity issues and write latency were killing agent responsiveness.

**Trust layer.** Every chunk carries `source_type` and `trust_score`. Recall weights results by trust, then ranks lexicographically by source tier — tier 0 `user_stated`, tier 1 `inferred`/`agent_generated`, tier 2 `tool_result`/`external_doc` — so external documents and tool results structurally cannot outrank user-stated directives, no matter their trust score or relevance (tunable via `sourceTiers`). Extraction and reflection prompts delimit memory content as labeled untrusted data — a prompt-injection mitigation, not a guarantee.

**SQLite over Postgres.** One file per agent — portable, git-committable, backup-friendly. sqlite-vec handles vector search; FTS5 handles keyword search. No Docker containers, no infrastructure dependencies.

**Batch reflect over real-time.** Reflection runs on a schedule, not on every query. This keeps LLM usage predictable while still building understanding over time.

**Two-tier extraction.** Tier 1 (CPU, inline) ensures the knowledge graph is never empty. Tier 2 (LLM, background) refines it. If Ollama goes down, the graph still works.

**Normalized dedup by default.** Storing the same fact twice wastes space and confuses ranking. Dedup reinforces trust instead.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Engram active" but zero chunks | Embedding model failed to load | Verify `@huggingface/transformers` installed and `node_modules/@huggingface/transformers/.cache/` writable |
| Health check passes, retain still empty | `shouldRetain`/`formatForPrompt` dynamically imported in framework | Pass function references from agent level, not via `import('engram')` |
| Reflection runs but produces nothing | Zero chunks in database | Fix retain first |
| Extraction queue stays at 0 | `skipExtraction: true` or non-world/experience types | Only `world` and `experience` queue for extraction |
| Entities empty despite chunks | Extraction tick not running | Call `processExtractions()` periodically |
| Recall returns empty | `sqlite-vec` not loaded | Install the `sqlite-vec` npm package |
| Trust scores not differentiating | All chunks same trust | Vary: user_stated=0.85, agent_generated=0.6, inferred=0.5, tool_result=0.4 |

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| better-sqlite3 | SQLite driver | Yes |
| sqlite-vec | Vector similarity search | Yes |
| @huggingface/transformers | In-process embeddings | Yes (default) |
| @modelcontextprotocol/sdk | MCP server support | Yes |
| Ollama + llama3.1:8b | Tier 2 extraction + reflection | **No** — core pipeline works without it |
| nomic-embed-text (Ollama) | Ollama embeddings | Only with `useOllamaEmbeddings: true` |

### Embedding model and cache

Engram embeds in-process via `@huggingface/transformers` 4.2.0 (the maintained successor to the deprecated `@xenova/transformers`). The default model is `nomic-ai/nomic-embed-text-v1.5` from the **public upstream nomic-ai repo, so no Hugging Face token is required**. It downloads once (~30MB quantized) to `node_modules/@huggingface/transformers/.cache/` on first use and loads from cache thereafter.

> The legacy `Xenova/nomic-embed-text-v1.5` mirror is now **gated** on the HF hub (401 without a token), which is why it's no longer the default. It ships identical 768-dim weights, so existing `.engram` files stay valid either way. Override the model via the `embedModel` option; for an unregistered model also pass an explicit `dimensions` so Engram records the correct vector size.

## File Format

Agent memory files use the `.engram` extension. Standard SQLite databases — inspect with any SQLite tool.

```
~/.valor/memory/myAgent.engram
~/.valor/memory/gage.engram
~/.valor/memory/shared.engram
```

## Development

```bash
npm install
npm run build        # TypeScript → dist/
npm test             # run the TypeScript test suite
npm run typecheck    # type check without emit
npm run audit:prod   # fail on high/critical production dependency advisories
npm run example      # run examples/basic-usage.ts
```

## License

Apache 2.0
