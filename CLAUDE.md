# Engram — Memory Traces for AI Agents

> *A memory trace physically encoded in neural tissue that strengthens with reinforcement.*

## What This Is

Engram is a lightweight, zero-infrastructure memory system for AI agents that **learns** over time. Each agent gets its own SQLite file — an engram — containing everything it knows, has experienced, believes, and has learned. Raw facts are retained, entities and relationships are extracted, and periodic reflection cycles synthesize higher-order observations and confidence-scored beliefs.

Agents running on Engram don't just remember. They form opinions, refine them with evidence, and get smarter the longer they run.

## Repo

`swift-innovate/engram` — Apache 2.0

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
│  │  opinions | observations | reflect_log | belief_journal   │     │
│  │  suggestions | suggestion_journal                         │     │
│  │  extraction_queue | bank_config | chunks_fts (FTS5)       │     │
│  │  working_memory                                           │     │
│  └──────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

## The Engram Pipeline (Biological Metaphor)

The system mirrors how biological memory works:

1. **Encoding** (retain) — Sensory input is rapidly stored as a raw trace. Fast, no deliberation. Entity extraction happens in the background, the way the hippocampus processes memories during idle time.

2. **Consolidation** (extract) — Background process strengthens the trace by linking it to existing knowledge structures. Entities are resolved, relationships mapped, the knowledge graph densifies.

3. **Reconsolidation** (reflect) — Periodic review of accumulated traces produces higher-order understanding. Observations emerge from patterns. Beliefs form and update with confidence. Old observations get refined as new evidence arrives. This is the learning loop.

4. **Retrieval** (recall) — Multi-pathway access to stored traces. Semantic similarity (pattern matching), keyword (direct access), graph traversal (associative recall), temporal (episodic memory). Results are fused via RRF, weighted by trace strength (trust score), then sorted lexicographically by source tier (see Trust Layer) so external content can never outrank user-stated directives. When temporal bounds are detected (auto-parsed or explicit), ALL strategies filter by date — not just the temporal strategy. Queries are sanitized before keyword/graph to prevent FTS5 syntax errors — balanced double-quoted segments are preserved as FTS5 phrase queries for the keyword strategy (`sanitizeQueryForFts`); everything else is punctuation-stripped, so unquoted multi-word keyword queries match as an implicit AND of terms. Bare 4-digit years only auto-parse as temporal bounds with corroborating context ("in 2024", "March 2026", "Q1 2026") — a bare number in "port 2020" or "error code 2048" no longer date-filters recall. `RecallOptions.minScore` drops fused results below a weighted-score threshold post-weighting; `explainScores?: boolean` (default false) attaches a per-result `strategyScores` breakdown (the rank/score each strategy contributed to RRF) for callers that need to see why something ranked where it did, without bloating the default payload. `RecallOptions.decayHalfLifeDays` is exposed on every transport, not just the library API: the `engram_recall` MCP tool takes a `decayHalfLifeDays` number param (min 0) and the CLI takes `--decay-half-life-days <n>`; both pass straight through to `recall()` and omitting either leaves the library default (180) unchanged. The `engram_recall` tool description itself now states the trust-tier guarantee and the decay default/override so an agent can discover both without reading source. `formatForPrompt` gained `showProvenance` / `showWhy` flags (2026-07-16, both default false): provenance renders a `[memoryType/sourceType, trust X.XX, YYYY-MM-DD]` bracket per result (backed by a new `createdAt` field on `RecallResult`), and `showWhy` renders a terse per-result `why:` line from `strategyScores` via the exported `formatWhyLine` helper — the CLI's human recall output and the Pi tool's formatted results reuse that same helper, so the renderings cannot drift. The MCP transport needs neither flag: it returns the full `RecallResponse` JSON, which already carries the provenance fields and (when `explainScores` is set) `strategyScores`.

## Key Design Decisions

### Fast Write / Slow Extract
The #1 lesson from Hearthmind: never block writes on LLM calls. `retain()` makes no LLM call — it embeds in-process and stores in one SQLite transaction. The SQLite write itself is ~5ms; end-to-end retain latency is dominated by the local embedding (typically tens of ms on CPU with the default in-process model, plus a one-time model download on first use). Entity extraction runs in a background queue via Ollama. The knowledge graph builds up over time without impacting agent responsiveness.

### Trust Layer (Unique to Engram)
Neither Memvid nor Hindsight tracks provenance. Every chunk carries `source_type` (user_stated, inferred, external_doc, tool_result, agent_generated) and `trust_score` (0.0-1.0). The recall pipeline weights results by trust. Security rule: external docs and tool results can NEVER override core agent directives regardless of trust score. This is enforced structurally in recall: final ranking sorts by (source tier, fused score) — tier 0 `user_stated`, tier 1 `inferred`/`agent_generated`, tier 2 `tool_result`/`external_doc` — so no trust score or relevance lets a lower tier outrank a higher one. **Memory type is NOT part of this floor** — it used to be a second lexicographic sort key (`world` > `observation` > `experience` > `opinion`) but that meant a weakly-related `observation`/`world` chunk always beat a strongly-related `experience` chunk regardless of relevance, which surfaced as a real bug (issue #18: a rank-1-by-cosine-distance `experience` chunk never appeared in results at all). Memory type is now a soft multiplicative weight applied in `applyWeighting()` alongside trust/decay/strategy-boost — world/observation get a gentle boost, opinion a gentle penalty (still consistent with the 0.85 opinion-confidence cap in `formatForPrompt`), but a strong match can beat a weak one regardless of type. **Within a tier, semantic hits now score cosine-primary** (issue #18 remediation, D6): the raw cosine similarity threaded out of `semanticSearch` is the base score, with only a gentle `0.94–0.99` trust tiebreak on top, so relevance magnitude — not just RRF rank ordinal — drives within-tier ranking and a strong single-strategy semantic match outranks a weak multi-strategy one. Keyword/graph/temporal-only chunks (no cosine) keep the old RRF-sum × `0.6–1.2` trust formula, so non-semantic recall is unchanged; `RecallOptions.minScore` doubles as a cosine relevance gate on the semantic path (guidance `0.4–0.45`). The `(tier, score)` comparator is byte-identical — cosine-primary scores, though larger-ranged, stay strictly within their own tier, so the security floor is untouched (proven by a tier-2-cosine-1.0-vs-tier-0-cosine-0.0 test). A truncation reserve re-fetches tier-0 matches when volume fills a strategy's candidate window. Tunable via `RecallOptions.sourceTiers` / `RecallOptions.memoryTypeRank` (defaults in `DEFAULT_SOURCE_TIERS` / `DEFAULT_MEMORY_TYPE_RANK`). Consumers must NOT read `results[0]` as the highest-relevance match overall — it is the best match in the highest-present tier; re-sort by `score` locally where pure relevance is genuinely needed (as `findToForget` in the Pi adapter does). At the ingest layer, extraction and reflection prompts delimit memory content inside labeled `untrusted_data` blocks and clamp `disposition` config to validated numbers — an injection mitigation, not a guarantee.

**Recency decay tradeoff (read before building a long-term-memory app):** `RecallOptions.decayHalfLifeDays` defaults to 180 — at that default, a chunk's score is multiplied by `2^(-ageDays/180)`, so anything roughly 18+ months old is functionally unrecallable regardless of relevance, silently, with no warning (issue #19). This is a reasonable default for a short-lived coding-session context where recent decisions genuinely matter more, but it is a real trap for a personal-assistant/journal-style use case meant to have continuity across years. If you're building that kind of consumer, set `decayHalfLifeDays: 0` (or a much longer half-life) explicitly — don't rely on the library default. `integrations/pi`'s general `engram_recall` tool does exactly this.

### SQLite Over Postgres
Hindsight uses embedded Postgres. Engram uses SQLite because:
- Proven in our OpenClaw memory-ragkg work
- Single file per agent = portable, git-committable, backup-friendly
- sqlite-vec for vector search, FTS5 for keyword search
- Zero Docker containers required
- Runs on the homelab without additional services

SQLite runs in WAL mode (`journal_mode = WAL`, `synchronous = NORMAL`) so
multiple processes can read while one writes — this is what lets the
`engram-aql` Rust binary share a live `.engram` file with the TypeScript
library, and what lets multiple agents hit a `shared.engram` concurrently.
At runtime the main `.engram` file is accompanied by `.engram-wal` and
`.engram-shm` sidecar files. For backups, use `engram.backup(destPath)`
rather than raw `cp` — the backup method runs SQLite's online backup API
and produces a single standalone file with no sidecars.

### Batch Reflect Over Real-Time
Hindsight reflects on every query. Engram reflects on a schedule (default: manual trigger, configurable to hourly/daily). This keeps Ollama usage bounded while still building observations and opinions over time. A single `reflect()` call drains ONE batch of ≤`batchSize` (default 50) unreflected facts, so a large backlog never catches up on a one-batch-per-trigger cadence — and raising `batchSize` doesn't help: an oversized batch overruns the model, produces zero insights (its facts are deliberately left unreflected, `reflect.ts`), and the issue-#17 adaptive-shrink hint halves the next batch. Per-batch size is model-bounded; throughput comes from looping. `reflectCatchUp()` (D5, issue #19-adjacent) is that loop: a bounded multi-batch pass around the untouched `reflect()` for off-peak backlog drain. It stops on `drained` (backlog < `minFactsThreshold`), `capped` (`maxBatches`/`maxFacts`/`maxDurationMs`), `stalled` (`maxStalls` **consecutive** zero-progress batches — leaving `batchSize` undefined lets the shrink hint self-heal one overrun before giving up), or `failed` (an inner batch errored, e.g. Ollama down), returning aggregated insight counts + `remainingBacklog`. Exposed as `Engram.reflectCatchUp()` and `new ReflectScheduler(cfg, { catchUp: true })`; **library-only, adds ZERO MCP tools** (surface-parity stays 16).

### Four Memory Types
Inspired by Hindsight's four networks, mapped to biological memory:
- **World** = semantic memory (facts about the world)
- **Experience** = episodic memory (what the agent itself did)
- **Observation** = consolidated knowledge (patterns across facts)
- **Opinion** = belief with confidence (strengthens or weakens with evidence)

## File Extension

Agent memory files use the `.engram` extension (they're SQLite files):
```
~/.valor/memory/myAgent.engram
~/.valor/memory/sit-agent.engram
~/.valor/memory/shared.engram
```

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| better-sqlite3 | SQLite driver for Node.js | Yes |
| sqlite-vec | Vector similarity search extension | Yes (for semantic recall) |
| @huggingface/transformers | In-process embeddings (retain + recall) | Yes (default embedding path) |
| Ollama | Local LLM for extraction + reflection | Yes (for extract + reflect) |
| llama3.1:8b | Fast model for extraction + reflection | Recommended |
| nomic-embed-text (Ollama) | Ollama embedding model | Only when `useOllamaEmbeddings: true` |

**Herd alternative:** swift-innovate/herd exposes the same Ollama HTTP API on port `40114`. Use `ollamaUrl: 'http://localhost:40114'` to point Engram at Herd instead.

## File Structure

```
engram/
├── CLAUDE.md           ← you are here (source of truth)
├── AGENTS.md           ← verbatim mirror of CLAUDE.md for AGENTS-standard tools (keep in sync)
├── README.md           ← public-facing docs
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── src/
│   ├── schema.sql               ← full database schema (13 tables, FTS5, triggers)
│   ├── engram.ts                ← unified Engram class + public API exports
│   ├── retain.ts                ← fast write + dedup + batch import + extraction queue
│   ├── recall.ts                ← four-way retrieval + RRF + trust/decay weighting + formatForPrompt + selectInsights() semantic opinion/observation ranking
│   ├── reflect.ts               ← scheduled learning engine + prompt templates + reflectCatchUp() backlog-drain runner (D5) + opinion gates, belief journal, counter-evidence pass & falsifier weakening (issue #38) + step-0.5 procedural-suggestion pass integration (issue #39) + backfillInsightEmbeddings()
│   ├── insight-shared.ts        ← gate/similarity/JSON-parse/embedding-codec helpers shared by reflect.ts + suggest.ts + recall.ts (issue #39 pure-move extraction, zero behavior change)
│   ├── suggest.ts               ← procedural-suggestion pass (issue #39): watermark scan (corrections/friction/workflow signals) → gated/deduped suggestions + audit journal
│   ├── extract-cpu.ts           ← zero-LLM inline entity extraction (Tier 1)
│   ├── temporal-parser.ts       ← natural language date parsing for temporal recall
│   ├── generation.ts            ← pluggable generation providers (Ollama, OpenAI-compat, Anthropic); model is REQUIRED (no default) + UnconfiguredGeneration fail-loud sentinel
│   ├── model-resolver.ts        ← single source of generation-model selection (no silent default) + Ollama /api/tags preflight; roles reflect/extract/integration
│   ├── local-embedder.ts        ← in-process embeddings via @huggingface/transformers
│   ├── working-memory-types.ts  ← types for working memory session management
│   ├── context-store.ts         ← ContextStore: task-scoped ephemeral artifacts (commit/query/expire/promote), reuses recall()'s RRF pipeline via chunks.scope='task'
│   ├── introspect.ts            ← projection-only read primitive for held state (opinions + observations by subject, no confidence floor); consistency check deferred
│   ├── readonly-engram.ts       ← ReadonlyEngram: capability-restricted read-only view over a {readonly:true} connection — the grounding layer's write guarantee (Engram.readonlyView())
│   ├── grounding.ts             ← Subagent Grounding Layer: groundSubagent/taskContext (belief-free read path) + SubagentReport/metabolizeReport (orchestrator-side metabolism)
│   ├── mcp-tools.ts             ← MCP tool definitions (16 tools: retain/recall/reflect/extract/forget/supersede/session/queue_stats/requeue_failed/introspect/embed/context_commit/context_query/context_promote/suggestions/resolve_suggestion)
│   ├── mcp-server.ts            ← standalone MCP stdio server (engram-mcp bin)
│   ├── cli.ts                   ← `engram` CLI: one subcommand per MCP tool, --json contract for Pi (engram bin)
│   └── cli-args.ts              ← CLI argv parser + Engram.open option-builder + shared validation/clamp helpers
├── tests/                        ← TS suites incl. aql-* cross-process (652 tests via npm test; +142 from integrations/pi, +67 from tools/openclaw-import)
│   ├── helpers.ts
│   ├── retain.test.ts
│   ├── retain-gate.test.ts
│   ├── recall.test.ts
│   ├── trust-tier.test.ts        ← source-tier ranking floor + prompt-injection mitigations
│   ├── reflect.test.ts
│   ├── belief-journal.test.ts     ← issue #38: opinion formation gates + per-belief audit journal (13 tests)
│   ├── counter-evidence.test.ts   ← issue #38 item 2: active counter-evidence pass before formation/reinforcement (10 tests)
│   ├── falsifier.test.ts          ← issue #38 item 3: would_change_this field + unanswered-contradiction weakening (10 tests)
│   ├── suggestions.test.ts        ← issue #39: procedural-suggestion pass — watermark scan, default-on gates, dedup/dismissal/reopen, recall/groundSubagent isolation (14 tests)
│   ├── extract-cpu.test.ts
│   ├── temporal-parser.test.ts
│   ├── generation.test.ts
│   ├── model-resolver.test.ts     ← model selection precedence + required-model throw + preflight
│   ├── engram.test.ts
│   ├── working-memory.test.ts
│   ├── context-store.test.ts      ← commit/query round-trip, TTL expiry, budget truncation, RRF parity, shared-parent integration example
│   ├── introspect.test.ts         ← held-state projection: no-floor visibility, full evidence/lifecycle shape, subject match, toggles/limit
│   ├── readonly-engram.test.ts    ← read-only guarantee: no write method on the surface + driver-level SQLITE_READONLY, read parity, WAL visibility
│   ├── grounding.test.ts          ← grounding layer: opinions never leak, empty-intersection fallback, parent-scope isolation, report→durable metabolize
│   ├── local-embedder.test.ts
│   ├── agent-integration.test.ts
│   ├── mcp-server.test.ts
│   ├── cli.test.ts               ← engram CLI: per-subcommand happy path, --json contract, stdin, exit codes (35 tests)
│   ├── surface-parity.test.ts     ← drift guard: ENGRAM_TOOLS ↔ CLI subcommands stay 1:1, tool count pinned
│   ├── aql-schema.test.ts         ← AQL schema/parse checks
│   ├── aql-equivalence.test.ts    ← L2: AQL results match TS recall/scan/load semantics
│   └── aql-e2e-process.test.ts    ← L3: cross-process WAL handoff (spawns the Rust binary; needs cargo)
├── docs/
│   ├── OPENCLAW-INTEGRATION.md   ← OpenClaw memory plugin setup guide
│   └── PI-INTEGRATION.md         ← Pi.dev (pi-mono) extension setup guide
├── integrations/
│   ├── README.md                 ← adapter map (OpenClaw external; Pi in-repo)
│   └── pi/                       ← Pi extension package (engram-pi)
│       ├── package.json          ← pi.extensions field; engram via file:../..
│       ├── src/
│       │   ├── index.ts          ← Pi binding: registers commands + LLM tools, lifecycle
│       │   ├── adapter.ts        ← pure logic: takes Engram, returns plain objects
│       │   └── types.ts          ← typebox schemas for the seven LLM tools (core + session)
│       └── tests/                ← 121 tests (adapter, scheduling, auto-retain, startup-recall, session-bridge, recall-passthrough, smoke + built-dist)
├── skills/
│   ├── engram.md                  ← portable agent skill (covers all 16 MCP tools)
│   ├── engram-session.md          ← working memory session skill
│   └── cli-memory/SKILL.md        ← Pi-facing `engram` CLI contract (per-command --json shapes, exit codes, when to recall/retain/supersede/session)
├── tools/
│   └── openclaw-import/           ← CLI to import OpenClaw memory files into .engram
│       ├── README.md
│       ├── package.json
│       ├── tsconfig.json
│       └── src/
│           ├── index.ts           ← CLI entry point (commander)
│           ├── import.ts          ← orchestrator: classify → parse → map → retainBatch
│           ├── classify.ts        ← path-based file classification + skip patterns
│           ├── dates.ts           ← date extraction from filenames and content
│           ├── parser.ts          ← H2-split markdown chunker with size limits
│           ├── mapping.ts         ← category → memory type + trust score mapping
│           ├── types.ts           ← shared interfaces
│           └── tests/             ← 67 unit tests (classify/parser/mapping/dates)
├── engram-aql/                   ← companion Rust crate (merged, PR #1): AQL read surface over the shared .engram
│   ├── Cargo.toml
│   ├── src/                       ← executor, memory_map, schema, result, error; mcp/, statements/, subcommand/, sql/
│   └── tests/                     ← L1 Rust integration tests (recall/scan/lookup/load/aggregate/graph/pipeline/mcp_roundtrip)
├── evals/                        ← retrieval-quality eval harness (`npm run eval`; embedding-only, report-only, NOT a CI gate)
│   ├── run.ts                     ← entry: seed fixtures → recall → P@5/R@5/MRR → results.json + stdout tables
│   ├── metrics.ts / fixture-builder.ts / types.ts   ← real retain()/supersede() seeding, raw-SQL created_at backdating
│   ├── fixtures/ + scenarios/     ← four families: relevance, contradiction, contamination, staleness
│   └── README.md                  ← methodology + committed baselines (the evidence base for staleness/expiry decisions)
└── examples/
    └── basic-usage.ts
```

## Usage

### Initialize an engram
```typescript
import { Engram } from 'engram';

const myAgent = await Engram.create('./myAgent.engram', {
  reflectMission: 'Focus on architecture preferences, project patterns, and infrastructure decisions.',
  retainMission: 'Prioritize technical decisions, code patterns, and project context. Ignore greetings.',
  ollamaUrl: 'http://localhost:11434',
});
```

### Store a memory
```typescript
await myAgent.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',
  source: 'conversation:valor-engine-planning',
  sourceType: 'user_stated',
  trustScore: 0.9,
  context: 'infrastructure',
});
```

### Recall memories
```typescript
const response = await myAgent.recall('What IaC tools does Tom use?', { topK: 5 });
// response.results   → ranked chunks from four retrieval strategies
// response.opinions  → relevant beliefs with confidence scores
// response.observations → synthesized knowledge
```

### Run reflection
```bash
# CLI
npx tsx src/reflect.ts ./myAgent.engram

# Or programmatically
const result = await myAgent.reflect();
// result: { observationsCreated: 3, opinionsFormed: 1, opinionsReinforced: 2, ... }
```

### Process entity extraction queue
```typescript
await myAgent.processExtractions();
// Runs Ollama against queued chunks, builds out entity graph
```

## Harness Integrations

Engram is harness-agnostic. Adapters live in `integrations/`. See `integrations/README.md` for the index.

### OpenClaw (production)

External `memory-engram` plugin in the OpenClaw workspace, consumed via `mcporter` subprocess. Production-verified by the Tracer agent (2026-03-24, 16/17 stress tests passed). See `docs/OPENCLAW-INTEGRATION.md`.

- Plugin bridges `memory_search` / `memory_get` → `engram_recall` via mcporter
- Markdown sync ingests `workspace/memory/*.md` into Engram automatically
- Migration CLI: `tools/openclaw-import/` (one-shot bulk-load of OpenClaw markdown into `.engram`)
- Known: ~10s latency from mcporter cold-start (fix: daemon mode or direct import)

### Pi.dev (`pi-mono`)

In-repo extension at `integrations/pi/`, loaded by Pi via Node.js + `jiti` (in-process, millisecond-latency). Project-local DB at `.engram/pi.db`. See `docs/PI-INTEGRATION.md`.

Surface: five slash commands (`/remember`, `/recall`, `/memory`, `/forget`, `/session`) and seven LLM tools (`engram_remember`, `engram_recall`, `engram_memory_stats`, `engram_forget`, `engram_session_resume`, `engram_session_update`, `engram_session_snapshot`).

- Adapter is pure (`integrations/pi/src/adapter.ts`); Pi binding (`index.ts`) registers commands/tools and owns lifecycle + transient state
- Lifecycle hooks: `session_start` (lazy DB open + fresh-session detection), `before_agent_start` (session-addendum + one-shot startup recall + pending-suggestions hint), `turn_end` (background extract/reflect scheduling), `message_end` (auto-retain), `session_shutdown` (flush + close)
- **Working-memory session bridge:** `engram_session_*` tools wrap infer/update/snapshot of working sessions across turns; a transient `currentSessionId` backs the `/session` command
- **Background consolidation:** every few turns drains the extraction queue / runs reflection, fire-and-forget so a turn never blocks on Ollama; warns once per session if Ollama is unreachable. Tunable via `ENGRAM_PI_EXTRACT_EVERY_TURNS` / `ENGRAM_PI_REFLECT_EVERY_TURNS` / `ENGRAM_PI_EXTRACT_BATCH`. **Backend selection** (`openWithResolvedModel`, `index.ts`) mirrors `engram-mcp`'s `--generation-endpoint`/`--generation-model` pair, from env: `ENGRAM_GENERATION_ENDPOINT` (+ `ENGRAM_GENERATION_API_KEY`, default `noauth`) routes consolidation to an OpenAI-compatible server (llama.cpp/vLLM/Herd) at the resolver's model; otherwise Ollama at the resolved host. The endpoint needs its own opt-in rather than being inferred from the host — `OllamaGeneration` speaks `/api/generate`, which llama.cpp does not serve, so a host alone cannot express the wire protocol. Pass the bare origin, NOT a `/v1` URL: `OpenAICompatibleGeneration` appends `/v1/chat/completions`. Until 2026-07-16 this factory dropped the resolver's `host` and passed only `reflectModel`, silently pinning every consolidation to `localhost:11434` regardless of `ENGRAM_OLLAMA_URL`/`ENGRAM_INTEGRATION_HOST` — a deployment could not point consolidation anywhere at all, and against an unreachable/unconfigured backend it burned each queued chunk's retry attempts and drove it to `failed`
- **Auto-retain:** captures conversation messages as `experience` chunks off `message_end` (on by default; `ENGRAM_PI_AUTO_RETAIN=0` disables). Tool/bash output is stored at the lowest trust tier (`tool_result`) so it can never outrank a user directive at recall
- **`engram_recall` passthrough:** widened (2026-07-09) to pass `memoryTypes` / `after` / `before` / `strategies` / `minScore` straight through to `engram.recall()`, matching what MCP/CLI callers already had — previously only `topK`/`minTrust` were exposed to the LLM tool. Widened again (2026-07-16) with `explainScores` and `decayHalfLifeDays` — Pi's no-decay default stays hardcoded 0 (issue #19), the param is per-call opt-IN to recency weighting (`input.decayHalfLifeDays ?? 0`). The tool's formatted output was converged onto core rendering at the same time (provenance bracket + the exported `formatWhyLine` — the hand-rolled trust/source line is gone), and `strategyScores` lands in the tool-result `details` only when `explainScores` is requested
- **Startup recall:** closes the "written automatically, read manually" gap for fresh sessions only. `session_start` sets a transient `sessionIsFresh` flag via `isFreshSessionStart(reason, priorMessageCount)` in `adapter.ts` — fresh for `reason === 'new'` unconditionally, and for `reason === 'startup'` only when there are zero prior `type === 'message'` session entries. This two-part check exists because live validation against a real Pi runtime found that non-interactive `pi -p` launches (and in fact every initial process launch, interactive or not) report `reason: 'startup'`, never `'new'` — `'new'` only fires for an explicit mid-process session switch — and Pi appends bookkeeping entries (`model_change`, `thinking_level_change`) before `session_start` fires on every launch, so a raw entry count is never zero even for a genuinely blank slate; only counting `'message'`-type entries fixes both false negatives. The very next `before_agent_start` (and only that one — the flag is consumed immediately) recalls against `event.prompt` and prepends the formatted result to the system prompt via `startupRecall()` in `adapter.ts` (reuses core's `formatForPrompt`, budget-capped, same convention as the session-resume `relatedContext`). Sessions continued via `--continue`/`--resume`/`--session` and later turns are unaffected — this is a one-time "create starting context" injection, not a per-turn recall. On by default; tunable via `ENGRAM_PI_STARTUP_RECALL` / `ENGRAM_PI_STARTUP_RECALL_MAX_CHARS` / `ENGRAM_PI_STARTUP_RECALL_TOPK`
- **Pending-suggestions hint (#39):** `before_agent_start` appends a one-line counter (`N pending improvement suggestions (run \`engram suggestions\` to view).`) when any suggestion sits in `proposed`, via `pendingSuggestionsHint()` in `adapter.ts`. Deliberately a COUNTER, never suggestion bodies — that is the issue's "pull-primary, push-minimal" contract, and injecting full bodies every session is the low-precision nag the feature's own design argues against. Returns null at zero pending so no line is appended rather than a token-costing "0 suggestions". Count saturates at `PENDING_SUGGESTIONS_HINT_CAP` (50, rendered `50+`) — past that the exact number prompts no different action and the cap bounds the query. One-shot on a fresh session, same gating as startup recall, but an independent try/catch and its own flag (`ENGRAM_PI_SUGGESTION_HINT`) so either feature can be run without the other and neither failure suppresses the other
- **Test env isolation (issue #45):** `integration-smoke.test.ts` reads/writes `ENGRAM_PI_DB_PATH` (override test) and depends on the extension's other `ENGRAM_PI_*` env-driven defaults; without isolation, an ambient value inherited from the invoking shell (e.g. exported in a systemd unit per the override-config guidance above) resolves `getEngram()` to a real store instead of the test's temp `projectDir` — hit live via `Embedding dimension mismatch: database contains 18004 chunks with 768d embeddings, but the current embedder produces 8d vectors`, which failed closed only because the dimension guard happened to catch it. `beforeEach`/`afterEach` now snapshot-clear-restore every `ENGRAM_PI_*` var (`clearEngramPiEnv`/`restoreEngramPiEnv`, scoped by prefix so unrelated env vars are untouched), so no test — including ones that set an override mid-test — can leak state into its neighbors regardless of the ambient shell.
- 142 tests (pure adapter + binding lifecycle + built-dist smoke), gated in CI on Node 20 and 24
- Deferred: memory-inspector UI widget (`ctx.ui.custom()`); `pi install`-able packaging

## Integration with valor-engine

Engram is a standalone library. valor-engine consumes it as a dependency:

```typescript
// valor-engine/engine/memory/index.ts
import { Engram } from 'engram';

// One engram per operative
const myAgentMemory = await Engram.open('./myAgent.engram');
const sitMemory = await Engram.open('./sit-agent.engram');

// Auto-retain conversation turns
gateway.on('message', async (msg) => {
  await myAgentMemory.retain(msg.text, {
    memoryType: 'experience',
    source: `conversation:${msg.conversationId}`,
    sourceType: 'user_stated',
    trustScore: 0.9,
  });
});

// Inject context before LLM calls
const context = await myAgentMemory.recall(userMessage, { topK: 10 });
const systemPrompt = buildPromptWithMemory(basePrompt, context);
```

## engram-aql (Rust) — merged

A companion crate, `engram-aql/`, ships in this repo (merged via PR #1). It is a separate Rust process that shares the `.engram` SQLite file with TypeScript Engram via WAL: TS owns writes (retain, embedding, extraction, reflection); the Rust binary owns the AQL declarative read surface (`RECALL`, `SCAN`, `LOOKUP`, `LOAD`, `AGGREGATE`, `ORDER BY`, `WITH LINKS`, `FOLLOW LINKS`, `PIPELINE`). Write statements `STORE`/`UPDATE`/`FORGET`/`REFLECT` are delegated to the canonical TS retain pipeline over the bridge (Phase 2b) — Rust never writes the file itself; only `LINK` is still rejected (no canonical TS manual-relation surface). Phase 2a adds query-side vector similarity: `RECALL ... LIKE $var` / `PATTERN $var THRESHOLD t` rank chunks by cosine distance via a native `vec_distance_cosine` scalar fn registered on the Rust connection (no `sqlite-vec` dependency in Rust — it decodes the same LE-f32 BLOBs TS writes). The probe is resolved from a bound variable: a query string is embedded through a lazily-spawned, warm `engram-mcp` child (the new `engram_embed` MCP tool, so the embedding is model-compatible with stored vectors), or a precomputed embedding array is used directly (no child spawn). Variables are passed via the `engram_aql` MCP tool's `variables` object or the `query` CLI's repeatable `--var name=value`.

**Where to look:**
- Crate: `engram-aql/` — `src/{executor,memory_map,schema,result,error,lib,main}.rs` plus `src/{mcp,statements,subcommand,sql}/`
- Subcommands: `engram-aql query <db> '<aql>'` (one-shot JSON), `engram-aql repl <db>` (interactive), `engram-aql mcp <db>` (stdio MCP server exposing `engram_aql` tool)
- Tests: L1 Rust integration tests in `engram-aql/tests/` (recall/scan/lookup/load/aggregate/graph/pipeline/mcp_roundtrip); L2 semantic-equivalence + L3 cross-process suites in TS (`tests/aql-*.test.ts`) validate the cross-process WAL handoff against TS-created `.engram` files. The L3 suite spawns the Rust binary, so the AQL tests need `cargo` present (they're skipped/failing without it).
- Spec: `docs/superpowers/specs/2026-04-12-engram-aql-rust-binary-design.md`
- Plan: `docs/superpowers/plans/2026-04-12-engram-aql-rust-binary.md`

**Phase 2a (vector search) — done.** `LIKE`/`PATTERN` semantic search wired in RECALL (`statements/recall.rs`); native `vec_distance_cosine` + LE-f32 codec (`src/vector/`); warm `engram-mcp` bridge for query embedding (`src/bridge/`, synchronous `std::process`); `ExecCtx` variable threading (`src/exec_ctx.rs`); `variables` surface in the MCP tool + `--var` in the `query` CLI. `LIKE`/`PATTERN` only apply to chunks-backed memory (SEMANTIC/EPISODIC); on PROCEDURAL/WORKING/TOOLS they warn (no embeddings). L2 equivalence (`tests/aql-vector-equivalence.test.ts`) proves the native cosine matches sqlite-vec ordering. Design: `docs/superpowers/specs/2026-06-24-engram-aql-writes-and-vector-search-design.md`; plan: `docs/superpowers/plans/2026-06-24-engram-aql-writes-and-vector-search.md`.

**Phase 2b (write delegation) — done.** Writes are translated to TS tool calls over the same bridge (`statements/write_delegate.rs`), so Rust stays DB-read-only: `STORE`→`engram_retain` (payload fields → retain options; **pass-through provenance** — unspecified `sourceType`/`trustScore` inherit retain's defaults), `UPDATE`→`engram_supersede` (target ids resolved via a RO read, superseded each), `FORGET`→`engram_forget` (RO-resolve ids, forget each), `REFLECT`→`engram_reflect` (global cycle; source/THEN filters not yet honored). Writes target chunk-backed memory only (SEMANTIC/EPISODIC). The bridge gained a generic `call_tool` (`src/bridge/call.rs`); `embed_query` reuses it. Cross-process WAL visibility holds with no refresh — Rust RO autocommit reads see the bridge child's commits immediately (`tests/write_delegate.rs`, gated on `engram-mcp`). `LINK` remains rejected.

**Status — FROZEN at Phase 2 (2026-07-14, see `tasks/decision-freeze-engram-aql.md`).** No Phase 3. The Step 6 audit found engram-aql has **zero in-repo consumers** beyond its own tests (Pi/OpenClaw call neither it nor ContextStore), against a high carrying cost (a whole Rust crate re-deriving TS read semantics, cargo-gated CI). Its only justification was a single inbound OSS question — the weakest demand signal — and every actual agent consumer is already served by the MCP surface; AQL only earns its keep for a Rust process wanting in-process, sub-MCP query access, which no consumer (internal or external) currently is. **Scope while frozen:** keep it building and CI-green, add no new AQL surface. **Thaw trigger (falsifiable):** a Rust consumer produces a concrete artifact — a PR/issue against the crate from the external user, OR a named internal fleet consumer (Substrate, Cradel/Mira-core, Herd) wanting in-process Rust reads. One more inbound *question* does not count. If keep-green itself starts costing real maintenance on toolchain bumps, the decision converts toward cut.

**Frozen leftover work (was "deferred"; not resumed unless thawed):** a canonical TS surface for manual `LINK` (relations are extraction-derived today); transactional `PIPELINE` mixing reads + writes (a stage that reads its own prior write within one pipeline).

## Git

```bash
git init && git branch -M main
```

## Decisions

- [x] **License**: Apache 2.0. Set in `package.json`. Maximizes adoption for a utility library. Revisit to BSL 1.1 only if commercial protection becomes a requirement.

- [x] **sqlite-vec loading**: Prebuilt binaries via npm (`sqlite-vec` package). No compile-from-source step. The `Engram` class loads it with a `try/catch` — graceful fallback to 3-strategy recall if absent.

- [x] **Shared engram**: Naming convention. Multiple agents call `Engram.open('./shared.engram')`. `Engram.open()` sets `journal_mode = WAL`, `synchronous = NORMAL`, and `busy_timeout = 5000` so concurrent readers never block each other or the writer, and a held write lock resolves within the timeout window. Concurrent `retain()` calls serialize at the writer (SQLite allows one writer at a time), but reads proceed fully in parallel. No separate API needed. The same pragma set enables cross-process sharing with the `engram-aql` Rust binary.

- [x] **MCP tool surface**: 16 tools exposed in `mcp-tools.ts` — `engram_retain`, `engram_recall`, `engram_reflect`, `engram_process_extractions`, `engram_forget`, `engram_supersede`, `engram_session`, `engram_queue_stats`, `engram_requeue_failed`, `engram_introspect`, `engram_embed`, `engram_context_commit`, `engram_context_query`, `engram_context_promote`, `engram_suggestions`, `engram_resolve_suggestion`. Agents pick what they need; `forget`/`supersede` enable correction loops, `session` carries working-memory context across turns, `queue_stats` reports extraction-queue health (counts + a `failed_reasons` error breakdown) for ops/diagnostics, `requeue_failed` re-drives terminally-failed extraction items after a transient outage (failed is otherwise permanent after 3 attempts — added from live-deployment feedback where an LLM-host outage stranded recoverable items), `introspect` is the structured read primitive for held state — returns opinions (belief + confidence + support/challenge provenance + lifecycle timestamps) and observations for a subject via a direct lookup with **no confidence floor**, so weakly-held or freshly-challenged beliefs (which `recall` gates out at 0.5) stay visible; it is **projection only** (every field maps to an existing `opinions`/`observations` column, no schema change, no LLM call) and REPORTS state rather than adjudicating truth — the consistency check (does a candidate statement agree/contradict a belief) is deliberately deferred as a separate primitive with its own cost/latency/model-role decision, `embed` returns a vector in the bank's stored embedding space (query|document mode — primarily the bridge surface `engram-aql` uses for AQL vector search, but any consumer needing model-compatible vectors can call it), `context_commit`/`context_query`/`context_promote` expose the previously-unreachable ContextStore (task-scoped agent-to-subagent handoff — see the ContextStore bullet below) to any MCP-only agent, `suggestions`/`resolve_suggestion` (issue #39) expose the procedural-suggestion read/write surface — list by status/kind/domain, and set lifecycle status where `'proposed'` is a legal target (manual reopen) — see the Procedural reflection bullet below. `engram_session` also gained an `action` enum (`resume` default | `update` | `snapshot`) so an MCP-only agent can drive the full working-session lifecycle without a direct API call — omitting `action` is byte-identical to the pre-existing tool. `engram_reflect` gained an optional `suggest: boolean` (maps to `suggestions: {}` all-defaults) so an MCP-only agent can form suggestions without a direct API call — omitting it is byte-identical to the pre-existing tool. `tests/surface-parity.test.ts` pins the tool count and the CLI↔MCP 1:1 mapping, so adding a tool without its CLI twin (or without updating the pinned count) fails the suite.

- [x] **CLI transport** (`engram` bin, `src/cli.ts`): a third transport over the same `Engram` core, structurally a sibling of `mcp-server.ts` — parse argv → `Engram.open()` → dispatch to the SAME methods → print. One kebab-cased subcommand per MCP tool (`retain`/`recall`/`reflect`/`process-extractions`/`forget`/`supersede`/`session`/`queue-stats`/`requeue-failed`/`introspect`/`embed`/`context-commit`/`context-query`/`context-promote`/`suggestions`/`resolve-suggestion`) so `skills/cli-memory/SKILL.md` maps 1:1 to the tool surface; the mapping is enforced by `tests/surface-parity.test.ts` against the canonical `CLI_COMMANDS` list in `cli-args.ts`. `session` takes a `--action update|snapshot` flag (requiring `--session-id`) mirroring the MCP tool's action enum; default (no `--action`) behavior is unchanged. `reflect` takes `--suggest` (maps to `suggestions: {}`, all defaults) mirroring `engram_reflect`'s `suggest` boolean — default (no flag) unchanged. Built for use as a Pi coding-agent skill: `--json` on every command emits the raw method return to stdout and nothing else (the stable Pi integration contract); diagnostics go to stderr; the primary text arg is read from stdin when omitted (pipe context in); exit codes are 0 success / 2 not-found (`forget`/`supersede` missing chunk, or `session`/`context-promote` on an unknown id) / 1 error. DB path resolves from `--db <path>` then `ENGRAM_DB`. No retain/recall/reflect logic is duplicated — `src/cli-args.ts` holds the argv parser, the `Engram.open` option-builder (same flags as `engram-mcp`), and the validation/clamp helpers (a canonical copy mirroring the private ones in the frozen `mcp-tools.ts`).

- [x] **Packaging / install**: `package.json` declares two bins — `engram-mcp` (MCP stdio server) and `engram` (CLI) — alongside the library `main`. A `prepare` script runs `npm run build` automatically on install, so installing this repo by **git ref** (e.g. `npm i github:tom-swift-tech/engram`, or `pi install` from a git source) produces a working `dist/` with all three entry points. No npm publish yet — consumers use a git ref or a local checkout; `files: ["dist"]` keeps any future tarball lean.

- [x] **Agent instructions live in two mirrored files**: `CLAUDE.md` is the source of truth; `AGENTS.md` is a verbatim mirror for tools that read the cross-vendor `AGENTS.md` standard. They differ only in the "you are here" marker. **Edit both together** — any change to architecture, file structure, or decisions must land in both files or they drift.

- [x] **Reflect schedule**: Library default is manual (call `engram.reflect()` or the CLI). `ReflectScheduler` class ships for timer-based use. Recommendation for valor-engine: `ReflectScheduler` with a 6-hour default, configurable per operative. For a store whose backlog outruns a one-batch-per-tick cadence, construct the scheduler with `{ catchUp: true }` (or call `engram.reflectCatchUp()`/`reflectCatchUp()` directly on an off-peak cron) so each tick drains many batches instead of one — see the catch-up runner under "Batch Reflect Over Real-Time".

- [x] **Embedding model**: `nomic-ai/nomic-embed-text-v1.5` (768d) runs in-process via `@huggingface/transformers` 4.2.0 (the maintained successor to the deprecated `@xenova/transformers`) as default — no Ollama required for retain/recall, and no Hugging Face token (the `nomic-ai` upstream repo is public). Override via `embedModel` option. `Xenova/all-MiniLM-L6-v2` (384d) is a valid alternative for lower disk/memory use. The legacy `Xenova/nomic-embed-text-v1.5` mirror is now gated (401 without a token) so it's no longer the default, but ships identical 768-dim weights — existing `.engram` files stay valid. Opt into Ollama embeddings via `useOllamaEmbeddings: true` (e.g., for GPU acceleration). Existing `.engram` files with Ollama-generated vectors are fully compatible — same model weights, same 768-dim space.

- [x] **No default generation model (fail-loud model selection)**: The library applies **no hardcoded model default** anywhere. `OllamaGeneration`/`AnthropicGeneration` require a non-empty `model` (throw at construction); the previous `?? 'llama3.1:8b'` / `?? 'claude-haiku-...'` defaults are gone, as are the copies in `engram.ts`, `reflect.ts`, `cli-args.ts`, and `mcp-server.ts`. A single resolver (`src/model-resolver.ts`) is the only place a model is chosen from config: `resolveModelSpec({role})` reads `ENGRAM_<ROLE>_MODEL` → `ENGRAM_MODEL` (role ∈ reflect/extract/integration; recall is embedding-based, no generation model) and **throws** if unconfigured; `resolveModelSpecOrNull` returns null for the optional path. When nothing is configured, `Engram.open` installs a fail-loud `UnconfiguredGeneration` sentinel — retain/recall still work with zero config, but the first reflect/extract throws instead of silently 404ing on a default the host may not serve. A `preflightModel({host,model})` hits the host's `/api/tags` and confirms the model is served **before** a pipeline commits, logging the served-model list on failure and flagging `:cloud`/non-LAN models as remote; it is wired into the generation entrypoints (reflect CLI entry, `engram` CLI reflect/process-extractions, `engram-mcp` startup, and the Pi integration's background consolidation, which warns once loudly). This closes a class of silent failure: a model name dropped before reaching the library used to fall back to an unserved default and 404 downstream, decoupled in time from the misconfiguration.

- [ ] **`.engram` MIME type**: Deferred. Extension is established; OS MIME registration is future work if IDE/tooling support becomes valuable.

- [x] **ContextStore (task-scoped ephemeral context)**: A fifth, short-lived scope alongside the four Hindsight memory types, for cheap agent-to-subagent context handoff (`commitContext`/`queryContext`/`expireContext`/`promoteToDurable` in `src/context-store.ts`). Discriminated via a new `chunks.scope IN ('durable', 'task')` column — deliberately **not** named "working memory": that name is already taken by the pre-existing `working_memory` table (session goal/progress state, its own cosine-matched retrieval, and the thing `engram-aql`'s AQL `Working` memory type already maps to). ContextStore artifacts are immutable, multi-artifact-per-task, and ranked through the **same RRF-fusion `recall()` pipeline** as durable memory — `scope`/`parentRef` are just two more `RecallOptions` filters (`recall.ts`), not a forked ranking path. `recall()` defaults `scope` to `['durable']`, so pre-existing callers and all prior tests are unaffected. New nullable `chunks` columns: `scope`, `expires_at`, `parent_ref` (chains to a parent `ContextRef`, never a copy), `agent_id` (originating agent provenance), `artifact_json` (the structured `DecisionArtifact`; `text` holds a flattened searchable rendering for FTS5/semantic/graph). TTL is enforced lazily at query time (`expires_at` checked via `datetime()`-normalized comparison — raw ISO-string vs. SQLite `datetime('now')` comparison silently never expires anything, a real bug caught during implementation), not swept by a background reaper. Task-scoped chunks are excluded from extraction (never enqueued) and from reflect/consolidation (`v_unreflected` view and `reflect.ts`'s inline query both require `scope = 'durable'`) unless explicitly promoted via `promoteToDurable()`, which is a mechanical seam only — it does not itself run `reflect()` or synthesize observations. Migration follows the existing `text_hash` guarded-`ALTER TABLE` pattern in `engram.ts`, with one correction: the new indexes are created *unconditionally after* the column-guards (not inside `schema.sql`'s unconditional index block), because `CREATE INDEX IF NOT EXISTS` on a column that doesn't exist yet fails outright on a pre-existing `.engram` file (unlike `CREATE TABLE IF NOT EXISTS`, which silently no-ops). **Now reachable via MCP + CLI** (2026-07-09): `engram_context_commit`/`_query`/`_promote` (and CLI `context-commit`/`context-query`/`context-promote`) expose the core fns to any agent, not just direct TypeScript callers — previously fully built but unreachable by an LLM. Gotcha worth knowing before use: `queryContext`/`engram_context_query` returns the **children** committed under a ref (rows with `parentRefId` pointing at it), not the artifact stored at that ref itself — querying a freshly-committed root with its own ref id back returns nothing until a child is committed under it.

- [x] **Subagent Grounding Layer (Product A)**: "Grounding in, report out, nothing written by the subagent." An orchestrator (any agent holding a read/write `Engram`) spawns a **stateless** subagent, injects scoped situated context at spawn, and gets a plain report back; every durable write happens on the orchestrator's side. Spec: `docs/GROUNDING-LAYER-SPEC.md`. Two net-new pieces over an otherwise-compositional feature: (1) **`ReadonlyEngram`** (`src/readonly-engram.ts`) — a capability-restricted view exposing only `recall`/`queryContext`/`introspect` (+ the grounding fns), over a **second `{readonly:true}` connection** (`Engram.readonlyView()`). Both enforcement layers from spec §5: no write method on the surface **and** a driver-level `SQLITE_READONLY` backstop so even a raw-SQL escape fails. Verified safe because `recall`/`introspect`/`queryContext` are pure reads (no temp tables/writes). Precondition: the parent `Engram` opened (and migrated) the file first — a readonly connection can't migrate. (2) **`groundSubagent()`/`taskContext()`** (`src/grounding.ts`) — the read path. `groundSubagent` runs `recall()` with `memoryTypes` **intersected** with `['world','experience','observation']` (`opinion` dropped even if asked; empty intersection falls back to all three — never zero, still belief-free), `includeOpinions:false`, durable scope only. **Belief-free by design** (spec §2): a stateless subagent has no revision loop, so injecting `opinion` would export confidence without the correction machinery; beliefs stay behind the orchestrator. `taskContext()` is a deliberate, explicit pass-through to `queryContext()` — task context is orchestrator-selected, never auto-inherited (spec §8). `SubagentReport` + `metabolizeReport()` are the hand-back: the orchestrator is the **single writer**, metabolizing `report.artifact`→`commitContext()` and `candidateExperiences`→`retain()` as `agent_generated` (tier 1, can't outrank user-stated), challengeable by the next reflect cycle. **Library-only — adds ZERO MCP tools**, so `surface-parity.test.ts` stays pinned at 16. Lives in a new file (not inside `recall.ts`), so it touches none of recall's scoring internals and does not collide with the in-flight D6 remediation lane. Tests: `tests/readonly-engram.test.ts` + `tests/grounding.test.ts` (opinions-never-leak, empty-intersection fallback, parent-scope isolation, report→durable→unreflected). Deferred (spec §6): belief injection for the orchestrator's own reasoning (blocked on the disconfirmation retrieval-gap fix), MCP exposure of `groundSubagent` (a deliberate surface-parity change), and any subagent working state.

- [x] **Node-Origin Provenance (groundwork)**: Every durable authored trace records **which Engram instance wrote it**, so a later sync/merge (dumb `.engram` union first, a message bus much later) has provenance natively — nothing to backfill onto un-tagged memories. Additive-only, no distribution: a new nullable `node_origin TEXT` column on `chunks`, `opinions`, AND `observations` (all three durable reflect/retain outputs — the draft plan's Step-2/Step-3 inconsistency was resolved toward including observations so no durable memory needs backfilling), each with a partial `WHERE node_origin IS NOT NULL` index. The instance's identity lives in `bank_config` under key `node_origin`, minted **once** on first open of a bank that lacks it (`INSERT ... ON CONFLICT(key) DO NOTHING` — never regenerated, survives restarts), format `node-<hostnameSlug>-<8hex>`; `Engram.init()` reads it back and holds it on the instance (`private readonly nodeOrigin`) so the write path never re-queries per retain. Stamped on write: `retain()`'s fresh-chunk INSERT (threaded in from the `Engram` wrapper), and `reflect()`'s `insertOpinion`/`insertObs` (reflect opens its own connection, so it reads `node_origin` from `bank_config` once up front). **First author wins** — the dedup UPDATE path never rewrites `node_origin`, and reinforce/challenge/decay/observation-refresh leave it untouched (mutating confidence doesn't change who *formed* the belief). `NULL` = pre-distribution / origin unknown; pre-migration rows stay `NULL` (backfilling would falsely claim this instance authored memories that predate origin tracking). Guarded `ALTER TABLE ADD COLUMN` migration in `engram.ts` (columns first, then indexes unconditionally — same pattern proven for `text_hash` / scope columns; `schema.sql` carries the columns for fresh installs but NOT the indexes). **Explicitly out of scope this sprint** (downstream, only justified by an actual merge): opinion mutability stays in-place (append+supersede is the sync sprint's job), no merge/union/conflict logic, no transport. **Library/schema-only — adds ZERO MCP tools**, so `surface-parity.test.ts` stays pinned at 16. Tests: `tests/node-origin.test.ts` (stable-identity-across-restart, chunk stamping, opinion+observation stamping via reflect, pre-distribution upgrade leaves legacy rows NULL, dedup preserves first author).

- [x] **Opinion formation gates + belief journal (issue #38, items 1+4)**: Reflection previously had no gate between "seen once" and "believed" — a single retained chunk could become an opinion (census on a live store: 66/303 opinions had exactly one supporting chunk), and `reflect_log` recorded only run-level counts, so neither the reasoning behind a belief nor the candidates reflection *declined* to form were reconstructable. Two additions in `reflect.ts`: (1) **`ReflectConfig.opinionGates`** (`minEvidenceCount` / `minDistinctDays` / `minDistinctSources`) gates `direction: 'new'` formation ONLY — reinforce/challenge of an existing opinion is evidence accumulation on an already-formed belief and stays ungated; omitting it applies safe defaults (3 verified chunks across 2 distinct days); `false` is the explicit opt-out. Gates measure **verified** evidence (cited chunk ids that exist and are active — hallucinated ids don't pass gates), days on `date(COALESCE(event_time, created_at))`, and sources with all-NULL bucketing as one. **Rejected evidence merges forward**: a below-threshold candidate is journaled `rejected` (reason `insufficient_evidence`, per-gate required/measured/pass), and when a later cycle re-derives the same belief (same domain, same 0.85 fuzzy bar as opinion dedup) the prior rejection's evidence counts toward the union — so per-batch evaluation can't permanently starve a belief whose evidence arrives one chunk per cycle; the formed opinion then carries the union in `supporting_chunks`/`evidence_count`. (2) **`belief_journal` table** (11th table; plain `CREATE TABLE IF NOT EXISTS` in `schema.sql` — a NEW table needs no guarded ALTER, and its indexes can live in schema.sql because the table exists before they run): append-only, one row per opinion decision per run (`formed`/`reinforced` incl. new-dedup-to-reinforce/`challenged`/`rejected`; `weakened` is reserved in the CHECK for the issue-#38 counter-evidence/falsifier follow-ups), keyed to `reflect_log` via `reflect_run_id`, carrying the candidate belief verbatim, evidence as evaluated, gate results, and an LLM-stated one-sentence `rationale` (new optional field in the reflect prompt's opinion_updates contract, clamped to 1000 chars). Unmatched reinforce/challenge verdicts — previously silent drops — are journaled as `rejected`/`no_matching_opinion`. **Engagement semantics fix**: a cycle whose only output is gate rejections or unmatched verdicts is NOT an issue-#17 silent failure (the model parsed and responded; it's not a context-size problem) — its facts are marked reflected and no shrink hint is written, otherwise the same batch would re-analyze and re-journal duplicates forever. `ReflectResult.opinionsRejected` (also aggregated in `CatchUpResult`) counts gate rejections; there is deliberately no reflect_log column for it — journal rows keyed by the run id ARE the per-run record. Read surface: `getBeliefJournal()` / `Engram.beliefJournal()` (filter by opinion/run/action, newest first) — **library-only, adds ZERO MCP tools**, `surface-parity.test.ts` stays pinned at 16. Tests: `tests/belief-journal.test.ts` (13). Item 2 (counter-evidence pass) and item 3 (falsifier field) landed next — see the following two bullets.

- [x] **Active counter-evidence pass (issue #38, item 2)**: The pre-existing challenge machinery was passive — `challengeOpinion` fired only when a contradiction happened to land in the *same reflect batch*, so a belief could ratchet confidence upward forever while disconfirming chunks sat unretrieved elsewhere in the store. `ReflectConfig.counterEvidence` is safe by default: omitting it enables the per-cycle audit with `{ onReinforce: false, topK: 8, maxContradictionRatio: 0.5, failOpen: false }`; `false` is the explicit opt-out and object settings merge over those defaults. For each `direction: 'new'` candidate (and reinforcements too, when `onReinforce`), related chunks are retrieved through the SAME `recall()` pipeline — `memoryTypes ['world','experience']`, no opinions/observations, **`decayHalfLifeDays: 0`** (recency decay must not hide an old contradiction — the whole point is reaching evidence outside the current batch), and the candidate's own cited evidence excluded (support must not be re-judged as potential contradiction). Then ONE batched judge LLM call per cycle — not per candidate — classifies which retrieved chunks contradict each candidate; judge-cited ids are intersected against the shown pool so hallucinated ids drop out, and retrieval text is delimited in `untrusted_data` blocks like the other reflect prompts. Formation is blocked when the contradiction ratio `c/(s+c)` exceeds `maxContradictionRatio` — journaled `rejected` with reason `counter_evidence` (gate_results carries ratio/threshold + per-candidate retrieval info), counted in `opinionsRejected`, facts still marked reflected; `maxContradictionRatio: 1` is record-only mode (never blocks). Surviving formations are **born with** `contradicting_chunks` + `last_challenged` set, so item 3's falsifier/decay machinery inherits an honest ledger; `onReinforce` merges contradictions onto the existing opinion (+ `last_challenged`) WITHOUT reversing the reinforcement delta — recording and adjudicating are deliberately separate concerns. **Fail-closed by default**: an unavailable or invalid judge result is journaled as unavailable, the candidate facts remain unreflected for retry, and the cycle is partial. `counterEvidence.failOpen: true` restores availability-first formation. Requires an embedder for retrieval: `Engram.reflect()`/`reflectCatchUp()` thread the instance's own (`ReflectConfig.embedder`); standalone `reflect()` without one throws clearly while the default audit is active, so callers must supply an embedder or opt out explicitly. sqlite-vec is loaded dynamically on reflect's own connection only when the pass is active. `ReflectResult.counterEvidenceChecked` (aggregated in `CatchUpResult`) counts judged candidates. Cost model: +1 generation call and one recall per eligible candidate per cycle. **No new MCP tool** — `engram_reflect` exposes `opinionGates` and `counterEvidence`, while `surface-parity.test.ts` stays pinned at 16. Tests: `tests/counter-evidence.test.ts` (10).

- [x] **Falsifier field + contradiction-driven weakening (issue #38, item 3)**: completes the #38 sequence (gates+journal → counter-evidence → falsifier). A new nullable `opinions.would_change_this` column (guarded `ALTER TABLE` in `engram.ts` for existing files; in `schema.sql` for fresh ones) records the model's own one-sentence answer to "what concrete evidence would change this belief," requested in the reflect prompt for every `direction: 'new'` opinion (optional in the contract — omissions store NULL; clamped 1000 chars like `rationale`). An opinion that predates the field (or whose model omitted it) is **backfilled** by the first reinforcement that states one — never overwritten once set (first-stated-wins, matching node_origin's first-author-wins). The falsifier then feeds both later-cycle surfaces: the reflect prompt's existing-opinions block renders `(would change if: ...)` with an instruction tying it to the `challenge` verdict, and the counter-evidence judge prompt shows `Stated falsifier:` for reinforcement candidates with an instruction that evidence matching the stated condition IS a contradiction. **Weakened decay** is the "principled decay" half, integrated into the EXISTING step-0 decay pass rather than added beside it: still one mechanism (same 0.02/cycle rate, 0.1 floor, ≥7-day throttle) with two eligibility arms — (a) the pre-existing 30-day idle arm, unchanged and unjournaled, and (b) **unanswered contradictions**: an opinion carrying recorded counter-evidence (`contradicting_chunks` non-empty) that has NOT been reinforced since it was last challenged decays without waiting out the idle window, journaled with the `weakened` action reserved since item 1 (`reason: 'unanswered_contradictions'`, contradiction count, decay amount, auto-generated rationale, keyed to the run's reflect_log id). Decay stops the moment a reinforcement answers the challenge (`last_reinforced` moves past `last_challenged`) — recording contradictions (item 2) and adjudicating them (this) stay separate concerns connected through the ledger. `ReflectResult.opinionsWeakened` (aggregated in `CatchUpResult`) counts arm-(b) decays. `introspect`'s `OpinionView` gains `wouldChangeThis` (additive projection — still no LLM call, every field maps to a column). **Library/schema-only, adds ZERO MCP tools** — `surface-parity.test.ts` stays pinned at 16. Tests: `tests/falsifier.test.ts` (10).

- [x] **Procedural reflection / suggestions (issue #39)**: A third insight kind alongside observations/opinions — "this recurring pattern would benefit from being codified as a skill/rule/workflow/config." Three signal classes are scanned from the durable chunk store: **correction** (a chunk was superseded or forgotten — `superseded_by IS NOT NULL OR is_active = FALSE`), **friction** (active `tool_result` chunks), **workflow** (active `experience` chunks that aren't tool output). Three design calls: (1) **own scan cursor** — a `suggest_watermark` `bank_config` key, separate from reflect's `reflected_at` cursor, because correction signal is an *event on an already-reflected chunk* (supersede/forget bumps `updated_at` on an old row that reflect already processed) — reusing reflect's cursor would never see it. The watermark advances only when the model **engaged** (parsed output, including the common parsed-empty answer); a generation or parse failure leaves it untouched (fail-open). Repetition that arrives one signal per cycle isn't lost to per-cycle evaluation instead of a rescan: a below-threshold candidate is journaled `rejected`/`insufficient_evidence`, and a later cycle re-deriving the same candidate (same 0.85 fuzzy-match bar as opinion dedup) merges the prior rejection's evidence into the union — the #38 merge-forward pattern, replicated against the new `suggestion_journal`. (2) **No event-trace table**: supersede/forget already leave queryable state (`superseded_by`, `is_active`), and chunk-state reconstruction sees pre-feature history an event log added today never would. Consequence: correction evidence is inactive *by definition* (a corrected/forgotten chunk), so the shared gate evaluator — `evaluateEvidenceGates` in the new `src/insight-shared.ts`, a pure-move extraction of the gate/similarity/JSON-parse helpers reflect.ts and suggest.ts both now call, zero behavior change — takes a `requireActive` toggle: `true` for opinion formation (byte-identical to pre-#39), `false` for suggestion formation. (3) **No confidence/priority score**: confidence without a revision loop is the smell the grounding layer already named; ranking is `evidence_count DESC, COALESCE(last_reinforced, formed_at) DESC` plus the human lifecycle (`proposed`/`accepted`/`dismissed`/`implemented`). Schema: two new tables — `suggestions` (summary + a summary-embedding BLOB for cosine dedup + evidence + lifecycle status/status_reason + `node_origin` first-author-wins) and `suggestion_journal` (append-only audit, own action vocabulary `proposed`/`reinforced`/`rejected`/`reopened`/`resolved` since a SQLite CHECK can't be widened in place to reuse `belief_journal`'s) — plus a `idx_chunks_correction_events` partial index; table count 11 → 13, all plain `CREATE TABLE`/`INDEX IF NOT EXISTS` (a brand-new table needs no guarded ALTER). Runs as **step 0.5** inside `reflect()`, opt-in via `ReflectConfig.suggestions?: SuggestionConfig`: after the decay pass, BEFORE the `minFactsThreshold` early return (so a store with correction backlog but too few fresh world/experience facts still gets suggestions considered), in its own try/catch (a suggestion-pass failure never disturbs opinion/observation formation below), at most **one extra generation call per cycle** (zero when new signal count is below `minSignalThreshold`, default 5). **Gates default ON** — `{minEvidenceCount: 3, minDistinctDays: 2}` . These match the safe default opinion gates; a low-precision procedural nag is worse than a missed one, so this feature is precision-over-recall from the start. Dedup at formation is cosine similarity over stored summary embeddings (embedder threaded from `ReflectConfig.embedder`, same as counter-evidence); no embedder → the pass still runs on a lexical `beliefSimilarity` fallback with a one-time warning (embedding stored NULL) rather than silently skipping the pass. Matching an existing `proposed`/`accepted`/`implemented` suggestion reinforces it (evidence union, status untouched — recurrence after `implemented` is itself signal); matching a **`dismissed`** one only reopens if the *new* (previously-unseen) evidence alone clears `minEvidenceCount` — dismissal is durable operator memory, and re-nagging about something already declined is exactly the failure mode gates exist to prevent. Surface: **2 new MCP tools, 14 → 16** — `engram_suggestions` (list, filter by status/kind/domain) and `engram_resolve_suggestion` (set status; `'proposed'` is a legal target = manual reopen) — with CLI twins `suggestions`/`resolve-suggestion` (unknown id → exit 2, supersede-style not-found convention). `engram_reflect` gained an optional `suggest: boolean` (maps to `suggestions: {}` all-defaults) and the CLI's `reflect` gained `--suggest`, so an MCP/CLI-only caller can form suggestions without touching `ReflectConfig` directly. Both transports also expose the safe reflection controls: MCP accepts `opinionGates: false | {...}` and `counterEvidence: false | {...}`; CLI provides `--unsafe-opinions`, `--no-counter-evidence`, and the corresponding gate/audit tuning flags. Suggestions **never enter `recall()` or `groundSubagent()`** — structurally, by living in their own table, and pinned by test. `ReflectResult`/`CatchUpResult` gain `suggestionsProposed`/`suggestionsReinforced`/`suggestionsRejected` (aggregated in `CatchUpResult` too). The scan's `datetime()`-wraps-both-sides timestamp comparison repeats the ContextStore TTL lesson — a raw ISO string vs. SQLite `CURRENT_TIMESTAMP` silently never matches otherwise. Library read surface: `Engram.suggestions()` / `resolveSuggestion()` / `suggestionJournal()`. Tests: `tests/suggestions.test.ts` (14). Landed in two slices: core pass + schema + read surface (`daae959`), then MCP + CLI exposure (`feec03f`).

- [x] **Insight embeddings (semantic recall of opinions/observations)**: Reflection's own output was the only part of the store not reachable semantically. `recall()` attached opinions/observations by `LOWER(belief|summary) LIKE '%term%'` over query terms longer than 3 characters, taking the top 5 **by confidence (opinions) or recency (observations)**. That ranks nothing by relevance: ordinary query words ("what", "like", "does", "work") match a large fraction of beliefs, so the attached insights were close to constant across unrelated questions. Measured on a live 305-opinion store, the candidate pool for *"what should I work on next"* was 70 of 305, and *"what music does Tom like"* — a subject never discussed, where the only correct answer is none — returned five confident beliefs about pipelines and work style. Fix: a nullable `embedding BLOB` on `opinions` AND `observations` (same LE-f32 space as `chunks.embedding`; guarded `ALTER TABLE` in `engram.ts` for existing files, in `schema.sql` for fresh ones — `suggestions` already carried one, so this is the established pattern, not a new one), written at formation by `reflect()` and **rewritten on observation refresh** (the summary changes, so a stale vector would rank the row against wording it no longer has; `COALESCE(?, embedding)` keeps an existing vector when a cycle has no embedder rather than blanking it). Selection moved into `selectInsights()` in `recall.ts`, which cosine-ranks embedded rows and gates them with the new `RecallOptions.insightMinScore` (default `0.45`, mirroring `minScore`'s `0.4–0.45` guidance; `0` ranks without filtering). This composes with — and does not undo — the query-scoping rule that removed the global top-N fallback: **both** paths can legitimately return nothing, and an empty list stays the correct answer for a query the store holds no belief about. **Three regimes keep it monotonic, so no deployment regresses**: a bank with nothing embedded takes the lexical path exactly as before; a fully-embedded bank is purely semantic; the mixed state a backfill passes through puts semantic hits first and fills any remaining slots lexically from the un-embedded rows. The query embedding is lazy and memoized, so it is shared with the semantic strategy (one embed call per recall) and costs nothing when a caller disables both that strategy and insights. `bufferToFloat32Array`/`cosineSimilarity` moved from `suggest.ts` to `insight-shared.ts` (pure move — three insight kinds now decode the same BLOB layout). Legacy rows are healed by `backfillInsightEmbeddings()`: bounded (`DEFAULT_INSIGHT_BACKFILL_LIMIT` = 200/call), resumable, opinions-before-observations, never rewrites an existing vector, and a row whose embed call throws stays NULL for a later pass. `reflect()` runs one bounded backfill per cycle whenever an embedder is present, placed **before** the `minFactsThreshold` early return — a store in the intended steady state (nothing left worth reflecting on) is exactly the store that would otherwise never backfill. Also exposed as `Engram.backfillInsightEmbeddings()` to drain a large legacy backlog in one pass. `insightMinScore` is exposed on `engram_recall` and the CLI (`--insight-min-score`) the same way `minScore`/`decayHalfLifeDays` already were, clamped at that boundary. **No new tools — schema plus one recall param**, so `surface-parity.test.ts` stays pinned at 16. Tests: `tests/insight-embeddings.test.ts` (22). **Known open:** on a real 305-opinion bank, nomic-embed puts 239/258 beliefs above 0.45 for a typical question (range 0.40–0.68, p50 0.53), so the floor filters little and vague queries still return vague beliefs. The top scorers are low-information generalities ("Tom's work style involves strategic planning, immersive engagement, and co-creative collaboration") that are near-equidistant from every query — belief *specificity* at formation, not ranking, is the remaining lever.
