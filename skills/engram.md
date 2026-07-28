# Engram Memory Skill

## Description

Engram is a local-first semantic memory system for AI agents. Store, search, and manage persistent memory across sessions using four retrieval strategies (semantic vector, keyword FTS, knowledge graph, temporal).

## MCP Server

Engram connects as an MCP server via mcporter. All tool parameters use **camelCase**.

## Tool Reference

### Store a fact — `engram_retain`

```bash
npx mcporter call engram.engram_retain text="Your fact here" memoryType=world sourceType=user_stated trustScore=0.9
```

| Parameter | Required | Values |
|-----------|----------|--------|
| `text` | yes | The memory content to store |
| `memoryType` | no | `world` (facts), `experience` (agent actions), `observation` (synthesized), `opinion` (beliefs) |
| `sourceType` | no | `user_stated`, `inferred`, `external_doc`, `tool_result`, `agent_generated` |
| `trustScore` | no | 0.0–1.0. Recommended: user_stated=0.85, agent_generated=0.6, inferred=0.5 |
| `source` | no | Identifier (e.g. `conversation:session-123`, `file:config.yaml`) |
| `context` | no | Freeform tag (e.g. `infrastructure`, `project:valor`) |
| `eventTime` | no | ISO 8601 timestamp for when the event occurred |
| `supersedes` | no | Stale `chk-…` id to atomically deactivate and link to this fact. Conflicting facts stay separate unless supplied explicitly. |

### Search memory — `engram_recall`

```bash
npx mcporter call engram.engram_recall query="Your search query" topK=5
```

| Parameter | Required | Values |
|-----------|----------|--------|
| `query` | yes | Natural language query. Temporal expressions are auto-parsed — "last week", "yesterday", "March 15th", "past 30 days", "Q1 2026" all activate the temporal strategy automatically. |
| `topK` | no | Max results (default: 10) |
| `strategies` | no | Array: `semantic`, `keyword`, `graph`, `temporal` |
| `memoryTypes` | no | Filter: `world`, `experience`, `observation`, `opinion` |
| `minTrust` | no | Minimum trust score 0.0–1.0 |
| `after` | no | ISO date — only facts after this date. Overrides auto-parsed dates. |
| `before` | no | ISO date — only facts before this date. Overrides auto-parsed dates. |
| `minScore` | no | Drop results whose final weighted score (post trust/decay/strategy-boost) falls below this 0.0–1.0 threshold. Default: no filtering. |
| `explainScores` | no | When `true`, each result gains a `strategyScores` breakdown (per-strategy rank/RRF contribution + weighting factors). Default: `false`, payload stays lean. |
| `decayHalfLifeDays` | no | Recency decay half-life in days (default: 180) — a chunk's score is multiplied by `2^(-ageDays/decayHalfLifeDays)`. Pass `0` to disable decay entirely (long-continuity recall over older facts). |

Returns: `results[]` (ranked chunks), `opinions[]` (beliefs with confidence), `observations[]` (synthesized knowledge).

`opinions` and `observations` are query-scoped. The query is case-folded and
punctuation-stripped before lexical matching; if no normalized term matches a
synthesized entry, those arrays are empty rather than falling back globally.

**`results[0]` is the best match in the highest-present source tier, not the best match overall** — re-sort by `score` locally where pure relevance is what you need. **`user_stated` memories structurally outrank `tool_result`/`external_doc` content regardless of score** — this floor holds no matter what `decayHalfLifeDays` or trust score is passed.

**Temporal query examples:**

```bash
npx mcporter call engram.engram_recall query="what happened last week"
npx mcporter call engram.engram_recall query="decisions yesterday"
npx mcporter call engram.engram_recall query="March 2026 deployments"
npx mcporter call engram.engram_recall query="progress past 30 days"
```

No need to pass `after`/`before` manually — the temporal strategy activates from the query text.

### Working memory session — `engram_session`

```bash
npx mcporter call engram.engram_session message="Current user message"
```

Returns session state + related long-term context. Auto-matches to existing sessions by topic similarity or creates new ones.

`action` selects the operation — `resume` (default, the call above — omit
`action` entirely for this, unchanged, backward-compatible behavior),
`update`, or `snapshot`:

| Parameter | Required | Values |
|-----------|----------|--------|
| `action` | no | `resume` (default), `update`, `snapshot` |
| `message` | yes for `resume` | The incoming user message |
| `maxActive` | no | `resume` only. Max active sessions before oldest is snapshotted (default: 5) |
| `threshold` | no | `resume` only. Cosine similarity threshold for matching (default: 0.55) |
| `sessionId` | yes for `update`/`snapshot` | The `wm-…` id returned by a prior resume |
| `progress` | no | `update` only. Free-form progress note merged into the session |
| `extensions` | no | `update` only. Object of agent-defined fields merged into the session |

```bash
# update: merge progress into an existing session
npx mcporter call engram.engram_session action=update sessionId=wm-abc123 \
  progress="Drafted the rollback plan"

# snapshot: collapse a session to long-term memory and end it
npx mcporter call engram.engram_session action=snapshot sessionId=wm-abc123
```

`update` returns the full updated session state. `snapshot` returns
`{sessionId, chunkId, ...}` — the retain result for the episodic chunk the
session was collapsed into. Both error (isError, "not found") if `sessionId`
doesn't resolve to an active session.

### Build knowledge graph — `engram_process_extractions`

```bash
npx mcporter call engram.engram_process_extractions batchSize=10
```

Requires Ollama. Processes queued chunks to extract entities and relationships.

### Run reflection — `engram_reflect`

```bash
npx mcporter call engram.engram_reflect
npx mcporter call engram.engram_reflect suggest=true
```

Requires Ollama. Synthesizes observations and forms/updates opinions from accumulated facts. Pass `suggest=true` to also run the procedural-suggestion pass this cycle (see `engram_suggestions` below) — omit it (the default) to skip that pass entirely, unchanged from before.

New opinions are safe by default: they require three verified chunks across two
distinct days, then an active counter-evidence audit (`topK: 8`, contradiction
ratio `0.5`). The audit is one extra batched generation call and fails closed:
an unavailable or invalid judgment is journaled and retried with the facts left
unreflected. `opinionGates` and `counterEvidence` accept either `false` for an
explicit opt-out or an object that overrides individual safe defaults; the
latter also supports `onReinforce` and `failOpen`.

### Forget a memory — `engram_forget`

```bash
npx mcporter call engram.engram_forget chunkId=chk-xxx
```

Soft-deletes a chunk. Excluded from recall but retained for audit.

### Replace outdated fact — `engram_supersede`

```bash
npx mcporter call engram.engram_supersede oldChunkId=chk-xxx newText="Updated fact"
```

Soft-deletes the old chunk, stores the new one, and links them via `superseded_by`.

### Inspect extraction queue — `engram_queue_stats`

```bash
npx mcporter call engram.engram_queue_stats
```

Returns counts by status (`pending`, `processing`, `completed`, `failed`), the age of the oldest pending item, and a `failed_reasons` breakdown (distinct error messages with counts, most common first). Use to diagnose why the knowledge graph is not growing or to decide when to call `engram_process_extractions`. Takes no parameters.

### Re-queue failed extractions — `engram_requeue_failed`

```bash
npx mcporter call engram.engram_requeue_failed
npx mcporter call engram.engram_requeue_failed errorLike="fetch failed"
```

Failed is a terminal state — after 3 attempts an item never retries on its own. Once the underlying cause is fixed (LLM host back online, missing model pulled), call this to reset failed items to pending with a fresh attempt counter. Optional `errorLike` substring targets one failure class from the `failed_reasons` breakdown. Returns `{"requeued": <count>}`. Items whose chunk was forgotten are skipped.

### Introspect held state — `engram_introspect`

```bash
npx mcporter call engram.engram_introspect subject="kubernetes"
npx mcporter call engram.engram_introspect                       # top held state overall
```

Read what the agent currently *believes* about a subject, not what chunks mention it. Returns `{"subject": ..., "opinions": [...], "observations": [...]}`. Each opinion carries `belief`, `confidence`, `domain`, `supportCount`/`challengeCount`, the `supportingChunks`/`contradictingChunks` provenance ids, `evidenceCount`, and lifecycle timestamps (`formedAt`, `lastReinforced`, `lastChallenged`). Unlike `engram_recall`, this is a **direct lookup with no confidence floor** — a weakly-held or freshly-challenged belief (which recall hides at `confidence < 0.5`) stays visible, which is exactly what you want when reasoning about your own beliefs. Optional `minConfidence` (default 0), `limit` (default 20), `includeOpinions`/`includeObservations`. It **reports** held state; it does not judge whether a candidate statement agrees with or contradicts a belief (that consistency check is a separate, deferred primitive). Read-only, no LLM call.

### Embed text — `engram_embed`

```bash
npx mcporter call engram.engram_embed text="deploy pipeline"
npx mcporter call engram.engram_embed text="stored document text" mode=document
```

Returns `{"embedding": [...], "dimensions": <n>}` — a vector in the bank's stored embedding space. `mode=query` (the default) applies the search prefix for asymmetric models like nomic-embed-text; `mode=document` matches how `engram_retain` embeds stored text. Most agents never need this directly — it exists for consumers that do their own vector math (it is the bridge `engram-aql` uses for AQL `LIKE`/`PATTERN` vector search). It does not store anything.

### List procedural suggestions — `engram_suggestions`

```bash
npx mcporter call engram.engram_suggestions
npx mcporter call engram.engram_suggestions status=proposed kind=rule
```

Reflection can spot a **third** kind of insight beyond observations/opinions:
recurring patterns worth codifying — repeated corrections, repeated tool
friction, or a repeated multi-step workflow. Call this when you want to check
whether the agent has noticed something worth turning into a skill, rule,
workflow, or config change (e.g. periodically, or when a user asks "have you
noticed anything I keep correcting?"). Suggestions **do not appear in
`engram_recall`** — this is the only way to see them.

| Parameter | Required | Values |
|-----------|----------|--------|
| `status` | no | `proposed`, `accepted`, `dismissed`, `implemented` |
| `kind` | no | `skill`, `rule`, `workflow`, `config` |
| `domain` | no | Filter by domain tag |
| `limit` | no | Max rows (default 20, clamped to [1, 1000]) |

Returns a list sorted by evidence strength then recency, each with `id`,
`kind`, `summary`, `rationale`, `supportingChunks`, `evidenceCount`, `status`,
`formedAt`, `lastReinforced`. Suggestions only form when `engram_reflect` is
called with `suggest=true` — a plain reflect cycle skips the pass entirely.
Formation gates default **on** (3+ evidence items across 2+ distinct days) — this is a
precision-over-recall feature, so an empty list after a reflect cycle is the
normal/common outcome, not a failure.

### Resolve a suggestion — `engram_resolve_suggestion`

```bash
npx mcporter call engram.engram_resolve_suggestion suggestionId=sug-xxx status=accepted
npx mcporter call engram.engram_resolve_suggestion suggestionId=sug-xxx status=dismissed reason="too narrow to codify"
```

| Parameter | Required | Values |
|-----------|----------|--------|
| `suggestionId` | yes | The `sug-…` id from `engram_suggestions` |
| `status` | yes | `proposed`, `accepted`, `dismissed`, `implemented` |
| `reason` | no | Freeform note recorded on the suggestion and its audit journal |

Set `accepted` once you've decided to act on it, `implemented` once you have
(e.g. you wrote the skill/rule it proposed), or `dismissed` if it's not worth
codifying — dismissal is remembered, so the same pattern won't be re-proposed
unless materially new evidence accumulates. Passing `proposed` manually
reopens a resolved suggestion. Returns `{suggestionId, status, resolved}` —
`resolved=false` (not an error) when the id doesn't exist.

## Task-scoped context (ContextStore)

A **fifth, short-lived scope** alongside the four durable memory types — for
cheap agent-to-subagent handoff, NOT for long-term memory. A lead agent
commits a structured decision, a subagent queries for only what's relevant
beneath it. Artifacts expire on their own (default 4 hours) unless promoted.
Use this instead of `engram_retain`/`engram_recall` when the content is
task-scoped scratch context that shouldn't pollute durable memory or survive
past the task.

### Commit a decision — `engram_context_commit`

```bash
npx mcporter call engram.engram_context_commit decision="Use blue/green deployment for the release" \
  rationale="Zero-downtime cutover with an easy rollback" domain="deployment-planning"
```

| Parameter | Required | Values |
|-----------|----------|--------|
| `decision` | yes | The decision/artifact text |
| `rationale` | no | Why this decision was made |
| `scoredOptions` | no | Array of `{option, score}` — options considered |
| `confidence` | no | 0.0–1.0 |
| `refsToSource` | no | Array of chunk ids or other identifiers this draws on |
| `domain` | no | Freeform tag |
| `agentId` | no | Originating agent Tier/callsign, for provenance |
| `parentRefId` | no | `ContextRef.id` of a parent scope to chain under (omit for a root commit) |
| `ttlMs` | no | Milliseconds until expiry (default: 4 hours) |

Returns a lightweight `{"id": "ctx-…", "scope": "task"}` — pass `id` to a
subagent as `parentRefId` when it commits its own findings, and as `refId`
when it queries.

**Important:** a ref is queryable as a **parent** — `engram_context_query`
returns the children committed *under* a ref (`parentRefId` pointing at it),
not the artifact at that ref itself. To hand a subagent a queryable scope,
commit a root artifact first, then have the subagent (or yourself) commit
follow-up artifacts with `parentRefId` set to the root's id.

### Query committed artifacts — `engram_context_query`

```bash
npx mcporter call engram.engram_context_query refId=ctx-abc123 query="deployment strategy"
```

| Parameter | Required | Values |
|-----------|----------|--------|
| `refId` | yes | `ContextRef.id` to query beneath (a prior commit's id) |
| `query` | yes | Relevance query used to rank artifacts committed under `refId` |
| `maxChars` | no | Character budget for returned artifacts (default: 4000) |

Ranked via the same RRF-fusion pipeline as durable `engram_recall`. Returns
`{"artifacts": [...], "truncated": bool, "totalCandidates": n}` — each
artifact carries its `ref`, the full `artifact` (decision/rationale/etc.),
`parentRef`, `createdAt`, `expiresAt`.

### Promote to durable memory — `engram_context_promote`

```bash
npx mcporter call engram.engram_context_promote refId=ctx-abc123
```

Moves the artifact into durable memory (survives past its TTL, becomes
eligible for `engram_reflect`). Returns `{"promoted": true}`, or
`{"promoted": false}` (not an error) if `refId` doesn't resolve to an active
task-scoped artifact. Does not itself run `engram_reflect`.

There is deliberately no `engram_context_expire` tool — TTL expiry is lazy
(enforced at query time), so an unwanted artifact simply ages out on its own.

## Usage Patterns

### Before answering a user question

```
1. engram_recall query="<relevant query>" topK=5
2. Read results, opinions, observations
3. Incorporate into response
```

### After learning something new

```
1. Evaluate if worth storing (is it a fact, decision, or preference?)
2. engram_retain text="<the fact>" memoryType=world sourceType=user_stated trustScore=0.85
```

### Starting a conversation turn

```
1. engram_session message="<user's message>"
2. Use returned session.goal and relatedContext for prompt context
3. After responding, update session progress if applicable
```

## Common Mistakes

- Parameters are **camelCase** not snake_case: `memoryType` not `memory_type`, `trustScore` not `trust_score`
- Use `key=value` syntax with mcporter, not JSON bodies
- Tool names are prefixed with the server name: `engram.engram_retain` not just `engram_retain`
- Don't retain trivial messages ("ok", "thanks", "got it") — they add noise
- Don't forget to run `engram_process_extractions` periodically to build the knowledge graph

## Configuration

The Engram MCP server is configured in your mcporter config file (typically `config/mcporter.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": [
        "/path/to/engram/dist/mcp-server.js",
        "/path/to/agent.engram"
      ],
      "transport": "stdio"
    }
  }
}
```

Add `--use-ollama-embeddings --ollama-url <url>` to args for Ollama-based embeddings. Default uses local Transformers.js (no external dependency).
