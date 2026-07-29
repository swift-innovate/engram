---
name: cli-memory
description: Persistent agent memory via the `engram` CLI. Recall before answering anything that depends on prior context (who/what/preferences/decisions/history); retain after a fact, decision, or preference is established; supersede on correction; run session once per incoming message. Every command takes `--json` for a machine-readable contract and signals not-found with exit code 2.
---

# Engram CLI Memory

You have a persistent memory file (an `.engram` SQLite database). The `engram`
command is your only interface to it. This skill is the contract for **deciding
whether and how to call it** — not a man page.

## When to reach for memory (read this first)

- **Before answering anything context-dependent → `recall`.** If the user
  references a person, project, preference, past decision, or "what we did", you
  do not know the answer from the prompt alone. Recall first, then answer. A
  recall that returns nothing is cheap; answering from a stale guess is not.
- **To check what you already believe about something → `introspect`.** When the
  question is "what's my current opinion on X and how strongly do I hold it",
  introspect returns beliefs with confidence + supporting/challenging evidence —
  including weakly-held ones `recall` hides. It reads held state; it does not
  fetch raw chunks (that's `recall`) or judge a statement's consistency.
- **After a fact, decision, or preference is established → `retain`.** The user
  states a preference ("I use Pulumi now"), you make a design decision, a
  constraint is confirmed. Store it so the next session knows. Do **not** retain
  small talk, acknowledgements, or things already in the repo/git history.
- **On a correction → `supersede`.** The user contradicts something you recalled
  ("no, it's Kubernetes not Docker"). Supersede the old chunk with the new text
  so the wrong fact stops surfacing.
- **Once per incoming user message → `session`.** Call it at the top of a turn to
  resume or open the working-memory session and pull related long-term context
  in one shot. Use the returned `relatedContext` to ground your reply.
- `reflect` / `process-extractions` are background maintenance (need an LLM).
  Run occasionally, not per turn. `queue-stats` tells you if the graph is behind;
  `requeue-failed` re-drives items stranded by an outage (see its section below).
  `reflect --suggest` also proposes procedural suggestions — recurring
  patterns worth codifying — readable via `suggestions` (see its section below).
  Keep maintenance out of the latency-critical turn path, prevent overlapping
  runs in the host scheduler, and surface failures for retry.

## Setup

Point the CLI at a database via `--db <path>` on each call, or set it once:

```bash
export ENGRAM_DB=./agent.engram
```

Keep that path stable for one agent identity. For long-lived assistant memory,
pass `--decay-half-life-days 0` to recall so old but relevant memories do not
silently decay away. Backup is host/operator work: use the library's SQLite
online-backup API, not a raw copy of an open `.engram` file.

`--db` wins over `ENGRAM_DB`. If neither is set the command exits 1.

## Exit codes (branch on these)

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `2`  | Not found — `forget`/`supersede` target chunk, `session --action` id, `context-promote` ref, or `resolve-suggestion` id does not exist |
| `1`  | Error — bad/missing argument, no DB path, or operation failure |

`--json` puts the raw method return on **stdout** and nothing else; all
diagnostics go to **stderr**. Parse stdout, branch on the exit code.

## Commands

### `recall <query>` — search memory

```bash
engram recall "Terraform IaC provider" --json
echo "long pasted context" | engram recall --json    # query from stdin if omitted
```

Use keywords and proper nouns, not full questions ("Tom role background", not
"Who is Tom?"). Temporal phrases auto-activate date filtering ("last week",
"March 2026"). Options: `--top-k <n>`, `--strategies semantic,keyword,graph,temporal`,
`--memory-types world,experience,observation,opinion`, `--min-trust <0..1>`,
`--after <iso>`, `--before <iso>`, `--no-opinions`, `--no-observations`,
`--min-score <0..1>` (drop results below this weighted score, post
trust/decay/strategy-boost — default: no filtering), `--explain-scores`
(add a `strategyScores` breakdown per result — per-strategy rank/RRF
contribution + weighting factors — default: off, keeps the payload lean),
`--decay-half-life-days <n>` (recency decay half-life in days, default 180 —
pass `0` to disable decay entirely for long-continuity recall over older facts).

Synthesized opinions and observations are query-scoped: after case-folding and
punctuation removal, only entries with a matching normalized query term are
returned. No match means no synthesized context, rather than a global fallback.

**`results[0]` is the best match in the highest-present source tier, not the
best match overall** — re-sort by `score` locally where pure relevance is
what you need. **`user_stated` memories structurally outrank `tool_result`/
`external_doc` content regardless of score** — this floor cannot be overridden
by trust score or `--decay-half-life-days`.

`--json` shape:
```json
{
  "results": [
    { "id": "chk-…", "text": "…", "memoryType": "world", "source": "…",
      "trustScore": 0.9, "sourceType": "user_stated", "eventTime": null,
      "score": 0.83, "strategies": ["semantic", "keyword"] }
  ],
  "opinions": [{ "belief": "…", "confidence": 0.7, "domain": "…" }],
  "observations": [{ "summary": "…", "domain": "…", "topic": "…" }],
  "totalCandidates": 12,
  "strategiesUsed": ["semantic", "keyword", "graph"]
}
```

With `--explain-scores`, each result also carries a `strategyScores` field:
```json
{
  "perStrategy": [{ "strategy": "keyword", "rank": 1, "rrfScore": 0.0164 }],
  "rawFusedScore": 0.0164,
  "weighting": { "trust": 1.14, "strategyBoost": 1.0, "decay": 0.98,
                 "sourceBoost": 1.0, "contextBoost": 1.0, "memoryType": 1.15 }
}
```

### `retain <text>` — store a fact

```bash
engram retain "Tom prefers Pulumi over Terraform" \
  --memory-type world --source-type user_stated --trust-score 0.9 --json
echo "fact text" | engram retain --json                # text from stdin if omitted
```

Options: `--memory-type <world|experience|observation|opinion>`,
`--source-type <user_stated|inferred|external_doc|tool_result|agent_generated>`,
`--trust-score <0..1>`, `--source <id>`, `--context <tag>`,
`--event-time <iso>`, `--temporal-label <text>`, `--supersedes <chunkId>`.
`--supersedes` atomically deactivates and links a known stale chunk; conflicting
facts otherwise remain separate until explicitly superseded. Trust guidance:
user_stated ≈ 0.85–0.9, agent_generated ≈ 0.6, inferred ≈ 0.5.

`--json` shape: `{ "chunkId": "chk-…", "queued": true, "deduplicated"?: false, "tier1"?: { "entitiesLinked": 0, "relationsCreated": 0 } }`

### `supersede <oldChunkId> <newText>` — correct a fact

```bash
engram supersede chk-abc123 "Tom switched to Kubernetes" --json
echo "new text" | engram supersede chk-abc123 --json    # newText from stdin if omitted
```

Takes the same write options as `retain`. Soft-deletes the old chunk and links
it to the new one. **Exit 2** if `oldChunkId` doesn't exist (and no new chunk is
created in that case). `--json` shape is the same as `retain`.

### `forget <chunkId>` — soft-delete

```bash
engram forget chk-abc123 --json
```

`--json` shape: `{ "forgotten": true }`. Exit `0` if it existed, **`2`** if not
(the JSON still reports `{ "forgotten": false }`).

### `session <message>` — working-memory session (once per turn)

```bash
engram session "Help me plan the deployment" --json
echo "user message" | engram session --json             # message from stdin if omitted
```

Options: `--max-active <n>` (default 5), `--threshold <0..1>` (default 0.55).
`--json` shape:
```json
{
  "session": { "id": "wm-…", "goal": "…", "updated_at": "…" },
  "relatedContext": "formatted long-term context for your prompt",
  "confidence": 0.82,
  "diagnostics": { "sessionId": "wm-…", "reason": "match", "candidatesEvaluated": 3 }
}
```
`reason` is `"new"` for a fresh session, `"match"` when resuming one.

`session` also takes `--action update|snapshot` (default is the resume
behavior above — unchanged when `--action` is omitted). Both require
`--session-id <id>` (the `wm-…` id a resume returned):

```bash
# merge progress (and/or --extensions '<json>') into an existing session
engram session --action update --session-id wm-abc123 \
  --progress "Drafted the rollback plan" --json
# --json shape: the full updated session state, e.g.
# { "id": "wm-abc123", "goal": "…", "progress": "Drafted the rollback plan", "updated_at": "…" }

# collapse a session to long-term memory and end it
engram session --action snapshot --session-id wm-abc123 --json
# --json shape: { "sessionId": "wm-abc123", "chunkId": "chk-…", "queued": true, ... }
```

Exit **2** if `--session-id` doesn't resolve to an active session.

### `queue-stats` — extraction queue health

```bash
engram queue-stats --json
```

`--json` shape: `{ "pending": 3, "processing": 0, "completed": 12, "failed": 2, "oldest_pending": "2026-06-05T12:00:00Z", "failed_reasons": [{ "error": "fetch failed", "count": 2 }] }`.
A growing `pending` means the knowledge graph is behind — run `process-extractions`.
A non-zero `failed` with a transient-looking reason (host down, model missing) means
those items need an explicit re-drive once the cause is fixed — run `requeue-failed`.

### `requeue-failed` — re-drive failed extractions

```bash
engram requeue-failed --json
engram requeue-failed --error-like "fetch failed" --json
```

Failed is terminal (3 attempts exhausted) — items never retry on their own. This
resets them to pending with a fresh attempt counter. `--error-like <substring>`
targets one failure class from the `failed_reasons` breakdown.
`--json` shape: `{ "requeued": 11 }`.

### `introspect [subject]` — read held state (opinions + observations)

```bash
engram introspect "kubernetes" --json
engram introspect --json                       # top held state overall
engram introspect "rust" --min-confidence 0.5 --limit 10 --json
```

What the agent currently *believes* about a subject, not what chunks mention it.
A direct lookup with **no confidence floor** — weakly-held or freshly-challenged
beliefs (which `recall` hides at `confidence < 0.5`) stay visible. Subject read
from stdin if omitted as a positional; omit entirely for top held state. Options:
`--min-confidence <0..1>` (default 0), `--limit <n>` (default 20),
`--no-opinions`, `--no-observations`. Read-only, no LLM call. Reports held state;
it does not judge whether a statement is consistent with a belief (deferred).
`--json` shape: `{ "subject": "kubernetes", "opinions": [{ "id", "belief",
"confidence", "domain", "supportCount", "challengeCount", "evidenceCount",
"supportingChunks", "contradictingChunks", "relatedEntities", "formedAt",
"lastReinforced", "lastChallenged", "updatedAt" }], "observations": [{ "id",
"summary", "domain", "topic", "sourceChunks", "sourceEntities", "synthesizedAt",
"lastRefreshed", "refreshCount" }] }`.

### `reflect` — synthesize observations/opinions (needs an LLM)

```bash
engram reflect --json
engram reflect --suggest --json    # also run the procedural-suggestion pass
```

`--json` shape: `{ "logId": "…", "factsProcessed": 8, "observationsCreated": 2, "observationsUpdated": 1, "opinionsFormed": 1, "opinionsReinforced": 2, "opinionsChallenged": 0, "suggestionsProposed": 0, "suggestionsReinforced": 0, "suggestionsRejected": 0, "status": "completed", "durationMs": 1430 }`.

`--suggest` also scans for recurring corrections/tool-friction/workflow
patterns worth codifying as a skill/rule/workflow/config — off by default
(the counters above are always present but stay `0` without the flag).
Formation gates default **on** (3+ evidence items across 2+ distinct days) so
a zero-suggestion cycle after `--suggest` is the normal outcome, not a
failure. See `suggestions` below to read what it proposed.

Opinion formation is safe by default: three verified chunks across two distinct
days plus active counter-evidence retrieval/judging (`topK: 8`, contradiction
ratio `0.5`). That audit adds one batched generation call and fails closed, so
an unavailable judgment leaves its facts unreflected for retry. Tune with
`--opinion-min-evidence`, `--opinion-min-distinct-days`,
`--opinion-min-distinct-sources`, `--counter-evidence-top-k`,
`--counter-evidence-max-contradiction-ratio`, and
`--counter-evidence-on-reinforce`. Use `--unsafe-opinions` or
`--no-counter-evidence` only for explicit opt-out; `--counter-evidence-fail-open`
restores availability-first handling for an unavailable audit.

### `process-extractions` — build the knowledge graph (needs an LLM)

```bash
engram process-extractions --batch-size 10 --json
```

`--json` shape: `{ "processed": 7, "failed": 0 }`.

### `embed <text>` — raw vector in the stored embedding space

```bash
engram embed "deploy pipeline" --json
echo "stored document text" | engram embed --mode document --json
```

`--json` shape: `{ "embedding": [0.01, …], "dimensions": 768 }`. `--mode query`
(default) applies the search prefix for asymmetric models; `--mode document`
matches how `retain` embeds stored text. Stores nothing. You will almost never
need this in a turn loop — it exists for consumers doing their own vector math
(it is the bridge surface `engram-aql` uses for AQL vector search).

### `context-commit`, `context-query`, `context-promote` — task-scoped handoff

A **fifth, short-lived scope**, separate from durable memory (`retain`/`recall`)
and separate from `session`. Use it for cheap agent-to-subagent handoff: a lead
commits a structured decision, a subagent queries for only what's relevant
beneath it. Artifacts expire on their own (default 4 hours) unless promoted —
there is no `context-expire` command, expiry is enforced lazily at query time.

```bash
# commit a root artifact (JSON payload as arg or stdin); required field: decision
engram context-commit '{"decision":"plan the release"}' --json
# --json shape: { "id": "ctx-…", "scope": "task" }

# commit a follow-up artifact chained under the root (--parent-ref-id, or
# "parentRefId" inside the JSON payload)
echo '{"decision":"use blue/green deployment","rationale":"zero-downtime cutover"}' \
  | engram context-commit --parent-ref-id ctx-root123 --json
```

Payload fields (all but `decision` optional): `decision`, `rationale`,
`scoredOptions` (`[{option,score}]`), `confidence` (0..1), `refsToSource`
(`string[]`), `domain`, `agentId`, `parentRefId`, `ttlMs`. `--parent-ref-id`/
`--ttl-ms` flags override the same fields in the JSON payload if both are given.

**Important:** a ref is queryable as a **parent** — `context-query` returns
the children committed *under* `refId`, not the artifact at `refId` itself.
Commit a root, then commit follow-ups with `--parent-ref-id`/`parentRefId`
pointing at the root, then query the root's id.

```bash
engram context-query ctx-root123 "deployment strategy" --json
echo "deployment strategy" | engram context-query ctx-root123 --json  # query from stdin if omitted
```

Options: `--max-chars <n>` (default 4000). `--json` shape:
```json
{
  "artifacts": [
    { "ref": { "id": "ctx-…", "scope": "task" },
      "artifact": { "decision": "…", "rationale": "…" },
      "parentRef": { "id": "ctx-root123", "scope": "task" },
      "createdAt": "…", "expiresAt": "…" }
  ],
  "truncated": false,
  "totalCandidates": 2
}
```

```bash
engram context-promote ctx-abc123 --json
```

Moves the artifact into durable memory (survives past its TTL, becomes
eligible for `reflect`). `--json` shape: `{ "promoted": true }`, or
**exit 2** with `{ "promoted": false }` (not an error) if `ctx-abc123` doesn't
resolve to an active task-scoped artifact.

### `suggestions`, `resolve-suggestion` — procedural suggestions from reflection

A **third** insight kind reflection can produce, separate from
opinions/observations: a recurring pattern (repeated corrections, repeated
tool friction, a repeated multi-step workflow) worth codifying as a skill,
rule, workflow, or config change. Only forms when `reflect --suggest` is
used — see the `reflect` section above. **Suggestions never appear in
`recall`** — `suggestions` is the only way to read them.

```bash
engram suggestions --json
engram suggestions --status proposed --kind rule --json
```

Options: `--status <proposed|accepted|dismissed|implemented>`,
`--kind <skill|rule|workflow|config>`, `--domain <tag>`, `--limit <n>`
(default 20). `--json` shape:
```json
{
  "suggestions": [
    { "id": "sug-…", "kind": "rule", "summary": "…", "rationale": "…",
      "supportingChunks": ["chk-…"], "evidenceCount": 3, "domain": "…",
      "status": "proposed", "statusReason": null, "formedAt": "…",
      "lastReinforced": null, "updatedAt": "…" }
  ]
}
```

```bash
engram resolve-suggestion sug-abc123 accepted --json
engram resolve-suggestion sug-abc123 dismissed --reason "too narrow" --json
```

Sets the lifecycle status: `accepted` once you've decided to act on it,
`implemented` once you have, `dismissed` if it's not worth codifying —
dismissal is remembered, so the same pattern won't be re-proposed unless
materially new evidence accumulates. Passing `proposed` manually reopens a
resolved suggestion. `--reason <text>` is optional, recorded on the
suggestion and its audit journal. **Exit 2** if `suggestionId` doesn't exist.
`--json` shape: `{ "suggestionId": "sug-abc123", "status": "accepted", "resolved": true }`.

## Turn loop (typical)

```bash
# 1. open/resume the session and pull context
engram session "$USER_MESSAGE" --json

# 2. recall anything the message depends on before answering
engram recall "relevant keywords" --top-k 5 --json

# 3. … answer the user …

# 4. retain durable facts/decisions with provenance; supersede on correction
engram retain "decision just made" \
  --memory-type world --source-type user_stated --trust-score 0.9 --json
```
