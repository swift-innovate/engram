# Engram Agent-Readiness Gaps — Claude Code Spec

**Repo:** `G:\Projects\SIT\engram`  
**Context:** Engram is being integrated into Gage (`C:\Users\tom-s\valor\agents\gage`) as its first real operational consumer. This spec addresses gaps discovered when evaluating Engram through the lens of an agent that runs all day processing conversations, managing projects, and building knowledge over time.

**Principle:** Every change ships with tests. Engram already has a solid test suite (52 tests) using vitest with mock embedders and mock Ollama. All new features follow the same pattern.

---

## Gap 1: Retain Gate — Lightweight Conversation Screening

**Problem:** An agent processes dozens of turns per session. "What time is it?" and "hey" don't deserve chunks. Without a filter, the engram fills with noise that degrades recall quality.

**Solution:** Add a `shouldRetain()` heuristic function that screens text before it hits `retain()`. No LLM call — pure pattern matching and length heuristics. This is a suggestion layer, not enforcement — the caller can override.

### Add to `src/retain.ts`:

```typescript
/**
 * Lightweight heuristic to determine if text is worth retaining.
 * No LLM call — pure pattern matching. Returns a score 0.0-1.0
 * where higher = more worth retaining.
 *
 * Intended use: agent's report phase calls this before retain().
 * Score < threshold → skip. Threshold is caller's choice (recommend 0.3).
 *
 * Factors:
 * - Length (very short = likely not substantive)
 * - Contains named entities, technical terms, decisions, preferences
 * - Is a question vs. a statement (statements more retainable)
 * - Contains temporal markers (dates, deadlines)
 * - Is purely social/phatic ("hey", "thanks", "ok")
 */
export function shouldRetain(text: string): { score: number; reason: string } {
  // Implementation: score starts at 0.5, adjust up/down based on signals
  // Return { score, reason } so callers can log the decision
}
```

**Scoring signals (adjust weight as needed):**
- Text length < 20 chars → -0.3 ("ok", "thanks", "hey")
- Text length > 200 chars → +0.1 (substantive content)
- Contains decision language ("decided", "chose", "will use", "switched to", "prefer") → +0.2
- Contains technical terms (heuristic: words with dots, slashes, camelCase, numbers mixed with letters) → +0.1
- Contains temporal markers ("yesterday", "next week", dates, "deadline") → +0.1
- Purely interrogative (starts with question word, ends with ?) with no embedded facts → -0.2
- Social/phatic patterns ("hey", "thanks", "ok", "sure", "got it", "sounds good") → -0.4
- Contains proper nouns (capitalized words not at sentence start) → +0.1

**Tests:** Add `tests/retain-gate.test.ts` with cases for each signal category. Verify scoring is deterministic.

---

## Gap 2: Prompt Formatting Helper for Recall Results

**Problem:** `recall()` returns `{ results, opinions, observations }` but every consumer must write their own formatting to inject this into a system prompt. Getting this wrong wastes tokens or loses signal.

### Add to `src/recall.ts`:

```typescript
/**
 * Format a RecallResponse into a string suitable for system prompt injection.
 * Handles token budgeting, prioritization, and clean formatting.
 *
 * @param response - The RecallResponse from recall()
 * @param options - Formatting options
 * @returns Formatted string ready for system prompt injection
 */
export function formatForPrompt(
  response: RecallResponse,
  options?: {
    /** Max characters for the entire block (default: 2000) */
    maxChars?: number;
    /** Include trust scores inline (default: false) */
    showTrust?: boolean;
    /** Include source attribution (default: true) */
    showSource?: boolean;
    /** Header text (default: "## Relevant Memory Context") */
    header?: string;
  }
): string {
  // Implementation:
  // 1. Start with header
  // 2. Add opinions first (they're the highest-signal, most condensed)
  //    Format: "- [confidence 0.85] Belief statement (domain)"
  // 3. Add observations
  //    Format: "- Observation text (topic)"
  // 4. Add recall results, ordered by score, truncating text to fit budget
  //    Format: "- [trust 0.9, source] Memory text..."
  // 5. Track character count, stop adding when maxChars would be exceeded
  // 6. If results were truncated, add "(N more results omitted)"
}
```

**Export** from `src/recall.ts` and re-export from `src/engram.ts`.

**Tests:** Add cases to `tests/recall.test.ts` verifying formatting output, character budget enforcement, and truncation behavior.

---

## Gap 3: Query-Scoped Opinions and Observations

**Problem:** `recall()` fetches top 5 opinions and top 5 observations globally — not filtered by the query. Asking about Terraform returns opinions about cooking.

### Modify `src/recall.ts` — the opinion and observation queries:

**Opinions:** Instead of a simple `ORDER BY confidence DESC LIMIT 5`, add domain matching and keyword overlap:

```typescript
// If we can extract a domain hint from the query or from the retrieved chunks,
// filter opinions by domain. Also do keyword matching against the belief text.
const queryTokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 3);

// Strategy: score opinions by relevance to query + retrieved chunk domains
// 1. Get domains from the top retrieved chunks
// 2. Filter opinions that match those domains OR whose belief text contains query tokens
// 3. Fall back to global top opinions if no domain-relevant ones found
```

**Observations:** Same approach — filter by domain extracted from retrieved chunks, with keyword overlap on summary text. Fall back to global if no matches.

**The key change:** Both opinion and observation queries should receive the `query` string and the `domains` extracted from the top-K results, and use them to filter.

**Tests:** Add test cases with opinions across multiple domains, verify that recall with a domain-specific query returns domain-relevant opinions/observations.

---

## Gap 4: Source and Context Filtering on Recall

**Problem:** No way to say "prefer memories from this project" or "only from this conversation." Gage has project IDs and session IDs stored in Engram's `source` and `context` fields, but `RecallOptions` has no filter for them.

### Modify `src/recall.ts` — add to `RecallOptions`:

```typescript
export interface RecallOptions {
  // ... existing options ...
  
  /** Filter results to chunks matching this source pattern (substring match) */
  sourceFilter?: string;
  /** Filter results to chunks matching this context (exact or substring) */
  contextFilter?: string;
  /** Boost score for chunks matching this source (multiplicative, default: 1.5) */
  sourceBoost?: { pattern: string; multiplier: number };
  /** Boost score for chunks matching this context (multiplicative, default: 1.5) */
  contextBoost?: { pattern: string; multiplier: number };
}
```

**`sourceFilter` and `contextFilter`** add WHERE clauses to all four retrieval strategies (semantic, keyword, graph, temporal). These are hard filters — non-matching results are excluded.

**`sourceBoost` and `contextBoost`** are soft preferences — they multiply the RRF score for matching chunks after fusion, before final ranking. This lets Gage say "boost memories from project:barracuda" without excluding everything else.

Apply the boost in `applyTrustWeighting()` (rename to `applyWeighting()` since it now handles more than trust).

**Tests:** Add cases verifying filter exclusion and boost reranking behavior.

---

## Gap 5: Forget and Supersede API

**Problem:** Schema has `superseded_by` and `is_active` but no API methods. When facts change, stale chunks persist indefinitely.

### Add to `src/engram.ts` (Engram class methods):

```typescript
/**
 * Soft-delete a memory chunk. Sets is_active = FALSE.
 * The chunk remains in the database for audit but is excluded from recall.
 */
async forget(chunkId: string): Promise<boolean> {
  // UPDATE chunks SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = ?
  // Return true if a row was affected
}

/**
 * Supersede an old fact with a new one. The old chunk is marked with
 * superseded_by pointing to the new chunk, and is_active set to FALSE.
 * The new chunk is retained normally.
 *
 * Use when correcting information: "Actually, I switched to Pulumi"
 * supersedes "Tom prefers Terraform".
 */
async supersede(
  oldChunkId: string,
  newText: string,
  options?: RetainOptions
): Promise<RetainResult> {
  // 1. retain() the new text → get new chunkId
  // 2. UPDATE old chunk: superseded_by = newChunkId, is_active = FALSE
  // 3. Return the RetainResult for the new chunk
}

/**
 * Forget all chunks matching a source pattern.
 * Useful for clearing out an entire conversation or document import.
 */
async forgetBySource(sourcePattern: string): Promise<number> {
  // UPDATE chunks SET is_active = FALSE WHERE source LIKE ? AND is_active = TRUE
  // Return count of affected rows
}
```

Also add `engram_forget` and `engram_supersede` to `src/mcp-tools.ts` tool definitions.

**Tests:** Add to `tests/engram.test.ts` — verify forgotten chunks don't appear in recall, superseded chunks point to replacement, forgetBySource clears matching chunks.

---

## Gap 6: Chunk Deduplication

**Problem:** Retaining "Tom prefers Terraform" three times creates three identical chunks competing for ranking.

### Modify `src/retain.ts` — add dedup check inside `retain()`:

Before creating a new chunk, check for near-duplicates:

```typescript
// Dedup check: if an active chunk with identical text already exists,
// update its trust_score (take the max) and updated_at instead of creating a new chunk.
// For near-duplicates (same text after normalization), do the same.

function normalizeForDedup(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

// In retain(), before the insert transaction:
const existing = db.prepare(`
  SELECT id, trust_score FROM chunks
  WHERE is_active = TRUE AND text = ?
  LIMIT 1
`).get(text) as { id: string; trust_score: number } | undefined;

if (existing) {
  // Reinforce existing chunk instead of creating duplicate
  const newTrust = Math.max(existing.trust_score, options.trustScore ?? 0.5);
  db.prepare(`
    UPDATE chunks SET trust_score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
  `).run(newTrust, existing.id);
  return { chunkId: existing.id, queued: false, deduplicated: true };
}
```

**Update `RetainResult`** to include `deduplicated?: boolean` so callers know what happened.

**Note:** Use exact text match, not normalized match, for the default. Near-dedup (normalized) can be an option (`dedupMode: 'exact' | 'normalized' | 'none'`) with `'exact'` as default. `'none'` preserves current behavior for callers that want it.

Add `dedupMode` to `RetainOptions`:

```typescript
export interface RetainOptions {
  // ... existing ...
  /** Dedup mode: 'exact' (default) skips if identical text exists, 'normalized' ignores case/whitespace, 'none' always creates new chunk */
  dedupMode?: 'exact' | 'normalized' | 'none';
}
```

**Tests:** Retain same text twice → verify only one chunk exists and trust was reinforced. Retain with `dedupMode: 'none'` → verify two chunks created. Retain with `dedupMode: 'normalized'` → verify case-different text deduplicates.

---

## Gap 7: Trust Score Decay

**Problem:** A fact from 6 months ago with `trust_score: 0.9` ranks the same as one from today. No temporal decay.

### Modify `src/recall.ts` — add decay to `applyWeighting()` (formerly `applyTrustWeighting()`):

```typescript
/**
 * Apply temporal decay to trust scores during recall ranking.
 * Recent memories get a slight boost; old memories get penalized.
 *
 * Decay model: exponential with configurable half-life.
 * At half-life age, score is multiplied by 0.5.
 * Default half-life: 180 days (6 months).
 *
 * Memories with verified_by_user = TRUE are exempt from decay.
 */
function temporalDecayMultiplier(createdAt: string, halfLifeDays: number = 180): number {
  const ageMs = Date.now() - new Date(createdAt).getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  // Exponential decay: multiplier = 2^(-age/halfLife)
  // At age=0: 1.0, at age=halfLife: 0.5, at age=2*halfLife: 0.25
  return Math.pow(2, -ageDays / halfLifeDays);
}
```

Apply this in the weighting function — multiply the fused score by the decay multiplier. The chunk data already includes `created_at` (and `event_time` if set).

**Add to `RecallOptions`:**

```typescript
/** Trust decay half-life in days (default: 180). Set to 0 to disable decay. */
decayHalfLifeDays?: number;
```

**Tests:** Create chunks with different created_at timestamps (use the test DB directly), verify that older chunks receive lower final scores than identical newer chunks. Verify `decayHalfLifeDays: 0` disables decay.

---

## Files Summary

| Action | File | Changes |
|--------|------|---------|
| MODIFY | `src/retain.ts` | Add `shouldRetain()`, add dedup check in `retain()`, add `dedupMode` to `RetainOptions`, update `RetainResult` |
| MODIFY | `src/recall.ts` | Add `formatForPrompt()`, add `sourceFilter`/`contextFilter`/`sourceBoost`/`contextBoost`/`decayHalfLifeDays` to `RecallOptions`, scope opinions/observations to query domain, add temporal decay, rename `applyTrustWeighting` → `applyWeighting` |
| MODIFY | `src/engram.ts` | Add `forget()`, `supersede()`, `forgetBySource()` methods, re-export `formatForPrompt` and `shouldRetain` |
| MODIFY | `src/mcp-tools.ts` | Add `engram_forget` and `engram_supersede` tool definitions |
| CREATE | `tests/retain-gate.test.ts` | Test suite for `shouldRetain()` |
| MODIFY | `tests/retain.test.ts` | Add dedup tests |
| MODIFY | `tests/recall.test.ts` | Add `formatForPrompt` tests, source/context filter tests, domain-scoped opinion tests, temporal decay tests |
| MODIFY | `tests/engram.test.ts` | Add `forget()`, `supersede()`, `forgetBySource()` tests |

## Files That Should NOT Change

- `src/schema.sql` — the schema already supports everything needed (`superseded_by`, `is_active`, `source`, `context`, `verified_by_user`). No schema changes required.
- `src/reflect.ts` — reflection logic is unchanged. The scheduling discussion (ReflectScheduler vs external scheduler) is an integration concern, not a library change.
- `vitest.config.ts` — no changes needed
- `tsconfig.json` — no changes needed

## Implementation Order

1. **Dedup** (Gap 6) — simplest, standalone, no new APIs
2. **Forget/Supersede** (Gap 5) — new Engram methods, straightforward
3. **Retain Gate** (Gap 1) — new export, no changes to existing code
4. **Source/Context Filtering** (Gap 4) — modifies recall internals
5. **Query-Scoped Opinions** (Gap 3) — modifies recall internals (do together with #4)
6. **Trust Decay** (Gap 7) — modifies recall scoring (do after #4/#5 since it touches the same weighting function)
7. **Prompt Formatting** (Gap 2) — depends on recall output shape being finalized

## Testing

```bash
npm test          # all tests pass (existing 52 + new)
npm run build     # clean compile
npm run typecheck # no type errors
```

All new tests use the existing `MockEmbedder` and `createTestDb()` helpers from `tests/helpers.ts` — no Ollama dependency for testing.
