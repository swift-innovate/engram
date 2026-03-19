# Engram Working Memory Refactor — Expose Primitives, Move Policy to Agent Layer

**Repo:** `G:\Projects\SIT\engram`  
**Context:** Working memory shipped in Sprint 1. `inferWorkingSession()` bundles three concerns: similarity search (infrastructure), match decision (policy), and context loading (convenience). This refactor separates them so agents can implement custom session resolution.

**Principle:** Engram is a storage/retrieval engine. Session matching _policy_ (thresholds, topic models, channel-specific behavior) belongs at the agent layer.

---

## Goal

Expose low-level working memory primitives alongside the existing `inferWorkingSession()` convenience method. No breaking changes — `inferWorkingSession()` keeps working exactly as it does today, but agents that need custom session resolution can use the primitives directly.

---

## New Public Methods to Add on `Engram` class

### `findSimilarSessions(embedding, limit?)`

Extract the similarity search from `inferWorkingSession()` into its own method:

```typescript
/**
 * Find active working memory sessions similar to the given embedding.
 * Returns candidates sorted by similarity (highest first).
 * 
 * This is the low-level primitive — it does NOT make a match/new decision.
 * The agent's adapter layer uses these results to implement its own policy.
 */
findSimilarSessions(
  embedding: Float32Array,
  limit?: number  // default: 3
): Array<{ id: string; state: WorkingMemoryState; similarity: number }>
```

Implementation: pull the `vec_distance_cosine` query + similarity conversion out of `inferWorkingSession()` into this method. Return parsed `WorkingMemoryState` (not raw `data_json`).

This should be a **synchronous** method — it only touches SQLite, no async needed.

### `embedText(text)`

Expose the embedder so agents can generate embeddings without calling retain/recall:

```typescript
/**
 * Generate an embedding for arbitrary text using the configured provider.
 * Useful for agents that need embeddings for session matching without
 * going through retain() or recall().
 */
async embedText(text: string): Promise<Float32Array>
```

Implementation: just `return this.embedder.embed(text)`.

---

## Refactor `inferWorkingSession()`

After extracting `findSimilarSessions()` and `embedText()`, refactor `inferWorkingSession()` to use them internally:

```typescript
async inferWorkingSession(message, options) {
  const embedding = await this.embedText(message);
  const embeddingBuffer = Buffer.from(embedding.buffer);
  const candidates = this.findSimilarSessions(embedding, 3);
  
  const threshold = options.threshold ?? 0.55;
  const best = candidates[0];
  
  if (best && best.similarity >= threshold) {
    // resume existing — same logic as before
  } else {
    // create new — same logic as before  
  }
  
  // load related context — same logic as before
}
```

The behavior is **identical** — this is a pure extraction refactor. `inferWorkingSession()` is now a convenience wrapper that uses the new primitives.

---

## Update Exports

In `src/engram.ts`, both new methods are public on the `Engram` class. No new files needed.

In `src/working-memory-types.ts`, add:

```typescript
export interface SessionCandidate {
  id: string;
  state: WorkingMemoryState;
  similarity: number;
}
```

Re-export `SessionCandidate` from `src/engram.ts`.

---

## Update MCP Tools (Optional)

Add `engram_find_sessions` tool that wraps `embedText()` + `findSimilarSessions()` for agent frameworks that want raw candidates without the match decision.

---

## Tests

Add to `tests/working-memory.test.ts`:

1. `findSimilarSessions()` returns empty array when no sessions exist
2. `findSimilarSessions()` returns candidates sorted by similarity descending
3. `findSimilarSessions()` excludes expired sessions
4. `findSimilarSessions()` respects limit parameter
5. `embedText()` returns a Float32Array of the correct dimensions
6. `inferWorkingSession()` still passes all existing tests (regression check)

Use existing `MockEmbedder` and `createTestDb()` helpers.

---

## Files Changed

| Action | File | What |
|--------|------|------|
| MODIFY | `src/engram.ts` | Add `findSimilarSessions()`, `embedText()`, refactor `inferWorkingSession()` to use them |
| MODIFY | `src/working-memory-types.ts` | Add `SessionCandidate` interface |
| MODIFY | `src/mcp-tools.ts` | Optional: add `engram_find_sessions` tool |
| MODIFY | `tests/working-memory.test.ts` | Add primitive tests, verify regression |

## Files NOT Changed

Everything else. This is purely additive + internal refactor.

---

## Verification

```bash
npm run typecheck  # zero errors
npm run build      # clean
npm test           # all existing tests pass + new tests pass
```

Then in Tracer:

```bash
cd G:\Projects\SIT\agents\tracer
npm run test       # all 19 pass (unchanged — inferWorkingSession API is identical)
```

---

## Why This Matters

After this refactor, an agent adapter can do:

```typescript
// Agent-level custom session resolution
const embedding = await engram.embedText(userInput);
const candidates = engram.findSimilarSessions(embedding, 5);

// Custom policy: ranch agent matches by animal name, not embedding similarity
const animalMention = extractAnimalName(userInput);
if (animalMention) {
  const match = candidates.find(c => c.state.animal === animalMention);
  if (match) return match;
}

// Fall back to similarity for non-animal topics  
if (candidates[0]?.similarity > 0.6) return candidates[0];
return engram.createWorkingSession(userInput, embedding);
```

This is the plugin path. Engram owns storage + retrieval primitives. The agent owns the decision.
