# Operative ← Engram Integration (Revised) — Claude Code Spec

**Operative Repo:** `G:\Projects\SIT\operative`  
**Engram Repo:** `G:\Projects\SIT\engram` (do not modify)  
**Gage Repo:** `C:\Users\tom-s\valor\agents\gage` (Phase 2 only)

**Design Change from Previous Spec:** Instead of scattering Engram calls across validate.ts, report.ts, loop.ts, persona.ts, and scheduler.ts, all Engram interaction is centralized in a single thin adapter module: `src/core/engram-adapter.ts`. The VALOR cycle files call the adapter. The adapter calls Engram. This makes the integration obvious to anyone reading the code and easy to replicate in other frameworks.

---

## Why an Adapter

An OpenClaw user integrating Engram does five things:

```typescript
// 1. Before LLM call — get context
const context = await engram.recall(input);
const memoryBlock = formatForPrompt(context);

// 2. After LLM call — store what matters
if (shouldRetain(input).score > 0.3) await engram.retain(input, { ... });

// 3. Background — extraction + reflection
setInterval(() => engram.processExtractions(), 5 * 60 * 1000);
setInterval(() => engram.reflect(), 6 * 60 * 60 * 1000);

// 4. Shutdown
engram.close();
```

The adapter does these same five things, mapped to the VALOR cycle. Anyone reading `engram-adapter.ts` sees the universal pattern. Anyone reading the VALOR cycle files sees `import { recallContext, retainTurn } from './engram-adapter.js'` — clean, contained, obvious.

---

## New File: `src/core/engram-adapter.ts`

This is the entire Engram integration for Operative. ~80 lines.

```typescript
// core/engram-adapter.ts — Thin adapter wiring Engram into the VALOR cycle.
//
// Centralizes all Engram interaction in one module. The VALOR cycle files
// import from here — they never import from 'engram' directly.
//
// If Engram is not configured (config.engram.instance is null), every
// function in this module is a no-op. Zero behavioral change.
//
// This adapter implements the same 5-step pattern any framework would use:
//   1. recall()         → before LLM call (Validate phase)
//   2. retain()         → after LLM call (Report phase)
//   3. processExtractions() → background tick
//   4. reflect()        → background tick
//   5. close()          → shutdown

import { config } from '../config.js';
import type { Session } from '../types/index.js';

// ─── Types ──────────────────────────────────────────────────────────────────

// We type the engram instance loosely to avoid a hard compile-time dependency.
// The agent passes a fully-typed Engram instance; we just call its methods.
interface EngramLike {
  recall(query: string, options?: { topK?: number }): Promise<any>;
  retain(text: string, options?: Record<string, unknown>): Promise<any>;
  processExtractions(batchSize?: number): Promise<any>;
  reflect(): Promise<any>;
  close(): void;
}

// Cached dynamic imports — resolved once, reused.
let _shouldRetain: ((text: string) => { score: number; reason: string }) | null = null;
let _formatForPrompt: ((response: any, options?: any) => string) | null = null;

async function loadEngramHelpers(): Promise<boolean> {
  if (_shouldRetain && _formatForPrompt) return true;
  try {
    const mod = await import('engram');
    _shouldRetain = mod.shouldRetain;
    _formatForPrompt = mod.formatForPrompt;
    return true;
  } catch {
    return false;
  }
}

function getEngram(): EngramLike | null {
  return config.engram?.instance ?? null;
}

// ─── 1. Recall — Before LLM Call (Validate Phase) ──────────────────────────

/**
 * Recall relevant memories for the current input.
 * Returns a formatted string for system prompt injection, or undefined if
 * Engram is not configured or recall returns nothing meaningful.
 */
export async function recallContext(userInput: string): Promise<string | undefined> {
  const engram = getEngram();
  if (!engram) return undefined;

  try {
    if (!(await loadEngramHelpers())) return undefined;

    const response = await engram.recall(userInput, {
      topK: config.engram.recallTopK,
    });

    // Only inject if we got meaningful results
    if (!response.results?.length && !response.opinions?.length && !response.observations?.length) {
      return undefined;
    }

    return _formatForPrompt!(response, {
      maxChars: config.engram.recallMaxChars,
    });
  } catch (err) {
    if (config.debug) console.debug('[ENGRAM] Recall failed:', err);
    return undefined;
  }
}

// ─── 2. Retain — After LLM Call (Report Phase) ─────────────────────────────

/**
 * Retain conversation turns that pass the shouldRetain() gate.
 * Retains user input and assistant response separately with different trust levels.
 */
export async function retainTurn(
  userInput: string,
  assistantResponse: string,
  session: Session,
): Promise<void> {
  const engram = getEngram();
  if (!engram) return;

  try {
    if (!(await loadEngramHelpers())) return;

    const threshold = config.engram.retainThreshold;
    const projectContext = session.activeProjectName ?? undefined;

    // User input — high trust (the operator said it)
    const userScore = _shouldRetain!(userInput);
    if (userScore.score >= threshold) {
      await engram.retain(userInput, {
        memoryType: 'experience',
        source: `session:${session.id}`,
        sourceType: 'user_stated',
        trustScore: 0.85,
        context: projectContext,
      });
    }

    // Assistant response — moderate trust (agent generated)
    const assistantScore = _shouldRetain!(assistantResponse);
    if (assistantScore.score >= threshold) {
      await engram.retain(assistantResponse, {
        memoryType: 'experience',
        source: `session:${session.id}`,
        sourceType: 'agent_generated',
        trustScore: 0.6,
        context: projectContext,
      });
    }
  } catch (err) {
    if (config.debug) console.debug('[ENGRAM] Retain failed:', err);
  }
}

// ─── 3 & 4. Background Ticks ────────────────────────────────────────────────

/** Run entity extraction batch. Called by the scheduler tick engine. */
export async function runExtraction(): Promise<void> {
  const engram = getEngram();
  if (!engram) return;

  try {
    await engram.processExtractions(10);
  } catch (err) {
    if (config.debug) console.debug('[ENGRAM] Extraction tick failed:', err);
  }
}

/** Run reflection cycle. Called by the scheduler tick engine. */
export async function runReflection(): Promise<void> {
  const engram = getEngram();
  if (!engram) return;

  try {
    await engram.reflect();
  } catch (err) {
    if (config.debug) console.debug('[ENGRAM] Reflect tick failed:', err);
  }
}

// ─── 5. Shutdown ────────────────────────────────────────────────────────────

/** Close the engram connection. Called on process exit. */
export function closeEngram(): void {
  const engram = getEngram();
  if (!engram) return;

  try {
    engram.close();
  } catch {
    // Non-fatal on shutdown
  }
}

// ─── Status ─────────────────────────────────────────────────────────────────

/** Whether Engram is active. Used by /status display. */
export function isEngramActive(): boolean {
  return getEngram() !== null;
}
```

---

## Modifications to Existing Files

### `src/config.ts` — Add Engram Config

Add to `InternalConfig` interface:

```typescript
  // Engram knowledge memory (optional)
  engram: {
    instance: any | null;
    retainThreshold: number;
    extractionIntervalMs: number;
    reflectIntervalMs: number;
    recallMaxChars: number;
    recallTopK: number;
  };
```

Add to `initConfig()` defaults:

```typescript
    engram: {
      instance: null,
      retainThreshold: 0.3,
      extractionIntervalMs: 5 * 60 * 1000,
      reflectIntervalMs: 6 * 60 * 60 * 1000,
      recallMaxChars: 2000,
      recallTopK: 10,
    },
```

### `src/index.ts` — Add Engram to Public API + Startup Wiring

Add to `CreateOperativeOptions`:

```typescript
  /**
   * Optional Engram instance for persistent knowledge memory.
   * The agent creates this (Engram.open() or Engram.create()) and passes it in.
   * If not provided, the operative runs without knowledge memory.
   */
  engram?: {
    instance: unknown;
    retainThreshold?: number;
    extractionIntervalMs?: number;
    reflectIntervalMs?: number;
    recallMaxChars?: number;
    recallTopK?: number;
  };
```

In `createOperative()`, after `initConfig()`:

```typescript
  // Wire engram config if provided
  if (options.engram?.instance) {
    config.engram = {
      instance: options.engram.instance,
      retainThreshold: options.engram.retainThreshold ?? 0.3,
      extractionIntervalMs: options.engram.extractionIntervalMs ?? 5 * 60 * 1000,
      reflectIntervalMs: options.engram.reflectIntervalMs ?? 6 * 60 * 60 * 1000,
      recallMaxChars: options.engram.recallMaxChars ?? 2000,
      recallTopK: options.engram.recallTopK ?? 10,
    };
  }
```

After registering VALOR ticks (step 6), add engram ticks:

```typescript
  // 6b. Engram background ticks
  if (config.engram.instance) {
    registerTick('engram-extract', runExtraction, config.engram.extractionIntervalMs);
    registerTick('engram-reflect', runReflection, config.engram.reflectIntervalMs);
  }
```

Add import at top:

```typescript
import { runExtraction, runReflection } from './core/engram-adapter.js';
```

### `src/core/loop.ts` — Recall Before Act, Retain After Report

Add import:

```typescript
import { recallContext, retainTurn, isEngramActive } from './engram-adapter.js';
```

In `runCycle()`, **add recall between Validate and Act:**

```typescript
  // ── VALIDATE ─────────────────────────────────────────────────────────────
  const quickValidation = validate(trimmed, session, 0);
  const systemPrompt = buildSystemPrompt(
    session,
    quickValidation.activeProject,
    quickValidation.preferences,
  );
  const systemTokenEstimate = estimateSystemPromptTokens(systemPrompt);
  const validation = validate(trimmed, session, systemTokenEstimate);

  // ── RECALL (Engram) ──────────────────────────────────────────────────────
  const memoryContext = await recallContext(trimmed);

  // ── ACT ───────────────────────────────────────────────────────────────────
  let response;
  try {
    response = await act(trimmed, validation, session, memoryContext);
  } catch (err: unknown) {
```

**Add retain after Report:**

```typescript
  // ── REPORT ────────────────────────────────────────────────────────────────
  report(trimmed, response, validation.intent, session, validation);

  // ── RETAIN (Engram) ──────────────────────────────────────────────────────
  if (response.content) {
    await retainTurn(trimmed, response.content, session);
  }

  return true;
```

In `printStatus()`, add Engram section:

```typescript
  if (isEngramActive()) {
    console.log('\n  ' + chalk.dim('─── Engram ───────────────────────────────────'));
    console.log(`  ${label('Status')}${chalk.green('active')}`);
    console.log(`  ${label('Retain gate')}score ≥ ${config.engram.retainThreshold}`);
    console.log(`  ${label('Extract tick')}every ${(config.engram.extractionIntervalMs / 60000).toFixed(0)}m`);
    console.log(`  ${label('Reflect tick')}every ${(config.engram.reflectIntervalMs / 3600000).toFixed(1)}h`);
    console.log(`  ${label('Recall budget')}${config.engram.recallMaxChars} chars / top ${config.engram.recallTopK}`);
  }
```

### `src/core/act.ts` — Accept Optional Memory Context

Change the `act()` signature:

```typescript
export async function act(
  userInput: string,
  validation: ValidationResult,
  session: Session,
  memoryContext?: string,
): Promise<ConduitResponse> {
```

Pass it to `buildSystemPrompt()`:

```typescript
  const systemPrompt = buildSystemPrompt(session, activeProject, preferences, memoryContext);
```

### `src/core/persona.ts` — Accept Optional Memory Context

Change `buildSystemPrompt()` signature:

```typescript
export function buildSystemPrompt(
  session: Session,
  activeProject: Project | null,
  preferences: Preference[],
  memoryContext?: string,
): string {
```

Inject memory context between project context and preferences:

```typescript
  if (activeProject) {
    parts.push(buildProjectBlock(activeProject));
  }

  // Engram memory context (optional — only present when Engram is active and recall returned results)
  if (memoryContext) {
    parts.push(memoryContext);
  }

  if (preferences.length > 0) {
    parts.push(buildPreferencesBlock(preferences));
  }
```

### `src/core/scheduler.ts` — Close Engram on Shutdown

Add import:

```typescript
import { closeEngram } from './engram-adapter.js';
```

In the SIGINT handler:

```typescript
  rl.on('SIGINT', () => {
    running = false;
    closeEngram();
    console.log(chalk.cyan(`\n\n  ${config.callsign} signing off. Mission log saved.\n`));
    rl.close();
    process.exit(0);
  });
```

Also fix: the current SIGINT handler says "Gage signing off" — change to `config.callsign`.

---

## Gage Integration (Phase 2)

### `C:\Users\tom-s\valor\agents\gage\package.json`

Add Engram dependency:

```json
"dependencies": {
  "operative": "file:G:/Projects/SIT/operative",
  "engram": "file:G:/Projects/SIT/engram",
  "dotenv": "^16.4.5"
}
```

### `C:\Users\tom-s\valor\agents\gage\index.ts`

```typescript
import 'dotenv/config';
import { createOperative } from 'operative';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Open Engram knowledge memory (optional — graceful fallback if unavailable)
let engramInstance: unknown;
try {
  const { Engram } = await import('engram');
  engramInstance = await Engram.open(
    path.resolve(__dirname, 'memory/gage.engram'),
    {
      ollamaUrl: process.env.OLLAMA_BASE_URL ?? 'http://starbase:40114',
      reflectMission: 'Focus on architecture preferences, infrastructure decisions, project patterns, and technical strategies.',
      retainMission: 'Prioritize technical decisions, project context, and stated preferences. Skip greetings and small talk.',
    },
  );
} catch (err) {
  console.warn(`  ⚠ Engram not available — running without knowledge memory.`);
}

await createOperative({
  callsign: 'Gage',
  soulPath: path.resolve(__dirname, 'config/gage.soul.md'),
  dbPath: path.resolve(__dirname, 'memory/gage.db'),
  repoRoot: __dirname,

  provider: {
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    anthropicModel: process.env.ANTHROPIC_MODEL,
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL,
    ollamaModel: process.env.OLLAMA_MODEL,
  },

  maxTokens: parseInt(process.env.MAX_OUTPUT_TOKENS || '0', 10) || undefined,
  maxContextTokens: parseInt(process.env.MAX_CONTEXT_TOKENS || '0', 10) || undefined,

  capabilities: [
    'architecture', 'code_review', 'technical_strategy', 'development', 'spec_writing',
  ],

  valorEngineUrl: process.env.VALOR_ENGINE_URL || undefined,
  valorDivisionId: process.env.VALOR_DIVISION_ID || undefined,

  containment: {
    protectedPaths: [
      'config/gage.soul.md', 'config/gage.keys.json',
      '.env', '.env.example', 'memory/',
      'package.json', 'package-lock.json', 'node_modules/', 'dist/',
    ],
    writablePaths: [],
  },

  engram: engramInstance ? {
    instance: engramInstance,
    retainThreshold: 0.3,
    extractionIntervalMs: 5 * 60 * 1000,
    reflectIntervalMs: 6 * 60 * 60 * 1000,
  } : undefined,

  debug: process.env.GAGE_DEBUG === 'true',
});
```

### `C:\Users\tom-s\valor\agents\gage\.gitignore`

Add:
```
memory/gage.engram
memory/gage.engram-shm
memory/gage.engram-wal
```

---

## Files Summary

### Operative (Phase 1)

| Action | File |
|--------|------|
| CREATE | `src/core/engram-adapter.ts` — all Engram interaction in ~80 lines |
| MODIFY | `src/config.ts` — add `engram` to InternalConfig |
| MODIFY | `src/index.ts` — add `engram` to CreateOperativeOptions, wire ticks |
| MODIFY | `src/core/loop.ts` — add `recallContext()` before Act, `retainTurn()` after Report, Engram in `/status` |
| MODIFY | `src/core/act.ts` — accept optional `memoryContext` param |
| MODIFY | `src/core/persona.ts` — accept optional `memoryContext` param |
| MODIFY | `src/core/scheduler.ts` — call `closeEngram()` on SIGINT, fix "Gage" → `config.callsign` |

### Gage (Phase 2)

| Action | File |
|--------|------|
| MODIFY | `index.ts` — open Engram, pass to createOperative |
| MODIFY | `package.json` — add engram dependency |
| MODIFY | `.gitignore` — add .engram files |

---

## Verification

### Phase 1 — Framework

```bash
cd G:\Projects\SIT\operative
npm run build    # Must compile — no hard engram import at top level
npm test         # All existing tests pass
```

The adapter uses `await import('engram')` — this only resolves at runtime. The framework compiles without Engram installed.

### Phase 2 — Gage with Engram

```bash
cd G:\Projects\SIT\engram && npm run build
cd C:\Users\tom-s\valor\agents\gage
npm install
npm start
```

1. Gage boots — `/status` shows Engram section with "active"
2. Conversation turns are retained (visible in `memory/gage.engram` via SQLite)
3. Recall injects context on subsequent turns
4. `GAGE_DEBUG=true` shows extraction/reflection tick activity
5. `/exit` — clean shutdown

### Phase 2 — Gage without Engram

Remove `engram` from Gage's `package.json`, `npm install`, `npm start`:

1. Warning: "Engram not available — running without knowledge memory"
2. Everything works identically to pre-Engram behavior
3. `/status` has no Engram section