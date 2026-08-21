#!/usr/bin/env node
// =============================================================================
// cli.ts — `engram` command-line transport over the Engram core
//
// A fourth transport alongside the library API and the MCP server. Structurally
// a sibling of mcp-server.ts: parse argv → Engram.open() → dispatch to the SAME
// engram methods → print. No retain/recall/reflect logic lives here.
//
// One subcommand per MCP tool, kebab-cased so skills/cli-memory/SKILL.md maps
// 1:1 to the MCP tool surface:
//
//   engram retain <text>            engram recall <query>
//   engram reflect                  engram process-extractions
//   engram forget <chunkId>         engram supersede <oldChunkId> <newText>
//   engram session <message>        engram queue-stats
//   engram requeue-failed           engram embed <text>
//
// Contract:
//   --json        emit the raw method return as JSON to stdout, nothing else.
//   (default)     human-readable output to stdout.
//   stdout        data only. All diagnostics/logging go to stderr.
//   stdin         primary text arg for retain/recall/session/supersede-newText
//                 is read from stdin when the positional is omitted.
//   exit codes    0 success · 2 not-found (forget/supersede missing chunk) · 1 error.
//
// Usage:
//   engram recall "terraform" --db ./agent.engram
//   ENGRAM_DB=./agent.engram engram retain "Tom prefers Pulumi" --memory-type world
//   echo "long context" | engram retain --db ./agent.engram --json
// =============================================================================

import Database from 'better-sqlite3';
import { pathToFileURL } from 'url';

import { Engram, formatWhyLine } from './engram.js';
import type {
  EngramOptions,
  RetainOptions,
  RecallOptions,
  RecallResponse,
  RetainResult,
  ReflectResult,
  WorkingSessionResult,
  WorkingMemoryState,
  DecisionArtifact,
  TaskScope,
  TokenBudget,
  ContextSlice,
  IntrospectResult,
  SuggestionKind,
  SuggestionStatus,
  SuggestionView,
} from './engram.js';
import type { QueueStats } from './retain.js';
import {
  parseArgs,
  buildEngramOptions,
  resolveDbPath,
  clampTrust,
  clampNonNegative,
  filterEnums,
  asNumber,
  asList,
  triStateFlag,
  VALID_MEMORY_TYPES,
  VALID_SOURCE_TYPES,
  VALID_STRATEGIES,
  VALID_SUGGESTION_STATUSES,
  VALID_SUGGESTION_KINDS,
  type ParsedArgs,
} from './cli-args.js';
import {
  resolveModelSpec,
  preflightModel,
  formatPreflightFailure,
} from './model-resolver.js';

// ─── Exit codes ──────────────────────────────────────────────────────────────

const EXIT_OK = 0;
const EXIT_ERROR = 1;
const EXIT_NOT_FOUND = 2;

// ─── IO abstraction (injectable for tests) ───────────────────────────────────

export interface CliIO {
  stdout(s: string): void;
  stderr(s: string): void;
  /** Read the full stdin as text. Called lazily, only when a text arg is omitted. */
  readStdin(): Promise<string>;
}

async function readAllStdin(): Promise<string> {
  if (process.stdin.isTTY) return '';
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8').trim();
}

const defaultIo: CliIO = {
  stdout: (s) => process.stdout.write(s),
  stderr: (s) => process.stderr.write(s),
  readStdin: readAllStdin,
};

// ─── Option builders (input → engram method options) ─────────────────────────

function buildRetainOptions(args: ParsedArgs): RetainOptions {
  const memoryType = args.values.get('--memory-type');
  const sourceType = args.values.get('--source-type');
  return {
    memoryType: VALID_MEMORY_TYPES.has(memoryType ?? '')
      ? (memoryType as RetainOptions['memoryType'])
      : undefined,
    sourceType: VALID_SOURCE_TYPES.has(sourceType ?? '')
      ? (sourceType as RetainOptions['sourceType'])
      : undefined,
    trustScore: clampTrust(asNumber(args.values.get('--trust-score'))),
    source: args.values.get('--source'),
    context: args.values.get('--context'),
    eventTime: args.values.get('--event-time'),
    temporalLabel: args.values.get('--temporal-label'),
    supersedes: args.values.get('--supersedes'),
  };
}

function buildReflectOptions(
  args: ParsedArgs,
): Parameters<Engram['reflect']>[0] {
  const minEvidenceCount = asNumber(args.values.get('--opinion-min-evidence'));
  const minDistinctDays = asNumber(
    args.values.get('--opinion-min-distinct-days'),
  );
  const minDistinctSources = asNumber(
    args.values.get('--opinion-min-distinct-sources'),
  );
  const counterTopK = asNumber(args.values.get('--counter-evidence-top-k'));
  const ratio = clampTrust(
    asNumber(args.values.get('--counter-evidence-max-contradiction-ratio')),
  );
  return {
    ...(args.bools.has('--suggest') ? { suggestions: {} } : {}),
    ...(args.bools.has('--unsafe-opinions')
      ? { opinionGates: false }
      : minEvidenceCount !== undefined ||
          minDistinctDays !== undefined ||
          minDistinctSources !== undefined
        ? {
            opinionGates: {
              minEvidenceCount: clampNonNegative(minEvidenceCount),
              minDistinctDays: clampNonNegative(minDistinctDays),
              minDistinctSources: clampNonNegative(minDistinctSources),
            },
          }
        : {}),
    ...(args.bools.has('--no-counter-evidence')
      ? { counterEvidence: false }
      : counterTopK !== undefined ||
          ratio !== undefined ||
          args.bools.has('--counter-evidence-on-reinforce') ||
          args.bools.has('--counter-evidence-fail-open')
        ? {
            counterEvidence: {
              topK:
                counterTopK !== undefined
                  ? Math.max(1, Math.floor(counterTopK))
                  : undefined,
              maxContradictionRatio: ratio,
              onReinforce: args.bools.has('--counter-evidence-on-reinforce')
                ? true
                : undefined,
              failOpen: args.bools.has('--counter-evidence-fail-open')
                ? true
                : undefined,
            },
          }
        : {}),
  };
}

function buildRecallOptions(args: ParsedArgs): RecallOptions {
  const topK = asNumber(args.values.get('--top-k'));
  return {
    topK: topK !== undefined ? Math.max(1, Math.floor(topK)) : undefined,
    strategies: filterEnums<'semantic' | 'keyword' | 'graph' | 'temporal'>(
      asList(args.values.get('--strategies')),
      VALID_STRATEGIES,
    ),
    memoryTypes: filterEnums<
      'world' | 'experience' | 'observation' | 'opinion'
    >(asList(args.values.get('--memory-types')), VALID_MEMORY_TYPES),
    minTrust: clampTrust(asNumber(args.values.get('--min-trust'))),
    after: args.values.get('--after'),
    before: args.values.get('--before'),
    includeOpinions: triStateFlag(args.bools, '--opinions', '--no-opinions'),
    includeObservations: triStateFlag(
      args.bools,
      '--observations',
      '--no-observations',
    ),
    minScore: clampTrust(asNumber(args.values.get('--min-score'))),
    insightMinScore: clampTrust(
      asNumber(args.values.get('--insight-min-score')),
    ),
    explainScores: args.bools.has('--explain-scores') ? true : undefined,
    decayHalfLifeDays: clampNonNegative(
      asNumber(args.values.get('--decay-half-life-days')),
    ),
  };
}

/**
 * Build a DecisionArtifact from a parsed JSON payload (the context-commit
 * primary arg/stdin body). Returns undefined when `decision` is missing —
 * the only required field, same convention as requireString elsewhere.
 */
function buildDecisionArtifactFromPayload(
  parsed: Record<string, unknown>,
): DecisionArtifact | undefined {
  if (typeof parsed.decision !== 'string' || parsed.decision.trim() === '') {
    return undefined;
  }
  return {
    decision: parsed.decision,
    rationale:
      typeof parsed.rationale === 'string' ? parsed.rationale : undefined,
    scoredOptions: Array.isArray(parsed.scoredOptions)
      ? (parsed.scoredOptions as DecisionArtifact['scoredOptions'])
      : undefined,
    confidence: clampTrust(parsed.confidence),
    refsToSource: Array.isArray(parsed.refsToSource)
      ? (parsed.refsToSource as unknown[]).filter(
          (v): v is string => typeof v === 'string',
        )
      : undefined,
    domain: typeof parsed.domain === 'string' ? parsed.domain : undefined,
    agentId: typeof parsed.agentId === 'string' ? parsed.agentId : undefined,
  };
}

// ─── Human-readable formatters ───────────────────────────────────────────────

function formatRetain(r: RetainResult): string {
  const flags = [
    `queued=${r.queued}`,
    r.deduplicated ? 'deduplicated' : null,
    r.tier1
      ? `entities=${r.tier1.entitiesLinked} relations=${r.tier1.relationsCreated}`
      : null,
  ]
    .filter(Boolean)
    .join(' ');
  return `retained ${r.chunkId} (${flags})\n`;
}

function formatRecall(r: RecallResponse): string {
  const lines: string[] = [];
  lines.push(
    `${r.results.length} result(s) · ${r.totalCandidates} candidate(s) · strategies: ${r.strategiesUsed.join(', ') || 'none'}`,
  );
  r.results.forEach((res, i) => {
    lines.push(
      `\n[${i + 1}] id=${res.id} score=${res.score.toFixed(3)} ${res.memoryType}/${res.sourceType} trust=${res.trustScore.toFixed(2)} created=${res.createdAt.slice(0, 10)} via=${res.strategies.join('+')}`,
    );
    lines.push(`    ${res.text}`);
    // Only present when the recall ran with --explain-scores; the shared
    // formatWhyLine helper keeps this rendering identical to formatForPrompt's.
    const why = formatWhyLine(res);
    if (why) lines.push(`  ${why}`);
  });
  if (r.opinions.length > 0) {
    lines.push('\nOpinions:');
    for (const o of r.opinions) {
      lines.push(`  (${o.confidence.toFixed(2)}) ${o.belief}`);
    }
  }
  if (r.observations.length > 0) {
    lines.push('\nObservations:');
    for (const o of r.observations) {
      lines.push(`  - ${o.summary}`);
    }
  }
  return lines.join('\n') + '\n';
}

function formatReflect(r: ReflectResult): string {
  return (
    `reflect: status=${r.status} facts=${r.factsProcessed} ` +
    `observations=+${r.observationsCreated}/~${r.observationsUpdated} ` +
    `opinions=new${r.opinionsFormed}/+${r.opinionsReinforced}/-${r.opinionsChallenged} ` +
    `(${r.durationMs}ms)\n`
  );
}

function formatSession(r: WorkingSessionResult): string {
  const lines = [
    `session ${r.session.id} (${r.diagnostics.reason}, confidence=${r.confidence.toFixed(2)}, candidates=${r.diagnostics.candidatesEvaluated})`,
    `goal: ${r.session.goal}`,
  ];
  if (r.relatedContext.trim()) {
    lines.push('related context:', r.relatedContext);
  }
  return lines.join('\n') + '\n';
}

function formatContextSlice(s: ContextSlice): string {
  const lines = [
    `${s.artifacts.length} artifact(s) · ${s.totalCandidates} candidate(s) · truncated=${s.truncated}`,
  ];
  s.artifacts.forEach((a, i) => {
    lines.push(`[${i + 1}] ${a.ref.id}: ${a.artifact.decision}`);
  });
  return lines.join('\n') + '\n';
}

function formatQueueStats(s: QueueStats): string {
  let out =
    `queue: pending=${s.pending} processing=${s.processing} ` +
    `completed=${s.completed} failed=${s.failed} ` +
    `oldest_pending=${s.oldest_pending ?? 'none'}\n`;
  if (s.failed_reasons.length > 0) {
    out += 'failed reasons:\n';
    for (const r of s.failed_reasons) {
      out += `  ${r.count}x ${r.error ?? '(no error recorded)'}\n`;
    }
  }
  return out;
}

function formatIntrospect(r: IntrospectResult): string {
  const subj = r.subject ? ` about "${r.subject}"` : ' (top held state)';
  let out = `held state${subj}: ${r.opinions.length} opinion(s), ${r.observations.length} observation(s)\n`;
  for (const o of r.opinions) {
    const dom = o.domain ? ` [${o.domain}]` : '';
    out +=
      `  opinion ${o.id}${dom} conf=${o.confidence.toFixed(2)} ` +
      `support=${o.supportCount} challenge=${o.challengeCount}: ${o.belief}\n`;
  }
  for (const o of r.observations) {
    const dom = o.domain ? ` [${o.domain}]` : '';
    out += `  observation ${o.id}${dom} refreshes=${o.refreshCount}: ${o.summary}\n`;
  }
  return out;
}

function formatSuggestions(rows: SuggestionView[]): string {
  if (rows.length === 0) return 'no suggestions\n';
  return rows
    .map(
      (s) =>
        `[${s.id}] (${s.kind ?? 'unknown'}, ${s.status}, evidence ${s.evidenceCount}) ${s.summary}\n`,
    )
    .join('');
}

// ─── Misc helpers ─────────────────────────────────────────────────────────────

/**
 * Does an active chunk with this id exist? Engram exposes no chunk getter and
 * engram.supersede() always succeeds (creating a new chunk regardless), so we
 * pre-check via a short-lived read-only connection — this both honors the
 * exit-code-2 contract and avoids leaving an orphan chunk when the target id
 * does not exist.
 */
function chunkExists(dbPath: string, chunkId: string): boolean {
  let db: Database.Database | undefined;
  try {
    db = new Database(dbPath, { readonly: true, fileMustExist: true });
    const row = db
      .prepare('SELECT 1 AS x FROM chunks WHERE id = ? AND is_active = TRUE')
      .get(chunkId);
    return Boolean(row);
  } catch {
    return false;
  } finally {
    db?.close();
  }
}

function emitJson(io: CliIO, value: unknown): void {
  io.stdout(JSON.stringify(value) + '\n');
}

const USAGE = `Usage: engram <command> [args] [options]

Commands:
  retain <text>                    Store a memory trace (text from stdin if omitted)
  recall <query>                   Four-way retrieval (query from stdin if omitted)
  reflect                          Run a reflection cycle (--suggest to also
                                   run the procedural-suggestion pass)
  process-extractions              Drain the entity extraction queue
  forget <chunkId>                 Soft-delete a chunk
  supersede <oldChunkId> <newText> Replace a fact (newText from stdin if omitted)
  session <message>                Infer/resume a working memory session (default action)
                                   --action update --session-id <id> [--progress <text>] [--extensions <json>]
                                   --action snapshot --session-id <id>
  queue-stats                      Extraction queue health
  requeue-failed                   Re-queue failed extractions for retry
                                   (--error-like <substring> to filter)
  introspect [subject]             Held state: opinions + observations about a
                                   subject (subject from stdin if omitted; omit
                                   entirely for top held state). No confidence
                                   floor — weakly-held beliefs stay visible.
  embed <text>                     Embed text in the bank's vector space
                                   (text from stdin if omitted)
  context-commit <json>            Commit a task-scoped DecisionArtifact
                                   (JSON from stdin if omitted; see below)
  context-query <refId> <query>    Query artifacts committed under refId
                                   (query from stdin if omitted)
  context-promote <refId>          Promote a task-scoped artifact to durable memory
  suggestions                      List procedural suggestions from reflection
                                   (--status, --kind, --domain, --limit)
  resolve-suggestion <id> <status> Set a suggestion's lifecycle status
                                   (--reason <text>); status reopens via "proposed"

Database:
  --db <path>                      Path to the .engram file (or set ENGRAM_DB)

Common options:
  --json                           Emit the raw method return as JSON to stdout
  --ollama-url <url>               Ollama endpoint (default: http://localhost:11434)
  --use-ollama-embeddings          Use Ollama for embeddings
  --reflect-model <model>          LLM for extraction + reflection
  --generation-endpoint <url>      OpenAI-compatible generation endpoint
  --generation-model <model>       Model for the OpenAI-compatible endpoint
  --generation-api-key <key>       API key for the generation endpoint
  --anthropic-api-key <key>        Anthropic API key (Claude generation)
  --anthropic-model <model>        Anthropic model

retain / supersede write options:
  --memory-type <world|experience|observation|opinion>
  --source <id>  --context <tag>
  --source-type <user_stated|inferred|external_doc|tool_result|agent_generated>
  --trust-score <0..1>  --event-time <iso8601>  --temporal-label <text>
  --supersedes <chunkId>             Explicitly replace an outdated fact

recall options:
  --top-k <n>  --strategies <semantic,keyword,graph,temporal>
  --memory-types <world,experience,...>  --min-trust <0..1>
  --after <iso8601>  --before <iso8601>
  --[no-]opinions  --[no-]observations
  --min-score <0..1>               Drop results below this weighted score
  --insight-min-score <0..1>       Cosine floor for attached opinions/observations (default 0.45)
  --explain-scores                 Include a strategyScores breakdown per result
                                   (results[0] is best-in-highest-tier, not
                                   best-overall; re-sort by score for that)
  --decay-half-life-days <n>       Recency decay half-life in days (default:
                                   180). 0 disables decay (long-continuity recall)

session options:
  --max-active <n>  --threshold <0..1>          (action=resume, the default)
  --action <resume|update|snapshot>
  --session-id <id>                             (required for update/snapshot)
  --progress <text>  --extensions <json>         (action=update)

process-extractions options:
  --batch-size <n>

introspect options:
  --min-confidence <0..1>          Opinion confidence floor (default 0 — none)
  --limit <n>                      Max opinions and observations, each (default 20)
  --no-opinions  --no-observations

embed options:
  --mode <query|document>          query applies the search prefix for
                                   asymmetric models (default: query)

context-commit JSON payload fields (all but decision optional):
  decision, rationale, scoredOptions ([{option,score}]), confidence (0..1),
  refsToSource (string[]), domain, agentId, parentRefId, ttlMs
context-commit options:
  --parent-ref-id <id>  --ttl-ms <n>             (override payload's parentRefId/ttlMs)

context-query options:
  --max-chars <n>                  Character budget for returned artifacts (default: 4000)

reflect options:
  --suggest                        Also run the procedural-suggestion pass
  --unsafe-opinions                 Disable safe default belief evidence gates
  --opinion-min-evidence <n>  --opinion-min-distinct-days <n>
  --opinion-min-distinct-sources <n>
  --no-counter-evidence             Disable the default active audit
  --counter-evidence-top-k <n>  --counter-evidence-max-contradiction-ratio <0..1>
  --counter-evidence-on-reinforce  --counter-evidence-fail-open

suggestions options:
  --status <proposed|accepted|dismissed|implemented>
  --kind <skill|rule|workflow|config>  --domain <tag>  --limit <n>

resolve-suggestion options:
  --reason <text>                  Freeform reason recorded on the suggestion
                                   and its journal entry

Exit codes: 0 success · 2 not-found · 1 error
`;

// ─── Main dispatch ────────────────────────────────────────────────────────────

/**
 * Run the CLI. Returns the process exit code (the entry point calls
 * process.exit with it). Injectable IO + option overrides keep this testable
 * in-process without spawning a subprocess.
 */
export async function runCli(
  argv: string[],
  io: CliIO = defaultIo,
  optionOverrides: Partial<EngramOptions> = {},
): Promise<number> {
  const args = parseArgs(argv);

  if (!args.command || args.command === 'help' || args.bools.has('--help')) {
    io.stderr(USAGE);
    return args.command ? EXIT_OK : EXIT_ERROR;
  }

  const dbPath = resolveDbPath(args, process.env);
  if (!dbPath) {
    io.stderr('engram: no database path. Pass --db <path> or set ENGRAM_DB.\n');
    return EXIT_ERROR;
  }

  let engramOptions: EngramOptions;
  try {
    engramOptions = { ...buildEngramOptions(args), ...optionOverrides };
  } catch (err) {
    // e.g. --anthropic-api-key without --anthropic-model (no default model).
    io.stderr(`engram: ${errMessage(err)}\n`);
    return EXIT_ERROR;
  }

  let engram: Engram;
  try {
    engram = await Engram.open(dbPath, engramOptions);
  } catch (err) {
    io.stderr(`engram: failed to open ${dbPath}: ${errMessage(err)}\n`);
    return EXIT_ERROR;
  }

  io.stderr(`[engram] ${args.command} on ${dbPath}\n`);

  const json = args.bools.has('--json');

  try {
    return await dispatch(args, engram, dbPath, io, json, engramOptions);
  } catch (err) {
    io.stderr(`engram: ${errMessage(err)}\n`);
    return EXIT_ERROR;
  } finally {
    engram.close();
  }
}

/**
 * Preflight the generation model for reflect/extract subcommands. Only the
 * default Ollama path is preflighted: resolve the model (no default → error)
 * and confirm the host actually serves it BEFORE running, so a misconfigured or
 * unserved model exits non-zero here at startup rather than 404-ing mid-run.
 * Skipped when the effective provider isn't Ollama's /api/tags surface — an
 * injected generator (programmatic/test), Anthropic, or an OpenAI-compatible
 * endpoint each validate on use instead.
 */
async function preflightGenerationModel(
  args: ParsedArgs,
  engramOptions: EngramOptions,
  io: CliIO,
): Promise<number> {
  if (
    engramOptions.generator ||
    engramOptions.anthropicGeneration ||
    engramOptions.generationEndpoint
  ) {
    return EXIT_OK;
  }

  let spec;
  try {
    spec = resolveModelSpec({
      role: 'reflect',
      explicitModel: args.values.get('--reflect-model'),
      explicitHost: args.values.get('--ollama-url'),
    });
  } catch (err) {
    io.stderr(`engram: ${errMessage(err)}\n`);
    return EXIT_ERROR;
  }

  const pf = await preflightModel(spec);
  if (!pf.ok) {
    io.stderr(`engram: ${formatPreflightFailure(pf)}\n`);
    return EXIT_ERROR;
  }
  return EXIT_OK;
}

async function dispatch(
  args: ParsedArgs,
  engram: Engram,
  dbPath: string,
  io: CliIO,
  json: boolean,
  engramOptions: EngramOptions,
): Promise<number> {
  switch (args.command) {
    case 'retain': {
      const text = await resolveText(args.positionals[0], io);
      if (!text) return missingArg(io, 'text');
      const result = await engram.retain(text, buildRetainOptions(args));
      if (json) emitJson(io, result);
      else io.stdout(formatRetain(result));
      return EXIT_OK;
    }

    case 'recall': {
      const query = await resolveText(args.positionals[0], io);
      if (!query) return missingArg(io, 'query');
      const result = await engram.recall(query, buildRecallOptions(args));
      if (json) emitJson(io, result);
      else io.stdout(formatRecall(result));
      return EXIT_OK;
    }

    case 'reflect': {
      const pf = await preflightGenerationModel(args, engramOptions, io);
      if (pf !== EXIT_OK) return pf;
      const result = await engram.reflect(buildReflectOptions(args));
      if (json) emitJson(io, result);
      else io.stdout(formatReflect(result));
      return EXIT_OK;
    }

    case 'process-extractions': {
      const pf = await preflightGenerationModel(args, engramOptions, io);
      if (pf !== EXIT_OK) return pf;
      const batchSize = asNumber(args.values.get('--batch-size'));
      const result = await engram.processExtractions(
        batchSize !== undefined
          ? Math.max(1, Math.floor(batchSize))
          : undefined,
      );
      if (json) emitJson(io, result);
      else
        io.stdout(
          `extractions: processed=${result.processed} failed=${result.failed}\n`,
        );
      return EXIT_OK;
    }

    case 'forget': {
      const chunkId = args.positionals[0];
      if (!chunkId) return missingArg(io, 'chunkId');
      const forgotten = await engram.forget(chunkId);
      if (json) emitJson(io, { forgotten });
      else
        io.stdout(
          forgotten ? `forgot ${chunkId}\n` : `not found: ${chunkId}\n`,
        );
      return forgotten ? EXIT_OK : EXIT_NOT_FOUND;
    }

    case 'supersede': {
      const oldChunkId = args.positionals[0];
      if (!oldChunkId) return missingArg(io, 'oldChunkId');
      const newText = await resolveText(args.positionals[1], io);
      if (!newText) return missingArg(io, 'newText');
      if (!chunkExists(dbPath, oldChunkId)) {
        io.stderr(`engram: chunk not found: ${oldChunkId}\n`);
        return EXIT_NOT_FOUND;
      }
      const result = await engram.supersede(
        oldChunkId,
        newText,
        buildRetainOptions(args),
      );
      if (json) emitJson(io, result);
      else io.stdout(`superseded ${oldChunkId} -> ${result.chunkId}\n`);
      return EXIT_OK;
    }

    case 'session': {
      const action = args.values.get('--action');

      if (action === 'update' || action === 'snapshot') {
        const sessionId = args.values.get('--session-id');
        if (!sessionId) return missingArg(io, 'sessionId');
        const existing = engram.getWorkingSession(sessionId);
        if (!existing) {
          io.stderr(`engram: working memory session not found: ${sessionId}\n`);
          return EXIT_NOT_FOUND;
        }

        if (action === 'update') {
          const updates: Record<string, unknown> = {};
          const progress = args.values.get('--progress');
          if (progress !== undefined) updates.progress = progress;
          const extensionsRaw = args.values.get('--extensions');
          if (extensionsRaw !== undefined) {
            try {
              const parsed = JSON.parse(extensionsRaw);
              if (
                parsed &&
                typeof parsed === 'object' &&
                !Array.isArray(parsed)
              ) {
                Object.assign(updates, parsed);
              }
            } catch {
              // malformed --extensions JSON — fall back to progress-only update
            }
          }
          await engram.updateWorkingSession(sessionId, updates);
          const state = engram.getWorkingSession(
            sessionId,
          ) as WorkingMemoryState;
          if (json) emitJson(io, state);
          else
            io.stdout(
              `session ${sessionId} updated (updated_at=${state.updated_at})\n`,
            );
          return EXIT_OK;
        }

        // action === 'snapshot'
        const result = await engram.snapshotWorkingSession(sessionId);
        const output = { sessionId, ...result };
        if (json) emitJson(io, output);
        else
          io.stdout(`session ${sessionId} snapshotted -> ${result.chunkId}\n`);
        return EXIT_OK;
      }

      const message = await resolveText(args.positionals[0], io);
      if (!message) return missingArg(io, 'message');
      const result = await engram.inferWorkingSession(message, {
        maxActive: asNumber(args.values.get('--max-active')),
        threshold: asNumber(args.values.get('--threshold')),
      });
      if (json) emitJson(io, result);
      else io.stdout(formatSession(result));
      return EXIT_OK;
    }

    case 'queue-stats': {
      const stats = engram.getQueueStats();
      if (json) emitJson(io, stats);
      else io.stdout(formatQueueStats(stats));
      return EXIT_OK;
    }

    case 'requeue-failed': {
      const result = engram.requeueFailedExtractions({
        errorLike: args.values.get('--error-like'),
      });
      if (json) emitJson(io, result);
      else io.stdout(`requeued ${result.requeued} failed item(s)\n`);
      return EXIT_OK;
    }

    case 'introspect': {
      // Subject is optional — an empty subject means "top held state overall".
      const subject = (await resolveText(args.positionals[0], io)) || undefined;
      const limit = asNumber(args.values.get('--limit'));
      const result = engram.introspect(subject, {
        minConfidence: asNumber(args.values.get('--min-confidence')),
        limit: limit !== undefined ? Math.max(1, Math.floor(limit)) : undefined,
        includeOpinions: args.bools.has('--no-opinions') ? false : undefined,
        includeObservations: args.bools.has('--no-observations')
          ? false
          : undefined,
      });
      if (json) emitJson(io, result);
      else io.stdout(formatIntrospect(result));
      return EXIT_OK;
    }

    case 'embed': {
      const text = await resolveText(args.positionals[0], io);
      if (!text) return missingArg(io, 'text');
      // Same clamp as the engram_embed MCP handler: anything not 'document' is 'query'.
      const mode =
        args.values.get('--mode') === 'document' ? 'document' : 'query';
      const vec = await engram.embedForMode(text, mode);
      const result = { embedding: Array.from(vec), dimensions: vec.length };
      if (json) emitJson(io, result);
      else io.stdout(`embedded: ${result.dimensions} dims (${mode} mode)\n`);
      return EXIT_OK;
    }

    case 'context-commit': {
      const raw = await resolveText(args.positionals[0], io);
      if (!raw) return missingArg(io, 'artifact JSON');
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        io.stderr('engram: context-commit payload must be valid JSON\n');
        return EXIT_ERROR;
      }
      const artifact = buildDecisionArtifactFromPayload(payload);
      if (!artifact) return missingArg(io, 'decision');

      const scope: TaskScope = {};
      const parentRefId =
        args.values.get('--parent-ref-id') ??
        (typeof payload.parentRefId === 'string'
          ? payload.parentRefId
          : undefined);
      if (parentRefId) scope.parent = { id: parentRefId, scope: 'task' };
      const ttlMs =
        asNumber(args.values.get('--ttl-ms')) ??
        (typeof payload.ttlMs === 'number' ? payload.ttlMs : undefined);
      if (ttlMs !== undefined) scope.ttlMs = ttlMs;

      const ref = await engram.commitContext(artifact, scope);
      if (json) emitJson(io, ref);
      else io.stdout(`committed context ${ref.id}\n`);
      return EXIT_OK;
    }

    case 'context-query': {
      const refId = args.positionals[0];
      if (!refId) return missingArg(io, 'refId');
      const query = await resolveText(args.positionals[1], io);
      if (!query) return missingArg(io, 'query');
      const maxChars = asNumber(args.values.get('--max-chars'));
      const budget: TokenBudget | undefined =
        maxChars !== undefined && maxChars > 0
          ? { maxChars: Math.floor(maxChars) }
          : undefined;
      const slice = await engram.queryContext(
        { id: refId, scope: 'task' },
        query,
        budget,
      );
      if (json) emitJson(io, slice);
      else io.stdout(formatContextSlice(slice));
      return EXIT_OK;
    }

    case 'context-promote': {
      const refId = args.positionals[0];
      if (!refId) return missingArg(io, 'refId');
      try {
        await engram.promoteContext({ id: refId, scope: 'task' });
        if (json) emitJson(io, { promoted: true });
        else io.stdout(`promoted ${refId} to durable\n`);
        return EXIT_OK;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (!message.includes('not found or already expired')) throw err;
        if (json) emitJson(io, { promoted: false });
        else io.stdout(`not found: ${refId}\n`);
        return EXIT_NOT_FOUND;
      }
    }

    case 'suggestions': {
      const statusArg = args.values.get('--status');
      const kindArg = args.values.get('--kind');
      const limit = asNumber(args.values.get('--limit'));
      const result = engram.suggestions({
        status: VALID_SUGGESTION_STATUSES.has(statusArg ?? '')
          ? (statusArg as SuggestionStatus)
          : undefined,
        kind: VALID_SUGGESTION_KINDS.has(kindArg ?? '')
          ? (kindArg as SuggestionKind)
          : undefined,
        domain: args.values.get('--domain'),
        limit: limit !== undefined ? Math.max(1, Math.floor(limit)) : undefined,
      });
      if (json) emitJson(io, result);
      else io.stdout(formatSuggestions(result));
      return EXIT_OK;
    }

    case 'resolve-suggestion': {
      const suggestionId = args.positionals[0];
      if (!suggestionId) return missingArg(io, 'suggestionId');
      const statusArg = args.positionals[1];
      if (!statusArg) return missingArg(io, 'status');
      if (!VALID_SUGGESTION_STATUSES.has(statusArg)) {
        io.stderr(
          'engram: status must be one of proposed|accepted|dismissed|implemented\n',
        );
        return EXIT_ERROR;
      }
      const status = statusArg as SuggestionStatus;
      const resolved = engram.resolveSuggestion(
        suggestionId,
        status,
        args.values.get('--reason'),
      );
      if (!resolved) {
        io.stderr(`engram: suggestion not found: ${suggestionId}\n`);
        return EXIT_NOT_FOUND;
      }
      if (json) emitJson(io, { suggestionId, status, resolved });
      else io.stdout(`resolved ${suggestionId} -> ${status}\n`);
      return EXIT_OK;
    }

    default: {
      io.stderr(`engram: unknown command "${args.command}"\n\n${USAGE}`);
      return EXIT_ERROR;
    }
  }
}

/** Use the positional if present, otherwise read from stdin. */
async function resolveText(
  positional: string | undefined,
  io: CliIO,
): Promise<string> {
  if (positional !== undefined && positional.trim() !== '') return positional;
  return (await io.readStdin()).trim();
}

function missingArg(io: CliIO, name: string): number {
  io.stderr(`engram: missing required argument: ${name}\n`);
  return EXIT_ERROR;
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ─── Entry point ──────────────────────────────────────────────────────────────
// Only auto-run when invoked directly (engram-cli), not when imported by tests.

const invokedDirectly =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (invokedDirectly) {
  runCli(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      process.stderr.write(`engram: fatal: ${errMessage(err)}\n`);
      process.exit(EXIT_ERROR);
    });
}
