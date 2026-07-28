# Engram + OpenClaw Integration Guide

OpenClaw is a multi-channel AI gateway. This guide covers integrating Engram as OpenClaw's memory backend, replacing the default flat-file FTS system with semantic four-strategy retrieval.

**Status:** Production-verified via Tracer agent stress test (2026-03-24) — 16/17 tests passed.

## Architecture

```
OpenClaw Agent (Tracer, etc.)
    │
    ▼
Plugin: memory-engram
    ├─ memory_search tool (replaces built-in FTS)
    ├─ memory_get tool (chunk retrieval by ID/path)
    └─ Markdown sync service (workspace/memory/*.md → Engram)
    │
    ▼
EngramClient (mcporter subprocess bridge)
    ├─ recall(query, options)
    ├─ retain(text, options)
    └─ exec(["npx", "mcporter", "call", ...])
    │
    ▼
mcporter (stdio transport)
    │
    ▼
Engram MCP Server (engram-mcp)
    ├─ engram_recall  (4-strategy retrieval)
    ├─ engram_retain  (store with context)
    ├─ engram_session (working memory)
    ├─ engram_reflect (synthesis)
    ├─ engram_process_extractions (knowledge graph)
    ├─ engram_forget  (soft-delete)
    └─ engram_supersede (replace outdated fact)
    │
    ▼
<agent>.engram (SQLite)
    with local or Ollama embeddings
```

## Prerequisites

- **Node.js** >= 20.0.0
- **OpenClaw** installed and configured
- **mcporter** installed globally: `npm install -g mcporter`
- **Engram** built: `cd /path/to/engram && npm install && npm run build`
- **Ollama** (optional — required for extraction, reflection, and Ollama embeddings)

## Step-by-Step Setup

### 1. Configure mcporter

Create or update `<openclaw-workspace>/config/mcporter.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": [
        "/path/to/engram/dist/mcp-server.js",
        "/path/to/openclaw/workspace/memory/agent.engram",
        "--use-ollama-embeddings",
        "--ollama-url", "http://localhost:11434"
      ],
      "transport": "stdio"
    }
  }
}
```

**Notes:**
- Replace paths with your actual Engram install and workspace locations.
- Omit `--use-ollama-embeddings` to use local Transformers.js embeddings (default, no Ollama needed for retain/recall).
- If using Herd instead of Ollama, set `--ollama-url http://localhost:40114`.

### 2. Verify mcporter can reach Engram

```bash
cd <openclaw-workspace>
npx mcporter call engram.engram_recall query="test"
```

Should return JSON with `results`, `strategiesUsed`, `totalCandidates`.

### 3. Install the memory-engram plugin

Create the plugin directory in your OpenClaw workspace:

```
workspace/plugins/memory-engram/
├── openclaw.plugin.json
├── index.ts
└── src/
    ├── engram-client.ts
    ├── result-adapter.ts
    └── markdown-sync.ts
```

**`openclaw.plugin.json`:**

```json
{
  "id": "memory-engram",
  "kind": "memory",
  "defaultConfig": {
    "mcporterServer": "engram",
    "mcporterConfigPath": "config/mcporter.json"
  }
}
```

### 4. Register in openclaw.json

Add the plugin to your `openclaw.json` under the plugins section:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-engram"
    }
  }
}
```

This replaces the default `memory-core` plugin. Only one memory plugin can be active at a time (exclusive slot).

### 5. (Optional) Configure SOUL.md for direct tool preference

If your agent should prefer Engram's native tools over the OpenClaw wrapper:

```markdown
## Memory Tool Priority

Use Engram tools (engram_recall, engram_retain, engram_session)
instead of the built-in memory_search tool.

- To search memory: use engram_recall (not memory_search)
- To store a fact: use engram_retain
- To manage working session: use engram_session
- To build knowledge graph: use engram_process_extractions
- To synthesize understanding: use engram_reflect
```

## Plugin Behavior

### Memory Tools

The plugin registers two tools that replace OpenClaw's built-in memory tools:

**`memory_search`** — Semantic search via Engram's four-strategy retrieval.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (required) | Search query |
| `maxResults` | number | Limit results (default: 10) |
| `minScore` | number | Confidence threshold 0-1 (default: 0) |
| `after` | string | ISO date filter — only results after this date |
| `before` | string | ISO date filter — only results before this date |

**`memory_get`** — Retrieve specific chunks by ID or source path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string (required) | Source reference or chunk ID (`chk-xxx` or `engram://chk-xxx`) |
| `from` | number | Start line (for file-based sources) |
| `lines` | number | Number of lines (for file-based sources) |

### Response Format

`memory_search` returns:

```json
{
  "results": [{
    "path": "engram://chk-xxx",
    "text": "Tom prefers Terraform with the bpg provider",
    "score": 0.041,
    "source": "conversation:session-123",
    "chunkId": "chk-xxx",
    "memoryType": "world",
    "trustScore": 0.9
  }],
  "provider": "engram",
  "mode": "semantic+keyword+graph",
  "totalCandidates": 42,
  "opinions": [],
  "observations": []
}
```

### Markdown Sync

The plugin automatically ingests `workspace/memory/*.md` into Engram:

- **Initial sync** on plugin load
- **Periodic sync** every 5 minutes (with 30s cooldown between runs)
- **Lazy sync** before each `memory_search` call
- Files are split by H2 sections; each section becomes a chunk
- State tracked in `memory/.engram-sync-manifest.json` (file hashes + chunk IDs)
- Retained with `memoryType: "experience"`, `sourceType: "external_doc"`, `trustScore: 0.85`

### Prompt Section Injection

The plugin injects a system prompt section guiding the LLM to use `memory_search` before answering questions about prior work, decisions, dates, people, preferences, or todos. Citations mode is respected (`auto` or `off`).

## Known Issues & Mitigations

### ~10s query latency

**Cause:** mcporter subprocess cold-start. Each `npx mcporter call` spawns a new Node process, loads Engram, loads the embedding model, executes the query, then exits.

**Mitigations:**
- Use `mcporter daemon start` for persistent connections (avoids cold-start per call)
- Replace mcporter subprocess with direct Engram library import in the plugin
- Accept the latency for async/background use cases

### SQLite lock under concurrent writes

**Cause:** Multiple parallel mcporter calls compete for the SQLite write lock.

**Mitigations:**
- Serialize writes (don't call `engram_retain` in parallel)
- Engram uses WAL mode, which handles concurrent readers + one writer
- Add retry-with-backoff in the plugin's EngramClient for `SQLITE_BUSY` errors

### Plugin loads multiple times

**Cause:** OpenClaw registers the plugin once per agent/channel combination.

**Mitigation:** The 30s sync cooldown prevents duplicate markdown ingestion. Each plugin instance creates its own EngramClient but they all target the same `.engram` file.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `memory_search` returns `{"results":[], "provider":"none"}` | Plugin not registered or wrong slot | Check `openclaw.json` has `plugins.slots.memory: "memory-engram"` |
| `Engram mcporter call failed` | mcporter can't reach Engram server | Run `npx mcporter call engram.engram_recall query="test"` manually to debug |
| Empty results despite retained data | Embedding model mismatch or not loaded | Ensure `--use-ollama-embeddings` flag matches how data was originally retained |
| `SQLITE_BUSY` errors | Concurrent writes | Serialize retain calls; consider mcporter daemon mode |
| Sync manifest grows stale | File hashes don't match | Delete `memory/.engram-sync-manifest.json` to force full re-sync |
| JSON parse errors from mcporter | mcporter status output mixed with JSON | Plugin strips non-JSON prefix; update mcporter if format changed |

## Production Config Reference

This is the production mcporter config used by the Tracer agent:

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": [
        "/mnt/g/projects/sit/engram/dist/mcp-server.js",
        "/home/tom/.openclaw/workspace/memory/tracer.engram",
        "--use-ollama-embeddings",
        "--ollama-url", "http://100.110.46.23:40114"
      ],
      "transport": "stdio"
    }
  }
}
```

The Ollama URL points to a Tailscale-accessible homelab node running Herd (Ollama-compatible API on port 40114).

## Migrating OpenClaw Memory to Engram

Engram ships with a CLI tool for importing existing OpenClaw memory directories into `.engram` files. This is a one-time migration — it reads your `memory/` directory, classifies files by path pattern, parses markdown into chunks, and batch-retains everything with appropriate trust scores.

### Quick Start

```bash
cd tools/openclaw-import
npm install

# Preview what would be imported
npx tsx src/index.ts -i /path/to/openclaw/memory -o ./agent.engram --dry-run

# Run the import (local embeddings, no Ollama needed)
npx tsx src/index.ts -i /path/to/openclaw/memory -o ./agent.engram

# With Ollama embeddings
npx tsx src/index.ts -i /path/to/openclaw/memory -o ./agent.engram \
  --use-ollama-embeddings --ollama-url http://localhost:11434
```

### How It Works

1. **Discover** — walks the memory directory for `.md` files, skips non-importable files (logs, JSON, backups)
2. **Classify** — assigns each file a category based on its path (`core/`, `daily/`, `decisions/`, etc.)
3. **Parse** — splits markdown on H2 headings into chunks (50–4000 chars)
4. **Map** — assigns memory type, trust score, and source type per category
5. **Retain** — batch-writes to Engram with deduplication (safe to re-run)

### Category Mapping

| Path Pattern | Category | Memory Type | Trust Score |
|-------------|----------|-------------|-------------|
| `core/*` | core | world | 0.90 |
| `daily/*` | daily | experience | 0.70 |
| `decisions/*` | decision | world | 0.85 |
| `dreams/*-creative.md` | dream | experience | 0.60 |
| `projects/*` | project | world | 0.80 |
| Root `YYYY-MM-DD*.md` | daily | experience | 0.75 |
| Everything else | memo | world | 0.75 |

### Post-Migration

After import, run extraction and reflection to build the knowledge graph:

```bash
# From engram root — process entity extraction queue
npx tsx -e "const {Engram}=await import('./dist/engram.js'); const e=await Engram.open('./agent.engram', {reflectModel:'llama3.1:8b'}); await e.processExtractions(100); e.close()"

# Run reflection to synthesize observations
REFLECT_MODEL=llama3.1:8b npx tsx src/reflect.ts ./agent.engram
```

See [`tools/openclaw-import/README.md`](../tools/openclaw-import/README.md) for full CLI options and architecture details.
