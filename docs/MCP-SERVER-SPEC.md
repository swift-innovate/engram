# Engram MCP Server — Claude Code Spec

**Repo:** `G:\Projects\SIT\engram`
**Goal:** Add a standalone MCP stdio server so OpenClaw (and any MCP-compatible client) can use Engram's memory tools natively.
**Priority:** This is the bridge between Engram and the agent ecosystem. Without it, agents can only use Engram via direct TypeScript imports.

---

## Context

Engram already has:
- `src/mcp-tools.ts` — tool schemas (JSON Schema, MCP-compliant) and a handler factory (`createEngramToolHandler`)
- 7 tools defined: `engram_retain`, `engram_recall`, `engram_reflect`, `engram_process_extractions`, `engram_forget`, `engram_supersede`, `engram_session`
- The handler factory takes an `Engram` instance and returns an async function that dispatches tool calls

What's missing is a **standalone process** that:
1. Opens an `.engram` file
2. Starts an MCP server over stdio (JSON-RPC)
3. Registers the tools
4. Routes incoming tool calls through the existing handler

---

## Implementation

### 1. Add MCP SDK dependency

```bash
npm install @modelcontextprotocol/sdk
```

This is the official MCP TypeScript SDK. It provides `Server`, `StdioServerTransport`, and the protocol types.

### 2. Create `src/mcp-server.ts`

```typescript
#!/usr/bin/env node
// =============================================================================
// mcp-server.ts — Standalone MCP stdio server for Engram
//
// Launches an MCP server over stdin/stdout that exposes Engram's memory
// tools to any MCP-compatible client (OpenClaw, Claude Desktop, Cursor, etc.)
//
// Usage:
//   npx engram-mcp ./path/to/agent.engram
//   npx engram-mcp ./agent.engram --ollama-url http://192.168.1.57:11434
//   npx engram-mcp ./agent.engram --use-ollama-embeddings
//
// OpenClaw config (add to settings or ~/.openclaw/config.json):
//   {
//     "mcpServers": {
//       "engram": {
//         "command": "npx",
//         "args": ["engram-mcp", "/path/to/tracer.engram", "--use-ollama-embeddings", "--ollama-url", "http://192.168.1.57:11434"]
//       }
//     }
//   }
// =============================================================================

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { Engram } from './engram.js';
import { ENGRAM_TOOLS, createEngramToolHandler } from './mcp-tools.js';
import type { EngramToolName } from './mcp-tools.js';

// ─── CLI Args ───────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dbPath = args.find(a => !a.startsWith('--'));

if (!dbPath) {
  console.error('Usage: engram-mcp <path-to-engram-file> [options]');
  console.error('');
  console.error('Options:');
  console.error('  --ollama-url <url>        Ollama endpoint (default: http://localhost:11434)');
  console.error('  --use-ollama-embeddings   Use Ollama for embeddings instead of local Transformers.js');
  console.error('  --reflect-model <model>   LLM for extraction + reflection (default: llama3.1:8b)');
  process.exit(1);
}

function getArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

const ollamaUrl = getArg('--ollama-url') ?? 'http://localhost:11434';
const useOllamaEmbeddings = args.includes('--use-ollama-embeddings');
const reflectModel = getArg('--reflect-model') ?? 'llama3.1:8b';

// ─── Server Setup ───────────────────────────────────────────────────────────

async function main() {
  // Open the engram file
  const engram = await Engram.open(dbPath!, {
    ollamaUrl,
    useOllamaEmbeddings,
    reflectModel,
  });

  // Create the tool handler bound to this engram instance
  const handleTool = createEngramToolHandler(engram);

  // Create MCP server
  const server = new Server(
    {
      name: 'engram',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools handler
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: ENGRAM_TOOLS.map(tool => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
    })),
  }));

  // Call tool handler
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: toolArgs } = request.params;
    const result = await handleTool(
      name as EngramToolName,
      (toolArgs ?? {}) as Record<string, unknown>
    );
    return result;
  });

  // Graceful shutdown
  process.on('SIGINT', () => {
    engram.close();
    process.exit(0);
  });
  process.on('SIGTERM', () => {
    engram.close();
    process.exit(0);
  });

  // Connect to stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Log to stderr (stdout is reserved for MCP protocol)
  console.error(`[engram-mcp] Serving ${dbPath} via MCP stdio`);
  console.error(`[engram-mcp] Ollama: ${ollamaUrl} | Ollama embeddings: ${useOllamaEmbeddings}`);
  console.error(`[engram-mcp] ${ENGRAM_TOOLS.length} tools registered`);
}

main().catch(err => {
  console.error('[engram-mcp] Fatal:', err);
  process.exit(1);
});
```

### 3. Add `bin` entry to `package.json`

```json
{
  "bin": {
    "engram-mcp": "./dist/mcp-server.js"
  }
}
```

Also add `@modelcontextprotocol/sdk` to dependencies:

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "@xenova/transformers": "^2.17.2",
    "better-sqlite3": "^9.4.3",
    "sqlite-vec": "^0.1.6"
  }
}
```

### 4. Add shebang handling

Make sure `dist/mcp-server.js` has the shebang line after compilation. Add to the build script in `package.json`:

```json
{
  "scripts": {
    "build": "tsc && node -e \"require('fs').copyFileSync('src/schema.sql', 'dist/schema.sql')\" && node -e \"const f='dist/mcp-server.js';const c=require('fs').readFileSync(f,'utf8');if(!c.startsWith('#!')){require('fs').writeFileSync(f,'#!/usr/bin/env node\\n'+c)}\"",
  }
}
```

Or simpler — just add the shebang as a comment at the top of `src/mcp-server.ts` (TypeScript preserves leading comments):

```typescript
#!/usr/bin/env node
```

TypeScript strips this during compilation. Better approach: use a small post-build step or just rely on `npx engram-mcp` which doesn't need the shebang.

### 5. Update `tsconfig.json`

The mcp-server.ts file should be included in compilation. It's already in `src/` so it should compile automatically. Verify it's not excluded.

### 6. Update exports in `package.json`

No change needed — the MCP server is a standalone binary, not a library export. The `bin` entry handles discovery.

---

## OpenClaw Integration

After building, configure OpenClaw to use Engram:

### Option A: Global config (`~/.openclaw/config.json`)

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": [
        "/mnt/g/projects/sit/engram/dist/mcp-server.js",
        "/home/tom/.openclaw/workspace/memory/tracer.engram",
        "--use-ollama-embeddings",
        "--ollama-url", "http://192.168.1.57:11434"
      ]
    }
  }
}
```

### Option B: Workspace-level (if OpenClaw supports it)

Place a `.mcp.json` or similar in `~/.openclaw/workspace/`:

```json
{
  "engram": {
    "command": "node",
    "args": [
      "/mnt/g/projects/sit/engram/dist/mcp-server.js",
      "./memory/tracer.engram",
      "--use-ollama-embeddings",
      "--ollama-url", "http://192.168.1.57:11434"
    ]
  }
}
```

### Verification

After configuring, restart the OpenClaw gateway. In the TUI, ask Tracer:

```
What tools do you have available?
```

It should list the 7 Engram tools. Then test:

```
Use engram_retain to store: "Tom prefers Terraform with the bpg provider for Proxmox IaC"
```

Then:

```
Use engram_recall to search for: "What IaC tools does Tom use?"
```

---

## Tests

Add `tests/mcp-server.test.ts`:

1. Server starts without error when given a valid `.engram` path
2. `ListTools` returns all 7 tool schemas
3. `CallTool` with `engram_retain` stores a chunk and returns a chunkId
4. `CallTool` with `engram_recall` retrieves the stored chunk
5. `CallTool` with `engram_session` creates a new working memory session
6. `CallTool` with `engram_forget` soft-deletes a chunk
7. Server shuts down cleanly on SIGTERM

Use the MCP SDK's test client (`Client` + `StdioClientTransport`) to test end-to-end without needing OpenClaw.

---

## Files Changed

| Action | File | What |
|--------|------|------|
| CREATE | `src/mcp-server.ts` | Standalone MCP stdio server |
| MODIFY | `package.json` | Add `@modelcontextprotocol/sdk` dep, `bin` entry |
| CREATE | `tests/mcp-server.test.ts` | MCP server integration tests |

## Files NOT Changed

| File | Why |
|------|-----|
| `src/mcp-tools.ts` | Already has schemas + handler — the server just wraps it |
| `src/engram.ts` | No changes needed |
| All existing tests | MCP server is additive |

---

## Verification

```bash
npm install
npm run build
npm test                    # existing 124 tests + new MCP tests pass
npm run typecheck           # clean

# Manual test
echo '{}' | node dist/mcp-server.js ./test.engram --use-ollama-embeddings --ollama-url http://192.168.1.57:11434
# Should start without error, log to stderr, then wait for JSON-RPC on stdin
```

---

## Why This Matters

This is the bridge that makes Engram a **platform component** rather than a TypeScript library. Any MCP-compatible client — OpenClaw, Claude Desktop, Cursor, Claude Code, custom agents — can connect to Engram memory without writing any integration code. Just point at the binary and the `.engram` file.

Combined with the working memory primitives from Sprint 1, this gives agents:
- Persistent long-term memory (retain/recall)
- Auto-switching working sessions (engram_session)
- Background knowledge graph building (process_extractions)
- Scheduled learning (reflect)
- Fact correction (supersede)
- Memory cleanup (forget)

All accessible over a standard protocol.
