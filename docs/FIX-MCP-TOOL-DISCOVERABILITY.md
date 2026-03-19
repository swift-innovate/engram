# Engram MCP Onboarding Fix — Tool Discoverability

**Repo:** `G:\Projects\SIT\engram`
**Problem:** Tracer (and future agents) stumbled on the MCP tools because:
1. Parameter names use camelCase (`memoryType`, `trustScore`, `sourceType`) but agents instinctively try snake_case
2. No usage examples in tool descriptions — agents have to guess the invocation syntax
3. `engram_session` schema still references threshold `0.72` (code uses `0.55`)
4. mcporter CLI uses `key=value` syntax that isn't obvious from the JSON Schema alone

**Fix:** Enhance tool descriptions with explicit parameter examples and common patterns.

---

## Changes to `src/mcp-tools.ts`

### 1. Add usage examples to each tool description

Update the `description` field for each tool to include a one-liner example:

```
engram_retain:
  old: "Store a memory trace in the agent's engram. Fast path — no LLM involved, ~5ms."
  new: "Store a memory trace. Fast path (~5ms, no LLM). Parameters use camelCase: text (required), memoryType (world|experience|observation|opinion), sourceType (user_stated|inferred|external_doc|tool_result|agent_generated), trustScore (0.0-1.0). Example: {text: 'Tom prefers Terraform', memoryType: 'world', sourceType: 'user_stated', trustScore: 0.9}"

engram_recall:
  old: "Retrieve relevant memories using semantic, keyword, graph, and temporal search..."
  new: "Retrieve relevant memories via four-strategy search fused with Reciprocal Rank Fusion. Example: {query: 'What tools does Tom use?', topK: 5}. Returns results[], opinions[], observations[]."

engram_session:
  old: "...Call once per incoming user message before the LLM call."
  new: "...Default similarity threshold: 0.55. Example: {message: 'plan the deployment'}. Returns session state + related long-term context."
```

### 2. Fix threshold reference

In the `engram_session` tool schema, update the `threshold` description:

```
old: "Cosine similarity threshold for session matching (default: 0.72)"
new: "Cosine similarity threshold for session matching (default: 0.55). Lower = more aggressive matching, higher = more new sessions."
```

### 3. Add a top-level description to the MCP server

In `src/mcp-server.ts`, when creating the Server, add a description:

```typescript
const server = new Server(
  {
    name: 'engram',
    version: '0.1.0',
    description: 'Memory system for AI agents. All parameters use camelCase (memoryType, trustScore, sourceType). Store facts with engram_retain, search with engram_recall, manage sessions with engram_session.',
  },
  ...
);
```

---

## Verification

After applying changes:

```bash
npm run build
cd ~/.openclaw/workspace
npx mcporter list engram --schema
```

The schema output should show the enhanced descriptions with examples, making it obvious to any agent how to call each tool.

---

Hand this to Claude Code — it's a 5-minute description update, zero behavioral changes, zero test impact.
