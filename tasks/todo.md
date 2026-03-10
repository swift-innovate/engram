# Engram Build Plan
## Goal: Go from 4 loose TypeScript files → a buildable, typed library with unified API

---

## Tasks

- [x] 1. Create `package.json` — deps: better-sqlite3, sqlite-vec, typescript, tsx, @types
- [x] 2. Create `tsconfig.json` — ESM, NodeNext, strict, rootDir: src, outDir: dist
- [x] 3. Move source files to `src/` — retain.ts, recall.ts, reflect.ts, schema.sql
- [x] 4. Create `src/engram.ts` — unified `Engram` class (static create/open + retain/recall/reflect/processExtractions)
- [x] 5. Create `src/mcp-tools.ts` — MCP tool definitions for retain, recall, reflect
- [x] 6. Create `examples/basic-usage.ts`
- [x] 7. `npm install` — 49 packages, 0 vulnerabilities
- [x] 8. `npx tsc --noEmit` — zero errors
- [x] 9. Verify reflect module imports cleanly — note: CLI entry point (`import.meta.url` check) doesn't trigger on Windows due to path format differences; works on Unix/Mac as written

---

## Key decisions

### Schema initialization (pending user input)
Option A: Read schema.sql from disk at runtime (schema lives in one canonical place)
Option B: Inline CREATE TABLE statements in engram.ts (no file I/O, but schema duplication risk)

### reflect.ts connection pattern
Keep existing pattern: reflect() opens/closes its own DB connection.
Engram class holds a persistent connection for retain/recall (WAL mode supports this).

---

## Review

- [x] Tests: 52/52 passing (retain × 16, recall × 18, reflect × 9, engram × 9)
- [x] Two source bugs found and fixed via tests:
  - `retainBatch()` was queueing observation/opinion types for extraction (logic error)
  - `reflect.ts` had a dead `updateObs` prepared statement with invalid SQL that killed all reflect cycles
- [x] TypeScript compiles with zero errors (`tsc --noEmit`)
- [x] Full build produces `dist/` with all JS, `.d.ts`, `.map`, and `schema.sql`
- [x] `Engram.create()` and `Engram.open()` implemented with idempotent schema bootstrap
- [x] MCP tool schemas follow JSON Schema spec; handler factory pattern allows any MCP server to integrate
- [x] 49 packages installed, 0 vulnerabilities
- [!] reflect CLI entry point: module imports cleanly; entry condition `import.meta.url` check is Windows-incompatible (pre-existing, needs `pathToFileURL(process.argv[1]).href` fix for Windows)
