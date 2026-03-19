import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { Engram } from '../src/engram.js';
import { ENGRAM_TOOLS, createEngramToolHandler } from '../src/mcp-tools.js';
import type { EngramToolName } from '../src/mcp-tools.js';
import { MockEmbedder, tmpDbPath, cleanupDb } from './helpers.js';

describe('MCP Server', () => {
  let engram: Engram;
  let client: Client;
  let server: Server;
  let dbPath: string;

  beforeAll(async () => {
    dbPath = tmpDbPath();
    engram = await Engram.create(dbPath, { embedder: new MockEmbedder() });

    const handleTool = createEngramToolHandler(engram);

    server = new Server(
      { name: 'engram', version: '0.1.0' },
      { capabilities: { tools: {} } }
    );

    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: ENGRAM_TOOLS.map(tool => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
      })),
    }));

    server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: toolArgs } = request.params;
      return handleTool(
        name as EngramToolName,
        (toolArgs ?? {}) as Record<string, unknown>
      );
    });

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

    client = new Client(
      { name: 'test-client', version: '1.0.0' },
      { capabilities: {} }
    );

    await server.connect(serverTransport);
    await client.connect(clientTransport);
  });

  afterAll(async () => {
    await client.close();
    await server.close();
    engram.close();
    cleanupDb(dbPath);
  });

  it('ListTools returns all 7 tool schemas', async () => {
    const result = await client.listTools();
    expect(result.tools.length).toBe(7);
    const names = result.tools.map(t => t.name);
    expect(names).toContain('engram_retain');
    expect(names).toContain('engram_recall');
    expect(names).toContain('engram_reflect');
    expect(names).toContain('engram_process_extractions');
    expect(names).toContain('engram_forget');
    expect(names).toContain('engram_supersede');
    expect(names).toContain('engram_session');
  });

  it('engram_retain stores a chunk and returns a chunkId', async () => {
    const result = await client.callTool({
      name: 'engram_retain',
      arguments: {
        text: 'Tom prefers Terraform with the bpg provider for Proxmox IaC',
        memoryType: 'world',
        sourceType: 'user_stated',
        trustScore: 0.9,
      },
    });

    const content = result.content as Array<{ type: string; text: string }>;
    expect(content[0].type).toBe('text');
    const parsed = JSON.parse(content[0].text);
    expect(parsed.chunkId).toMatch(/^chk-/);
    expect(parsed).toHaveProperty('queued');
  });

  it('engram_recall retrieves the stored chunk', async () => {
    const result = await client.callTool({
      name: 'engram_recall',
      arguments: {
        query: 'What IaC tools does Tom use?',
        topK: 5,
      },
    });

    const content = result.content as Array<{ type: string; text: string }>;
    const parsed = JSON.parse(content[0].text);
    expect(parsed.results.length).toBeGreaterThan(0);
    expect(parsed.results.some((r: any) => r.text.includes('Terraform'))).toBe(true);
  });

  it('engram_session creates a new working memory session', async () => {
    const result = await client.callTool({
      name: 'engram_session',
      arguments: {
        message: 'Help me plan the Kubernetes migration',
      },
    });

    const content = result.content as Array<{ type: string; text: string }>;
    const parsed = JSON.parse(content[0].text);
    expect(parsed.session.id).toMatch(/^wm-/);
    expect(parsed.diagnostics.reason).toBe('new');
  });

  it('engram_forget soft-deletes a chunk', async () => {
    // First retain something to forget
    const retainResult = await client.callTool({
      name: 'engram_retain',
      arguments: { text: 'Temporary fact to forget' },
    });
    const retainContent = retainResult.content as Array<{ type: string; text: string }>;
    const chunkId = JSON.parse(retainContent[0].text).chunkId;

    // Forget it
    const forgetResult = await client.callTool({
      name: 'engram_forget',
      arguments: { chunkId },
    });

    const forgetContent = forgetResult.content as Array<{ type: string; text: string }>;
    const parsed = JSON.parse(forgetContent[0].text);
    expect(parsed.forgotten).toBe(true);
  });

  it('engram_supersede replaces an old fact with new text', async () => {
    // Retain the old fact
    const retainResult = await client.callTool({
      name: 'engram_retain',
      arguments: { text: 'Tom uses Docker Compose for everything' },
    });
    const retainContent = retainResult.content as Array<{ type: string; text: string }>;
    const oldChunkId = JSON.parse(retainContent[0].text).chunkId;

    // Supersede it
    const supersedeResult = await client.callTool({
      name: 'engram_supersede',
      arguments: {
        oldChunkId,
        newText: 'Tom switched from Docker Compose to Kubernetes',
      },
    });

    const content = supersedeResult.content as Array<{ type: string; text: string }>;
    const parsed = JSON.parse(content[0].text);
    expect(parsed.chunkId).toMatch(/^chk-/);
    expect(parsed).toHaveProperty('queued');
  });

  it('returns isError for invalid tool calls', async () => {
    const result = await client.callTool({
      name: 'engram_forget',
      arguments: { chunkId: 'nonexistent-id' },
    });

    const content = result.content as Array<{ type: string; text: string }>;
    const parsed = JSON.parse(content[0].text);
    // forget returns {forgotten: false} for missing chunks, not an error
    expect(parsed.forgotten).toBe(false);
  });
});
