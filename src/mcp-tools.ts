// =============================================================================
// mcp-tools.ts - MCP Tool Definitions
//
// Exposes Engram operations as Model Context Protocol tools for agent
// integration. Provides:
//   - ENGRAM_TOOLS: static tool schema array (register with your MCP server)
//   - createEngramToolHandler: factory that binds tools to an Engram instance
//
// Usage:
//   import { ENGRAM_TOOLS, createEngramToolHandler } from 'engram/mcp-tools';
//   const handle = createEngramToolHandler(mira);
//   server.registerTools(ENGRAM_TOOLS);
//   server.onToolCall((name, input) => handle(name, input));
// =============================================================================

import type { Engram } from './engram.js';
import type { RetainOptions } from './retain.js';
import type { RecallOptions } from './recall.js';

// =============================================================================
// Tool Schemas (JSON Schema — MCP spec compliant)
// =============================================================================

export const ENGRAM_TOOLS = [
  {
    name: 'engram_retain' as const,
    description: 'Store a memory trace in the agent\'s engram. Fast path — no LLM involved, ~5ms.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        text: {
          type: 'string',
          description: 'The memory content to store',
        },
        memoryType: {
          type: 'string',
          enum: ['world', 'experience', 'observation', 'opinion'],
          description: 'world=facts about the world, experience=agent\'s own actions, observation=synthesized knowledge, opinion=belief with confidence',
        },
        source: {
          type: 'string',
          description: 'Source identifier: conversation ID, filename, tool name, etc.',
        },
        context: {
          type: 'string',
          description: 'Freeform context tag (e.g. "infrastructure", "career", "mission:VALOR-042")',
        },
        sourceType: {
          type: 'string',
          enum: ['user_stated', 'inferred', 'external_doc', 'tool_result', 'agent_generated'],
          description: 'Provenance classification. Affects trust weighting during recall.',
        },
        trustScore: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description: 'Trust level 0.0–1.0. user_stated typically 0.8–0.9, inferred 0.4–0.6.',
        },
        eventTime: {
          type: 'string',
          description: 'ISO 8601 timestamp for when the event/fact occurred (may differ from storage time)',
        },
        temporalLabel: {
          type: 'string',
          description: 'Human-readable time reference, e.g. "last spring", "Q4 2025"',
        },
      },
      required: ['text'],
    },
  },

  {
    name: 'engram_recall' as const,
    description: 'Retrieve relevant memories using semantic, keyword, graph, and temporal search fused via Reciprocal Rank Fusion. Returns ranked chunks plus synthesized opinions and observations.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: {
          type: 'string',
          description: 'Natural language query',
        },
        topK: {
          type: 'number',
          description: 'Max results to return (default: 10)',
        },
        strategies: {
          type: 'array',
          items: { type: 'string', enum: ['semantic', 'keyword', 'graph', 'temporal'] },
          description: 'Retrieval strategies to use. Omit to use all four.',
        },
        memoryTypes: {
          type: 'array',
          items: { type: 'string', enum: ['world', 'experience', 'observation', 'opinion'] },
          description: 'Filter to specific memory types. Omit to search all.',
        },
        minTrust: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description: 'Minimum trust score filter (default: 0.0)',
        },
        after: {
          type: 'string',
          description: 'ISO 8601 date — only include facts after this date',
        },
        before: {
          type: 'string',
          description: 'ISO 8601 date — only include facts before this date',
        },
        includeOpinions: {
          type: 'boolean',
          description: 'Include relevant opinions in response (default: true)',
        },
        includeObservations: {
          type: 'boolean',
          description: 'Include synthesized observations in response (default: true)',
        },
      },
      required: ['query'],
    },
  },

  {
    name: 'engram_reflect' as const,
    description: 'Run a reflection cycle: processes unreflected memories through the LLM to synthesize observations and update opinions. Requires Ollama. Typically run on a schedule rather than per-turn.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },

  {
    name: 'engram_process_extractions' as const,
    description: 'Drain the entity extraction queue to build the knowledge graph from retained chunks. Requires Ollama. Run after retain() calls to enable graph-based recall.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        batchSize: {
          type: 'number',
          description: 'Number of chunks to process per call (default: 10)',
        },
      },
    },
  },

  {
    name: 'engram_forget' as const,
    description: 'Soft-delete a memory chunk. The chunk is excluded from recall but remains in the database for audit.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        chunkId: {
          type: 'string',
          description: 'The chunk ID to deactivate (returned by engram_retain)',
        },
      },
      required: ['chunkId'],
    },
  },

  {
    name: 'engram_supersede' as const,
    description: 'Replace an outdated fact with new text. The old chunk is soft-deleted and linked to the new one. Use when correcting information.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        oldChunkId: {
          type: 'string',
          description: 'The chunk ID of the fact being replaced',
        },
        newText: {
          type: 'string',
          description: 'The updated fact text',
        },
        memoryType: {
          type: 'string',
          enum: ['world', 'experience', 'observation', 'opinion'],
        },
        source: { type: 'string' },
        context: { type: 'string' },
        sourceType: {
          type: 'string',
          enum: ['user_stated', 'inferred', 'external_doc', 'tool_result', 'agent_generated'],
        },
        trustScore: { type: 'number', minimum: 0, maximum: 1 },
      },
      required: ['oldChunkId', 'newText'],
    },
  },
] as const;

export type EngramToolName = typeof ENGRAM_TOOLS[number]['name'];

// =============================================================================
// MCP Response Types
// =============================================================================

interface McpContent {
  type: 'text';
  text: string;
}

export interface McpToolResult {
  content: McpContent[];
  isError?: boolean;
}

// =============================================================================
// Handler Factory
// =============================================================================

/**
 * Bind the ENGRAM_TOOLS to a live Engram instance.
 * Returns a handler function suitable for passing to your MCP server's onToolCall.
 */
export function createEngramToolHandler(engram: Engram) {
  return async function handleTool(
    name: EngramToolName,
    input: Record<string, unknown>
  ): Promise<McpToolResult> {
    try {
      switch (name) {
        case 'engram_retain': {
          const { text, ...opts } = input as unknown as { text: string } & RetainOptions;
          const result = await engram.retain(text, opts);
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }

        case 'engram_recall': {
          const { query, ...opts } = input as unknown as { query: string } & RecallOptions;
          const result = await engram.recall(query, opts);
          return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
        }

        case 'engram_reflect': {
          const result = await engram.reflect();
          return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
        }

        case 'engram_process_extractions': {
          const batchSize = typeof input.batchSize === 'number' ? input.batchSize : 10;
          const result = await engram.processExtractions(batchSize);
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }

        case 'engram_forget': {
          const chunkId = input.chunkId as string;
          const forgotten = await engram.forget(chunkId);
          return { content: [{ type: 'text', text: JSON.stringify({ forgotten }) }] };
        }

        case 'engram_supersede': {
          const { oldChunkId, newText, ...opts } = input as unknown as { oldChunkId: string; newText: string } & RetainOptions;
          const result = await engram.supersede(oldChunkId, newText, opts);
          return { content: [{ type: 'text', text: JSON.stringify(result) }] };
        }
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        content: [{ type: 'text', text: `engram tool error: ${message}` }],
        isError: true,
      };
    }
  };
}
