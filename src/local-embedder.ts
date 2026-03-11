// =============================================================================
// local-embedder.ts — In-Process Embeddings via Transformers.js
//
// Replaces the Ollama HTTP embedding provider with a fully local,
// in-process embedding pipeline. No server, no network, no Docker.
//
// Model downloads once to ~/.cache/xenova/ on first use.
// Subsequent runs load from cache instantly.
//
// Supports query vs document prefixing for models that use it
// (nomic-embed-text uses 'search_query:' / 'search_document:' prefixes).
// =============================================================================

import type { EmbeddingProvider } from './retain.js';

// Lazy-load the pipeline to avoid blocking module import.
// Keyed by model name so multiple LocalEmbedder instances for
// different models each get their own singleton pipeline.
const pipelineCache: Map<string, Promise<unknown>> = new Map();

async function getPipeline(modelName: string, quantized: boolean): Promise<unknown> {
  const cacheKey = `${modelName}:${quantized}`;
  if (!pipelineCache.has(cacheKey)) {
    const { pipeline } = await import('@xenova/transformers');
    pipelineCache.set(cacheKey, pipeline('feature-extraction', modelName, { quantized }));
  }
  return pipelineCache.get(cacheKey)!;
}

// =============================================================================
// Model Registry — known models and their characteristics
// =============================================================================

interface ModelInfo {
  dimensions: number;
  /** Whether the model uses query/document prefix distinction */
  usePrefixes: boolean;
  queryPrefix: string;
  documentPrefix: string;
}

const MODEL_REGISTRY: Record<string, ModelInfo> = {
  'Xenova/nomic-embed-text-v1.5': {
    dimensions: 768,
    usePrefixes: true,
    queryPrefix: 'search_query: ',
    documentPrefix: 'search_document: ',
  },
  'Xenova/all-MiniLM-L6-v2': {
    dimensions: 384,
    usePrefixes: false,
    queryPrefix: '',
    documentPrefix: '',
  },
  'Xenova/snowflake-arctic-embed-m': {
    dimensions: 768,
    usePrefixes: false,
    queryPrefix: '',
    documentPrefix: '',
  },
};

// =============================================================================
// LocalEmbedder — EmbeddingProvider implementation
// =============================================================================

export class LocalEmbedder implements EmbeddingProvider {
  private readonly modelName: string;
  private readonly modelInfo: ModelInfo;
  private readonly quantized: boolean;
  private initialized = false;

  public readonly dimensions: number;

  constructor(
    model: string = 'Xenova/nomic-embed-text-v1.5',
    options?: { quantized?: boolean },
  ) {
    this.modelName = model;
    this.quantized = options?.quantized ?? true;
    this.modelInfo = MODEL_REGISTRY[model] ?? {
      dimensions: 768,
      usePrefixes: false,
      queryPrefix: '',
      documentPrefix: '',
    };
    this.dimensions = this.modelInfo.dimensions;
  }

  /**
   * Initialize the embedding pipeline.
   * Downloads the model on first use (~30MB for quantized nomic).
   * Subsequent calls load from ~/.cache/xenova/ instantly.
   */
  async init(): Promise<void> {
    if (this.initialized) return;
    await getPipeline(this.modelName, this.quantized);
    this.initialized = true;
  }

  /**
   * Embed text for storage (document embedding).
   * Called by retain() — text being stored in memory.
   * Satisfies the EmbeddingProvider interface.
   */
  async embed(text: string): Promise<Float32Array> {
    return this.embedWithMode(text, 'document');
  }

  /**
   * Embed text for search (query embedding).
   * Called by recall() — the search query.
   *
   * For models that use prefix distinction (nomic-embed-text),
   * this produces better recall quality than document mode for queries.
   */
  async embedQuery(text: string): Promise<Float32Array> {
    return this.embedWithMode(text, 'query');
  }

  private async embedWithMode(text: string, mode: 'query' | 'document'): Promise<Float32Array> {
    const extractor = await getPipeline(this.modelName, this.quantized) as (
      text: string,
      options: { pooling: string; normalize: boolean }
    ) => Promise<{ data: ArrayLike<number> }>;

    const prefix = mode === 'query'
      ? this.modelInfo.queryPrefix
      : this.modelInfo.documentPrefix;

    const output = await extractor(prefix + text, {
      pooling: 'mean',
      normalize: true,
    });

    return new Float32Array(output.data);
  }
}
