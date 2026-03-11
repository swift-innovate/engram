/**
 * Basic Engram usage example
 *
 * Prerequisites:
 *   - Ollama running at http://starbase:40114
 *   - nomic-embed-text model: ollama pull nomic-embed-text
 *   - llama3.1:8b model:      ollama pull llama3.1:8b
 *
 * Run:
 *   npm run example
 *   # or: npx tsx examples/basic-usage.ts
 */

import { Engram } from '../src/engram.js';

// Initialize — creates ./mira.engram if it doesn't exist
const mira = await Engram.create('./mira.engram', {
  reflectMission: 'Focus on architecture preferences, project patterns, and infrastructure decisions.',
  retainMission: 'Prioritize technical decisions, code patterns, and project context. Ignore greetings.',
  ollamaUrl: 'http://starbase:40114',
});

console.log('Engram initialized. Storing memories...\n');

// Store world facts
await mira.retain('Tom prefers Terraform with the bpg provider for Proxmox IaC', {
  memoryType: 'world',
  source: 'conversation:valor-engine-planning',
  sourceType: 'user_stated',
  trustScore: 0.9,
  context: 'infrastructure',
});

await mira.retain('The valor-engine project uses TypeScript and runs on Node.js 22', {
  memoryType: 'world',
  source: 'conversation:valor-engine-planning',
  sourceType: 'user_stated',
  trustScore: 0.9,
  context: 'projects',
});

await mira.retain('Tom decided to use SQLite over Postgres for zero-infrastructure deployments', {
  memoryType: 'world',
  source: 'conversation:engram-design',
  sourceType: 'user_stated',
  trustScore: 0.85,
  context: 'architecture',
});

await mira.retain('Tom values fast write paths — never block agent turns on LLM calls', {
  memoryType: 'world',
  source: 'conversation:engram-design',
  sourceType: 'user_stated',
  trustScore: 0.9,
  context: 'architecture',
});

await mira.retain('Tom is building a homelab on Proxmox with three NUC nodes', {
  memoryType: 'world',
  source: 'conversation:infrastructure',
  sourceType: 'user_stated',
  trustScore: 0.8,
  context: 'infrastructure',
});

console.log('5 memories stored.\n');

// Recall
console.log('--- Recall: "What IaC tools does Tom use?" ---');
const response = await mira.recall('What IaC tools does Tom use?', { topK: 5 });

for (const r of response.results) {
  console.log(`  [${r.memoryType}] trust=${r.trustScore} score=${r.score.toFixed(4)} via [${r.strategies.join(',')}]`);
  console.log(`  → ${r.text}\n`);
}

if (response.observations.length > 0) {
  console.log('--- Observations ---');
  for (const o of response.observations) {
    console.log(`  [${o.domain}/${o.topic}] ${o.summary}`);
  }
  console.log();
}

if (response.opinions.length > 0) {
  console.log('--- Opinions ---');
  for (const o of response.opinions) {
    console.log(`  [${o.domain}] (${o.confidence.toFixed(2)}) ${o.belief}`);
  }
  console.log();
}

console.log(`Total candidates before ranking: ${response.totalCandidates}`);
console.log(`Strategies used: ${response.strategiesUsed.join(', ')}\n`);

// Process entity extraction queue (builds knowledge graph via Ollama)
console.log('--- Processing entity extraction queue ---');
const extracted = await mira.processExtractions(5);
console.log(`  processed: ${extracted.processed}, failed: ${extracted.failed}\n`);

// Reflect (synthesizes observations + opinions — needs ≥5 unreflected facts)
console.log('--- Running reflection cycle ---');
const reflectResult = await mira.reflect();
console.log('  Result:', JSON.stringify(reflectResult, null, 2));

mira.close();
console.log('\nDone. Memory file: ./mira.engram');
