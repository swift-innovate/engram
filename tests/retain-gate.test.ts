import { describe, it, expect } from 'vitest';
import { shouldRetain } from '../src/retain.js';

describe('shouldRetain()', () => {
  it('returns a score and reason', () => {
    const result = shouldRetain('Tom prefers Terraform for infrastructure');
    expect(result).toHaveProperty('score');
    expect(result).toHaveProperty('reason');
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it('is deterministic — same text always produces the same score', () => {
    const text = 'Alice decided to use Rust for the embedded project';
    const r1 = shouldRetain(text);
    const r2 = shouldRetain(text);
    expect(r1.score).toBe(r2.score);
    expect(r1.reason).toBe(r2.reason);
  });

  // --- Low-score cases ---

  it('scores very short text below 0.3', () => {
    expect(shouldRetain('ok').score).toBeLessThan(0.3);
    expect(shouldRetain('hey').score).toBeLessThan(0.3);
  });

  it('scores phatic expressions very low', () => {
    expect(shouldRetain('thanks').score).toBeLessThan(0.2);
    expect(shouldRetain('sounds good').score).toBeLessThan(0.2);
    expect(shouldRetain('got it').score).toBeLessThan(0.2);
  });

  it('scores pure interrogatives lower than statements', () => {
    const question = shouldRetain('What tools does Tom use?');
    const statement = shouldRetain('Tom uses Terraform and Pulumi for infrastructure');
    expect(question.score).toBeLessThan(statement.score);
  });

  // --- High-score cases ---

  it('scores decision language higher (>= 0.6)', () => {
    const result = shouldRetain('Tom decided to switch from Terraform to Pulumi');
    expect(result.score).toBeGreaterThanOrEqual(0.6);
    expect(result.reason).toMatch(/decision/);
  });

  it('scores text with technical terms higher', () => {
    const result = shouldRetain('The camelCase function in src/retain.ts handles dedup logic');
    expect(result.score).toBeGreaterThan(0.5);
    expect(result.reason).toMatch(/technical/);
  });

  it('scores text with temporal markers higher', () => {
    const result = shouldRetain('The deadline for the project is next week');
    expect(result.score).toBeGreaterThan(0.5);
    expect(result.reason).toMatch(/temporal/);
  });

  it('scores text with proper nouns higher', () => {
    const result = shouldRetain('Alice and Bob agreed on the architecture for Barracuda');
    expect(result.score).toBeGreaterThan(0.5);
    expect(result.reason).toMatch(/proper/);
  });

  it('scores substantive-length text higher than short text', () => {
    const long = shouldRetain(
      'Tom consistently prefers lightweight, zero-infrastructure solutions: SQLite over Postgres, ' +
      'static sites over dynamic ones, and Pulumi over Terraform when possible.'
    );
    const short = shouldRetain('ok sure');
    expect(long.score).toBeGreaterThan(short.score);
  });

  // --- Combinations ---

  it('accumulates signals: decision + technical + length produces high score', () => {
    const result = shouldRetain(
      'We decided to switch the entire homelab IaC stack from Terraform bpg to Pulumi ' +
      'because the provider supports better drift detection in proxmox environments.'
    );
    expect(result.score).toBeGreaterThanOrEqual(0.7);
  });

  it('negative signals can bring score to 0 but not below', () => {
    // Very short phatic: starts at 0.5, -0.3 (short) -0.4 (phatic) = -0.2, clamped to 0
    const result = shouldRetain('hey');
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it('positive signals cannot push score above 1', () => {
    // Multiple positive signals
    const result = shouldRetain(
      'Tom decided to use Pulumi today with the TypeScript SDK for all infrastructure at Swift-Innovate, ' +
      'including the barracuda project and proxmox clusters.'
    );
    expect(result.score).toBeLessThanOrEqual(1);
  });
});
