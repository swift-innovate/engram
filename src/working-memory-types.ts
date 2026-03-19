// =============================================================================
// working-memory-types.ts — Types for Working Memory Sessions
// =============================================================================

export interface WorkingMemoryState {
  /** Session ID (matches working_memory.id) */
  id: string;
  /** The user's apparent goal for this session */
  goal: string;
  /** Agent-written summary of work done so far — captured in snapshot */
  progress?: string;
  /** When this session state was last updated */
  updated_at: string;
  /** Agent-defined extensions — any additional session state */
  [key: string]: unknown;
}

export interface WorkingMemoryOptions {
  /** Max active (non-expired) sessions to keep. Oldest auto-snapshot when exceeded. Default: 5 */
  maxActive?: number;
  /** Cosine similarity threshold for matching an existing session. Default: 0.55
   *  Calibrated for nomic-embed-text 768d via Ollama. Lower = more aggressive matching,
   *  higher = more new sessions. Range: 0.0–1.0 */
  threshold?: number;
  /** Hours before an untouched session auto-expires. Default: 48 */
  expireAfterHours?: number;
}

export interface SessionCandidate {
  id: string;
  state: WorkingMemoryState;
  similarity: number;
}

export interface WorkingSessionResult {
  /** The resolved or newly created session state */
  session: WorkingMemoryState;
  /** Formatted related context from long-term memory (output of formatForPrompt) */
  relatedContext: string;
  /** Cosine similarity score of the match (1.0 = new session) */
  confidence: number;
  /** Diagnostic info */
  diagnostics: {
    sessionId: string;
    reason: 'match' | 'new' | 'forced';
    candidatesEvaluated: number;
  };
}
