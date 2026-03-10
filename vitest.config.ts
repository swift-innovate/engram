import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Native addons (better-sqlite3) require forked child processes,
    // not worker_threads (the vitest default).
    pool: 'forks',
    // Auto-cleanup after each test
    restoreMocks: true,
    unstubGlobals: true,
  },
});
