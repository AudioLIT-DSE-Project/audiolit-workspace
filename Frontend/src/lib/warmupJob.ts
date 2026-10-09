/**
 * Persistence for the in-flight dataset-warmup job id.
 *
 * A batch warmup runs server-side for up to its 24h RQ job timeout, and
 * cancellation is addressed by job id. The id previously lived only in
 * MainLayout's React state, so a reload, a navigation, or the browser
 * discarding a backgrounded tab dropped it: the progress banner vanished, the
 * run kept consuming CPU, and because Stop needs the id, it could no longer be
 * cancelled from the UI at all.
 *
 * Remembering the id locally is the first recovery path. The second is the
 * server's own list of live runs (`GET /api/inference/warmup/active`), which
 * still works when storage was cleared, when the job was started from another
 * browser, or in a private window where these calls throw.
 *
 * Kept in its own module so it can be tested without mounting the workbench.
 */

export const ACTIVE_WARMUP_JOB_KEY = "audiolit.warmup.activeJobId";

/** The remembered job id, or null - never throws. */
export const readActiveWarmupJobId = (): string | null => {
  try {
    return window.localStorage.getItem(ACTIVE_WARMUP_JOB_KEY);
  } catch {
    return null;
  }
};

/** Remember a job id. A failure is non-fatal: server discovery still finds it. */
export const writeActiveWarmupJobId = (jobId: string): void => {
  try {
    window.localStorage.setItem(ACTIVE_WARMUP_JOB_KEY, jobId);
  } catch {
    /* non-fatal */
  }
};

/** Forget the job id, once the run has reached a terminal state. */
export const clearActiveWarmupJobId = (): void => {
  try {
    window.localStorage.removeItem(ACTIVE_WARMUP_JOB_KEY);
  } catch {
    /* non-fatal */
  }
};

/**
 * Statuses after which a warmup run will never change again. "interrupted"
 * is set by the backend when the run's worker died under it (e.g. the
 * containers were recreated mid-run) - it must stop polling and reattaching
 * just like a run the user cancelled.
 */
export const TERMINAL_WARMUP_STATUSES = ["completed", "cancelled", "failed", "interrupted"] as const;

export const isTerminalWarmupStatus = (status: string | undefined | null): boolean =>
  !!status && (TERMINAL_WARMUP_STATUSES as readonly string[]).includes(status);
