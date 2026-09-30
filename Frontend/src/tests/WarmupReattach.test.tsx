import "@testing-library/jest-dom";
import {
  ACTIVE_WARMUP_JOB_KEY,
  readActiveWarmupJobId,
  writeActiveWarmupJobId,
  clearActiveWarmupJobId,
} from "@/lib/warmupJob";

// A batch warmup runs server-side for up to its 24h RQ job timeout, and
// cancellation is addressed by job id. The id used to live only in React
// state, so a reload or a discarded background tab stranded the run: no
// banner, and no way to stop it. These cover the persistence half of the fix
// (the other half is the server-side GET /api/inference/warmup/active lookup,
// which covers cleared storage and other browsers).
describe("active warmup job id persistence", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns null when no warmup has been started", () => {
    expect(readActiveWarmupJobId()).toBeNull();
  });

  it("round-trips a job id so a reload can reattach to the run", () => {
    writeActiveWarmupJobId("warmup_abc123");
    expect(readActiveWarmupJobId()).toBe("warmup_abc123");
    expect(window.localStorage.getItem(ACTIVE_WARMUP_JOB_KEY)).toBe("warmup_abc123");
  });

  it("clears the id once the run reaches a terminal state", () => {
    writeActiveWarmupJobId("warmup_abc123");
    clearActiveWarmupJobId();
    expect(readActiveWarmupJobId()).toBeNull();
    expect(window.localStorage.getItem(ACTIVE_WARMUP_JOB_KEY)).toBeNull();
  });

  it("overwrites a stale id rather than accumulating ids", () => {
    writeActiveWarmupJobId("warmup_old");
    writeActiveWarmupJobId("warmup_new");
    expect(readActiveWarmupJobId()).toBe("warmup_new");
  });

  // A private window with site data blocked throws on access. The workbench
  // must still render - the server-side lookup is the fallback - so none of
  // these may propagate.
  it("read returns null instead of throwing when getItem throws", () => {
    const spy = jest
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new Error("storage blocked");
      });
    expect(() => readActiveWarmupJobId()).not.toThrow();
    expect(readActiveWarmupJobId()).toBeNull();
    spy.mockRestore();
  });

  it("write does not throw when setItem throws", () => {
    const spy = jest
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("storage blocked");
      });
    expect(() => writeActiveWarmupJobId("warmup_abc123")).not.toThrow();
    spy.mockRestore();
  });

  it("clear does not throw when removeItem throws", () => {
    const spy = jest
      .spyOn(Storage.prototype, "removeItem")
      .mockImplementation(() => {
        throw new Error("storage blocked");
      });
    expect(() => clearActiveWarmupJobId()).not.toThrow();
    spy.mockRestore();
  });
});
