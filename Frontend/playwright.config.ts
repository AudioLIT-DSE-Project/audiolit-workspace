import { defineConfig, devices } from "@playwright/test";

/**
 * LIT-160: cross-browser responsive-layout verification. Scoped to
 * rendering/layout only (the app's own workbench UI booted against the
 * real Vite dev server) - it deliberately does not depend on the backend,
 * Redis, or RQ workers being up, so it stays fast and runs anywhere the
 * frontend suite does. Full data-flow E2E (upload -> inference -> results)
 * would need that whole stack live and is a separate, heavier effort.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:8080",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:8080",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    // LIT-160's layout projects. `testIgnore` keeps the stack-dependent
    // data-flow specs out of them, so `npm run test:e2e` stays exactly as fast
    // and as backend-free as it was designed to be.
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
      testIgnore: /dataflow\.spec\.ts/,
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
      testIgnore: /dataflow\.spec\.ts/,
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
      testIgnore: /dataflow\.spec\.ts/,
    },

    // Opt-in: `npm run test:e2e:dataflow`. Requires the backend, Redis and the
    // RQ workers to be running. Chromium only - what these assert is that data
    // survives the round trip from model to panel, which is not a per-browser
    // property; running it three times would triple a suite whose individual
    // requests already take tens of seconds on CPU.
    {
      name: "dataflow",
      use: { ...devices["Desktop Chrome"] },
      testMatch: /dataflow\.spec\.ts/,
      timeout: 180_000,
    },
  ],
});
