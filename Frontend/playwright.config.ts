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
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
