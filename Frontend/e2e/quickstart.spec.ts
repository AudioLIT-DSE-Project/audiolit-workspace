import { test, expect } from "@playwright/test";

// LIT-261: SRS §3.7's quick-start walkthrough. Backend-free like layout.spec.ts
// (dev server only) - the dialog's content is static instructional text, not
// live data, so nothing here needs the real stack.
test.describe("Quick-start walkthrough", () => {
  test("first load shows the dialog, all four tabs work, and it does not reappear after dismissal", async ({ page }) => {
    await page.goto("/");

    const dialog = page.getByTestId("quickstart-dialog");
    await expect(dialog).toBeVisible({ timeout: 15_000 });

    // Step through all four tabs.
    for (const track of ["deepfake", "mutation", "bias", "faithfulness"]) {
      await page.getByTestId(`quickstart-tab-${track}`).click();
      await expect(page.getByTestId(`quickstart-panel-${track}`)).toBeVisible();
    }

    // Tick "Don't show again" and close.
    await page.getByTestId("quickstart-dont-show-again").click();
    await page.getByTestId("quickstart-close").click();
    await expect(dialog).not.toBeVisible();

    // Reload - the dialog must not auto-open again. Radix unmounts a closed
    // dialog entirely, so absence from the DOM (not just non-visibility) is
    // the correct check.
    await page.reload();
    await page.waitForLoadState("networkidle");
    await expect(page.locator('[data-testid="quickstart-dialog"]')).toHaveCount(0);

    // The toolbar button reopens it on demand regardless of dismissal.
    await page.getByTestId("quickstart-reopen-button").click();
    await expect(page.getByTestId("quickstart-dialog")).toBeVisible();
  });

  test("dismissing without checking 'Don't show again' means it reopens on next load", async ({ page }) => {
    await page.goto("/");
    const dialog = page.getByTestId("quickstart-dialog");
    await expect(dialog).toBeVisible({ timeout: 15_000 });

    await page.getByTestId("quickstart-close").click();
    await expect(dialog).not.toBeVisible();

    await page.reload();
    await expect(page.getByTestId("quickstart-dialog")).toBeVisible({ timeout: 15_000 });
  });
});
