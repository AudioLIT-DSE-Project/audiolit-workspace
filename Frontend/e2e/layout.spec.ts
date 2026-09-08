import { test, expect, type Page } from "@playwright/test";

/**
 * LIT-160: responsive layout verification across Chrome/Firefox/Safari
 * (playwright.config.ts's three projects) at a few standard developer
 * viewport widths, checking for the "flexible grid constraints where charts
 * overflow" failure mode the issue names - a panel or chart forcing
 * horizontal scroll on the page instead of shrinking within its resizable
 * panel.
 */
const VIEWPORTS = [
  { name: "wide-desktop", width: 1920, height: 1080 },
  { name: "laptop", width: 1366, height: 768 },
  { name: "small-desktop", width: 1024, height: 768 },
];

async function getDocumentOverflow(page: Page) {
  return page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
}

// Browsers legitimately surface this as a `pageerror`/window error even
// though it's an inert, well-known ResizeObserver quirk (the callback didn't
// finish before the next frame) rather than an application bug - filtering
// it out here, not from the app, is the standard treatment.
const isBenignResizeObserverNotice = (message: string) =>
  message.includes("ResizeObserver loop");

test.describe("Main workbench layout - responsive rendering across browsers", () => {
  for (const viewport of VIEWPORTS) {
    test(`renders without horizontal overflow at ${viewport.name} (${viewport.width}x${viewport.height})`, async ({
      page,
    }) => {
      const pageErrors: string[] = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto("/");

      // Root of the three-panel workbench (embedding / prediction+dataset /
      // datapoint editor) - wait for the real layout rather than a fixed sleep.
      await page.waitForSelector("[data-panel-group]", { timeout: 15_000 });

      // A couple of px of slack for scrollbar-width differences across browsers.
      const { scrollWidth, clientWidth } = await getDocumentOverflow(page);
      expect(
        scrollWidth,
        `document is ${scrollWidth}px wide but the viewport is only ${clientWidth}px - something is forcing horizontal overflow at ${viewport.name}`,
      ).toBeLessThanOrEqual(clientWidth + 2);

      const realErrors = pageErrors.filter((msg) => !isBenignResizeObserverNotice(msg));
      expect(realErrors, `Uncaught page errors: ${realErrors.join("; ")}`).toEqual([]);
    });
  }

  test("all three top-level panels stay within the viewport at a standard desktop size", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");

    // Only the root horizontal group's direct panels - the middle one nests
    // its own vertical PanelGroup, whose panels also carry [data-panel] and
    // would otherwise inflate this count to 5.
    const panels = page.locator(
      '[data-panel-group][data-panel-group-direction="horizontal"] > [data-panel]',
    );
    await expect(panels).toHaveCount(3, { timeout: 15_000 });

    const viewportSize = page.viewportSize();
    if (!viewportSize) throw new Error("Page reported no viewport size");

    const boxes = await panels.evaluateAll((elements) =>
      elements.map((el) => {
        const rect = el.getBoundingClientRect();
        return { x: rect.x, right: rect.x + rect.width };
      }),
    );

    for (const box of boxes) {
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.right).toBeLessThanOrEqual(viewportSize.width + 2);
    }
  });
});
