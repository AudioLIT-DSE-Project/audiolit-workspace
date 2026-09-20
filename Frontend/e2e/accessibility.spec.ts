import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Automated accessibility checks against WCAG 2.1 A and AA.
 *
 * The Master Test Plan lists accessibility under section 3.1.3 as an
 * outstanding manual item. This covers the part a machine can decide -
 * contrast, names, roles, landmarks, form labels - so the remaining manual
 * pass can concentrate on what automation genuinely cannot judge, such as
 * keyboard order making sense or a screen reader announcement being useful.
 *
 * Axe finds roughly a third of WCAG issues in practice, so a clean run here
 * is a floor, not a certificate.
 *
 * These run backend-free against the Vite dev server, like the layout suite.
 */

const WCAG_AA = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

// Serious and critical are treated as failures. Minor and moderate are
// reported but do not fail the build, so a genuine regression is not buried
// under pre-existing low-severity noise in an inherited UI.
const BLOCKING_IMPACTS = new Set(["serious", "critical"]);

/**
 * Wait for the workbench shell to mount.
 *
 * Deliberately not `networkidle`: the workbench holds a task WebSocket open
 * and polls progress, so the network never goes idle and every scan timed out
 * at 30s. The panel group is the same anchor the layout suite uses, and it
 * means the DOM axe scans is the real, mounted one.
 */
async function waitForWorkbench(page: import("@playwright/test").Page) {
  await page.waitForSelector("[data-panel-group]", { timeout: 20_000 });
}

/**
 * Scan the workbench, not the dialog covering it.
 *
 * Every Playwright context is a fresh profile, so the first-run quick-start
 * dialog auto-opens and its modal overlay sits over the whole workbench. axe
 * then scans the dialog and reports the page as clean, because the controls
 * underneath are inert and hidden from the accessibility tree. That is a false
 * pass: a separate scan of the dismissed state found button-name violations at
 * critical severity across 18 nodes, none of which this suite could see.
 *
 * The dialog keeps its own coverage in quickstart.spec.ts.
 */
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("audiolit.quickstart.dismissed", "true");
    } catch {
      /* private mode: the dialog opens and the scan will report it */
    }
  });
});

test.describe("accessibility (WCAG 2.1 AA)", () => {
  test("workbench has no serious or critical violations", async ({ page }, testInfo) => {
    await page.goto("/");
    await waitForWorkbench(page);

    const results = await new AxeBuilder({ page }).withTags(WCAG_AA).analyze();

    const blocking = results.violations.filter((v) =>
      BLOCKING_IMPACTS.has(v.impact ?? "")
    );
    const advisory = results.violations.filter(
      (v) => !BLOCKING_IMPACTS.has(v.impact ?? "")
    );

    // Attach the full result so the report has evidence, pass or fail.
    await testInfo.attach("axe-results.json", {
      body: JSON.stringify(
        {
          url: page.url(),
          passes: results.passes.length,
          violations: results.violations.map((v) => ({
            id: v.id,
            impact: v.impact,
            help: v.help,
            nodes: v.nodes.length,
          })),
        },
        null,
        2
      ),
      contentType: "application/json",
    });

    if (advisory.length) {
      console.log(
        `advisory (non-blocking) violations: ${advisory
          .map((v) => `${v.id}[${v.impact}]x${v.nodes.length}`)
          .join(", ")}`
      );
    }

    const summary = blocking
      .map((v) => `${v.id} (${v.impact}, ${v.nodes.length} nodes): ${v.help}`)
      .join("\n");

    expect(blocking, `serious/critical accessibility violations:\n${summary}`).toEqual([]);
  });

  test("every page image carries an accessible name", async ({ page }) => {
    await page.goto("/");
    await waitForWorkbench(page);

    const results = await new AxeBuilder({ page })
      .withRules(["image-alt", "input-image-alt", "role-img-alt"])
      .analyze();

    expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([]);
  });

  test("text meets AA contrast", async ({ page }, testInfo) => {
    await page.goto("/");
    await waitForWorkbench(page);

    const results = await new AxeBuilder({ page })
      .withRules(["color-contrast"])
      .analyze();

    await testInfo.attach("contrast.json", {
      body: JSON.stringify(results.violations, null, 2),
      contentType: "application/json",
    });

    // Contrast is a colour-token decision, so a failure here is a design fix
    // and is reported with the offending selectors rather than summarised.
    const offenders = results.violations.flatMap((v) =>
      v.nodes.map((n) => n.target.join(" "))
    );
    expect(results.violations, `low-contrast elements:\n${offenders.join("\n")}`).toEqual([]);
  });

  test("the document exposes a main landmark and a page title", async ({ page }) => {
    await page.goto("/");
    await waitForWorkbench(page);

    await expect(page).toHaveTitle(/\S+/);

    const results = await new AxeBuilder({ page })
      .withRules(["document-title", "html-has-lang", "landmark-one-main"])
      .analyze();

    const ids = results.violations.map((v) => `${v.id}: ${v.help}`);
    expect(results.violations, ids.join("\n")).toEqual([]);
  });
});
