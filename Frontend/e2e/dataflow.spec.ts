import { test, expect, type Page } from "@playwright/test";

/**
 * Data-flow E2E: does a model's output actually reach the panel that claims to
 * show it?
 *
 * LIT-160's layout.spec.ts deliberately runs without a backend, so it can only
 * see whether the shell renders. The failures this project has actually shipped
 * were all a layer deeper and invisible to that: a saliency map that was really
 * an encoder-energy fallback, word labels naming words the transcript never
 * contained, attention extracted at full cost and dropped before the response,
 * a deepfake model handed the emotion model's embeddings. Every one of them
 * rendered a perfectly laid-out panel full of the wrong thing.
 *
 * So these assert provenance and content, not presence. "A heatmap appeared" is
 * not a passing condition here; "the heatmap is flagged as measured" is.
 *
 * Requires the full stack - Redis, API, and the RQ workers:
 *
 *     cd Backend && docker compose up -d
 *     cd Backend && python -m uvicorn app.main:app --port 8000
 *     cd Backend && python -m app.orchestration.worker all
 *     cd Frontend && npm run test:e2e:dataflow
 *
 * Rather than fail obscurely when the backend is down, every test skips with a
 * message saying so.
 */

// 127.0.0.1, not localhost: on Windows `localhost` resolves to ::1 first while
// the backend binds IPv4 only (`--host 0.0.0.0`), so each new connection waits
// out an IPv6 connect timeout first - measured 2063 ms per fresh connection
// against 23 ms via 127.0.0.1. These tests open fresh connections, so the wrong
// name silently adds two seconds to every direct API call below.
const API = process.env.AUDIOLIT_API ?? "http://127.0.0.1:8000";

/** Measured on an idle CPU box: Grad-CAM ~17 s, IG ~21 s, SHAP ~34 s, LIME ~110 s. */
const SALIENCY_TIMEOUT_MS = 150_000;

async function backendIsUp(): Promise<boolean> {
  try {
    const res = await fetch(`${API}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

test.beforeEach(async () => {
  test.skip(
    !(await backendIsUp()),
    `No backend at ${API}. These tests drive real inference - start Redis, the API and the workers first (see the file header).`,
  );
});

/** Click the first dataset row and wait for the workspace to bind to it. */
async function selectFirstClip(page: Page): Promise<string> {
  await page.goto("/");

  const firstFilename = page.locator("text=/sample-\\d+\\.mp3/").first();
  await expect(firstFilename).toBeVisible({ timeout: 30_000 });

  const name = (await firstFilename.textContent())?.trim() ?? "";
  await firstFilename.click();
  return name;
}

test.describe("Dataset table to workspace", () => {
  test("selecting a clip binds it to the datapoint editor", async ({ page }) => {
    const name = await selectFirstClip(page);
    expect(name).toMatch(/sample-\d+\.mp3/);

    // The editor starts at "No file selected"; binding is what clears it.
    await expect(page.getByText("No file selected").first()).toBeHidden({
      timeout: 30_000,
    });
  });

  test("the predicted-transcript column is never raw JSON", async ({ page }) => {
    await selectFirstClip(page);

    // A dict reaching a string slot is how this has failed before - the column
    // rendered "[object Object]" or a serialised payload instead of words.
    const table = page.locator("table");
    await expect(table).toBeVisible({ timeout: 30_000 });
    const body = (await table.textContent()) ?? "";

    expect(body).not.toContain("[object Object]");
    expect(body).not.toContain('{"prediction"');
    expect(body).not.toContain('{"text"');
  });
});

test.describe("Saliency panel", () => {
  test("Grad-CAM renders a map that is flagged measured, not a fallback", async ({
    page,
  }) => {
    await selectFirstClip(page);

    // Capture the saliency response directly. Reading provenance off the wire
    // is the point: a fallback map and a real one look identical on screen,
    // which is exactly why the fallback went unnoticed for so long.
    const saliency = page.waitForResponse(
      (r) => r.url().includes("/saliency/generate") && r.status() === 200,
      { timeout: SALIENCY_TIMEOUT_MS },
    );

    await page.getByRole("tab", { name: "Saliency" }).click();
    const body = await (await saliency).json();

    expect(body.saliency_matrix?.length ?? 0).toBeGreaterThan(0);
    expect(body.base_spectrogram?.length ?? 0).toBeGreaterThan(0);
    expect(["measured", "fallback", "unavailable"]).toContain(body.provenance);
    expect(
      body.provenance,
      `saliency came back as "${body.provenance}" (${body.provenance_reason}) - ` +
        "the panel will still draw a heatmap, but it is not attribution",
    ).toBe("measured");
  });

  test("word segments name words the transcript actually contains", async ({
    page,
  }) => {
    await selectFirstClip(page);

    const saliency = page.waitForResponse(
      (r) => r.url().includes("/saliency/generate") && r.status() === 200,
      { timeout: SALIENCY_TIMEOUT_MS },
    );
    await page.getByRole("tab", { name: "Saliency" }).click();
    const body = await (await saliency).json();

    const segments = (body.segments ?? []) as Array<{ word?: string }>;
    test.skip(
      segments.length === 0 || segments.every((s) => s.word === undefined),
      "this model's saliency segments are not word-labelled",
    );

    // Two decode paths disagreed here once: the transcript panel said "Mines in
    // the door." while the saliency segments said "Minds". Both were correct
    // outputs of different decodes, and the pair was nonsense to a reader.
    const words = segments
      .map((s) => (s.word ?? "").trim().toLowerCase().replace(/[.,!?]/g, ""))
      .filter(Boolean);
    expect(words.length).toBeGreaterThan(0);

    const transcriptRes = await fetch(`${API}/inferences/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "whisper-base",
        dataset: "common-voice",
        dataset_file: body.dataset_file ?? undefined,
        file_path: body.file_path ?? undefined,
      }),
    });

    test.skip(
      !transcriptRes.ok,
      "could not fetch the canonical transcript to compare against",
    );
    const transcript = String(await transcriptRes.json())
      .toLowerCase()
      .replace(/[.,!?]/g, "");

    for (const word of words) {
      expect(
        transcript.includes(word),
        `saliency labels a segment "${word}", which the transcript "${transcript}" does not contain`,
      ).toBeTruthy();
    }
  });
});

test.describe("Deepfake panel", () => {
  test("a genuine speech clip is not reported as spoof at full confidence", async ({
    page,
  }) => {
    await selectFirstClip(page);

    // Common Voice is human speech. A detector calling it synthetic at 0.9999
    // is not a UI bug, but it is the single most visible way this product can
    // be wrong, and it has been - a checkpoint scoring 38.5% on 200 labelled
    // clips shipped as the default.
    const res = await fetch(`${API}/inferences/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "wav2vec2-add",
        dataset: "common-voice",
        dataset_file: "sample-000037.mp3",
      }),
    });
    expect(res.ok).toBeTruthy();

    const verdict = await res.json();
    expect(verdict).toHaveProperty("predicted_label");
    expect(
      verdict.predicted_label,
      `genuine Common Voice speech was called "${verdict.predicted_label}" ` +
        `with P(spoof)=${verdict.synthetic_probability}`,
    ).toBe("bona-fide");
  });
});

test.describe("Cache behaviour through the UI", () => {
  test("the same clip and model return the same prediction twice", async () => {
    const request = {
      model: "whisper-base",
      dataset: "common-voice",
      dataset_file: "sample-000037.mp3",
    };
    const call = async () => {
      const res = await fetch(`${API}/inferences/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      expect(res.ok).toBeTruthy();
      return res.json();
    };

    const first = await call();
    const second = await call();

    // A warm read must equal the cold computation it claims to be. Cache bugs
    // here have served one model's output under another model's key.
    expect(second).toEqual(first);
    expect(typeof first).toBe("string");
    expect(String(first).trim().length).toBeGreaterThan(0);
  });
});
