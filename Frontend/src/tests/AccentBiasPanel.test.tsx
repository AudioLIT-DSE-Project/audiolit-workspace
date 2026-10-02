/**
 * Accent Bias Dashboard (FR15.1). The panel had no test, and the report it
 * renders never arrived - the job's result was lost between worker and hook -
 * so nothing noticed that the dashboard had never once drawn a chart.
 */
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { jest } from "@jest/globals";

const mockUseTaskStatus = jest.fn();
jest.mock("../hooks/useTaskStatus", () => ({
  useTaskStatus: (taskId: string | null) => mockUseTaskStatus(taskId),
}));

import { AccentBiasPanel } from "../components/panels/AccentBiasPanel";

const cohort = (accent: string, mean_wer: number | null) => ({
  accent,
  sample_count: 10,
  scored_count: mean_wer === null ? 0 : 10,
  mean_wer,
  median_wer: mean_wer,
  stdev_wer: 0,
  min_wer: mean_wer,
  max_wer: mean_wer,
});

// Ranked worst-first, as the backend returns it. Vietnamese is above 1.0.
const REPORT = {
  corpus: "l2-arctic",
  model_id: "openai/whisper-base",
  cohorts: [cohort("Vietnamese", 1.25), cohort("Arabic", 0.31), cohort("Hindi", 0.12)],
};

const idle = { state: "QUEUED", result: null, error: null };

const startDiagnostic = async () => {
  fireEvent.click(screen.getByRole("button", { name: /Run Accent-Bias Diagnostic/i }));
  await waitFor(() => expect(global.fetch).toHaveBeenCalled());
};

describe("AccentBiasPanel (FR15.1)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseTaskStatus.mockReturnValue(idle);
    (global as any).fetch = jest.fn(async () => ({
      ok: true,
      json: async () => ({ job_id: "bias-1", websocket_url: "ws://x", schema_version: "1", family_jobs: {} }),
    }));
  });

  it("offers no diagnostic for a model that does not transcribe", () => {
    render(<AccentBiasPanel model="wav2vec2" />);
    expect(screen.queryByRole("button", { name: /Run Accent-Bias Diagnostic/i })).not.toBeInTheDocument();
    expect(screen.getByText(/Select a Whisper model/i)).toBeInTheDocument();
  });

  it("starts the diagnostic for the built-in Whisper model", async () => {
    render(<AccentBiasPanel model="whisper-base" />);
    await startDiagnostic();

    const [url, options] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://localhost:8000/evaluation/accent-bias");
    expect(JSON.parse((options as RequestInit).body as string)).toEqual({
      model_id: "openai/whisper-base",
      corpus: "l2-arctic",
    });
  });

  it("passes a custom Whisper checkpoint through under its own id", async () => {
    render(<AccentBiasPanel model="openai/whisper-small" />);
    await startDiagnostic();

    const [, options] = (global.fetch as jest.Mock).mock.calls[0];
    expect(JSON.parse((options as RequestInit).body as string).model_id).toBe("openai/whisper-small");
  });

  it("renders the ranked report once the job succeeds", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1" ? { state: "SUCCESS", result: REPORT, error: null } : idle,
    );
    render(<AccentBiasPanel model="whisper-base" />);
    await startDiagnostic();

    await waitFor(() => expect(screen.getByText("cohorts: 3")).toBeInTheDocument());
    expect(screen.getByText("worst: Vietnamese (1.250)")).toBeInTheDocument();
    expect(screen.getByText("best: Hindi (0.120)")).toBeInTheDocument();
    expect(screen.getByText("disparity: 1.130")).toBeInTheDocument();
  });

  it("does not treat a result without cohorts as a report", async () => {
    // What the hook used to hand over: the worker's timing payload.
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1" ? { state: "SUCCESS", result: { duration_s: 41.2 }, error: null } : idle,
    );
    render(<AccentBiasPanel model="whisper-base" />);
    await startDiagnostic();

    expect(screen.queryByText(/cohorts:/)).not.toBeInTheDocument();
  });

  it("says so when no cohort could be scored", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1"
        ? { state: "SUCCESS", result: { ...REPORT, cohorts: [cohort("Hindi", null)] }, error: null }
        : idle,
    );
    render(<AccentBiasPanel model="whisper-base" />);
    await startDiagnostic();

    await waitFor(() => expect(screen.getByText(/no cohort could be scored/i)).toBeInTheDocument());
  });

  it("shows the worker's error on failure and lets the user retry", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1"
        ? { state: "FAILURE", result: null, error: "FileNotFoundError: L2-ARCTIC root not found" }
        : idle,
    );
    render(<AccentBiasPanel model="whisper-base" />);
    await startDiagnostic();

    await waitFor(() =>
      expect(screen.getByText(/FileNotFoundError: L2-ARCTIC root not found/)).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /Run Accent-Bias Diagnostic/i })).toBeEnabled();
  });
});
