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
  fireEvent.click(screen.getByRole("button", { name: /Run (Accent-)?Bias Diagnostic/i }));
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
    render(<AccentBiasPanel model="wav2vec2" dataset="l2-arctic" />);
    expect(screen.queryByRole("button", { name: /Run Accent-Bias Diagnostic/i })).not.toBeInTheDocument();
    expect(screen.getByText(/Select a Whisper model/i)).toBeInTheDocument();
  });

  it("starts the diagnostic for the built-in Whisper model", async () => {
    render(<AccentBiasPanel model="whisper-base" dataset="l2-arctic" />);
    await startDiagnostic();

    const [url, options] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://localhost:8000/evaluation/accent-bias");
    expect(JSON.parse((options as RequestInit).body as string)).toEqual({
      model_id: "openai/whisper-base",
      corpus: "l2-arctic",
    });
  });

  it("passes a custom Whisper checkpoint through under its own id", async () => {
    render(<AccentBiasPanel model="openai/whisper-small" dataset="l2-arctic" />);
    await startDiagnostic();

    const [, options] = (global.fetch as jest.Mock).mock.calls[0];
    expect(JSON.parse((options as RequestInit).body as string).model_id).toBe("openai/whisper-small");
  });

  it("renders the ranked report once the job succeeds", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1" ? { state: "SUCCESS", result: REPORT, error: null } : idle,
    );
    render(<AccentBiasPanel model="whisper-base" dataset="l2-arctic" />);
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
    render(<AccentBiasPanel model="whisper-base" dataset="l2-arctic" />);
    await startDiagnostic();

    expect(screen.queryByText(/cohorts:/)).not.toBeInTheDocument();
  });

  it("says so when no cohort could be scored", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1"
        ? { state: "SUCCESS", result: { ...REPORT, cohorts: [cohort("Hindi", null)] }, error: null }
        : idle,
    );
    render(<AccentBiasPanel model="whisper-base" dataset="l2-arctic" />);
    await startDiagnostic();

    await waitFor(() => expect(screen.getByText(/no group could be scored/i)).toBeInTheDocument());
  });

  it("shows the worker's error on failure and lets the user retry", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1"
        ? { state: "FAILURE", result: null, error: "FileNotFoundError: L2-ARCTIC root not found" }
        : idle,
    );
    render(<AccentBiasPanel model="whisper-base" dataset="l2-arctic" />);
    await startDiagnostic();

    await waitFor(() =>
      expect(screen.getByText(/FileNotFoundError: L2-ARCTIC root not found/)).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /Run Accent-Bias Diagnostic/i })).toBeEnabled();
  });
});

describe("AccentBiasPanel follows the selected dataset", () => {
  const requestBody = () => JSON.parse(((global.fetch as jest.Mock).mock.calls[0][1] as RequestInit).body as string);

  const emotionCohort = (group: string, accuracy: number | null) => ({
    group,
    sample_count: 25,
    scored_count: accuracy === null ? 0 : 25,
    correct_count: accuracy === null ? 0 : Math.round(accuracy * 25),
    accuracy,
  });

  // Ranked lowest accuracy first, as the backend returns it.
  const CREMA_REPORT = {
    corpus: "crema-d",
    model_id: "wav2vec2",
    metric: "emotion_accuracy",
    group_by: "race",
    cohorts: [emotionCohort("Asian", 0.4), emotionCohort("African American", 0.6), emotionCohort("Caucasian", 0.72)],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseTaskStatus.mockReturnValue(idle);
    (global as any).fetch = jest.fn(async () => ({
      ok: true,
      json: async () => ({ job_id: "bias-1", websocket_url: "ws://x", schema_version: "1", family_jobs: {} }),
    }));
  });

  it("says which datasets it runs on when another one is selected", () => {
    for (const dataset of ["common-voice", "ravdess", "asvspoof-2021", "custom:abc:mine"]) {
      const { unmount } = render(<AccentBiasPanel model="whisper-base" dataset={dataset} />);
      expect(screen.getByText(/available for the L2-ARCTIC, CREMA-D and ESD datasets only/i)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /Diagnostic/i })).not.toBeInTheDocument();
      unmount();
    }
  });

  it("asks for the SER model on an emotion dataset", () => {
    render(<AccentBiasPanel model="whisper-base" dataset="crema-d" />);
    expect(screen.getByText(/Select the Wav2Vec2 \(SER\) model/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Diagnostic/i })).not.toBeInTheDocument();
  });

  it("still asks for a Whisper model on L2-ARCTIC", () => {
    render(<AccentBiasPanel model="wav2vec2" dataset="l2-arctic" />);
    expect(screen.getByText(/Select a Whisper model/i)).toBeInTheDocument();
  });

  it("runs emotion bias on CREMA-D, grouped by race by default", async () => {
    render(<AccentBiasPanel model="wav2vec2" dataset="crema-d" />);
    await startDiagnostic();
    expect(requestBody()).toEqual({
      model_id: "wav2vec2",
      corpus: "crema-d",
      group_by: "race",
      samples_per_cohort: 25,
    });
  });

  it("lets CREMA-D be grouped by sex or ethnicity instead", async () => {
    render(<AccentBiasPanel model="wav2vec2" dataset="crema-d" />);
    fireEvent.change(screen.getByLabelText(/Group by/i), { target: { value: "sex" } });
    await startDiagnostic();
    expect(requestBody().group_by).toBe("sex");
  });

  it("runs emotion bias on ESD by language, with no grouping choice to make", async () => {
    render(<AccentBiasPanel model="wav2vec2" dataset="esd" />);
    expect(screen.queryByLabelText(/Group by/i)).not.toBeInTheDocument();
    await startDiagnostic();
    expect(requestBody()).toMatchObject({ corpus: "esd", group_by: "language" });
  });

  it("ranks an accuracy report with the lowest accuracy as the worst", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1" ? { state: "SUCCESS", result: CREMA_REPORT, error: null } : idle,
    );
    render(<AccentBiasPanel model="wav2vec2" dataset="crema-d" />);
    await startDiagnostic();

    await waitFor(() => expect(screen.getByText("cohorts: 3")).toBeInTheDocument());
    expect(screen.getByText("worst: Asian (40.0%)")).toBeInTheDocument();
    expect(screen.getByText("best: Caucasian (72.0%)")).toBeInTheDocument();
    expect(screen.getByText("disparity: 32.0%")).toBeInTheDocument();
    expect(screen.getByText(/Emotion accuracy by race/)).toBeInTheDocument();
  });

  it("drops the report when the dataset changes", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === "bias-1" ? { state: "SUCCESS", result: CREMA_REPORT, error: null } : { ...idle, state: "SUCCESS", result: CREMA_REPORT },
    );
    const { rerender } = render(<AccentBiasPanel model="wav2vec2" dataset="crema-d" />);
    await startDiagnostic();
    await waitFor(() => expect(screen.getByText("cohorts: 3")).toBeInTheDocument());

    // The hook keeps its last result for a null id; the panel must not show it.
    rerender(<AccentBiasPanel model="wav2vec2" dataset="esd" />);
    expect(screen.queryByText("cohorts: 3")).not.toBeInTheDocument();
  });

  it("shows the backend's reason when the run cannot start", async () => {
    (global as any).fetch = jest.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Corpus 'esd' cannot be grouped by 'race'. Choose from: language." }),
    }));
    render(<AccentBiasPanel model="wav2vec2" dataset="esd" />);
    fireEvent.click(screen.getByRole("button", { name: /Run Bias Diagnostic/i }));
    await waitFor(() => expect(screen.getByText(/cannot be grouped by 'race'/)).toBeInTheDocument());
  });
});
