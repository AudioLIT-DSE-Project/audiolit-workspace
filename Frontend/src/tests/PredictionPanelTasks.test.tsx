/**
 * Which result cards the Analytics tab shows.
 *
 * The result cache is keyed by clip, and a dataset warmup fills it with ASR,
 * SER and acoustic results for every clip. On a warmed-up corpus an ASR model
 * therefore got an Emotion Analytics card it had never been asked for. A
 * recovered result is shown only for the selected model's own task; a
 * multi-task job the user actually ran still shows everything it produced.
 */
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { jest } from "@jest/globals";
import { PredictionPanel } from "../components/panels/PredictionPanel";

// The clip's cache holds all three tasks, as it does after a dataset warmup.
const CACHED = {
  asr: { transcript: "Work hard just to have food and water like this.", tokens: [] },
  ser: { predicted_emotion: "neutral", probabilities: { neutral: 0.465, angry: 0.387 }, confidence: 0.465 },
  add: { label: "bona-fide", confidence: 0.91, synthetic_probability: 0.09 },
};

const DATASET_ROW = {
  file_id: "sample-000179.mp3",
  filename: "sample-000179.mp3",
  file_path: "cv-valid-dev/sample-000179.mp3",
  message: "Selected from dataset",
};

const renderWith = (props: Partial<React.ComponentProps<typeof PredictionPanel>>) =>
  render(<PredictionPanel selectedFile={DATASET_ROW} dataset="common-voice" {...props} />);

describe("PredictionPanel task cards", () => {
  beforeEach(() => {
    (global as any).fetch = jest.fn(async (url: unknown) => {
      if (String(url).includes("/api/inference/cached-results")) {
        return { ok: true, json: async () => ({ tasks: CACHED, cached: true }) };
      }
      // Saliency and acoustic profile: not under test here.
      return { ok: false, status: 404, json: async () => ({}) };
    });
  });

  it("shows only the transcript for an ASR model, even when emotion is cached for the clip", async () => {
    renderWith({ model: "whisper-base" });

    await waitFor(() => expect(screen.getByText("Transcription Timeline")).toBeInTheDocument());
    expect(screen.getByText(/Work hard just to have food/)).toBeInTheDocument();
    expect(screen.queryByText("Emotion Analytics")).not.toBeInTheDocument();
    expect(screen.queryByText(/Bona-fide Audio|Deepfake Detected/)).not.toBeInTheDocument();
  });

  it("shows only the emotion result for an SER model", async () => {
    renderWith({ model: "wav2vec2" });

    await waitFor(() => expect(screen.getByText("Emotion Analytics")).toBeInTheDocument());
    expect(screen.queryByText("Transcription Timeline")).not.toBeInTheDocument();
    expect(screen.queryByText(/Bona-fide Audio|Deepfake Detected/)).not.toBeInTheDocument();
  });

  it("shows only the deepfake verdict for a deepfake model", async () => {
    renderWith({ model: "melody-machine" });

    await waitFor(() => expect(screen.getByText("Bona-fide Audio")).toBeInTheDocument());
    expect(screen.queryByText("Emotion Analytics")).not.toBeInTheDocument();
    expect(screen.queryByText("Transcription Timeline")).not.toBeInTheDocument();
  });

  it("still shows every task of a multi-task job the user ran", async () => {
    renderWith({
      model: "whisper-base",
      unifiedResult: {
        tasks: {
          asr: { transcript: "hello there" },
          ser: { predicted_emotion: "happy", probabilities: { happy: 0.8, sad: 0.2 } },
        },
      },
    });

    await waitFor(() => expect(screen.getByText("Transcription Timeline")).toBeInTheDocument());
    expect(screen.getByText("Emotion Analytics")).toBeInTheDocument();
  });

  it("does not present another model's per-clip prediction", async () => {
    // MainLayout's per-model fetches can still hold the previous model's result
    // for a moment after the model is switched.
    (global as any).fetch = jest.fn(async (url: unknown) =>
      String(url).includes("/api/inference/cached-results")
        ? { ok: true, json: async () => ({ tasks: {} }) }
        : { ok: false, status: 404, json: async () => ({}) },
    );
    renderWith({
      model: "whisper-base",
      wav2vecPrediction: { predicted_emotion: "sad", probabilities: { sad: 0.9 }, confidence: 0.9 },
    });

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(screen.queryByText("Emotion Analytics")).not.toBeInTheDocument();
  });
});
