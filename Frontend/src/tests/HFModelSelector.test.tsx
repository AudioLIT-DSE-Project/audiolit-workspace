/**
 * Class names for a custom checkpoint published without them.
 *
 * Such a model reports LABEL_0..LABEL_n, which is what the dataset table showed
 * as its predictions. The dialog asks for the real names only in that case and
 * sends them to the backend, which applies them to every prediction.
 */
import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { jest } from "@jest/globals";
import { HFModelSelector } from "../components/layout/HFModelSelector";
import { ModelRegistryProvider } from "../context/ModelRegistryContext";

const RESOLVED = {
  model_id: "myorg/wav2vec2-emotion",
  revision: "abc123def456",
  family: "wav2vec2",
  weights_sha256: "deadbeefdeadbeefdeadbeef",
  available_layers: [],
};

const mockResolve = (extra: Record<string, unknown>) => {
  const fetchMock = jest.fn(async (url: unknown, init?: any) => {
    if (String(url).endsWith("/models/labels")) {
      const { labels } = JSON.parse(init.body);
      return { ok: true, json: async () => ({ model_id: RESOLVED.model_id, labels, labels_are_placeholders: true }) };
    }
    return { ok: true, json: async () => ({ ...RESOLVED, ...extra }) };
  });
  (global as any).fetch = fetchMock;
  return fetchMock;
};

const resolveInDialog = async () => {
  render(
    <ModelRegistryProvider>
      <HFModelSelector />
    </ModelRegistryProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: /custom model/i }));
  fireEvent.change(screen.getByLabelText("Model ID"), { target: { value: RESOLVED.model_id } });
  fireEvent.click(screen.getByRole("button", { name: "Resolve Model" }));
  await screen.findByText("Model ready");
};

describe("HFModelSelector class names", () => {
  it("asks for names when the checkpoint only has placeholders, and saves them", async () => {
    const fetchMock = mockResolve({ labels: ["LABEL_0", "LABEL_1"], labels_are_placeholders: true });
    await resolveInDialog();

    const save = screen.getByRole("button", { name: "Save class names" });
    expect(save).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Name for class 0"), { target: { value: "angry" } });
    fireEvent.change(screen.getByLabelText("Name for class 1"), { target: { value: "neutral" } });
    fireEvent.click(save);

    await waitFor(() => expect(screen.getByText(/Saved\./)).toBeInTheDocument());
    const [, init] = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/models/labels"))!;
    expect((init as any).method).toBe("PUT");
    expect(JSON.parse((init as any).body)).toEqual({
      model_id: RESOLVED.model_id,
      revision: RESOLVED.revision,
      labels: ["angry", "neutral"],
    });
  });

  it("prefills names saved earlier", async () => {
    mockResolve({ labels: ["angry", "neutral"], labels_are_placeholders: true });
    await resolveInDialog();
    expect(screen.getByLabelText("Name for class 0")).toHaveValue("angry");
  });

  it("stays out of the way when the checkpoint has real names", async () => {
    mockResolve({ labels: ["angry", "neutral"], labels_are_placeholders: false });
    await resolveInDialog();
    expect(screen.queryByText("Class names")).not.toBeInTheDocument();
  });
});
