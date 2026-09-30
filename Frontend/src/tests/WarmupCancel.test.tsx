import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import { WarmupModal, WarmupProgress } from "@/components/dataset/WarmupModal";
import { isTerminalWarmupStatus } from "@/lib/warmupJob";

// A warmup whose worker died (containers recreated mid-run) used to stay
// "running" forever: Cancel set a flag nothing read, the next poll flipped the
// UI back to running, and every reload reattached to the ghost. The backend
// now reports such runs as "interrupted", and cancel returns the real state.
describe("warmup terminal statuses", () => {
  it.each(["completed", "cancelled", "failed", "interrupted"])("%s is terminal", (s) => {
    expect(isTerminalWarmupStatus(s)).toBe(true);
  });

  it.each(["running", "cancelling", "not_found", "", undefined, null])(
    "%s is not terminal",
    (s) => {
      expect(isTerminalWarmupStatus(s)).toBe(false);
    },
  );
});

describe("WarmupModal status labels", () => {
  const progress = (status: string): WarmupProgress => ({
    completed: 4,
    total: 100,
    current_file: "sample.mp3",
    status,
    percent: 4,
  });

  const renderModal = (status: string) =>
    render(
      <WarmupModal
        isOpen
        onClose={jest.fn()}
        dataset="common-voice"
        model="whisper-base"
        warmupJobId="warmup_abc"
        warmupProgress={progress(status)}
        isStarting={false}
        onStartWarmup={jest.fn()}
        onCancelWarmup={jest.fn()}
        onMinimize={jest.fn()}
      />,
    );

  it("labels a run whose worker died as interrupted, not cancelled", () => {
    renderModal("interrupted");
    expect(screen.getByText("Warmup Interrupted")).toBeInTheDocument();
    expect(screen.getByText(/worker running this warmup stopped/i)).toBeInTheDocument();
    expect(screen.queryByText("Cancel Warmup")).not.toBeInTheDocument();
  });

  it("shows that a cancel is in flight while the worker finishes its step", () => {
    renderModal("cancelling");
    expect(screen.getByText("Cancelling Warmup…")).toBeInTheDocument();
    expect(screen.queryByText("Cancel Warmup")).not.toBeInTheDocument();
  });

  it("still offers Cancel on a running warmup", () => {
    renderModal("running");
    expect(screen.getByText("Cancel Warmup")).toBeInTheDocument();
  });
});
