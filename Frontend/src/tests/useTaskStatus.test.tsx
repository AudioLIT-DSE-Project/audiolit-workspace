/**
 * useTaskStatus is the one place a job's outcome enters the UI. Every panel
 * that waits on a job mocks this hook in its own tests, so the hook itself -
 * including the bug where it handed consumers the worker's `{duration_s}` as
 * their result - had no coverage at all.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { useTaskStatus } from "@/hooks/useTaskStatus";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close = jest.fn();

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  emit(message: unknown) {
    act(() => {
      this.onmessage?.({ data: JSON.stringify(message) });
    });
  }
}

const latestSocket = () => MockWebSocket.instances[MockWebSocket.instances.length - 1];

const REPORT = { cohorts: [{ accent: "Hindi", mean_wer: 0.21 }] };

describe("useTaskStatus", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    MockWebSocket.instances = [];
    fetchMock.mockReset();
    (global as unknown as { WebSocket: unknown }).WebSocket = MockWebSocket;
    (global as unknown as { fetch: unknown }).fetch = fetchMock;
    jest.spyOn(console, "log").mockImplementation(() => undefined);
    jest.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("opens the socket against the API origin, not the page origin", () => {
    renderHook(() => useTaskStatus("job-1"));
    expect(latestSocket().url).toBe("ws://localhost:8000/api/ws/tasks/job-1");
  });

  it("takes the result from payload.result on SUCCESS", () => {
    const { result } = renderHook(() => useTaskStatus("job-1"));
    latestSocket().emit({ state: "SUCCESS", stage: "SUCCESS", payload: { duration_s: 1.2, result: REPORT } });

    expect(result.current.state).toBe("SUCCESS");
    expect(result.current.result).toEqual(REPORT);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("never mistakes the worker's timing payload for the result", async () => {
    fetchMock.mockResolvedValue({
      json: async () => ({ task_id: "job-1", state: "SUCCESS", result: REPORT, error: null }),
    });
    const { result } = renderHook(() => useTaskStatus("job-1"));

    // The shape a worker publishes on its own: no result on the event.
    latestSocket().emit({ stage: "SUCCESS", payload: { duration_s: 1.2 } });

    await waitFor(() => expect(result.current.state).toBe("SUCCESS"));
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/tasks/job-1/status");
    expect(result.current.result).toEqual(REPORT);
  });

  it("does not report SUCCESS before the result is available", () => {
    fetchMock.mockReturnValue(new Promise(() => undefined));
    const { result } = renderHook(() => useTaskStatus("job-1"));
    latestSocket().emit({ stage: "PROCESSING", payload: {} });
    latestSocket().emit({ stage: "SUCCESS", payload: { duration_s: 1.2 } });

    expect(result.current.state).toBe("PROCESSING");
    expect(result.current.result).toBeNull();
  });

  it("surfaces the error on FAILURE", () => {
    const { result } = renderHook(() => useTaskStatus("job-1"));
    latestSocket().emit({ state: "FAILURE", stage: "FAILURE", payload: { error: "ValueError: bad region" } });

    expect(result.current.state).toBe("FAILURE");
    expect(result.current.error).toBe("ValueError: bad region");
  });

  it("treats a fine-grained worker stage as PROCESSING", () => {
    const { result } = renderHook(() => useTaskStatus("job-1"));
    latestSocket().emit({ stage: "mutation.running", payload: { perturbation_count: 1 } });
    expect(result.current.state).toBe("PROCESSING");
  });

  it("clears the previous task's outcome when the task id changes", () => {
    const { result, rerender } = renderHook(({ id }) => useTaskStatus(id), {
      initialProps: { id: "job-1" as string | null },
    });
    latestSocket().emit({ state: "SUCCESS", payload: { result: REPORT } });
    expect(result.current.result).toEqual(REPORT);

    rerender({ id: "job-2" });
    expect(result.current.state).toBe("QUEUED");
    expect(result.current.result).toBeNull();
  });
});
