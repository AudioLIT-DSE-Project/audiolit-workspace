import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import {
  QuickStartDialog,
  QUICKSTART_DISMISSED_KEY,
  readQuickStartDismissed,
  writeQuickStartDismissed,
} from "@/components/layout/QuickStartDialog";

// LIT-261: the localStorage gate must never crash the app - a private
// window throws on both read and write, and the dialog should just behave
// as "nothing was ever dismissed" in that case.
describe("QuickStartDialog localStorage gate", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("readQuickStartDismissed defaults to false when nothing is stored", () => {
    expect(readQuickStartDismissed()).toBe(false);
  });

  it("writeQuickStartDismissed(true) then readQuickStartDismissed() round-trips", () => {
    writeQuickStartDismissed(true);
    expect(readQuickStartDismissed()).toBe(true);
    expect(window.localStorage.getItem(QUICKSTART_DISMISSED_KEY)).toBe("true");
  });

  it("writeQuickStartDismissed(false) clears the key", () => {
    writeQuickStartDismissed(true);
    writeQuickStartDismissed(false);
    expect(readQuickStartDismissed()).toBe(false);
    expect(window.localStorage.getItem(QUICKSTART_DISMISSED_KEY)).toBeNull();
  });

  it("readQuickStartDismissed returns false, not throws, when localStorage.getItem throws", () => {
    jest.spyOn(window.localStorage.__proto__, "getItem").mockImplementation(() => {
      throw new Error("SecurityError: storage disabled");
    });
    expect(() => readQuickStartDismissed()).not.toThrow();
    expect(readQuickStartDismissed()).toBe(false);
    jest.restoreAllMocks();
  });

  it("writeQuickStartDismissed does not throw when localStorage.setItem throws", () => {
    jest.spyOn(window.localStorage.__proto__, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });
    expect(() => writeQuickStartDismissed(true)).not.toThrow();
    jest.restoreAllMocks();
  });
});

describe("QuickStartDialog", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("renders all four tracks and shows the deepfake track by default", () => {
    render(<QuickStartDialog open onOpenChange={() => {}} />);
    expect(screen.getByTestId("quickstart-dialog")).toBeInTheDocument();
    expect(screen.getByTestId("quickstart-tab-deepfake")).toBeInTheDocument();
    expect(screen.getByTestId("quickstart-tab-mutation")).toBeInTheDocument();
    expect(screen.getByTestId("quickstart-tab-bias")).toBeInTheDocument();
    expect(screen.getByTestId("quickstart-tab-faithfulness")).toBeInTheDocument();
  });

  it("persists the dismissal only when 'Don't show again' is checked at close", () => {
    const onOpenChange = jest.fn();
    render(<QuickStartDialog open onOpenChange={onOpenChange} />);

    fireEvent.click(screen.getByTestId("quickstart-close"));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(readQuickStartDismissed()).toBe(false);
  });

  it("persists the dismissal when 'Don't show again' is checked before closing", () => {
    const onOpenChange = jest.fn();
    render(<QuickStartDialog open onOpenChange={onOpenChange} />);

    fireEvent.click(screen.getByTestId("quickstart-dont-show-again"));
    fireEvent.click(screen.getByTestId("quickstart-close"));

    expect(readQuickStartDismissed()).toBe(true);
  });
});
