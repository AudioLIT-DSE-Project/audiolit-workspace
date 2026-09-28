// Storage helpers for the first-run walkthrough, kept out of the component
// module so that file exports only a component and stays Fast Refresh safe.

// first-visit open only - the toolbar's "Quick start" button always reopens
// this regardless of the dismissal state.
export const QUICKSTART_DISMISSED_KEY = "audiolit.quickstart.dismissed";

// localStorage throws in a private window with storage blocked - a failed
// read/write here must not crash the app, it should just behave as if
// nothing was ever dismissed/saved.
export function readQuickStartDismissed(): boolean {
  try {
    return window.localStorage.getItem(QUICKSTART_DISMISSED_KEY) === "true";
  } catch {
    return false;
  }
}

export function writeQuickStartDismissed(value: boolean): void {
  try {
    if (value) {
      window.localStorage.setItem(QUICKSTART_DISMISSED_KEY, "true");
    } else {
      window.localStorage.removeItem(QUICKSTART_DISMISSED_KEY);
    }
  } catch {
    // Private window / storage disabled - nothing we can persist.
  }
}
