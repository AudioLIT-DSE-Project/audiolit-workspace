import { Component, ErrorInfo, ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Without this, any uncaught render-time exception anywhere in the tree
 * unmounts the entire app to a blank white page - React 16+'s default
 * behaviour with no error boundary. This has bitten the deepfake (ADD) flow
 * in particular: a backend response shaped slightly differently than a
 * component assumed (e.g. a missing `probabilities` map) throws during
 * render, and the whole SPA disappears with nothing in the UI to explain why.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Uncaught render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background p-6">
          <div className="max-w-md text-center space-y-3">
            <h1 className="text-lg font-semibold text-foreground">Something went wrong</h1>
            <p className="text-sm text-muted-foreground">
              {this.state.error.message || "An unexpected error occurred while rendering this page."}
            </p>
            <button
              type="button"
              onClick={() => {
                this.setState({ error: null });
                window.location.reload();
              }}
              className="text-sm px-3 py-1.5 rounded border border-border hover:bg-accent"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
