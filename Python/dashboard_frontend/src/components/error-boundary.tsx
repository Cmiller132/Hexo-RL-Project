import type { FallbackProps } from "react-error-boundary";

export function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <main className="app">
      <div className="error">
        <strong>Dashboard error</strong>
        <p>{message}</p>
        <button onClick={resetErrorBoundary}>Reset</button>
      </div>
    </main>
  );
}
