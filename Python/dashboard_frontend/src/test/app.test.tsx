import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "../app";

function renderRoute(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("dashboard routes", () => {
  for (const [path, text] of [
    ["/suite", "Autotune Suite"],
    ["/suite/trials/trial-a/architecture", "Trial trial-a"],
    ["/suite/family-space", "Family Search Space"],
    ["/suite/scheduler", "Scheduler State"],
    ["/suite/runtime-sweep", "Runtime Sweep Scatter"],
    ["/charts?run=trial-a", "Losses"],
    ["/games?run=trial-a", "Game Browser"],
    ["/arena", "Arena Spectator"],
    ["/checkpoints?run=trial-a", "Checkpoint Index"],
    ["/axis", "Axis Target Board"]
  ]) {
    it(`renders ${path}`, async () => {
      renderRoute(path);
      expect(await screen.findByText(text)).toBeInTheDocument();
    });
  }
});
