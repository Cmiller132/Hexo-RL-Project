# Dashboard Refactor Plan

Goal: make the Hexo-RL dashboard reliable, maintainable, and robust without a
big-bang rewrite. Every step in this plan is independently shippable.

## Current state (baseline)

**Frontend** (`Python/dashboard_frontend/`):

- Single file: [src/main.tsx](../../Python/dashboard_frontend/src/main.tsx)
  is **1706 lines**, holds 51 components.
- Single stylesheet: `src/styles.css` is 848 lines.
- 108 instances of `type AnyRow = Record<string, any>` — no domain typing.
- 34 hardcoded `/api/...` strings sprinkled across components.
- 18 `useEffect` blocks for fetch/poll/autoplay; manual `useRef` flags
  (`loadInFlight`, `runDetailInFlight`) used to dedupe in-flight requests.
- Tabs are stored in `useState` ([main.tsx:46](../../Python/dashboard_frontend/src/main.tsx)),
  not in the URL — refresh resets state, no deep linking.
- 15-second polling of nine endpoints regardless of which tab is open.
- Replay autoplay polls every 650 ms via `setInterval`.
- One global error sink: `<div className="error">{lastError}</div>`.
- No tests. No `*.test.tsx`, no `*.spec.tsx`, no Playwright.
- Dependencies: React 18, Vite 5, TypeScript 5, lucide-react. No router,
  no query lib, no test runner.

**Backend** (`Python/src/hexorl/dashboard/`):

- [app.py](../../Python/src/hexorl/dashboard/app.py) is **1431 lines** with
  42 routes and 9 inline `BaseModel` request classes. No `APIRouter` split.
- Sibling modules (`db.py`, `play.py`, `replay.py`, `render.py`,
  `inspection_services.py`, …) implement services and are reasonable in
  size; the sprawl is concentrated in `app.py`.

## Target stack

| Concern | Pick | Notes |
|---|---|---|
| Data fetching | TanStack Query v5 | Replaces ad-hoc `fetch` + `loadInFlight` plumbing. |
| Backend types | `openapi-typescript` | Read `/openapi.json`, emit `schema.d.ts`. |
| Runtime validation | Zod | Validate at the trust boundary even with generated types. |
| Routing | React Router v6 (data routers) | Tabs become URLs; deep-linkable. |
| Live updates | FastAPI SSE + `EventSource` | For Suite status and Arena history. |
| UI state | Zustand | Tiny. Server state stays in TanStack Query. |
| Component tests | Vitest + React Testing Library + MSW | One MSW handler set, reused for tests and dev mocks. |
| Smoke E2E | Playwright | One spec per route, runs in CI. |

Avoid: Redux, MobX, Recoil.

## Target layout

**Frontend:**

```
Python/dashboard_frontend/
  src/
    main.tsx                   # mount <App/>, providers
    app.tsx                    # router + layout chrome
    api/
      schema.d.ts              # generated; gitignored
      client.ts                # fetch wrapper using generated types
      schemas.ts               # zod runtime schemas
      hooks.ts                 # useRuns, useGame, useSuiteStatus, …
    routes/
      suite.tsx
      charts.tsx
      games.tsx
      replay.tsx
      play.tsx
      arena.tsx
      checkpoints.tsx
      axis-lab.tsx
    components/
      board.tsx · panel.tsx · kpi-row.tsx · table.tsx
      error-boundary.tsx · connection-banner.tsx
    state/
      ui-store.ts              # zustand for UI-only state
    test/
      msw-handlers.ts · setup.ts
  e2e/
    smoke.spec.ts
  vitest.config.ts
  playwright.config.ts
  package.json
```

**Backend:**

```
Python/src/hexorl/dashboard/
  app.py                       # ~50 lines: middleware + include_router calls
  routes/
    health.py · runs.py · games.py · metrics.py · checkpoints.py
    sessions.py · suite.py · axis.py · arena.py · replay.py
  schemas/                     # pydantic request/response models per route
  services/                    # existing modules stay where they are
```

## File-size budgets

- No frontend file >300 lines.
- No backend route file >250 lines.
- `app.py` ≤ 100 lines after Step 5.
- `main.tsx` ≤ 50 lines after Step 4.
- One route = one file on each side.

## Migration steps

Each step is independently shippable. Do not start a step until the previous
step's tests pass. Do not skip the test-adding steps.

### Step 1 — Generate types from the backend OpenAPI

**Goal:** kill `AnyRow` for one route and prove the dev loop works.

- Add `openapi-typescript` to `devDependencies`.
- Add `npm run gen:api` that hits `http://127.0.0.1:<port>/openapi.json`
  and writes `src/api/schema.d.ts`. Add `schema.d.ts` to `.gitignore`.
- Add to backend startup docs: dashboard build expects the API server to
  be reachable for type generation, OR add a CI step that runs the API
  briefly to dump `openapi.json` to disk.
- Convert ONE component's `AnyRow` to `components["schemas"]["…"]` from
  the generated file. Pick something small: the KPI row or the
  `Health` response. Land it.

**Done when:** `tsc --noEmit` fails if you rename a field on the chosen
backend model.

### Step 2 — Introduce TanStack Query

**Goal:** stop hand-managing fetch/dedupe.

- Add `@tanstack/react-query` and `@tanstack/react-query-devtools`.
- Wrap `<App/>` with `QueryClientProvider`. Default options:
  `staleTime: 5_000`, `refetchOnWindowFocus: true`,
  `retry: 2`.
- Create `src/api/hooks.ts`. Migrate fetches one at a time into
  `useQuery`/`useMutation` hooks: `useHealth`, `useRuns`, `useMetrics`,
  `useGames`, `useGame`, `useReplay`, `usePosition`,
  `useCheckpoints`, `useArenaHistory`, `useAxisPrototypes`,
  `useSuiteStatus`, `useSuiteTrials`, etc.
- Delete `loadInFlight` and `runDetailInFlight` refs once their last
  caller is gone.
- Suite status and arena history keep `refetchInterval: 15_000`
  for now; everything else stops polling.

**Done when:** `App` has no `useEffect` that calls `fetch` directly.

### Step 3 — Routes instead of tab state

**Goal:** URLs become the source of truth for navigation.

- Add `react-router-dom` v6.
- Replace `tabs` array driven by `useState` with `<Routes>` mounting one
  child per tab. Tab buttons become `<NavLink>`s.
- Move `selectedRun`, `selectedGame`, and `turn` into URL search params
  via `useSearchParams`. Drop the corresponding `useState`s.
- Verify deep links: `/replay?game=42&run=foo&turn=12` should load the
  replay viewer at turn 12.

**Done when:** browser refresh on any non-Suite tab returns to the same
view, and the URL is shareable.

### Step 4 — Split `main.tsx` per route

**Goal:** the 1706-line file dies.

- For each tab, move its function (`SuitePanel`, `Charts`, `Games`,
  `Replay`, `PlayPanel`, `ArenaPanel`, `CheckpointPanel`, `AxisLab`) into
  its own file under `src/routes/`.
- Move shared sub-components (`Panel`, `Board`, `Table`, `KeyValue`,
  `MetricCard`, `TrialDetail`) into `src/components/`.
- `main.tsx` shrinks to mounting `<App/>`. `app.tsx` holds the router and
  the layout chrome (header, KPI row, error banner).
- No behavioral changes in this step. Move + import only.

**Done when:** `wc -l src/main.tsx` < 50 and `wc -l src/app.tsx` < 200.

### Step 5 — Split `app.py` per concern

**Goal:** the 1431-line backend file dies.

- Create `Python/src/hexorl/dashboard/routes/` and `schemas/`.
- Move 5–7 routes per commit into a router module. Order: `health.py`,
  `runs.py`, `games.py`, `metrics.py`, `checkpoints.py`, `sessions.py`,
  `suite.py`, `axis.py`, `arena.py`, `replay.py`.
- The `BaseModel` request classes move next to their route file under
  `schemas/`.
- `app.py` becomes `FastAPI()` + middleware + a list of
  `app.include_router(...)` calls.
- Re-run `gen:api` after each move; the frontend should keep working
  without changes.

**Done when:** `wc -l Python/src/hexorl/dashboard/app.py` ≤ 100 and the
contract test (Step 6.1) still passes.

### Step 6 — Add tests

**Goal:** future refactors stop being silent breakage.

6.1 **Contract test (backend, fast).** A pytest that boots the FastAPI
app in-process, snapshots `app.openapi()`, and diffs against a committed
`Docs/refactor/artifacts/dashboard_openapi_snapshot.json`. PR fails on
unintended schema drift; intended changes update the snapshot.

6.2 **Component tests (frontend).** Add Vitest + React Testing Library +
MSW. Add `src/test/msw-handlers.ts` with a handler for every route the
frontend calls. One smoke test per route file: render the route, assert a
key piece of data appears, assert no console error. Target: ≥ 80 % line
coverage on `src/routes/`.

6.3 **Smoke E2E.** Add Playwright. One spec, `e2e/smoke.spec.ts`: starts
the FastAPI app + Vite dev server, visits each route, asserts no console
error, asserts a key element renders. Runs in CI on every PR.

**Done when:** `npm run test`, `npm run e2e`, and `pytest
Python/tests/dashboard` all run in CI and gate merges.

### Step 7 — Switch live tabs to SSE

**Goal:** stop polling for things that should be pushed.

- Backend: add `GET /api/suite/status/stream` and
  `GET /api/arena/history/stream` as SSE endpoints emitting JSON events
  on change. Reuse existing service code; the streaming wrapper is thin.
- Frontend: add `useEventSource(url)` hook (or use TanStack Query's
  experimental SSE adapter). Replace the 15-second polling on those two
  endpoints. Auto-reconnect on disconnect.
- Other endpoints stay on TanStack Query polling — they are low-cost.

**Done when:** the network tab shows no `/api/suite/status` or
`/api/arena/history` requests after the initial connection while the page
is open.

### Step 8 — Error boundaries and health-aware UI

**Goal:** one panel crashing doesn't blank the dashboard, and a dead
backend produces one clear banner instead of an error flash every poll.

- Add `<ErrorBoundary>` (from `react-error-boundary`). Wrap each route
  element. Fallback shows "this panel failed to load — reload" and logs
  the error.
- Add `<ConnectionBanner>` driven by `useHealth`. When the query has
  errored for >10 s, show a fixed banner; pause non-essential queries
  while in the disconnected state by setting their `enabled: connected`.
- Add explicit loading skeletons for routes that previously rendered
  empty arrays during fetch.

**Done when:** killing the backend produces one banner (not 4
errors/min), and a thrown render error in one tab leaves other tabs
working.

### Step 9 — Runtime validation at the trust boundary

**Goal:** schema drift between regen cycles fails loud.

- Add `zod`. For each `useQuery` hook, wrap the fetched data in
  `Schema.parse(data)` before returning from `queryFn`. Schemas live
  in `src/api/schemas.ts` and mirror the generated types.
- Optional: use `openapi-zod-client` or `ts-to-zod` to derive the zod
  schemas from the generated types, and add a CI check that they stay
  in sync.

**Done when:** a hand-edited backend response that doesn't match the
schema produces a Zod error in the dev console and an error boundary
fallback in the UI.

## Sequencing summary

```
Step 1  generate types                         (1 day)
Step 2  TanStack Query migration               (2-3 days)
Step 3  routes instead of tabs                 (1 day)
Step 4  split main.tsx                         (1 day; mechanical)
Step 5  split app.py                           (2 days; mechanical)
Step 6  tests (contract + component + E2E)    (3 days)
Step 7  SSE for live tabs                      (1-2 days)
Step 8  error boundaries + health banner       (1 day)
Step 9  zod runtime validation                 (1-2 days)
─────────────────────────────────────────────
Total   ≈ 13-16 days of focused work
```

Steps 1–5 deliver the maintainability win. Steps 6–9 deliver the
reliability and robustness wins. If you have to stop early, stopping
after Step 6 still leaves the project meaningfully better.

## Verification gates

A PR for any step must satisfy:

```
# frontend
( cd Python/dashboard_frontend && npm run gen:api && tsc --noEmit && npm run test -- --run )
# backend
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/dashboard
# size budgets (after Step 4 / Step 5)
[ $(wc -l < Python/dashboard_frontend/src/main.tsx) -le 50 ]
[ $(wc -l < Python/src/hexorl/dashboard/app.py) -le 100 ]
```

CI must enforce the same checks.

## Out of scope

- New dashboard features (charts, panels, exports). This plan is
  structural.
- Authentication / authorization. The dashboard is local-only today; if
  that changes it deserves its own plan.
- Visual redesign. Component split is by route, not by visual hierarchy;
  styling stays where it is in `styles.css` until somebody owns a UI
  refresh.
- Mobile / responsive layout.

## Stopping conditions

- If Step 1 fails because the OpenAPI schema is incomplete (missing
  response models, untyped `dict[str, Any]` returns), fix the affected
  backend handlers' return types before continuing. Do not paper over
  with `unknown` casts.
- If Step 3 reveals a tab that needs more state than fits in URL params
  (e.g. axis lab parameter overrides), use Zustand for that piece —
  don't push everything into the URL or back into App state.
- If Step 7 produces unstable connections in the dev environment, ship
  Steps 1–6+8+9 first and keep polling for the live tabs. SSE is a
  performance/reliability nicety, not a blocker.
