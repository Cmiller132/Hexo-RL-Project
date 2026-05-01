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
      suite/
        index.tsx              # /suite — overview hero, leaderboard, events feed
        trial.tsx              # /suite/trials/:trialId — full trial detail w/ tabs
        family-space.tsx       # /suite/family-space — search-space inspector
        scheduler.tsx          # /suite/scheduler — autotune scheduler state
        runtime-sweep.tsx      # /suite/runtime-sweep — cross-trial sweep results
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
    sessions.py · axis.py · arena.py · replay.py
    suite/
      __init__.py              # router aggregator; mounted as /api/suite
      status.py                # current activity + totals
      trials.py                # leaderboard + per-trial detail
      events.py                # event feed + SSE stream
      family_space.py          # tuning/recipes.py + tuning/family_spaces.py
      scheduler.py             # tuning/scheduler.py state
      runtime_sweep.py         # tuning/runtime_sweep.py history per trial
  schemas/                     # pydantic request/response models per route
  services/                    # existing modules stay where they are
```

## Suite & autotuning surface

The Suite tab today crams six panels into one page driven by four endpoints
and one prop-drilled `selectedTrialId`. The refactor splits Suite into its
own `/suite/*` route group and exposes the autotuning data that already
exists in `Python/src/hexorl/tuning/` and the per-trial filesystem layout
(`<run_root>/trials/<trial_id>/{trial.json,LATEST.json,scores.jsonl,
events.jsonl,summary.jsonl,dashboard.sqlite3}`) but is currently invisible
or buried in the trial-detail panel.

### Routes and what each surfaces

**`/suite` — overview**
- Hero card: stage, current trial, current model, best trial, best score,
  positions/sec, total positions, total games, workers alive/total, last
  event + time. (Today: `SuitePanel` hero strip.)
- Live activity strip: `current_activity.action`, supervisor decisions,
  click-through to the inspected trial.
- Leaderboard: best checkpoints with rank, score, scheduler_score, epoch,
  loadability, path; click → `/suite/trials/:id`.
- Trial table: sortable + filterable (by stage, family, pruned status,
  architecture, score range). Today the table is unsorted/unfiltered.
- Recent events feed (suite-scoped) with stage transitions, sweeps,
  pruning, epoch completions.

**`/suite/trials/:trialId` — trial detail (tabbed)**
- **Architecture** tab: family, architecture, channels, blocks, heads,
  graph_token_set, graph_token_budget, graph_layers, sparse_policy,
  candidate_budget, sparse_prior_stage/mix. Pull from
  `model_metadata` / `trial.family` / `static`. Add a "compare to family
  space" diff showing which params are sweep-driven vs. fixed.
- **Search & self-play** tab: mcts_simulations, pcr_low_sims,
  pcr_low_sim_prob, c_puct, dirichlet_*, max_game_moves, states/games per
  epoch, pair strategy.
- **Runtime** tab: num_workers, batch_size_per_worker, max_batch_size,
  max_wait_us, fp16, cpu_threads, compile_model/inference,
  `runtime_sweep.selected`, plus a **runtime sweep history chart** (probe
  results sorted by positions/sec with stability flag — currently only
  the chosen value is displayed).
- **Trainer** tab: latest epoch, loss_total, per-head losses
  (loss_policy, loss_value, loss_sparse_policy, loss_pair_policy,
  loss_regret_*, loss_entropy), top-1 accuracies, plus a **loss curve**
  (read `metrics_json` series across epochs from
  `dashboard.sqlite3` — already persisted, just not graphed).
- **Score history** tab: line chart of `scores.jsonl`
  (scheduler_score + score over epochs) with the prune/promote events
  overlaid.
- **Events** tab: `events.jsonl` for this trial with severity coloring
  (runtime_sweep_*, prune, stage_change, epoch_complete).
- **Checkpoints** tab: same `checkpoints` table currently in the
  `Checkpoints` global tab, scoped to this trial; click → load into the
  Replay or Play views.
- **Raw config** disclosure: existing `<details>` block.

**`/suite/family-space` — search-space inspector** *(new)*
- Reads from `hexorl.tuning.family_spaces` and
  `hexorl.tuning.recipes`. Shows every registered family, its declared
  param ranges/choices, the recipes that target it, and which trials
  have been spawned from each cell. This makes "what is the autotuner
  searching over and what has it tried" answerable from the UI; today
  it's only knowable by reading the source.

**`/suite/scheduler` — scheduler state** *(new)*
- Reads from `hexorl.tuning.scheduler` and the suite manifest. Surfaces
  current stage, planned stages, scheduler decisions (last sweep,
  promotion/pruning thresholds, queue), and budget remaining (positions,
  wall time, GPU time). Pulls `manifest.json` host/args data that's
  fetched today but only displayed as a path string at the bottom of
  the hero card.

**`/suite/runtime-sweep` — cross-trial sweep results** *(new)*
- Reads `runtime_sweep` payloads across trials. Shows a scatter of
  (workers, batch) → positions/sec with stability flags, grouped by
  family/architecture. The current dashboard only shows the per-trial
  selected setting in a string — the sweep data needed to *audit* the
  selection is not exposed.

### Backend endpoints to add

These live under `Python/src/hexorl/dashboard/routes/suite/` (see
backend layout above). Codex implements the body; the data sources are
listed for each.

| Endpoint | Source | Purpose |
|---|---|---|
| `GET /api/suite/status` | existing `_suite_status` | unchanged surface; move into `routes/suite/status.py` |
| `GET /api/suite/manifest` | `<run_root>/manifest.json` | host, args, scheduler config, budget; today only the path string is shown |
| `GET /api/suite/family-space` | `hexorl.tuning.family_spaces`, `hexorl.tuning.recipes` | declared search space per family |
| `GET /api/suite/scheduler` | `hexorl.tuning.scheduler`, manifest, state | current/planned stages, thresholds, budget remaining |
| `GET /api/suite/runtime-sweep` | aggregate of per-trial `runtime_sweep` blobs | cross-trial sweep results |
| `GET /api/suite/trials` | existing `_suite_trials` | unchanged |
| `GET /api/suite/trials/{id}` | existing `_suite_trial_detail`, *minus* the bundled scores/events/summaries (split below) | leaner core detail; scores/events become their own endpoints so the trial page can lazy-load tabs |
| `GET /api/suite/trials/{id}/scores` | `scores.jsonl` | full score history for the chart |
| `GET /api/suite/trials/{id}/events` | `events.jsonl` | trial-scoped event feed |
| `GET /api/suite/trials/{id}/loss-curve` | `dashboard.sqlite3` metrics rows | per-epoch metrics series for the trainer chart |
| `GET /api/suite/trials/{id}/runtime-sweep` | per-trial sweep blob | sweep probe history |
| `GET /api/suite/events` | existing | unchanged surface; backed by SSE in Step 7 |
| `GET /api/suite/events/stream` | tail of `events.jsonl` (SSE) | live event feed (Step 7) |
| `GET /api/suite/status/stream` | recompute on event (SSE) | live hero card (Step 7) |

The split of the current monolithic `/api/suite/trials/{id}` payload is
deliberate: today the panel always fetches scores + summary + events +
checkpoints together even when the user only wants Architecture. Tabbed
lazy-loading via TanStack Query (`enabled: tab === "scores"`) replaces
that.

### Charts to add

These all consume data we already persist; they just aren't visualized:

- Score history (per trial): `scores.jsonl` → line chart with prune /
  promote markers from `events.jsonl`.
- Loss curves (per trial): `dashboard.sqlite3.metrics` → faceted line
  chart (one panel per head loss + one for `loss_total`).
- Runtime sweep scatter (per trial and cross-trial): probe results from
  `runtime_sweep` → scatter or small-multiples by family.
- Stage timeline (suite-wide): event timestamps grouped by stage →
  Gantt-style strip showing how long each stage took.

For chart rendering: `recharts` (low-friction, declarative, fits
TanStack Query data shape). Add to `dependencies` in Step 4. Avoid d3
direct or chart.js — too heavy / too imperative for this dashboard's
needs.

### Filtering and sorting

Trial table gets URL-bound filter state (`?family=graph_hybrid&pruned=0&sort=score`)
so Suite views are deep-linkable. Implement once in `routes/suite/index.tsx`;
do not introduce a global filter store.

## File-size budgets

- No frontend file >300 lines.
- No backend route file >250 lines.
- `app.py` ≤ 100 lines after Step 5.
- `main.tsx` ≤ 50 lines after Step 4.
- Each `routes/suite/*.tsx` and `routes/suite/trial-tabs/*.tsx` ≤ 250
  lines.
- Each `Python/src/hexorl/dashboard/routes/suite/*.py` ≤ 200 lines (route
  module) and `services/suite.py` ≤ 400 lines (helpers, replaces the
  `_suite_*` block in today's `app.py`).
- One route = one file on each side, with one exception: the Suite
  trial-detail tabs share a parent route (`routes/suite/trial.tsx`) and
  are split into a `trial-tabs/` directory.

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
- **Suite is its own subdirectory** (see "Suite & autotuning surface").
  In this step, only carry over the existing Suite content into
  `routes/suite/index.tsx` and `routes/suite/trial.tsx`. The new family-
  space, scheduler, and runtime-sweep routes ship in their own follow-up
  PRs after Step 5 lands their endpoints — keep the move mechanical.
- Move shared sub-components (`Panel`, `Board`, `Table`, `KeyValue`,
  `MetricCard`, `TrialDetail`) into `src/components/`. `TrialDetail`
  splits into per-tab subcomponents under
  `routes/suite/trial-tabs/{architecture,search,runtime,trainer,scores,events,checkpoints}.tsx`.
- Add `recharts` to `dependencies` for the score / loss / sweep charts
  introduced in the new Suite tabs.
- `main.tsx` shrinks to mounting `<App/>`. `app.tsx` holds the router and
  the layout chrome (header, KPI row, error banner).
- Behavioral changes are limited to: lazy-loading per Suite trial tab,
  URL-bound trial selection (`/suite/trials/:id?tab=architecture`), and
  URL-bound trial-table filters. No data-shape changes.

**Done when:** `wc -l src/main.tsx` < 50, `wc -l src/app.tsx` < 200, and
no file in `routes/suite/` exceeds 250 lines.

### Step 5 — Split `app.py` per concern

**Goal:** the 1431-line backend file dies and the autotune surface gets
its missing endpoints.

5a. **Mechanical split.** Create `Python/src/hexorl/dashboard/routes/`
and `schemas/`. Move 5–7 routes per commit into a router module. Order:
`health.py`, `runs.py`, `games.py`, `metrics.py`, `checkpoints.py`,
`sessions.py`, `axis.py`, `arena.py`, `replay.py`. The `BaseModel`
request classes move next to their route file under `schemas/`. Re-run
`gen:api` after each move; the frontend should keep working unchanged.

5b. **Suite split + extraction.** Suite gets a subpackage
(`routes/suite/`) instead of a single file because it owns more
endpoints than any other concern. In this commit:
- Move existing endpoints (`status`, `trials`, `trials/{id}`,
  `best-checkpoints`, `events`) into the corresponding files in
  `routes/suite/`.
- Lift the `_suite_*` helpers currently buried in `app.py:700-900` into
  `Python/src/hexorl/dashboard/services/suite.py`. The route files
  import from `services/suite.py`; no business logic stays in route
  modules.
- **Split `/api/suite/trials/{id}`**: today it bundles
  scores + summary + events + checkpoints. Trim its response to the core
  detail (trial + state + latest + checkpoint_metadata + architecture)
  and add the four lazy-load endpoints listed in "Suite & autotuning
  surface": `/scores`, `/events`, `/loss-curve`, `/runtime-sweep`. Bump
  the response model name (e.g. `SuiteTrialDetailV2`) and update the
  OpenAPI snapshot test (Step 6.1).

5c. **New autotune endpoints.** Add `GET /api/suite/manifest`,
`GET /api/suite/family-space`, `GET /api/suite/scheduler`,
`GET /api/suite/runtime-sweep`. Each is < 50 lines and reads from the
existing `tuning/` modules and the per-trial filesystem layout. Pydantic
response models for each go in `schemas/suite.py`. No new business logic
in `tuning/` — this step only exposes what's already computed.

`app.py` becomes `FastAPI()` + middleware + the list of
`app.include_router(...)` calls.

**Done when:** `wc -l Python/src/hexorl/dashboard/app.py` ≤ 100,
no file in `routes/suite/` exceeds 250 lines,
the contract test (Step 6.1) passes,
and the new endpoints have at least one test fixture each (a synthetic
`run_root` directory with a couple of trials, used by Step 6.2).

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

**Goal:** stop polling for things that should be pushed. Suite is the
top beneficiary because trial events (pruning, stage changes, runtime
sweep results, epoch completions) are inherently push-shaped and today
only surface after the next 15-second poll.

- Backend: add three SSE endpoints, all backed by tailing the existing
  JSONL files (no new persistence):
  - `GET /api/suite/status/stream` — recompute `_suite_status` and emit
    on any change to `<run_root>/state.json` or `events.jsonl`.
  - `GET /api/suite/events/stream` — emit each new line appended to
    `<run_root>/events.jsonl`.
  - `GET /api/arena/history/stream` — emit each new arena result.
  Use a small `tail_jsonl()` helper in `services/suite.py` that opens
  the file, seeks to end, and yields appended lines via async sleep.
- Frontend: add `useEventSource(url, { onMessage })` hook that
  integrates with TanStack Query's `setQueryData` so the rest of the
  app keeps reading from the same query key. Replace the 15-second
  polling on the three endpoints above. Auto-reconnect on disconnect
  with exponential backoff (max 30 s).
- Per-trial event tab (`/suite/trials/:id?tab=events`) optionally
  consumes the same `/api/suite/events/stream` and filters client-side
  to the selected `trial_id`. Don't add a per-trial SSE endpoint; the
  shared stream is fine for the cardinality involved (single supervisor).
- Other endpoints stay on TanStack Query polling — they are low-cost.

**Done when:** the network tab shows no `/api/suite/status`,
`/api/suite/events`, or `/api/arena/history` requests after the initial
connection while the page is open, and a deliberately injected event in
`events.jsonl` appears in the UI within 1 s.

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
Step 1  generate types                                (1 day)
Step 2  TanStack Query migration                      (2-3 days)
Step 3  routes instead of tabs                        (1 day)
Step 4  split main.tsx (incl. suite/ subroutes)       (2 days; mechanical)
Step 5  split app.py + new autotune endpoints         (3-4 days)
        5a mechanical split            (1 day)
        5b suite split + detail trim   (1-2 days)
        5c new autotune endpoints      (1 day)
Step 6  tests (contract + component + E2E)           (3 days)
Step 7  SSE for suite + arena                         (2 days)
Step 8  error boundaries + health banner              (1 day)
Step 9  zod runtime validation                        (1-2 days)
────────────────────────────────────────────────────
Total   ≈ 16-19 days of focused work
```

Steps 1–5 deliver the maintainability win and the new autotune surface.
Steps 6–9 deliver the reliability and robustness wins. If you have to
stop early, stopping after Step 5 leaves a usable autotune view; stopping
after Step 6 leaves the project meaningfully better and CI-protected.

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

- Dashboard features that require new backend persistence. The Suite
  expansion is bounded to **exposing data already written to disk by
  `tuning/`, `selfplay/`, and the supervisor**. If a chart requires data
  that isn't currently persisted, flag it and stop — that needs a
  separate plan.
- Visualizations outside the Suite expansion (no new charts on Charts,
  Games, Arena, Axis Lab tabs).
- Changes to `Python/src/hexorl/tuning/`. Step 5c only *reads* from it.
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
- If Step 5c reveals that a `tuning/` module's state isn't persisted in
  a form the dashboard can read (e.g. scheduler state lives only in
  memory in the supervisor process), stop and report. The fix is a
  small persistence change in the supervisor; do not "scrape it from
  events.jsonl by string match" as a workaround.
- If trimming the `/api/suite/trials/{id}` payload (Step 5b) breaks an
  external consumer (no current ones, but check before shipping), keep
  the old shape under `/api/suite/trials/{id}?include=all` and have the
  new dashboard call the trimmed endpoint by default. Don't keep the
  fat shape as the default.
- If Step 7 produces unstable connections in the dev environment, ship
  Steps 1–6+8+9 first and keep polling for the live tabs. SSE is a
  performance/reliability nicety, not a blocker.
