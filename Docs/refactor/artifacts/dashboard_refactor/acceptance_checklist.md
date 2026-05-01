# Dashboard Refactor Acceptance Checklist

This checklist maps every step in `Docs/refactor/dashboard_refactor_plan.md` to the evidence required before that step can be claimed complete. Checkboxes must remain open until implementation, tests, audits, and artifacts are command-backed.

## Step 1 - Generate Types From Backend OpenAPI

| Evidence field | Required proof |
|---|---|
| Implementation | `openapi-typescript` added to dashboard frontend dev dependencies; `npm run gen:api` added; generated `src/api/schema.d.ts` ignored or otherwise handled as planned; one small `AnyRow` usage replaced with generated OpenAPI schema type. |
| Tests/checks | `npm run gen:api`; `tsc --noEmit`; intentional backend model field rename or equivalent type-break proof demonstrating `tsc` fails. |
| Artifacts | OpenAPI generation command transcript; note of chosen converted component/schema. |
| Not complete until | Type generation works against a running/dumped backend OpenAPI schema and one route/component is genuinely typed. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 2 - Introduce TanStack Query

| Evidence field | Required proof |
|---|---|
| Implementation | `@tanstack/react-query` and devtools installed; app wrapped in `QueryClientProvider`; `src/api/hooks.ts` created; fetch/dedupe calls migrated into `useQuery`/`useMutation`; obsolete in-flight refs removed after last use. |
| Tests/checks | `rg -n "useEffect\\([^)]*fetch|fetch\\(" Python/dashboard_frontend/src` reviewed for direct fetches; `npm run test -- --run` or available frontend test command; `tsc --noEmit`. |
| Artifacts | Import/search audit showing no direct `fetch` in `App` after migration; command transcript. |
| Not complete until | `App` has no `useEffect` that calls `fetch` directly and polling is limited to suite status/arena history for this step. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 3 - Routes Instead Of Tab State

| Evidence field | Required proof |
|---|---|
| Implementation | `react-router-dom` v6 installed; tab state replaced by route elements and `NavLink`; selected run/game/turn represented in URL search params. |
| Tests/checks | Deep-link smoke for `/replay?game=42&run=foo&turn=12`; `tsc --noEmit`; route/component tests where available. |
| Artifacts | Browser or E2E transcript proving refresh preserves non-Suite route state. |
| Not complete until | Refreshing any non-Suite tab returns to the same view and URLs are shareable. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 4 - Split `main.tsx` Per Route

| Evidence field | Required proof |
|---|---|
| Implementation | Route components moved under `src/routes/`; Suite moved under `src/routes/suite/`; shared components moved under `src/components/`; `main.tsx` only mounts `<App/>`; `app.tsx` owns router/layout; `recharts` added for Suite charts. |
| Tests/checks | `wc -l Python/dashboard_frontend/src/main.tsx`; `wc -l Python/dashboard_frontend/src/app.tsx`; route module size audit; `tsc --noEmit`; frontend tests; dashboard smoke. |
| Artifacts | Size audit output proving `main.tsx` < 50, `app.tsx` < 200, no frontend file > 300, and no `routes/suite/*.tsx` or `routes/suite/trial-tabs/*.tsx` > 250. |
| Not complete until | Data-shape changes are limited to planned lazy-loading/URL-bound state and size budgets pass. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 5 - Split `app.py` Per Concern

| Evidence field | Required proof |
|---|---|
| Implementation | `dashboard/routes/` and `dashboard/schemas/` created; route modules split by concern; Suite helpers lifted into `services/suite.py`; trial detail payload trimmed; lazy-load trial endpoints added; manifest/family-space/scheduler/runtime-sweep endpoints added. |
| Tests/checks | Backend OpenAPI contract snapshot test; endpoint fixture tests with synthetic `run_root`; `pytest Python/tests/dashboard`; `wc -l Python/src/hexorl/dashboard/app.py`; backend route module size audit. |
| Artifacts | Size audit proving `app.py` <= 100, no backend route file > 250, no `routes/suite/*.py` > 200, and `services/suite.py` <= 400. |
| Not complete until | `app.py` is middleware plus `include_router` calls and new autotune endpoints each have at least one fixture-backed test. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 6 - Add Tests

| Evidence field | Required proof |
|---|---|
| Implementation | Backend OpenAPI snapshot test and `Docs/refactor/artifacts/dashboard_openapi_snapshot.json`; Vitest/RTL/MSW setup; one smoke test per frontend route file; Playwright smoke E2E. |
| Tests/checks | `npm run test`; `npm run e2e`; `pytest Python/tests/dashboard`; CI config or transcript showing these gate merges. |
| Artifacts | OpenAPI snapshot artifact; frontend coverage report or summary showing target coverage for `src/routes/`; E2E transcript. |
| Not complete until | All three suites run in CI and gate merges. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 7 - Switch Live Tabs To SSE

| Evidence field | Required proof |
|---|---|
| Implementation | Suite status/events SSE endpoints; arena history SSE endpoint; `tail_jsonl()` helper; frontend `useEventSource` hook integrated with TanStack Query cache; polling removed from the three live endpoints after initial connection. |
| Tests/checks | Backend SSE tests; frontend hook/component tests; browser/network or Playwright proof that `/api/suite/status`, `/api/suite/events`, and `/api/arena/history` are not repeatedly polled after initial connection. |
| Artifacts | Injected `events.jsonl` event transcript proving UI update within 1 second. |
| Not complete until | SSE reconnect/backoff is implemented and live tabs update without 15-second polling. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 8 - Error Boundaries And Health-Aware UI

| Evidence field | Required proof |
|---|---|
| Implementation | `react-error-boundary` added; each route wrapped; `ConnectionBanner` driven by `useHealth`; non-essential queries pause while disconnected; loading skeletons added. |
| Tests/checks | Component tests for thrown route render error, disconnected backend, query pausing, and loading skeletons; `tsc --noEmit`; frontend tests. |
| Artifacts | Browser/E2E transcript showing backend kill produces one banner and other tabs remain usable after one route crash. |
| Not complete until | Dead backend does not produce repeated error flashes and isolated render failure does not blank the dashboard. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Step 9 - Runtime Validation At Trust Boundary

| Evidence field | Required proof |
|---|---|
| Implementation | `zod` added; `src/api/schemas.ts` created; every `useQuery` hook parses fetched data with a schema before returning; optional generated-schema sync tool added if selected. |
| Tests/checks | Unit/component test with deliberately malformed backend/MSW response; `npm run test`; `tsc --noEmit`; schema sync CI check if generator is used. |
| Artifacts | Transcript showing malformed response produces a Zod error and route error-boundary fallback. |
| Not complete until | Schema drift fails loudly at the UI trust boundary. |

Status: [x] Claimed complete with evidence in `agent_completion_packet.md` and `command_transcripts.md`.

## Cross-Step Required Gates

- [x] `cd Python/dashboard_frontend && npm run gen:api && npx tsc --noEmit && npm run test`
- [x] `PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/dashboard`
- [x] Size budget audit after Step 4 and Step 5.
- [x] Import/deletion audit for removed direct-fetch/dedupe paths and split backend route ownership.
- [x] Local gate evidence for frontend, backend, size, OpenAPI snapshot, and E2E checks; CI dashboard job now runs build, Vitest, and E2E, while the Python shard includes dashboard tests.

No skipped, deferred, flaky, quarantined, or manual-only requirement is claimed complete by this checklist.
