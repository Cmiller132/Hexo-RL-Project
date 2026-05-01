# Dashboard Refactor Artifact Manifest

Artifact scope: implementation and evidence layer for `Docs/refactor/dashboard_refactor_plan.md`.

This directory records dashboard refactor verification planning and command-backed completion evidence.

## Files

| File | Purpose | Completion state |
|---|---|---|
| `acceptance_checklist.md` | Maps plan Steps 1-9 to required implementation, test, audit, documentation, and artifact evidence fields. | Drafted for implementers to fill as command-backed evidence exists. |
| `command_transcripts.md` | Records commands personally run by this QA/docs worker and their exit statuses. | Filled with current QA/docs commands. |
| `size_audit.md` | Defines exact commands for current and final size-budget proof for `main.tsx`, `app.py`, frontend route modules, and backend route modules. | Drafted with current baseline output and future gate commands. |
| `agent_completion_packet.md` | Reports closed steps, changed consumers, deleted legacy paths, tests, artifacts, performance evidence, and blockers. | Filled for the implemented dashboard refactor slice. |

## Source Plan Reviewed

- `Docs/refactor/dashboard_refactor_plan.md`

## Evidence Policy

- A step is not complete until its checklist row has command-backed evidence.
- Manual-only, skipped, flaky, quarantined, or deferred checks cannot close a requirement.
- Size-budget claims require `wc -l` command output archived in `command_transcripts.md` or a linked CI artifact.
- Import/deletion claims require `rg`/`git grep` command output archived in `command_transcripts.md` or a linked CI artifact.

## Final Size Evidence

- `Python/dashboard_frontend/src/main.tsx`: 30 lines.
- `Python/dashboard_frontend/src/app.tsx`: 110 lines.
- `Python/dashboard_frontend/src/styles.css`: 190 lines.
- `Python/src/hexorl/dashboard/app.py`: 81 lines.
- `Python/src/hexorl/dashboard/services/suite.py`: 395 lines.
- All `Python/src/hexorl/dashboard/routes/suite/*.py`: <= 65 lines.
- All `Python/dashboard_frontend/src/routes/suite/**/*.tsx`: <= 94 lines.

## Verification Summary

- `npm run gen:api`: exit 0.
- `npx tsc --noEmit`: exit 0.
- `npm run test`: exit 0.
- `npm run build`: exit 0.
- `npm run e2e`: exit 0.
- `PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/dashboard`: exit 0.
- `.github/workflows/ci.yml` dashboard job now gates frontend build, Vitest, and Playwright E2E; the Python shard already gates `Python/tests/dashboard`.
- `npm audit --audit-level=moderate`: exit 1 because Vite 5 depends on an esbuild version with a moderate dev-server advisory; the automated fix is a breaking Vite major upgrade.
