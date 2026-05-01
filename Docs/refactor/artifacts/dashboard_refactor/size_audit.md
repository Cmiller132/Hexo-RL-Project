# Dashboard Refactor Size Audit

This file defines exact commands for proving the dashboard refactor file-size budgets. Current baseline commands were run by the QA/docs worker; final budget commands must be rerun by implementation workers after each relevant step.

## Current Baseline

Command:

```bash
wc -l Python/dashboard_frontend/src/main.tsx Python/src/hexorl/dashboard/app.py Python/dashboard_frontend/src/styles.css
```

Observed output:

```text
    1706 Python/dashboard_frontend/src/main.tsx
    1431 Python/src/hexorl/dashboard/app.py
     848 Python/dashboard_frontend/src/styles.css
    3985 total
```

Command:

```bash
find Python/dashboard_frontend/src -type f \( -name '*.tsx' -o -name '*.ts' \) -print0 | xargs -0 wc -l | sort -nr | sed -n '1,40p'
```

Observed output:

```text
    1706 Python/dashboard_frontend/src/main.tsx
```

Command:

```bash
find Python/src/hexorl/dashboard -type f -name '*.py' -print0 | xargs -0 wc -l | sort -nr | sed -n '1,60p'
```

Observed output:

```text
    4666 total
    1431 Python/src/hexorl/dashboard/app.py
     577 Python/src/hexorl/dashboard/db.py
     562 Python/src/hexorl/dashboard/inspection_services.py
     464 Python/src/hexorl/dashboard/render.py
     429 Python/src/hexorl/dashboard/replay.py
     232 Python/src/hexorl/dashboard/recorder.py
     203 Python/src/hexorl/dashboard/fixtures.py
     165 Python/src/hexorl/dashboard/arena_service.py
     154 Python/src/hexorl/dashboard/play.py
     153 Python/src/hexorl/dashboard/checkpoints.py
      99 Python/src/hexorl/dashboard/contract_inspector.py
      90 Python/src/hexorl/dashboard/model_cache.py
      75 Python/src/hexorl/dashboard/model_inference.py
      18 Python/src/hexorl/dashboard/pseudocode.py
      14 Python/src/hexorl/dashboard/__init__.py
```

## Required Step 4 Frontend Budget Commands

Run after `main.tsx` is split.

```bash
wc -l Python/dashboard_frontend/src/main.tsx Python/dashboard_frontend/src/app.tsx
```

Pass criteria:

- `Python/dashboard_frontend/src/main.tsx` < 50 lines.
- `Python/dashboard_frontend/src/app.tsx` < 200 lines.

```bash
find Python/dashboard_frontend/src -type f \( -name '*.tsx' -o -name '*.ts' \) -print0 | xargs -0 wc -l | sort -nr
```

Pass criteria:

- No frontend TypeScript/TSX file is > 300 lines.
- No file under `Python/dashboard_frontend/src/routes/suite/` is > 250 lines.
- No file under `Python/dashboard_frontend/src/routes/suite/trial-tabs/` is > 250 lines.

Shell gate:

```bash
test "$(wc -l < Python/dashboard_frontend/src/main.tsx)" -lt 50
test "$(wc -l < Python/dashboard_frontend/src/app.tsx)" -lt 200
find Python/dashboard_frontend/src -type f \( -name '*.tsx' -o -name '*.ts' \) -print0 \
  | xargs -0 wc -l \
  | awk '$2 != "total" && $1 > 300 { print; bad=1 } END { exit bad }'
find Python/dashboard_frontend/src/routes/suite -type f \( -name '*.tsx' -o -name '*.ts' \) -print0 \
  | xargs -0 wc -l \
  | awk '$2 != "total" && $1 > 250 { print; bad=1 } END { exit bad }'
```

## Required Step 5 Backend Budget Commands

Run after `app.py` is split.

```bash
wc -l Python/src/hexorl/dashboard/app.py
```

Pass criteria:

- `Python/src/hexorl/dashboard/app.py` <= 100 lines.

```bash
find Python/src/hexorl/dashboard/routes -type f -name '*.py' -print0 | xargs -0 wc -l | sort -nr
```

Pass criteria:

- No backend route file is > 250 lines.
- No `Python/src/hexorl/dashboard/routes/suite/*.py` module is > 200 lines.

```bash
wc -l Python/src/hexorl/dashboard/services/suite.py
```

Pass criteria:

- `Python/src/hexorl/dashboard/services/suite.py` <= 400 lines.

Shell gate:

```bash
test "$(wc -l < Python/src/hexorl/dashboard/app.py)" -le 100
find Python/src/hexorl/dashboard/routes -type f -name '*.py' -print0 \
  | xargs -0 wc -l \
  | awk '$2 != "total" && $1 > 250 { print; bad=1 } END { exit bad }'
find Python/src/hexorl/dashboard/routes/suite -type f -name '*.py' -print0 \
  | xargs -0 wc -l \
  | awk '$2 != "total" && $1 > 200 { print; bad=1 } END { exit bad }'
test "$(wc -l < Python/src/hexorl/dashboard/services/suite.py)" -le 400
```

## Import And Ownership Audit Commands

Run after Steps 2, 4, 5, 7, and 9 as applicable.

```bash
rg -n "loadInFlight|runDetailInFlight|useEffect\(|fetch\(" Python/dashboard_frontend/src
```

Purpose: prove direct fetch/dedupe patterns are removed from `App` and migrated into API hooks.

```bash
rg -n "\"/api/|'/api/" Python/dashboard_frontend/src
```

Purpose: locate hardcoded endpoint strings that should be centralized in API client/hooks.

```bash
rg -n "_suite_|@app\.(get|post|put|delete|patch)|BaseModel" Python/src/hexorl/dashboard/app.py Python/src/hexorl/dashboard/routes Python/src/hexorl/dashboard/schemas Python/src/hexorl/dashboard/services
```

Purpose: prove `app.py` no longer owns route/business logic and Pydantic models live in schemas/route-owned files.

```bash
rg -n "EventSource|refetchInterval|setInterval|/api/suite/status|/api/suite/events|/api/arena/history" Python/dashboard_frontend/src
```

Purpose: prove Step 7 replaces live-tab polling with SSE while preserving low-cost polling elsewhere.

```bash
rg -n "zod|\\.parse\\(" Python/dashboard_frontend/src/api Python/dashboard_frontend/src/routes
```

Purpose: prove Step 9 validates fetched data at the trust boundary.

