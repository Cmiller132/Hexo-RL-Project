# 05 Dashboard Spec

## Scope

This document captures the legacy dashboard as a rebuild target. The goal is to preserve the visual style and major workflows while replacing brittle implementation details.

## Legacy Source Anchors

- Backend: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/ui/dashboard.py`
- Frontend: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/ui/static/dashboard.html`
- DB schema: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/data/schema.py`
- DB session: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/data/database.py`
- Migration: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/cli/migrate.py`
- Older standalone viewer: `/Users/coltonmiller/Documents/GitHub/Hexagon/viewer.html`
- Older viewer CLI: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/cli/viewer.py`
- Older play server: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/cli/play.py`

## High-Level Architecture

Legacy dashboard is:

- FastAPI backend.
- One static HTML file with inline CSS and JavaScript.
- Chart.js from CDN.
- SQLite persistence.
- WebSocket for live metrics.
- WebSocket for live arena matches.
- In-memory model cache, game sessions, and arena match state.
- Optional training subprocess launcher.
- Optional embedded `DashboardServer` used by the training process.

The frontend file is about 5.7k lines and contains every UI feature in one place.

## Visual Style To Preserve

The dashboard uses a dense GitHub-dark operational style:

| Token | Legacy Value |
|---|---|
| Background | `#0d1117` |
| Surface | `#161b22` |
| Border | `#30363d` |
| Text | `#c9d1d9` |
| Muted text | `#8b949e` |
| Accent | `#58a6ff` |
| Success | `#3fb950` |
| Danger | `#f85149` |
| Warning | `#d29922` |

UI qualities to preserve:

- Dense, dashboard-first layout.
- Small uppercase section labels.
- Compact tabs.
- Compact tables and filters.
- Monospaced move/debug panels.
- SVG hex boards with dark field background.
- Blue/red player colors.
- Small badges for analysis flags.
- Thin borders and 4-8px radius.
- Status dot for WebSocket/training state.

Do not rebuild it as a marketing page or a large-card hero UI. It should feel like a training control room.

## Main Navigation

Legacy tabs:

1. Charts
2. Games
3. Viewer
4. Model Lab
5. Arena
6. Checkpoints
7. Training
8. Config
9. Analysis

Top KPIs:

- Epoch
- Avg loss
- Elo
- Eval win rate
- Sparring win rate
- Average game length
- Buffer size
- Learning rate

## Data Model

Legacy SQLite tables:

- `TrainingRun`
- `Checkpoint`
- `EpochMetrics`
- `GameRecord`
- `GameAnalysis`
- `CanonicalOpening`

Dashboard should continue to have a persistent run/game/metrics model. The rewrite currently lacks this. A rebuilt dashboard should either:

- Introduce a clean SQLite schema in `hexorl.dashboard`, or
- Define append-only JSONL/Parquet run logs with a background indexer.

SQLite is likely enough and matches the legacy workflow.

## Backend Route Inventory

Legacy backend groups:

### Runs, Metrics, Checkpoints

- `GET /api/runs`
- `GET /api/metrics/{run_name}`
- `GET /api/checkpoints/{run_name}`
- `GET /api/checkpoint-dirs`
- `GET /api/latest`
- `POST /api/import/checkpoints`
- `GET /api/import/status`

### Games And Viewer

- `GET /api/games`
- `GET /api/games/{run_name}`
- `GET /api/game/{game_id}`
- `GET /api/game/{game_id}/model-input`
- `GET /api/game/{game_id}/train-targets`
- `GET /api/game/{game_id}/alternatives`

### Config And Control

- `GET /api/config`
- `GET /api/status`
- `POST /api/config/update`
- `GET /api/config/file`
- `POST /api/control/{action}`
- `GET /api/control/status`

### Training Process

- `POST /api/training/start`
- `GET /api/training/status`
- `POST /api/training/stop`
- `GET /api/training/log`

### Analysis And Corpus

- `POST /api/analyze/{game_id}`
- `POST /api/analyze/batch`
- `GET /api/corpus/summary`
- `GET /api/corpus/openings`
- `PATCH /api/corpus/openings/{opening_id}`
- `POST /api/corpus/refresh-openings`
- `GET /api/corpus/opening-trajectory`
- `GET /api/corpus/openings/{opening_id}/games`

### Model Lab

- `POST /api/model/load`
- `GET /api/model/loaded`
- `DELETE /api/model/{model_id}`
- `POST /api/model/{model_id}/infer`
- `GET /api/sessions`
- `POST /api/position/create`

### Play Sessions

- `POST /api/session/create`
- `GET /api/session/{session_id}`
- `POST /api/session/{session_id}/move`
- `POST /api/session/{session_id}/undo`
- `POST /api/session/{session_id}/reset`
- `DELETE /api/session/{session_id}`

### Arena

- `POST /api/arena/start`
- `GET /api/arena/history`
- `GET /api/arena/{match_id}`
- `DELETE /api/arena/{match_id}`
- `WS /ws/arena/{match_id}`

### WebSockets

- `WS /ws`: main live metrics.
- `WS /ws/arena/{match_id}`: arena events.

## Charts Tab

Preserve these panels:

- Total loss.
- Loss decomposition.
- Elo rating.
- Win rates.
- P0/P1/draw balance.
- Average game length.
- Buffer size and learning rate.
- Timing breakdown.
- Gradient norms.
- Positions per epoch.
- Regret losses.
- Policy entropy.
- PCR full versus low moves.
- PRB regret mean/max.

Rebuild requirements:

- Bundle chart dependency locally.
- Fetch latest metrics correctly; legacy `limit` behavior returns earliest rows because of ascending order before limit.
- Support both historical DB load and live updates.
- Use typed metric payloads with schema version.

## Games Tab

Preserve:

- Cross-run game browsing.
- Filters by epoch, source, winner, player type, move count, analyzed status, sort, limit.
- Click-through to viewer.
- Analysis status/accuracy columns.

Rebuild requirements:

- Server-side pagination.
- Stable sort semantics.
- Explicit empty/loading/error states.
- Avoid building rows with unsafe string interpolation.

## Viewer Tab

Preserve:

- Replay mode.
- Play mode.
- SVG hex board pan/zoom/fit.
- Move slider.
- First/prev/play/next/last controls.
- Move list.
- Board overlays for stones, current move, legal moves, winning line, threats.
- PRB restart/start-ply markers.
- Analysis badges.
- Eval bar and explanation.
- Rhythm strip.
- Move detail panel.
- Best-alternative lookup.
- Exact model-input debug channel grid.
- Train-target debug view.

Rebuild requirements:

- Extract board renderer into a reusable component/module.
- Keep board state serializable.
- Put all geometry and coordinate conversion in one tested module.
- Use canvas or SVG consistently; SVG is fine for inspectability.
- Make debug panes virtualized or collapsible for long games.

## Model Lab Tab

Preserve:

- Load checkpoint into model cache.
- List loaded models.
- Unload model.
- Choose source position from recorded game, session, or custom board.
- Run inference.
- Compare multiple models.
- Show policy/value/axis/regret/moves outputs.
- Show input channels.
- Probe cell values.
- Overlay policy/value/debug maps on board.

Rebuild requirements:

- Put model cache behind a service with explicit GPU memory accounting.
- Persist model-cache metadata enough for UI recovery after refresh.
- Use typed inference request/response schemas.
- Do not let arbitrary paths load without path validation.

## Arena Tab

Preserve:

- Player types: neural, classical, random.
- Checkpoint selection for neural players.
- MCTS sims/time controls.
- Number of games.
- Start/cancel.
- Live board spectator.
- Score and progress bar.
- Match history.
- Arena WebSocket events.

Rebuild requirements:

- Persist match history if it matters after restart.
- Keep arena worker state isolated from dashboard process state.
- Add clear terminal states: completed, failed, cancelled.
- Stream move events with monotonically increasing sequence ids.

## Training Tab

Preserve:

- New run/start controls.
- Stop controls.
- Training status badge.
- Log viewer.
- Import checkpoints/games.
- Auto-refresh log while tab active.

Rebuild requirements:

- Separate dashboard process control from trainer internals.
- Make stop semantics explicit: immediate, after batch, after epoch.
- Avoid hidden file-only signals unless wrapped behind a tested control service.
- Track child process stdout/stderr and exit status.

## Config Tab

Preserve:

- Grouped config editor.
- Tooltips/descriptions.
- Save/apply behavior.
- Source-of-truth clarity.

Legacy improvement to keep:

- Config source-of-truth should be disk file, not in-memory dashboard state.
- Writes should be atomic.
- Training should hot-reload at a documented boundary.

Rebuild requirements:

- Use the rewrite's Pydantic schema as the canonical config schema.
- Expose JSON schema to the frontend.
- Validate server-side and client-side.
- Display which file or run config is active.

## Analysis Tab

Preserve:

- Corpus summary.
- Game length distribution.
- Wins by source.
- Opening explorer.
- Opening naming.
- Example games for an opening.
- Opening trajectory chart.
- Refresh/recompute openings.

Rebuild requirements:

- Move slow analysis to jobs.
- Store job status.
- Cache corpus summaries.
- Support recomputation without blocking dashboard request threads.

## Known Legacy Dashboard Risks

- External CDN Chart.js can fail offline.
- One giant HTML/JS file is hard to reason about and easy to break.
- Standalone training launch does not push embedded dashboard WebSocket metrics, so live charts can feel stale.
- Many frontend updates use `innerHTML` with data from DB/config/player/opening strings.
- Most REST errors are HTTP 200 with `{ "error": ... }`.
- No authentication; okay only for trusted local single-user usage.
- Sessions, model cache, and arena history are in memory and lost on restart.
- Model cache has max size 3 and ref-counting; leaks or stale refs can strand GPU memory.
- Pause/stop mostly happen at epoch boundaries, so controls feel unresponsive during long phases.
- Mobile responsiveness is minimal.

## Dashboard Rebuild Architecture

Recommended architecture:

```text
hexorl.dashboard
+-- api/              FastAPI routers by domain
+-- db/               SQLite schema and migrations
+-- services/         run indexer, model cache, arena, training control
+-- events/           metrics/event schemas and WebSocket fanout
+-- static/ or web/   componentized frontend
+-- board/            shared coordinate/geometry helpers
```

Suggested route groups:

- `/api/runs`
- `/api/metrics`
- `/api/games`
- `/api/checkpoints`
- `/api/config`
- `/api/training`
- `/api/model`
- `/api/session`
- `/api/arena`
- `/api/analysis`
- `/ws/runs/{run_id}`
- `/ws/arena/{match_id}`

## Minimum Viable Dashboard Rebuild

Phase 1:

- Run selection.
- KPIs.
- Charts from run metrics.
- Training status/logs.
- Config read-only view.
- Local chart bundle.

Phase 2:

- Game browser.
- Replay viewer.
- Board renderer.
- Model-input debug endpoint.

Phase 3:

- Config editing.
- Training start/stop.
- Checkpoint import/indexing.
- Analysis job status.

Phase 4:

- Model Lab.
- Interactive play.
- Arena.
- Corpus/opening explorer.

## Definition Of Done For Dashboard Replacement

- Can monitor a new rewrite training run live without training importing dashboard internals.
- Can reload browser without losing selected run context.
- Can view historical metrics without a running process.
- Can open a saved game and replay it with overlays.
- Can inspect exact encoder channels for a position.
- Can validate and save config atomically.
- Can start/stop training with clear status.
- Can run local/offline without CDN dependency.
- Has typed request/response schemas.
- Has smoke tests for key API routes.
- Has visual regression or Playwright screenshots for major tabs.
