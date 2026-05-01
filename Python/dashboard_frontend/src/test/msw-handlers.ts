import { http, HttpResponse } from "msw";

const rows = {
  runs: [{ run_id: "trial-a", name: "trial-a" }],
  games: [{ game_id: 7, run_id: "trial-a", trial_id: "trial-a", source: "selfplay", epoch: 1, move_count: 2 }],
  checkpoints: [{ checkpoint_id: 1, trial_id: "trial-a", run_id: "trial-a", score: 1.2, path: "/tmp/ckpt.pt" }],
  metrics: [{ epoch: 1, global_step: 10, phase: "train", metrics_json: { train: { epoch: 1, loss_total: 0.5 }, loss_total: 0.5 } }],
  events: [{ event: "stage_start", trial_id: "trial-a", stage: "seed", time: 1 }],
  scores: [{ epoch: 1, score: 1.0, scheduler_score: 1.2 }],
  sweep: [{ trial_id: "trial-a", workers: 2, batch: 16, positions_per_sec: 42, stable: true }]
};

export const handlers = [
  http.get("/api/health", () => HttpResponse.json({ ok: true, db_path: "/tmp/dashboard.sqlite3" })),
  http.get("/api/runs", () => HttpResponse.json(rows.runs)),
  http.get("/api/metrics/:run", () => HttpResponse.json(rows.metrics)),
  http.get("/api/games", () => HttpResponse.json(rows.games)),
  http.get("/api/checkpoints", () => HttpResponse.json(rows.checkpoints)),
  http.get("/api/arena/history", () => HttpResponse.json([])),
  http.get("/api/axis/prototypes", () => HttpResponse.json([{ id: "center", label: "Center", description: "center control" }])),
  http.get("/api/axis/fixtures", () => HttpResponse.json([])),
  http.get("/api/suite/status", () => HttpResponse.json({ enabled: true, current_stage: "seed", current_trial_id: "trial-a", best_trial_id: "trial-a", best_score: 1.2, total_games: 1, total_positions: 2, current_positions_per_sec: 42, current_activity: { trial_id: "trial-a", action: "training" } })),
  http.get("/api/suite/trials", () => HttpResponse.json([{ trial_id: "trial-a", family: "cnn", architecture: "cnn", stage: "seed", epoch: 1, score: 1.2, games: 1, positions: 2 }])),
  http.get("/api/suite/best-checkpoints", () => HttpResponse.json(rows.checkpoints)),
  http.get("/api/suite/events", () => HttpResponse.json(rows.events)),
  http.get("/api/suite/trials/:trial", () => HttpResponse.json({ trial_id: "trial-a", latest: { epoch: 1, train: { loss_total: 0.5 }, selfplay: { positions_per_min: 600 } }, architecture_summary: "CNN residual trunk", config: { selfplay: { mcts_simulations: 16 } }, trial: { family: { name: "cnn" } }, state: { runtime_sweep: { selected: { workers: 2 } } } })),
  http.get("/api/suite/trials/:trial/scores", () => HttpResponse.json(rows.scores)),
  http.get("/api/suite/trials/:trial/events", () => HttpResponse.json(rows.events)),
  http.get("/api/suite/trials/:trial/loss-curve", () => HttpResponse.json(rows.metrics)),
  http.get("/api/suite/trials/:trial/runtime-sweep", () => HttpResponse.json({ probes: rows.sweep })),
  http.get("/api/suite/family-space", () => HttpResponse.json({ families: [{ name: "cnn", choices: { channels: [32] } }], recipes: [{ recipe_id: "cnn:default", model_family: "cnn" }] })),
  http.get("/api/suite/scheduler", () => HttpResponse.json({ current_stage: "seed", planned_stages: [{ stage: "seed" }], decisions: [], scheduler: {}, budget: {}, state: {} })),
  http.get("/api/suite/runtime-sweep", () => HttpResponse.json({ probes: rows.sweep, selected: [] }))
  ,http.post("/api/session/create", () => HttpResponse.json({ session_id: "session-a", position: { current_player: 0, legal_moves: [{ q: 0, r: 0 }], stones: [], moves: [], encoding: { channels: [] } } }))
  ,http.post("/api/session/:id/move", () => HttpResponse.json({ session_id: "session-a", position: { current_player: 1, legal_moves: [{ q: 1, r: 0 }], stones: [{ q: 0, r: 0, player: 0 }], moves: [{ q: 0, r: 0, player: 0 }], encoding: { channels: [] } } }))
  ,http.post("/api/session/:id/reset", () => HttpResponse.json({ session_id: "session-a", position: { current_player: 0, legal_moves: [{ q: 0, r: 0 }], stones: [], moves: [], encoding: { channels: [] } } }))
  ,http.post("/api/axis/evaluate", () => HttpResponse.json({ prototype_id: "center", cells: [{ q: 0, r: 0, score: 1 }] }))
  ,http.post("/api/axis/fixtures/generate", () => HttpResponse.json({ fixtures: [] }))
];
