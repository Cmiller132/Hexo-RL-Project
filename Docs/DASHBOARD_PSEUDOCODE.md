# Dashboard Pseudocode

The dashboard should be a read-only Rich TUI that watches structured runtime
stats emitted by the epoch pipeline, inference server, self-play orchestrator,
trainer, replay buffer, and arena.

```python
class DashboardState:
    run_name: str
    epoch: int
    elapsed_s: float
    selfplay: dict      # games_done, games/min, workers alive, crash count
    inference: dict     # batches, positions/s, p50/p99 wait, max batch, device
    buffer: dict        # size, capacity, full-search %, max_game_id
    training: dict      # step, lr, losses per head, batches/s, grad norm
    eval: dict          # win rate, elo diff, avg moves
    events: deque[str]  # warnings, checkpoints, worker restarts


def dashboard_main(run_dir: Path):
    state = DashboardState()
    with Live(render(state), refresh_per_second=4) as live:
        while not should_exit():
            state.selfplay = read_json(run_dir / "selfplay_stats.json")
            state.inference = read_json(run_dir / "inference_stats.json")
            state.buffer = read_json(run_dir / "buffer_stats.json")
            state.training = read_json(run_dir / "train_stats.json")
            state.eval = read_json(run_dir / "eval_stats.json")
            state.events.extend(read_new_lines(run_dir / "events.log"))
            live.update(render(state))
            sleep(0.25)


def render(state: DashboardState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(header(state), size=3),
        Layout(name="body"),
        Layout(event_log(state.events), size=8),
    )
    layout["body"].split_row(
        worker_panel(state.selfplay),
        inference_panel(state.inference),
        train_panel(state.training),
        buffer_eval_panel(state.buffer, state.eval),
    )
    return layout
```

Expected panels:

- **Run Header:** run name, epoch, wall time, device, checkpoint path.
- **Self-Play:** worker alive/total, games/min, positions/min, crash/restart count.
- **Inference:** device, FP16/BF16 mode, current batch size, positions/s, queue wait p50/p99.
- **Replay Buffer:** size/capacity bar, full-search percentage, regret sample fraction.
- **Training:** total loss sparkline, per-head losses, learning rate, batches/s, EMA decay.
- **Arena:** current opponent, win rate, Elo delta, average moves, latest result.
- **Events:** compact rolling log of checkpoint saves, warnings, worker deaths, OOM recovery.

Writers should use atomic JSON updates:

```python
def write_stats(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)
```

The TUI must never own training resources. If it crashes, the run continues.
