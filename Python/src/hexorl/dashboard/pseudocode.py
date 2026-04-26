"""Dashboard pseudocode sketch.

See ``Docs/DASHBOARD_PSEUDOCODE.md`` for the full written design.
"""

DASHBOARD_PSEUDOCODE = r"""
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
"""
