import torch
from torch.utils.data import DataLoader

from hexorl.contracts.candidates import CANDIDATE_FEATURE_VERSION
from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.config import Config
from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.recorder import RunRecorder
from hexorl.dashboard.replay import replay_game
from hexorl.epoch.pipeline import _make_bootstrap_game_records
from hexorl.eval.scorecard import compute_phase3_scorecard
from hexorl.model.network import build_model_from_config
from hexorl.train.losses import compute_losses


def test_tiny_production_pipeline_records_games_metrics_and_checkpoint(tmp_path):
    cfg = Config()
    cfg.run.seed = 123
    cfg.model.channels = 4
    cfg.model.blocks = 1
    cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
    cfg.model.sparse_policy = True
    cfg.model.candidate_budget = 16
    cfg.model.sparse_prior_stage = 0
    cfg.model.pair_prior_mix = 0.25
    cfg.buffer.capacity = 64
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.selfplay.policy_target_top_k = 8
    cfg.selfplay.max_game_moves = 12
    cfg.train.batch_size = 2
    cfg.train.batches_per_epoch = 1
    cfg.train.lr_schedule = "constant"
    cfg.train.loss_weights = {
        "policy": 1.0,
        "value": 1.0,
        "sparse_policy": 0.2,
        "pair_policy": 0.05,
        "entropy": 0.0,
    }
    cfg.runtime.dataloader_workers = 0

    recorder = RunRecorder.for_run_dir(tmp_path, run_id="tiny-production")
    game_records = _make_bootstrap_game_records(cfg, 2)
    for game in game_records:
        recorder.game(game, source="bootstrap", epoch=1)

    replay_buffer = RingBuffer(
        capacity=cfg.buffer.capacity,
        max_policy_entries=cfg.selfplay.policy_target_top_k,
        max_policy_v2_entries=min(
            max(cfg.selfplay.policy_target_top_k, cfg.model.candidate_budget),
            512,
        ),
        num_lookahead=0,
    )
    for game in game_records:
        replay_buffer.extend(game.positions)

    dataset = ReplayDataset(
        replay_buffer,
        batch_size=cfg.train.batch_size,
        recency_decay=cfg.buffer.recency_decay,
        pcr_weight=cfg.buffer.pcr_weight,
        use_symmetry=True,
        lookahead_horizons=[],
        regret_fraction=cfg.buffer.regret_fraction,
        include_sparse_policy=True,
        include_pair_policy=True,
        candidate_budget=cfg.model.candidate_budget,
    )
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    tensors, policies, values, _lookahead, aux_targets = batch
    targets = {"policy": policies, "value": values, **aux_targets}
    model = build_model_from_config(cfg, device=torch.device("cpu"), inference=False)
    model.train()
    predictions = model(
        tensors,
        candidate_features=targets.get("candidate_features"),
        candidate_indices=targets.get("candidate_indices"),
        candidate_mask=targets.get("candidate_mask"),
        pair_candidate_indices=targets.get("pair_candidate_indices"),
        pair_candidate_mask=targets.get("pair_candidate_mask"),
    )
    total_loss, per_head = compute_losses(
        predictions,
        targets,
        loss_weights=cfg.train.loss_weights,
        n_bins=model.n_bins,
    )
    total_loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= 1e-4 * param.grad
    train_stats = {
        "loss_total": float(total_loss.detach().cpu()),
        **{
            f"loss_{key}": float(value.detach().cpu())
            for key, value in per_head.items()
            if torch.is_tensor(value)
        },
    }

    checkpoint_path = tmp_path / "epoch_0001.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "epoch": 1,
            "global_step": 1,
            "model_metadata": {
                "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
                "architecture": cfg.model.architecture,
                "heads": list(cfg.model.heads),
            },
        },
        checkpoint_path,
    )
    recorder.metric({"train": train_stats, "buffer": replay_buffer.stats}, phase="train", epoch=1, global_step=1)
    recorder.checkpoint(checkpoint_path, {"buffer": replay_buffer.stats}, epoch=1, global_step=1)

    assert checkpoint_path.exists()
    assert train_stats["loss_total"] >= 0.0
    assert "pair_fallback_prior_use" in replay_buffer.stats
    assert "fallback_prior_use_on_mcts_topk" in replay_buffer.stats

    indexed = scan_checkpoints(tmp_path, recorder.store, run_id="tiny-production")
    assert indexed
    assert indexed[0].is_loadable
    assert "sparse_policy" in indexed[0].model_heads
    assert "pair_policy" in indexed[0].model_heads

    games = recorder.store.rows("SELECT game_id FROM games ORDER BY game_id")
    assert games
    replay_payload = replay_game(recorder.store, int(games[0]["game_id"]))
    assert replay_payload["positions"]
    debug = replay_payload["positions"][0]["debug"]
    for key in (
        "policy_target_v2",
        "pair_policy_target_v2",
        "selected_action_value",
        "value_weight",
        "policy_weight",
        "fallback_prior_use",
        "pair_fallback_prior_use",
    ):
        assert key in debug

    scorecard = compute_phase3_scorecard(
        {
            "policy_target_quality": 1.0 - float(replay_buffer.stats["avg_missing_target_policy_mass"]),
            "value_calibration_score": 0.0,
            "outside_window_robustness": 0.0,
            "illegal_or_crash_rate": 0.0,
            "critical_overflow_count": 0.0,
        },
        epoch=1,
        candidate_model=True,
    )
    assert scorecard.mode == "health_warmup"
    assert "illegal_or_crash_rate" not in scorecard.hard_failures
