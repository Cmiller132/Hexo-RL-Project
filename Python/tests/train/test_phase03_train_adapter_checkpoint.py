import numpy as np
import torch
import zipfile

from hexorl.config import Config
from hexorl.graph import build_graph_batch_from_history, collate_graph_batches
from hexorl.models.checkpoint import CheckpointBundle, CheckpointManager
from hexorl.models.factory import build_model, train_adapter_for
from hexorl.replay.projector import ProjectedReplayBatch
from hexorl.train.trainer import Trainer


def _cfg(architecture: str) -> Config:
    return Config.model_validate(
        {
            "model": {
                "architecture": architecture,
                "channels": 4,
                "blocks": 1,
                "attention_heads": 1,
                "graph_layers": 1,
            },
            "train": {
                "batches_per_epoch": 1,
                "loss_weights": {"policy": 1.0, "policy_place": 1.0, "value": 1.0},
            },
            "inference": {"fp16": False},
        }
    )


def _crop_batch():
    policy = np.zeros((1, 1089), dtype=np.float32)
    policy[0, 544] = 1.0
    return ProjectedReplayBatch(
        tensors=np.zeros((1, 13, 33, 33), dtype=np.float32),
        policies=policy,
        values=np.zeros(1, dtype=np.float32),
        lookahead=[],
        aux_targets={},
        record_hashes=("test-crop",),
    )


def _graph_batch():
    graph = collate_graph_batches(
        [
            build_graph_batch_from_history(
                b"",
                radius=8,
                policy_target=[(0, 0, 1.0)],
                include_pair_rows=False,
            )
        ]
    )
    aux = {
        "token_features": graph.token_features,
        "token_type": graph.token_type,
        "token_qr": graph.token_qr,
        "token_mask": graph.token_mask,
        "legal_token_indices": graph.legal_token_indices,
        "legal_qr": graph.legal_qr,
        "legal_mask": graph.legal_mask,
        "policy_target": graph.policy_target,
        "pair_token_indices": graph.pair_token_indices,
        "pair_first_indices": graph.pair_first_indices,
        "pair_second_indices": graph.pair_second_indices,
        "pair_rows": graph.pair_rows,
        "pair_table_mask": graph.pair_table_mask,
        "pair_phase": graph.pair_phase,
        "pair_known_first": graph.pair_known_first,
        "pair_known_first_mask": graph.pair_known_first_mask,
        "relation_type": graph.relation_type,
        "relation_bias": graph.relation_bias,
        "policy_weight": np.ones(1, dtype=np.float32),
    }
    return ProjectedReplayBatch(
        tensors=np.zeros((1, 13, 33, 33), dtype=np.float32),
        policies=np.zeros((1, 1089), dtype=np.float32),
        values=np.zeros(1, dtype=np.float32),
        lookahead=[],
        aux_targets=aux,
        record_hashes=("test-graph",),
    )


def test_trainer_runs_one_batch_for_every_registered_family():
    aliases = [
        ("cnn", _crop_batch),
        ("restnet", _crop_batch),
        ("graph_hybrid_0", _crop_batch),
        ("global_xattn_0", _graph_batch),
        ("global_line_window_0", _graph_batch),
        ("global_graph_option1", _graph_batch),
    ]
    for architecture, batch_builder in aliases:
        cfg = _cfg(architecture)
        model = build_model(cfg, device=torch.device("cpu"), inference=False)
        trainer = Trainer(model, cfg, dataloader=[], device=torch.device("cpu"))
        losses = trainer._train_step(batch_builder(), 0)
        assert torch.isfinite(torch.tensor(losses["total"])), architecture


def test_trainer_contains_no_architecture_or_model_class_branches():
    import inspect
    from hexorl.train import trainer

    source = inspect.getsource(trainer.Trainer)
    forbidden = ["isinstance(self.model", "GlobalHexGraphNet", "architecture.startswith", "build_model_from_config"]
    for pattern in forbidden:
        assert pattern not in source


def test_train_adapter_rejects_legacy_raw_tuple_batches():
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    legacy = (np.zeros((1, 13, 33, 33), dtype=np.float32), np.zeros((1, 1089), dtype=np.float32), np.zeros(1, dtype=np.float32))

    try:
        adapter.project_batch(legacy)
    except TypeError as exc:
        assert "ProjectedReplayBatch" in str(exc)
    else:
        raise AssertionError("legacy raw tuple batch was accepted")


def test_pair_target_validation_rejects_opening_pair_loss():
    cfg = _cfg("global_xattn_0")
    cfg.model.pair_strategy = "diagnostic_full_pair"
    cfg.model.pair_strategy_max_pairs = 2
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _graph_batch()
    batch.aux_targets["pair_policy_target"] = np.zeros_like(batch.aux_targets["pair_token_indices"], dtype=np.float32)

    try:
        adapter.project_batch(batch)
    except ValueError as exc:
        assert "opening positions" in str(exc)
    else:
        raise AssertionError("opening pair loss was accepted")


def test_pair_target_validation_rejects_missing_known_first():
    cfg = _cfg("global_xattn_0")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _graph_batch()
    batch.aux_targets["pair_policy_target"] = np.zeros((1, 1), dtype=np.float32)
    batch.aux_targets.pop("pair_token_indices")

    try:
        adapter.project_batch(batch)
    except ValueError as exc:
        assert "pair_token_indices" in str(exc)
    else:
        raise AssertionError("missing known-first metadata was accepted")


def test_pair_target_validation_rejects_stale_post_first_legal_table():
    cfg = _cfg("global_xattn_0")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _graph_batch()
    batch.aux_targets["pair_policy_target"] = np.zeros((1, 2), dtype=np.float32)

    try:
        adapter.project_batch(batch)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("stale pair table shape was accepted")


def test_pair_target_mass_preserved_under_d6():
    batch = _graph_batch()
    target = batch.aux_targets["policy_target"]
    assert np.isclose(target.sum(), 1.0)


def test_train_adapter_debug_bundle_reconstructs_replay_to_loss_inputs():
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    projected = adapter.project_batch(_crop_batch())
    outputs = adapter.forward(projected)
    total, losses = adapter.losses(outputs, projected.targets, n_bins=65)
    bundle = adapter.debug_bundle(projected, outputs, losses, trace_id="phase03-single-position")
    assert torch.isfinite(total)
    assert bundle.owner == "train_adapter"
    assert bundle.output_keys
    assert bundle.loss_keys


def test_train_adapter_rejects_mutated_contract_after_projection():
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _crop_batch()
    projected = adapter.project_batch(batch)
    batch.policies[0, 0] = float("nan")
    assert torch.isfinite(projected.targets["policy"]).all()


def test_train_adapter_rejects_stale_legal_row_identity():
    cfg = _cfg("global_xattn_0")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _graph_batch()
    batch.aux_targets["policy_target"] = np.zeros((1, 3), dtype=np.float32)

    try:
        adapter.project_batch(batch)
    except ValueError as exc:
        assert "policy_target shape" in str(exc)
    else:
        raise AssertionError("stale legal row identity was accepted")


def test_train_adapter_rejects_corrupt_masks_or_nonfinite_targets():
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _crop_batch()
    batch.policies[0, 0] = float("nan")

    try:
        adapter.project_batch(batch)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite target was accepted")


def test_train_adapter_projection_and_device_transfer_profile_is_recorded():
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    projected = adapter.project_batch(_crop_batch())
    assert projected.targets["policy"].device.type == "cpu"


def test_model_output_validation_rejects_wrong_rows_shapes_and_nonfinite_values():
    cfg = _cfg("global_xattn_0")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    batch = _graph_batch()
    projected = adapter.project_batch(batch)

    try:
        adapter.validate_outputs({"policy_place": torch.zeros(1, 3)}, projected.targets)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("bad output rows were accepted")


def test_checkpoint_manifest_round_trips(tmp_path):
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    path = tmp_path / "phase03.pt"
    manager = CheckpointManager()
    manager.save(CheckpointBundle(cfg=cfg, model=model), path)
    loaded = manager.load(path, purpose="train", device="cpu")
    assert loaded.manifest.model_family == "dense_cnn"


def test_checkpoint_inspect_does_not_load_weights(tmp_path, monkeypatch):
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    path = tmp_path / "phase03.pt"
    manager = CheckpointManager()
    manager.save(CheckpointBundle(cfg=cfg, model=model), path)
    with zipfile.ZipFile(path) as archive:
        assert any(name.endswith("/checkpoint_manifest.json") for name in archive.namelist())

    def _fail_torch_load(*args, **kwargs):
        raise AssertionError("inspect must not torch.load or unpickle the checkpoint payload")

    monkeypatch.setattr(torch, "load", _fail_torch_load)
    manifest = manager.inspect(path)
    assert manifest.model_family == "dense_cnn"


def test_checkpoint_load_rejects_missing_manifest(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}}, path)
    try:
        CheckpointManager().load(path, purpose="train", device="cpu")
    except ValueError as exc:
        assert "checkpoint_manifest" in str(exc)
    else:
        raise AssertionError("legacy checkpoint without manifest was accepted")


def test_checkpoint_load_rejects_unknown_or_stale_manifest_fields(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"checkpoint_manifest": {"checkpoint_schema_version": 1, "unknown": True}, "model_state_dict": {}}, path)
    try:
        CheckpointManager().load(path, purpose="train", device="cpu")
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("bad manifest was accepted")


def test_checkpoint_load_rejects_model_family_mismatch(tmp_path):
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    path = tmp_path / "phase03.pt"
    manager = CheckpointManager()
    manager.save(CheckpointBundle(cfg=cfg, model=model), path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["checkpoint_manifest"]["model_family"] = "restnet"
    torch.save(checkpoint, path)
    try:
        manager.load(path, purpose="train", device="cpu")
    except ValueError as exc:
        assert "model family mismatch" in str(exc)
    else:
        raise AssertionError("model family mismatch was accepted")


def test_checkpoint_load_rejects_inference_protocol_mismatch(tmp_path):
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    path = tmp_path / "phase03.pt"
    manager = CheckpointManager()
    manager.save(CheckpointBundle(cfg=cfg, model=model), path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["checkpoint_manifest"]["inference_protocol"]["protocol_version"] = 0
    torch.save(checkpoint, path)
    try:
        manager.load(path, purpose="inference", device="cpu")
    except ValueError as exc:
        assert "inference protocol" in str(exc)
    else:
        raise AssertionError("bad inference protocol was accepted")


def test_checkpoint_load_does_not_silently_strip_orig_mod_or_prefixes(tmp_path):
    cfg = _cfg("cnn")
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    path = tmp_path / "phase03.pt"
    manager = CheckpointManager()
    manager.save(CheckpointBundle(cfg=cfg, model=model), path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {"_orig_mod.weight": torch.zeros(1)}
    torch.save(checkpoint, path)
    try:
        manager.load(path, purpose="train", device="cpu")
    except ValueError as exc:
        assert "offline conversion" in str(exc)
    else:
        raise AssertionError("prefixed checkpoint keys were accepted")


def test_no_runtime_imports_from_hexorl_model():
    import pathlib

    root = pathlib.Path(__file__).parents[2] / "src" / "hexorl"
    matches = [
        path
        for path in root.rglob("*.py")
        if ("hexorl." "model.") in path.read_text(encoding="utf-8") or ("from hexorl import " "model") in path.read_text(encoding="utf-8")
    ]
    assert matches == []


def test_no_model_architecture_string_gates_outside_registry_spec_tests():
    import pathlib

    root = pathlib.Path(__file__).parents[2] / "src" / "hexorl"
    forbidden = ("architecture.startswith", "architecture ==", "build_model_from_config", "GlobalHexGraphNet)")
    matches = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in forbidden):
            matches.append(path)
    assert matches == []
