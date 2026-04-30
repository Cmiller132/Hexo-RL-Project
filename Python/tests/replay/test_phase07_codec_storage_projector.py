import math

import pytest
import torch
from torch.utils.data import DataLoader

from hexorl.config import Config
from hexorl.models.factory import build_model, train_adapter_for
from hexorl.replay.codec import ReplayCodecError, decode_replay_game, encode_replay_game
from hexorl.replay.fixtures import corrupt_replay_bytes, golden_replay_game
from hexorl.replay.projector import ReplayProjectionConfig, ReplayProjector
from hexorl.replay.sampler import ReplayDataset
from hexorl.replay.storage import ReplayStorage


def test_replay_codec_roundtrip_preserves_semantic_identities():
    record = golden_replay_game()
    decoded = decode_replay_game(encode_replay_game(record))

    assert decoded.game_hash == record.game_hash
    assert decoded.final_history_hash == record.final_history_hash
    assert decoded.positions[0].history_hash == record.positions[0].history_hash
    assert decoded.positions[0].legal_table_hash == record.positions[0].legal_table_hash
    assert decoded.positions[0].record_hash == record.positions[0].record_hash


@pytest.mark.parametrize("kind,match", [
    ("bad_magic", "magic"),
    ("bad_version", "schema"),
    ("truncated", "length"),
])
def test_replay_codec_rejects_corruption_with_owner(kind, match):
    with pytest.raises(ReplayCodecError, match=match) as exc:
        decode_replay_game(corrupt_replay_bytes(kind))
    assert "replay.codec" in str(exc.value)


def test_replay_record_rejects_stale_legal_hash_and_bad_target():
    record = golden_replay_game()
    payload = record.to_dict()
    payload["positions"][0]["legal_table_hash"] = "stale"
    with pytest.raises(ReplayCodecError, match="stale legal hash"):
        type(record).from_dict(payload)

    payload = record.to_dict()
    payload["positions"][0]["policy_target_v2"][0][2] = float("nan")
    with pytest.raises(ReplayCodecError, match="invalid target probability"):
        type(record).from_dict(payload)


def test_storage_sampler_reads_only_new_replay_records():
    storage = ReplayStorage(capacity=8)
    storage.append_game(golden_replay_game())
    dataset = ReplayDataset(storage, batch_size=2, use_symmetry=False, include_sparse_policy=True, candidate_budget=8)
    batch = next(iter(dataset))

    assert batch.source == "replay/projector.py"
    assert batch.tensors.shape == (2, 13, 33, 33)
    assert batch.policies.shape == (2, 1089)
    assert storage.stats["read_samples"] == 2

    with pytest.raises(ReplayCodecError, match="ReplayDataset reads only ReplayStorage"):
        ReplayDataset(object(), batch_size=1)


def test_projector_d6_preserves_policy_mass_without_mutating_record():
    record = golden_replay_game().positions[1]
    before = record.record_hash
    projector = ReplayProjector(ReplayProjectionConfig(use_symmetry=True, include_sparse_policy=True, candidate_budget=8))
    batch = projector.project([record, record])

    assert record.record_hash == before
    assert batch.aux_targets["sparse_policy_target"].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert batch.projection_id


def test_sample_to_loss_uses_projected_replay_batch():
    cfg = Config()
    cfg.model.channels = 4
    cfg.model.blocks = 1
    cfg.model.heads = ["policy", "value", "sparse_policy"]
    cfg.model.sparse_policy = True
    cfg.model.candidate_budget = 8
    cfg.train.batch_size = 2
    cfg.train.loss_weights = {"policy": 1.0, "value": 1.0, "sparse_policy": 0.1}
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []

    storage = ReplayStorage(capacity=8)
    storage.append_game(golden_replay_game())
    loader = DataLoader(
        ReplayDataset(storage, batch_size=2, include_sparse_policy=True, candidate_budget=8, use_symmetry=False),
        batch_size=None,
        num_workers=0,
    )
    batch = next(iter(loader))
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))

    projected = adapter.project_batch(batch)
    outputs = adapter.forward(projected)
    loss, parts = adapter.losses(outputs, projected.targets, n_bins=model.n_bins)

    assert batch.source == "replay/projector.py"
    assert math.isfinite(float(loss.detach()))
    assert {"policy", "value", "sparse_policy"} <= set(parts)
