import torch
import pytest
from pydantic import ValidationError

import hexorl.epoch.pipeline as epoch_pipeline
from hexorl.config import Config
from hexorl.runtime import HostProfile, dataloader_worker_count


def _host(*, system: str = "linux", cuda: bool = True, physical: int = 16, logical: int = 32) -> HostProfile:
    return HostProfile(
        logical_cpus=logical,
        physical_cpus=physical,
        system=system,
        cuda_available=cuda,
        cuda_name="test",
        cuda_memory_gb=12.0 if cuda else 0.0,
        system_memory_gb=32.0,
    )


def _graph_cfg(**runtime):
    return Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "heads": ["policy_place", "value"],
            },
            "train": {
                "loss_weights": {
                    "policy": 1.0,
                    "policy_place": 1.0,
                    "value": 1.0,
                },
            },
            "runtime": runtime,
            "inference": {"fp16": False},
        }
    )


def test_linux_cuda_global_graph_defaults_to_7950x_safe_worker_count():
    assert dataloader_worker_count(_graph_cfg(), _host()) == 8
    assert dataloader_worker_count(_graph_cfg(), _host(physical=5, logical=10)) == 2


def test_windows_keeps_dataloader_workers_disabled_by_default():
    assert dataloader_worker_count(_graph_cfg(), _host(system="windows")) == 0


def test_explicit_dataloader_worker_overrides_win():
    assert dataloader_worker_count(_graph_cfg(dataloader_workers=3), _host()) == 3
    assert dataloader_worker_count(_graph_cfg(graph_dataloader_workers=5), _host()) == 5
    assert dataloader_worker_count(
        _graph_cfg(dataloader_workers=3, graph_dataloader_workers=0),
        _host(),
    ) == 0


def test_non_graph_models_keep_default_workers_disabled_on_linux_cuda():
    cfg = Config.model_validate({"model": {"architecture": "cnn"}, "inference": {"fp16": False}})
    assert dataloader_worker_count(cfg, _host(), global_graph_model=False) == 0


def test_graph_worker_thread_config_is_validated():
    cfg = _graph_cfg(graph_worker_torch_threads=1, dataloader_prefetch_factor=2)
    assert cfg.runtime.graph_worker_torch_threads == 1
    assert cfg.runtime.dataloader_prefetch_factor == 2
    assert torch.get_num_threads() >= 1


def test_legacy_graph_collate_in_worker_config_is_rejected():
    with pytest.raises(ValidationError):
        _graph_cfg(graph_collate_in_worker=True)


def test_run_epoch_retries_single_process_on_dataloader_worker_failure(monkeypatch, tmp_path):
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "heads": ["policy_place", "value"],
            },
            "train": {
                "batch_size": 1,
                "batches_per_epoch": 1,
                "loss_weights": {
                    "policy": 1.0,
                    "policy_place": 1.0,
                    "value": 1.0,
                },
            },
            "runtime": {"graph_dataloader_workers": 2},
            "inference": {"fp16": False},
        }
    )

    class FakeDataLoader:
        def __init__(self, _dataset, **kwargs):
            self.num_workers = int(kwargs.get("num_workers", 0))
            self.prefetch_factor = int(kwargs.get("prefetch_factor", 0) or 0)

    class FakeTrainer:
        def __init__(self, model, cfg, dataloader, device=None):
            self.model = model
            self.cfg = cfg
            self.dataloader = dataloader
            self.device = device
            self.batches_per_epoch = cfg.train.batches_per_epoch
            self.global_step = 0
            self.epoch = 0

        def train_epoch(self):
            self.epoch += 1
            if self.dataloader.num_workers > 0:
                raise RuntimeError("DataLoader worker exited unexpectedly")
            return {
                "epoch": float(self.epoch),
                "batches": float(self.batches_per_epoch),
                "elapsed_s": 0.01,
                "batches_per_sec": 100.0,
            }

        def save_checkpoint(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    monkeypatch.setattr(epoch_pipeline, "DataLoader", FakeDataLoader)
    monkeypatch.setattr(epoch_pipeline, "Trainer", FakeTrainer)

    result = epoch_pipeline.run_epoch(
        cfg,
        model=torch.nn.Linear(1, 1),
        output_dir=tmp_path,
        train=True,
    )

    assert result.train_stats["dataloader_worker_fallback"] == 1.0
    assert result.train_stats["epoch"] == 1.0
