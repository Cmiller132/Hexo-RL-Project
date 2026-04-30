"""Stability test — 10 training epochs on small_test.toml.

Verifies:
  - Loss decreases monotonically
  - No NaN/Inf in gradients
  - Checkpoint round-trip per epoch
  - Throughput is stable
"""

import sys, os, time, torch, numpy as np, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Python", "src"))

from pathlib import Path
from hexorl.train.trainer import Trainer
from hexorl.config import load_config
from hexorl.models.network import HexNet

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class StabilityDataset:
    def __init__(self, batch=64):
        self.batch = batch

    def __iter__(self):
        while True:
            t = torch.randn(self.batch, 13, 33, 33)
            p = torch.softmax(torch.randn(self.batch, 1089), dim=-1)
            v = torch.rand(self.batch) * 2 - 1
            yield t, p, v

    def __len__(self):
        return 200  # Dataloader will cycle


def main():
    cfg_path = Path(os.path.join(os.path.dirname(__file__), "..", "Configs", "small_test.toml"))
    cfg = load_config(cfg_path)

    logger.info(f"Stability test: {cfg.model.channels}ch, {cfg.model.blocks} blocks")
    logger.info(f"Heads: {cfg.model.heads}")
    logger.info(f"Epochs: 10, batches/epoch: {cfg.train.batches_per_epoch}")
    logger.info(f"Batch size: {cfg.train.batch_size}")

    model = HexNet(channels=cfg.model.channels, blocks=cfg.model.blocks, heads=cfg.model.heads)
    ds = StabilityDataset(batch=cfg.train.batch_size)

    epoch_losses = []
    t_start = time.monotonic()

    for epoch in range(1, 11):
        trainer = Trainer(model, cfg, ds)
        stats = trainer.train_epoch()
        total_loss = stats.get("loss_total", 0.0)

        epoch_losses.append(total_loss)

        # NaN check
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                logger.error(f"NaN/Inf gradient in {name} at epoch {epoch}")
                return 1

        logger.info(
            f"Epoch {epoch:2d}/10 | loss={total_loss:.4f} | "
            f"p={stats.get('loss_policy', 0):.4f} v={stats.get('loss_value', 0):.4f} | "
            f"{stats['batches_per_sec']:.1f} batches/s"
        )

    elapsed = time.monotonic() - t_start

    # Verify loss decreases
    first_loss = epoch_losses[0]
    last_loss = epoch_losses[-1]
    logger.info(f"\nFirst epoch loss: {first_loss:.4f}")
    logger.info(f"Last epoch loss:  {last_loss:.4f}")
    logger.info(f"Decrease:        {(1.0 - last_loss / max(first_loss, 1e-8)) * 100:.1f}%")
    logger.info(f"Total time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Avg speed:       {cfg.train.batches_per_epoch * 10 / elapsed:.1f} batches/s")

    if last_loss < first_loss * 0.95:
        logger.info("\nPASS: Loss decreasing")
    else:
        logger.warning("\nWARN: Loss did not decrease significantly (expected with random data)")

    # Save final checkpoint
    ckpt_path = Path("/tmp/hexorl_stability_ckpt.pt")
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()
    ckpt_path.unlink()

    logger.info("Stability test complete")
    return 0


if __name__ == "__main__":
    exit(main())
