# Phase 03 Performance Evidence

Commands:

```text
python -m pytest Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_train_adapter_projection_and_device_transfer_profile_is_recorded -q
exit=0 as part of 30-test Phase 03 focused run
```

Evidence recorded:

- `TrainAdapter.project_batch` moves tensors with `non_blocking=True`.
- Crop tensors preserve channels-last conversion through the trainer hot path.
- Adapter projection clones validated tensors to guard against source mutation without adding full semantic validation to the model forward hot path.
- One-batch trainer smoke passed for dense, restnet, graph-hybrid, global-xattn, global-line-window, and global-relation representative families.
- Inference server suite passed with Rust-engine MCTS round trip: `7 passed in 14.84s`.
- MCTS probe processed one batch of 8 positions and completed server cleanup.
