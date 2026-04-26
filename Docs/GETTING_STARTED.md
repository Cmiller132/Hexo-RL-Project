# Getting Started with Hexo-RL

## Prerequisites

- Rust 1.87+ with `cargo`
- Python 3.10+ with `pip`
- PyTorch 2.0+ (CUDA recommended for training)
- `maturin` for building the Rust extension

## Installation

```bash
# Clone
git clone <repo> && cd Hexo-RL-Project

# Install Rust dependencies
cargo build --workspace

# Install Python package and dev tools
python -m pip install -e 'Python[dev]' pytest maturin

# Build the Rust Python extension
python -m maturin develop --manifest-path crates/hexgame-py/Cargo.toml
```

## Verify Installation

```bash
# Run Rust tests
cargo test -p hexgame-core

# Run Python smoke test
PYTHONPATH=Python/src python -m pytest Python/tests/test_engine_smoke.py -v

# Check Python imports
PYTHONPATH=Python/src python -c "import hexorl, _engine; print(hexorl.__version__, _engine.BOARD_SIZE)"
```

## Quick Test: Inference Server

```bash
PYTHONPATH=Python/src python -c "
from hexorl.config import load_config
from hexorl.inference.server import InferenceServer

cfg = load_config()
cfg.model.channels = 8
cfg.model.blocks = 2
cfg.inference.fp16 = False

server = InferenceServer(cfg, num_workers=1)
server.start()
print('Server running...')
server.stop()
server.join()
print('Server stopped')
"
```

## Quick Test: One Training Epoch

```bash
PYTHONPATH=Python/src python -m hexorl.cli smoke-train --epochs 3 --output-dir /tmp/hexorl_smoke
```

## Running a Full Pipeline

```bash
# Self-play + training
PYTHONPATH=Python/src python -m hexorl.cli epoch --config configs/small_test.toml --output-dir runs/small --bootstrap-games 16

# Arena evaluation
PYTHONPATH=Python/src python -m hexorl.cli arena --games 20 --time-ms 250 --depth 2
```

## Directory Structure

```
crates/          — Rust workspace crates
Python/src/      — Python package (hexorl)
configs/         — TOML configuration files
benches/         — Benchmark scripts
Docs/            — Technical documentation
```

## Common Issues

1. **Rust extension not found:** Run `python -m maturin develop --manifest-path crates/hexgame-py/Cargo.toml`
2. **CUDA not available:** Training works on CPU/MPS (slower)
3. **Shared memory errors:** macOS may need `ulimit -n` increased
4. **Config validation fails:** Check `configs/small_test.toml` against `schema.py`
