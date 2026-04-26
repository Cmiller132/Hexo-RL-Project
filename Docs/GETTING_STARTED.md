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

# Install Python dependencies
pip install torch numpy pydantic tomli maturin

# Build the Rust Python extension
maturin develop --features python
```

## Verify Installation

```bash
# Run Rust tests
cargo test -p hexgame-core

# Run Python smoke test
pytest python/tests/test_engine_smoke.py -v

# Check Python imports
python -c "import sys; sys.path.insert(0,'python/src'); import hexorl; print(hexorl.__version__)"
```

## Quick Test: Inference Server

```bash
# Start inference server with stub model
python -c "
import sys; sys.path.insert(0,'python/src')
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
python benches/train_epoch.py  # Or create: python models/train_epoch.py
```

## Running a Full Pipeline

```bash
# Self-play + training
python -m hexorl.cli epoch configs/small_test.toml

# Arena evaluation
python -m hexorl.cli arena configs/small_test.toml
```

## Directory Structure

```
crates/          — Rust workspace crates
python/src/      — Python package (hexorl)
configs/         — TOML configuration files
benches/         — Benchmark scripts
Docs/            — Technical documentation
```

## Common Issues

1. **Rust extension not found:** Run `maturin develop --features python`
2. **CUDA not available:** Training works on CPU/MPS (slower)
3. **Shared memory errors:** macOS may need `ulimit -n` increased
4. **Config validation fails:** Check `configs/small_test.toml` against `schema.py`
