# Phase 03 Command Index

Command transcripts are recorded in `command_transcripts.md`.

Required closing command classes:

- Python compile
- focused model registry/spec/checkpoint pytest
- focused train adapter pytest
- one-batch trainer smoke pytest
- dashboard/eval/inference smoke updates for model import cutover
- import/deletion audits
- performance/debug bundle probe

Current status:

- Focused model registry/spec/checkpoint tests passed.
- Focused train adapter tests passed.
- Existing config, graph, production smoke, and training data suites passed.
- Inference server suite including Rust-engine MCTS round-trip passed.
- Import/deletion audits passed with documented `rg` OS fallback.
