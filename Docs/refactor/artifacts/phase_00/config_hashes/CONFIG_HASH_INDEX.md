# Phase 00 Config Hash Index

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`
- Digest algorithm: SHA-256 over exact file bytes or exact inline-config source script.

| Hash ID | Input class | Source | SHA-256 | Consuming commands |
|---|---|---|---|---|
| `P00-CONFIG-DEFAULT` | config | `Configs/default.toml` | `96f5d1a3a3bb7365dca1a7a0dd4cd1c33e173f4201c5a5043ae04b2f39729eb2` | baseline capture; runtime smokes where applicable |
| `P00-CONFIG-DEFAULT_CONFIG` | config | `Configs/default_config.toml` | `afdc6b660cd21b42a3741bf645cb34a55b0b6d2c1fa2a1cea803a8ee6e0bc419` | baseline capture; runtime smokes where applicable |
| `P00-CONFIG-PRODUCTION` | config | `Configs/production.toml` | `3ec548a50e18f4ac0d5142e5b6471fcf0bdd2e652f253f78d835a31962e54835` | baseline capture; runtime smokes where applicable |
| `P00-CONFIG-REPRODUCIBLE` | config | `Configs/reproducible.toml` | `df3cc23192cca290155773a9c3260586255563222aa55f7be9e9e022ad3ba7f9` | baseline capture; runtime smokes where applicable |
| `P00-CONFIG-SMALL_TEST` | config | `Configs/small_test.toml` | `939338e34a349b30267f229e7ff909c55589b471b9307bd0276fdf2cc00af04f` | baseline capture; runtime smokes where applicable |
| `P00-CONFIG-WSL_SPEED_PROBE` | config | `Configs/wsl_speed_probe.toml` | `dc08e68ec2cd17906e00c7bc29e09bc66dd7149007ceefc6b656e26952123b55` | baseline capture; runtime smokes where applicable |
| `P00-HASH-RUST-PYO3-MANIFEST` | rust-python-boundary | `crates/hexgame-py/Cargo.toml` | `0524b08db4f4ad3c29d7ef5e869b23272fac01aceffd19889dc13473dcb585d4` | see COMMAND_INDEX.md |
| `P00-HASH-CARGO-LOCK` | rust-workspace | `Cargo.lock` | `09cc669a4a7a37330e363996f0f5b1aa7d6722d0906bbadadc9e722b4811e8e6` | see COMMAND_INDEX.md |
| `P00-HASH-PYTHON-PROJECT` | python-env | `Python/pyproject.toml` | `0a8c38ee08ffb803b5c135e60d05f6fb788fd53614d0d0c39191738a455c8cbf` | see COMMAND_INDEX.md |
| `P00-HASH-DASHBOARD-DEPS` | dashboard-deps | `Python/dashboard_frontend/package-lock.json` | `b155c0055d8266a834861711efecf348db4bbadf9e7436bb2b2ecaf2a413050a` | see COMMAND_INDEX.md |
| `P00-HASH-BASELINE-SCRIPT` | telemetry-baseline-script | `scripts/phase00_capture_baseline.py` | `b5a8b86cdd54e4db60651c0749b741c5e0671c189b112105fc4587f3a5adbdc0` | see COMMAND_INDEX.md |
| `P00-HASH-RUNTIME-SMOKE-SCRIPT` | inline_phase00_tiny_cfg | `scripts/phase00_runtime_smoke.py` | `b21761fabd3a2b9c0b6443a842e7a385b9f1710bc5f78da62760b28455736d8c` | see COMMAND_INDEX.md |
| `P00-HASH-FINALIZE-SCRIPT` | phase00_artifact_inputs | `scripts/phase00_finalize_artifacts.py` | `9253815f67acbf60472280a45eaa98bad2409fe6586bb5b47ed09c3e6e0d5c35` | see COMMAND_INDEX.md |
| `P00-HASH-V2-MATRIX` | matrix | `Docs/refactor/V2_REQUIREMENTS_MATRIX.md` | `11992aadcd4ab9bcd59586ec075ff3dc38de2e827204cd40736195c487a1dd7a` | see COMMAND_INDEX.md |
