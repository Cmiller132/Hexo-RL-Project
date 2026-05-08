# Rust Engine API Reference

This document summarizes the current Rust-facing API exposed to Python through the `_engine` PyO3 module.

## Crates

### `hexgame-core`

Core Rust crate for game rules, encoding, tactics, classical search, and MCTS.

Important public areas:

- `hexgame_core::rules`: axial coordinates, turn placement rules, board state, game errors, and rule constants.
- `hexgame_core::encoding`: canonical 13-channel board encoding and shape constants.
- `hexgame_core::tactics`: tactical classification, blocking constraints, masks, and tactical filters.
- `hexgame_core::classical`: classical alpha-beta / iterative deepening search.
- `hexgame_core::mcts`: MCTS engine and MCTS error types used by the Python FFI crate.

Important constants:

| Constant | Meaning |
|---|---|
| `BOARD_SIZE = 33` | Board tensor width/height. |
| `NUM_CHANNELS = 13` | Encoder channel count. |
| `BOARD_AREA = 1089` | Dense policy size. |
| `TENSOR_SIZE = 14157` | `13 * 33 * 33`. |
| `PLACEMENT_RADIUS = 8` | Canonical global legal-row radius. |

### `hexgame-py`

PyO3 crate that exposes `_engine` to Python.

The active production pipeline uses `HexGame`, `MCTSEngine`, encoding helpers, D6 transforms, and classical bootstrap helpers.

## Python Module Constants

The `_engine` module exports:

```text
FEATURE_COUNT
WIN_LENGTH
PLACEMENT_RADIUS
BOARD_SIZE
NUM_CHANNELS
TENSOR_SIZE
```

## `HexGame`

Python wrapper around Rust `HexGameState`.

Common methods:

- `place(q, r)`
- `unplace()`
- `is_over`
- `winner`
- `current_player`
- `placements_remaining`
- `legal_moves()`
- `legal_moves_near(radius)`
- `encode_board_and_legal(near_radius, constrain_threats)`
- `classical_search(time_ms, max_depth, near_radius, noise_level)`

## `MCTSEngine`

Python wrapper around Rust `hexgame_core::mcts::MCTSEngine`.

Constructor:

```text
MCTSEngine(
    game,
    num_simulations,
    c_puct=1.4,
    near_radius=8,
    c_puct_init=19652.0,
    constrain_threats=true,
    arena_sim_hint=None,
    seed=0,
)
```

### Root Initialization

```text
init_root() -> None | (tensor_3d, offset_q, offset_r, legal_bytes, root_generation)
```

- Returns `None` for terminal roots.
- Returns a 13-channel tensor and the exact legal rows for root expansion.
- `root_generation` is a stale-token guard. Root expansion APIs reject stale generations.

### Root Expansion APIs

Dense root expansion:

```text
expand_root(policy, value, offset_q, offset_r, legal_bytes, root_generation)
```

Sparse prior expansion:

```text
expand_root_with_sparse_priors(
    policy,
    value,
    offset_q,
    offset_r,
    legal_bytes,
    root_generation,
    sparse_qr,
    sparse_logits,
    stage,
    sparse_mix,
)
```

Global graph prior expansion:

```text
expand_root_with_global_priors(
    legal_bytes,
    root_generation,
    global_qr,
    global_logits,
    value,
)
```

Validation performed by these APIs includes:

- policy/logit arrays are contiguous where required
- dense policy length is `BOARD_AREA`
- logits/values are finite
- `legal_bytes` decode correctly
- root generation matches the initialized root
- global and sparse row arrays have expected shapes
- global logits have at least one logit per global row
- global rows are validated by the Rust MCTS core against legal rows

### Root Pair-Prior APIs

First-placement unordered pair prior expansion:

```text
apply_root_pair_priors(pair_qr, pair_logits, pair_mix)
```

First-action prior expansion from `policy_pair_first`:

```text
apply_root_pair_first_priors(action_logits, pair_mix)
```

Second-placement known-first pair prior expansion:

```text
apply_root_pair_second_priors(pair_qr, pair_logits, pair_mix)
```

Pair APIs validate pair row shape and delegate legality/identity checks to the Rust MCTS core. Illegal pairs, duplicate cells, reversed duplicate unordered pairs, and wrong known-first rows are rejected by Rust tests.

### Leaf Selection And Backprop

Common leaf APIs:

```text
select_leaves(batch_size) -> (tensor_4d, count, batch_generation)
pending_leaf_metadata() -> metadata for selected pending leaves
expand_and_backprop(policies, values, batch_generation)
```

Important behavior:

- `batch_generation` guards selected leaves against stale expansion.
- Global graph leaf expansion uses `pending_leaf_metadata()` to rebuild graph batches in Python from compact histories and legal rows.
- Expansion APIs reject stale batch tokens, wrong lengths, non-finite values, and malformed sparse/global metadata.

### Search Results And Tree Utilities

Common methods:

- `add_dirichlet_noise(noise, noise_fraction)`
- `sample_action(temperature, rng_state)`
- `should_resign(threshold)`
- `re_root(q, r, new_num_simulations)`
- `get_results() -> (moves_q, moves_r, visits, root_value)`
- `root_child_priors()`
- `root_child_q_values()`
- `root_child_prior_sources()`
- `prior_source_summary()`
- `root_pair_visit_targets()`
- `extract_tree_node_histories(min_visits)`
- `extract_tree_node_states(min_visits)`

The Python self-play worker uses these methods for replay targets, RGSC candidate extraction, pair-policy targets, and telemetry.

## Standalone Functions

Common module-level helpers:

- `encode_compact_record(history_bytes, near_radius)` -> encoded tensor batch
- `apply_d6_symmetry(tensor, sym_idx)` -> transformed `(13, 33, 33)` tensor
- `classical_self_play(num_games, time_ms, max_depth, near_radius, max_moves)` -> classical bootstrap records

## Current Integration Notes

- Rust remains the canonical rules and MCTS boundary.
- Python owns model inference, batching, training, replay orchestration, dashboards, and config.
- Dense, sparse, and global graph roots use different expansion APIs but all validate legal-row identity before MCTS consumes priors.
- Pair priors are only valid at root today and are gated by Python `pair_strategy` configuration.
- The planned modular model refactor should move Python-side interpretation of model outputs behind policy providers and pair strategies, but the Rust APIs above remain the current validated boundary.
