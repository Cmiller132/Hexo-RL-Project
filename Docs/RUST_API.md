# Rust Engine API Reference

## hexgame-core

### Stable Facades
- `hexgame_core::rules` — coordinates, turns, board state, move records, rule constants, and rule errors.
- `hexgame_core::encoding` — canonical 13-channel tensor encoding and shape constants.
- `hexgame_core::tactics` — complete tactical classification, blocking constraints, and tactical mask/filter helpers.
- `hexgame_core::classical` — classical alpha-beta search.

### Rules Types
- `rules::Hex` — axial coordinate (q, r)
- `rules::Turn` — 1 or 2 placements
- `rules::HexGameState` — board state, place/unplace, win detection
- `rules::GameError` — rule error type

### Tactics Types
- `tactics::TacticalStatus` — complete tactical classification
- `tactics::BlockConstraint` — exact cells/pairs that satisfy mandatory blocks

### Encoder Constants
- `encoding::BOARD_SIZE = 33` — tensor width/height
- `encoding::NUM_CHANNELS = 13` — feature channels
- `encoding::BOARD_AREA = 33 * 33 = 1089` — policy output size
- `encoding::TENSOR_SIZE = 13 * 33 * 33 = 14157`

### Active Implementation Module
- `hexgame_core::mcts::MCTSEngine` and `hexgame_core::mcts::MCTSError` remain public for the Python FFI crate. They are not re-exported from the crate root.

## hexgame-py (FFI)

### PyHexGame
Python wrapper around HexGameState. Methods:
- `place(q, r)`, `unplace()`, `is_over`, `winner`, `current_player`
- `legal_moves()`, `legal_moves_near(radius)`
- `encode_board_and_legal(near_radius, constrain_threats)` → (tensor, offset_q, offset_r, legal_bytes)
- `classical_search(time_ms, max_depth, near_radius, noise_level)` → (q, r, score, depth, nodes)

### PyMCTSEngine
- `new(game, num_simulations, c_puct, near_radius, c_puct_init, constrain_threats, arena_sim_hint, seed)`
- `init_root()` → (tensor_3d, offset_q, offset_r, legal_bytes, root_generation) or None
- `expand_root(policy, value, offset_q, offset_r, legal_bytes, root_generation)` and `expand_root_with_sparse_priors` / `expand_root_with_global_priors` validate the current root token, row identity, finite priors, and policy shape before expansion.
- `select_leaves(batch_size)` → (tensor_4d, count, batch_generation)
- `pending_leaf_metadata()` → legal rows and compact histories for the currently selected pending leaves; callers must pair this metadata with the returned `batch_generation`
- `expand_and_backprop(policies, values, batch_generation)` and sparse/source variants reject stale batch tokens, wrong lengths, non-finite values, and malformed sparse metadata.
- root pair-prior methods accept only canonical pair rows and fail on stale root tokens or invalid pair identities.
- `sample_action(temperature, rng_state)` → (q, r)
- `should_resign(threshold)` → bool
- `re_root(q, r, new_num_simulations)` → PyResult
- `get_results()` → (moves_q, moves_r, visits, root_value)
- `root_child_priors()`, `root_child_q_values()`
- `extract_tree_node_states(min_visits)` → (tensors, histories, count)

### Standalone Functions
- `encode_compact_record(history_bytes, near_radius)` → ndarray (N, 13, 33, 33)
- `apply_d6_symmetry(tensor, sym_idx)` → ndarray (13, 33, 33)
- `classical_self_play(num_games, time_ms, max_depth, near_radius, max_moves)` → list of (features, outcome, board_snap)

### Module Constants
- `FEATURE_COUNT`, `WIN_LENGTH`, `PLACEMENT_RADIUS`, `BOARD_SIZE`, `NUM_CHANNELS`, `TENSOR_SIZE`
