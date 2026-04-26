# Rust Engine API Reference

## hexgame-core

### Core Types
- `Hex` — axial coordinate (q, r)
- `Turn` — 1 or 2 placements
- `HexGameState` — board state, place/unplace, win detection
- `EvalState` — incremental pattern evaluation
- `ThreatStatus` — threat classification
- `MCTSEngine` — neural MCTS with PUCT
- `MCTSError` — error type for MCTS operations

### Encoder Constants
- `BOARD_SIZE = 33` — tensor width/height
- `NUM_CHANNELS = 13` — feature channels
- `BOARD_AREA = 33 * 33 = 1089` — policy output size
- `TENSOR_SIZE = 13 * 33 * 33 = 14157`

## hexgame-py (FFI)

### PyHexGame
Python wrapper around HexGameState. Methods:
- `place(q, r)`, `unplace()`, `is_over`, `winner`, `current_player`
- `legal_moves()`, `legal_moves_near(radius)`
- `encode_board_and_legal(near_radius, constrain_threats)` → (tensor, offset_q, offset_r, legal_bytes)
- `classical_search(time_ms, max_depth, near_radius, noise_level)` → (q, r, score, depth, nodes)

### PyMCTSEngine
- `new(game, num_simulations, c_puct, near_radius, c_puct_init, constrain_threats, arena_sim_hint, seed)`
- `init_root()` → (tensor_3d, offset_q, offset_r, legal_bytes) or None
- `select_leaves(batch_size)` → (tensor_4d, count)
- `expand_and_backprop(policies, values)`
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
