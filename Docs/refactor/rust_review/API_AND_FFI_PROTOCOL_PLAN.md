# Rust Public API And FFI Protocol Plan

Date: 2026-04-29

Purpose: define the stable Rust surface that downstream Rust crates and the Python extension may depend on during the refactor. This plan covers the public API and FFI protocol sections of the Rust review slice.

## Public API Boundary

The stable `hexgame-core` API is organized around facade modules:

| Facade | Stable responsibility | Public examples |
| --- | --- | --- |
| `rules` | Board state, coordinates, turns, legal placement constants, move records, and rule errors. | `HexGameState`, `Hex`, `Turn`, `GameError`, `PLACEMENT_RADIUS`, `WIN_LENGTH` |
| `encoding` | Canonical neural tensor encoding and shape constants. | `encode_board_into`, `EncodedBoard`, `BOARD_AREA`, `NUM_CHANNELS`, `TENSOR_SIZE` |
| `tactics` | Complete tactical classification and mask/filter helpers. | `TacticalStatus`, `BlockConstraint`, `tactical_status`, `turn_satisfies_tactical`, `tactical_mask_cells` |
| `classical` | Classical alpha-beta search entry points. | `iterative_deepening`, `SearchResult` |

The crate root should not be a catch-all convenience namespace. New callers should import through the facade that owns the concept. Root exports of implementation details should stay absent unless a caller has an active, documented need that cannot use a facade.

`mcts` remains a public module because `hexgame-py` is a sibling crate and needs direct access to the neural MCTS engine for the compiled Python extension. It is not re-exported from the crate root, and it is not part of the stable facade list. Any future stable MCTS API should be designed as its own facade rather than restoring root convenience exports.

## ThreatStatus Removal

`ThreatStatus` is not part of the public API. The public tactical model is `tactics::TacticalStatus`, because it preserves all immediate winning turns rather than collapsing them to a single deterministic move. Legacy `ThreatStatus` language should not appear in public docs, README material, or root exports.

The compatibility helpers have been removed rather than kept private. Oracle
tests now exercise `TacticalStatus` directly, so there is no second tactical
model inside the core crate.

## FFI Protocol Ownership

The Python extension may depend on:

- `rules` for board state, move history rows, legal rows, constants, and errors.
- `encoding` for tensor and legal-mask layout.
- `tactics` for tactical oracle responses exposed through `PyHexGame`.
- `classical` for classical search wrappers.
- `mcts` for the active PyO3 MCTS engine wrapper.

The FFI byte protocols are owned by `crates/hexgame-py/src/protocol.rs`. They should remain explicit row encodings instead of ad hoc Python tuples at Rust boundaries:

| Protocol | Row layout | Owner |
| --- | --- | --- |
| Legal rows | `i32 q`, `i32 r` | `protocol::encode_legal_rows` / `decode_legal_rows` |
| Board piece rows | `i32 q`, `i32 r`, `i32 player` | `protocol::encode_board_piece_rows` |
| Compact history rows | `i32 player`, `i32 q`, `i32 r` | `protocol::encode_compact_history_rows` / `decode_compact_history_rows` |
| Pair rows | `i32 q1`, `i32 r1`, `i32 q2`, `i32 r2` ndarray view | `protocol::decode_pair_rows` |

FFI protocol changes must include a Python smoke or invariant test that proves shape, row width, ordering, and stale-token behavior. Protocol readers should validate row widths before decoding and return structured Python exceptions rather than panicking.

## Downstream Migration Rules

1. In-repo Rust callers use facade imports, for example `hexgame_core::rules::HexGameState`.
2. Python binding code imports root constants only through their owning facades.
3. Public docs name only the stable facades and the active `mcts` FFI module exception.
4. New implementation modules stay private until they have an intentional facade or FFI reason to be public.
