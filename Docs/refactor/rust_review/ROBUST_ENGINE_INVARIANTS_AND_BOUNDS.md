# Robust Engine Invariants And Bounds Plan

Date: 2026-04-29

## 3. Consistency And Invariant Hooks

### Target Shape

`HexGameState` has a debug/test consistency helper that recomputes derived state
from authoritative history and stones.  The helper is expensive and not part of
the release hot path.

### Required Checks

- move history length matches stone count;
- every history record has a matching board stone;
- current player and placements remaining match replayed turn state;
- zobrist hash equals a full recompute;
- candidate reference counts match a full radius scan;
- winner and winning line match full win detection;
- incremental eval score matches full recompute inside the eval grid;
- `ThreatCounts` match full recompute inside the eval grid;
- hot windows match full recompute inside the eval grid.

### Required Tests

- place/unplace round trips assert consistency after every individual mutation;
- intermediate unplace states are checked, not only full reset;
- branch replay after partial unplace is checked;
- far-board tactical fixtures prove tactical correctness is full-board even
  when bounded eval intentionally clips scoring data.

## 4. `WindowKey` And Eval Bounds

### Target Shape

Runtime construction of `WindowKey` is release-safe.  Coordinates outside the
representable range and invalid directions return an error or use a widened key
representation.  They never silently truncate.

### Required Changes

- Introduce a checked constructor or widen `WindowKey`.
- Use the checked path for runtime window origins.
- Keep an infallible constructor only for compile-time/test values whose inputs
  are statically known safe.
- Preserve the separation between bounded eval-grid score caches and complete
  full-board tactical scanning.

### Completion Tests

- release-mode tests cover out-of-range `WindowKey` construction.
- far-coordinate tactical tests continue to pass.
- eval-grid clipping remains explicit and does not affect tactical masks or
  tactical status.
