# Phase 02 Golden Fixture Plan

Golden positions:

- opening state
- first-placement state with many legal rows
- second-placement known-first state
- terminal or near-terminal state
- pair-heavy state requiring explicit cap behavior
- graph-token-heavy state with stones, windows, cover sets, hot cells, and components
- invalid history and corrupted row-link cases

Each fixture must record:

- compact history bytes or fixture rows
- legal table hash
- candidate table hash
- pair table hash when requested
- graph semantic hash
- graph tensor projection shape summary
- D6 variant identity for at least one non-trivial transform
- expected failure owner for negative cases

