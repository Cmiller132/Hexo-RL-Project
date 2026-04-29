# Refactor Phase Checklist

Use this checklist during execution. A phase is complete only if every box is checked.

## Universal Checklist (applies to every phase)

- [ ] Scope for this phase is frozen and documented.
- [ ] Entry criteria satisfied.
- [ ] Unit tests for new behavior added.
- [ ] Integration tests updated/added.
- [ ] Legacy/new parity checks executed (if replacement phase).
- [ ] Telemetry fields added for contract source/version.
- [ ] Performance smoke compared to baseline.
- [ ] CI green on branch.
- [ ] Rollback point tagged.
- [ ] Exit criteria evidence attached.

## Phase-by-Phase Quick Gates

### Phase 0
- [ ] Baseline `cargo test --workspace` pass recorded.
- [ ] Baseline `pytest -q Python/tests` pass recorded.
- [ ] Baseline inference/self-play benchmark snapshots saved.

### Phase 1
- [ ] `contracts/` package created with versioned contract objects.
- [ ] Contract validation + hash tests pass.

### Phase 2
- [ ] `engine/` boundary in place.
- [ ] Legal/replay parity suite reports zero mismatches.
- [ ] Production fallbacks disabled.

### Phase 3
- [ ] `models/` registry/spec/capability system active.
- [ ] Legacy `model/network.py` only adapter/delegation role.

### Phase 4
- [ ] Inference protocol object model adopted by all callers.
- [ ] Latency/throughput smoke within agreed envelope.

### Phase 5
- [ ] Pair strategy explicit and default=`none`.
- [ ] No implicit pair scoring via architecture/config side effects.

### Phase 6
- [ ] Self-play worker decomposed and responsibilities separated.
- [ ] Replay/tactical completeness regression tests pass.

### Phase 7
- [ ] `replay/` canonical pipeline active.
- [ ] Training/eval consume unified contracts.

### Phase 8
- [ ] Dashboard views consume runtime contracts.
- [ ] Dashboard private reconstruction paths removed.

### Phase 9
- [ ] Legacy modules deleted.
- [ ] CI policy checks enforce invariants.
- [ ] Final end-to-end smoke run archived.
