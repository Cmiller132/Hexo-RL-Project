# V1 Pair MCTS Response1 Review Verification

Date: May 8, 2026

This note verifies the claims in `Docs/Response1.md` against the current V1 pair
runtime source. It separates source-correctness issues from remaining
acceptance/performance evidence.

## Verified Concerns And Fixes

### Terminal Pair Ordering

Review claim: unordered pair rows can apply a filler cell before an immediate
winning cell, creating arbitrary terminal traces.

Status: real concern, fixed in source.

Fix:

- Added terminal-safe pair application in
  `crates/hexgame-core/src/v1_pair_search.rs`.
- If exactly one cell in a selected pair is an immediate current-player win,
  Rust applies that cell first for root simulation, interior simulation, and
  final selected-action application.
- Added `v1_terminal_pair_order_applies_winning_cell_before_filler`.

### Proposal Correction In Search Priors

Review claim: using proposal correction weights directly in PUCT priors can
double-correct a learned proposal distribution and distort search.

Status: real concern, fixed in source.

Fix:

- Root and interior search priors now use raw model logits.
- Correction weights and modes remain validated and preserved for replay,
  training, and telemetry.
- Added `v1_search_priors_ignore_proposal_correction_weights`.

### Progressive Widening Prior Normalization

Review claim: if PUCT considers only revealed rows but priors are normalized over
the full hidden reservoir, early exploration mass is suppressed by unrevealed
actions.

Status: real concern, fixed in source.

Fix:

- Interior reservoirs store raw prior logits.
- `select_interior_puct_edge`, `widen_tree_node_reservoir`, and standalone
  `widen_interior_reservoir` compute PUCT priors from a softmax over the
  currently revealed rows.
- Added `v1_widened_reservoir_renormalizes_priors_over_revealed_rows`.

### Final Selection Visit Bias

Review claim: final root selection can overweight early visits instead of
completed value estimates.

Status: real concern, fixed in source.

Fix:

- `select_root_action` still requires completed simulations and at least one
  visited admitted row, but `compare_candidate_final` now orders by
  completed-Q first, then visit count, prior, and deterministic pair key.
- `Docs/V1_PAIR_MCTS_SEARCH.md` now documents this ordering.

## Verified As Already Correct

### PUCT Zero-Visit Parent Term

Review claim: first selection can lose all exploration pressure if
`sqrt(parent_visits)` is zero.

Status: already handled.

Evidence:

- Root PUCT uses the root visit sum with `.max(1)`.
- Interior PUCT uses `node.visit_count.max(1)`.
- Progressive widening also uses `max(1)`.

### Full State Identity

Review claim: legal-row hash alone may not be enough to detect Python/Rust graph
drift.

Status: mostly already handled by the V1 legal row table.

Evidence:

- `LegalRowTableV1` contains `turn_start_state_hash` and
  `current_state_hash`.
- `legal_table_hash_v1(...)` includes phase, player, placements remaining,
  move count, state hash, and every legal row.
- PyO3 exposes both state hashes in the legal row table sent to Python.

Remaining evidence:

- A Rust/PyO3 lifecycle test should still assert Python sees and preserves these
  hashes across expansion requests in a Rust-capable environment.

### One-Placement Phase

Review claim: `one_placement` could be confused with normal post-opening turns.

Status: already handled.

Evidence:

- `turn_phase_v1(...)` returns `NormalTwoPlacement` whenever the turn-start
  state has two placements remaining.
- `one_placement` is reserved for structural single-placement roots.

### Root Targets Over Raw Visits

Review claim: training targets could be biased if they use raw visit counts
from sampled/admitted support.

Status: already handled for current V1 targets.

Evidence:

- `V1SearchPairMetadata.support_type` is
  `completed_q_candidate_posterior`.
- `Python/src/hexorl/train/v1_pair_targets.py` builds policy and completion
  targets from completed-Q over sampled trainable rows, while preserving the
  absolute rule that unsampled legal pairs are not negatives.

### Per-Candidate Metadata

Review claim: replay might compress away candidate-level source and correction
details.

Status: already handled.

Evidence:

- `V1SearchPairMetadata` retains per-candidate `V1CandidatePair` records plus
  visit, Q, completed-Q, Gumbel, allocation, correction, canary, and tactical
  flags.

## Real Concerns Not Claimed Fixed

### Root Gumbel Sequential Halving

Review claim: the current root behavior is not true Gumbel sequential halving.

Status: verified real concern; documentation corrected, source not changed to
Gumbel SH in this patch.

Current source behavior:

- Rust runs recursive pair-action PUCT.
- Gumbel values participate in deterministic admission/tie metadata.
- Replay stores Gumbel values, allocation, visits, Q, and completed-Q.
- Final choice is completed-Q-first over visited admitted rows.

Required product decision:

- Either implement a formal `gumbel_sequential_halving_v1` root schedule and
  corresponding improved-policy target, or amend the V1 architecture to accept
  recursive completed-Q PUCT as the root operator.

### Batched Expansion Performance

Review claim: per-request proposal and pair-scoring passes may violate the
wall-clock performance target.

Status: verified real concern; request-batch graph submission fixed in source.

Current source behavior:

- The V1 provider uses `_v1_complete_expansion_requests(...)` for each request
  batch returned by `run_search_step(...)`.
- It submits all proposal graphs together, builds all admitted-pair scoring
  graphs, submits those together, then completes each expansion back into Rust.
- Test fakes without `submit_graph_many(...)` use a fallback to
  `submit_graph(...)`.

Remaining evidence:

- Gate performance with encoder-forward count, pair-scorer-forward count,
  proposal latency, pair-scoring latency, pair-scores/sec,
  queue/backpressure, and GPU utilization artifacts.

## Local Verification Limits

Rust source tests were added but not executed locally because `cargo` and
`rustfmt` are not available on PATH in this environment. The required external
check remains:

```powershell
cargo test -p hexgame-core v1_pair_search
```

PyO3 lifecycle tests also require a built `_engine` extension.
