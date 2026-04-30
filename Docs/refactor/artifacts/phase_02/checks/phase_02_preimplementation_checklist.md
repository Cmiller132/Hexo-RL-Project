# Phase 02 Pre-Implementation Checklist

- Created: `2026-04-30T00:29:18-04:00`
- Git SHA: `549c14d8f4837b98257a95f4fd5b4496e26b76b1`
- Branch: `codex/phase-02-builder-convergence`
- Source docs: `Docs/refactor/phases/PHASE_02.md`, `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`
- Scope: implement and close Phase 02 rows `V2-020` through `V2-025`.

## Goal

Converge candidate, pair-action, and global-graph construction onto shared canonical builders consumed by runtime, training, replay, evaluation, and dashboard paths.

## Success Criteria

- [ ] `V2-020`: `CandidateContractBuilder` is the one production owner for candidate rows, dense indices, masks, targets, diagnostics, feature extensions, and contract hash.
- [ ] `V2-021`: `PairActionTableBuilder` is the one production owner for pair row identity, phase semantics, known-first handling, caps, counts, masks, targets, and table hash.
- [ ] `V2-022`: `PairCandidateBatch` is deleted or reduced to a tensor-only projection from `PairActionTable` with no independent semantics.
- [ ] `V2-023`: `GraphSemanticBuilder` owns graph token/relation/link semantics and `GraphTensorizer` plus collator own only projection, padding, batching, and shape validation.
- [ ] `V2-024`: self-play, replay sampler/projector, training adapters, evaluation debug payloads, dashboard fixtures/inspectors, and model-input fixture generation consume the same builders.
- [ ] `V2-025`: candidate, pair, and graph projections are pure, mutation-safe, D6-verified, and corruption-tested.
- [ ] Runtime import/code-search audits prove private candidate, pair, graph, legal, D6, and history rebuild paths are absent from Phase 02 consumers.
- [ ] Golden, negative, mutation, D6, extension-proof, debug-bundle, telemetry, and performance evidence is produced under this artifact directory.
- [ ] Phase packet files are reconciled: `agent_completion_packet.md`, `evidence_reconciliation.md`, and `exit_gate_report.md`.

## Constraints

- No later-phase implementation is claimed complete by Phase 02.
- `PairStrategy` work is limited to the Phase 02 minimum needed to make pair selection/caps explicit; final policy/search strategy ownership remains Phase 05.
- No runtime compatibility shim may keep old and new semantic owners alive together.
- Full pair enumeration is rejected unless an explicit capped strategy requests it.
- Rust legal/history data remains canonical but is semantically validated before contract use.
- Graph tensorization must not regenerate legal rows, candidates, pair rows, tactical facts, history, or D6 transforms.
- `rg` is preferred, but local `rg.exe` access previously failed; use `git grep` or PowerShell fallback and record command output.

## Required Evidence

- Phase scope and matrix-row checklist in this file.
- Interface freeze notes in `checks/interface_freeze_notes.md`.
- Fixture/artifact plan in `fixtures_or_references/golden_fixture_plan.md`.
- CI routing plan in `commands/ci_routing_plan.md`.
- Command transcripts with exit codes in `commands/` and `test_output/`.
- Import/code-search audits in `import_audits/`.
- Deletion manifest in `deletion_manifest/phase02_deletion_manifest.md`.
- Telemetry/debug samples in `telemetry_samples/`.
- Performance artifacts in `performance/`.
- Contract examples/docs in `contract_examples/`.
- Adversarial review in `checks/adversarial_review.md`.
- Completion packet, evidence reconciliation, and exit gate report at the phase artifact root.

## Stop Rules

- Stop if a Phase 02 deletion would remove the only consumed runtime path before the replacement builder is wired.
- Stop if a phase-closing invariant would need to be skipped, xfailed, flaky-only, or manual-only.
- Stop if hot-path performance evidence cannot be produced for touched candidate, pair, or graph projection paths.
- Stop if implementing Phase 02 requires a runtime compatibility shim or fallback not approved by the phase.
- Stop if subagent work overlaps write ownership without a clear integration plan.

No stop rule is active at scope freeze. Existing private paths have replacement owners defined below, and code edits will proceed only after those interfaces are frozen.

## Matrix Row Checklist

| Row | Owner | Required close evidence |
|---|---|---|
| `V2-020` | `Python/src/hexorl/contracts/candidates.py` | builder API, golden parity, feature extension test, mutation/corruption tests, runtime consumer imports |
| `V2-021` | `Python/src/hexorl/contracts/pairs.py`, Phase 02 pair strategy cap helper | phase-aware pair tests, known-first tests, cap rejection, D6 pair tests, telemetry |
| `V2-022` | pair projection code | deletion/import audit, projection-only tests, no semantic fields outside `PairActionTable` |
| `V2-023` | `Python/src/hexorl/graph/semantic_builder.py`, `tensorize.py`, `collate.py` | projection tests, graph schema tests, no semantic rebuild in tensorizer/collator |
| `V2-024` | self-play, replay/sampler, training, eval, dashboard consumers | golden equality tests and import audits across consumers |
| `V2-025` | candidate/pair/graph projection contracts | mutation safety, D6 inverse/composition checks, corruption failure localization |

