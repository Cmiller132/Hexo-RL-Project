# Phase 00 Adversarial Review

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`

| Attack attempt | Evidence | Resolution |
|---|---|---|
| Enable pair scoring through `global_xattn_0` architecture string | Pair policy audit and tests cover architecture/head/mix coupling | Runtime now gates on `pair_strategy != none`; default self-play smoke scored zero pair rows. |
| Enable pair scoring with pair-capable heads plus nonzero `pair_prior_mix` | `test_global_xattn_pair_heads_do_not_enable_pair_scoring_without_strategy` in full pytest transcript | Pair heads and mix no longer enable `SelfPlayWorker.pair_policy_enabled`. |
| Run full pair enumeration without explicit diagnostic cap | `test_pair_scoring_requires_explicit_diagnostic_strategy_and_cap` in full pytest transcript | Scoring helper raises when cap is non-positive; non-none config requires positive cap. |
| Treat Rust outputs as self-validating | Rust/Python boundary inventory and verification inventory map malformed bytes, stale MCTS tokens, and source/hash requirements | Phase 00 records suspicion and negative-test owners; no Python fallback is added. |
| Close inventories without import/code-search proof | Four import audit files under `import_audits/` show remaining surfaces and owner phases | Remaining legacy paths are not claimed deleted; owner phases are explicit. |
| Use manual-only watchdog proof | `phase00_watchdog_smoke_expected_abort.txt` records a command-backed expected abort | The event artifact includes last phase, last engine op, counters, pair strategy, pair rows scored, and next action. |

No unresolved Phase 00 adversarial finding remains. Later-phase findings are tracked as owner-phase inventory, not Phase 00 blockers.
