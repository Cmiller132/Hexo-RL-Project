# Rust Phase 1 Risk Register

| ID | Classification | Area | Severity | Confidence | Phase 2 Action |
|---|---|---|---|---|---|
| D1 | Direct issue | `set_position` transactionality | High | High | Add state-preservation regression test; fix with prevalidation or temp-state swap. |
| D2 | Direct issue | MCTS pending virtual loss | High | High | Test repeated `select_leaves`; require cleanup before clearing pending. |
| D3 | Direct issue | MCTS Python root expansion | High | High | Stale/mutated `legal_bytes` tests; add root row/hash/generation guard. |
| D4 | Direct issue | Python CI Cargo feature | High | High | Run workflow command; add feature or remove invalid flag. |
| D5 | Direct issue | Unordered pair priors | Medium | High | Reversed-duplicate pair test; reject or canonicalize. |
| D6 | Direct issue | `PLACEMENT_RADIUS` rustdoc | Low | High | Correct documentation. |
| H1 | Hypothesis | `set_position` terminal/history semantics | Medium-High | Medium | Same-board/different-order and post-terminal fixtures. |
| H2 | Hypothesis | Candidate-set drift | Medium-High | Medium | Cached-vs-bruteforce proptests after every place/unplace. |
| H3 | Hypothesis | MCTS `i16` coordinate truncation | Medium | High | Far-coordinate action round-trip tests. |
| H4 | Hypothesis | Dense non-finite policy logits | Medium | Medium | Non-finite dense root/leaf policy tests. |
| H5 | Hypothesis | MCTS batch identity | Medium | Medium | Stale batch misuse tests; evaluate generation IDs. |
| H6 | Hypothesis | Finite eval grid tactical blind spot | High/Medium | High | Far-threat fixtures against full scanner. |
| H7 | Hypothesis | Oracle shared dependencies | Medium | High | Add radius-3 independent tactical scanner for 4+ windows, not using candidates/hot windows. |
| H8 | Hypothesis | Multiple winning turns collapsed | Medium | High | Decide complete-vs-sufficient tactical API and test accordingly. |
| H9 | Hypothesis | Search caps omit mandatory blocks | High | Medium | Radius-3 oracle-generated mandatory win/block tests for 4+ windows. |
| H10 | Hypothesis | Release-silent eval/hot drift | Medium | Medium | Release differential fuzzer/recompute tests. |
| Q1 | Question | Internal MCTS threat constraints | Medium | Low-Medium | Compare root and internal leaf legal rows in forced-threat positions. |
| Q2 | Question | Legal row ordering contract | Low-Medium | Medium | Repeated-process/order tests; document/sort. |

## Priority Recommendation

Phase 2 should start with D1, D2, D3, D4, H2, H6, and H9. These have the clearest path to either a confirmed defect or a high-confidence safety gate.
