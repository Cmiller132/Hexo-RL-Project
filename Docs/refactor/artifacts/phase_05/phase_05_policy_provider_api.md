# Phase 05 PolicyProvider API

`Python/src/hexorl/search/policy_provider.py` defines the runtime search policy boundary:

```text
PolicyProvider.evaluate_root(SearchContext) -> SearchEvaluation
PolicyProvider.evaluate_leaves(list[SearchContext]) -> list[SearchEvaluation]
```

Registered providers:

| Provider | Model kinds | Mapping rule |
|---|---|---|
| `DensePolicyProvider` | `dense_cnn` | Dense policy is indexed by `LegalActionTable.dense_indices`. |
| `RestNetPolicyProvider` | `restnet` | Same legal-row mapping as dense, with RestNet family telemetry. |
| `GraphHybridPolicyProvider` | `graph_hybrid` | Sparse candidate logits map through canonical `CandidateTable` rows to `LegalActionTable` rows. |
| `GlobalGraphPolicyProvider` | `global_xattn`, `global_line_window`, `global_relation_graph` | `policy_place` logits map through graph `legal_qr` metadata to the Rust legal row order. |

Hard rules enforced by tests:

- Providers return `SearchEvaluation`, never raw logits.
- Providers do not enable pair scoring.
- Providers do not inspect architecture prefixes, pair head presence, or `pair_prior_mix`.
- Every provider records provider name, model family, protocol, timing, raw output metadata, prior source, and legal row identity.

Primary tests:

- `Python/tests/search/test_policy_provider.py`
- `Python/tests/search/test_global_graph_pair_contracts.py`
