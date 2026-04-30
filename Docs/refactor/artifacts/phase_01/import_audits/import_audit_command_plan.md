# Phase 01 Import Audit Command Plan

- Created: `2026-04-30T03:35:54Z`
- Git SHA: `4055227a880a3f6995bc2d18e30f61a11b4a7ef4`
- Scope: audit plan only. This file defines commands and expected classification; it does not claim the audits are clean.

## Search Tool Rule

`rg` was attempted during setup and failed with access denied. Use `git grep` as the primary audit tool in this workspace. Use the listed PowerShell fallbacks only if `git grep` cannot express the needed filter.

## Direct `_engine` Imports

Purpose: prove runtime direct `_engine` imports are absent outside `Python/src/hexorl/engine/`, tests, and explicit fixture tooling.

```powershell
git grep -n "_engine" -- Python/src/hexorl Python/tests crates Docs/RUST_API.md Docs/refactor/rust_review
```

Classification:

- Allowed after implementation: `Python/src/hexorl/engine/rust.py`, tests, explicit fixture builders.
- Runtime findings outside `Python/src/hexorl/engine/` must be deleted, routed through `engine/`, or quarantined as fixture-only.
- Docs references are informational and do not satisfy deletion proof.

## Private Compact-History Parsers

Purpose: find private compact-history decode helpers and row parsers that must move behind `MoveHistory` and `engine/history.py`.

```powershell
git grep -n -E "decode_compact|encode_compact|compact_history|from_compact|history_bytes|compact record|compact_record" -- Python/src/hexorl Python/tests crates
git grep -n -E "struct.unpack|from_le_bytes|np.frombuffer|int.from_bytes" -- Python/src/hexorl crates
```

Classification:

- Allowed after implementation: centralized protocol owner in `crates/hexgame-py/src/protocol.rs`, `engine/history.py`, `contracts/history.py`, tests, explicit fixture tooling.
- Runtime helper clones in graph, sampler, dashboard, tactical oracle, RGSC, epoch bootstrap, replay, or self-play must be deleted or made fixture-only with no runtime imports.

## Private Legal Helpers And Python Legal Fallbacks

Purpose: find private legal row reconstruction, legal bytes builders, fallback legal tables, and Python legal generation in runtime paths.

```powershell
git grep -n -E "legal_moves_for_stones|legal_rows|legal_bytes|build_legal|make_legal|python legal|fallback legal|legal fallback|valid_moves|legal_table" -- Python/src/hexorl Python/tests crates
git grep -n -E "source=`"fixture`"|source='fixture'|fixture legal|fixture-only" -- Python/src/hexorl Python/tests Docs/refactor/artifacts/phase_01
```

Classification:

- Allowed after implementation: Rust-backed `engine/legal.py`, `contracts/legal.py`, tests, explicit fixture tooling using `source="fixture"`.
- Production Python legal fallback and private legal row reconstruction must be deleted or isolated from runtime imports.

## Private D6 Helpers

Purpose: find D6 transform helpers outside `contracts/symmetry.py`.

```powershell
git grep -n -E "d6|symmetry|transform_qr|transform_history|transform_legal|transform_policy|rotate|reflect|axis_map|axis_label" -- Python/src/hexorl Python/tests crates
```

Classification:

- Allowed after implementation: `contracts/symmetry.py`, tests, Rust parity surfaces, and call sites that consume the contract API.
- Runtime helper clones must be deleted or routed through `contracts/symmetry.py`.

## Forbidden Contracts Imports

Purpose: prove `contracts/` remains pure and does not import runtime subsystems.

```powershell
git grep -n -E "hexorl\\.model|hexorl\\.models|hexorl\\.inference|hexorl\\.search|hexorl\\.train|hexorl\\.dashboard|hexorl\\.tuning|hexorl\\.selfplay|import _engine" -- Python/src/hexorl/contracts
```

PowerShell fallback:

```powershell
Get-ChildItem -Path Python/src/hexorl/contracts -Recurse -File |
  Select-String -Pattern "hexorl\.model|hexorl\.models|hexorl\.inference|hexorl\.search|hexorl\.train|hexorl\.dashboard|hexorl\.tuning|hexorl\.selfplay|import _engine"
```

Expected result after implementation: no matches.

## Fixture-Only Fallbacks

Purpose: prove fixture/test fallbacks are explicit and unavailable to production runtime imports.

```powershell
git grep -n -E "fixture|fixture-only|fixture_only|allow_fixture|test mode|test_mode|fallback" -- Python/src/hexorl Python/tests
```

Classification:

- Allowed after implementation: test modules, explicit fixture builders, and source-guard code requiring fixture/test opt-in.
- Runtime imports of fixture-only providers must fail the audit.

## `source="fallback"` And Degraded Sources

Purpose: prove fallback sources cannot enter production runtime accidentally.

```powershell
git grep -n -E "source=`"fallback`"|source='fallback'|python_fallback|fallback_source|degraded" -- Python/src/hexorl Python/tests crates Docs/refactor
```

Expected result after implementation:

- `source="fallback"` absent from production runtime.
- Negative tests may contain fallback labels only as rejection fixtures.
- Any degraded source must be telemetry-visible and test-visible, not a silent runtime path.

## Duplicate Byte Parsers

Purpose: prove no Python or Rust runtime path duplicates FFI byte decoding already owned by `crates/hexgame-py/src/protocol.rs`.

```powershell
git grep -n -E "row_width|from_le_bytes|to_le_bytes|struct.unpack|struct.pack|np.frombuffer|memoryview|legal_bytes|history_bytes|pair_bytes|compact bytes|compact_bytes" -- Python/src/hexorl crates
```

Classification:

- Allowed after implementation: `crates/hexgame-py/src/protocol.rs`, thin engine calls that delegate to protocol-owned Rust/PyO3 APIs, tests, and explicit malformed-byte fixtures.
- Runtime duplicate decoders outside protocol ownership must be deleted or replaced with protocol calls.

## Tactical Compatibility Names

Purpose: prove new runtime code does not recreate `ThreatStatus` as a Python compatibility model.

```powershell
git grep -n -E "ThreatStatus|threat_status" -- Python/src/hexorl Python/tests crates Docs/refactor
```

Classification:

- Allowed after implementation: Rust internal implementation references, historical docs, tests that assert absence/rejection, and migration notes.
- Python runtime compatibility models named `ThreatStatus` or collapsed tactical status helpers are not allowed.

## Audit Output Requirements

For Phase 01 closure, each audit must be captured as a transcript under `Docs/refactor/artifacts/phase_01/import_audits/` with:

- command line
- exit status
- timestamp
- current Git SHA
- raw matches or explicit no-match result
- classification summary
- deletion/quarantine proof for each formerly runtime-owned helper

## Non-Claim Statement

This file is a command plan and classification guide. It does not claim that the current repository passes any audit.
