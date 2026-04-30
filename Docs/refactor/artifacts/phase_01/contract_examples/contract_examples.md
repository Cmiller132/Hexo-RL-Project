# Phase 01 Contract Examples

## MoveHistory

```python
from hexorl.contracts.history import MoveHistory, encode_move_history

payload = encode_move_history([(0, 0, 0), (1, 1, 0)])
history = MoveHistory.decode(payload, source="rust")
assert history.history_hash == MoveHistory.decode(history.encode(), source="rust").history_hash
```

Malformed byte payloads, duplicate cells, invalid player order, invalid radius, fixture source without opt-in, and `source="fallback"` are rejected in `Python/tests/contracts/test_phase01_contract_semantics.py`.

## LegalActionTable

```python
from hexorl.engine.legal import LegalTableProvider

table = LegalTableProvider(near_radius=8, constrain_threats=False).from_history(payload)
assert table.source == "rust:legal"
assert not table.rows.flags.writeable
assert table.debug_payload()["table_hash"] == table.table_hash
```

Legal rows are decoded by `hexorl.engine.legal.decode_legal_bytes`, then validated into `LegalActionTable`. Direct runtime legal-byte parsing was removed from sampler, dashboard model cache, eval players, and self-play worker paths.

## D6

```python
from hexorl.contracts.symmetry import transform_history, inverse_symmetry

rotated = transform_history(payload, 1)
restored = transform_history(rotated, inverse_symmetry(1))
assert restored == payload
```

D6 composition, inverse, dense policy mass preservation, pair target mass preservation, legal table transformation, and mutation safety are covered by the Phase 01 contract tests.
