# Replay Contract Examples

Create and encode a canonical replay game:

```python
from hexorl.replay.fixtures import golden_replay_game
from hexorl.replay.codec import encode_replay_game, decode_replay_game

record = golden_replay_game()
payload = encode_replay_game(record)
decoded = decode_replay_game(payload)
assert decoded.game_hash == record.game_hash
```

Project samples to trainable batch:

```python
from hexorl.replay.storage import ReplayStorage
from hexorl.replay.sampler import ReplayDataset

storage = ReplayStorage(capacity=16)
storage.append_game(record)
batch = next(iter(ReplayDataset(storage, batch_size=2, include_sparse_policy=True)))
assert batch.source == "replay/projector.py"
```

Failure behavior:

```python
from hexorl.replay.codec import ReplayCodecError, decode_replay_game

try:
    decode_replay_game(b"legacy")
except ReplayCodecError as exc:
    assert "replay.codec" in str(exc)
```
