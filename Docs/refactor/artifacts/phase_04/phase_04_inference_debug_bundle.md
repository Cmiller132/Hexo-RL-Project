# Phase 04 Inference Debug Bundle

Debuggable identities now available per request/response:

- manifest hash
- request id
- trace id
- request kind
- position/history/legal/pair hashes
- slot generation
- deadline
- response generation
- transport state
- wait time
- queue depth
- adapter name

Primary code paths:

- `Python/src/hexorl/inference/protocol.py`
- `Python/src/hexorl/inference/client/transport.py`
- `Python/src/hexorl/inference/server/scheduler.py`
- `Python/src/hexorl/inference/telemetry.py`
- `Python/src/hexorl/inference/adapters/`

No raw public client method owns private shared-memory lifecycle. Runtime callers in `Python/src/hexorl/selfplay/worker.py` consume typed request methods.
