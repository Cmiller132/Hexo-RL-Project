# Phase 07 Acceptance Checklist

Assignment

Goal
Replace runtime replay/buffer ownership with canonical replay records, storage, sampler, and projector.

Success criteria
- New self-play writes only `ReplayGameRecord`.
- New replay records validate schema, history hash, Rust legal-row hash, reconstructed legal-row hash, targets, pair metadata, and finite values.
- Sampler reads only `ReplayStorage`.
- Training batches are produced by `replay/projector.py`.
- Self-play, replay, train, and epoch runtime have no `hexorl.buffer` imports.
- Corruption, mutation, D6, projection, sample-to-loss, and performance evidence exists.

Constraints
- No legacy runtime fallback or old-buffer decode in the Phase 07 runtime path.
- Rust remains canonical for history/legal replay, but record decoding revalidates Rust-derived data.
- Hot-path projection keeps batched arrays and cheap identity/hash checks.

Required evidence
- Tests: replay codec/storage/projector/import audit, self-play record writer, production smoke, search guardrails.
- Audits: banned runtime import search, transient MCTS token search, magic-less legacy decode rejection.
- Performance: write/read/project samples/sec and memory high-watermark JSON.

Stop rules
- Stop if old replay decode must remain in self-play/sampler/train/epoch runtime.
- Stop if sampler needs legacy schema fallback.
- Stop if sample-to-loss cannot be produced through `replay/projector.py`.

Checklist
- [x] V2-070 complete.
- [x] V2-071 complete.
- [x] V2-072 complete.
- [x] V2-073 complete.
- [x] V2-074 complete.
- [x] V2-075 complete.
