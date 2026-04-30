# Fixture And Artifact Plan

- Eval uses registry enumeration over every registered `ModelFamily` with a unit `PolicyProvider`.
- Dashboard uses compact move-history fixtures and required-view route assertions.
- Dashboard parity fixture compares `model-input` tensor hash to training-input hash for the same golden history.
- Autotune dry-run fixtures include one valid typed recipe per family plus rejected invalid pair-strategy caps.
- Runtime sweep fixture simulates no-progress outcomes for self-play, inference, training, evaluation, and artifact writing.
