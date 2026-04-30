# Interface Freeze Notes

- Eval frozen interface: `PolicyPlayer(provider, model_spec, ...)` and `NoisyModelPlayer` compatibility constructor; both route through `hexorl.search.policy_provider.PolicyProvider`.
- Dashboard frozen interface: `ContractInspector.inspect(view, **kwargs)` with registerable `InspectorService` extensions and required views from `required_view_names()`.
- Autotune frozen interfaces: `ModelRecipe`, `RecipeTransform`, `FamilySpace`, `RuntimeSpec`, `WatchdogSpec`, `ScoreComponents`, `TrialManifest`, `AutotuneScheduler`, and reporting helpers.
- Forbidden Phase 08 interfaces: direct eval model-class dispatch, dashboard private candidate/pair/graph/D6 reconstruction outside `contract_inspector.py`, old ASHA/BOHB/PB2 raw-config modules, and Phase 3 autotune scripts.
