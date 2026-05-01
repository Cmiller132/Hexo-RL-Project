from hexorl.contracts.pair_strategy import (
    PAIR_STRATEGY_REGISTRY,
    PairStrategyDescriptor,
    PairStrategyRegistry,
)


def test_throwaway_registry_can_define_new_strategy_for_runner_path(runner_factory, fake_pair_scorer):
    fake_registry = PairStrategyRegistry(
        {
            "none": PAIR_STRATEGY_REGISTRY.resolve("none"),
            "experimental_root": PairStrategyDescriptor(
                name="experimental_root",
                aliases=frozenset({"experimental"}),
                generation_mode="capped_fill",
                root_enabled=True,
                leaf_enabled=False,
                diagnostic=False,
                max_pair_rows_field="max_root_pair_rows",
                chunk_cap=4,
                allow_full=False,
                requires_pair_head=True,
            ),
        }
    )
    runner, _telemetry, _writer = runner_factory(
        pair_strategy_name="experimental",
        pair_strategy_max_pairs=1,
        pair_scorer=fake_pair_scorer,
        pair_strategy_registry=fake_registry,
    )

    assert runner._pair_strategy_spec().name == "experimental_root"
    assert runner.pair_strategy == "experimental_root"
