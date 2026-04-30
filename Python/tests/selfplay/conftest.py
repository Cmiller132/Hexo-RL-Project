from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hexorl.contracts.legal import LegalActionTable
from hexorl.search.context import SearchContext
from hexorl.search.pair_strategy import PairStrategySpec, create_pair_strategy
from hexorl.search.priors import PRIOR_SOURCE_DENSE, SearchEvaluation
from hexorl.selfplay.game_runner import (
    GameRunner,
    GameRunnerConfig,
    RuntimeResourceSpec,
    SelfPlayContractBuilders,
)
from hexorl.selfplay.record_writer import InMemorySelfPlayRecordWriter
from hexorl.selfplay.telemetry import InMemorySelfPlayTelemetrySink


class FakePolicyProvider:
    name = "FakePolicyProvider"

    def __init__(self, model_family: str = "dense_cnn"):
        self.model_family = model_family

    def evaluate_root(self, context: SearchContext) -> SearchEvaluation:
        return self._eval(context)

    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]:
        return [self._eval(context) for context in contexts]

    def _eval(self, context: SearchContext) -> SearchEvaluation:
        width = int(context.legal_table.rows.shape[0])
        return SearchEvaluation(
            context=context,
            value=0.25,
            legal_row_ids=np.arange(width, dtype=np.int64),
            legal_dense_indices=context.legal_table.dense_indices,
            row_priors=np.ones(width, dtype=np.float32),
            prior_source=np.full(width, PRIOR_SOURCE_DENSE, dtype=np.uint8),
            policy_provider=self.name,
            model_family=self.model_family,
            model_spec_version="v2",
            inference_protocol="fake",
        )


class FakeEngineAdapter:
    def __init__(self):
        self.is_over = False
        self.winner = 0
        self._game = SimpleNamespace(current_player=0, placements_remaining=1)
        self._legal = np.asarray([[0, 0]], dtype=np.int32)

    def init_root(self):
        tensor = np.zeros((13, 33, 33), dtype=np.float32)
        return tensor, -16, -16, self._legal.tobytes(order="C"), 1

    def expand_root(self, *args):
        return None

    def expand_root_with_sparse_priors(self, *args):
        return None

    def expand_root_with_global_priors(self, *args):
        return None

    def done(self):
        return True

    def get_results(self):
        return [0], [0], [3], 0.5

    def root_pair_visit_targets(self):
        return []

    def prior_source_summary(self):
        return {"root_total_count": 2, "root_dense_count": 2, "leaf_total_count": 0}

    def root_child_prior_sources(self):
        return [PRIOR_SOURCE_DENSE, PRIOR_SOURCE_DENSE]

    def root_child_q_values(self):
        return [0.5, 0.1]

    def sample_action(self, temperature):
        return 0, 0

    def re_root(self, q, r, sims):
        self.is_over = True
        self._game.current_player = 1
        return True


class FakeGraphBatch:
    token_qr = np.zeros((2, 2), dtype=np.int32)
    edge_index = np.zeros((2, 1), dtype=np.int64)
    graph_semantic_hash = "fake-graph"


@pytest.fixture
def runner_factory():
    def _make(*, model_family: str = "dense_cnn", is_global_graph: bool = False):
        telemetry = InMemorySelfPlayTelemetrySink()
        writer = InMemorySelfPlayRecordWriter(telemetry_sink=telemetry)
        config = GameRunnerConfig(
            worker_id=0,
            run_seed=7,
            num_simulations=1,
            max_game_moves=4,
            batch_size=1,
            c_puct=1.0,
            c_puct_init=1.0,
            near_radius=2,
            constrain_threats=False,
            temperature_schedule=((0.0, 0.0),),
            pcr_low_sim_prob=0.0,
            pcr_low_sims=1,
            policy_target_top_k=4,
            dirichlet_alpha=0.0,
            dirichlet_fraction=0.0,
            sparse_prior_stage=0,
            sparse_prior_mix=0.0,
            sparse_policy_enabled=False,
            candidate_budget=4,
            subtree_reuse=False,
            pair_strategy_name="none",
            pair_strategy_max_pairs=0,
        )
        runtime = RuntimeResourceSpec(
            worker_processes=1,
            inference_queue_capacity=1,
            record_queue_capacity=2,
            leaf_batch_size=1,
            max_in_flight_requests_per_worker=1,
            rust_threads=1,
            torch_threads=1,
            shutdown_timeout_s=1.0,
        )
        model_spec = SimpleNamespace(kind=model_family, version="v2", is_global_graph=is_global_graph)
        builders = SelfPlayContractBuilders(graph_batch_builder=lambda *args, **kwargs: FakeGraphBatch())
        runner = GameRunner(
            policy_provider=FakePolicyProvider(model_family),
            pair_strategy=create_pair_strategy(PairStrategySpec()),
            engine_adapter_factory=lambda **kwargs: FakeEngineAdapter(),
            record_writer=writer,
            telemetry_sink=telemetry,
            contract_builders=builders,
            runtime_spec=runtime,
            runner_config=config,
            model_spec=model_spec,
        )
        return runner, telemetry, writer

    return _make


@pytest.fixture
def legal_context():
    legal = LegalActionTable.from_rows(
        [(0, 0), (1, 0)],
        source="fixture",
        allow_fixture=True,
        history_hash="h",
        placements_remaining=1,
    )
    return SearchContext.create(phase="root", legal_table=legal, model_family="dense_cnn")
