import multiprocessing as mp
from pathlib import Path

import numpy as np

from hexorl.config import Config
from hexorl.selfplay import worker as worker_module
from hexorl.selfplay.worker import SelfPlayWorker


V1_OUTPUTS = [
    "cell_marginal_logits",
    "pair_completion_logits",
    "pair_proposal_score",
    "pair_joint_logits",
    "value",
    "terminal_tactical_v1",
]


def _v1_cfg() -> Config:
    return Config.model_validate(
        {
            "run": {"seed": 11},
            "model": {
                "architecture": "global_pair_biaffine_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": V1_OUTPUTS,
                "pair_strategy": "sampled_joint_pair_v1",
                "pair_strategy_max_pairs": 4,
                "candidate_budget": 32,
            },
            "selfplay": {
                "mcts_simulations": 4,
                "max_game_moves": 3,
                "pcr_low_sim_prob": 0.0,
                "dirichlet_alpha": 0.0,
                "legal_row_mode": "full_rust_legal",
                "tactical_mode": "proposal_and_label",
                "constrain_threats": False,
            },
            "inference": {"fp16": False},
        }
    )


class FakeGraphClient:
    def __init__(self) -> None:
        self.graph_batches = []
        self.graph_many_batches = []

    def submit_graph(self, graph_batch):
        self.graph_batches.append(graph_batch)
        pair_count = int(np.asarray(graph_batch.pair_first_indices).shape[0])
        legal_count = int(np.asarray(graph_batch.legal_qr).shape[0])
        rank = 4
        legal_projection = np.zeros((legal_count, rank), dtype=np.float32)
        if legal_count:
            legal_projection[:, 0] = np.linspace(0.0, 1.0, legal_count, dtype=np.float32)
            legal_projection[:, 1] = np.linspace(1.0, 0.0, legal_count, dtype=np.float32)
        return {
            "value": np.asarray([0.25], dtype=np.float32),
            "cell_marginal_logits": np.zeros(
                legal_count,
                dtype=np.float32,
            ),
            "legal_proposal_embeddings": legal_projection,
            "legal_completion_query": legal_projection,
            "legal_completion_key": legal_projection,
            "pair_completion_logits": np.zeros(pair_count, dtype=np.float32),
            "pair_proposal_score": np.linspace(0.0, 1.0, pair_count, dtype=np.float32),
            "pair_joint_logits": np.linspace(0.0, 3.0, pair_count, dtype=np.float32),
            "terminal_tactical_v1": np.zeros(8, dtype=np.float32),
            "metadata": {
                "legal_qr": np.asarray(graph_batch.legal_qr, dtype=np.int32),
                "outputs": {
                    "value": {
                        "value_decoder": {"perspective": "current_player"},
                    },
                },
            },
        }

    def submit_graph_many(self, graph_batches):
        batch = list(graph_batches)
        self.graph_many_batches.append(batch)
        return [self.submit_graph(graph_batch) for graph_batch in batch]


class FakeHexGame:
    def __init__(self) -> None:
        self.move_count = 0
        self.placements_remaining = 1
        self.current_player = 0
        self._moves = []

    def legal_moves(self):
        if self.move_count == 0:
            return [(0, 0)]
        return [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)]

    def place(self, q, r):
        self._moves.append((int(self.current_player), int(q), int(r)))
        self.move_count += 1
        if self.move_count == 1:
            self.placements_remaining = 2
            self.current_player = 1
        elif self.placements_remaining > 1:
            self.placements_remaining -= 1
        else:
            self.placements_remaining = 2
            self.current_player = 1 - self.current_player

    def is_over(self):
        return self.move_count >= 3

    def winner(self):
        return 0 if self.is_over() else None


def fake_graph_batch_from_history(
    _history,
    *,
    legal_moves,
    **_kwargs,
):
    legal = np.asarray(legal_moves, dtype=np.int32).reshape(-1, 2)
    n = int(legal.shape[0])
    return worker_module.GraphBatch(
        token_features=np.zeros((n, 12), dtype=np.float32),
        token_type=np.full(n, 3, dtype=np.int16),
        token_qr=legal.copy(),
        token_mask=np.ones(n, dtype=np.bool_),
        legal_token_indices=np.arange(n, dtype=np.int64),
        legal_qr=legal.copy(),
        legal_mask=np.ones(n, dtype=np.bool_),
        pair_token_indices=np.zeros(0, dtype=np.int64),
        pair_first_indices=np.zeros(0, dtype=np.int64),
        pair_second_indices=np.zeros(0, dtype=np.int64),
        relation_bias=np.zeros((0,), dtype=np.float32),
        relation_type=np.zeros((0,), dtype=np.int16),
        policy_target=np.zeros(n, dtype=np.float32),
        opp_legal_qr=np.zeros((0, 2), dtype=np.int32),
        opp_legal_mask=np.zeros(0, dtype=np.bool_),
        opp_policy_target=np.zeros(0, dtype=np.float32),
        pair_first_policy_target=np.zeros(n, dtype=np.float32),
        pair_policy_target=np.zeros(0, dtype=np.float32),
        pair_second_policy_target=np.zeros(0, dtype=np.float32),
        tactical_target=np.zeros(8, dtype=np.float32),
        placements_remaining=2 if n > 1 else 1,
        current_player=0,
    )


def test_v1_proposal_matrix_reader_accepts_legal_by_legal_outputs():
    values = np.arange(9, dtype=np.float32).reshape(3, 3)

    direct = worker_module._v1_pair_matrix_from_proposal_output(
        {"pair_completion_logits": values},
        "pair_completion_logits",
        3,
    )
    batched = worker_module._v1_pair_matrix_from_proposal_output(
        {"pair_proposal_score": values.reshape(1, 3, 3)},
        "pair_proposal_score",
        3,
    )
    row_vector = worker_module._v1_pair_matrix_from_proposal_output(
        {"pair_proposal_score": np.arange(3, dtype=np.float32)},
        "pair_proposal_score",
        3,
    )

    np.testing.assert_array_equal(direct, values)
    np.testing.assert_array_equal(batched, values)
    assert row_vector is None


class FakeV1PairSearchEngine:
    instances = []

    def __init__(
        self,
        game,
        num_simulations,
        c_puct=1.4,
        seed=1,
        max_root_admitted=None,
        min_root_admitted=1,
        prior_temperature=1.0,
        min_log_correction=-4.0,
        max_log_correction=4.0,
        alpha_pw=0.5,
        c_pw=2.0,
    ) -> None:
        self.game = game
        self.num_simulations = int(num_simulations)
        self.seed = int(seed)
        self.max_root_admitted = None if max_root_admitted is None else int(max_root_admitted)
        self.root = None
        self.admitted_pair_qr = np.zeros((0, 4), dtype=np.int32)
        self.pair_logits = np.zeros(0, dtype=np.float32)
        self.correction_weights = np.zeros(0, dtype=np.float32)
        self.correction_modes = np.zeros(0, dtype=np.uint8)
        self.selected = None
        self.expansion_requests_issued = 0
        self.expansion_completed = 0
        FakeV1PairSearchEngine.instances.append(self)

    def init_root_v1(self):
        move_count = int(getattr(self.game, "move_count", 0))
        placements_remaining = int(getattr(self.game, "placements_remaining", 1))
        phase = (
            "opening_single"
            if move_count == 0
            else "normal_two_placement"
            if placements_remaining == 2
            else "single_exception"
        )
        legal = [(int(q), int(r)) for q, r in self.game.legal_moves()]
        rows = [[idx, q, r] for idx, (q, r) in enumerate(legal)]
        table_hash = 100000 + move_count
        self.root = {
            "root_generation": 17 + move_count,
            "legal_pair_count": len(legal) * max(0, len(legal) - 1) // 2,
            "phase": phase,
            "legal_row_table": {
                "schema_version": 7,
                "row_count": len(rows),
                "rows": rows,
                "hash": table_hash,
            },
            "terminal_tactical": {
                "pair_row_schema_version": 11,
                "hot_completion_pairs": [],
                "hot_cover_pairs": [],
                "terminal_equivalent_pairs": [],
            },
        }
        return self.root

    def admit_root_pairs(
        self,
        pair_qr,
        pair_logits,
        correction_weights,
        correction_modes,
        root_generation,
    ):
        assert self.root is not None
        assert int(root_generation) == int(self.root["root_generation"])
        self.admitted_pair_qr = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
        self.pair_logits = np.asarray(pair_logits, dtype=np.float32).reshape(-1)
        self.correction_weights = np.asarray(correction_weights, dtype=np.float32).reshape(-1)
        self.correction_modes = np.asarray(correction_modes, dtype=np.uint8).reshape(-1)
        assert self.admitted_pair_qr.shape[0] == self.pair_logits.shape[0]
        assert self.correction_weights.shape[0] == self.pair_logits.shape[0]
        assert self.correction_modes.shape[0] == self.pair_logits.shape[0]

    def run_root_search(self):
        assert self.root is not None
        table_hash = int(self.root["legal_row_table"]["hash"])
        if self.root["phase"] == "opening_single":
            self.selected = {
                "action_kind": "single",
                "cell": (0, 0),
                "first": (0, 0),
                "root_generation": int(self.root["root_generation"]),
                "legal_row_table_hash": table_hash,
                "reason": "opening_center",
            }
            return self.selected
        selected_idx = int(np.argmax(self.pair_logits))
        row = self.admitted_pair_qr[selected_idx]
        first = (int(row[0]), int(row[1]))
        second = (int(row[2]), int(row[3]))
        self.selected = {
            "action_kind": "pair",
            "first": first,
            "second": second,
            "pair_key": 9000 + selected_idx,
            "root_generation": int(self.root["root_generation"]),
            "legal_row_table_hash": table_hash,
        }
        return self.selected

    def run_search_step(self, max_expansions):
        assert self.root is not None
        if self.root["phase"] == "opening_single":
            return []
        if self.expansion_requests_issued >= 2:
            return []
        legal = [(int(q), int(r)) for q, r in self.game.legal_moves()]
        rows = [[idx, q, r] for idx, (q, r) in enumerate(legal)]
        node_key = 12345 if self.expansion_requests_issued == 0 else 67890
        table_hash = 200000 + self.expansion_requests_issued
        self.expansion_requests_issued += 1
        return [
            {
                "node_key": node_key,
                "move_history_bytes": b"",
                "phase": "normal_two_placement",
                "parent_visits": 1,
                "node_visit_count": 0,
                "root_generation": int(self.root["root_generation"]),
                "legal_row_table_hash": table_hash,
                "legal_row_table": {
                    "schema_version": 7,
                    "row_count": len(rows),
                    "rows": rows,
                    "hash": table_hash,
                },
                "terminal_tactical": {
                    "pair_row_schema_version": 11,
                    "hot_completion_pairs": [],
                    "hot_cover_pairs": [],
                    "terminal_equivalent_pairs": [],
                },
            }
        ][: int(max_expansions)]

    def complete_expansion(
        self,
        node_key,
        value,
        pair_qr,
        pair_logits,
        correction_weights,
        correction_modes,
    ):
        assert int(node_key) in {12345, 67890}
        assert np.asarray(pair_qr).reshape(-1, 4).shape[0] == np.asarray(pair_logits).reshape(-1).shape[0]
        self.expansion_completed += 1
        return {
            "node_key": int(node_key),
            "candidate_count": int(np.asarray(pair_qr).reshape(-1, 4).shape[0]),
            "revealed_count": 1,
            "reservoir_build_count": 1,
            "scoring_pass_count": 1,
            "widening_events": 1,
            "reservoir_refill_events": 0,
        }

    def select_root_action(self):
        assert self.expansion_completed == 2
        return self.run_root_search()

    def apply_selected_action(self, root_generation, legal_row_table_hash, pair_key=None):
        assert self.selected is not None
        assert int(root_generation) == int(self.selected["root_generation"])
        assert int(legal_row_table_hash) == int(self.selected["legal_row_table_hash"])
        if self.selected["action_kind"] == "single":
            return {
                "action_kind": "single",
                "placements_applied": 1,
                "first": self.selected["first"],
            }
        assert int(pair_key) == int(self.selected["pair_key"])
        return {
            "action_kind": "pair",
            "placements_applied": 2,
            "first": self.selected["first"],
            "second": self.selected["second"],
        }

    def replay_telemetry(self):
        if self.root is None or self.root["phase"] == "opening_single":
            return {
                "candidate_selector_version": "fake_v1_search",
                "search_performed": False,
                "hardcoded_action": True,
                "reservoir_build_count": 0,
                "scoring_pass_count": 0,
                "supplied_candidate_count": 0,
                "admitted_pair_count": 0,
                "legal_row_count": 0,
                "reservoir_refill_events": 0,
                "candidate_pairs": [],
                "root_gumbel_values_or_admission_order": [],
                "neural_calls_per_expanded_full_turn_node": 0,
            }
        n = int(self.admitted_pair_qr.shape[0])
        selected_idx = int(np.argmax(self.pair_logits))
        visits = [0 for _ in range(n)]
        allocations = [0 for _ in range(n)]
        visits[selected_idx] = self.num_simulations
        allocations[selected_idx] = self.num_simulations
        rows = []
        for idx in range(n):
            q1, r1, q2, r2 = (int(value) for value in self.admitted_pair_qr[idx])
            q_value = float(self.pair_logits[idx] / 10.0)
            rows.append(
                {
                    "candidate_id": idx,
                    "row_id": idx,
                    "first_legal_row_id": idx,
                    "second_legal_row_id": idx + 1,
                    "first": (q1, r1),
                    "second": (q2, r2),
                    "pair_key": 9000 + idx,
                    "prior_logit": float(self.pair_logits[idx]),
                    "prior": 1.0 / float(max(n, 1)),
                    "gumbel": float(idx) / 100.0,
                    "visit_count": visits[idx],
                    "q_value": q_value,
                    "completed_q": q_value + 0.05,
                    "allocation": allocations[idx],
                    "admitted": True,
                    "forced_exploration_flag": False,
                    "terminal_exact_flag": False,
                    "terminal_equivalence_flag": False,
                    "target_support_flags": ["admitted"],
                    "correction_mode": int(self.correction_modes[idx]),
                }
            )
        return {
            "candidate_selector_version": "fake_v1_search",
            "search_performed": True,
            "hardcoded_action": False,
            "reservoir_build_count": 1,
            "scoring_pass_count": 1,
            "supplied_candidate_count": n,
            "admitted_pair_count": n,
            "legal_row_count": int(self.root["legal_row_table"]["row_count"]),
            "simulation_count": self.num_simulations,
            "reservoir_refill_events": 0,
            "interior_expanded_full_turn_nodes": self.expansion_completed,
            "interior_reservoir_build_count": self.expansion_completed,
            "interior_scoring_pass_count": self.expansion_completed,
            "candidate_pairs": rows,
            "root_gumbel_values_or_admission_order": [[idx, float(idx) / 100.0] for idx in range(n)],
            "root_simulation_allocation": allocations,
            "visit_counts": visits,
            "neural_calls_per_expanded_full_turn_node": 1,
        }


def test_sampled_joint_pair_v1_requires_explicit_strategy_not_v1_heads():
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_pair_biaffine_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": V1_OUTPUTS,
            },
            "inference": {"fp16": False},
        }
    )
    queue = mp.Queue()
    try:
        worker = SelfPlayWorker(0, cfg, queue, num_workers=1, max_batch_size=1)
        assert worker.v1_pair_runtime_enabled is False
        assert worker.pair_policy_enabled is False
        assert worker.pair_strategy == "none"
    finally:
        queue.close()
        queue.join_thread()


def test_sampled_joint_pair_v1_worker_uses_pair_native_runtime(monkeypatch):
    FakeV1PairSearchEngine.instances.clear()
    monkeypatch.setattr(worker_module, "HAS_ENGINE", True)
    monkeypatch.setattr(
        worker_module._engine,
        "PyV1PairSearchEngine",
        FakeV1PairSearchEngine,
        raising=False,
    )
    monkeypatch.setattr(worker_module._engine, "PyHexGame", FakeHexGame, raising=False)
    monkeypatch.setattr(worker_module, "build_graph_batch_from_history", fake_graph_batch_from_history)

    cfg = _v1_cfg()
    record_queue = mp.Queue()
    diagnostic_queue = mp.Queue()
    try:
        worker = SelfPlayWorker(
            0,
            cfg,
            record_queue,
            num_workers=1,
            max_batch_size=1,
            diagnostic_queue=diagnostic_queue,
        )
        client = FakeGraphClient()
        record = worker._play_one_game(client)
    finally:
        record_queue.close()
        record_queue.join_thread()
        diagnostic_queue.close()
        diagnostic_queue.join_thread()

    assert record is not None
    assert record.game_length == 3
    assert len(record.final_move_history) == 3 * 12
    assert len(record.positions) == 2
    assert len(client.graph_batches) == 6
    assert len(FakeV1PairSearchEngine.instances) == 2

    opening = record.positions[0]
    assert opening.policy_target_v2 == [(0, 0, 1.0)]
    assert opening.v1_search_metadata is not None
    assert opening.v1_search_metadata.candidate_pairs == ()

    pair_position = record.positions[1]
    metadata = pair_position.v1_search_metadata
    assert metadata is not None
    assert pair_position.pair_policy_target_v2 == []
    assert pair_position.pair_policy_complete is False
    assert pair_position.policy_target == {}
    assert pair_position.policy_target_v2 == []
    assert metadata.legal_row_schema_version == 7
    assert metadata.pair_row_schema_version == 11
    assert metadata.selected_pair is not None
    assert metadata.selected_pair in {candidate.pair_key for candidate in metadata.candidate_pairs}
    assert all(
        candidate.source_contributions and candidate.proposal_propensity_metadata
        for candidate in metadata.candidate_pairs
    )
    assert metadata.neural_calls_per_expanded_full_turn_node == 6.0
    assert metadata.reservoir_refill_events == ()
    assert metadata.search_surprise_metrics["model_eval_count"] == 6.0
    assert metadata.search_surprise_metrics["reservoir_build_count"] == 1.0
    assert metadata.search_surprise_metrics["bounded_scoring_pass_count"] == 3.0
    assert metadata.search_surprise_metrics["proposal_batch_size"] == 2.0
    assert metadata.search_surprise_metrics["scoring_batch_size"] == 2.0
    assert metadata.search_surprise_metrics["rust_reservoir_build_count"] == 1.0
    assert metadata.search_surprise_metrics["rust_scoring_pass_count"] == 1.0
    assert metadata.search_surprise_metrics["rust_reservoir_refill_events"] == 0.0
    assert metadata.search_surprise_metrics["rust_interior_expanded_full_turn_nodes"] == 2.0
    assert metadata.search_surprise_metrics["interior_widening_events"] >= 2.0
    assert len(client.graph_many_batches) == 4
    assert all(len(batch) == 1 for batch in client.graph_many_batches)

    proposal_batch = client.graph_batches[0]
    assert np.asarray(proposal_batch.pair_first_indices).shape[0] == 0
    assert np.asarray(proposal_batch.pair_token_indices).shape[0] == 0

    graph_batch = client.graph_batches[1]
    assert np.asarray(graph_batch.pair_first_indices).shape[0] == len(metadata.candidate_pairs)
    assert np.asarray(graph_batch.pair_token_indices).shape[0] == len(metadata.candidate_pairs)
    assert np.all(np.asarray(graph_batch.pair_token_indices) == -1)
    graph_legal = {tuple(row) for row in np.asarray(graph_batch.legal_qr, dtype=np.int32).tolist()}
    assert np.asarray(graph_batch.legal_qr).shape[0] <= int(
        metadata.search_surprise_metrics["graph_legal_row_count"]
    )
    assert np.asarray(graph_batch.legal_qr).shape[0] <= int(
        metadata.search_surprise_metrics["selector_legal_row_count"]
    )
    for candidate in metadata.candidate_pairs:
        first, second = candidate.pair_key
        assert tuple(first) in graph_legal
        assert tuple(second) in graph_legal

    interior_proposal_batch = client.graph_batches[2]
    interior_graph_batch = client.graph_batches[3]
    deeper_proposal_batch = client.graph_batches[4]
    deeper_graph_batch = client.graph_batches[5]
    assert np.asarray(interior_proposal_batch.pair_first_indices).shape[0] == 0
    assert np.asarray(interior_graph_batch.pair_first_indices).shape[0] > 0
    assert np.asarray(deeper_proposal_batch.pair_first_indices).shape[0] == 0
    assert np.asarray(deeper_graph_batch.pair_first_indices).shape[0] > 0


def test_sampled_joint_pair_v1_real_engine_smoke_outputs_distinct_history():
    if not worker_module.HAS_ENGINE:
        import pytest

        pytest.skip("Rust _engine extension is unavailable")

    from scripts.run_v1_selfplay_coherence_smoke import run_smoke

    summary = run_smoke(
        target_states=8,
        mcts_simulations=4,
        max_game_moves=8,
        pair_budget=32,
    )

    assert summary["ok"] is True
    assert summary["positions"] >= 8
    assert summary["games"] >= 1
    assert summary["graph_calls"] >= 1
    assert summary["max_graph_pair_count"] <= 32
    first = summary["records"][0]
    assert first["positions"] > 0
    assert first["game_length"] <= 8
    assert len(first["first_moves"]) == len(set(tuple(move) for move in first["first_moves"]))


def test_sampled_joint_pair_v1_source_path_has_no_legacy_projection_authority():
    worker_path = Path(__file__).resolve().parents[1] / "src" / "hexorl" / "selfplay" / "worker.py"
    source = worker_path.read_text(encoding="utf-8")
    start = source.index("    def _v1_build_and_score_root")
    end = source.index("    def _play_one_game(", start)
    v1_segment = source[start:end]
    banned = (
        "pair_logits_to_action_logits",
        "apply_root_pair_priors",
        "apply_root_pair_first_priors",
        "apply_root_pair_second_priors",
        "apply_root_pair_rows",
    )
    for token in banned:
        assert token not in v1_segment
    assert "constrain_threats=False" in v1_segment
    assert "select_pair_candidates_v1" in v1_segment
    assert "graph_batch_with_admitted_pair_rows" in v1_segment
    assert ".admit_root_pairs(" in v1_segment
    assert ".run_search_step(" in v1_segment
    assert ".complete_expansion(" in v1_segment
    assert ".select_root_action(" in v1_segment
    assert ".apply_selected_action(" in v1_segment
