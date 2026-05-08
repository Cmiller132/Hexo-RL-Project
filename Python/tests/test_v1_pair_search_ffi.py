import numpy as np
import pytest

_engine = pytest.importorskip("_engine")


def _root_rows(root):
    return np.frombuffer(root["legal_row_table"]["rows_bytes"], dtype=np.int32).reshape(-1, 3)


def _request_rows(request):
    return np.frombuffer(request["legal_row_table"]["rows_bytes"], dtype=np.int32).reshape(-1, 3)


def _exact_modes(count):
    return np.full(count, _engine.V1_CORRECTION_EXACT_IMPORTANCE, dtype=np.uint8)


def _complete_request(engine, request, value):
    rows = _request_rows(request)
    pair_qr = np.array([[*rows[0, 1:].tolist(), *rows[1, 1:].tolist()]], dtype=np.int32)
    return engine.complete_expansion(
        request["node_key"],
        value,
        pair_qr,
        np.array([0.0], dtype=np.float32),
        np.ones(1, dtype=np.float32),
        _exact_modes(1),
    )


def test_v1_pair_search_selects_and_applies_canonical_pair():
    game = _engine.PyHexGame()
    game.place(0, 0)
    engine = _engine.PyV1PairSearchEngine(
        game,
        num_simulations=2,
        seed=7,
        max_root_admitted=4,
    )
    root = engine.init_root_v1()
    rows = _root_rows(root)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()
    c = rows[2, 1:].tolist()
    d = rows[3, 1:].tolist()

    pair_qr = np.array([[a[0], a[1], b[0], b[1]], [c[0], c[1], d[0], d[1]]], dtype=np.int32)
    engine.admit_root_pairs(
        pair_qr,
        np.array([0.0, 8.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        _exact_modes(2),
        root["root_generation"],
    )
    requests = engine.run_search_step(8)
    assert len(requests) == 2
    winning_key = next(
        candidate["pair_key"]
        for candidate in engine.replay_telemetry()["candidate_pairs"]
        if candidate["first_legal_row_id"] == 0 and candidate["second_legal_row_id"] == 1
    )
    for request in requests:
        _complete_request(engine, request, 1.0 if request["node_key"] == winning_key else -1.0)
    selected = engine.select_root_action()

    assert selected["action_kind"] == "pair"
    assert selected["first_legal_row_id"] == 0
    assert selected["second_legal_row_id"] == 1
    assert selected["first"] == tuple(a)
    assert selected["second"] == tuple(b)

    applied = engine.apply_selected_action(
        selected["root_generation"],
        selected["legal_row_table_hash"],
        selected["pair_key"],
    )
    assert applied["placements_applied"] == 2
    assert engine.move_count == 3
    assert engine.current_player == 0

    telemetry = engine.replay_telemetry()
    assert telemetry["candidate_selector_version"] == "rust_v1_pair_search_foundation"
    assert telemetry["search_performed"] is True
    assert telemetry["selected_pair_key"] == selected["pair_key"]
    assert telemetry["admitted_pair_count"] == 2
    assert sum(telemetry["root_simulation_allocation"]) == telemetry["simulation_count"]
    assert sum(telemetry["visit_counts"]) == telemetry["simulation_count"]
    assert telemetry["neural_calls_per_expanded_full_turn_node"] == 1
    assert telemetry["reservoir_refill_events"] == 0
    assert telemetry["interior_expanded_full_turn_nodes"] == 2
    assert telemetry["interior_reservoir_build_count"] == 2


def test_v1_pair_search_exceptions_and_identity_rejection():
    opening = _engine.PyV1PairSearchEngine(_engine.PyHexGame(), num_simulations=0)
    root = opening.init_root_v1()
    selected = opening.run_root_search()
    assert root["phase"] == "opening_single"
    assert selected["action_kind"] == "single"
    assert selected["reason"] == "opening_center"
    opening.apply_selected_action(
        selected["root_generation"],
        selected["legal_row_table_hash"],
    )
    assert opening.move_count == 1

    game = _engine.PyHexGame()
    game.place(0, 0)
    engine = _engine.PyV1PairSearchEngine(game, num_simulations=4)
    root = engine.init_root_v1()
    rows = _root_rows(root)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()
    pair_qr = np.array([[a[0], a[1], b[0], b[1]]], dtype=np.int32)

    with pytest.raises(ValueError, match="root token mismatch"):
        engine.admit_root_pairs(
            pair_qr,
            np.array([1.0], dtype=np.float32),
            np.ones(1, dtype=np.float32),
            _exact_modes(1),
            root["root_generation"] + 1,
        )

    engine.admit_root_pairs(
        pair_qr,
        np.array([1.0], dtype=np.float32),
        np.ones(1, dtype=np.float32),
        _exact_modes(1),
        root["root_generation"],
    )
    with pytest.raises(ValueError, match="run_search_step/complete_expansion"):
        engine.run_root_search()
    requests = engine.run_search_step(1)
    _complete_request(engine, requests[0], 1.0)
    selected = engine.select_root_action()
    with pytest.raises(ValueError, match="pair_key mismatch"):
        engine.apply_selected_action(
            selected["root_generation"],
            selected["legal_row_table_hash"],
            selected["pair_key"] + 1,
        )


def test_v1_pair_search_expansion_lifecycle_and_stale_child_identity_rejection():
    game = _engine.PyHexGame()
    game.place(0, 0)
    engine = _engine.PyV1PairSearchEngine(game, num_simulations=4)
    root = engine.init_root_v1()
    rows = _root_rows(root)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()
    pair_qr = np.array([[a[0], a[1], b[0], b[1]]], dtype=np.int32)

    engine.admit_root_pairs(
        pair_qr,
        np.array([1.0], dtype=np.float32),
        np.ones(1, dtype=np.float32),
        _exact_modes(1),
        root["root_generation"],
    )
    requests = engine.run_search_step(1)
    assert len(requests) == 1
    assert requests[0]["phase"] == "normal_two_placement"
    assert len(requests[0]["move_history_bytes"]) > 0
    assert requests[0]["terminal_tactical"]["schema_version"] == 2

    with pytest.raises(ValueError, match="illegal"):
        engine.complete_expansion(
            requests[0]["node_key"],
            0.0,
            pair_qr,
            np.array([0.0], dtype=np.float32),
            np.ones(1, dtype=np.float32),
            _exact_modes(1),
        )


def test_v1_pair_search_recurses_to_deeper_expansion_before_selection():
    game = _engine.PyHexGame()
    game.place(0, 0)
    engine = _engine.PyV1PairSearchEngine(game, num_simulations=2)
    root = engine.init_root_v1()
    rows = _root_rows(root)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()
    pair_qr = np.array([[a[0], a[1], b[0], b[1]]], dtype=np.int32)

    engine.admit_root_pairs(
        pair_qr,
        np.array([1.0], dtype=np.float32),
        np.ones(1, dtype=np.float32),
        _exact_modes(1),
        root["root_generation"],
    )
    first = engine.run_search_step(1)
    assert len(first) == 1
    _complete_request(engine, first[0], 0.25)
    second = engine.run_search_step(1)
    assert len(second) == 1
    assert second[0]["node_key"] != first[0]["node_key"]
    assert second[0]["root_generation"] == root["root_generation"]
    assert second[0]["legal_row_table_hash"] == second[0]["legal_row_table"]["hash"]
    _complete_request(engine, second[0], 0.5)

    selected = engine.select_root_action()
    assert selected["action_kind"] == "pair"
    telemetry = engine.replay_telemetry()
    assert telemetry["simulation_count"] == 2
    assert telemetry["interior_expanded_full_turn_nodes"] == 2


def test_v1_interior_reservoir_is_cached_and_widened_once():
    game = _engine.PyHexGame()
    game.place(0, 0)
    engine = _engine.PyV1PairSearchEngine(game, num_simulations=4, c_pw=2.0, alpha_pw=0.5)
    root = engine.init_root_v1()
    rows = _root_rows(root)
    pairs = np.array(
        [
            [*rows[0, 1:].tolist(), *rows[1, 1:].tolist()],
            [*rows[2, 1:].tolist(), *rows[3, 1:].tolist()],
        ],
        dtype=np.int32,
    )

    telemetry = engine.cache_interior_reservoir(
        99,
        pairs,
        np.array([2.0, 0.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        _exact_modes(2),
    )
    assert telemetry["reservoir_build_count"] == 1
    assert telemetry["scoring_pass_count"] == 1

    widened = engine.widen_interior_reservoir(99, 4)
    assert widened["telemetry"]["reservoir_build_count"] == 1
    assert widened["telemetry"]["scoring_pass_count"] == 1
    assert len(widened["revealed_rows"]) >= 1
    assert len(widened["puct_scores"]) >= 1

    with pytest.raises(ValueError, match="already has a cached reservoir"):
        engine.cache_interior_reservoir(
            99,
            pairs[:1],
            np.array([1.0], dtype=np.float32),
            np.ones(1, dtype=np.float32),
            _exact_modes(1),
        )
