import pytest

from hexorl.contracts.validation import ContractValidationError
from hexorl.selfplay.telemetry import (
    ContractTrace,
    InMemorySelfPlayTelemetrySink,
    SelfPlayDebugBundle,
    SelfPlayMutationGuard,
    heartbeat_payload,
    no_progress_payload,
)


def test_contract_trace_contains_required_spans(legal_context):
    trace = ContractTrace.from_context(legal_context, timings_ms={"model_forward_ms": 1.25})

    payload = trace.to_event_payload()
    assert payload["legal_count"] == 2
    assert "history_parse_ms" in payload["timings_ms"]
    assert payload["timings_ms"]["model_forward_ms"] == pytest.approx(1.25)


def test_heartbeat_no_progress_and_required_event_types_emit():
    sink = InMemorySelfPlayTelemetrySink()
    sink.emit(
        "selfplay_worker_heartbeat",
        heartbeat_payload(
            worker_id=0,
            process_id=123,
            run_id="r",
            game_id=1,
            phase="waiting_ipc",
            move_index=0,
            positions_completed=0,
            model_family="dense_cnn",
            recipe_id="default",
            policy_provider="fake",
            pair_strategy="none",
        ),
    )
    sink.emit(
        "selfplay_no_progress",
        no_progress_payload(phase="record_writer_wait", elapsed_ms=10.0, last_completed_position=1),
    )

    assert sink.events[0]["last_engine_operation"] == ""
    assert sink.events[1]["suggested_next_subsystem"] == "engine replay/legal"


def test_debug_bundle_sections_and_mutation_guard_owner():
    sections = {
        "engine": {"legal_hash": "l"},
        "contract": {"trace_id": "t"},
        "d6": {"transform": "identity"},
        "model_input": {"shape": [1, 13, 33, 33]},
        "raw_output": {"finite": True},
        "policy": {"rows": 2},
        "pair": {"rows": 0},
        "mcts": {"selected": [0, 0]},
        "replay": {"schema_version": 9},
    }
    bundle = SelfPlayDebugBundle(
        owner_subsystem="policy provider row mapping",
        run_id="r",
        game_id=1,
        move_index=0,
        seed=7,
        phase="root",
        sections=sections,
    )
    assert bundle.to_event_payload()["owner_subsystem"] == "policy provider row mapping"

    guard = SelfPlayMutationGuard("contract validation", "legal_table", "a")
    with pytest.raises(ContractValidationError, match="mutated after validation"):
        guard.check("b")
