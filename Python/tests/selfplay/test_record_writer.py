import queue

from hexorl.replay.codec import REPLAY_RECORD_SCHEMA_VERSION, ReplayGameRecord
from hexorl.selfplay.record_writer import InMemorySelfPlayRecordWriter, QueueSelfPlayRecordWriter
from hexorl.selfplay.records import GameRecord, PositionRecord
from hexorl.selfplay.telemetry import InMemorySelfPlayTelemetrySink


def _record(game_id: int = 1) -> GameRecord:
    record = GameRecord(
        positions=[
            PositionRecord(
                move_history=b"",
                policy_target={544: 1.0},
                policy_target_v2=[(0, 0, 1.0)],
                root_value=0.5,
                player=0,
                game_id=game_id,
            )
        ],
        outcome=1.0,
        game_id=game_id,
        game_length=1,
        final_move_history=(0).to_bytes(4, "little", signed=True)
        + (0).to_bytes(4, "little", signed=True)
        + (0).to_bytes(4, "little", signed=True),
        terminal_reason="win",
    )
    record.assign_outcomes()
    return record


def test_replay_records_are_written_outside_worker_with_hash_and_schema():
    sink = InMemorySelfPlayTelemetrySink()
    writer = InMemorySelfPlayRecordWriter(telemetry_sink=sink)
    result = writer.write(_record(), run_request=object())

    assert result.ok is True
    assert result.positions_written == 1
    assert isinstance(writer.records[0], ReplayGameRecord)
    assert result.record_hash == writer.records[0].game_hash
    assert result.schema_version == REPLAY_RECORD_SCHEMA_VERSION


def test_record_writer_backpressure_is_structured():
    sink = InMemorySelfPlayTelemetrySink()
    out = queue.Queue(maxsize=1)
    out.put(_record(9))
    writer = QueueSelfPlayRecordWriter(
        out,
        lookahead_horizons=[],
        lookahead_lambdas=[],
        telemetry_sink=sink,
        max_backpressure_events=1,
        put_timeout_s=0.001,
    )
    result = writer.write(_record(2), run_request=object())

    assert result.ok is False
    assert any(event["event"] == "selfplay_backpressure" for event in sink.events)
    assert any(event["event"] == "contract_validation_failure" for event in sink.events)
