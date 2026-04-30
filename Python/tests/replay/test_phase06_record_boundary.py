from hexorl.selfplay.record_writer import record_hash, validate_game_record
from hexorl.selfplay.records import GameRecord, PositionRecord


def test_phase06_record_boundary_validates_traceable_game_record():
    record = GameRecord(
        positions=[
            PositionRecord(
                move_history=b"",
                policy_target={544: 1.0},
                policy_target_v2=[(0, 0, 1.0)],
                root_value=0.5,
                player=0,
                game_id=10,
            )
        ],
        outcome=1.0,
        game_id=10,
        game_length=1,
        final_move_history=b"\x00" * 12,
        terminal_reason="win",
    )
    record.assign_outcomes()

    validate_game_record(record)
    assert record_hash(record)
