from pathlib import Path


WORKER = Path("Python/src/hexorl/selfplay/worker.py")


def test_worker_contains_no_architecture_or_search_logic():
    source = WORKER.read_text(encoding="utf-8")
    banned = [
        "architecture",
        'startswith("global_',
        "pair_prior_mix",
        "pair_head",
        "Candidate",
        "PairAction",
        "PAIR_ACTION",
        "graph_token",
        "graph_relation",
        "chunk",
        "MCTS",
        "prior",
        "process_game_record",
        "uniform",
        "_engine.MCTSEngine",
        "submit_",
    ]
    for item in banned:
        assert item not in source
