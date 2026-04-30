from pathlib import Path


def test_no_mode_specific_inference_submit_methods_remain():
    client_text = (Path(__file__).resolve().parents[2] / "src" / "hexorl" / "inference" / "client.py").read_text()
    banned = ("def submit_sparse", "def submit_sparse_pair", "def submit_graph", "def submit_regret_rank")
    assert [name for name in banned if name in client_text] == []
