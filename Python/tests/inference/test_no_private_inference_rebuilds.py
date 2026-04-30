from pathlib import Path


INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "hexorl" / "inference"


def test_no_mode_specific_inference_submit_methods_remain():
    assert not (INFERENCE_ROOT / "client.py").exists()
    client_text = (INFERENCE_ROOT / "client" / "api.py").read_text()
    banned = ("def submit_sparse", "def submit_sparse_pair", "def submit_graph", "def submit_regret_rank")
    assert [name for name in banned if name in client_text] == []


def test_deleted_flat_inference_modules_do_not_return():
    deleted = ("client.py", "server.py", "batching.py", "shm_transport.py")
    assert [path for path in deleted if (INFERENCE_ROOT / path).exists()] == []


def test_runtime_imports_use_client_server_packages():
    root = Path(__file__).resolve().parents[2] / "src" / "hexorl"
    banned = (
        "hexorl.inference.batching",
        "hexorl.inference.shm_transport",
    )
    hits = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        for needle in banned:
            if needle in text:
                hits.append((str(path.relative_to(root)), needle))
    assert hits == []
