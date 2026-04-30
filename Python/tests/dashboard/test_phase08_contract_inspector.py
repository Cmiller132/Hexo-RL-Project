import base64
import struct

import pytest

from hexorl.dashboard.app import create_app
from hexorl.dashboard.contract_inspector import ContractInspector, required_view_names


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def test_contract_inspector_required_views_and_hash_fields():
    pytest.importorskip("_engine")
    inspector = ContractInspector()
    history = _move(0, 0, 0)
    for view in required_view_names():
        payload = inspector.inspect(view, history=history)
        assert payload["view"] == view
        facts = payload["facts"]
        assert "history_hash" in facts
        assert "trace_id" in facts
        assert "model_family" in facts
        assert "recipe_id" in facts
        assert "inference_protocol_version" in facts


def test_dashboard_routes_use_contract_inspector_and_render_required_views(tmp_path):
    pytest.importorskip("_engine")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing"))
    views = client.get("/api/inspect/views").json()
    assert set(required_view_names()) <= set(views["registered"])
    history_b64 = base64.b64encode(_move(0, 0, 0)).decode("ascii")
    graph = client.post("/api/inspect/graph", json={"history_b64": history_b64}).json()
    assert graph["inspector"]["dispatcher"] == "ContractInspector"
    assert graph["legal_count"] > 0
    d6 = client.post("/api/debug/d6", json={"history_b64": history_b64}).json()
    assert d6["symmetry_count"] == 12
    assert d6["transforms"][0]["contracts"]["sparse_candidates"]["source"]


def test_dashboard_extension_and_mismatch_owner():
    class FakeInspector:
        name = "fake"

        def inspect(self, request, inspector):
            return {"ok": True, "history_seen": bool(request.history)}

    inspector = ContractInspector()
    inspector.register("fake", FakeInspector())
    assert inspector.inspect("fake", history=b"")["ok"] is True
    mismatch = inspector.inspect("mismatch", history=b"", compare_to={"legal_table_hash": "wrong"})
    assert mismatch["likely_owner"] == "engine/legal"


def test_dashboard_import_audit_rejects_private_reconstruction_imports():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hexorl" / "dashboard"
    banned = (
        "CandidateContractBuilder",
        "PairActionTableBuilder",
        "build_graph_batch_from_history",
        "transform_history",
    )
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "contract_inspector.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            if needle in text:
                offenders.append(f"{path.name}:{needle}")
    assert offenders == []
