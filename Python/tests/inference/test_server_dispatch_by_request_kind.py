import numpy as np

from hexorl.inference.protocol import InferenceRequestKind, REQUEST_KIND_TO_CODE
from hexorl.inference.server import InferenceServer


class _Queue:
    def __init__(self, kind_code):
        self.slot = type(
            "Slot",
            (),
            {"req_kind": np.array([kind_code], dtype=np.uint8)},
        )()

    def get_slot(self, _worker_id):
        return self.slot


def test_server_dispatches_graph_path_by_request_kind_not_architecture_string():
    server = object.__new__(InferenceServer)
    server._queue = _Queue(REQUEST_KIND_TO_CODE[InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE])
    assert server._is_graph_request([0]) is True
    server._queue = _Queue(REQUEST_KIND_TO_CODE[InferenceRequestKind.DENSE_POLICY_VALUE])
    assert server._is_graph_request([0]) is False
