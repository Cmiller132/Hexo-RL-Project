import unittest

import numpy as np
import torch

from hexorl.config import load_config
from hexorl.inference.client import InferenceClient
from hexorl.inference.server import InferenceServer
from hexorl.inference.server.outputs import bounded_policy_logits, bounded_value_logits
from hexorl.models.inference_contracts import OP_PLACE_VALUE


class TestInferenceServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.model.channels = 8
        cls.cfg.model.blocks = 1
        cls.cfg.model.heads = ["policy", "value"]
        cls.cfg.inference.max_batch_size = 4
        cls.cfg.inference.fp16 = False

    def test_server_starts_and_stops(self):
        server = InferenceServer(self.cfg, num_workers=1)
        server.start()
        self.assertTrue(server.is_running())
        server.stop()
        server.join(timeout=5.0)
        self.assertFalse(server.is_running())

    def test_single_client_generic_operation_round_trip(self):
        server = InferenceServer(self.cfg, num_workers=1)
        server.start()
        client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=4, timeout_ms=10000)
        try:
            client.connect()
            tensor = np.random.randn(2, 13, 33, 33).astype(np.float32)
            response = client.evaluate(OP_PLACE_VALUE, {"tensor": tensor})
            policies = response.head_outputs["policy"]
            values = response.head_outputs["value"]
            self.assertEqual(policies.shape, (2, 1089))
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.isfinite(policies).all())
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(response.telemetry["operation_name"], OP_PLACE_VALUE)
        finally:
            client.disconnect()
            server.stop()
            server.join(timeout=5.0)

    def test_non_finite_outputs_are_rejected_before_mcts(self):
        policy = torch.tensor([[float("nan"), float("inf"), -float("inf"), 5.0]])
        value = torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.0]])
        with self.assertRaises(RuntimeError):
            bounded_policy_logits(policy, head_name="policy")
        with self.assertRaises(RuntimeError):
            bounded_value_logits(value, head_name="value")


if __name__ == "__main__":
    unittest.main()
