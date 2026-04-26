"""End-to-end smoke test for the inference server + MCTSEngine integration.

These tests spin up a real inference server with a small stub model and
run through the full MCTS cycle: init_root → select_leaves → submit →
expand_and_backprop → get_results.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hexorl.config import load_config
from hexorl.inference.server import InferenceServer
from hexorl.inference.client import InferenceClient
from hexorl.inference.shm_queue import connect_inference_queue

# Try to import the compiled Rust extension.
try:
    import _engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


class TestInferenceServer(unittest.TestCase):
    """Tests that require the inference server but not the Rust engine."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.model.channels = 8
        cls.cfg.model.blocks = 2
        cls.cfg.inference.max_batch_size = 16
        cls.cfg.inference.fp16 = False

    def test_server_starts_and_stops(self):
        """Server starts and stops cleanly."""
        server = InferenceServer(self.cfg, num_workers=1)
        server.start()
        self.assertTrue(server.is_running())
        server.stop()
        server.join(timeout=5.0)
        self.assertFalse(server.is_running())

    def test_single_client_round_trip(self):
        """A single client can submit and receive correct results."""
        server = InferenceServer(self.cfg, num_workers=1)
        server.start()

        client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=16, timeout_ms=10000)
        client.connect()

        # Fix event references (the shm_queue.connect_inference_queue gets the right events)
        server_q = connect_inference_queue(1, 16)
        s0 = server_q.get_slot(0)
        client._slot.req_ready = s0.req_ready
        client._slot.res_ready = s0.res_ready

        count = 4
        tensor = np.random.randn(count, 13, 33, 33).astype(np.float32)
        policies, values = client.submit(tensor, count)

        self.assertEqual(policies.shape, (count * 1089,))
        self.assertEqual(values.shape, (count,))
        self.assertTrue(np.isfinite(policies).all(), "Policies contain NaN/Inf")
        self.assertTrue(np.isfinite(values).all(), "Values contain NaN/Inf")
        self.assertTrue(np.all(values >= -1.0) and np.all(values <= 1.0),
                        "Values outside [-1, 1]")

        client.disconnect()
        server.stop()
        server.join(timeout=5.0)
        server_q.close()

    def test_adaptive_batching_two_clients(self):
        """Two clients submitting simultaneously get correct results."""
        server = InferenceServer(self.cfg, num_workers=2)
        server.start()

        client0 = InferenceClient(worker_id=0, num_workers=2, max_batch_size=16, timeout_ms=10000)
        client1 = InferenceClient(worker_id=1, num_workers=2, max_batch_size=16, timeout_ms=10000)
        client0.connect()
        client1.connect()

        server_q = connect_inference_queue(2, 16)
        s0, s1 = server_q.get_slot(0), server_q.get_slot(1)
        client0._slot.req_ready = s0.req_ready
        client0._slot.res_ready = s0.res_ready
        client1._slot.req_ready = s1.req_ready
        client1._slot.res_ready = s1.res_ready

        for i in range(5):
            t0 = np.random.randn(3, 13, 33, 33).astype(np.float32)
            t1 = np.random.randn(2, 13, 33, 33).astype(np.float32)
            p0, v0 = client0.submit(t0, 3)
            p1, v1 = client1.submit(t1, 2)
            self.assertEqual(p0.shape, (3 * 1089,))
            self.assertEqual(p1.shape, (2 * 1089,))

        client0.disconnect()
        client1.disconnect()
        server.stop()
        server.join(timeout=5.0)
        server_q.close()


@unittest.skipUnless(HAS_ENGINE, "Rust _engine extension not available")
class TestInferenceServerWithEngine(unittest.TestCase):
    """Tests that require both the inference server and the Rust MCTS engine."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.model.channels = 8
        cls.cfg.model.blocks = 2
        cls.cfg.inference.max_batch_size = 16
        cls.cfg.inference.fp16 = False

    def test_mcts_round_trip(self):
        """Full MCTS cycle through the inference server produces valid results."""
        server = InferenceServer(self.cfg, num_workers=1)
        server.start()

        client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=16, timeout_ms=30000)
        client.connect()

        server_q = connect_inference_queue(1, 16)
        s0 = server_q.get_slot(0)
        client._slot.req_ready = s0.req_ready
        client._slot.res_ready = s0.res_ready

        # Create an MCTSEngine
        game = _engine.PyHexGame()
        engine = _engine.PyMCTSEngine(game, num_simulations=100, c_puct=1.5,
                                       near_radius=2, c_puct_init=19652.0,
                                       constrain_threats=False, seed=42)

        # Init root
        init = engine.init_root()
        self.assertIsNotNone(init, "init_root returned None (game over?)")
        tensor_3d, oq, or_, legal_bytes = init
        self.assertEqual(tensor_3d.shape, (13, 33, 33))

        # Expand root with mock uniform policy
        uniform = np.ones(1089, dtype=np.float32) / 1089.0
        engine.expand_root(uniform, 0.0, oq, or_, legal_bytes)

        # MCTS loop with inference server
        while not engine.done():
            tensor_4d, count = engine.select_leaves(8)
            tensor_np = np.array(tensor_4d).astype(np.float32)
            policies, values = client.submit(tensor_np, count)
            engine.expand_and_backprop(policies, values)

        # Get results
        moves_q, moves_r, visits, root_value = engine.get_results()
        self.assertGreater(len(moves_q), 0, "No moves from MCTS")
        self.assertGreaterEqual(max(visits), 1, "No visits recorded")
        self.assertGreaterEqual(root_value, -1.0)
        self.assertLessEqual(root_value, 1.0)

        client.disconnect()
        server.stop()
        server.join(timeout=5.0)
        server_q.close()


if __name__ == "__main__":
    unittest.main()
