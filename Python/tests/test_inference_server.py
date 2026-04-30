"""End-to-end smoke test for the inference server + MCTSEngine integration.

These tests spin up a real inference server with a small stub model and
run through the full MCTS cycle: init_root → select_leaves → submit →
expand_and_backprop → get_results.
"""

import sys
import os
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hexorl.config import load_config
from hexorl.inference.server import InferenceServer
from hexorl.inference.client import InferenceClient
from hexorl.inference.shm_queue import CANDIDATE_FEATURES, connect_inference_queue
from hexorl.graph.batch import build_graph_batch_from_history, collate_graph_batches
from hexorl.models.factory import build_inference_model

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

    def test_server_forward_returns_sparse_pair_logits(self):
        """Server forward path returns active pair-policy logits when pair rows are supplied."""
        cfg = load_config()
        cfg.model.channels = 8
        cfg.model.blocks = 2
        cfg.model.sparse_policy = True
        cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
        cfg.inference.max_batch_size = 16
        cfg.inference.fp16 = False
        server = InferenceServer(cfg, num_workers=1)
        server._device = torch.device("cpu")
        server._model = build_inference_model(cfg, device=server._device)
        server._model.eval()

        count = 2
        k = 4
        p_rows = 3
        tensor = torch.randn(count, 13, 33, 33)
        candidate_indices = torch.from_numpy(np.tile(np.arange(k, dtype=np.int64), (count, 1)))
        candidate_features = torch.randn(count, k, CANDIDATE_FEATURES)
        candidate_mask = torch.ones(count, k, dtype=torch.bool)
        pair_indices = np.array(
            [
                [[0, 1], [0, 2], [2, 3]],
                [[0, 1], [1, 2], [0, 3]],
            ],
            dtype=np.int64,
        )
        pair_mask = torch.ones(count, p_rows, dtype=torch.bool)

        policies, values, sparse, pair, regret = server._forward(
            tensor,
            {
                "candidate_indices": candidate_indices,
                "candidate_features": candidate_features,
                "candidate_mask": candidate_mask,
                "pair_candidate_indices": torch.from_numpy(pair_indices),
                "pair_candidate_mask": pair_mask,
            },
        )

        self.assertEqual(policies.shape, (count, 1089))
        self.assertEqual(values.shape, (count,))
        self.assertEqual(sparse.shape, (count, k))
        self.assertEqual(pair.shape, (count, p_rows))
        self.assertIsNone(regret)
        self.assertTrue(np.isfinite(sparse).all())
        self.assertTrue(np.isfinite(pair).all())

    def test_server_forward_graph_returns_keyed_logits(self):
        cfg = load_config()
        cfg.model.architecture = "global_xattn_0"
        cfg.model.channels = 16
        cfg.model.attention_heads = 4
        cfg.model.graph_layers = 1
        cfg.model.heads = ["value", "policy_place", "policy_pair_joint", "opp_policy"]
        cfg.inference.max_batch_size = 2
        cfg.inference.fp16 = False
        server = InferenceServer(cfg, num_workers=1)
        server._device = torch.device("cpu")
        server._model = build_inference_model(cfg, device=server._device)
        server._model.eval()
        graph = collate_graph_batches([
            build_graph_batch_from_history(b"", include_pair_rows=False),
        ])
        graph_inputs = {
            "token_features": torch.from_numpy(graph.token_features),
            "token_type": torch.from_numpy(graph.token_type),
            "token_qr": torch.from_numpy(graph.token_qr),
            "token_mask": torch.from_numpy(graph.token_mask),
            "legal_token_indices": torch.from_numpy(graph.legal_token_indices),
            "legal_mask": torch.from_numpy(graph.legal_mask),
            "opp_legal_qr": torch.from_numpy(graph.opp_legal_qr),
            "opp_legal_mask": torch.from_numpy(graph.opp_legal_mask),
            "pair_token_indices": torch.from_numpy(graph.pair_token_indices),
            "pair_first_indices": torch.from_numpy(graph.pair_first_indices),
            "pair_second_indices": torch.from_numpy(graph.pair_second_indices),
            "relation_type": torch.from_numpy(graph.relation_type),
            "relation_bias": torch.from_numpy(graph.relation_bias),
        }

        place, values, opp, pair_first, pair_joint, pair_second, regret = server._forward_graph(graph_inputs)

        self.assertEqual(place.shape, graph.legal_mask.shape)
        self.assertEqual(values.shape, (1,))
        self.assertTrue(np.isfinite(place).all())
        self.assertTrue(np.isfinite(values).all())
        self.assertIsNotNone(opp)
        self.assertIsNotNone(pair_first)
        self.assertEqual(pair_first.shape, graph.legal_mask.shape)
        self.assertIsNotNone(pair_joint)
        self.assertIsNotNone(pair_second)
        self.assertIsNotNone(regret)
        self.assertEqual(regret.shape, (1,))

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

    def test_non_finite_outputs_are_rejected_before_mcts(self):
        policy = torch.tensor([[float("nan"), float("inf"), -float("inf"), 5.0]])
        value = torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.0]])

        with self.assertRaises(RuntimeError):
            InferenceServer._bounded_policy_logits(policy, head_name="policy")
        with self.assertRaises(RuntimeError):
            InferenceServer._bounded_value_logits(value, head_name="value")


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
        try:
            s0 = server_q.get_slot(0)
            client._slot.req_ready = s0.req_ready
            client._slot.res_ready = s0.res_ready

            game = _engine.PyHexGame()
            engine = _engine.PyMCTSEngine(
                game,
                num_simulations=8,
                c_puct=1.5,
                near_radius=2,
                c_puct_init=19652.0,
                constrain_threats=False,
                seed=42,
            )

            init = engine.init_root()
            self.assertIsNotNone(init, "init_root returned None (game over?)")
            tensor_3d, oq, or_, legal_bytes, root_generation = init
            self.assertEqual(tensor_3d.shape, (13, 33, 33))

            uniform = np.ones(1089, dtype=np.float32) / 1089.0
            engine.expand_root(uniform, 0.0, oq, or_, legal_bytes, root_generation)

            for _step in range(16):
                if engine.done():
                    break
                tensor_4d, count, batch_generation = engine.select_leaves(8)
                tensor_np = np.array(tensor_4d).astype(np.float32)
                policies, values = client.submit(tensor_np, count)
                engine.expand_and_backprop(policies, values, batch_generation)
            self.assertTrue(engine.done(), "MCTS did not finish within bounded inference loop")

            moves_q, moves_r, visits, root_value = engine.get_results()
            self.assertGreater(len(moves_q), 0, "No moves from MCTS")
            self.assertGreaterEqual(max(visits), 1, "No visits recorded")
            self.assertGreaterEqual(root_value, -1.0)
            self.assertLessEqual(root_value, 1.0)
        finally:
            client.disconnect()
            server.stop()
            server.join(timeout=5.0)
            server_q.close()


if __name__ == "__main__":
    unittest.main()
