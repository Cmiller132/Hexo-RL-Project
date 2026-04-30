import numpy as np

from hexorl.config import load_config
from hexorl.inference.client import InferenceClient
from hexorl.inference.server import InferenceServer
from hexorl.inference.shm_queue import connect_inference_queue

import _engine


def main() -> None:
    cfg = load_config()
    cfg.model.channels = 8
    cfg.model.blocks = 2
    cfg.inference.max_batch_size = 16
    cfg.inference.fp16 = False

    print("before server", flush=True)
    server = InferenceServer(cfg, num_workers=1)
    server.start()
    print(f"after server start running={server.is_running()}", flush=True)

    client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=16, timeout_ms=5000)
    client.connect()
    server_q = connect_inference_queue(1, 16)
    slot = server_q.get_slot(0)
    client._slot.req_ready = slot.req_ready
    client._slot.res_ready = slot.res_ready
    print("after client connect", flush=True)

    try:
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
        print("engine built", flush=True)
        init = engine.init_root()
        print(f"init_root none={init is None}", flush=True)
        tensor_3d, offset_q, offset_r, legal_bytes = init
        print(f"init tensor shape={tensor_3d.shape}", flush=True)
        uniform = np.ones(1089, dtype=np.float32) / 1089.0
        print("before expand_root", flush=True)
        engine.expand_root(uniform, 0.0, offset_q, offset_r, legal_bytes)
        print(f"after expand_root done={engine.done()}", flush=True)

        for step in range(20):
            print(f"loop={step} done={engine.done()}", flush=True)
            if engine.done():
                break
            print("before select", flush=True)
            tensor_4d, count = engine.select_leaves(8)
            print(f"after select count={count} shape={getattr(tensor_4d, 'shape', None)}", flush=True)
            tensor_np = np.array(tensor_4d).astype(np.float32)
            print("before submit", flush=True)
            policies, values = client.submit(tensor_np, count)
            print(f"after submit policies={policies.shape} values={values.shape}", flush=True)
            print("before backprop", flush=True)
            engine.expand_and_backprop(policies, values)
            print("after backprop", flush=True)

        print("before results", flush=True)
        print(engine.get_results(), flush=True)
    finally:
        print("cleanup", flush=True)
        client.disconnect()
        server.stop()
        server.join(timeout=5.0)
        server_q.close()
        print("cleanup done", flush=True)


if __name__ == "__main__":
    main()
