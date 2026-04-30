"""Inference server process facade."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import signal
from typing import Optional

from hexorl.config import Config
from hexorl.inference.protocol import default_protocol_manifest, publish_server_manifest, remove_server_manifest
from hexorl.inference.server.batching import BatchingPolicy
from hexorl.inference.server.collation import ServerCollator
from hexorl.inference.server.execution import ServerExecutor
from hexorl.inference.server.metrics import ServerMetrics
from hexorl.inference.server.runtime import apply_latest_weight_update, initialize_runtime, state_to_cpu
from hexorl.inference.server.scatter import ServerScatterer
from hexorl.inference.server.scheduler import InferenceScheduler
from hexorl.inference.shm_queue import create_inference_queue, connect_inference_queue
from hexorl.models.factory import model_uses_global_graph
from hexorl.models.specs import model_spec_from_config


class InferenceServer:
    """GPU inference server process facade."""

    def __init__(
        self,
        cfg: Config,
        num_workers: int,
        initial_state_dict: Optional[dict] = None,
    ):
        self.cfg = cfg
        self._global_graph_kind = model_uses_global_graph(cfg)
        model_spec = model_spec_from_config(cfg)
        required_heads = {"value"} if self._global_graph_kind else {"policy", "value"}
        missing_heads = sorted(required_heads - set(cfg.model.heads))
        if missing_heads:
            raise ValueError(
                "InferenceServer requires model heads for self-play inference: "
                f"{missing_heads}"
            )
        self.num_workers = int(num_workers)
        self.max_batch = int(cfg.inference.max_batch_size)
        self.max_wait_us = int(cfg.inference.max_wait_us)
        self.fp16 = bool(cfg.inference.fp16)
        self.manifest = default_protocol_manifest(
            max_batch_size=self.max_batch,
            timeout_ms=float(getattr(cfg.inference, "timeout_ms", 30000.0)),
            heads=tuple(str(head) for head in cfg.model.heads),
            adapter_name="hexorl-shm-server",
            model_family=model_spec.kind,
            model_spec_version=str(model_spec.version),
            config_hash=str(getattr(cfg.run, "name", "server")),
        )
        self._batching_policy = BatchingPolicy(
            max_batch_size=self.max_batch,
            max_wait_us=self.max_wait_us,
        )

        self._mp_ctx = mp.get_context("spawn")
        self._process: Optional[mp.Process] = None
        self._stop_event = self._mp_ctx.Event()
        self._ready_event = self._mp_ctx.Event()
        self._weight_queue = self._mp_ctx.Queue(maxsize=2)
        self._initial_state_dict = state_to_cpu(initial_state_dict)
        self._metrics = ServerMetrics()

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_owned_queue", "_queue", "_process", "_mp_ctx"):
            state.pop(key, None)
        return state

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Server already running")

        self._owned_queue = create_inference_queue(self.num_workers, self.max_batch)
        publish_server_manifest(
            self.manifest,
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
        )

        self._process = self._mp_ctx.Process(
            target=self._run,
            name="hexorl-inference-server",
            daemon=False,
        )
        self._process.start()
        timeout_s = max(1.0, float(getattr(self.cfg.runtime, "inference_start_timeout_s", 30.0)))
        if not self._ready_event.wait(timeout=timeout_s):
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            self.close()
            self._process = None
            raise RuntimeError(f"Inference server failed to start within {timeout_s:g}s")

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.exitcode is None:
                self._process.terminate()
            self._process = None
        self.close()

    def close(self) -> None:
        if hasattr(self, "_owned_queue") and self._owned_queue is not None:
            self._owned_queue.close()
            self._owned_queue = None
        remove_server_manifest(num_workers=self.num_workers, max_batch_size=self.max_batch)

    def update_weights(self, state_dict: dict) -> None:
        cpu_state = state_to_cpu(state_dict)
        if cpu_state is None:
            return
        while self._weight_queue.full():
            try:
                self._weight_queue.get_nowait()
            except Exception:
                break
        self._weight_queue.put(cpu_state)

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        runtime = initialize_runtime(self.cfg, self._initial_state_dict)
        queue = connect_inference_queue(self.num_workers, self.max_batch)
        metrics = ServerMetrics()
        try:
            collator = ServerCollator(
                cfg=self.cfg,
                queue=queue,
                device=runtime.device,
                max_batch=self.max_batch,
                global_graph_kind=self._global_graph_kind,
            )
            executor = ServerExecutor(
                model=runtime.model,
                device=runtime.device,
                forward_stream=runtime.forward_stream,
                fp16=self.fp16,
                metrics=metrics,
            )
            scatterer = ServerScatterer(queue=queue)
            scheduler = InferenceScheduler(
                queue=queue,
                num_workers=self.num_workers,
                max_batch=self.max_batch,
                max_wait_us=self.max_wait_us,
                stop_event=self._stop_event,
                batching_policy=self._batching_policy,
                collator=collator,
                executor=executor,
                scatterer=scatterer,
                metrics=metrics,
                weight_poll=lambda: apply_latest_weight_update(
                    runtime.model,
                    self._weight_queue,
                    device=runtime.device,
                ),
                device=runtime.device,
                fp16=self.fp16,
            )
            self._ready_event.set()
            asyncio.run(scheduler.run())
        except Exception as exc:
            print(f"[inference-server] Fatal error: {exc}", flush=True)
            raise
        finally:
            queue.close()

    @property
    def positions_per_sec(self) -> float:
        return self._metrics.positions_per_sec


__all__ = ["InferenceServer"]
