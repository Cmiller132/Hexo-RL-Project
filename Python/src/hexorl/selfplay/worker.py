"""Self-play worker lifecycle and IPC shell."""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Optional

from hexorl.config import Config
from hexorl.inference.client import InferenceClient
from hexorl.selfplay.game_runner import GameRunRequest, create_default_game_runner

logger = logging.getLogger(__name__)


class SelfPlayWorker:
    """Own process lifecycle, inference IPC connection, and request forwarding."""

    def __init__(
        self,
        worker_id: int,
        cfg: Config,
        output_queue,
        num_workers: int = 24,
        max_batch_size: int = 128,
        stop_event: Optional[object] = None,
    ) -> None:
        self.worker_id = int(worker_id)
        self.cfg = cfg
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.num_workers = int(num_workers)
        self.max_batch_size = int(max_batch_size)
        self._game_counter = 0
        self._crash_count = 0
        self._run_id = f"selfplay-worker-{self.worker_id}"

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        logger.info("Self-play worker %s starting", self.worker_id)

        client = InferenceClient(
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            max_batch_size=self.max_batch_size,
            timeout_ms=30000,
        )
        try:
            try:
                client.connect()
            except Exception as exc:
                logger.warning("Self-play worker %s IPC unavailable: %s", self.worker_id, exc)
                client = None

            runner = create_default_game_runner(
                cfg=self.cfg,
                worker_id=self.worker_id,
                output_queue=self.output_queue,
                client=client,
                num_workers=self.num_workers,
                max_batch_size=self.max_batch_size,
            )

            while not self._stopping():
                request = self._next_request()
                runner.emit_worker_heartbeat(
                    process_id=os.getpid(),
                    run_id=request.run_id,
                    game_id=request.game_id,
                    phase="request_forwarded",
                    move_index=0,
                    positions_completed=0,
                )
                result = runner.run_game(request)
                if result.ok:
                    self._game_counter += 1
                    self._crash_count = 0
                else:
                    self._crash_count += 1
                    logger.error(
                        "Self-play worker %s run failure #%s: %s",
                        self.worker_id,
                        self._crash_count,
                        result.failure_status,
                    )
                    if self._crash_count > 10:
                        break
                    time.sleep(0.5)
        finally:
            if client is not None:
                client.disconnect()

    def _next_request(self) -> GameRunRequest:
        game_id = ((self.worker_id & 0xFF) << 24) | (self._game_counter & 0xFF_FFFF)
        seed = int(self.cfg.run.seed) + self.worker_id * 10000 + self._game_counter
        return GameRunRequest(
            run_id=self._run_id,
            game_id=int(game_id),
            game_index=int(self._game_counter),
            seed=int(seed),
        )

    def _stopping(self) -> bool:
        return self.stop_event is not None and bool(self.stop_event.is_set())
