"""Self-play orchestrator — supervisor process for N self-play workers.

Spawns workers, manages the inference server, collects game records,
pushes to the ring buffer, and monitors throughput.

§6.3 of SYSTEM_DESIGN.md — Workers are daemon=False so they survive
orchestrator restarts. On crash: drops the in-progress game, respawns.
"""

import time
import signal
import logging
import threading
import multiprocessing as mp
import queue
from typing import Optional, List

from hexorl.config import Config
from hexorl.inference.server import InferenceServer
from hexorl.buffer.ring import RingBuffer
from hexorl.selfplay.worker import SelfPlayWorker
from hexorl.dashboard.recorder import RunRecorder

logger = logging.getLogger(__name__)


class SelfPlayOrchestrator:
    """Supervisor that manages self-play workers and the inference server."""

    def __init__(
        self,
        cfg: Config,
        buffer_capacity: int = 100_000,
        initial_model_state: Optional[dict] = None,
        recorder: Optional[RunRecorder] = None,
    ):
        self.cfg = cfg
        self.num_workers = cfg.selfplay.num_workers
        self.max_batch = cfg.inference.max_batch_size
        self.games_per_epoch = cfg.selfplay.games_per_epoch
        self.states_per_epoch = cfg.selfplay.states_per_epoch

        # Inference server
        self._server: Optional[InferenceServer] = None
        self._initial_model_state = initial_model_state
        self._recorder = recorder

        # Ring buffer
        self._buffer = RingBuffer(
            capacity=buffer_capacity,
            recency_decay=cfg.buffer.recency_decay,
            num_lookahead=len(cfg.buffer.lookahead_horizons),
        )

        # Worker management
        self._workers: List[mp.Process] = []
        self._record_queue = mp.Queue(maxsize=5000)
        self._stop_event = mp.Event()
        self._collector_thread: Optional[threading.Thread] = None

        # Stats
        self._games_done = 0
        self._positions_done = 0
        self._start_time = 0.0
        self._stats_lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Launch inference server and worker processes."""
        self._start_time = time.monotonic()

        # 1. Start inference server
        logger.info(f"Starting inference server (workers={self.num_workers})")
        self._server = InferenceServer(
            self.cfg,
            num_workers=self.num_workers,
            initial_state_dict=self._initial_model_state,
        )
        self._server.start()

        # 2. Start record collector thread
        self._collector_thread = threading.Thread(
            target=self._collect_records,
            name="record-collector",
            daemon=True,
        )
        self._collector_thread.start()

        # 3. Spawn worker processes
        logger.info(f"Spawning {self.num_workers} workers")
        for i in range(self.num_workers):
            self._spawn_worker(i)

    def stop(self, drain_timeout: float = 10.0):
        """Graceful shutdown: drain queues, stop server, join workers."""
        logger.info("Orchestrator shutting down...")

        # Signal workers to stop
        self._stop_event.set()

        # Let workers observe the stop event before forcing termination.
        for p in self._workers:
            if p.is_alive():
                p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        self._workers.clear()

        # Wait for collector to drain records queued during worker shutdown.
        if self._collector_thread and self._collector_thread.is_alive():
            self._collector_thread.join(timeout=drain_timeout)

        # Stop inference server
        if self._server:
            self._server.stop()
            self._server.join(timeout=5.0)

        elapsed = time.monotonic() - self._start_time
        logger.info(
            f"Orchestrator stopped. "
            f"Games: {self._games_done}, "
            f"Positions: {self._positions_done}, "
            f"Rate: {self._games_done / max(elapsed, 0.1):.1f} games/min"
        )

    # ── Worker Management ────────────────────────────────────────────────

    def _spawn_worker(self, worker_id: int):
        """Spawn a single self-play worker process."""
        worker = SelfPlayWorker(
            worker_id=worker_id,
            cfg=self.cfg,
            record_queue=self._record_queue,
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
            stop_event=self._stop_event,
        )

        p = mp.Process(
            target=worker.run,
            name=f"selfplay-worker-{worker_id}",
            daemon=False,
        )
        p.start()
        self._workers.append(p)
        logger.info(f"Worker {worker_id} started (pid={p.pid})")

    def _respawn_worker(self, worker_id: int) -> mp.Process:
        """Replace a dead worker with a new one."""
        logger.warning(f"Worker {worker_id} died — respawning")
        worker = SelfPlayWorker(
            worker_id=worker_id,
            cfg=self.cfg,
            record_queue=self._record_queue,
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
            stop_event=self._stop_event,
        )
        p = mp.Process(
            target=worker.run,
            name=f"selfplay-worker-{worker_id}",
            daemon=False,
        )
        p.start()
        logger.info(f"Worker {worker_id} restarted (pid={p.pid})")
        return p

    def _monitor_workers(self):
        """Check worker health and respawn dead ones."""
        for i, p in enumerate(self._workers):
            if not p.is_alive():
                self._workers[i] = self._respawn_worker(i)

    # ── Record Collection ────────────────────────────────────────────────

    def _collect_records(self):
        """Continuously drain game records from the worker queue into the buffer."""
        while not self._stop_event.is_set() or not self._record_queue.empty():
            try:
                game_record = self._record_queue.get(timeout=0.5)
                self._ingest_game(game_record)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning("Record collector error: %s", e)

    def _ingest_game(self, game_record):
        """Process and store one completed game record."""
        try:
            # Targets are already computed by the worker before pushing.
            # Do not reprocess — it overwrites correct EMA lookahead values.

            if self._recorder is not None:
                self._recorder.game(game_record, source="selfplay")

            # Push terminal/resigned games into the training buffer. Truncated
            # games are useful for dashboard review but their fake draw outcome
            # is a bad value target unless explicitly allowed.
            is_truncated = bool(getattr(game_record, "truncated", False))
            valid_positions = (
                list(game_record.positions)
                if self.cfg.selfplay.train_on_truncated_games or not is_truncated
                else []
            )
            if valid_positions:
                self._buffer.extend(valid_positions)

            with self._stats_lock:
                self._games_done += 1
                self._positions_done += len(valid_positions)

        except Exception as e:
            logger.error(f"Failed to ingest game: {e}")

    # ── Status & Monitoring ──────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return current orchestrator statistics."""
        elapsed = max(time.monotonic() - self._start_time, 0.1)
        with self._stats_lock:
            return {
                "games_done": self._games_done,
                "positions_done": self._positions_done,
                "games_per_min": self._games_done / elapsed * 60.0,
                "positions_per_min": self._positions_done / elapsed * 60.0,
                "buffer_size": len(self._buffer),
                "buffer_capacity": self._buffer.capacity,
                "workers_alive": sum(1 for p in self._workers if p.is_alive()),
                "workers_total": len(self._workers),
                "elapsed_s": elapsed,
            }

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    @property
    def progress(self) -> float:
        """Fraction of epoch completed (0.0 to 1.0)."""
        targets = []
        with self._stats_lock:
            if self.games_per_epoch > 0:
                targets.append(self._games_done / self.games_per_epoch)
            if self.states_per_epoch > 0:
                targets.append(self._positions_done / self.states_per_epoch)
        if not targets:
            return 0.0
        return min(1.0, min(targets))

    @property
    def epoch_complete(self) -> bool:
        """Whether every configured epoch quota has been met."""
        with self._stats_lock:
            games_done = self.games_per_epoch <= 0 or self._games_done >= self.games_per_epoch
            states_done = self.states_per_epoch <= 0 or self._positions_done >= self.states_per_epoch
        return games_done and states_done


def run_orchestrator(
    cfg: Config,
    buffer_capacity: int = 100_000,
    initial_model_state: Optional[dict] = None,
    recorder: Optional[RunRecorder] = None,
):
    """Run the orchestrator until interrupted, then clean up.

    This is the main entry point called from the epoch runner.
    """
    orchestrator = SelfPlayOrchestrator(
        cfg,
        buffer_capacity=buffer_capacity,
        initial_model_state=initial_model_state,
        recorder=recorder,
    )

    # Handle SIGINT/SIGTERM gracefully
    def _shutdown(signum, frame):
        logger.info("Received shutdown signal")
        orchestrator.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        orchestrator.start()

        # Main monitoring loop
        while not orchestrator._stop_event.is_set():
            time.sleep(2.0)
            orchestrator._monitor_workers()

            stats = orchestrator.stats
            logger.info(
                f"Progress: {orchestrator.progress*100:.1f}% | "
                f"Games: {stats['games_done']} "
                f"({stats['games_per_min']:.1f}/min) | "
                f"Buffer: {stats['buffer_size']} | "
                f"Workers: {stats['workers_alive']}/{stats['workers_total']}"
            )

            # Check epoch completion
            if orchestrator.epoch_complete:
                logger.info("Epoch complete!")
                break

    finally:
        orchestrator.stop()

    return orchestrator
