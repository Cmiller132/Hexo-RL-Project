import multiprocessing as mp

from hexorl.config import Config
from hexorl.selfplay.worker import SelfPlayWorker


def test_worker_creates_run_requests_without_game_execution_state():
    worker = SelfPlayWorker(2, Config(), output_queue=None, num_workers=1, max_batch_size=1)
    request = worker._next_request()

    assert request.game_id == ((2 & 0xFF) << 24)
    assert request.seed == Config().run.seed + 2 * 10000


def test_worker_cancellation_state_is_lifecycle_only():
    event = mp.Event()
    worker = SelfPlayWorker(0, Config(), output_queue=None, stop_event=event)

    assert worker._stopping() is False
    event.set()
    assert worker._stopping() is True
