"""Suite dashboard routers."""

from fastapi import APIRouter

from hexorl.dashboard.routes.suite import autotune, events, status, trials

router = APIRouter()
router.include_router(status.router)
router.include_router(trials.router)
router.include_router(events.router)
router.include_router(autotune.router)
