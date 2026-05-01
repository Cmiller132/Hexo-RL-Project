"""FastAPI dashboard application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from hexorl.dashboard.arena_service import ArenaManager
from hexorl.dashboard.contract_inspector import ContractInspector
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.model_cache import ModelCache
from hexorl.dashboard.routes import arena, axis, checkpoints, games, health, inspection, metrics, model, runs, sessions
from hexorl.dashboard.routes.suite import router as suite_router

def create_app(
    db_path: Path | str = "runs/dashboard.sqlite3",
    *,
    frontend_dist: Path | str | None = None,
    run_root: Path | str | None = None,
) -> FastAPI:
    store = DashboardStore(db_path)
    suite_root = Path(run_root).expanduser().resolve() if run_root else None
    app = FastAPI(title="Hexo-RL Dashboard", version="0.1.0")
    app.state.store = store
    app.state.suite_root = suite_root
    app.state.model_cache = ModelCache()
    app.state.arena = ArenaManager(store)
    app.state.contract_inspector = ContractInspector()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    routers = (health.router, runs.router, metrics.router, checkpoints.router, games.router, sessions.router, axis.router, model.router, inspection.router, arena.router, suite_router)
    for router in routers:
        app.include_router(router)

    _mount_frontend(app, Path(frontend_dist) if frontend_dist else _default_frontend_dist())
    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(dist / "index.html")

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            target = dist / path
            if target.exists() and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    else:

        @app.get("/")
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(
                "<html><body><h1>Hexo-RL Dashboard API</h1>"
                "<p>Build Python/dashboard_frontend to serve the React UI.</p></body></html>"
            )


def _default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "dashboard_frontend" / "dist"


def default_app() -> FastAPI:
    """Factory target for ASGI runners that prefer a zero-argument callable."""
    return create_app()
