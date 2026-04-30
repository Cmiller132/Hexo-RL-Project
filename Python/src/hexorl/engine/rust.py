"""Central Python-facing loader for the Rust `_engine` extension."""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType


class EngineUnavailableError(RuntimeError):
    """Raised when production code needs the Rust engine but it is unavailable."""


@lru_cache(maxsize=1)
def engine_module(*, required: bool = True) -> ModuleType | None:
    try:
        import _engine  # type: ignore
    except ImportError as exc:
        if required:
            raise EngineUnavailableError("Rust _engine extension is required") from exc
        return None
    return _engine


def engine_available() -> bool:
    return engine_module(required=False) is not None


def hex_game_class(*, required: bool = True):
    module = engine_module(required=required)
    if module is None:
        return None
    return getattr(module, "HexGame", None) or getattr(module, "PyHexGame")


def mcts_engine_class(*, required: bool = True):
    module = engine_module(required=required)
    if module is None:
        return None
    return getattr(module, "MCTSEngine", None) or getattr(module, "PyMCTSEngine")
