"""Contract-first model architecture authority."""

from hexorl.models.registry import (
    architecture_ids,
    architecture_spec,
    deprecated_aliases,
    global_graph_architecture_ids,
    is_global_graph_architecture,
    normalize_architecture_id,
    relation_required_architecture_ids,
    resolve_model_spec,
)


def build_model_from_config(*args, **kwargs):
    from hexorl.models.assembly import build_model_from_config as _build

    return _build(*args, **kwargs)


def from_config(*args, **kwargs):
    from hexorl.models.assembly import from_config as _from_config

    return _from_config(*args, **kwargs)


def bins_to_value(*args, **kwargs):
    from hexorl.models.assembly import bins_to_value as _bins_to_value

    return _bins_to_value(*args, **kwargs)


def load_model_state(*args, **kwargs):
    from hexorl.models.assembly import load_model_state as _load_model_state

    return _load_model_state(*args, **kwargs)


def is_global_graph_model(*args, **kwargs):
    from hexorl.models.assembly import is_global_graph_model as _is_global_graph_model

    return _is_global_graph_model(*args, **kwargs)

__all__ = [
    "architecture_ids",
    "architecture_spec",
    "bins_to_value",
    "build_model_from_config",
    "deprecated_aliases",
    "from_config",
    "global_graph_architecture_ids",
    "is_global_graph_architecture",
    "is_global_graph_model",
    "load_model_state",
    "normalize_architecture_id",
    "relation_required_architecture_ids",
    "resolve_model_spec",
]
