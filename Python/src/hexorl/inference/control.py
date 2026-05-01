"""Generic shared-memory control block and dynamic dimension table."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


CONTROL_WORDS = 96
CTL_PROTOCOL_HASH = 0
CTL_CONTRACT_HASH = 1
CTL_LAYOUT_HASH = 2
CTL_OPCODE = 3
CTL_STATUS = 4
CTL_GENERATION = 5
CTL_DEADLINE_NS = 6
CTL_ENQUEUED_NS = 7
CTL_DYN_COUNT = 8
CTL_DYN_START = 9
MAX_DYN_DIMS = (CONTROL_WORDS - CTL_DYN_START) // 2

STATUS_EMPTY = 0
STATUS_READY = 1
STATUS_DRAINING = 2
STATUS_OK = 3
STATUS_ERROR = 4


def hash_word(value: str) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "little")


def dim_hash(name: str) -> int:
    return hash_word(f"dim:{name}")


def clear_dyn_dims(control: np.ndarray) -> None:
    control[CTL_DYN_COUNT] = 0
    control[CTL_DYN_START:] = 0


def write_dyn_dims(control: np.ndarray, dims: Mapping[str, int]) -> None:
    if len(dims) > MAX_DYN_DIMS:
        raise ValueError(f"too many dynamic dimensions for control block: {len(dims)}")
    clear_dyn_dims(control)
    control[CTL_DYN_COUNT] = len(dims)
    for idx, (name, value) in enumerate(sorted(dims.items())):
        base = CTL_DYN_START + idx * 2
        control[base] = dim_hash(name)
        control[base + 1] = int(value)


def read_dyn_dims(control: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    wanted = {dim_hash(name): name for name in names}
    found = {name: 0 for name in names}
    for idx in range(int(control[CTL_DYN_COUNT])):
        base = CTL_DYN_START + idx * 2
        name = wanted.get(int(control[base]))
        if name is not None:
            found[name] = int(control[base + 1])
    return found


def read_all_dyn_dims(control: np.ndarray, known_names: tuple[str, ...]) -> dict[str, int]:
    return {name: value for name, value in read_dyn_dims(control, known_names).items() if value > 0}

