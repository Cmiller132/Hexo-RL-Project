"""Stable identity helpers for V2 contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


HASH_ALGORITHM = "sha256"


def stable_digest(parts: Iterable[bytes | str | int]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            data = part
        elif isinstance(part, int):
            data = str(part).encode("utf-8")
        else:
            data = str(part).encode("utf-8")
        hasher.update(len(data).to_bytes(8, "little", signed=False))
        hasher.update(data)
    return hasher.hexdigest()


def ndarray_digest(array: np.ndarray, *, schema_version: int, source: str) -> str:
    arr = np.ascontiguousarray(array)
    return stable_digest(
        (
            "ndarray",
            schema_version,
            source,
            str(arr.dtype),
            ",".join(str(dim) for dim in arr.shape),
            arr.tobytes(),
        )
    )


def readonly_array(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = np.array(array, dtype=dtype, copy=True) if dtype is not None else np.array(array, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ContractIdentity:
    contract: str
    schema_version: int
    source: str
    content_hash: str
