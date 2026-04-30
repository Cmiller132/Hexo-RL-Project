"""Search-facing immutable position context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hexorl.contracts.candidates import CandidateTable
from hexorl.contracts.identity import ndarray_digest, stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.pairs import PairActionTable
from hexorl.graph.tensorize import GraphBatch

SearchPhase = Literal["root", "leaf"]


@dataclass(frozen=True)
class SearchContext:
    """Canonical search view of one root or leaf position.

    The context carries existing contract objects and byte payload identities.
    It does not rebuild legal rows, candidates, graph rows, pair rows, compact
    history, or transforms.
    """

    phase: SearchPhase
    legal_table: LegalActionTable
    trace_id: str
    model_family: str
    model_spec_version: str
    recipe_id: str
    search_id: str
    pair_strategy_id: str
    tensor: np.ndarray | None = None
    history_bytes: bytes = b""
    root_generation: int | None = None
    batch_generation: int | None = None
    candidate_table: CandidateTable | None = None
    pair_table: PairActionTable | None = None
    graph_batch: GraphBatch | None = None
    graph_contract_hash: str = ""
    inference_protocol: str = "v1"
    extra: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        phase: SearchPhase,
        legal_table: LegalActionTable,
        model_family: str,
        tensor: np.ndarray | None = None,
        history_bytes: bytes = b"",
        root_generation: int | None = None,
        batch_generation: int | None = None,
        candidate_table: CandidateTable | None = None,
        pair_table: PairActionTable | None = None,
        graph_batch: GraphBatch | None = None,
        trace_id: str | None = None,
        model_spec_version: str = "v2",
        recipe_id: str = "default",
        search_id: str = "selfplay",
        pair_strategy_id: str = "none",
        inference_protocol: str = "v1",
        extra: dict[str, Any] | None = None,
    ) -> "SearchContext":
        return cls(
            phase=phase,
            legal_table=legal_table,
            trace_id=trace_id or uuid.uuid4().hex,
            model_family=str(model_family),
            model_spec_version=str(model_spec_version),
            recipe_id=str(recipe_id),
            search_id=str(search_id),
            pair_strategy_id=str(pair_strategy_id),
            tensor=None if tensor is None else np.array(tensor, dtype=np.float32, copy=True),
            history_bytes=bytes(history_bytes),
            root_generation=root_generation,
            batch_generation=batch_generation,
            candidate_table=candidate_table,
            pair_table=pair_table,
            graph_batch=graph_batch,
            graph_contract_hash="" if graph_batch is None else str(getattr(graph_batch, "graph_semantic_hash", "")),
            inference_protocol=str(inference_protocol),
            extra=dict(extra or {}),
        )

    @property
    def position_hash(self) -> str:
        tensor_hash = "none" if self.tensor is None else ndarray_digest(self.tensor, schema_version=1, source="search")
        return stable_digest(
            (
                "SearchContext",
                self.phase,
                self.legal_table.table_hash,
                tensor_hash,
                self.history_hash,
                self.root_generation or -1,
                self.batch_generation or -1,
            )
        )

    @property
    def history_hash(self) -> str:
        return stable_digest(("history", self.history_bytes))

    @property
    def candidate_hash(self) -> str:
        return "" if self.candidate_table is None else self.candidate_table.table_hash

    @property
    def pair_hash(self) -> str:
        return "" if self.pair_table is None else self.pair_table.table_hash

    def identity_payload(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "phase": self.phase,
            "position_hash": self.position_hash,
            "history_hash": self.history_hash,
            "legal_table_hash": self.legal_table.table_hash,
            "candidate_table_hash": self.candidate_hash,
            "pair_table_hash": self.pair_hash,
            "root_generation": self.root_generation,
            "batch_generation": self.batch_generation,
            "model_family": self.model_family,
            "model_spec_version": self.model_spec_version,
            "pair_strategy_id": self.pair_strategy_id,
        }
