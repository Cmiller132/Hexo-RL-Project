"""Pure V2 contract package."""

from hexorl.contracts.history import MoveHistory, decode_move_history, encode_move_history
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.symmetry import (
    apply_tensor_symmetry,
    compose_symmetries,
    inverse_symmetry,
    transform_axis_label,
    transform_axis_maps,
    transform_dense_policy,
    transform_history,
    transform_legal_table,
    transform_pair_policy_target,
    transform_policy_target,
    transform_qr,
)
from hexorl.contracts.targets import PairPolicyTarget, PolicyTarget
from hexorl.contracts.candidates import CandidateContractBuilder, CandidateDiagnostics, CandidateTable
from hexorl.contracts.pairs import PairActionTable, PairActionTableBuilder, PairStrategy
from hexorl.contracts.telemetry import ContractTrace
from hexorl.contracts.validation import ContractValidationError

__all__ = [
    "CandidateDiagnostics",
    "CandidateContractBuilder",
    "CandidateTable",
    "ContractTrace",
    "ContractValidationError",
    "LegalActionTable",
    "MoveHistory",
    "PairActionTable",
    "PairActionTableBuilder",
    "PairPolicyTarget",
    "PairStrategy",
    "PolicyTarget",
    "apply_tensor_symmetry",
    "compose_symmetries",
    "decode_move_history",
    "encode_move_history",
    "inverse_symmetry",
    "transform_axis_label",
    "transform_axis_maps",
    "transform_dense_policy",
    "transform_history",
    "transform_legal_table",
    "transform_pair_policy_target",
    "transform_policy_target",
    "transform_qr",
]
