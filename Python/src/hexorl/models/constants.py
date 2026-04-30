"""Shared constants for crop-based Hex models."""

from hexorl.contracts.candidates import CANDIDATE_FEATURES

BOARD_SIZE = 33
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
DEFAULT_CANDIDATE_FEATURES = CANDIDATE_FEATURES

__all__ = ["BOARD_AREA", "BOARD_SIZE", "DEFAULT_CANDIDATE_FEATURES"]
