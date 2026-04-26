"""Bayesian ELO rating system with confidence intervals.

Computes ratings from head-to-head match results using maximum-likelihood
ELO with conjugate prior uncertainty.
"""

import math
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class EloRating:
    """ELO rating with confidence interval."""
    rating: float
    lower: float
    upper: float
    games: int


def compute_elo(
    wins_a: int,
    wins_b: int,
    draws: int = 0,
    rating_a: float = 1500.0,
    rating_b: float = 1500.0,
) -> Tuple[float, float]:
    total = wins_a + wins_b + draws
    if total == 0:
        return rating_a, rating_b

    score_a = (wins_a + 0.5 * draws) / total
    expected_a = _expected_score(rating_a, rating_b)
    k = 32.0 / (1.0 + total / 100.0)
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_confidence_interval(wins: int, games: int, rating: float) -> Tuple[float, float]:
    if games == 0:
        return rating, rating
    p = wins / games
    sigma = 400.0 * math.sqrt(p * (1.0 - p) / games)
    return rating - 1.96 * sigma, rating + 1.96 * sigma


def compute_elo_from_results(
    results: List[Tuple[int, int, int]],
    initial_ratings: List[float] = None,
    iterations: int = 10,
) -> List[EloRating]:
    n = len(results) + 1 if results else 1
    if initial_ratings is None:
        ratings = [1500.0] * n
    else:
        ratings = list(initial_ratings)

    games_played = [0] * n
    wins = [0] * n

    for i, (wa, wb, draws) in enumerate(results):
        games_played[i] += wa + wb + draws
        wins[i] += wa
        if i + 1 < n:
            games_played[i + 1] += wa + wb + draws
            wins[i + 1] += wb

    for _ in range(iterations):
        for i in range(n - 1):
            if i >= len(results):
                continue
            wa, wb, draws = results[i]
            ratings[i], ratings[i + 1] = compute_elo(
                wa, wb, draws, ratings[i], ratings[i + 1]
            )

    elos = []
    for i in range(n):
        lo, hi = elo_confidence_interval(wins[i], max(games_played[i], 1), ratings[i])
        elos.append(EloRating(
            rating=ratings[i],
            lower=lo,
            upper=hi,
            games=games_played[i],
        ))

    return elos
