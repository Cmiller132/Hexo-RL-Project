"""Persistent checkpoint league ratings for Phase 3 champion selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColorRecord:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


@dataclass(frozen=True)
class LeagueRating:
    checkpoint_id: str
    rating_mean: float
    rating_std: float
    lcb: float
    games: int
    by_color: dict[str, ColorRecord]


@dataclass
class CheckpointLeague:
    ratings: dict[str, LeagueRating] = field(default_factory=dict)
    matches: list[dict[str, Any]] = field(default_factory=list)
    lcb_sigma: float = 1.0

    def record_match(
        self,
        checkpoint_id: str,
        *,
        color: str,
        wins: int,
        losses: int,
        draws: int = 0,
    ) -> LeagueRating:
        if color not in {"black", "white"}:
            raise ValueError("color must be black or white")
        games = wins + losses + draws
        if games <= 0:
            raise ValueError("match record must contain at least one game")
        self.matches.append(
            {
                "checkpoint_id": checkpoint_id,
                "color": color,
                "wins": wins,
                "losses": losses,
                "draws": draws,
            }
        )
        rating = self._compute_rating(checkpoint_id)
        self.ratings[checkpoint_id] = rating
        return rating

    def both_colors_recorded(self, checkpoint_id: str) -> bool:
        colors = {
            match["color"]
            for match in self.matches
            if match["checkpoint_id"] == checkpoint_id
        }
        return {"black", "white"}.issubset(colors)

    def champion_by_lcb(self) -> LeagueRating:
        if not self.ratings:
            raise ValueError("league has no ratings")
        eligible = [
            rating
            for rating in self.ratings.values()
            if self.both_colors_recorded(rating.checkpoint_id)
        ]
        if not eligible:
            raise ValueError("no checkpoint has both-color league evidence")
        return max(eligible, key=lambda rating: (rating.lcb, rating.checkpoint_id))

    def _compute_rating(self, checkpoint_id: str) -> LeagueRating:
        by_color = {
            "black": ColorRecord(),
            "white": ColorRecord(),
        }
        for match in self.matches:
            if match["checkpoint_id"] != checkpoint_id:
                continue
            prior = by_color[match["color"]]
            by_color[match["color"]] = ColorRecord(
                games=prior.games + match["wins"] + match["losses"] + match["draws"],
                wins=prior.wins + match["wins"],
                losses=prior.losses + match["losses"],
                draws=prior.draws + match["draws"],
            )
        total_games = sum(record.games for record in by_color.values())
        total_score = sum(record.score for record in by_color.values())
        score_rate = total_score / total_games
        rating_mean = 1500.0 + 800.0 * (score_rate - 0.5)
        variance = max(score_rate * (1.0 - score_rate), 1e-9) / total_games
        rating_std = 800.0 * math.sqrt(variance)
        return LeagueRating(
            checkpoint_id=checkpoint_id,
            rating_mean=rating_mean,
            rating_std=rating_std,
            lcb=rating_mean - self.lcb_sigma * rating_std,
            games=total_games,
            by_color=by_color,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lcb_sigma": self.lcb_sigma,
            "matches": self.matches,
            "ratings": {
                checkpoint_id: {
                    **asdict(rating),
                    "by_color": {
                        color: asdict(record)
                        for color, record in rating.by_color.items()
                    },
                }
                for checkpoint_id, rating in self.ratings.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointLeague":
        league = cls(lcb_sigma=float(payload.get("lcb_sigma", 1.0)))
        league.matches = list(payload.get("matches", []))
        for checkpoint_id, rating in payload.get("ratings", {}).items():
            league.ratings[checkpoint_id] = LeagueRating(
                checkpoint_id=rating["checkpoint_id"],
                rating_mean=rating["rating_mean"],
                rating_std=rating["rating_std"],
                lcb=rating["lcb"],
                games=rating["games"],
                by_color={
                    color: ColorRecord(**record)
                    for color, record in rating["by_color"].items()
                },
            )
        return league

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointLeague":
        return cls.from_dict(json.loads(Path(path).read_text()))
