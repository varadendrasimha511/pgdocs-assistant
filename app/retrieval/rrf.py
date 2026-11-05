from collections.abc import Sequence
from dataclasses import replace

from .types import Hit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """score(doc) = sum over rankings of weight / (k + rank)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (k + rank)
            first_seen.setdefault(hit.id, hit)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [replace(first_seen[i], score=scores[i]) for i in ordered]