from collections.abc import Sequence
from dataclasses import replace

from .types import Hit


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    """score(doc) = sum over each ranking of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.id, hit)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [replace(first_seen[i], score=scores[i]) for i in ordered]