from dataclasses import replace
from functools import lru_cache

from ..config import settings
from .types import Hit


@lru_cache(maxsize=1)
def _encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """Score each (query, passage) pair jointly and re-sort."""
    if not hits:
        return hits
    docs = [f"{h.section}\n{h.content}" for h in hits]
    scores = list(_encoder().rerank(query, docs))
    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
    return [replace(h, score=float(s)) for h, s in ranked]