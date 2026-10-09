from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(settings.embed_model)

def embed_texts(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]

def embed_query(query: str) -> list[float]:
    # bge models expect a special instruction prefix for queries; query_embed adds it
    return next(iter(_model().query_embed(query))).tolist()