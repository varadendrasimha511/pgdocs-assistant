import time
from dataclasses import dataclass, field
from typing import Literal

from pgvector import Vector

from ..config import settings
from ..db import get_pool
from ..embeddings import embed_query
from .filters import build_where
from .rerank import rerank as rerank_hits
from .rrf import reciprocal_rank_fusion
from .types import Hit

Mode = Literal["vector", "keyword", "hybrid"]

COLS = "id, content, section, chapter, url, version"

VECTOR_SQL = """
SELECT {cols}, 1 - (embedding <=> %(q)s) AS score
FROM chunks WHERE {where}
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""

# plainto_tsquery ANDs every term, which finds nothing for natural-language questions.
# Replacing & with | gives OR semantics; ts_rank_cd then favours chunks matching more terms.
KEYWORD_SQL = """
WITH q AS (
  SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS tsq
)
SELECT {cols}, ts_rank_cd(tsv, tsq, 32) AS score
FROM chunks, q
WHERE {where} AND tsv @@ tsq
ORDER BY score DESC
LIMIT %(k)s
"""


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _run(sql: str, params: dict) -> list[Hit]:
    with get_pool().connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Hit(**r) for r in rows]


def search(
    query: str,
    mode: Mode = "hybrid",
    version: str | None = None,
    chapter: str | None = None,
    top_k: int = 5,
    candidates: int = 30,
    rerank: bool | None = None,
) -> SearchResult:
    version = version or settings.default_version
    rerank = settings.rerank_enabled if rerank is None else rerank
    where, params = build_where(version, chapter)
    t: dict[str, float] = {}

    vec_hits: list[Hit] = []
    kw_hits: list[Hit] = []

    if mode in ("vector", "hybrid"):
        s = time.perf_counter()
        qvec = Vector(embed_query(query))
        t["embed"] = (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        vec_hits = _run(
            VECTOR_SQL.format(cols=COLS, where=where), {**params, "q": qvec, "k": candidates}
        )
        t["vector"] = (time.perf_counter() - s) * 1000

    if mode in ("keyword", "hybrid"):
        s = time.perf_counter()
        kw_hits = _run(
            KEYWORD_SQL.format(cols=COLS, where=where), {**params, "text": query, "k": candidates}
        )
        t["keyword"] = (time.perf_counter() - s) * 1000

    if mode == "vector":
        hits = vec_hits
    elif mode == "keyword":
        hits = kw_hits
    else:
        hits = reciprocal_rank_fusion([vec_hits, kw_hits])

    if rerank and hits:
        s = time.perf_counter()
        hits = rerank_hits(query, hits[:candidates])
        t["rerank"] = (time.perf_counter() - s) * 1000

    return SearchResult(hits=hits[:top_k], timings_ms={k: round(v, 1) for k, v in t.items()})