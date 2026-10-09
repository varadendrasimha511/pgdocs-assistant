import argparse
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from .. import db
from ..config import settings
from ..embeddings import embed_texts
from .chunker import chunk_html
from .crawl import crawl

INSERT = """
INSERT INTO chunks (version, chapter, section, url, chunk_index, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (version, url, chunk_index) DO UPDATE
SET chapter = EXCLUDED.chapter, section = EXCLUDED.section,
    content = EXCLUDED.content, embedding = EXCLUDED.embedding
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=settings.default_version)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--cache", default="data/raw")
    args = ap.parse_args()

    db.init_db()
    total_pages = total_chunks = 0
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        for url, html in crawl(args.version, Path(args.cache), args.max_pages):
            chunks = chunk_html(html)
            if not chunks:
                continue
            vectors = embed_texts([c.embed_text for c in chunks])
            with conn.cursor() as cur:
                cur.executemany(
                    INSERT,
                    [
                        (args.version, c.chapter, c.section, url, c.index, c.text, Vector(v))
                        for c, v in zip(chunks, vectors)
                    ],
                )
            total_pages += 1
            total_chunks += len(chunks)
            print(f"[{total_pages}] {url} -> {len(chunks)} chunks", flush=True)
    print(f"Done: {total_pages} pages, {total_chunks} chunks")


if __name__ == "__main__":
    main()