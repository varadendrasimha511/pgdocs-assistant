CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    version     TEXT    NOT NULL,
    chapter     TEXT    NOT NULL,
    section     TEXT    NOT NULL,
    url         TEXT    NOT NULL,
    chunk_index INT     NOT NULL,
    content     TEXT    NOT NULL,
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(section, '')), 'B') ||
                  setweight(to_tsvector('english', content), 'A')
                ) STORED,
    UNIQUE (version, url, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_version_idx   ON chunks (version);
