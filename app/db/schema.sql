CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    page_num    INT  NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops);
