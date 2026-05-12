import os
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


def _get_conn(register_vec: bool = True):
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "visionrag"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )
    if register_vec:
        register_vector(conn)
    return conn


def init_schema():
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()
    # Connect without registering vector type — extension doesn't exist yet
    conn = _get_conn(register_vec=False)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        conn.close()


def insert_document(filename: str, page_num: int, content: str, embedding: list[float]):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, page_num, content, embedding) VALUES (%s, %s, %s, %s)",
                (filename, page_num, content, embedding),
            )
        conn.commit()
    finally:
        conn.close()


def insert_documents_batch(records: list[tuple[str, int, str, list[float]]]):
    """records: list of (filename, page_num, content, embedding)"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (filename, page_num, content, embedding) VALUES %s",
                records,
            )
        conn.commit()
    finally:
        conn.close()


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, page_num, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"filename": r[0], "page_num": r[1], "content": r[2], "score": float(r[3])}
        for r in rows
    ]


def get_all_filenames() -> list[str]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT filename FROM documents ORDER BY filename")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def delete_document(filename: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE filename = %s", (filename,))
        conn.commit()
    finally:
        conn.close()
