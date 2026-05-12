from unittest.mock import patch, MagicMock, call
from app.db.vector_store import insert_document, search, get_all_filenames


def _mock_conn(cursor_rows=None):
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: mock_cursor
    mock_cursor.__exit__ = MagicMock(return_value=False)
    if cursor_rows is not None:
        mock_cursor.fetchall.return_value = cursor_rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


@patch("app.db.vector_store._get_conn")
def test_insert_document(mock_get_conn):
    mock_conn, mock_cursor = _mock_conn()
    mock_get_conn.return_value = mock_conn

    insert_document("invoice.pdf", 1, "Total: $500", [0.1] * 1536)

    mock_cursor.execute.assert_called_once()
    sql = mock_cursor.execute.call_args[0][0]
    assert "INSERT INTO documents" in sql
    mock_conn.commit.assert_called_once()


@patch("app.db.vector_store._get_conn")
def test_search_returns_results(mock_get_conn):
    rows = [("invoice.pdf", 1, "Total: $500", 0.92)]
    mock_conn, _ = _mock_conn(cursor_rows=rows)
    mock_get_conn.return_value = mock_conn

    results = search([0.1] * 1536, top_k=3)

    assert len(results) == 1
    assert results[0]["filename"] == "invoice.pdf"
    assert results[0]["page_num"] == 1
    assert results[0]["score"] == 0.92


@patch("app.db.vector_store._get_conn")
def test_get_all_filenames(mock_get_conn):
    rows = [("doc1.pdf",), ("doc2.png",)]
    mock_conn, _ = _mock_conn(cursor_rows=rows)
    mock_get_conn.return_value = mock_conn

    result = get_all_filenames()

    assert result == ["doc1.pdf", "doc2.png"]
