from app.retrieval.filters import build_where


def test_no_filters():
    sql, params = build_where(None, None)
    assert sql == "TRUE" and params == {}


def test_version_and_chapter():
    sql, params = build_where("16", "Indexes")
    assert "version = %(version)s" in sql and "chapter ILIKE" in sql
    assert params == {"version": "16", "chapter": "%Indexes%"}