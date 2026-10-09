def build_where(version: str | None, chapter: str | None) -> tuple[str, dict]:
    """Metadata filters, applied inside SQL before ranking."""
    clauses, params = ["TRUE"], {}
    if version:
        clauses.append("version = %(version)s")
        params["version"] = version
    if chapter:
        clauses.append("chapter ILIKE %(chapter)s")
        params["chapter"] = f"%{chapter}%"
    return " AND ".join(clauses), params