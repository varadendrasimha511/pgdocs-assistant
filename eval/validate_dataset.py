import json
from collections import Counter
from pathlib import Path

import psycopg

from app.config import settings

DATASET = Path("eval/dataset.jsonl")
TYPES = {"factual", "multi", "unanswerable"}


def main() -> None:
    with psycopg.connect(settings.database_url) as conn:
        known = {r[0] for r in conn.execute("SELECT DISTINCT url FROM chunks")}

    problems: list[str] = []
    counts: Counter = Counter()
    seen_ids: set[str] = set()

    for n, line in enumerate(DATASET.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e})")
            continue

        rid = row.get("id", f"line{n}")
        if rid in seen_ids:
            problems.append(f"{rid}: duplicate id")
        seen_ids.add(rid)

        if row.get("type") not in TYPES:
            problems.append(f"{rid}: type must be one of {sorted(TYPES)}")
        counts[row.get("type")] += 1

        if not row.get("question"):
            problems.append(f"{rid}: missing question")

        urls = row.get("expected_urls", [])
        if row.get("type") == "unanswerable":
            if urls:
                problems.append(f"{rid}: unanswerable questions need empty expected_urls")
        else:
            if not urls:
                problems.append(f"{rid}: needs at least one expected URL")
            if not row.get("reference_answer"):
                problems.append(f"{rid}: missing reference_answer")
        for u in urls:
            if u not in known:
                problems.append(f"{rid}: URL not in your database -> {u}")

    print("Counts:", dict(counts), "| total:", sum(counts.values()))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
    else:
        print("Dataset OK")


if __name__ == "__main__":
    main()