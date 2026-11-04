import json
import sys
from pathlib import Path

from app import db
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
DEFAULT_IDS = ["f17", "f19", "m04", "f15", "f07", "f24", "m03", "m10"]


def page(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def main() -> None:
    wanted = sys.argv[1:] or DEFAULT_IDS
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {r["id"]: r for r in rows}
    db.open_pool()
    for qid in wanted:
        row = by_id[qid]
        print("=" * 90)
        print(qid, "|", row["question"])
        print("expected:", [page(u) for u in row["expected_urls"]])
        for mode, rerank in (("vector", False), ("hybrid", True)):
            label = mode + (" + rerank" if rerank else "")
            print(f"\n  {label}")
            res = search(row["question"], mode=mode, rerank=rerank, top_k=5)
            for i, h in enumerate(res.hits, start=1):
                mark = "*" if h.url in row["expected_urls"] else " "
                print(f"   {mark}{i}. {h.score:8.3f}  {page(h.url)}  | {h.section[:60]}")
    db.close_pool()


if __name__ == "__main__":
    main()