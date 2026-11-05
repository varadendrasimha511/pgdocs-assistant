import json
import sys
from pathlib import Path

from app import db
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
DEFAULT_IDS = ["f15", "f17", "f19", "f07", "f24", "m03", "m04", "m10"]
DEPTH = 200


def page(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def best_ranks(hits, expected: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for rank, hit in enumerate(hits, start=1):
        if hit.url in expected and hit.url not in found:
            found[hit.url] = rank
    return found


def main() -> None:
    wanted = sys.argv[1:] or DEFAULT_IDS
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    by_id = {r["id"]: r for r in (json.loads(x) for x in lines if x.strip())}
    db.open_pool()
    for qid in wanted:
        row = by_id[qid]
        print(f"\n{qid} | {row['question']}")
        for mode in ("vector", "keyword", "hybrid"):
            res = search(row["question"], mode=mode, rerank=False, top_k=DEPTH, candidates=DEPTH)
            found = best_ranks(res.hits, row["expected_urls"])
            cells = []
            for url in row["expected_urls"]:
                rank = found.get(url)
                cells.append(f"{page(url)}: {rank if rank else '>' + str(DEPTH)}")
            print(f"  {mode:<8}" + " | ".join(cells))
    db.close_pool()


if __name__ == "__main__":
    main()