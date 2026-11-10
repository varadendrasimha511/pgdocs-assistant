import json
import sys
from pathlib import Path

from app import db
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
GUARD = Path("eval/results/guardrail.json")
DEFAULT_IDS = ["f10", "f25", "m03", "m06", "m07", "f22"]


def page(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def main() -> None:
    wanted = sys.argv[1:] or DEFAULT_IDS
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    rows = {r["id"]: r for r in (json.loads(x) for x in lines if x.strip())}
    verdicts = {r["id"]: r for r in json.loads(GUARD.read_text(encoding="utf-8"))["records"]}
    db.open_pool()
    for qid in wanted:
        row = rows[qid]
        print("=" * 90)
        print(qid, "|", row["question"])
        print("model verdict:", verdicts[qid]["verdict"])
        print("expected:", [page(u) for u in row["expected_urls"]])
        res = search(row["question"], mode="vector", rerank=False, top_k=5)
        for i, h in enumerate(res.hits, start=1):
            mark = "*" if h.url in row["expected_urls"] else " "
            print(f"  {mark}{i}. {page(h.url)} | {h.section[:60]}")
            print(f"       {h.content[:300]!r}")
        if verdicts[qid]["verdict"] == "yes":
            print("\nmodel reply:", verdicts[qid]["reply"])
    db.close_pool()


if __name__ == "__main__":
    main()