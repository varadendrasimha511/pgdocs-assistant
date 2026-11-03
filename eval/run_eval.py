import json
import time
from pathlib import Path

from app import db
from app.config import settings
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
OUT = Path("eval/results/retrieval.json")
K = 10  # retrieve 10; hit@5 and MRR are computed from this list

CONFIGS = {
    "vector": {"mode": "vector", "rerank": False},
    "keyword": {"mode": "keyword", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "rerank": True},
}


def load_rows() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_relevant_rank(hits, expected: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.url in expected:
            return rank
    return None


def percentile(values: list[float], p: int) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def main() -> None:
    rows = load_rows()
    db.open_pool()
    search("warm up", mode="hybrid", rerank=True)  # load models before timing anything

    summary = {}
    details = {}
    for name, cfg in CONFIGS.items():
        ranks, latencies, misses = [], [], []
        top_scores = {"answerable": [], "unanswerable": []}
        details[name] = []

        for row in rows:
            start = time.perf_counter()
            res = search(row["question"], top_k=K, **cfg)
            latencies.append((time.perf_counter() - start) * 1000)

            top = res.hits[0].score if res.hits else 0.0
            if row["type"] == "unanswerable":
                top_scores["unanswerable"].append(top)
                continue
            top_scores["answerable"].append(top)
            rank = first_relevant_rank(res.hits, row["expected_urls"])
            ranks.append(rank)
            if rank is None or rank > 5:
                misses.append(row["id"])
            details[name].append(
                {
                    "id": row["id"],
                    "rank": rank,
                    "top_urls": [h.url.rsplit("/", 1)[-1] for h in res.hits[:3]],
                }
            )

        n = len(ranks)
        n_unans = max(len(top_scores["unanswerable"]), 1)
        summary[name] = {
            "questions": n,
            "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "p50_ms": round(percentile(latencies, 50)),
            "p95_ms": round(percentile(latencies, 95)),
            "missed_ids": misses,
            "avg_top_score_answerable": round(sum(top_scores["answerable"]) / n, 3),
            "avg_top_score_unanswerable": round(sum(top_scores["unanswerable"]) / n_unans, 3),
        }

    db.close_pool()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"embed_model": settings.embed_model, "summary": summary, "details": details},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'config':<18}{'hit@5':>8}{'MRR':>8}{'p50 ms':>9}{'p95 ms':>9}")
    for name, s in summary.items():
        print(f"{name:<18}{s['hit@5']:>8}{s['mrr']:>8}{s['p50_ms']:>9}{s['p95_ms']:>9}")

    print("\nMissed questions (not in top 5):")
    for name, s in summary.items():
        print(f"  {name:<18}{s['missed_ids']}")

    r = summary["hybrid + rerank"]
    print(
        "\nAvg top rerank score: answerable",
        r["avg_top_score_answerable"],
        "| unanswerable",
        r["avg_top_score_unanswerable"],
    )


if __name__ == "__main__":
    main()