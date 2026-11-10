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
    "hybrid (vector 2x)": {"mode": "hybrid", "rerank": False, "vector_weight": 2.0},
    "hybrid (vector 3x)": {"mode": "hybrid", "rerank": False, "vector_weight": 3.0},
    "vector + rerank": {"mode": "vector", "rerank": True},
    "vector + rerank (10)": {"mode": "vector", "rerank": True, "candidates": 10},
    "vector + rerank (5)": {"mode": "vector", "rerank": True, "candidates": 5},
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


def coverage(hits, expected: list[str], k: int = 5) -> float:
    """Share of the expected pages that appear in the top k results."""
    found = {h.url for h in hits[:k]} & set(expected)
    return len(found) / len(expected)


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
        ranks, latencies, misses, multi_cov = [], [], [], []
        details[name] = []

        for row in rows:
            start = time.perf_counter()
            res = search(row["question"], top_k=K, **cfg)
            latencies.append((time.perf_counter() - start) * 1000)

            top_score = res.hits[0].score if res.hits else 0.0
            entry = {
                "id": row["id"],
                "type": row["type"],
                "top_score": round(top_score, 3),
                "top_urls": [h.url.rsplit("/", 1)[-1] for h in res.hits[:3]],
            }

            if row["type"] != "unanswerable":
                rank = first_relevant_rank(res.hits, row["expected_urls"])
                ranks.append(rank)
                entry["rank"] = rank
                if rank is None or rank > 5:
                    misses.append(row["id"])
                if row["type"] == "multi":
                    multi_cov.append(coverage(res.hits, row["expected_urls"]))
            details[name].append(entry)

        n = len(ranks)
        summary[name] = {
            "questions": n,
            "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "multi_coverage@5": round(sum(multi_cov) / max(len(multi_cov), 1), 3),
            "p50_ms": round(percentile(latencies, 50)),
            "p95_ms": round(percentile(latencies, 95)),
            "missed_ids": misses,
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

    print(f"\n{'config':<22}{'hit@5':>8}{'MRR':>8}{'multi@5':>9}{'p50 ms':>9}{'p95 ms':>9}")
    for name, s in summary.items():
        print(
            f"{name:<22}{s['hit@5']:>8}{s['mrr']:>8}{s['multi_coverage@5']:>9}"
            f"{s['p50_ms']:>9}{s['p95_ms']:>9}"
        )

    print("\nMissed questions (not in top 5):")
    for name, s in summary.items():
        print(f"  {name:<22}{s['missed_ids']}")

    # Data for the 'no evidence, no answer' guardrail (hybrid + rerank scores)
    scores = details["hybrid + rerank"]
    answerable = sorted((e["top_score"], e["id"]) for e in scores if e["type"] != "unanswerable")
    unanswerable = sorted(
        ((e["top_score"], e["id"]) for e in scores if e["type"] == "unanswerable"), reverse=True
    )
    print("\nLowest 5 answerable top scores :", answerable[:5])
    print("Unanswerable top scores (high to low):", unanswerable)


if __name__ == "__main__":
    main()