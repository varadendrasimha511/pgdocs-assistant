import json
from pathlib import Path

RESULTS = Path("eval/results/retrieval.json")


def best_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, int, int]:
    """Return (threshold, wrongly_refused, wrongly_answered) with the fewest total errors.

    Rule: answer when the top score >= threshold, otherwise refuse.
    """
    best: tuple[int, int, float, int] | None = None
    for t in sorted(set(answerable + unanswerable)):
        refused = sum(1 for a in answerable if a < t)
        leaked = sum(1 for u in unanswerable if u >= t)
        key = (refused + leaked, refused, t, leaked)
        if best is None or key < best:
            best = key
    assert best is not None
    return best[2], best[1], best[3]


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    first = next(iter(data["details"].values()))
    n_unans = sum(1 for r in first if r["type"] == "unanswerable")
    print(f"{len(first) - n_unans} answerable / {n_unans} unanswerable questions")
    print("Rule: answer if top score >= threshold, else refuse\n")

    print(
        f"{'config':<22}{'lowest ans':>12}{'highest unans':>15}"
        f"{'wrong refusals':>16}{'wrong answers':>15}  threshold"
    )
    for name, rows in data["details"].items():
        answerable = [r["top_score"] for r in rows if r["type"] != "unanswerable"]
        unanswerable = [r["top_score"] for r in rows if r["type"] == "unanswerable"]
        t, refused, leaked = best_threshold(answerable, unanswerable)
        print(
            f"{name:<22}{min(answerable):>12}{max(unanswerable):>15}"
            f"{refused:>16}{leaked:>15}  {t}"
        )


if __name__ == "__main__":
    main()