import json
import time
from pathlib import Path

from openai import OpenAIError

from app import db
from app.config import settings
from app.generation.guard import parse_verdict
from app.generation.llm import stream_answer
from app.generation.prompt import GUARDED_SYSTEM, build_user_prompt
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
OUT = Path("eval/results/guardrail.json")
PAUSE_S = 5.0  # stay under the free tier's requests-per-minute limit


def load_rows() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def ids(group: list[dict], verdict: str) -> list[str]:
    return [r["id"] for r in group if r["verdict"] == verdict]


def main() -> None:
    rows = load_rows()
    db.open_pool()
    search("warm up", mode="vector", rerank=False)

    records = []
    for n, row in enumerate(rows, start=1):
        res = search(row["question"], mode="vector", rerank=False, top_k=5)
        prompt = build_user_prompt(row["question"], res.hits)
        try:
            reply = "".join(stream_answer(GUARDED_SYSTEM, prompt))
            verdict = parse_verdict(reply)
        except OpenAIError as e:
            reply, verdict = f"{type(e).__name__}: {e}", "error"
        records.append(
            {"id": row["id"], "type": row["type"], "verdict": verdict, "reply": reply[:300]}
        )
        print(f"[{n}/{len(rows)}] {row['id']:<5} {verdict}", flush=True)
        time.sleep(PAUSE_S)
    db.close_pool()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"model": settings.llm_model, "records": records}, indent=2), encoding="utf-8"
    )

    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    false_refusals = ids(answerable, "no") + ids(answerable, "unclear")
    leaks = ids(unanswerable, "yes")
    errors = ids(answerable, "error") + ids(unanswerable, "error")

    print(f"\nModel: {settings.llm_model}")
    print(f"Answerable ({len(answerable)}): answered {len(ids(answerable, 'yes'))}, "
          f"refused {len(false_refusals)}")
    print("  false refusals:", false_refusals)
    print(f"Unanswerable ({len(unanswerable)}): refused "
          f"{len(unanswerable) - len(leaks) - len(ids(unanswerable, 'error'))}, "
          f"answered {len(leaks)}")
    print("  leaks (answered when it should refuse):", leaks)
    if errors:
        print("\nWARNING: API errors on", errors, "- re-run before trusting these numbers")


if __name__ == "__main__":
    main()