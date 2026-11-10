import asyncio
import json
import logging
import math
from pathlib import Path

from openai import AsyncOpenAI

from app import db
from app.config import settings
from app.generation.guard import parse_verdict, strip_verdict
from app.generation.llm import BASE_URLS, stream_answer
from app.generation.prompt import GUARDED_SYSTEM, build_user_prompt
from app.retrieval.search import search

DATASET = Path("eval/dataset.jsonl")
OUT = Path("eval/results/faithfulness.json")
LIMIT = 15  # questions scored; keeps the free-tier quota safe
PAUSE_S = 5.0

logger = logging.getLogger("faithfulness")


def load_rows() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def main() -> None:
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    answerable = [r for r in load_rows() if r["type"] != "unanswerable"]
    step = max(1, len(answerable) // LIMIT)
    subset = answerable[::step][:LIMIT]  # evenly spaced, so the same questions every run

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or BASE_URLS[settings.llm_provider],
        max_retries=3,
    )
    judge = llm_factory(settings.llm_model, client=client, max_tokens=4096)
    metric = Faithfulness(llm=judge)

    db.open_pool()
    search("warm up", mode="vector", rerank=False)

    records = []
    for n, row in enumerate(subset, start=1):
        res = search(row["question"], mode="vector", rerank=False, top_k=5)
        prompt = build_user_prompt(row["question"], res.hits)
        record = {"id": row["id"], "status": "error", "score": None, "answer": ""}
        try:
            reply = "".join(stream_answer(GUARDED_SYSTEM, prompt))
            if parse_verdict(reply) != "yes":
                record["status"] = "refused"
            else:
                answer = strip_verdict(reply)
                result = await metric.ascore(
                    user_input=row["question"],
                    response=answer,
                    retrieved_contexts=[h.content for h in res.hits],
                )
                score = float(result.value)
                is_nan = math.isnan(score)
                record.update(
                    status="error" if is_nan else "scored",
                    score=None if is_nan else round(score, 3),
                    answer=answer[:400],
                )
        except Exception:
            logger.exception("question %s failed", row["id"])
        records.append(record)
        print(f"[{n}/{len(subset)}] {row['id']:<5} {record['status']:<8} {record['score']}",
              flush=True)
        await asyncio.sleep(PAUSE_S)
    db.close_pool()

    scored = [r["score"] for r in records if r["status"] == "scored"]
    summary = {
        "model": settings.llm_model,
        "questions": len(records),
        "scored": len(scored),
        "refused": sum(1 for r in records if r["status"] == "refused"),
        "errors": sum(1 for r in records if r["status"] == "error"),
        "mean_faithfulness": round(sum(scored) / len(scored), 3) if scored else None,
        "lowest": sorted(scored)[:3],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    if summary["errors"]:
        print("WARNING: some questions errored; re-run before trusting the mean")


if __name__ == "__main__":
    asyncio.run(main())