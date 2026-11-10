import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .generation.guard import gate_tokens
from .generation.llm import stream_answer
from .generation.prompt import GUARDED_SYSTEM, build_user_prompt
from .retrieval.search import Mode, search

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger("pgdocs")

REFUSAL = (
    "I couldn't find this in the PostgreSQL documentation, so I won't guess. "
    "Try rephrasing, or ask about a specific command or setting."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.open_pool()
    search("warm up", mode="vector", rerank=False, top_k=1)  # load the embedding model once
    yield
    db.close_pool()


app = FastAPI(title="PgDocs Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    version: str | None = None
    chapter: str | None = None
    mode: Mode = "vector"
    top_k: int = Field(5, ge=1, le=10)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/search")
def search_endpoint(
    q: str,
    mode: Mode = "hybrid",
    version: str | None = None,
    chapter: str | None = None,
    top_k: int = 5,
    rerank: bool | None = None,
):
    """Retrieval only. Used for debugging and for the ablation study."""
    res = search(q, mode=mode, version=version, chapter=chapter, top_k=top_k, rerank=rerank)
    return {"hits": [h.public() for h in res.hits], "timings_ms": res.timings_ms}


@app.post("/ask")
def ask(req: AskRequest):
    if not settings.llm_api_key and settings.llm_provider != "ollama":
        raise HTTPException(503, "LLM_API_KEY is not configured")
    start = time.perf_counter()
    res = search(
        req.question,
        mode=req.mode,
        version=req.version,
        chapter=req.chapter,
        top_k=req.top_k,
        rerank=False,
    )
    prompt = build_user_prompt(req.question, res.hits)

    def events():
        first = None
        verdict = None
        try:
            for kind, value in gate_tokens(stream_answer(GUARDED_SYSTEM, prompt)):
                if kind == "verdict":
                    verdict = value
                    if value == "yes":  # sources are shown only for answers
                        yield sse("sources", [h.public() for h in res.hits])
                    continue
                if first is None:
                    first = (time.perf_counter() - start) * 1000
                yield sse("token", value)
        except Exception as e:
            logger.exception("LLM stream failed")
            yield sse("error", f"{type(e).__name__}: {e}")
        else:
            if verdict != "yes":
                logger.info("refused (verdict=%s): %r", verdict, req.question[:80])
                yield sse("refusal", REFUSAL)
        yield sse(
            "done",
            {
                "verdict": verdict,
                "retrieval_ms": res.timings_ms,
                "ttft_ms": round(first or 0, 1),
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream")