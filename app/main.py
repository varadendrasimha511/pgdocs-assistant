import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .generation.llm import stream_answer
from .generation.prompt import SYSTEM, build_user_prompt
from .retrieval.search import Mode, search

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.open_pool()
    yield
    db.close_pool()


app = FastAPI(title="PgDocs Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    version: str | None = None
    chapter: str | None = None
    mode: Mode = "hybrid"
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
def search_endpoint(q: str, mode: Mode = "hybrid", version: str | None = None,
                    chapter: str | None = None, top_k: int = 5, rerank: bool | None = None):
    """Retrieval only. Used for debugging and for the ablation study."""
    res = search(q, mode=mode, version=version, chapter=chapter, top_k=top_k, rerank=rerank)
    return {"hits": [h.public() for h in res.hits], "timings_ms": res.timings_ms}


@app.post("/ask")
def ask(req: AskRequest):
    if not settings.anthropic_api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured")
    start = time.perf_counter()
    res = search(req.question, mode=req.mode, version=req.version,
                 chapter=req.chapter, top_k=req.top_k)

    def events():
        yield sse("sources", [h.public() for h in res.hits])
        first = None
        for token in stream_answer(SYSTEM, build_user_prompt(req.question, res.hits)):
            if first is None:
                first = (time.perf_counter() - start) * 1000
            yield sse("token", token)
        yield sse("done", {"retrieval_ms": res.timings_ms, "ttft_ms": round(first or 0, 1)})

    return StreamingResponse(events(), media_type="text/event-stream")