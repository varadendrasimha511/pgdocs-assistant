import sys

from app.observability import RequestTrace


def test_tracing_never_breaks_the_request(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)  # makes `import langfuse` fail
    trace = RequestTrace("What does VACUUM do?")
    span = trace.start("retrieval", as_type="retriever", input="q")
    assert span is None
    trace.finish(span, output="x")  # must be a silent no-op
    trace.close(output={"verdict": "yes"})


def test_trace_works_without_keys_configured(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    trace = RequestTrace("q?")
    gen = trace.start("llm", as_type="generation", model="m", input="prompt")
    trace.finish(gen, output="answer", metadata={"verdict": "yes"})
    trace.close(output={"verdict": "yes"})