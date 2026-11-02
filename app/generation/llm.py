from collections.abc import Iterator

from ..config import settings

# Providers that expose an OpenAI-compatible API
BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def _openai_compatible(system: str, user: str) -> Iterator[str]:
    from openai import OpenAI

    base_url = settings.llm_base_url or BASE_URLS[settings.llm_provider]
    # max_retries: the client backs off and retries automatically on 429 rate-limit errors
    client = OpenAI(api_key=settings.llm_api_key or "none", base_url=base_url, max_retries=3)
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=1000,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _anthropic(system: str, user: str) -> Iterator[str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.llm_api_key)
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        yield from stream.text_stream


def stream_answer(system: str, user: str) -> Iterator[str]:
    """One function the rest of the app calls. Switch providers with LLM_PROVIDER in .env."""
    if settings.llm_provider == "anthropic":
        yield from _anthropic(system, user)
    else:
        yield from _openai_compatible(system, user)