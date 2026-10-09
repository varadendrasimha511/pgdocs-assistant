from collections.abc import Iterator

from ..config import settings


def stream_answer(system: str, user: str) -> Iterator[str]:
    """Thin LLM interface: swap this one function to change provider."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        yield from stream.text_stream