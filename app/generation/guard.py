import re
from collections.abc import Iterable, Iterator

VERDICT = re.compile(r"^\s*ANSWERABLE:\s*(YES|NO)\b", re.IGNORECASE)


def parse_verdict(reply: str) -> str:
    """Return 'yes', 'no' or 'unclear' from the first line of a model reply."""
    match = VERDICT.match(reply)
    return match.group(1).lower() if match else "unclear"


def strip_verdict(reply: str) -> str:
    """Remove the verdict line so only the answer is shown."""
    return VERDICT.sub("", reply, count=1).strip()


def gate_tokens(tokens: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Hold back the stream until the verdict line is complete.

    Yields ("verdict", "yes" | "no" | "unclear") first. Answer text is yielded as
    ("token", text) only when the verdict is "yes"; otherwise nothing more is yielded.
    """
    buffer = ""
    decided = False
    for token in tokens:
        if decided:
            yield ("token", token)
            continue
        buffer += token
        if "\n" not in buffer:
            continue
        verdict = parse_verdict(buffer)
        yield ("verdict", verdict)
        decided = True
        if verdict != "yes":
            return
        rest = strip_verdict(buffer)
        if rest:
            yield ("token", rest)
    if not decided:  # the stream ended before a newline arrived
        verdict = parse_verdict(buffer)
        yield ("verdict", verdict)
        if verdict == "yes":
            rest = strip_verdict(buffer)
            if rest:
                yield ("token", rest)