import re

VERDICT = re.compile(r"^\s*ANSWERABLE:\s*(YES|NO)\b", re.IGNORECASE)


def parse_verdict(reply: str) -> str:
    """Return 'yes', 'no' or 'unclear' from the first line of a model reply."""
    match = VERDICT.match(reply)
    return match.group(1).lower() if match else "unclear"


def strip_verdict(reply: str) -> str:
    """Remove the verdict line so only the answer is shown."""
    return VERDICT.sub("", reply, count=1).strip()