from app.generation.guard import gate_tokens, parse_verdict, strip_verdict


def test_yes_and_no_are_parsed_case_insensitively():
    assert parse_verdict("ANSWERABLE: YES\nVACUUM reclaims space [1].") == "yes"
    assert parse_verdict("answerable: no") == "no"
    assert parse_verdict("\n  ANSWERABLE: NO\n") == "no"


def test_missing_or_misplaced_verdict_is_unclear():
    assert parse_verdict("Sure! ANSWERABLE: YES") == "unclear"
    assert parse_verdict("") == "unclear"
    assert parse_verdict("The docs say...") == "unclear"


def test_strip_verdict_keeps_only_the_answer():
    assert strip_verdict("ANSWERABLE: YES\nUse COPY [1].") == "Use COPY [1]."


def test_yes_streams_the_answer_after_the_verdict_line():
    out = list(gate_tokens(["ANSWER", "ABLE: YES\n", "Use ", "COPY [1]."]))
    assert out == [("verdict", "yes"), ("token", "Use "), ("token", "COPY [1].")]


def test_no_verdict_blocks_all_text():
    assert list(gate_tokens(["ANSWERABLE: NO\n", "made up"])) == [("verdict", "no")]


def test_missing_verdict_is_unclear_and_blocked():
    assert list(gate_tokens(["Sure, here is the answer\n", "..."])) == [("verdict", "unclear")]


def test_stream_without_newline_is_still_judged():
    assert list(gate_tokens(["ANSWERABLE: NO"])) == [("verdict", "no")]
    assert list(gate_tokens(["ANSWERABLE: YES"])) == [("verdict", "yes")]