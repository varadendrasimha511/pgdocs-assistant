from app.generation.guard import parse_verdict, strip_verdict


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