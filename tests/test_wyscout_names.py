from fbrecruit.sources.wyscout import decode


def test_escape_is_decoded():
    assert decode("I. Konat\\u00e9") == "I. Konaté"


def test_text_without_escape_is_unchanged():
    assert decode("Real Sociedad") == "Real Sociedad"
    assert decode("Atlético Madrid") == "Atlético Madrid"
