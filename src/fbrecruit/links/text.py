import difflib
import re
import unicodedata

import pandas as pd

SPECIAL = str.maketrans(
    {"ø": "o", "đ": "d", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe", "ð": "d", "þ": "th", "ı": "i"}
)
CLUB_STOP = {
    "fc", "cf", "afc", "ac", "sc", "as", "us", "ss", "ssc", "ud", "cd", "sd", "rc", "rcd", "sv",
    "fk", "calcio", "spa", "sad", "club", "de",
}  # fmt: skip


def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).replace("\xa0", " ").lower().translate(SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_club(s):
    return " ".join(t for t in norm(s).split() if t not in CLUB_STOP)


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def tokens_contained(name, *others, min_tokens=1):
    """True when every token of name is among the tokens of the other strings."""
    wanted = set(norm(name).split())
    pool = {t for o in others for t in norm(o).split()}
    return len(wanted) >= min_tokens and wanted <= pool
