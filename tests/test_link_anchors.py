import pytest

from fbrecruit import cohorts
from fbrecruit.links import data, statsbomb_tm, wyscout_tm
from fbrecruit.links.teams import build_team_map, name_overlap
from fbrecruit.paths import INTERIM, PROCESSED, RAW

pytestmark = pytest.mark.slow

REQUIRED = [
    INTERIM / "transfermarkt" / "appearances.parquet",
    INTERIM / "statsbomb" / "la_liga" / "lineups.parquet",
    INTERIM / "wyscout" / "spain" / "lineups.parquet",
    PROCESSED / "minutes_player_wyscout.parquet",
    RAW / "wyscout" / "players.json",
]

STATSBOMB_RATES = {
    "la_liga": 0.9482,
    "ligue_1": 0.9688,
    "premier_league": 0.9745,
    "serie_a": 0.9797,
}
DIFFERING_PAIRS = {
    ("AS Roma", "Associazione Sportiva Roma"),
    ("Lazio", "Società Sportiva Lazio S.p.A."),
    ("Rennes", "Stade Rennais FC"),
    ("Lyon", "Olympique Lyon"),
    ("Stade Malherbe Caen", "SM Caen"),
}

# league, n1, n2, n3, n4, n5, stayers n1, stayers n3, stayers n4
FUNNEL_A = [
    ("La Liga", 417, 56, 53, 43, 28, 224, 214, 201),
    ("Ligue 1", 419, 47, 47, 35, 24, 219, 216, 192),
    ("Premier League", 386, 29, 28, 23, 20, 257, 252, 232),
    ("Serie A", 415, 54, 53, 46, 40, 227, 222, 205),
    ("ALL", 1637, 186, 181, 147, 112, 927, 904, 830),
]
FUNNEL_B = [
    ("La Liga", 416, 83, 69, 60, 51, 227, 190, 179),
    ("Ligue 1", 396, 70, 68, 59, 47, 217, 204, 192),
    ("Premier League", 383, 36, 33, 31, 20, 275, 265, 245),
    ("Serie A", 400, 85, 81, 67, 56, 210, 201, 184),
    ("Bundesliga", 348, 47, 45, 41, 34, 233, 226, 209),
    ("ALL", 1943, 321, 296, 258, 208, 1162, 1086, 1009),
]
FUNNEL_BOTH_A = [
    ("La Liga", 417, 56, 55, 45, 30, 224, 223, 210),
    ("Ligue 1", 419, 47, 47, 35, 24, 219, 219, 195),
    ("Premier League", 386, 29, 28, 23, 20, 257, 255, 235),
    ("Serie A", 415, 54, 54, 46, 40, 227, 225, 208),
    ("ALL", 1637, 186, 184, 149, 114, 927, 922, 848),
]
FUNNEL_BOTH_B = [
    ("La Liga", 416, 83, 77, 67, 57, 227, 216, 204),
    ("Ligue 1", 396, 70, 69, 60, 47, 217, 209, 195),
    ("Premier League", 383, 36, 34, 32, 21, 275, 268, 246),
    ("Serie A", 400, 85, 82, 68, 57, 210, 205, 188),
    ("Bundesliga", 348, 47, 46, 42, 35, 233, 229, 212),
    ("ALL", 1943, 321, 308, 269, 217, 1162, 1127, 1045),
]
LABELS_A = {
    "paid": 63,
    "paid_mirror": 9,
    "loan_return": 60,
    "loan_out": 32,
    "both": 4,
    "free_other": 18,
}
LABELS_B = {
    "paid": 99,
    "paid_mirror": 9,
    "loan_return": 116,
    "loan_out": 66,
    "both": 3,
    "free_other": 28,
}
# label: n3, n4, n5 with pass-1 links
LEVELS_A = {
    "paid": (63, 51, 42),
    "paid_mirror": (9, 8, 7),
    "loan_return": (58, 52, 36),
    "loan_out": (31, 23, 17),
    "both": (4, 4, 3),
    "free_other": (16, 9, 7),
}
LEVELS_B = {
    "paid": (91, 84, 77),
    "paid_mirror": (8, 8, 7),
    "loan_return": (106, 91, 69),
    "loan_out": (62, 51, 33),
    "both": (2, 2, 1),
    "free_other": (27, 22, 21),
}


@pytest.fixture(scope="module", autouse=True)
def caches():
    for path in REQUIRED:
        if not path.exists():
            pytest.skip(f"local data cache missing: {path}")


@pytest.fixture(scope="module")
def appearances():
    return data.tm_appearances()


@pytest.fixture(scope="module")
def team_map():
    return build_team_map()


@pytest.fixture(scope="module")
def sb_pass1(appearances, team_map):
    cands = statsbomb_tm.candidates(appearances)
    return cands, statsbomb_tm.match_pass1(data.statsbomb_players(), team_map, cands)


@pytest.fixture(scope="module")
def sb_final(sb_pass1):
    cands, pass1 = sb_pass1
    return statsbomb_tm.match_pass2(pass1, cands)


@pytest.fixture(scope="module")
def wy_pass1():
    by_dob = wyscout_tm.tm_by_dob()
    links = wyscout_tm.link_pass1(wyscout_tm.wyscout_players(), by_dob)
    links = links.merge(wyscout_tm.wyscout_minutes(), on="wy_id", how="left")
    return by_dob, links.assign(minutes=links.minutes.fillna(0))


@pytest.fixture(scope="module")
def wy_final(wy_pass1):
    by_dob, pass1 = wy_pass1
    return wyscout_tm.link_pass2(pass1, by_dob)


def test_team_map(team_map):
    assert len(team_map) == 80
    assert (team_map.groupby("league").tm_club_id.nunique() == 20).all()
    assert int((team_map.similarity < 0.85).sum()) == 20
    differing = team_map[team_map.differs_from_best]
    assert set(zip(differing.sb_team, differing.tm_club, strict=True)) == DIFFERING_PAIRS


def test_team_overlap(team_map, appearances):
    ov = name_overlap(team_map, data.statsbomb_players(), appearances)
    assert (ov.overlap.min(), ov.overlap.median(), ov.overlap.max()) == (19, 26, 35)
    assert not (ov.max_other_overlap > ov.overlap).any()


def test_statsbomb_pass1(sb_pass1):
    _, links = sb_pass1
    counts = links.status.value_counts()
    assert (len(links), counts["matched"], counts["unmatched"]) == (2279, 2182, 97)
    assert "ambiguous" not in counts
    assert round(statsbomb_tm.rate(links), 4) == 0.9678
    for league, expected in STATSBOMB_RATES.items():
        assert round(statsbomb_tm.rate(links[links.league == league]), 4) == expected
    big = links[links.minutes >= 450]
    assert (len(big), int((big.status == "matched").sum())) == (1685, 1629)
    assert round(statsbomb_tm.rate(big), 4) == 0.9688
    matched = links[links.status == "matched"]
    assert int((matched.score < 1.0).sum()) == 30
    assert int(((links.status == "unmatched") & (links.minutes >= 900)).sum()) == 44


def test_wyscout_pass1_all_players(wy_pass1):
    _, links = wy_pass1
    counts = links.status.value_counts().to_dict()
    assert len(links) == 3603
    assert counts == {
        "linked": 3260,
        "below_threshold": 303,
        "no_candidate_same_dob": 26,
        "ambiguous": 14,
    }
    linked = links[links.status == "linked"]
    assert int((linked.best_score == 1.0).sum()) == 2677
    assert not linked.tm_player_id.duplicated().any()


def test_wyscout_pass1_450_plus(wy_pass1):
    _, links = wy_pass1
    big = links[links.minutes >= 450]
    assert len(big) == 1974
    assert big.status.value_counts().to_dict() == {
        "linked": 1844,
        "below_threshold": 121,
        "ambiguous": 6,
        "no_candidate_same_dob": 3,
    }
    rates = wyscout_tm.rates_table(links).set_index("league")
    assert round(rates.loc["ALL", "rate"], 4) == 0.9341
    assert round(rates.loc["ALL", "minutes_weighted_rate"], 4) == 0.9309
    assert round(rates.loc["spain", "rate"], 4) == 0.8416
    others = rates.drop(["spain", "ALL"]).rate
    assert (round(others.min(), 4), round(others.max(), 4)) == (0.9505, 0.9715)


def test_wyscout_minutes_agreement(wy_pass1, appearances):
    _, links = wy_pass1
    agree = wyscout_tm.minutes_agreement(links, appearances)
    assert len(agree) == 1844
    assert int((agree.tm_minutes == 0).sum()) == 0
    assert round(agree.ratio.median(), 3) == 0.949
    assert round(((agree.ratio >= 0.8) & (agree.ratio <= 1.2)).mean(), 3) == 0.999


def links_for(df, status):
    return df[df.status == status][["tm_player_id", "minutes", "link_pass"]].astype(
        {"tm_player_id": int}
    )


@pytest.fixture(scope="module")
def transfers():
    return data.tm_table("transfers").rename(columns={"transfer_date": "date"})


@pytest.fixture(scope="module")
def cohort_a(appearances, transfers, sb_final):
    return cohorts.build("A", appearances, transfers, links_for(sb_final, "matched"), legacy=True)


@pytest.fixture(scope="module")
def cohort_b(appearances, transfers, wy_final):
    return cohorts.build("B", appearances, transfers, links_for(wy_final, "linked"), legacy=True)


@pytest.fixture(scope="module")
def corrected_a(appearances, transfers, sb_final):
    return cohorts.build("A", appearances, transfers, links_for(sb_final, "matched"))


@pytest.fixture(scope="module")
def corrected_b(appearances, transfers, wy_final):
    return cohorts.build("B", appearances, transfers, links_for(wy_final, "linked"))


def rows(table):
    return [tuple(r) for r in table.itertuples(index=False)]


def per_label(mv):
    grouped = mv.groupby("mirror_label")[["n3_p1", "n4_p1", "n5_p1"]].sum()
    return {label: tuple(int(v) for v in r) for label, r in grouped.iterrows()}


@pytest.mark.parametrize(
    ("name", "funnel", "labels", "levels", "june30", "july1", "returned"),
    [
        ("A", FUNNEL_A, LABELS_A, LEVELS_A, 64, 34, 26),
        ("B", FUNNEL_B, LABELS_B, LEVELS_B, 118, 54, 37),
    ],
)
def test_cohort_anchors(request, name, funnel, labels, levels, june30, july1, returned):
    mv, tables = request.getfixturevalue(f"cohort_{name.lower()}")
    assert rows(tables["pass 1"]) == funnel
    assert mv.mirror_label.value_counts().to_dict() == labels
    assert per_label(mv) == levels
    assert int(mv.returned.sum()) == returned
    day = mv.date.dt.strftime("%m-%d")
    assert (int((day == "06-30").sum()), int((day == "07-01").sum())) == (june30, july1)


@pytest.mark.parametrize(
    ("name", "funnel", "funnel_both", "labels", "levels", "permanent", "sensitivity"),
    [
        ("A", FUNNEL_A, FUNNEL_BOTH_A, LABELS_A, LEVELS_A, 88, 32),
        ("B", FUNNEL_B, FUNNEL_BOTH_B, LABELS_B, LEVELS_B, 136, 66),
    ],
)
def test_corrected_mover_selection(
    request, name, funnel, funnel_both, labels, levels, permanent, sensitivity
):
    mv, tables = request.getfixturevalue(f"corrected_{name.lower()}")
    assert rows(tables["pass 1"]) == funnel
    assert rows(tables["passes 1 and 2"]) == funnel_both
    assert mv.mirror_label.value_counts().to_dict() == labels
    assert per_label(mv) == levels
    assert (int(mv.permanent.sum()), int(mv.sensitivity.sum())) == (permanent, sensitivity)
