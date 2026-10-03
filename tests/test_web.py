import itertools
import json
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from fbrecruit import gems, scout, squad, style, web
from fbrecruit import shrinkage as sh
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT

# own players by group: in the 4-3-3, one club fills the ten and the other is a CB and the FWD
# short; in a back three with two strikers and no WIDE place, one club fills the ten and the
# other is a FB and a MID short, and both have WIDE players
OWN_FULL = {"CB": 3, "FB": 2, "MID": 4, "WIDE": 3, "FWD": 1}
OWN_SHORT = {"CB": 1, "FB": 2, "MID": 3, "WIDE": 2, "FWD": 0}
BACK_THREE = {"CB": 3, "FB": 2, "MID": 3, "WIDE": 0, "FWD": 2}
OWN_BACK_THREE_FULL = {"CB": 4, "FB": 2, "MID": 3, "WIDE": 2, "FWD": 3}
OWN_BACK_THREE_SHORT = {"CB": 3, "FB": 1, "MID": 2, "WIDE": 3, "FWD": 2}
PLAN_BUDGETS = [5_000_000, 12_000_000]
# from p10a_build_summary.log
DIGEST = "63766f9b6cc03331565bf4f6b36f050ef9b979a3ac9dbcb1982c28d506e76520"
COUNTS = {
    "players": 1247, "CB": 257, "FB": 254, "MID": 284, "WIDE": 286, "FWD": 166, "priced": 1166,
    "with a 2015 value": 1130, "with a year-later value": 1166, "clubs": 80,
}  # fmt: skip
SMALLEST = 75_000
SHAPES = {
    "2-2-2-3-1": 32, "2-2-3-2-1": 15, "2-2-2-2-2": 12, "2-2-3-1-2": 7, "3-2-2-2-1": 5,
    "2-3-2-2-1": 3, "3-2-3-0-2": 2, "2-2-3-0-3": 1, "3-1-2-3-1": 1, "3-1-3-2-1": 1,
    "3-2-2-1-2": 1,
}  # fmt: skip
BASELINES = 79
SHORT = {"serie_a-233": {"FB": 1}}
PLANS = {
    "ok": 2400, "more places than signings": 0, "too few candidates": 0, "budget too small": 0,
}  # fmt: skip
SIGNINGS = {0: 0, 1: 800, 2: 801, 3: 799}
LARGEST = 3647
FILES = (83, 8_536_224)


def test_a_band_includes_its_lower_edge():
    cases = {999_999: 0, 1_000_000: 1, 2_499_999: 1, 2_500_000: 2, 39_999_999: 5, 40_000_000: 6}
    assert {price: web.band(price) for price in cases} == cases


def test_each_reason_and_the_first_that_applies():
    need = {"CB": 1, "FB": 0, "MID": 1, "WIDE": 0, "FWD": 0}
    groups = np.array(["CB", "CB", "MID", "FWD"])
    prices = np.array([3_000_000, 4_000_000, 2_000_000, 500_000])
    no_mid = groups != "MID"
    assert web.infeasible(need, groups, prices, 1, 80_000_000) == "more places than signings"
    assert web.infeasible(need, groups[no_mid], prices[no_mid], 2, 80_000_000) == (
        "too few candidates"
    )
    # the cheapest CB and the only MID cost 5,000,000
    assert web.infeasible(need, groups, prices, 2, 4_999_999) == "budget too small"
    assert web.infeasible(need, groups, prices, 2, 5_000_000) is None
    assert web.infeasible(need, groups[no_mid], prices[no_mid], 1, 0) == (
        "more places than signings"
    )
    assert web.infeasible(need, groups[no_mid], prices[no_mid], 2, 0) == "too few candidates"


def test_largest_remainder_by_hand():
    cases = [
        # exact shares
        ((6_000, 6_000, 6_000, 9_000, 3_000), (2, 2, 2, 3, 1)),
        # 2.333, 2.167, 2.567, 1.933 and 1: WIDE and MID take the two places left
        ((7_000, 6_500, 7_700, 5_800, 3_000), (2, 2, 3, 2, 1)),
        # 3, 2, 2.5, 0.5 and 2: MID and WIDE tie for the last place and MID takes it
        ((9_000, 6_000, 7_500, 1_500, 6_000), (3, 2, 3, 0, 2)),
        # 3.333 three times: a three-way tie goes to CB
        ((3, 3, 3, 0, 0), (4, 3, 3, 0, 0)),
    ]
    for minutes, want in cases:
        shape = web.shape_of(dict(zip(sh.OUTFIELD, minutes, strict=True)))
        assert shape == dict(zip(sh.OUTFIELD, want, strict=True))


def test_a_shape_has_ten_places_each_within_one_of_its_share():
    rng = np.random.default_rng(0)
    for _ in range(500):
        minutes = rng.integers(0, 4_001, 5) * (rng.random(5) < 0.7)
        if not minutes.any():
            minutes[rng.integers(5)] = rng.integers(1, 4_001)
        shape = web.shape_of(dict(zip(sh.OUTFIELD, minutes, strict=True)))
        total = int(minutes.sum())
        assert list(shape) == sh.OUTFIELD and all(type(n) is int for n in shape.values())
        assert sum(shape.values()) == 10
        for g, m in zip(sh.OUTFIELD, minutes, strict=True):
            assert abs(shape[g] - Fraction(10 * int(m), total)) < 1


def made_up_season():
    """One club's minutes and the three tables of groups: player 1 is in gems_value and in
    style_player_w2 with another group, 2 only in player_group, 3 only in style_player_w2, 4 has
    no minutes and no group, and 5 and 6 are a GK and an UNKNOWN in player_group."""
    club = {"league": "la_liga", "team_id": 1}
    season = pd.DataFrame(
        {**club, "player_id": [1, 2, 3, 4, 5, 6], "minutes": [900, 600, 300, 0, 3_420, 200]}
    )
    value = pd.DataFrame({**club, "player_id": [1], "group": ["MID"], "minutes": [900]})
    w1 = pd.DataFrame({**club, "player_id": [2, 5, 6], "group": ["CB", "GK", "UNKNOWN"]})
    w2 = pd.DataFrame({**club, "player_id": [1, 3], "group": ["WIDE", "FB"]})
    return season, value, w1, w2


def test_a_player_team_takes_the_first_group_that_holds_its_key():
    season, value, w1, w2 = made_up_season()
    rows = web.season_groups(season, value, w1, w2)
    assert rows.set_index("player_id")[["group", "source"]].to_dict("index") == {
        1: {"group": "MID", "source": "gems_value"},
        2: {"group": "CB", "source": "player_group"},
        3: {"group": "FB", "source": "style_player_w2"},
        5: {"group": "GK", "source": "player_group"},
        6: {"group": "UNKNOWN", "source": "player_group"},
    }
    # 600 CB, 300 FB and 900 MID minutes give shares of 3.333, 1.667 and 5; the GK's and the
    # UNKNOWN's minutes count toward no group
    minutes = rows.groupby("group").minutes.sum().reindex(sh.GROUPS, fill_value=0)
    assert web.shape_of(minutes) == {"CB": 3, "FB": 2, "MID": 5, "WIDE": 0, "FWD": 0}
    extra = pd.DataFrame({"league": ["la_liga"], "team_id": [1], "player_id": [7], "minutes": [90]})
    with pytest.raises(AssertionError, match="minutes and no group"):
        web.season_groups(pd.concat([season, extra], ignore_index=True), value, w1, w2)


def made_up_club(seed, sizes):
    """Own players of the given group sizes and four candidates per group, from a generator
    seeded seed: values from a normal, fit scores from a standard normal, prices in whole
    euros."""
    rng = np.random.default_rng(seed)
    own_groups = np.repeat(list(sizes), list(sizes.values()))
    groups = np.repeat(list(squad.QUOTAS), 4)
    own = pd.DataFrame(
        {
            "player_id": np.arange(len(own_groups)),
            "group": own_groups,
            "P3_full": rng.normal(0.0, 0.1, len(own_groups)),
            "P3_sd": rng.uniform(0.02, 0.06, len(own_groups)),
        }
    )
    cands = pd.DataFrame(
        {
            "player_id": 100 + np.arange(len(groups)),
            "group": groups,
            "P3_full": rng.normal(0.05, 0.1, len(groups)),
            "P3_sd": rng.uniform(0.02, 0.06, len(groups)),
            "price": rng.integers(500_000, 8_000_001, len(groups)),
            "score": rng.normal(0.0, 1.0, len(groups)),
        }
    )
    return own, cands


def best_total(own, cands, budget, k, shape):
    """The highest summed P3_full over every set of at most k candidates within the budget, the
    places of the shape they leave filled by the own players of highest P3_full, who cost
    nothing; None if no set fills the shape."""
    values, prices, groups = (cands[c].to_numpy() for c in ("P3_full", "price", "group"))
    top = {g: np.sort(own.P3_full[own.group == g].to_numpy())[::-1] for g in shape}
    best = None
    for n in range(k + 1):
        for s in map(list, itertools.combinations(range(len(cands)), n)):
            if prices[s].sum() > budget:
                continue
            left = {g: q - int((groups[s] == g).sum()) for g, q in shape.items()}
            if any(m < 0 or m > len(top[g]) for g, m in left.items()):
                continue
            total = values[s].sum() + sum(top[g][:m].sum() for g, m in left.items())
            best = total if best is None else max(best, total)
    return best


@pytest.mark.parametrize(
    "seed, sizes, shape, short",
    [
        (0, OWN_FULL, squad.QUOTAS, {}),
        (1, OWN_SHORT, squad.QUOTAS, {"CB": 1, "FWD": 1}),
        (2, OWN_BACK_THREE_FULL, BACK_THREE, {}),
        (3, OWN_BACK_THREE_SHORT, BACK_THREE, {"FB": 1, "MID": 1}),
    ],
)
def test_plans_match_brute_force(seed, sizes, shape, short):
    own, cands = made_up_club(seed, sizes)
    base, got, plans = web.club_plans(own, cands, shape, PLAN_BUDGETS)
    assert got == short
    if short:
        assert base is None
    else:
        best = [own[own.group == g].nlargest(n, "P3_full") for g, n in shape.items()]
        assert sorted(base.player_id) == sorted(pd.concat(best).player_id)
    order = [(b, k, f) for b in PLAN_BUDGETS for k in web.CAPS for f in (False, True)]
    assert [(p["budget"], p["cap"], p["fit"]) for p in plans] == order
    for p in plans:
        c = cands[cands.score > 0] if p["fit"] else cands
        best = best_total(own, c, p["budget"], p["cap"], shape)
        rows = p["rows"]
        if best is None:
            assert rows is None and p["reason"] in web.REASONS
            continue
        assert p["reason"] is None and abs(rows.P3_full.sum() - best) <= 1e-9
        # all five groups are counted, so in the back three no WIDE player is picked
        assert {g: int((rows.group == g).sum()) for g in sh.OUTFIELD} == shape
        signed = rows[~rows.player_id.isin(own.player_id)]
        assert signed.player_id.isin(c.player_id).all()
        assert len(signed) <= p["cap"] and signed.price.sum() <= p["budget"]
        assert not p["fit"] or (signed.score > 0).all()
        assert (rows.price[rows.player_id.isin(own.player_id)] == 0).all()


def test_gain_and_its_interval_come_from_the_players_moved():
    base = pd.DataFrame(
        {"player_id": [1, 2, 3], "P3_full": [0.9, 0.2, 0.1], "P3_sd": [0.05, 0.1, 0.2]}
    )
    rows = pd.DataFrame(
        {"player_id": [1, 4, 5], "P3_full": [0.9, 0.6, 0.3], "P3_sd": [0.05, 0.2, 0.4]}
    )
    # in 4 and 5, out 2 and 3: 1.8 less 1.2, and the root of 0.04 + 0.16 + 0.01 + 0.04 is 0.5
    got = web.gain(rows, base)
    want = [0.6, 0.6 - 1.645 * 0.5, 0.6 + 1.645 * 0.5]
    assert np.allclose(got, want, rtol=0, atol=1e-12)
    plan = {"budget": 10_000_000, "cap": 2, "fit": True, "reason": None, "rows": rows}
    assert web.plan_record(plan, base, base) == {
        "budget": 10,
        "cap": 2,
        "fit": True,
        "status": "ok",
        "reason": None,
        "in": [4, 5],
        "out": [2, 3],
        "total": 1.8,
        "gain": 0.6,
        "low": -0.2225,
        "high": 1.4225,
    }


def made_up_rows():
    """Three midfielders: two priced, the first with a nickname, and one not priced with an
    empty nickname and two missing scores."""
    centred = {c: [0.12345, -0.98765, np.nan if c in style.C[1::4] else 1.5] for c in style.C}
    return pd.DataFrame(
        {
            "league": "la_liga",
            "player_id": [30, 10, 20],
            "team_id": [217, 217, 206],
            "player_name": ["Ana Uno", "Bea Dos", "Cai Tres"],
            "group": "MID",
            "minutes": [1000, 2000, 900],
            "P3_full": [0.123456789, 0.2, -0.05],
            "P3_sd": [0.01, 0.02, 0.031],
            "price": pd.array([2_500_000, 999_999, None], dtype="Int64"),
            "value_2015": pd.array([2_000_000, None, 700_000], dtype="Int64"),
            "age": [24.99, 31.0, np.nan],
            "reason": ["priced", "priced", "no summer valuation"],
            "gem": [0.01234567, -0.5, np.nan],
            "value_1": [3_086_420.0, 999_999.0, np.nan],
            "how_1": ["window", "price", np.nan],
            "nickname": ["Ana", None, ""],
            **centred,
        }
    )


def test_writing_gives_the_same_bytes_rejects_nan_and_rounds():
    obj = {"b": [1, 2.5, None, True], "a": "Kanté, €1m"}
    want = '{"b":[1,2.5,null,true],"a":"Kanté, €1m"}\n'.encode()
    assert web.dumps(obj) == web.dumps(obj) == want
    with pytest.raises(ValueError):
        web.dumps({"x": float("nan")})
    records = web.player_records(made_up_rows())
    assert list(records[0]) == [
        "id", "name", "full", "club", "group", "minutes", "age", "band", "value", "low", "high",
        "pct", "gem", "rank", "momentum", "later", "how", "style", "missing",
    ]  # fmt: skip
    common = {"group": "MID", "how": None}
    assert records == [
        common
        | {
            "id": 10,
            "name": "Bea Dos",
            "full": "Bea Dos",
            "club": "la_liga-217",
            "minutes": 2000,
            "age": 31,
            "band": 0,
            "value": 0.2,
            "low": 0.1671,
            "high": 0.2329,
            "pct": 66,
            "gem": -0.5,
            "rank": 2,
            "momentum": None,
            "later": 1.0,
            "how": "price",
            "style": [-0.988] * 8,
            "missing": [],
        },
        common
        | {
            "id": 20,
            "name": "Cai Tres",
            "full": "Cai Tres",
            "club": "la_liga-206",
            "minutes": 900,
            "age": None,
            "band": None,
            "value": -0.05,
            "low": -0.101,
            "high": 0.001,
            "pct": 0,
            "gem": None,
            "rank": None,
            "momentum": None,
            "later": None,
            "style": [1.5, 0.0, 1.5, 1.5, 1.5, 0.0, 1.5, 1.5],
            "missing": [1, 5],
        },
        common
        | {
            "id": 30,
            "name": "Ana",
            "full": "Ana Uno",
            "club": "la_liga-217",
            "minutes": 1000,
            "age": 24,
            "band": 2,
            "value": 0.1235,
            "low": 0.107,
            "high": 0.1399,
            "pct": 33,
            "gem": 0.0123,
            "rank": 1,
            "momentum": 1.25,
            "later": 1.235,
            "how": "window",
            "style": [0.123] * 8,
            "missing": [],
        },
    ]


def test_the_ceiling_check_rejects_an_amount_and_accepts_an_id():
    files = {
        "players.json": {"players": [{"id": 40_000, "minutes": 900, "band": 3}]},
        "clubs.json": {"clubs": [{"key": "a-1", "players": [40_000]}]},
        "clubs/a-1.json": {
            "fit": [[40_000, 0.5]],
            "similar": {"40000": [[40_001, 2.5, 0.5, 1]]},
            "baseline": {"players": [40_000], "total": 1.0},
            "plans": [{"budget": 80, "fit": True, "in": [40_001], "out": [40_000], "gain": 0.2}],
        },
    }
    assert web.check_numbers(files) == 900
    files["players.json"]["players"][0]["band"] = 25_000_000
    with pytest.raises(AssertionError):
        web.check_numbers(files)
    files["players.json"]["players"][0]["band"] = 3
    files["clubs/a-1.json"]["fit"][0][1] = 25_000_000
    with pytest.raises(AssertionError):
        web.check_numbers(files)


def test_the_committed_files_have_the_logged_digest():
    assert web.digest(web.read_files(web.OUT)) == DIGEST


def test_the_committed_plans_fill_each_club_shape():
    files = {path: json.loads(content) for path, content in web.read_files(web.OUT).items()}
    assert "quotas" not in files["meta.json"]
    group = {p["id"]: p["group"] for p in files["players.json"]["players"]}
    for club in files["clubs.json"]["clubs"]:
        shape = club["shape"]
        assert list(shape) == sh.OUTFIELD and sum(shape.values()) == 10
        f = files[f"clubs/{club['key']}.json"]
        if f["baseline"] is None:
            continue
        base = set(f["baseline"]["players"])
        tens = [(base - set(p["out"])) | set(p["in"]) for p in f["plans"] if p["status"] == "ok"]
        for ten in [base, *tens]:
            assert {g: sum(group[p] == g for p in ten) for g in sh.OUTFIELD} == shape


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


@pytest.mark.slow
def test_a_rebuild_gives_the_committed_bytes(tmp_path):
    tables = [gems.value_path(), gems.decision_path(), gems.scores_path(), gems.outcomes_path()]
    tables += [style.season_path(), style.team_season_path(), scout.fit_path(), scout.pairs_path()]
    tables += [PROCESSED / "minutes_player_team_statsbomb.parquet", sh.group_path()]
    tables += [style.window2_path()]
    for lg in LEAGUES:
        tables += [OUT / lg / f"{name}.parquet" for name in ("games", "teams", "lineups")]
    for path in tables:
        need(path)
    out = tmp_path / "web" / "data"
    web.build(out)
    assert web.read_files(out) == web.read_files(web.OUT)


@pytest.mark.slow
def test_the_counts_of_the_summary_log():
    cols = ["price", "value_2015", "reason"]
    decision = pd.read_parquet(need(gems.decision_path()), columns=cols)
    priced = decision[decision.reason == "priced"]
    later = pd.read_parquet(need(gems.outcomes_path()), columns=["value_1"]).value_1
    assert min(priced.price.min(), priced.value_2015.min(), later.min()) == SMALLEST
    raw = web.read_files(web.OUT)
    files = {path: json.loads(content) for path, content in raw.items()}
    players = files["players.json"]["players"]
    clubs = files["clubs.json"]["clubs"]
    got = {"players": len(players)}
    got |= {g: sum(p["group"] == g for p in players) for g in web.GROUPS}
    got |= {
        "priced": sum(p["band"] is not None for p in players),
        "with a 2015 value": sum(p["momentum"] is not None for p in players),
        "with a year-later value": sum(p["later"] is not None for p in players),
        "clubs": len(clubs),
    }
    assert got == COUNTS
    shapes = Counter("-".join(str(n) for n in club["shape"].values()) for club in clubs)
    assert shapes == SHAPES
    short, plans, made = {}, dict.fromkeys(PLANS, 0), dict.fromkeys(SIGNINGS, 0)
    for club in clubs:
        f = files[f"clubs/{club['key']}.json"]
        if f["baseline"] is None:
            short[club["key"]] = f["short"]
        for p in f["plans"]:
            plans[p["reason"] or "ok"] += 1
            if p["status"] == "ok":
                made[len(set(p["in"]) - set(club["players"]))] += 1
    assert len(clubs) - len(short) == BASELINES and short == SHORT
    assert plans == PLANS and made == SIGNINGS
    assert web.check_numbers(files) == LARGEST
    assert (len(raw), sum(len(content) for content in raw.values())) == FILES
