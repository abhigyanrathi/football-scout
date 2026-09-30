import numpy as np
import pandas as pd
import pytest

from fbrecruit import identify, scout, style
from fbrecruit.paths import PROCESSED
from fbrecruit.shrinkage import OUTFIELD
from fbrecruit.sources.statsbomb import LEAGUES, OUT
from fbrecruit.split import windows

DAYS = 38
KEYS = scout.KEYS
RAW = [*style.DIMENSIONS, *[f"n_{d}" for d in style.DIMENSIONS]]
# from p9b_season_summary.log
TOOLS_GROUPS = {"CB": 257, "FB": 254, "MID": 284, "WIDE": 286, "FWD": 166}
TOOLS_LEAGUES = {"la_liga": 318, "premier_league": 299, "serie_a": 317, "ligue_1": 313}
HALF_ROWS = 4988
DEALING_ROWS = 3034
# from p9b_tools_summary.log: per split and group, the reference rows and their median u
REFERENCE = {
    ("venue", "CB"): (244, 50.89171422595),
    ("venue", "FB"): (232, 47.308841608322524),
    ("venue", "MID"): (274, 44.226114408195954),
    ("venue", "WIDE"): (264, 43.505326798033295),
    ("venue", "FWD"): (154, 51.90841576582536),
    ("parity", "CB"): (243, 51.61282606567464),
    ("parity", "FB"): (238, 45.566039161489016),
    ("parity", "MID"): (271, 42.77442094131743),
    ("parity", "WIDE"): (260, 45.47683986291146),
    ("parity", "FWD"): (155, 50.360427875652235),
}
PAIRS = {"CB": 65155, "FB": 63631, "MID": 79536, "WIDE": 80559, "FWD": 27128}
FEWEST = 161
# fit rows and the fewest and most candidates of a club
FIT = (98450, 1225, 1235)
# from p9b_beside_summary.log, 6f: every value of the results entries
ENTRIES = {
    "P": 1247, "P_CB": 257, "P_FB": 254, "P_MID": 284, "P_WIDE": 286, "P_FWD": 166, "K": 80,
    "Q": 1168, "QP": 1167,
    "UV_CB": 50.89171422595, "UP_CB": 51.61282606567464, "RT_CB": 0.9906826322979028,
    "UV_FB": 47.308841608322524, "UP_FB": 45.566039161489016, "RT_FB": 1.0221337273091782,
    "UV_MID": 44.226114408195954, "UP_MID": 42.77442094131743, "RT_MID": 0.9686376039504566,
    "UV_WIDE": 43.505326798033295, "UP_WIDE": 45.47683986291146, "RT_WIDE": 0.9974958326817187,
    "UV_FWD": 51.90841576582536, "UP_FWD": 50.360427875652235, "RT_FWD": 1.0109530784535037,
    "NN": 0.7142857142857143,
    "SL_C": 0.2440256615878108, "SL_U": 0.3006415396952687, "SL_0": 0.24226947742117638,
    "BL_1": 0.17789260840275709, "BL_2": 0.021324950013683144, "BL_3": 0.05688135161356365,
    "BL_4": 0.019417323481564087, "BL_5": 0.02254657262687911, "BL_6": 0.12115904151049708,
    "BL_7": 0.2846760533480407, "BL_8": 0.2725150554579439,
    "G1": 0.30303030303030304, "G1X": 1.0, "G1H": 32,
    "G2": 0.4642857142857143, "G2X": 1.0, "G2H": 25,
    "GS": 0.35964912280701755, "GSX": 1.0, "GSH": 23,
    "VA_1": 0.804344967616759, "VB_1": 0.7538743592367398, "VC_1": 0.9217700514290122,
    "VA_2": 0.8531557633518276, "VB_2": 0.7693790737699661, "VC_2": 0.9492946581670975,
    "VA_3": 0.8304124081200063, "VB_3": 0.7921420511986448, "VC_3": 0.8956540766805534,
    "VA_4": 0.895055408481624, "VB_4": 0.8881935623459312, "VC_4": 0.869258991284655,
    "VA_5": 0.818468307545073, "VB_5": 0.7785699462556505, "VC_5": 0.8926173065348505,
    "VA_6": 0.5585599895030645, "VB_6": 0.42243702650854653, "VC_6": 0.8309298698948381,
    "VA_7": 0.746878276454618, "VB_7": 0.6910923429788552, "VC_7": 0.9374471284985497,
    "VA_8": 0.4587961189950392, "VB_8": 0.3786516464303156, "VC_8": 0.8096112601746903,
}  # fmt: skip


def club_games(venues, windows=None):
    """Club 1's games, one per matchday from 1, against a different club each time; venues is a
    string of H and A."""
    day = np.arange(1, len(venues) + 1)
    home = np.array([v == "H" for v in venues])
    return pd.DataFrame(
        {
            "league": "x",
            "game_id": 1000 + day,
            "game_day": day,
            "window": np.where(day <= 23, 1, 2) if windows is None else windows,
            "home_team_id": np.where(home, 1, 100 + day),
            "away_team_id": np.where(home, 100 + day, 1),
        }
    )


def club_one(dealing):
    return dealing[dealing.team_id == 1].reset_index(drop=True)


def shuffled_venues(home, away):
    order = np.random.default_rng(0).permutation(np.array(["H"] * home + ["A"] * away))
    return "".join(order)


def test_nineteen_home_and_nineteen_away_games_give_halves_of_nineteen():
    d = club_one(style.deal(club_games(shuffled_venues(19, 19))))
    assert d.game_day.tolist() == list(range(1, DAYS + 1))
    assert d.half_season.value_counts().to_dict() == {"A": 19, "B": 19}
    home = d[d.home]
    assert home.half_season.value_counts().to_dict() == {"A": 10, "B": 9}
    assert home.half_season.tolist() == ["A", "B"] * 9 + ["A"]
    assert d[~d.home].half_season.tolist() == ["B", "A"] * 9 + ["B"]


def test_nineteen_home_and_eighteen_away_games_give_halves_of_nineteen_and_eighteen():
    d = club_one(style.deal(club_games(shuffled_venues(19, 18))))
    assert d.half_season.value_counts().to_dict() == {"A": 19, "B": 18}


def test_a_window_is_dealt_again_from_its_first_game():
    games = club_games("HHHAHA", windows=[1, 1, 1, 1, 2, 2])
    d = club_one(style.deal(games))
    assert d.half_season.tolist() == ["A", "B", "A", "B", "B", "A"]
    assert d.half_window.tolist() == ["A", "B", "A", "B", "A", "B"]


def test_games_are_taken_in_matchday_order_then_game_id():
    games = club_games("HHH").iloc[[2, 0, 1]]
    games = pd.concat([games, games.iloc[[1]].assign(game_id=999)], ignore_index=True)
    d = club_one(style.deal(games))
    # matchday 1 holds games 999 and 1001: 999 first
    assert d.game_id.tolist() == [999, 1001, 1002, 1003]
    assert d.half_season.tolist() == ["A", "B", "A", "B"]


def parts(values, counts):
    """A frame with every dimension set to values and every n_ to counts."""
    columns = {d: values for d in style.DIMENSIONS}
    columns |= {f"n_{d}": counts for d in style.DIMENSIONS}
    return pd.DataFrame(columns, dtype=np.float64)


def test_a_missing_part_or_an_n_of_0_adds_nothing_and_a_zero_total_gives_nan():
    first = parts([1.0, 5.0, np.nan, np.nan], [2.0, 0.0, np.nan, np.nan])
    second = parts([4.0, 3.0, 7.0, np.nan], [1.0, 3.0, 2.0, np.nan])
    got = style.pooled(first, second)
    for d in style.DIMENSIONS:
        assert np.array_equal(got[d], [2.0, 3.0, 7.0, np.nan], equal_nan=True)
        assert got[f"n_{d}"].tolist() == [3.0, 3.0, 2.0, 0.0]


def test_identity_gaps_are_zero_for_a_total_and_its_parts():
    first = parts([1.0, 5.0, np.nan], [2.0, 4.0, np.nan])
    second = parts([4.0, np.nan, 7.0], [1.0, np.nan, 2.0])
    whole = parts([2.0, 5.0, 7.0], [3.0, 4.0, 2.0])
    gaps = style.identity_gaps(whole, first, second)
    assert {kind: len(g) for kind, g in gaps.items()} == {"n_": 24, "values": 18, "one part": 16}
    assert all(g.max() == 0.0 for g in gaps.values())
    whole.loc[1, "sb_pressing"] = 6.0
    # sb_pressing is left out of the values, but a row in one part must have that part's value
    assert style.identity_gaps(whole, first, second)["one part"].max() == 1 / 6


def scored(frame, rng):
    z = rng.normal(size=(len(frame), len(style.DIMENSIONS)))
    z[0, 2] = np.nan
    return frame.assign(**dict(zip(style.Z, z.T, strict=True)))


def test_centred_scores_average_zero_within_each_league_and_group():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"league": ["a", "a", "a", "b", "b", "a"], "group": ["CB", "CB", "FB", "CB", "CB", "CB"]}
    )
    frame = scored(frame, rng)
    by = ["league", "group"]
    c = style.centred(frame, frame.groupby(by)[style.Z].mean(), by)
    means = c.groupby(by)[style.C].mean()
    assert np.allclose(means.to_numpy(), 0.0, rtol=0, atol=1e-15)
    # a missing standard score stays missing, and its centre leaves it out
    assert np.isnan(c.loc[0, style.C[2]])
    centre = frame.loc[[1, 5], style.Z[2]].mean()
    assert np.isclose(c.loc[1, style.C[2]], frame.loc[1, style.Z[2]] - centre, rtol=0, atol=1e-15)


def test_a_missing_score_counts_as_0_in_distances_and_products():
    season = pd.DataFrame(
        {
            "league": "a",
            "player_id": [1, 2],
            "team_id": [10, 20],
            "group": "MID",
            "minutes": [900, 1600],
            **{c: [1.0, 2.0] for c in style.C},
            **{z: [0.5, 1.5] for z in style.Z},
        }
    )
    season.loc[0, "c_width"] = np.nan
    t = season[scout.KEYS]
    ref = pd.DataFrame({"group": "MID", "u": [10.0]})
    pr = scout.pairs(season, t, ref)
    # seven dimensions 1 apart and width 0 against 2
    assert np.allclose(pr.distance, np.sqrt(7 + 4), rtol=0, atol=1e-15)
    teams = pd.DataFrame({"league": "a", "team_id": [30], **{c: [3.0] for c in style.C}})
    fits = scout.fit(season, teams, t)
    assert fits.loc[fits.player_id == 1, "p_width"].tolist() == [0.0]
    assert fits.loc[fits.player_id == 1, "p_depth"].tolist() == [3.0]


def test_closeness_counts_reference_values_at_or_above_v():
    u = [3.0, 1.0, 2.0, 2.0]
    got = scout.closeness(u, np.array([2.0, 0.5, 3.5, 3.0]))
    assert got.tolist() == [0.75, 1.0, 0.0, 0.25]


def test_a_candidate_with_any_row_at_the_club_is_left_out():
    season = pd.DataFrame(
        {
            "league": "a",
            "player_id": [1, 2, 3, 4],
            "team_id": [10, 20, 30, 10],
            "group": "FB",
            "minutes": 1000,
            **{c: [0.0, 1.0, 2.0, 3.0] for c in style.C},
            **{z: [0.0, 1.0, 2.0, 3.0] for z in style.Z},
        }
    )
    # player 3 also played for club 10, and player 2 for a club of another league
    t = pd.concat(
        [
            season[scout.KEYS],
            pd.DataFrame({"league": ["a", "b"], "player_id": [3, 2], "team_id": [10, 10]}),
        ],
        ignore_index=True,
    )
    ref = pd.DataFrame({"group": "FB", "u": [1.0, 2.0]})
    pr = scout.pairs(season, t, ref)
    assert pr[pr.player_id == 1].player_id_c.tolist() == [2]
    assert pr[pr.player_id == 4].player_id_c.tolist() == [2]
    assert pr[pr.player_id == 2].player_id_c.tolist() == [1, 3, 4]
    teams = pd.DataFrame({"league": ["a", "a"], "team_id": [10, 20], **{c: 1.0 for c in style.C}})
    fits = scout.fit(season, teams, t)
    assert fits[fits.team_id_k == 10].player_id.tolist() == [2]
    assert fits[fits.team_id_k == 20].player_id.tolist() == [1, 3, 4]


def test_the_fit_score_weighs_possession_buildup_and_verticality_a_third_each():
    p = pd.DataFrame(np.eye(len(style.DIMENSIONS)), columns=scout.P)
    third = {"possession", "buildup", "verticality"}
    expected = [1 / 18 if d in third else 1 / 6 for d in style.DIMENSIONS]
    assert np.allclose(scout.fit_score(p), expected, rtol=0, atol=1e-15)
    assert np.isclose(scout.fit_score(pd.DataFrame([[1.0] * 8], columns=scout.P))[0], 1.0)


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def stored(path):
    return pd.read_parquet(need(path))


def minutes_table():
    need(PROCESSED / "player_window_vaep_pooled_v2.parquet")
    return identify.window_minutes()


def season_games():
    for lg in LEAGUES:
        need(OUT / lg / "games.parquet")
    return style.season_games()


@pytest.mark.slow
def test_tools_rows_clubs_halves_and_dealing():
    season = stored(style.season_path())
    assert season.group.value_counts().to_dict() == TOOLS_GROUPS
    assert season.league.value_counts().to_dict() == TOOLS_LEAGUES
    assert len(stored(style.team_season_path())) == len(stored(style.team_w2_path())) == 80
    assert len(stored(style.halves_path())) == HALF_ROWS
    dealing = stored(style.dealing_path())
    assert len(dealing) == DEALING_ROWS
    pd.testing.assert_frame_equal(style.deal(season_games()), dealing)


@pytest.mark.slow
def test_reference_counts_and_median_u():
    ref = stored(scout.reference_path())
    pd.testing.assert_frame_equal(scout.reference(stored(style.halves_path())), ref)
    for (split, group), (n, median) in REFERENCE.items():
        u = ref.u[(ref.split == split) & (ref.group == group)]
        assert (len(u), u.median()) == (n, median)


@pytest.mark.slow
def test_pairs_and_fit_recomputed_from_the_stored_tables():
    season, t = stored(style.season_path()), minutes_table()
    ref, pr = stored(scout.reference_path()), stored(scout.pairs_path())
    pd.testing.assert_frame_equal(scout.pairs(season, t, ref[ref.split == "venue"]), pr)
    assert pr.group.value_counts().to_dict() == PAIRS
    assert pr.groupby("player_id").size().min() == FEWEST
    fits = stored(scout.fit_path())
    pd.testing.assert_frame_equal(scout.fit(season, stored(style.team_season_path()), t), fits)
    per = fits.groupby(["league_k", "team_id_k"]).size()
    assert (len(fits), per.min(), per.max()) == FIT


@pytest.mark.slow
def test_the_values_of_the_results_entries():
    season, teams = stored(style.season_path()), stored(style.team_season_path())
    ref, pr = stored(scout.reference_path()), stored(scout.pairs_path())
    wins = windows()[["league", "game_id", "window"]]
    venue = scout.venue_gaps(stored(style.dealing_path()).merge(wins, on=["league", "game_id"]))
    pd.testing.assert_frame_equal(venue, stored(scout.venue_path()))
    ratios = scout.minutes_thirds(ref, season)
    shares = scout.league_shares(pr)
    between = scout.club_shares(teams)
    got = {"P": len(season), **{f"P_{g}": int((season.group == g).sum()) for g in OUTFIELD}}
    got |= {"K": len(teams), "Q": int((ref.split == "venue").sum())}
    got["QP"] = int((ref.split == "parity").sum())
    for g in OUTFIELD:
        got[f"UV_{g}"] = ref.u[(ref.split == "venue") & (ref.group == g)].median()
        got[f"UP_{g}"] = ref.u[(ref.split == "parity") & (ref.group == g)].median()
        got[f"RT_{g}"] = ratios[g]
    got["NN"] = scout.nearest(pr, "distance", 1).closeness.median()
    got |= {"SL_C": shares["centred"], "SL_U": shares["uncentred"], "SL_0": shares["all"]}
    got |= {f"BL_{i}": between[d] for i, d in enumerate(style.DIMENSIONS, 1)}
    for name, scope in (("G1", "window 1"), ("G2", "window 2"), ("GS", "season")):
        gap = venue.gap[venue.scope == scope]
        got |= {name: gap.median(), f"{name}X": gap.max(), f"{name}H": int((gap >= 0.5).sum())}
    rows = stored(scout.reliability_path())
    rows = rows[rows.split == "venue"].set_index("dimension")
    for i, d in enumerate(style.DIMENSIONS, 1):
        got |= {f"VA_{i}": rows.s_w1[d], f"VB_{i}": rows.s_w2[d], f"VC_{i}": rows.corrected[d]}
    assert got == ENTRIES


@pytest.mark.slow
def test_split_half_values_recomputed_from_the_stored_window_halves():
    w1 = stored(style.player_path())
    pool = identify.pool(w1, minutes_table())
    halves = stored(scout.window_halves_path())
    table = stored(scout.reliability_path()).set_index(["split", "dimension"])
    w2 = pool[KEYS].merge(stored(style.window2_path()), on=KEYS, how="left", validate="one_to_one")
    across = {c["dimension"]: c["r"] for c in style.half_correlations(pool, w2, "windows")}
    for split in style.SPLITS:
        for w in (1, 2):
            h = halves[(halves.window == w) & (halves.split == split)]
            a, b = (
                pool[KEYS].merge(h[h.half == half], on=KEYS, how="left", validate="one_to_one")
                for half in "AB"
            )
            both = ((a.minutes > 0) & (b.minutes > 0)).to_numpy()
            for c in style.half_correlations(a[both], b[both], split):
                row = table.loc[(split, c["dimension"])]
                assert (c["r"], c["n"]) == (row[f"r_w{w}"], row[f"n_w{w}"])
                assert 2 * c["r"] / (1 + c["r"]) == row[f"s_w{w}"]
        for d in style.DIMENSIONS:
            row = table.loc[(split, d)]
            assert row.r_windows == across[d]
            assert row.corrected == row.r_windows / np.sqrt(row.s_w1 * row.s_w2)


def worst(gaps):
    return max(float(np.max(g, initial=0.0)) for g in gaps.values())


@pytest.mark.slow
def test_identities_rechecked_from_the_stored_tables():
    season = stored(style.season_path())
    windows_ = [stored(style.player_path()), stored(style.window2_path())]
    a, b = (season[KEYS].merge(w, on=KEYS, how="left", validate="one_to_one") for w in windows_)
    assert worst(style.identity_gaps(season, a, b)) <= style.RELATIVE
    teams = stored(style.team_season_path())
    by = style.TEAM_KEYS
    clubs = [stored(style.team_path()), stored(style.team_w2_path())]
    a, b = (teams[by].merge(w, on=by, how="left", validate="one_to_one") for w in clubs)
    assert worst(style.identity_gaps(teams, a, b)) <= style.RELATIVE
    halves = stored(style.halves_path())
    for split in style.SPLITS:
        h = halves[halves.split == split]
        a, b = (
            season[KEYS].merge(h[h.half == half], on=KEYS, how="left", validate="one_to_one")
            for half in "AB"
        )
        assert (a.minutes.fillna(0) + b.minutes.fillna(0) == season.minutes).all()
        assert worst(style.identity_gaps(season, a, b)) <= style.RELATIVE


@pytest.mark.slow
def test_one_league_recomputed_from_the_event_caches():
    lg = "ligue_1"
    for name in ["actions", "pressures", "lineups"]:
        need(OUT / lg / f"{name}.parquet")
    season, teams = stored(style.season_path()), stored(style.team_season_path())
    halves, dealing = stored(style.halves_path()), stored(style.dealing_path())
    games = season_games()
    g, a, press, lineups = style.league_inputs(games[games.league == lg])
    player, team = style.profiles_of(g, a, press, lineups)
    mine = season[season.league == lg].reset_index(drop=True)
    got = mine[KEYS].merge(player, on=KEYS, how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(got[RAW], mine[RAW], check_dtype=False, check_exact=True)
    clubs = teams[teams.league == lg].reset_index(drop=True)
    pd.testing.assert_frame_equal(team[RAW], clubs[RAW], check_dtype=False, check_exact=True)

    d = dealing[dealing.league == lg]
    h = d[[*style.TEAM_KEYS, "game_id"]].assign(half=d.half_season)
    p = style.club_profiles(g, a, press, lineups, h)
    venue = halves[(halves.league == lg) & (halves.split == "venue")].reset_index(drop=True)
    on = [*KEYS, "half"]
    got = venue[[*on, "group"]].merge(p, on=on, how="left", validate="one_to_one")
    got = style.standardize_players(got, style.moments(season, season))
    by = ["league", "group"]
    got = style.centred(got, season.groupby(by)[style.Z].mean(), by)
    cols = [*RAW, *style.Z, *style.C]
    pd.testing.assert_frame_equal(got[cols], venue[cols], check_dtype=False, check_exact=True)
    assert (venue.minutes == got.n_sb_pressing).all()
