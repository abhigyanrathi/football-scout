import pandas as pd
import pytest

from fbrecruit.paths import INTERIM
from fbrecruit.split import assign_windows, boundary, games

WINDOWS = {
    # league: (games w1, games w2, actions w1, actions w2)
    "la_liga": (230, 150, 455229, 302081),
    "premier_league": (230, 150, 456422, 302012),
    "serie_a": (230, 150, 459943, 301802),
    "ligue_1": (228, 149, 461148, 304923),
}


def rounds(n, null=0):
    days = list(range(1, n + 1)) + [None] * null
    return pd.DataFrame({"game_id": range(len(days)), "game_day": pd.array(days, dtype="Int64")})


def test_boundary_38_rounds():
    assert boundary(38) == 23
    w = assign_windows(rounds(38)).set_index("game_day").window
    assert (w.loc[:23] == 1).all() and (w.loc[24:] == 2).all()
    assert (w == 1).sum() == 23 and (w == 2).sum() == 15


def test_boundary_34_rounds():
    assert boundary(34) == 21
    w = assign_windows(rounds(34)).window
    assert (w == 1).sum() == 21 and (w == 2).sum() == 13


def test_exact_multiple_is_not_rounded_up():
    assert boundary(10) == 6


def test_null_game_day_gets_no_window():
    out = assign_windows(rounds(38, null=2))
    assert out.window.isna().sum() == 2
    assert out[out.game_day.isna()].window.isna().all()
    assert (out.window == 1).sum() == 23


@pytest.mark.slow
@pytest.mark.parametrize("league", WINDOWS)
def test_window_counts(league):
    path = INTERIM / "statsbomb" / league / "actions.parquet"
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    g = assign_windows(games(league))
    actions = pd.read_parquet(path, columns=["game_id"])
    per_action = actions.game_id.map(g.set_index("game_id").window)
    counts = ((g.window == 1).sum(), (g.window == 2).sum())
    counts += ((per_action == 1).sum(), (per_action == 2).sum())
    assert counts == WINDOWS[league]
