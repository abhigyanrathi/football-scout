import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadl
from socceraction.spadl import play_left_to_right

from fbrecruit import gems, manifest, minutes
from fbrecruit import shrinkage as sh
from fbrecruit.logs import Tee, key, show
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT, Loader, retry
from fbrecruit.split import windows

LOGS = Path("C:/Users/abhig/fb-scratch")
IN_POSSESSION = [
    "pass", "cross", "throw_in", "freekick_crossed", "freekick_short", "corner_crossed",
    "corner_short", "take_on", "dribble", "shot", "shot_freekick", "shot_penalty", "bad_touch",
    "goalkick",
]  # fmt: skip
OPEN_PLAY = ["pass", "cross", "take_on", "dribble", "shot", "bad_touch"]
DEFENSIVE = ["tackle", "interception", "clearance"]
SHOTS = ["shot", "shot_freekick", "shot_penalty"]
DIMENSIONS = [
    "sb_pressing", "possession", "buildup", "width", "verticality", "depth", "dribble", "creation",
]  # fmt: skip
KEYS = sh.KEYS
TEAM_KEYS = ["league", "team_id"]
FINAL_THIRD = 70.0
LONG_SEQUENCE = 10.0
SB_HALFWAY = 60.0
NEEDED = ["location", "team_id", "player_id", "type_name", "related_events"]
PRESSURE_COLUMNS = [
    "game_id", "event_id", "period_id", "seconds", "team_id", "player_id", "position_name",
    "x", "y", "duration", "counterpress", "possession", "pressed_player_id",
]  # fmt: skip
Z = [f"z_{d}" for d in DIMENSIONS]
C = [f"c_{d}" for d in DIMENSIONS]
# the dimensions whose value is a total over games over the total counted in n_; verticality's
# n_ counts passes while its value is a ratio of passed distances
RATIOS = ["possession", "buildup", "width", "depth", "dribble", "creation"]
SPLITS = ["venue", "parity"]
HALF_MINUTES = 450
RELATIVE = 1e-12
MINUTES_GAP = 1e-9
CLUBS = 80


def part_paths(league, game_id):
    parts = OUT / league / "parts"
    return parts / f"{game_id}_pressures.parquet", parts / f"{game_id}_events_summary.parquet"


def pressed_player(pressures, events):
    """The player of the first related event from another team than the pressure's, or null."""
    team = dict(zip(events.event_id, events.team_id, strict=True))
    player = dict(zip(events.event_id, events.player_id, strict=True))
    out = []
    for own, related in zip(pressures.team_id, pressures.related_events, strict=True):
        other = next((e for e in related if e in team and team[e] != own), None)
        out.append(pd.NA if other is None else player[other])
    return pd.array(out, dtype="Int64")


def pressure_rows(events):
    p = events[events.type_name == "Pressure"]
    xy = p.location.apply(lambda v: v if isinstance(v, list | tuple) else [np.nan, np.nan])
    return pd.DataFrame(
        {
            "game_id": p.game_id.astype("int64"),
            "event_id": p.event_id,
            "period_id": p.period_id.astype("int64"),
            # StatsBomb timestamps restart at zero in every period
            "seconds": p.timestamp.dt.total_seconds(),
            "team_id": p.team_id.astype("int64"),
            "player_id": p.player_id.astype("Int64"),
            "position_name": p.position_name,
            "x": xy.str[0].astype(float),
            "y": xy.str[1].astype(float),
            "duration": p.duration.astype(float),
            "counterpress": p.counterpress.fillna(False).astype(bool),
            "possession": p.possession.astype("int64"),
            "pressed_player_id": pressed_player(p, events),
        }
    )[PRESSURE_COLUMNS].reset_index(drop=True)


def game_summary(game_id, events):
    shots = events[events.type_name == "Shot"]
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "events": [len(events)],
            "pressures": [int((events.type_name == "Pressure").sum())],
            "shots": [len(shots)],
            "shot_mean_x": [shots.location.str[0].astype(float).mean()],
        }
    )


def check_schema(events):
    key("2a: columns and dtypes of the first game's events")
    key(events.dtypes.to_string())
    missing = [c for c in NEEDED if c not in events.columns]
    key(f"2a: required columns missing: {missing}")
    assert not missing, f"HARD STOP: events lack {missing}"
    key(f"2a: timestamp dtype {events.timestamp.dtype}; seconds = timestamp.dt.total_seconds()")


def read_games():
    loader = Loader("remote")
    games = {lg: pd.read_parquet(OUT / lg / "games.parquet") for lg in LEAGUES}
    first = games[next(iter(LEAGUES))].game_id.iloc[0]
    check_schema(retry(loader.events, int(first)))
    for lg, g in games.items():
        read, skipped = 0, 0
        for game_id in g.game_id:
            paths = part_paths(lg, game_id)
            if all(p.exists() for p in paths):
                skipped += 1
                continue
            try:
                events = retry(loader.events, int(game_id))
            except Exception as e:
                print(f"FAILED {lg} {game_id}: {e!r}", flush=True)
                continue
            pressure_rows(events).to_parquet(paths[0], index=False)
            game_summary(game_id, events).to_parquet(paths[1], index=False)
            read += 1
            print(f"{lg} {game_id}: events {len(events)}", flush=True)
        key(f"2b {lg}: games read now {read}, skipped as already read {skipped}")


def consolidate(lg, games, lineups):
    done = [g for g in games.game_id if all(p.exists() for p in part_paths(lg, g))]
    key(f"\n2c {lg}: games read {len(done)} of {len(games)} in games.parquet")
    short = sorted(set(games.game_id) - set(done))
    key(f"2c {lg}: games not read {short}")
    assert not short, f"HARD STOP: {lg} games not read"

    pr = pd.concat(pd.read_parquet(part_paths(lg, g)[0]) for g in done).reset_index(drop=True)
    sm = pd.concat(pd.read_parquet(part_paths(lg, g)[1]) for g in done).reset_index(drop=True)
    pr.to_parquet(OUT / lg / "pressures.parquet", index=False)
    sm.to_parquet(OUT / lg / "pressures_summary.parquet", index=False)

    m = sm.merge(games[["game_id", "n_events"]], on="game_id", validate="one_to_one")
    differ = m[m.events != m.n_events]
    key(f"2c {lg}: events read {int(m.events.sum())}, stored n_events {int(m.n_events.sum())}")
    key(f"2c {lg}: games whose event count differs {len(differ)} {differ.game_id.tolist()}")
    assert differ.empty, f"HARD STOP: {lg} event counts differ"

    team_games = pd.concat(
        [games[["game_id", "home_team_id"]].set_axis(["game_id", "team_id"], axis=1)]
        + [games[["game_id", "away_team_id"]].set_axis(["game_id", "team_id"], axis=1)]
    )
    per = team_games.merge(
        pr.groupby(["game_id", "team_id"]).size().rename("n").reset_index(),
        on=["game_id", "team_id"],
        how="left",
    ).n.fillna(0)
    key(
        f"2c {lg}: pressure events {len(pr)}; per team-game over {len(per)} team-games: "
        f"min {per.min()!r}, median {per.median()!r}, max {per.max()!r}"
    )

    lu = lineups[lineups.league == lg][["game_id", "team_id", "player_id"]].astype("int64")
    j = pr.merge(lu.drop_duplicates().assign(found=True), how="left")
    orphan = pr[j.found.isna().to_numpy()]
    key(f"2c {lg}: pressure events with no lineup row {len(orphan)}")
    key(orphan.to_string(index=False))

    dup = int(pr.event_id.duplicated().sum())
    key(f"2c {lg}: duplicate event_ids {dup}")
    assert dup == 0, f"{lg}: duplicate event_ids"
    key(f"2c {lg}: share with a pressed_player_id {pr.pressed_player_id.notna().mean()!r}")
    shot_x = (sm.shot_mean_x * sm.shots).sum() / sm.shots.sum()
    key(f"2c {lg}: shots {int(sm.shots.sum())}, mean Shot x {shot_x!r}")
    assert shot_x > 60, f"HARD STOP: {lg} mean Shot x {shot_x!r} is not above 60"
    return pr


def pressures():
    read_games()
    lineups = minutes.load_lineups("statsbomb")
    total = 0
    for lg in LEAGUES:
        games = pd.read_parquet(OUT / lg / "games.parquet")
        total += len(consolidate(lg, games, lineups))
    key(f"\n2c: pressure events in all four leagues {total}")
    manifest.record(
        "statsbomb_pressures",
        {"competition_ids": LEAGUES, "getter": "remote (open data)", "events": "Pressure"},
        manifest.file_sizes(OUT.glob("*/pressures*.parquet"), OUT),
    )
    key("2d: recorded under statsbomb_pressures in data/manifest.json")


def player_path():
    return PROCESSED / "style_player_w1.parquet"


def team_path():
    return PROCESSED / "style_team_w1.parquet"


def window2_path():
    return PROCESSED / "style_player_w2.parquet"


def season_path():
    return PROCESSED / "style_player_season.parquet"


def team_season_path():
    return PROCESSED / "style_team_season.parquet"


def team_w2_path():
    return PROCESSED / "style_team_w2.parquet"


def dealing_path():
    return PROCESSED / "style_dealing.parquet"


def halves_path():
    return PROCESSED / "style_halves_season.parquet"


def window1_games(window=1):
    """The games of window 1, or of the window given, with their home and away teams."""
    w = windows()
    w = w[w.window == window]
    cols = ["game_id", "home_team_id", "away_team_id"]
    teams = pd.concat(
        pd.read_parquet(OUT / lg / "games.parquet", columns=cols).assign(league=lg)
        for lg in LEAGUES
    )
    return w.merge(teams, on=["league", "game_id"], validate="one_to_one").reset_index(drop=True)


def sequence_ids(a):
    """A new sequence starts at every change of game, period or team; a must be in order."""
    new = (
        (a.game_id != a.game_id.shift())
        | (a.period_id != a.period_id.shift())
        | (a.team_id != a.team_id.shift())
    )
    return new.cumsum().to_numpy()


def creating(a):
    """Actions whose followers in the sequence, up to the first shot, are all one other player."""
    n = len(a)
    pos = np.arange(n)
    seq = a.seq.to_numpy()
    player = a.player_id.to_numpy()
    shot_pos = pd.Series(np.where(a.type_name.isin(SHOTS), pos, n))
    first_shot_from = shot_pos.iloc[::-1].groupby(seq[::-1]).cummin().sort_index().to_numpy()
    same_seq = np.r_[seq[1:] == seq[:-1], False]
    changes = np.r_[True, (player[1:] != player[:-1]) | ~same_seq[:-1]]
    run_end = pd.Series(pos).groupby(changes.cumsum()).transform("max").to_numpy()
    nxt = np.minimum(pos + 1, n - 1)
    other = np.r_[player[1:] != player[:-1], False]
    return same_seq & other & (first_shot_from[nxt] <= run_end[nxt])


def with_sequences(a):
    """Sequence id, sequence duration and the creation flag; a must be in action order."""
    a = a.assign(seq=sequence_ids(a))
    t = a.groupby("seq").time_seconds
    a["seq_duration"] = t.transform("last") - t.transform("first")
    a["creates"] = creating(a)
    return a


def load_actions(games):
    """Window-1 SPADL actions, each game played left to right, in game, period, action order."""
    frames = []
    for lg in LEAGUES:
        g = games[games.league == lg]
        home = dict(zip(g.game_id, g.home_team_id, strict=True))
        a = pd.read_parquet(OUT / lg / "actions.parquet")
        a = a[a.game_id.isin(home)]
        frames += [
            play_left_to_right(x, home[gid]).assign(league=lg) for gid, x in a.groupby("game_id")
        ]
    a = pd.concat(frames, ignore_index=True)
    a["player_id"] = a.player_id.astype("int64")
    a["type_name"] = a.type_id.map(dict(enumerate(spadl.actiontypes)))
    a = a.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
    return with_sequences(a)


def buildup_times(a):
    """Per sequence starting before x = 70 and reaching it: seconds to its first end_x >= 70."""
    g = a.groupby("seq")
    first_x, first_t = g.start_x.first(), g.time_seconds.first()
    reach = a[a.end_x >= FINAL_THIRD].groupby("seq").time_seconds.first()
    t = (reach - first_t.reindex(reach.index))[first_x.reindex(reach.index) < FINAL_THIRD]
    return t.rename("seconds").reset_index()


def opponent_share(a, games):
    """Per team-game, the opponent's in-possession actions over both teams' in-possession ones."""
    g = games[["league", "game_id", "home_team_id", "away_team_id"]]
    tg = pd.concat(
        [
            g.set_axis(["league", "game_id", "team_id", "opp_id"], axis=1),
            g.set_axis(["league", "game_id", "opp_id", "team_id"], axis=1),
        ],
        ignore_index=True,
    )
    ip = a[a.type_name.isin(IN_POSSESSION)].groupby(["game_id", "team_id"]).size()
    own = ip.reindex(pd.MultiIndex.from_frame(tg[["game_id", "team_id"]]), fill_value=0)
    opp = ip.reindex(pd.MultiIndex.from_frame(tg[["game_id", "opp_id"]]), fill_value=0)
    return tg.assign(opp_share=opp.to_numpy() / (own.to_numpy() + opp.to_numpy()))


def action_dimensions(a, by, times):
    """Every dimension but pressing, over the rows of a grouped by `by`."""
    op = a[a.type_name.isin(OPEN_PLAY)]
    op = op.assign(
        long=op.seq_duration > LONG_SEQUENCE,
        off_axis=(op.start_y - spadl.field_width / 2).abs(),
        take_on=op.type_name == "take_on",
    )
    g = op.groupby(by)
    out = {"possession": g.long.mean(), "width": g.off_axis.mean()}
    out |= {"dribble": g.take_on.mean(), "creation": g.creates.mean()}
    out |= {f"n_{d}": g.size() for d in ("possession", "width", "dribble", "creation")}

    pc = a[a.type_name.isin(["pass", "cross"])]
    length = np.hypot(pc.end_x - pc.start_x, pc.end_y - pc.start_y)
    pc = pc.assign(length=length, forward=length.where(pc.end_x > pc.start_x, 0.0))
    v = pc.groupby(by).agg(
        n=("length", "size"), forward=("forward", "sum"), total=("length", "sum")
    )
    out |= {"verticality": (v.forward / v.total).where(v.total > 0), "n_verticality": v.n}

    d = a[a.type_name.isin(DEFENSIVE)]
    d = d.assign(depth=d.start_x / spadl.field_length).groupby(by).depth
    out |= {"depth": d.mean(), "n_depth": d.size()}

    s = a[[*by, "seq"]].drop_duplicates().merge(times, on="seq").groupby(by).seconds
    out |= {"buildup": -s.mean(), "n_buildup": s.size()}
    return pd.DataFrame(out)


def player_pressing(press, lineups, share):
    share = share[["game_id", "team_id", "opp_share"]]
    lu = lineups[lineups.minutes_played > 0].merge(
        share, on=["game_id", "team_id"], validate="many_to_one"
    )
    m = (
        lu.assign(w=lu.minutes_played * lu.opp_share)
        .groupby(KEYS)
        .agg(minutes=("minutes_played", "sum"), w=("w", "sum"))
    )
    high = press[press.x > SB_HALFWAY].groupby(KEYS).size().reindex(m.index, fill_value=0)
    share_w = m.w / m.minutes
    return pd.DataFrame(
        {"sb_pressing": high / m.minutes * 90 * 0.5 / share_w, "n_sb_pressing": m.minutes}
    )


def team_pressing(press, share):
    s = share.groupby(TEAM_KEYS).agg(games=("game_id", "size"), opp=("opp_share", "mean"))
    high = press[press.x > SB_HALFWAY].groupby(TEAM_KEYS).size().reindex(s.index, fill_value=0)
    return pd.DataFrame({"sb_pressing": high / s.games * 0.5 / s.opp, "n_sb_pressing": s.games})


def ordered(frame):
    return frame[[c for d in DIMENSIONS for c in (d, f"n_{d}")]].reset_index()


def profiles_of(games, a, press, lineups):
    """Player and team dimensions over the given games alone."""
    ids = set(games.game_id)
    a = a[a.game_id.isin(ids)]
    press = press[press.game_id.isin(ids)]
    lineups = lineups[lineups.game_id.isin(ids)]
    share = opponent_share(a, games)
    times = buildup_times(a)
    player = action_dimensions(a, KEYS, times).join(
        player_pressing(press, lineups, share), how="outer"
    )
    team = action_dimensions(a, TEAM_KEYS, times).join(team_pressing(press, share), how="outer")
    return ordered(player), ordered(team)


def load_inputs():
    games = window1_games()
    a = load_actions(games)
    press = pd.concat(
        pd.read_parquet(OUT / lg / "pressures.parquet").assign(league=lg) for lg in LEAGUES
    )
    assert press.player_id.notna().all()
    press["player_id"] = press.player_id.astype("int64")
    lineups = minutes.load_lineups("statsbomb")[[*KEYS, "game_id", "minutes_played"]]
    return games, a, press, lineups.astype({"team_id": "int64"})


def window_inputs(window):
    """load_inputs for the games of one window, with each file read for those games only."""
    games = window1_games(window)
    frames, press, lineups = [], [], []
    for lg in LEAGUES:
        g = games[games.league == lg]
        home = dict(zip(g.game_id, g.home_team_id, strict=True))
        only = [("game_id", "in", g.game_id.tolist())]
        a = pd.read_parquet(OUT / lg / "actions.parquet", filters=only)
        frames += [
            play_left_to_right(x, home[gid]).assign(league=lg) for gid, x in a.groupby("game_id")
        ]
        p = pd.read_parquet(OUT / lg / "pressures.parquet", filters=only)
        press.append(p.assign(league=lg))
        cols = ["game_id", "team_id", "player_id", "minutes_played"]
        lu = pd.read_parquet(OUT / lg / "lineups.parquet", columns=cols, filters=only)
        lineups.append(lu.assign(league=lg))
    a = pd.concat(frames, ignore_index=True)
    a["player_id"] = a.player_id.astype("int64")
    a["type_name"] = a.type_id.map(dict(enumerate(spadl.actiontypes)))
    a = a.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
    press = pd.concat(press)
    assert press.player_id.notna().all()
    press["player_id"] = press.player_id.astype("int64")
    lineups = pd.concat(lineups, ignore_index=True)[[*KEYS, "game_id", "minutes_played"]]
    return games, with_sequences(a), press, lineups.astype({"team_id": "int64"})


def season_games():
    """The games of both windows, with their game days, windows and home and away teams."""
    return pd.concat([window1_games(1), window1_games(2)], ignore_index=True)


def league_inputs(games):
    """window_inputs for the given games, all of one league, with each file read for those games
    only."""
    (lg,) = games.league.unique()
    home = dict(zip(games.game_id, games.home_team_id, strict=True))
    only = [("game_id", "in", games.game_id.tolist())]
    a = pd.read_parquet(OUT / lg / "actions.parquet", filters=only)
    frames = [play_left_to_right(x, home[gid]).assign(league=lg) for gid, x in a.groupby("game_id")]
    a = pd.concat(frames, ignore_index=True)
    a["player_id"] = a.player_id.astype("int64")
    a["type_name"] = a.type_id.map(dict(enumerate(spadl.actiontypes)))
    a = a.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
    press = pd.read_parquet(OUT / lg / "pressures.parquet", filters=only).assign(league=lg)
    assert press.player_id.notna().all()
    press["player_id"] = press.player_id.astype("int64")
    cols = ["game_id", "team_id", "player_id", "minutes_played"]
    lineups = pd.read_parquet(OUT / lg / "lineups.parquet", columns=cols, filters=only)
    lineups = lineups.assign(league=lg)[[*KEYS, "game_id", "minutes_played"]]
    return games, with_sequences(a), press, lineups.astype({"team_id": "int64"})


def moments(player, pop):
    """Per group mean and standard deviation of each dimension over the estimation population."""
    est = player.merge(pop[KEYS], on=KEYS, validate="one_to_one")
    est = est[est.group != "UNKNOWN"]
    return est.groupby("group")[DIMENSIONS].agg(["mean", "std"])


def standardize_players(player, mom):
    z = {}
    for d in DIMENSIONS:
        mean, sd = player.group.map(mom[(d, "mean")]), player.group.map(mom[(d, "std")])
        z[f"z_{d}"] = (player[d] - mean) / sd
    return player.assign(**z)


def standardize_teams(team, ref):
    return team.assign(**{f"z_{d}": (team[d] - ref[d].mean()) / ref[d].std() for d in DIMENSIONS})


def player_table(games, a, press, lineups, pop):
    """The player table of the profiles step, standardized with the moments over pop, unwritten."""
    player, _ = profiles_of(games, a, press, lineups)
    rows = pd.read_parquet(sh.group_path())[[*KEYS, "group", "window1_minutes"]]
    player = rows.merge(player, on=KEYS, how="left", validate="one_to_one")
    return standardize_players(player, moments(player, pop))


def split_halves(games, a, press, lineups, rows, mom):
    """The rows' profiles on the odd and on the even game days of games, standardized with mom, over
    the rows with minutes in both, as the reliability step computes them."""
    halves = []
    for parity in (1, 0):
        p, _ = profiles_of(games[games.game_day % 2 == parity], a, press, lineups)
        halves.append(standardize_players(rows.merge(p, on=KEYS, how="left"), mom))
    odd, even = halves
    both = ((odd.n_sb_pressing > 0) & (even.n_sb_pressing > 0)).to_numpy()
    return odd[both].reset_index(drop=True), even[both].reset_index(drop=True)


def alternate(rows, by):
    """A and B in turn over the home games of each group of by from A and over its away games from
    B, in the order of rows."""
    turn = rows.groupby([*by, "home"]).cumcount() % 2
    return np.where(rows.home == (turn == 0), "A", "B")


def deal(games):
    """Each club's games in matchday order, then game_id, with its halves of the season and of the
    game's window, dealt by venue."""
    cols = ["league", "game_id", "game_day", "window"]
    rows = pd.concat(
        [
            games[cols].assign(team_id=games.home_team_id, home=True),
            games[cols].assign(team_id=games.away_team_id, home=False),
        ],
        ignore_index=True,
    )
    rows = rows.sort_values([*TEAM_KEYS, "game_day", "game_id"]).reset_index(drop=True)
    rows["half_season"] = alternate(rows, TEAM_KEYS)
    rows["half_window"] = alternate(rows, [*TEAM_KEYS, "window"])
    return rows[[*TEAM_KEYS, "game_id", "game_day", "home", "half_season", "half_window"]]


def club_profiles(games, a, press, lineups, halves):
    """Each club's player rows of profiles_of over its games in each of its halves; halves has one
    row per club and game, with league, team_id, game_id and half."""
    out = []
    for (lg, team, half), h in halves.groupby([*TEAM_KEYS, "half"]):
        p, _ = profiles_of(games[games.game_id.isin(h.game_id)], a, press, lineups)
        out.append(p[(p.league == lg) & (p.team_id == team)].assign(half=half))
    return pd.concat(out, ignore_index=True)


def pooled(first, second):
    """Per row of two aligned parts, each n_ summed and each value the n_-weighted mean of the
    parts' values; a part with no row or an n_ of 0 adds nothing, and a sum of 0 gives NaN."""
    out = {}
    for d in DIMENSIONS:
        n1 = first[f"n_{d}"].fillna(0).to_numpy(np.float64)
        n2 = second[f"n_{d}"].fillna(0).to_numpy(np.float64)
        s1 = np.where(n1 > 0, first[d].to_numpy(np.float64) * n1, 0.0)
        s2 = np.where(n2 > 0, second[d].to_numpy(np.float64) * n2, 0.0)
        n = n1 + n2
        out[d] = np.where(n > 0, (s1 + s2) / np.where(n > 0, n, 1.0), np.nan)
        out[f"n_{d}"] = n
    return pd.DataFrame(out)


def relative_gap(a, b):
    """|a - b| / max(1, |a|, |b|), 0 where both are NaN and infinite where one is."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    gap = np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    both, one = np.isnan(a) & np.isnan(b), np.isnan(a) | np.isnan(b)
    return np.where(both, 0.0, np.where(one, np.inf, gap))


def identity_gaps(whole, first, second):
    """Relative gaps of whole, taken over two parts, from them, the three frames aligned: of every
    n_ from the parts' sum, of each value in RATIOS from their n_-weighted mean and, on rows with
    minutes in one part only, of every value from that part's."""
    p = pooled(first, second)
    out = {
        "n_": [relative_gap(whole[f"n_{d}"].fillna(0), p[f"n_{d}"]) for d in DIMENSIONS],
        "values": [relative_gap(whole[d], p[d]) for d in RATIOS],
        "one part": [],
    }
    has = [part.n_sb_pressing.fillna(0).to_numpy() > 0 for part in (first, second)]
    for part, only in ((first, has[0] & ~has[1]), (second, has[1] & ~has[0])):
        out["one part"] += [
            relative_gap(whole[d].to_numpy()[only], part[d].to_numpy()[only]) for d in DIMENSIONS
        ]
    return {kind: np.concatenate(gaps) for kind, gaps in out.items()}


def weighted_gap(whole, first, second, d):
    """The largest relative gap of d from the n_-weighted mean of the parts' values, over the rows
    with minutes in both parts."""
    both = (first.n_sb_pressing.fillna(0) > 0) & (second.n_sb_pressing.fillna(0) > 0)
    both = both.to_numpy()
    gaps = relative_gap(whole[d].to_numpy()[both], pooled(first, second)[d].to_numpy()[both])
    return float(np.max(gaps, initial=0.0))


def centred(frame, centres, by):
    """frame with c_ columns: each standard score less the centre of its row's group of by."""
    c = frame[by].merge(centres.reset_index(), on=by, how="left", validate="many_to_one")
    z = frame[Z].to_numpy(np.float64) - c[Z].to_numpy(np.float64)
    return frame.assign(**dict(zip(C, z.T, strict=True)))


def profiles():
    names = set(spadl.actiontypes)
    missing = [t for t in [*IN_POSSESSION, *OPEN_PLAY, *DEFENSIVE] if t not in names]
    key(f"3a: type names not in the installed spadl config: {missing}")
    assert not missing, "HARD STOP: type names missing from the spadl config"
    games, a, press, lineups = load_inputs()
    key(f"3a: window-1 games {len(games)}, actions {len(a)}")
    seqs = a.groupby("seq").agg(league=("league", "first"), duration=("seq_duration", "first"))
    key("3b: sequences per league and median duration in seconds")
    key(seqs.groupby("league").duration.agg(["size", "median"]).to_string())

    player, team = profiles_of(games, a, press, lineups)
    rows = pd.read_parquet(sh.group_path())[[*KEYS, "group", "window1_minutes"]]
    player = rows.merge(player, on=KEYS, how="left", validate="one_to_one")
    assert (player.n_sb_pressing == player.window1_minutes).all()
    pop = sh.population("pooled")
    mom = moments(player, pop)
    player = standardize_players(player, mom)
    team = standardize_teams(team, team)
    player.to_parquet(player_path(), index=False)
    team.to_parquet(team_path(), index=False)

    z = [f"z_{d}" for d in DIMENSIONS]
    key(f"\n3h: player rows {len(player)}, team rows {len(team)}")
    key("3h: non-null counts per dimension, players")
    key(player[[*DIMENSIONS, *z]].notna().sum().to_string())
    key("3h: non-null counts per dimension, teams")
    key(team[[*DIMENSIONS, *z]].notna().sum().to_string())
    key("\n3f: group means and standard deviations over the estimation population")
    for g in mom.index:
        for d in DIMENSIONS:
            key(f"3f {g} {d}: mean {mom.loc[g, (d, 'mean')]!r}, std {mom.loc[g, (d, 'std')]!r}")
    key("\n3h: team table, raw")
    key(show(team[[*TEAM_KEYS, *DIMENSIONS]].set_index(TEAM_KEYS), 4))
    key("\n3h: team table, standardized")
    key(show(team[[*TEAM_KEYS, *z]].set_index(TEAM_KEYS), 3))
    est = player.merge(pop[KEYS], on=KEYS)
    est = est[est.group.isin(sh.OUTFIELD)]
    key(f"\n3h: correlation of the standardized player dimensions, outfield population {len(est)}")
    key(show(est[z].corr(), 3))


def half_correlations(first, second, scope):
    rows = []
    for d in DIMENSIONS:
        x, y = first[f"z_{d}"].to_numpy(), second[f"z_{d}"].to_numpy()
        ok = ~np.isnan(x) & ~np.isnan(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        rows.append({"scope": scope, "dimension": d, "n": int(ok.sum()), "r": r})
    return rows


def reliability():
    games, a, press, lineups = load_inputs()
    player = pd.read_parquet(player_path())
    team = pd.read_parquet(team_path())
    pop = sh.population("pooled")
    mom = moments(player, pop)
    est = pop[KEYS].merge(player[[*KEYS, "group"]], on=KEYS, validate="one_to_one")
    est = est[est.group != "UNKNOWN"]

    halves = {}
    for name, parity in (("odd", 1), ("even", 0)):
        p, t = profiles_of(games[games.game_day % 2 == parity], a, press, lineups)
        p = est.merge(p, on=KEYS, how="left")
        halves[name] = (standardize_players(p, mom), standardize_teams(t, team))
    odd, even = halves["odd"][0], halves["even"][0]
    both = ((odd.n_sb_pressing > 0) & (even.n_sb_pressing > 0)).to_numpy()
    odd, even = odd[both].reset_index(drop=True), even[both].reset_index(drop=True)
    key(f"4: estimation-population players with minutes in both halves {len(odd)}")
    key(odd.group.value_counts().to_string())

    rows = []
    for g in [x for x in sh.GROUPS if x != "UNKNOWN"]:
        m = (odd.group == g).to_numpy()
        rows += half_correlations(odd[m], even[m], g)
    m = odd.group.isin(sh.OUTFIELD).to_numpy()
    rows += half_correlations(odd[m], even[m], "outfield")
    t_odd, t_even = (h[1].set_index(TEAM_KEYS) for h in (halves["odd"], halves["even"]))
    t_even = t_even.reindex(t_odd.index)
    key(f"4: teams in both halves {len(t_odd)}")
    rows += half_correlations(t_odd, t_even, "teams")

    out = pd.DataFrame(rows)
    out["spearman_brown"] = 2 * out.r / (1 + out.r)
    key("\n4: split-half reliability, odd against even matchdays, reported only")
    key(show(out.pivot(index="scope", columns="dimension", values="r")[DIMENSIONS], 3))
    for r in out.itertuples():
        key(f"4 {r.scope} {r.dimension}: n {r.n}, r {r.r!r}, spearman-brown {r.spearman_brown!r}")


def window2():
    """Window-2 profiles of every player-team with window-2 minutes, on the window-1 yardstick."""
    w1 = pd.read_parquet(player_path())
    pop = sh.population("pooled")
    same = player_table(*window_inputs(1), pop).equals(w1)
    key(
        "5: the player table recomputed with the changed style.py equals style_player_w1.parquet "
        f"in every column, NaN matching NaN: {same}"
    )
    assert same, "HARD STOP: style_player_w1.parquet is not reproduced"
    key("5: lineup_groups on the window-1 lineups")
    rule = sh.lineup_groups(sh.window1_lineups())
    m = w1[[*KEYS, "group"]].merge(
        rule, on=KEYS, how="outer", suffixes=("", "_rule"), indicator=True
    )
    differ = int(((m._merge != "both") | (m.group != m.group_rule)).sum())
    key(f"5: window-1 rows {len(w1)}, rule rows {len(rule)}, keys or groups that differ {differ}")
    assert differ == 0, "HARD STOP: the group rule does not reproduce every window-1 group"

    games, a, press, lineups = window_inputs(2)
    key(
        f"\n5a: windows of the games read {sorted(games.window.unique().tolist())}; "
        f"games {len(games)}, actions {len(a)}, pressure events {len(press)}, "
        f"lineup rows {len(lineups)}"
    )
    player, _ = profiles_of(games, a, press, lineups)
    t = sh.v2_table("pooled")
    rows = t[t.window == 2][[*KEYS, "minutes"]].reset_index(drop=True)
    m = rows.merge(player[[*KEYS, "n_sb_pressing"]], on=KEYS, how="outer", indicator=True)
    only = int((m._merge != "both").sum())
    gap = float((m.minutes - m.n_sb_pressing).abs().max())
    key(
        f"5a: v2 window-2 rows {len(rows)}, profile rows {len(player)}, keys in only one of them "
        f"{only}; largest difference of the profile minutes from the v2 minutes {gap!r}"
    )
    assert only == 0 and gap <= 1e-9, "HARD STOP: the profiles are not the v2 table's window-2 rows"

    rows = rows.merge(w1[[*KEYS, "group"]], on=KEYS, how="left", validate="one_to_one")
    source = pd.Series(np.where(rows.group.notna(), "window 1", "window 2"), name="source")
    key("5a: lineup_groups on the window-2 lineups")
    rule = sh.lineup_groups(sh.window1_lineups(2))
    new = rows[KEYS].merge(rule, on=KEYS, how="left", validate="one_to_one")
    key(f"5a: window-2 rows with no row in the rule {int(new.group.isna().sum())}")
    assert new.group.notna().all(), "HARD STOP: a window-2 row has no lineup rows"
    rows["group"] = rows.group.fillna(new.group)

    out = rows.merge(player, on=KEYS, how="left", validate="one_to_one")
    out = standardize_players(out, moments(w1, pop)).rename(columns={"minutes": "window2_minutes"})
    out = out[[c.replace("window1", "window2") for c in w1.columns]]
    out.to_parquet(window2_path(), index=False)

    z = [f"z_{d}" for d in DIMENSIONS]
    unknown = out.group == "UNKNOWN"
    key(f"\n5b: wrote {window2_path()} with {len(out)} rows, columns {list(out.columns)}")
    came = {s: int(((source == s) & ~unknown).sum()) for s in ("window 1", "window 2")}
    key(
        f"5b: rows whose group came from window 1 {came['window 1']}, from window 2 "
        f"{came['window 2']}, UNKNOWN {int(unknown.sum())}"
    )
    key(pd.crosstab(source, out.group).reindex(columns=sh.GROUPS, fill_value=0).to_string())
    key("5b: how the groups taken from window 2 were resolved")
    key(new.resolved_by[source == "window 2"].value_counts().to_string())
    key("5b: non-null counts per dimension")
    key(out[[*DIMENSIONS, *z]].notna().sum().to_string())
    key(f"5b: infinite standard scores {int(np.isinf(out[z].to_numpy()).sum())}")
    key(out.league.value_counts().reindex(list(LEAGUES)).to_string())


def tools_rows():
    """gems_value.parquet, the scouting tools' players; HARD STOP unless one row per player, each
    outfield with at least 900 minutes, on the keys of gems_scores.parquet."""
    tools = pd.read_parquet(gems.value_path())
    scored = pd.read_parquet(gems.scores_path(), columns=KEYS)[KEYS]
    same = (
        tools[KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
        .equals(scored.sort_values(KEYS).reset_index(drop=True))
    )
    outside = int((~tools.group.isin(sh.OUTFIELD)).sum())
    key(
        f"4a: tools' rows {len(tools)}; player_id unique {tools.player_id.is_unique}; groups "
        f"outside the outfield {outside}; smallest minutes {tools.minutes.min()!r}; keys equal "
        f"those of gems_scores.parquet ({len(scored)} rows): {same}"
    )
    ok = tools.player_id.is_unique and outside == 0 and same
    assert ok and (tools.minutes >= gems.MINUTES).all(), "HARD STOP: the tools' rows"
    key(tools.league.value_counts().reindex(list(LEAGUES)).to_string())
    key(tools.group.value_counts().reindex(sh.OUTFIELD).to_string())
    w1 = pd.read_parquet(player_path())
    g1 = tools[KEYS].merge(w1[[*KEYS, "group"]], on=KEYS, how="left", validate="one_to_one").group
    other = g1.notna().to_numpy() & (g1.to_numpy() != tools.group.to_numpy())
    key(
        f"4a: tools' rows with no row in style_player_w1 {int(g1.isna().sum())}, with another "
        f"group there {int(other.sum())}"
    )
    return tools


def reproduced(games):
    """Each league's window-1 and window-2 profiles from its season inputs, one league at a time;
    HARD STOP unless they reproduce style_player_w1, style_team_w1 and the raw dimensions and n_
    columns of style_player_w2. Returns window 2's club profiles."""
    parts = {w: ([], []) for w in (1, 2)}
    for lg in LEAGUES:
        g, a, press, lineups = league_inputs(games[games.league == lg])
        for w, (players, teams) in parts.items():
            p, t = profiles_of(g[g.window == w], a, press, lineups)
            players.append(p)
            teams.append(t)
        del a, press, lineups
    rows = pd.read_parquet(sh.group_path())[[*KEYS, "group", "window1_minutes"]]
    p1 = pd.concat(parts[1][0], ignore_index=True)
    player = rows.merge(p1, on=KEYS, how="left", validate="one_to_one")
    player = standardize_players(player, moments(player, sh.population("pooled")))
    team = pd.concat(parts[1][1]).sort_values(TEAM_KEYS).reset_index(drop=True)
    stored = pd.read_parquet(window2_path())
    p2 = pd.concat(parts[2][0], ignore_index=True)
    w2 = stored[KEYS].merge(p2, on=KEYS, how="left", validate="one_to_one")
    raw = [*KEYS, *[c for d in DIMENSIONS for c in (d, f"n_{d}")]]
    same = {
        "style_player_w1": player.equals(pd.read_parquet(player_path())),
        "style_team_w1": standardize_teams(team, team).equals(pd.read_parquet(team_path())),
        "the raw dimensions and n_ columns of style_player_w2": w2[raw].equals(stored[raw]),
    }
    for name, ok in same.items():
        key(f"4: {name} reproduced by the changed style.py, NaN matching NaN: {ok}")
    assert all(same.values()), "HARD STOP: a stored profile table is not reproduced"
    return pd.concat(parts[2][1]).sort_values(TEAM_KEYS).reset_index(drop=True)


def halves_of(games, dealing, rows, mom, venue):
    """The rows' profiles on each half of the games, per split, from each club's games in the half,
    one league at a time, with raw, n_ and standard scores: by venue with dealing's column venue,
    and by parity with odd matchdays in half A. Also split_halves' odd and even rows on the
    games."""
    dealing = dealing.merge(games[["league", "game_id"]], on=["league", "game_id"])
    splits = {
        "venue": dealing[venue].to_numpy(),
        "parity": np.where(dealing.game_day % 2 == 1, "A", "B"),
    }
    parts, odd, even = [], [], []
    for lg in LEAGUES:
        g, a, press, lineups = league_inputs(games[games.league == lg])
        mine = rows[rows.league == lg][[*KEYS, "group"]]
        for split, half in splits.items():
            h = dealing[[*TEAM_KEYS, "game_id"]].assign(half=half)
            p = club_profiles(g, a, press, lineups, h[h.league == lg])
            parts.append(mine.merge(p, on=KEYS).assign(split=split))
        o, e = split_halves(g, a, press, lineups, mine, mom)
        odd.append(o)
        even.append(e)
        del a, press, lineups
    halves = pd.concat(parts, ignore_index=True)
    halves = standardize_players(halves[halves.n_sb_pressing > 0].reset_index(drop=True), mom)
    return halves, pd.concat(odd, ignore_index=True), pd.concat(even, ignore_index=True)


def parity_matches(halves, odd, even):
    """Whether the parity halves' standard scores equal those of split_halves' odd and even rows,
    exactly."""
    par = halves[halves.split == "parity"]
    same = []
    for frame, half in ((odd, "A"), (even, "B")):
        mine = frame[KEYS].merge(par[par.half == half], on=KEYS, how="left", validate="one_to_one")
        a, b = mine[Z].to_numpy(np.float64), frame[Z].to_numpy(np.float64)
        same.append(np.array_equal(a, b, equal_nan=True))
    return all(same)


def season_profiles(games):
    """Each league's season profiles of its players and clubs, one league at a time; HARD STOP
    unless the games read are the league's window-1 and window-2 games and the player rows are the
    v2 table's rows over both windows, with their minutes."""
    wins = windows()
    players, teams = [], []
    for lg in LEAGUES:
        g, a, press, lineups = league_inputs(games[games.league == lg])
        own = wins[(wins.league == lg) & wins.window.isin([1, 2])].game_id
        exact = g.game_id.is_unique and len(g) == len(own) and set(g.game_id) == set(own)
        stray = sum(int((~f.game_id.isin(g.game_id)).sum()) for f in (a, press, lineups))
        key(
            f"\n4b {lg}: games {len(g)}, window 1 {int((g.window == 1).sum())}, window 2 "
            f"{int((g.window == 2).sum())}; actions {len(a)}, pressure events {len(press)}, "
            f"lineup rows {len(lineups)}; the games are exactly the league's window-1 and "
            f"window-2 games: {exact}; rows from another game {stray}"
        )
        assert exact and stray == 0, "HARD STOP: the games read are not the league's two windows"
        p, club = profiles_of(g, a, press, lineups)
        players.append(p)
        teams.append(club)
        del a, press, lineups
    player = pd.concat(players, ignore_index=True)
    t = sh.v2_table("pooled")
    cols = [*KEYS, "minutes"]
    v2 = t[t.window == 1][cols].merge(
        t[t.window == 2][cols], on=KEYS, how="outer", suffixes=("_1", "_2"), validate="one_to_one"
    )
    v2["minutes"] = v2.minutes_1.fillna(0) + v2.minutes_2.fillna(0)
    m = v2.merge(player[[*KEYS, "n_sb_pressing"]], on=KEYS, how="outer", indicator=True)
    only = int((m._merge != "both").sum())
    worst = float((m.minutes - m.n_sb_pressing).abs().max())
    key(
        f"\n4b: player rows {len(player)}, v2 keys over both windows {len(v2)}, keys in only one "
        f"of them {only}; largest difference of n_sb_pressing from the window-1 plus window-2 "
        f"minutes {worst!r}"
    )
    assert only == 0 and worst <= MINUTES_GAP, "HARD STOP: the player rows are not the v2 rows"
    return player, pd.concat(teams).sort_values(TEAM_KEYS).reset_index(drop=True)


def identity_report(label, whole, first, second):
    """Print the identity gaps of whole from its two aligned parts, per kind; return the largest."""
    worst = 0.0
    for kind, gaps in identity_gaps(whole, first, second).items():
        largest = float(np.max(gaps, initial=0.0))
        worst = max(worst, largest)
        key(
            f"{label}, {kind}: values compared {len(gaps)}, largest relative difference {largest!r}"
        )
    return worst


def window_identities(player, team, team2):
    """HARD STOP unless every season player row and club is, in each n_ and each dimension of
    RATIOS, the total over its windows' rows, and a row in one window has that window's values."""
    w1, w2 = pd.read_parquet(player_path()), pd.read_parquet(window2_path())
    sides = {
        "players": (player, w1, w2, KEYS),
        "clubs": (team, pd.read_parquet(team_path()), team2, TEAM_KEYS),
    }
    worst = 0.0
    for name, (whole, one, two, by) in sides.items():
        first = whole[by].merge(one, on=by, how="left", validate="one_to_one")
        second = whole[by].merge(two, on=by, how="left", validate="one_to_one")
        worst = max(worst, identity_report(f"4c {name}", whole, first, second))
        for d in ("sb_pressing", "verticality"):
            key(
                f"4c {name}, as information: largest relative difference of {d} from the "
                f"n_-weighted mean of its window values, rows in both windows, "
                f"{weighted_gap(whole, first, second, d)!r}"
            )
    key(
        "4c: left out of the value identity: sb_pressing, whose value divides high pressures by "
        "minutes times the opponent's share (clubs: by the summed share) while n_ holds minutes "
        "(clubs: games), and verticality, whose value divides forward by total passed distance "
        "while n_ counts passes and crosses"
    )
    assert worst <= RELATIVE, "HARD STOP: a season value is not the total over its windows"


def standard_scores(tools, player, team):
    """The tools' rows with their season profiles and the clubs' season profiles, each with z_ and
    c_ columns, and the players' moments and centres; HARD STOP if a standard deviation is 0 or not
    finite or a club value is missing."""
    rows = tools[[*KEYS, "group", "minutes"]].merge(
        player, on=KEYS, how="left", validate="one_to_one"
    )
    gap = float((rows.minutes - rows.n_sb_pressing).abs().max())
    key(f"\n4d: largest difference of the tools' rows' n_sb_pressing from their minutes {gap!r}")
    mom = moments(rows, tools)
    rows = standardize_players(rows, mom)
    centres = rows.groupby(["league", "group"])[Z].mean()
    rows = centred(rows, centres, ["league", "group"])
    club_mom = team[DIMENSIONS].agg(["mean", "std"])
    team = standardize_teams(team, team)
    club_centres = team.groupby("league")[Z].mean()
    team = centred(team, club_centres, ["league"])
    key("4d: group means and standard deviations (ddof 1) over the tools' rows")
    for grp in mom.index:
        for d in DIMENSIONS:
            mean, sd = mom.loc[grp, (d, "mean")], mom.loc[grp, (d, "std")]
            key(f"4d {grp} {d}: mean {mean!r}, sd {sd!r}")
    key("4d: centres, the mean standard score over the tools' rows of each league and group")
    for (lg, grp), c in centres.iterrows():
        key(f"4d centre {lg} {grp}: " + ", ".join(f"{d} {c[f'z_{d}']!r}" for d in DIMENSIONS))
    key("4d: club means and standard deviations (ddof 1) over the season club profiles")
    for d in DIMENSIONS:
        key(f"4d clubs {d}: mean {club_mom.loc['mean', d]!r}, sd {club_mom.loc['std', d]!r}")
    for lg, c in club_centres.iterrows():
        key(f"4d club centre {lg}: " + ", ".join(f"{d} {c[f'z_{d}']!r}" for d in DIMENSIONS))
    key("4d: missing standard scores among the tools' rows")
    key(rows[Z].isna().sum().to_string())
    sds = np.concatenate(
        [mom.xs("std", axis=1, level=1).to_numpy().ravel(), club_mom.loc["std"].to_numpy()]
    )
    finite = bool(np.isfinite(sds).all() and (sds != 0).all())
    missing = int(team[DIMENSIONS].isna().sum().sum())
    key(f"4d: every standard deviation finite and not zero {finite}; missing club values {missing}")
    assert finite and missing == 0, "HARD STOP: a standard deviation or a club value"
    return rows, team, mom, centres


def check_dealing(games, dealing):
    """HARD STOP unless every game is dealt once for each of its two clubs and each club's halves
    differ by at most one in games and in home games, over the season and in each window."""
    cols = [*TEAM_KEYS, "game_id", "home"]
    venues = [(games.home_team_id, True), (games.away_team_id, False)]
    expect = pd.concat(
        [games[["league", "game_id"]].assign(team_id=club, home=home) for club, home in venues],
        ignore_index=True,
    )[cols]
    got = dealing[cols].sort_values(cols, ignore_index=True)
    once = got.equals(expect.sort_values(cols, ignore_index=True))
    key(
        f"\n4f: dealing rows {len(dealing)}, games {len(games)}; every game dealt once for each "
        f"of its two clubs {once}"
    )
    d = dealing.merge(games[["league", "game_id", "window"]], on=["league", "game_id"])
    worst = 0
    for scope, part, half in (
        ("the season", d, "half_season"),
        ("window 1", d[d.window == 1], "half_window"),
        ("window 2", d[d.window == 2], "half_window"),
    ):
        n = part.groupby([*TEAM_KEYS, half]).agg(games=("game_id", "size"), home=("home", "sum"))
        n = n.unstack(half, fill_value=0)
        dg = int((n[("games", "A")] - n[("games", "B")]).abs().max())
        dh = int((n[("home", "A")] - n[("home", "B")]).abs().max())
        worst = max(worst, dg, dh)
        key(
            f"4f {scope}: clubs {len(n)}; largest difference between a club's halves in games "
            f"{dg}, in home games {dh}"
        )
    assert once and worst <= 1, "HARD STOP: the dealing"


def season_halves(games, dealing, rows, mom, centres):
    """The rows of style_halves_season.parquet; HARD STOP unless the parity halves match
    split_halves and, per split, every row's halves add up to his season row."""
    halves, odd, even = halves_of(games, dealing, rows, mom, "half_season")
    halves = centred(halves, centres, ["league", "group"])
    same = parity_matches(halves, odd, even)
    key(
        f"\n4g: rows split_halves returns on the season's games {len(odd)}; the parity halves' "
        f"standard scores equal its own exactly: {same}"
    )
    assert same, "HARD STOP: the parity halves differ from split_halves"
    out, worst = [], 0.0
    for split in SPLITS:
        s = halves[halves.split == split]
        first = rows[KEYS].merge(s[s.half == "A"], on=KEYS, how="left", validate="one_to_one")
        second = rows[KEYS].merge(s[s.half == "B"], on=KEYS, how="left", validate="one_to_one")
        m_a, m_b = first.n_sb_pressing.fillna(0), second.n_sb_pressing.fillna(0)
        gap = float((m_a + m_b - rows.minutes).abs().max())
        key(f"4g {split}: largest difference of the halves' minutes from the season's {gap!r}")
        assert gap <= MINUTES_GAP, "HARD STOP: the halves' minutes do not add to the season's"
        worst = max(worst, identity_report(f"4g {split}", rows, first, second))
        for grp in sh.OUTFIELD:
            at = rows.group == grp
            both = int((at & (m_a > 0) & (m_b > 0)).sum())
            each = int((at & (m_a >= HALF_MINUTES) & (m_b >= HALF_MINUTES)).sum())
            key(
                f"4g {split} {grp}: rows with minutes in both halves {both}, with at least "
                f"{HALF_MINUTES} in each {each}"
            )
        out += [first[m_a > 0], second[m_b > 0]]
    assert worst <= RELATIVE, "HARD STOP: a season value is not the total over its halves"
    table = pd.concat(out, ignore_index=True)
    table["minutes"] = table.n_sb_pressing.astype("int64")
    raw = [*DIMENSIONS, *[f"n_{d}" for d in DIMENSIONS]]
    return table[[*KEYS, "group", "split", "half", "minutes", *raw, *Z, *C]]


def season():
    start = time.perf_counter()
    tools = tools_rows()
    games = season_games()
    team2 = reproduced(games)
    player, team = season_profiles(games)
    key(f"4b: club rows over the season {len(team)}, over window 2 {len(team2)}")
    assert len(team) == len(team2) == CLUBS, "HARD STOP: a club table does not have 80 rows"
    window_identities(player, team, team2)
    rows, team, mom, centres = standard_scores(tools, player, team)

    raw = [*DIMENSIONS, *[f"n_{d}" for d in DIMENSIONS]]
    rows = rows[[*KEYS, "group", "minutes", *raw, *Z, *C]]
    written = {
        season_path(): rows,
        team_season_path(): team[[*TEAM_KEYS, *raw, *Z, *C]],
        team_w2_path(): team2[[*TEAM_KEYS, *raw]],
    }
    for path, frame in written.items():
        frame.to_parquet(path, index=False)
        key(f"\n4e: wrote {path} with {len(frame)} rows, columns {list(frame.columns)}")

    dealing = deal(games)
    check_dealing(games, dealing)
    dealing.to_parquet(dealing_path(), index=False)
    key(f"4f: wrote {dealing_path()} with {len(dealing)} rows, columns {list(dealing.columns)}")

    table = season_halves(games, dealing, rows, mom, centres)
    table.to_parquet(halves_path(), index=False)
    key(f"4g: wrote {halves_path()} with {len(table)} rows, columns {list(table.columns)}")
    key(f"4: {time.perf_counter() - start:.1f} s")


def main(argv):
    step = argv[0]
    name = {"window2": "p9a_style", "season": "p9b_season"}.get(step, f"p4d_{step}")
    LOGS.mkdir(parents=True, exist_ok=True)
    full = open(LOGS / f"{name}.log", "w", encoding="utf-8", errors="replace")
    brief = open(LOGS / f"{name}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "pressures":
            pressures()
        elif step == "profiles":
            profiles()
        elif step == "reliability":
            reliability()
        elif step == "window2":
            window2()
        elif step == "season":
            season()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
