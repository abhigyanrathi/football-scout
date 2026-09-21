import sys
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadl
from socceraction.spadl import play_left_to_right

from fbrecruit import manifest, minutes
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


def window1_games():
    w = windows()
    w = w[w.window == 1]
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


def main(argv):
    step = argv[0]
    LOGS.mkdir(parents=True, exist_ok=True)
    full = open(LOGS / f"p4d_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(LOGS / f"p4d_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "pressures":
            pressures()
        elif step == "profiles":
            profiles()
        elif step == "reliability":
            reliability()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
