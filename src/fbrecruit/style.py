import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fbrecruit import manifest, minutes
from fbrecruit.logs import Tee, key
from fbrecruit.sources.statsbomb import LEAGUES, OUT, Loader, retry

LOGS = Path("C:/Users/abhig/fb-scratch")
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


def main(argv):
    step = argv[0]
    LOGS.mkdir(parents=True, exist_ok=True)
    full = open(LOGS / f"p4d_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(LOGS / f"p4d_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "pressures":
            pressures()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
