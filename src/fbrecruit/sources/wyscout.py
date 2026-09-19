import gc
import json
import re
import sys
import time

import pandas as pd
from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl.wyscout import convert_to_actions

from fbrecruit import manifest
from fbrecruit.paths import DATA, INTERIM, RAW

LEAGUES = {
    "italy": (524, 181248),
    "england": (364, 181150),
    "spain": (795, 181144),
    "france": (412, 181189),
    "germany": (426, 181137),
}
URLS = {
    "competitions": "https://ndownloader.figshare.com/files/15073685",
    "teams": "https://ndownloader.figshare.com/files/15073697",
    "players": "https://ndownloader.figshare.com/files/15073721",
    "matches": "https://ndownloader.figshare.com/files/14464622",
    "events": "https://ndownloader.figshare.com/files/14464685",
}
RAW_DIR = RAW / "wyscout"
OUT = INTERIM / "wyscout"
CHECK_AFTER = 20
PROJECTION_LIMIT_S = 60 * 60


def decode(text):
    """Turn literal backslash-u escapes into characters."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), text)


def download():
    needed = ["teams.json", "players.json"]
    for name in LEAGUES:
        needed += [f"matches_{name.title()}.json", f"events_{name.title()}.json"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not all((RAW_DIR / f).exists() for f in needed):
        PublicWyscoutLoader(root=str(RAW_DIR), download=True)


def team_names(team_ids):
    teams = pd.DataFrame(json.loads((RAW_DIR / "teams.json").read_text(encoding="utf-8")))
    teams = teams[teams.wyId.isin(team_ids)]
    return pd.DataFrame(
        {
            "team_id": teams.wyId,
            "team_name": teams.name.map(decode),
            "official_name": teams.officialName.map(decode),
        }
    ).reset_index(drop=True)


def ingest_league(name, clock):
    competition_id, season_id = LEAGUES[name]
    loader = PublicWyscoutLoader(root=str(RAW_DIR))
    games = loader.games(competition_id, season_id)
    n_events, lineups, actions = [], [], []
    for game in games.itertuples():
        events = loader.events(game.game_id)
        n_events.append(len(events))
        lineups.append(loader.players(game.game_id))
        actions.append(convert_to_actions(events, game.home_team_id))
        clock()
    games = games.assign(n_events=n_events)
    teams = team_names(set(games.home_team_id) | set(games.away_team_id))
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    teams.to_parquet(out / "teams.parquet", index=False)
    lineups = pd.concat(lineups, ignore_index=True)
    lineups.to_parquet(out / "lineups.parquet", index=False)
    actions = pd.concat(actions, ignore_index=True)
    actions.to_parquet(out / "actions.parquet", index=False)
    return len(games), sum(n_events), len(actions), len(lineups)


def ingest():
    download()
    total = sum(len(PublicWyscoutLoader(root=str(RAW_DIR)).games(*ids)) for ids in LEAGUES.values())
    started, done, totals = time.perf_counter(), [0], {}

    def clock():
        done[0] += 1
        if done[0] == CHECK_AFTER:
            elapsed = time.perf_counter() - started
            projection = elapsed / CHECK_AFTER * total
            print(
                f"after {CHECK_AFTER} games: {elapsed:.1f} s; projection for {total} games: "
                f"{projection:.0f} s (limit {PROJECTION_LIMIT_S} s)",
                flush=True,
            )
            if projection > PROJECTION_LIMIT_S:
                sys.exit("projection over the limit, stopping")

    for name in LEAGUES:
        totals[name] = ingest_league(name, clock)
        gc.collect()
        print(f"{name}: games, events, actions, lineup rows = {totals[name]}", flush=True)
    manifest.record(
        "wyscout",
        {"competition_season_ids": LEAGUES, "downloads": URLS},
        manifest.file_sizes(
            [*RAW_DIR.glob("*.json"), *RAW_DIR.glob("*.zip"), *OUT.glob("*/*.parquet")], DATA
        ),
    )
    return totals


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("totals per league (games, events, actions, lineup rows):", ingest())
