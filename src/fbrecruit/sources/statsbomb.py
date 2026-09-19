import sys
import time

import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl.statsbomb import convert_to_actions

from fbrecruit import manifest
from fbrecruit.paths import INTERIM

SEASON_ID = 27
LEAGUES = {"la_liga": 11, "premier_league": 2, "serie_a": 12, "ligue_1": 7}
OUT = INTERIM / "statsbomb"
TRIES = 5
CHECK_AFTER = 20
PROJECTION_LIMIT_S = 90 * 60


class Loader(StatsBombLoader):
    """players() fetches the events of a game again, so keep the last fetch."""

    _last = None

    def events(self, game_id, load_360=False):
        if self._last is None or self._last[0] != game_id:
            self._last = (game_id, super().events(game_id, load_360))
        return self._last[1]


def retry(fn, *args):
    for attempt in range(1, TRIES + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt == TRIES:
                raise
            time.sleep(2**attempt)


def fetch_game(loader, game):
    events = loader.events(game.game_id)
    lineups = loader.players(game.game_id)
    actions = convert_to_actions(events, game.home_team_id)
    return len(events), lineups, actions


def fetch_teams(loader, games, failures):
    """One game per team is enough: skip a game once both its teams are known."""
    seen, frames = set(), []
    for game in games.itertuples():
        if {game.home_team_id, game.away_team_id} <= seen:
            continue
        try:
            frames.append(retry(loader.teams, game.game_id))
        except Exception as e:
            failures.append((game.game_id, f"teams: {e!r}"))
            continue
        seen |= set(frames[-1].team_id)
    return pd.concat(frames).drop_duplicates("team_id").reset_index(drop=True)


def consolidate(out, games, teams, failures):
    parts = out / "parts"
    done = [g for g in games.game_id if (parts / f"{g}_events.txt").exists()]
    counts = {g: int((parts / f"{g}_events.txt").read_text()) for g in done}
    games = games.assign(n_events=games.game_id.map(counts).astype("Int64"))
    lineups = pd.concat(pd.read_parquet(parts / f"{g}_lineups.parquet") for g in done)
    actions = pd.concat(pd.read_parquet(parts / f"{g}_actions.parquet") for g in done)
    games.to_parquet(out / "games.parquet", index=False)
    teams.to_parquet(out / "teams.parquet", index=False)
    lineups.to_parquet(out / "lineups.parquet", index=False)
    actions.to_parquet(out / "actions.parquet", index=False)
    pd.DataFrame(failures, columns=["game_id", "error"]).to_csv(out / "failures.csv", index=False)
    return games, lineups, actions


def ingest():
    loader = Loader("remote")
    games = {name: retry(loader.games, comp, SEASON_ID) for name, comp in LEAGUES.items()}
    for name in LEAGUES:
        (OUT / name / "parts").mkdir(parents=True, exist_ok=True)

    def is_done(name, game_id):
        return (OUT / name / "parts" / f"{game_id}_events.txt").exists()

    remaining = sum(not is_done(n, g) for n, df in games.items() for g in df.game_id)
    started, fetched, all_failures, totals = time.perf_counter(), 0, {}, {}
    for name in LEAGUES:
        out, failures = OUT / name, []
        for game in games[name].itertuples():
            if is_done(name, game.game_id):
                continue
            try:
                n_events, lineups, actions = retry(fetch_game, loader, game)
            except Exception as e:
                failures.append((game.game_id, repr(e)))
                print(f"FAILED {name} {game.game_id}: {e!r}", flush=True)
                continue
            parts = out / "parts"
            lineups.to_parquet(parts / f"{game.game_id}_lineups.parquet", index=False)
            actions.to_parquet(parts / f"{game.game_id}_actions.parquet", index=False)
            (parts / f"{game.game_id}_events.txt").write_text(str(n_events))
            fetched += 1
            if fetched == CHECK_AFTER:
                elapsed = time.perf_counter() - started
                projection = elapsed / CHECK_AFTER * remaining
                print(
                    f"after {CHECK_AFTER} games: {elapsed:.1f} s; projection for {remaining} "
                    f"games: {projection:.0f} s (limit {PROJECTION_LIMIT_S} s)",
                    flush=True,
                )
                if projection > PROJECTION_LIMIT_S:
                    sys.exit("projection over the limit, stopping")
        teams_path = out / "teams.parquet"
        if teams_path.exists() and not failures:
            teams = pd.read_parquet(teams_path)
        else:
            teams = fetch_teams(loader, games[name], failures)
        g, lineups, actions = consolidate(out, games[name], teams, failures)
        all_failures[name] = failures
        totals[name] = (len(g), int(g.n_events.sum()), len(actions), len(lineups))
        print(f"{name}: games, events, actions, lineup rows = {totals[name]}", flush=True)
    manifest.record(
        "statsbomb",
        {"competition_ids": LEAGUES, "season_id": SEASON_ID, "getter": "remote (open data)"},
        manifest.file_sizes(OUT.glob("*/*.parquet"), OUT),
    )
    return totals, all_failures


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    totals, failures = ingest()
    print("totals per league (games, events, actions, lineup rows):", totals)
    print("failures:", failures)
