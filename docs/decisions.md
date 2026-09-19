# Decisions

Append-only log of project decisions. Each entry: date, decision, alternatives considered, reason.

## 2026-09-19: Event data sources
Decision: Use StatsBomb open data for 2015/16 (La Liga, Premier League, Serie A, Ligue 1) and the public Wyscout dataset for 2017/18 (Serie A, Premier League, La Liga, Ligue 1, Bundesliga) as the two event-data seasons. National-team competitions are excluded.
Alternatives considered: one provider only; aggregated statistics from Understat or FBref (deferred).
Reason: StatsBomb's open data has five league-wide competition-seasons (the four above and the Indian Super League 2021/22) and 49 single-team ones, with no consecutive full league seasons. Wyscout adds a second full season and the Bundesliga.

## 2026-09-19: Transfermarkt for transfers and costs
Decision: Use the Transfermarkt dataset compiled by dcaribou (CC0 repository licence) for transfers, appearances and valuations. Raw data is downloaded locally and never committed. Player costs will come from market valuations, not transfer fees.
Alternatives considered: transfer fees as costs.
Reason: The transfers table has a positive fee for only 17,554 of 175,165 rows and no loan flag.

## 2026-09-19: Minutes come from match lineups
Decision: Player minutes are computed from each provider's match lineups, not taken from Transfermarkt.
Alternatives considered: Transfermarkt appearance minutes.
Reason: Minutes must describe the same matches the event features are built from, and the minimum-minutes filter must be enforceable on them.

## 2026-09-19: Wyscout limits
Decision: Wyscout-based features exclude pressing intensity, and results from the two providers are compared as relationships, not levels.
Alternatives considered: a pressing proxy built from other event types.
Reason: Wyscout's public data has no pressure events, and its event density is about half of StatsBomb's (about 1,670 events per game in ten measured games, against about 3,400 in the 2015/16 La Liga).

## 2026-09-19: Environment
Decision: Python 3.12, with multimethod pinned to 1.10 and the other dependency versions pinned in pyproject.toml.
Alternatives considered: Python 3.13 (socceraction 1.5.x requires Python below 3.13); multimethod 2.x (it breaks pandera 0.17.2, which socceraction's schemas use).
Reason: Both constraints were hit while loading StatsBomb data.
