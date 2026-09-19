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

## 2026-09-19: Linking StatsBomb players to Transfermarkt
Decision: Map each StatsBomb team to one Transfermarkt club per league by an optimal one-to-one assignment on club-name similarity, then match players by normalised name or nickname within the mapped club and season (similarity 0.85, margin 0.05). A second pass accepts a Transfermarkt name whose tokens are all contained in the StatsBomb name and nickname tokens.
Alternatives considered: mapping each team to its best name match independently; the published record-linkage tables.
Reason: The independent mapping sent 5 of 80 teams to the wrong club and left two leagues below a 90% minutes-weighted match rate (overall 90.63%). The one-to-one mapping gives 96.78% (La Liga 94.82%, Premier League 97.45%, Serie A 97.97%, Ligue 1 96.88%), and every mapped team shares at least 19 players by exact name with its club. The published link tables could not be retrieved when tried. The second pass adds 62 pairs and 2.32 minutes-weighted points, and Transfermarkt minutes are within 0.8 to 1.2 times StatsBomb minutes for 93.98% of matched pairs.

## 2026-09-19: Linking Wyscout players to Transfermarkt
Decision: Link Wyscout players to Transfermarkt players by exact date of birth plus name similarity (first and last name, or short name; similarity 0.75; exactly one candidate at or above it). A second pass accepts a same-birthday candidate whose name tokens are all contained in the Wyscout name tokens.
Alternatives considered: matching on names alone; the published record-linkage tables.
Reason: Exact date of birth is a strong key. The first pass links 93.41% of the Wyscout players with 450 or more minutes (Spain 84.16%, the other four leagues 95.05% to 97.15%; the misses are mostly long legal names). For linked players, Transfermarkt minutes are within 0.8 to 1.2 times Wyscout minutes in 99.9% of cases, which supports the links without a ground truth. The second pass raises the rate to 97.06% (Spain 95.27%).

## 2026-09-19: Transfers, loans and the transfer cohorts
Decision: The Transfermarkt transfers table has no loan flag, so each mover in the summer window is labelled by whether the same player has a mirror move (clubs swapped) within three years: paid, paid_mirror, loan_return, loan_out, both or free_other. The primary transfer cohort is permanent moves: paid, paid_mirror, and free_other not dated 30 June. Loan endings and moves labelled both are excluded; loan-outs are kept as a sensitivity set.
Alternatives considered: treating every mover in the window as a transfer; flagging loans by whether the player reappears for the old club two seasons later.
Reason: In the summer 2016 cohort 64 of 186 movers, and in summer 2018 118 of 321, are dated 30 June with a fee of zero or null, and the mirror label identifies 62 of the 64 and all 118 as loan endings. The reappearance heuristic flags only 7 of 32 and 9 of 66 loan-outs. At the last funnel stage (linked, 450 or more minutes the next season, destination in the same leagues) the permanent cohorts number 54 and 108.
