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

## 2026-09-19: Correction to the StatsBomb minutes-agreement figure
Decision: Judge the StatsBomb links by minutes agreement among matched pairs with 450 or more StatsBomb minutes, as for Wyscout.
Alternatives considered: all matched pairs.
Reason: The earlier entry reports 93.98% of all matched pairs within 0.8 to 1.2 times. The farthest pairs there are players with fewer than 60 minutes, where a few minutes of difference is a large ratio. Among pairs with 450 or more StatsBomb minutes the share is 99.94% (median ratio 0.954); the Wyscout figure on the same basis is 99.9%.

## 2026-09-19: Earliest whole transfer row per mover
Decision: Each mover in the window is represented by the player's earliest qualifying transfer as a whole row.
Alternatives considered: taking the first non-null value of each column, which the first version did.
Reason: The old rule could mix two transfers: when the earliest transfer had a null fee, it took the fee of a later one. No mover in either cohort triggers this. All 10 players with more than one qualifying transfer (1 in the summer 2016 cohort, 9 in summer 2018) have a fee of 0.0 on the earliest, and the legacy and corrected rules give identical rows, labels and funnel levels.

## 2026-09-19: Within-season matchday split
Decision: Each StatsBomb 2015/16 league is split by matchday. Window 1 is matchdays 1 to ceil(0.6 × the league's last matchday), window 2 the rest. All four leagues run to matchday 38, so the boundary is matchday 23 everywhere: 230 and 150 games per window (Ligue 1 228 and 149), about 60.1% to 60.4% of actions in window 1. No game has a missing matchday. Leave-one-league-out across the four leagues is the second split.
Alternatives considered: splitting by calendar date.
Reason: A forward holdout inside one season gives features and a later target for the same players without needing a second season. Matchday order and date order differ only in La Liga: one window-1 game (matchday 16, played 17 February 2016) falls after the first window-2 game, and 10 window-2 games are dated before it; the largest gap is 4.88 days. The Premier League, Serie A and Ligue 1 have no violations. The split is left unchanged.

## 2026-09-19: Leakage-safe action values
Decision: The VAEP scores and concedes classifiers are fit five times, each on window-1 actions only. The pooled fit uses all four leagues and rates every action in all four. Each leave-one-league-out fit uses the other three leagues and rates only the actions of the league it left out. Features are the installed socceraction defaults (568 columns, three previous actions). The learner is xgboost 2.1.4 with tree_method hist, random_state 0, n_jobs 16, objective binary:logistic and 100 rounds, otherwise library defaults, with no tuning. Refitting the pooled scores model gives a bit-identical window-2 Brier score (0.009096510708332062). The installed default feature set includes the current action's result, so VAEP here is a retrospective value of what the action did, not an expected value of the attempt.
Alternatives considered: one fit on all actions; socceraction's own tree settings (depth 3 with early stopping on a random 25% of training rows).
Reason: A window-2 target must come from a model that never trained on window-2 actions, and a leave-one-league-out target from one that never saw that league. Evaluated on rows each model did not train on, mean predicted scores probability is within 6.1% of the observed rate everywhere (pooled on window 2: 0.010410 against 0.010746, Brier 0.009097, log loss 0.047711). Concedes probability is under-predicted: pooled on window 2 0.002103 against 0.002274 (−7.5%, Brier 0.002190). Three leave-one-league-out evaluations miss by more than 10%: La Liga window 1 (−14.9%), La Liga window 2 (−13.2%) and Ligue 1 window 2 (−14.3%). Every reliability table under-predicts in the lowest deciles and over-predicts in the top decile (pooled scores: 0.0584 predicted against 0.0539 observed).

## 2026-09-19: Minutes filter for window pairs
Decision: A player-team enters a window-1 to window-2 pair with at least 450 minutes in window 1 and 270 in window 2. That gives 1,144 pairs in the pooled table: La Liga 299, Premier League 278, Serie A 280, Ligue 1 287.
Alternatives considered: window-1 minimums of 180, 270, 360 and 450 crossed with window-2 minimums of 180, 270 and 360. Pair counts run from 1,335 (180 and 180) to 1,073 (450 and 360).
Reason: 450 minutes is the project's existing minimum for a player season. In window 2, 270 minutes is three full matches and 20% of the 1,350-minute maximum; raising the window-2 minimum from 180 to 270 costs 46 pairs. The threshold was chosen on minutes and cohort size, not on the correlations below. As a pipeline diagnostic only (no shrinkage, no controls, no held-out structure), the correlation between window-1 and window-2 pooled VAEP/90 ranges from 0.3244 to 0.3851 (Pearson) and 0.2671 to 0.3047 (Spearman) across the grid, and is 0.3807 and 0.3018 at the chosen threshold. It is not a result.

## 2026-09-20: Recalibration method
Decision: Scores and concedes probabilities are recalibrated with a logistic map on the logit, p_cal = 1 / (1 + exp(-(a + b * logit(p)))), fit by unpenalized maximum likelihood, one map per rating model, league and label.
Alternatives considered: isotonic regression.
Reason: Out of sample the raw classifiers are over-dispersed (for the pooled scores model on window 2, the lowest decile is under-predicted by a factor of 2.65 and the top decile over-predicted), and action values are differences of these probabilities. Isotonic regression was rejected because it is piecewise constant: consecutive states in the same block would get identical probabilities and a value change of exactly zero, and one league's window 1 holds only 932 to 1,085 concedes positives to fit on.

## 2026-09-20: Calibration data
Decision: Every calibrator is fit on window-1 rows only, using predictions from a classifier that never trained on those rows: for the pooled model, out-of-fold predictions from five game-level folds, the pooled model itself unchanged; for each leave-one-league-out model, its own predictions for its held-out league. Window-2 outcomes enter no fitted component and are used only to report calibration.
Alternatives considered: fitting calibrators on window 2; fitting the pooled calibrators on the pooled model's own window-1 predictions.
Reason: Fitting on window 2 would use the target, and the pooled model's own window-1 predictions are in-sample. Folds are whole games because labels and features span neighbouring actions within a game and each value is a difference of consecutive states.

## 2026-09-20: Leakage rule for leave-one-league-out, amended
Decision: The classifier never trains on the held-out league; its calibrator may be fit on that league's window-1 outcomes; neither uses window 2. The leave-one-league-out result therefore measures transfer to a new league after recalibration on that league's first 23 matchdays, and is described that way. The earlier leakage-rule entry still governs the classifier.
Alternatives considered: a calibrator fit on the three training leagues.
Reason: Miscalibration is league-specific (concedes under-predicted by 13% to 15% for held-out La Liga and Ligue 1), so a calibrator fit on the training leagues would not correct the held-out one.

## 2026-09-20: Pooled window-1 values
Decision: Pooled window-1 action values come from calibrated out-of-fold predictions and window-2 values from the calibrated pooled model; the uncalibrated tables are kept and the calibrated tables are written beside them.
Alternatives considered: calibrating the pooled model's own window-1 predictions; replacing the uncalibrated tables.
Reason: The pooled model trained on every window-1 row, so its window-1 predictions are in-sample (window-1 log loss in-sample against out-of-fold: scores 0.039959 against 0.046413, concedes 0.009418 against 0.013928), and a calibrator learned on out-of-sample predictions does not describe them.

## 2026-09-20: Calibration fit at depth 6
Decision: Compare classifier depth 6 with depth 3, under a rule fixed in advance, before building on calibrated values.
Alternatives considered: keeping the depth-6 two-parameter calibration as it is; a more flexible calibration map.
Reason: On its own fit rows, the two-parameter map leaves the depth-6 concedes probabilities with an S-shaped residual (window 1, all four leagues: the lowest four deciles under-predicted by 16% to 65%, the sixth to ninth deciles over-predicted by 10% to 31%), and the lowest two scores deciles under-predicted by 38% and 25%. Well-calibrated league maps would give a well-calibrated mixture, so the residual comes from the map's shape. Near zero the map behaves like p raised to a power below one, which steepens differences between small probabilities, and action values are built from those differences.

## 2026-09-20: Classifier depth
Decision: Scores use depth 3 and concedes use depth 3, chosen by a rule fixed before the comparison: for each label, depth 3 replaces depth 6 if its calibrated window-1 out-of-fold log loss is no higher and its Hosmer-Lemeshow statistic is lower (ten equal-count bins, all four leagues together); depth 6 stays if it is no worse on both. Only the tree depth changes. The depth is chosen once, by window-1 cross-validation across all four leagues, and used for every model of that label, including leave-one-league-out; no model or choice uses window 2. The depth-6 models, predictions and tables are kept. This amends the entry "Leakage-safe action values".
Alternatives considered: depth 6 for scores; depth 6 for concedes.
Reason: Calibrated window-1 out-of-fold log loss, depth 6 against depth 3: scores 0.046221 against 0.045498, concedes 0.013728 against 0.013315; Hosmer-Lemeshow: scores 104.43 against 14.20, concedes 303.92 against 48.45. Depth 6 overfits (window-1 log loss in-sample against out-of-fold: 0.039959 against 0.046413 for scores, 0.009418 against 0.013928 for concedes), and depth 3 is socceraction's default.
