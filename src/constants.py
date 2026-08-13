"""Shared constants used across multiple src/ modules.

Audit finding (AUDIT_2026-07-25.md, Technical Debt #3): BLEND_RATIO_BY_POSITION
was copy-pasted into team_aggregation.py, sos_adjustment.py, and
phase3_diagnostic.py independently, and had already diverged - sos_adjustment.py's
copy was missing the "LB" entry (harmless today since that module's only
consumer only loops CB/S, but a real KeyError risk the moment anyone extends
it to LB). Centralized here so there is exactly one copy to keep correct.

Audit finding #3 (AUDIT_2026-07-27_backend_review.md): recurred a THIRD time -
MATCHUP_FITTED_COEFFICIENT was independently hardcoded in both rest_tracking.py
and integrated_predictions.py. Centralized here; both files now import it.

Audit finding #2 (AUDIT_2026-07-27.md): the same duplication pattern recurred
for the Elo/EPA/Vegas baseline and hyperparameter constants introduced by the
Elo work (elo_model.py, elo_game_prediction.py, ensemble_model.py each had
their own hardcoded copies - two of them under the identical name
EPA_BASELINE_CORR/MAE, meaning different things: elo_model.py's copy is the
2025 season-level number, ensemble_model.py's was a separate hardcoded copy
of the same number, and elo_game_prediction.py's EPA_BASELINE_CORR_2024 is a
different, game-level number entirely). Centralized here, with SEASON vs.
GAME kept as explicitly separate constants rather than collapsed into one
generically-named pair - they are different real markets (a season win
total vs. an individual game spread) and conflating them under one name is
exactly the bug this cleanup found and fixed in the pre-commit README draft
(which cited the season-level Vegas number, +0.798/1.78, as if it were the
game-level spread accuracy, which is actually +0.504/9.72).
"""

# CB/S/LB winning blend ratios (tackle_weight, leverage_weight) from Phase 2
# Refinement Task 2's holdout search (LB matches CB - both landed at the edge
# of the tested grid; S found a genuine interior optimum).
BLEND_RATIO_BY_POSITION = {"CB": (0.8, 0.2), "S": (0.5, 0.5), "LB": (0.8, 0.2)}

# --- Elo hyperparameters -----------------------------------------------
# Learned via grid search on real 2015-2024 games (elo_model.learn_elo_
# hyperparameters, minimizing Brier score); home-field advantage fit
# empirically from the real 2015-2024 home win rate (54.7%).
ELO_K_FACTOR = 10
ELO_POINTS_PER_WIN = 45
ELO_HOME_FIELD_ADVANTAGE = 32.4

# --- EPA baselines, SEASON-level (real 2025 outcomes) -------------------
# Team strength -> projected season wins, vs. real 2025 final standings.
EPA_BASELINE_SEASON_CORR_2025 = 0.216
EPA_BASELINE_SEASON_MAE_2025 = 2.882

# --- EPA baselines, GAME-level (real 2024 holdout) -----------------------
# Team strength differential -> point spread, vs. real 2024 game results.
EPA_BASELINE_GAME_CORR_2024 = 0.255
EPA_BASELINE_GAME_MAE_2024 = 10.82

# --- Elo carryover accuracy (real, non-circular - no Vegas signal used) --
ELO_CARRYOVER_SEASON_CORR_2025 = 0.316
ELO_CARRYOVER_SEASON_MAE_2025 = 2.74
ELO_CARRYOVER_GAME_CORR_2024 = 0.393
ELO_CARRYOVER_GAME_MAE_2024 = 10.21

# --- Vegas baselines - SEASON win totals and GAME spreads are different --
# --- real markets, deliberately kept as separate constants (see module ---
# --- docstring - conflating them was a real bug caught before commit).  --
VEGAS_BASELINE_SEASON_CORR_2025 = 0.798  # devigged moneyline-implied season win totals
VEGAS_BASELINE_SEASON_MAE_2025 = 1.781
VEGAS_BASELINE_GAME_CORR_2025 = 0.504    # real spread_line vs. real point differential
VEGAS_BASELINE_GAME_MAE_2025 = 9.72

# --- Reference only - NOT imported/used programmatically anywhere. -------
# The probability->spread conversion is always refit live from real 2015-
# 2023 games by elo_game_prediction.fit_probability_to_spread_conversion(),
# specifically so it can never go stale the way a hardcoded copy would.
# These are just the most recent fit's values, kept here for docs/README
# reference only.
PROB_TO_SPREAD_COEFFICIENT_REFERENCE = 72.596
PROB_TO_SPREAD_INTERCEPT_REFERENCE = -1.641
PROB_TO_SPREAD_RESIDUAL_STD_REFERENCE = 13.436

# --- Matchup adjustment (Component 2.3, real 2015-2023 regression, n=2090 games) ---
MATCHUP_FITTED_COEFFICIENT = 1.065  # pts per unit net EPA/play edge differential

# --- Trade/injury signal engineering conventions ------------------------
# Audit finding (AUDIT_2026-08-12_DEEP.md Section 4.1): MIN_GAMES_FOR_SEASON
# was independently hardcoded in both build_trade_signals.py and compute_
# injury_consistency_scores_2026.py; a year-over-year trend-smoothing
# epsilon was independently duplicated in build_trade_signals.py and
# generate_trade_scores_2026.py. Centralized here.
MIN_GAMES_FOR_SEASON = 4  # min real games played to count a player-season
TREND_EPSILON = 0.01  # denominator smoothing for year-over-year % change signals
