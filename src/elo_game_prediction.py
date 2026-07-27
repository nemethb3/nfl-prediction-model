"""Game Prediction Engine Improvement, Test A: Elo for Spreads.

Corrects 3 issues found in the spec before building:

1. The spec says to use carryover (non-circular) Elo but then says to load
   it from data/processed/elo_ratings_2025.csv. Checked: that file was
   actually built by Component 1's validate_elo_backtest() calling
   run_elo_season() with no initial_ratings argument, which defaults to the
   VEGAS-INFORMED variant - exactly what the spec says not to use. Not
   loaded here at all. Instead, genuine carryover Elo is regenerated fresh
   via elo_model.run_multi_season_elo() (no Vegas signal anywhere), which
   also gives the EXACT pre-game rating for every game, not just "the week
   before" - equivalent here anyway since each NFL team plays at most once
   per week, but more precise in general.

2. The spec's calculate_win_probability_from_elo() formula has no home-field
   term and claims home-field is "already baked into home_elo" from
   Component 1 - false. Component 1's team ratings never include a home
   adjustment; home_field_elo (+32.4, empirically fit) is applied
   separately at prediction time. Following the spec's literal formula
   would silently drop home-field advantage. Reuses elo_model.py's real
   calculate_expected_win_probability(elo_home, elo_away, home_field_elo)
   directly instead of re-deriving an incomplete one.

3. data/processed/game_predictions_2024_epa.csv doesn't exist (checked).
   Reconstructed the 2024 EPA holdout game-level predictions locally
   (_build_epa_holdout_game_predictions) using the same merge fit_game_
   level_epa_to_points() already does internally (train 2016-2023, holdout
   2024 - reproduces the real, already-verified +0.255 corr / 10.82 MAE
   baseline) rather than skipping the game-by-game comparison.

Disclosed, not fixed: Component 1's Elo hyperparameters (K=10, points_per_
win=45, home_field_elo=+32.4) were fit on 2015-2024, so 2024 isn't a
completely untouched holdout for the ELO MECHANISM itself - only for the
NEW probability->spread conversion fit here fresh on 2015-2023. Refitting
Elo hyperparameters with 2024 also excluded is out of scope for this test.
2025 (reported for context) is actually a cleaner double-holdout than 2024,
since it was untouched by both fits.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from constants import (
    ELO_K_FACTOR,
    ELO_POINTS_PER_WIN,
    ELO_HOME_FIELD_ADVANTAGE as ELO_HOME_FIELD,
    EPA_BASELINE_GAME_CORR_2024 as EPA_BASELINE_CORR_2024,
    EPA_BASELINE_GAME_MAE_2024 as EPA_BASELINE_MAE_2024,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

# Component 1's already-learned, real Elo hyperparameters (2015-2024 fit) -
# reused, not refit (see docstring). Centralized in constants.py (AUDIT_2026-07-27.md).
ELO_EARLIEST_SEASON = 2015


def calculate_win_probability_from_elo(elo_home, elo_away, home_field_elo=ELO_HOME_FIELD):
    """P(home win). Thin wrapper around elo_model's real function - see
    module docstring #2 for why the spec's own formula (no home term) would
    have been wrong."""
    from elo_model import calculate_expected_win_probability
    return calculate_expected_win_probability(elo_home, elo_away, home_field_elo)


def _load_game_results(seasons):
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    games = games[(games["season"].isin(list(seasons))) & (games["game_type"] == "REG")].copy()
    games["point_diff"] = games["home_score"] - games["away_score"]
    return games


def fit_probability_to_spread_conversion(train_seasons=range(2015, 2024)):
    """Fits win_prob -> point_diff on real 2015-2023 games (genuinely fresh,
    not reused from Component 1). Tries linear, logit, and normal/probit
    forms, picks the lowest-train-MAE one. Returns the fitted dict and
    pickles it."""
    from elo_model import run_multi_season_elo

    backtest_df, _, _ = run_multi_season_elo(
        range(ELO_EARLIEST_SEASON, max(train_seasons) + 1), k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
    backtest_df = backtest_df[backtest_df["season"].isin(list(train_seasons))]

    results = _load_game_results(train_seasons)[["game_id", "point_diff"]]
    merged = backtest_df.merge(results, on="game_id", how="inner")
    merged["win_prob"] = calculate_win_probability_from_elo(merged["home_elo_before"], merged["away_elo_before"])

    p = np.clip(merged["win_prob"].to_numpy(), 0.01, 0.99)
    point_diff = merged["point_diff"].to_numpy()

    candidates = {}
    a, b = np.polyfit(p - 0.5, point_diff, 1)
    candidates["linear"] = (a, b, np.mean(np.abs((a * (p - 0.5) + b) - point_diff)))

    logit_p = np.log(p / (1 - p))
    a, b = np.polyfit(logit_p, point_diff, 1)
    candidates["logit"] = (a, b, np.mean(np.abs((a * logit_p + b) - point_diff)))

    probit_p = norm.ppf(p)
    a, b = np.polyfit(probit_p, point_diff, 1)
    candidates["normal"] = (a, b, np.mean(np.abs((a * probit_p + b) - point_diff)))

    best_form = min(candidates, key=lambda k: candidates[k][2])
    a, b, train_mae = candidates[best_form]

    if best_form == "linear":
        pred = a * (p - 0.5) + b
    elif best_form == "logit":
        pred = a * logit_p + b
    else:
        pred = a * probit_p + b
    resid_std = float(np.std(point_diff - pred))

    print(f"[fit_probability_to_spread_conversion] train {min(train_seasons)}-{max(train_seasons)} (n={len(merged)}): "
          f"form comparison (train MAE) - " + ", ".join(f"{k}={v[2]:.3f}" for k, v in candidates.items()))
    print(f"[fit_probability_to_spread_conversion] winner: {best_form} (a={a:.3f}, b={b:.3f}, "
          f"train MAE={train_mae:.3f} pts, resid_std={resid_std:.3f} pts)")

    fitted_model = {"form": best_form, "a": float(a), "b": float(b), "resid_std": resid_std,
                     "train_mae": float(train_mae), "k_factor": ELO_K_FACTOR,
                     "points_per_win": ELO_POINTS_PER_WIN, "home_field_elo": ELO_HOME_FIELD}
    with open(os.path.join(PROCESSED_DIR, "elo_probability_to_spread_model.pkl"), "wb") as f:
        pickle.dump(fitted_model, f)
    return fitted_model


def predict_game_spread_from_elo(elo_home, elo_away, fitted_model):
    p = np.clip(calculate_win_probability_from_elo(elo_home, elo_away), 0.01, 0.99)
    form, a, b = fitted_model["form"], fitted_model["a"], fitted_model["b"]
    if form == "linear":
        return a * (p - 0.5) + b
    elif form == "logit":
        return a * np.log(p / (1 - p)) + b
    else:
        return a * norm.ppf(p) + b


def generate_elo_game_spreads(season, fitted_model):
    """Real pre-game Elo -> predicted spread + 90% CI (from the conversion
    model's own residual std), for every REG game of `season`. Seasons with
    real completed games (<=2025) use the exact chained pre-game rating from
    run_multi_season_elo's game-by-game backtest; season=2026 (no games
    played) uses the single preseason carryover snapshot against the full
    schedule, same static-preseason convention as every other 2026
    deliverable in this project."""
    from elo_model import run_multi_season_elo
    from game_predictions import _load_schedule_for_season

    if season <= 2025:
        backtest_df, _, _ = run_multi_season_elo(range(ELO_EARLIEST_SEASON, season + 1),
                                                   k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
        games = backtest_df[backtest_df["season"] == season].copy()
        games["home_elo"] = games["home_elo_before"]
        games["away_elo"] = games["away_elo_before"]
    else:
        _, ratings_at_season_start, _ = run_multi_season_elo(range(ELO_EARLIEST_SEASON, season),
                                                               k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
        preseason = ratings_at_season_start[season - 1] if (season - 1) in ratings_at_season_start else \
            ratings_at_season_start[max(ratings_at_season_start)]
        # roll the last known ratings forward one more (unplayed) season boundary regression
        regressed = {t: r + (1.0 / 3.0) * (1500.0 - r) for t, r in preseason.items()}
        schedule = _load_schedule_for_season(season)
        reg = schedule[schedule["game_type"] == "REG"].copy()
        reg["home_elo"] = reg["home_team"].map(regressed)
        reg["away_elo"] = reg["away_team"].map(regressed)
        games = reg[["week", "home_team", "away_team", "home_elo", "away_elo"]].copy()
        games["game_id"] = games["home_team"] + "_" + games["away_team"] + "_" + games["week"].astype(str)
        games["season"] = season

    games["predicted_spread"] = predict_game_spread_from_elo(games["home_elo"], games["away_elo"], fitted_model)
    band = 1.645 * fitted_model["resid_std"]
    games["ci_low_90"] = games["predicted_spread"] - band
    games["ci_high_90"] = games["predicted_spread"] + band

    return games[["game_id", "season", "week", "home_team", "away_team", "home_elo", "away_elo",
                   "predicted_spread", "ci_low_90", "ci_high_90"]].reset_index(drop=True)


def validate_elo_game_spreads(holdout_season=2024, fitted_model=None):
    """Real validation against the completed holdout season, plus real 2025
    for context (see module docstring - 2025 is actually the cleaner
    double-holdout of the two)."""
    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()

    def _score(season):
        preds = generate_elo_game_spreads(season, fitted_model)
        actual = _load_game_results([season])[["game_id", "point_diff"]]
        merged = preds.merge(actual, on="game_id", how="inner")
        corr = merged["predicted_spread"].corr(merged["point_diff"])
        mae = float(np.mean(np.abs(merged["predicted_spread"] - merged["point_diff"])))
        coverage = float(((merged["point_diff"] >= merged["ci_low_90"]) &
                           (merged["point_diff"] <= merged["ci_high_90"])).mean())
        return merged, corr, mae, coverage

    merged_holdout, corr, mae, coverage = _score(holdout_season)
    merged_2025, corr_2025, mae_2025, coverage_2025 = _score(2025)

    print(f"\n{'=' * 70}\nELO GAME SPREAD VALIDATION (holdout {holdout_season}, form={fitted_model['form']})\n{'=' * 70}")
    print(f"Elo   ({holdout_season} holdout): corr={corr:+.3f} MAE={mae:.2f} pts | CI coverage={coverage:.1%} (target 90%)")
    print(f"EPA baseline ({holdout_season})  : corr={EPA_BASELINE_CORR_2024:+.3f} MAE={EPA_BASELINE_MAE_2024:.2f} pts")
    delta_corr, delta_mae = corr - EPA_BASELINE_CORR_2024, mae - EPA_BASELINE_MAE_2024
    print(f"Delta: corr {delta_corr:+.3f} | MAE {delta_mae:+.2f} pts ({'Elo better' if delta_mae < 0 else 'EPA better'} on MAE)")
    print(f"\n2025 (context only, cleaner double-holdout - not part of any fit): "
          f"corr={corr_2025:+.3f} MAE={mae_2025:.2f} pts | CI coverage={coverage_2025:.1%}")

    if corr > EPA_BASELINE_CORR_2024 and mae < EPA_BASELINE_MAE_2024:
        verdict = "GREEN - Elo beats EPA on both corr and MAE"
    elif corr < 0.240 or mae > 11.0:
        verdict = "RED - EPA beats Elo"
    else:
        verdict = "MIXED - comparable, trade-offs between corr/MAE"
    print(f"\nVerdict: {verdict}")

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    merged_holdout.to_csv(os.path.join(PROCESSED_DIR, f"elo_game_spreads_{holdout_season}.csv"), index=False, encoding="utf-8")
    print(f"\nSaved data/processed/elo_game_spreads_{holdout_season}.csv")
    print("=" * 70)

    return {"corr": corr, "mae": mae, "coverage": coverage, "corr_2025": corr_2025, "mae_2025": mae_2025,
            "verdict": verdict, "fitted_model": fitted_model, "merged_holdout": merged_holdout}


def generate_elo_game_predictions_2026(fitted_model=None):
    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()
    preds = generate_elo_game_spreads(2026, fitted_model)
    out_path = os.path.join(PROCESSED_DIR, "elo_game_predictions_2026.csv")
    preds.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path} ({len(preds)} games)")
    return preds


def _build_epa_holdout_game_predictions(season=2024, train_seasons=range(2016, 2024)):
    """Reconstructs game_predictions_2024_epa.csv's real content, since that
    file doesn't exist on disk (see module docstring #3) - same merge
    fit_game_level_epa_to_points() already does internally."""
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa
    from game_predictions import fit_game_level_epa_to_points

    team_off = compute_team_offense_epa()
    team_def = compute_team_defense_epa()
    game_results = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    slope, intercept, resid_std, _ = fit_game_level_epa_to_points(
        team_off, team_def, game_results, train_seasons=train_seasons, holdout_season=season)

    reg = game_results[(game_results["game_type"] == "REG")].copy()
    off = team_off[["team", "season", "off_epa"]]
    defn = team_def[["team", "season", "def_epa_allowed"]]
    reg = reg.merge(off.rename(columns={"team": "home_team", "off_epa": "home_off_epa"}), on=["home_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "home_team", "def_epa_allowed": "home_def_epa"}), on=["home_team", "season"])
    reg = reg.merge(off.rename(columns={"team": "away_team", "off_epa": "away_off_epa"}), on=["away_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "away_team", "def_epa_allowed": "away_def_epa"}), on=["away_team", "season"])
    reg["epa_diff"] = (reg["home_off_epa"] - reg["away_def_epa"]) - (reg["away_off_epa"] - reg["home_def_epa"])
    reg["point_diff"] = reg["home_score"] - reg["away_score"]

    hold = reg[reg["season"] == season].copy()
    hold["epa_predicted_spread"] = slope * hold["epa_diff"] + intercept
    return hold[["game_id", "season", "week", "home_team", "away_team", "point_diff", "epa_predicted_spread"]]


def compare_elo_vs_epa_detailed(holdout_season=2024, fitted_model=None):
    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()

    epa_preds = _build_epa_holdout_game_predictions(holdout_season)
    elo_preds = generate_elo_game_spreads(holdout_season, fitted_model)

    merged = epa_preds.merge(elo_preds[["game_id", "predicted_spread", "home_elo", "away_elo"]], on="game_id", how="inner")
    merged["epa_error"] = (merged["epa_predicted_spread"] - merged["point_diff"]).abs()
    merged["elo_error"] = (merged["predicted_spread"] - merged["point_diff"]).abs()
    merged["elo_won"] = merged["elo_error"] < merged["epa_error"]

    n = len(merged)
    elo_wins = int(merged["elo_won"].sum())
    print(f"\n{'=' * 70}\nELO vs. EPA GAME-BY-GAME COMPARISON ({holdout_season})\n{'=' * 70}")
    print(f"Elo won on {elo_wins}/{n} games | EPA won on {n - elo_wins}/{n} games")

    merged["elo_diff_abs"] = (merged["home_elo"] - merged["away_elo"]).abs()
    merged["elo_diff_bucket"] = pd.qcut(merged["elo_diff_abs"], 4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (biggest mismatch)"])
    bucket_summary = merged.groupby("elo_diff_bucket", observed=True)["elo_won"].mean()
    print(f"\nElo win rate by |Elo differential| quartile (does Elo do better on lopsided games?):")
    print(bucket_summary.to_string())

    biggest = merged.reindex((merged["epa_error"] - merged["elo_error"]).abs().sort_values(ascending=False).index).head(5)
    print(f"\nBiggest single-game divergences (|EPA error - Elo error|):")
    for _, row in biggest.iterrows():
        winner = "Elo" if row["elo_error"] < row["epa_error"] else "EPA"
        print(f"  {row['home_team']} vs {row['away_team']} (wk{row['week']}): actual={row['point_diff']:+.0f} | "
              f"Elo pred={row['predicted_spread']:+.1f} (err {row['elo_error']:.1f}) | "
              f"EPA pred={row['epa_predicted_spread']:+.1f} (err {row['epa_error']:.1f}) | {winner} closer")

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    merged.to_csv(os.path.join(DIAGNOSTIC_DIR, f"elo_game_validation_{holdout_season}.csv"), index=False, encoding="utf-8")
    summary_lines = [
        f"Elo vs EPA game-by-game comparison ({holdout_season})",
        f"Elo won {elo_wins}/{n} games ({elo_wins / n:.1%})",
        f"Elo win rate by |Elo diff| quartile:\n{bucket_summary.to_string()}",
    ]
    with open(os.path.join(DIAGNOSTIC_DIR, "elo_vs_epa_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines))
    print(f"\nSaved data/diagnostic/elo_game_validation_{holdout_season}.csv, elo_vs_epa_comparison.txt")
    print("=" * 70)

    return merged, {"elo_wins": elo_wins, "n": n, "bucket_summary": bucket_summary}


if __name__ == "__main__":
    fitted = fit_probability_to_spread_conversion()
    validate_elo_game_spreads(2024, fitted)
    compare_elo_vs_epa_detailed(2024, fitted)
    generate_elo_game_predictions_2026(fitted)
