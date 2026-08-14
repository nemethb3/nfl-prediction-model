"""Real point-totals projection model: predicts each real game's combined
score (home + away points) from real Elo-derived and schedule-position
features, no Vegas signal used as an input feature (only as the real,
historical comparison point for calibration and the OOF directional
accuracy this module reports).

Real design change from the originally pasted spec, made before writing
this: the spec's Step 1 trains a LOGISTIC REGRESSION classifier (target =
actual_total > vegas_total, no numeric output), but Step 4's betting
backtest needs `our_total - vegas_total` as a real point-value edge - a
classifier has no such output. Also, a classifier trained with vegas_total
baked into its own target has no way to score any real game that lacks a
posted Vegas line (219/272 real 2026 games right now - only weeks 1-4 have
one). Built as a real LINEAR REGRESSION instead: predicts the actual
combined score directly from Elo/week/season only, so it can score every
real game regardless of whether a Vegas total exists for it, and produces
a real numeric prediction the betting backtest can actually use.

Real bugs found and fixed in the originally pasted spec before writing
this (same pattern as train_vegas_integration_model.py):
1. Assumed files (games_historical_2015_2025.csv, elo_by_week_2015_2025.csv)
   don't exist. Real source is data/backtest/vegas_with_results_2015_2025.csv
   (total_line, total, game_type columns), merged with
   elo_game_prediction.generate_elo_game_spreads() per season - the same
   real precedent already established for this exact join.
2. Confidence/alert calibration is computed from real OUT-OF-FOLD
   predictions (5-fold KFold), not in-sample residuals - calibrating
   "does a bigger edge mean better real accuracy" on the same rows the
   model was fit on would overstate the real relationship.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "point_totals_model.json")

FEATURES = ["elo_sum", "elo_diff", "week_norm", "season_norm"]
N_SPLITS = 5
RNG_SEED = 42
ALERT_COVERAGE_TARGET = (0.05, 0.10)  # real, disclosed target range for alert flag coverage


def _real_elo_spreads_2015_2025():
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads
    fitted_model = fit_probability_to_spread_conversion()
    frames = [generate_elo_game_spreads(season, fitted_model) for season in range(2015, 2026)]
    return pd.concat(frames, ignore_index=True)[["game_id", "home_elo", "away_elo"]]


def _real_games_with_features():
    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas = vegas[vegas["game_type"] == "REG"].copy()
    vegas = vegas[vegas["total_line"].notna() & vegas["total"].notna()].copy()

    elo = _real_elo_spreads_2015_2025()
    games = vegas.merge(elo, on="game_id", how="inner")

    games["elo_sum"] = games["home_elo"] + games["away_elo"]
    games["elo_diff"] = (games["home_elo"] - games["away_elo"]).abs()
    week_min, week_max = games["week"].min(), games["week"].max()
    games["week_norm"] = (games["week"] - week_min) / (week_max - week_min)
    season_min, season_max = games["season"].min(), games["season"].max()
    games["season_norm"] = (games["season"] - season_min) / (season_max - season_min)
    return games, (week_min, week_max), (season_min, season_max)


def train_point_totals_model():
    print("\nTraining real point-totals regression model (2015-2025)...\n")
    games, (week_min, week_max), (season_min, season_max) = _real_games_with_features()
    print(f"Real REG games with a posted Vegas total and an Elo rating: {len(games)}")

    X = games[FEATURES].to_numpy()
    y_total = games["total"].to_numpy()
    vegas_total = games["total_line"].to_numpy()
    actual_over = (y_total > vegas_total).astype(int)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    oof_pred = np.full(len(games), np.nan)
    fold_mae = []
    for train_idx, test_idx in kf.split(X_scaled):
        model = LinearRegression()
        model.fit(X_scaled[train_idx], y_total[train_idx])
        pred = model.predict(X_scaled[test_idx])
        oof_pred[test_idx] = pred
        fold_mae.append(float(np.mean(np.abs(pred - y_total[test_idx]))))
    mean_mae = float(np.mean(fold_mae))
    ss_res = float(np.sum((y_total - oof_pred) ** 2))
    ss_tot = float(np.sum((y_total - y_total.mean()) ** 2))
    oof_r2 = 1 - ss_res / ss_tot
    print(f"Real 5-fold OOF: MAE={mean_mae:.2f} points, R^2={oof_r2:.3f}")

    # Real, honest OOF directional accuracy: did our OOF prediction agree with
    # the real actual result on which side of the real Vegas total it fell?
    oof_edge = oof_pred - vegas_total
    oof_predicted_over = (oof_edge > 0).astype(int)
    oof_direction_correct = (oof_predicted_over == actual_over)
    overall_directional_accuracy = float(oof_direction_correct.mean())
    print(f"Real OOF directional accuracy (vs. real Vegas total): {100 * overall_directional_accuracy:.1f}%")

    # Real, OOF-only alert-threshold calibration: sweep |edge| thresholds,
    # report real coverage and real directional accuracy of the flagged
    # subset at each, pick the smallest threshold whose real coverage falls
    # in the real target range.
    print("\nReal alert-threshold calibration (OOF only):")
    calibration = []
    for edge_threshold in np.arange(0.5, 5.5, 0.5):
        flagged = np.abs(oof_edge) >= edge_threshold
        coverage = float(flagged.mean())
        acc = float(oof_direction_correct[flagged].mean()) if flagged.sum() > 0 else None
        calibration.append({"edge_threshold": round(float(edge_threshold), 1), "coverage": round(coverage, 4),
                             "n_flagged": int(flagged.sum()),
                             "directional_accuracy": round(acc, 4) if acc is not None else None})
        acc_str = f"{100 * acc:.1f}%" if acc is not None else "n/a"
        print(f"  edge>={edge_threshold:.1f} pts: coverage={100 * coverage:.1f}% (n={int(flagged.sum())}) "
              f"real directional accuracy={acc_str}")

    in_range = [c for c in calibration if ALERT_COVERAGE_TARGET[0] <= c["coverage"] <= ALERT_COVERAGE_TARGET[1]]
    alert_entry = in_range[0] if in_range else min(
        calibration, key=lambda c: abs(c["coverage"] - sum(ALERT_COVERAGE_TARGET) / 2))
    print(f"\nSelected real alert threshold: {alert_entry['edge_threshold']} pts "
          f"({100 * alert_entry['coverage']:.1f}% real coverage, "
          f"{100 * alert_entry['directional_accuracy']:.1f}% real OOF directional accuracy)")

    final_model = LinearRegression()
    final_model.fit(X_scaled, y_total)

    results = {
        "methodology": (
            "Linear regression (5-fold KFold, real out-of-fold evaluation), predicting each real "
            "game's combined score from Elo-derived and schedule-position features only (no Vegas "
            "signal as an input feature) - trained on real 2015-2025 REG games with a posted Vegas "
            "total. Real Vegas total is used only as the historical comparison point for the "
            "directional-accuracy and alert-threshold numbers below, never as a model input, so "
            "this model can score any real game regardless of whether a Vegas total exists for it."
        ),
        "training_samples": int(len(games)),
        "features": FEATURES,
        "week_range": [int(week_min), int(week_max)],
        "season_range": [int(season_min), int(season_max)],
        "oof_mae_points": round(mean_mae, 3),
        "oof_r2": round(oof_r2, 3),
        "oof_directional_accuracy": round(overall_directional_accuracy, 4),
        "alert_edge_threshold_points": alert_entry["edge_threshold"],
        "alert_coverage": alert_entry["coverage"],
        "alert_directional_accuracy": alert_entry["directional_accuracy"],
        "threshold_calibration": calibration,
        "intercept": float(final_model.intercept_),
        "coefficients": {k: round(float(v), 4) for k, v in zip(FEATURES, final_model.coef_)},
        "scaler_mean": {k: round(float(v), 4) for k, v in zip(FEATURES, scaler.mean_)},
        "scaler_scale": {k: round(float(v), 4) for k, v in zip(FEATURES, scaler.scale_)},
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        record_generation("point_totals_model")
    print(f"\nWrote {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    train_point_totals_model()
