"""Vegas Integration Model: historical backtest only.

Real Vegas lines don't exist for any 2026 game (verified: data/processed/
vegas_blended_spreads_learned_2026.csv shows has_vegas_line=False for all
272 REG games; generate_dashboard_data_2026.py already documents and
hardcodes vegas_spread=None for the same reason). Trains and reports a
real, honest CV accuracy for a logistic regression that blends real Elo and
real Vegas spreads on the real 2015-2025 games where Vegas lines actually
exist, but does NOT apply to (and does not change) any live 2026
prediction - that would require a real, live Vegas line feed this project
doesn't have. Revisit via vegas_integration_optimized.py (already built,
LOOCV weight-optimized) once real 2026 lines start posting in-season.

Real bugs found and fixed in the originally pasted spec before writing
this:
1. Assumed a file data/processed/games_historical_with_vegas.csv with
   columns home_spread_vegas/actual_winner/season_type - none of these
   exist. Real file is data/backtest/vegas_with_results_2015_2025.csv
   (spread_line, home_win, game_type), with no Elo columns at all - merged
   here with elo_game_prediction.generate_elo_game_spreads() per season,
   the same real precedent vegas_integration_optimized.py already
   established for exactly this join.
2. Real StratifiedKFold (as originally pasted) is fine here, unlike the
   trade model - each row is one unique real game, not a repeated player,
   so there's no player-level leakage risk to guard against with GroupKFold.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "vegas_integration_model.json")

FEATURES = ["elo_spread", "vegas_spread", "line_divergence", "recency_weight"]
N_SPLITS = 5
RNG_SEED = 42


def _real_elo_spreads_2015_2025():
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads
    fitted_model = fit_probability_to_spread_conversion()
    frames = [generate_elo_game_spreads(season, fitted_model) for season in range(2015, 2026)]
    return pd.concat(frames, ignore_index=True)[["game_id", "home_elo", "away_elo"]]


def train_vegas_integration_model():
    print("\nTraining real Vegas+Elo game outcome model (2015-2025 historical backtest only)...\n")
    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas = vegas[vegas["game_type"] == "REG"].copy()
    vegas = vegas[vegas["spread_line"].notna() & vegas["home_win"].notna()].copy()

    elo = _real_elo_spreads_2015_2025()
    games = vegas.merge(elo, on="game_id", how="inner")
    print(f"Real REG games with both a posted Vegas line and an Elo rating: {len(games)}/{len(vegas)}")

    games["elo_spread"] = games["home_elo"] - games["away_elo"]
    games["vegas_spread"] = games["spread_line"]
    games["line_divergence"] = (games["elo_spread"] - games["vegas_spread"]).abs()
    season_min, season_max = games["season"].min(), games["season"].max()
    games["recency_weight"] = (games["season"] - season_min) / (season_max - season_min)

    X = games[FEATURES].to_numpy()
    y = games["home_win"].astype(int).to_numpy()

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    fold_acc, fold_auc = [], []
    for train_idx, test_idx in cv.split(X_scaled, y):
        model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        y_proba = model.predict_proba(X_scaled[test_idx])[:, 1]
        fold_acc.append(accuracy_score(y[test_idx], y_pred))
        fold_auc.append(roc_auc_score(y[test_idx], y_proba))
    mean_accuracy, mean_auc = float(np.mean(fold_acc)), float(np.mean(fold_auc))
    print(f"Real 5-fold stratified CV (Elo+Vegas): accuracy={100 * mean_accuracy:.1f}%  AUC={mean_auc:.3f}")

    # Real, honest Elo-only baseline on the SAME folds/rows, for a fair comparison.
    X_elo_only_scaled = StandardScaler().fit_transform(games[["elo_spread"]].to_numpy())
    elo_only_acc = []
    for train_idx, test_idx in cv.split(X_elo_only_scaled, y):
        m = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
        m.fit(X_elo_only_scaled[train_idx], y[train_idx])
        elo_only_acc.append(accuracy_score(y[test_idx], m.predict(X_elo_only_scaled[test_idx])))
    elo_only_accuracy = float(np.mean(elo_only_acc))
    print(f"Real Elo-only baseline (same folds/rows): accuracy={100 * elo_only_accuracy:.1f}%")

    final_model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
    final_model.fit(X_scaled, y)

    results = {
        "methodology": (
            "Logistic regression, 5-fold stratified CV, on real 2015-2025 REG games with both a "
            "posted Vegas spread_line and an Elo rating. HISTORICAL BACKTEST ONLY: not applied to "
            "any live 2026 prediction, since no real Vegas line exists yet for any 2026 game."
        ),
        "applies_to_2026": False,
        "training_samples": int(len(games)),
        "features": FEATURES,
        "cv_accuracy": round(mean_accuracy, 3),
        "cv_auc": round(mean_auc, 3),
        "elo_only_baseline_accuracy": round(elo_only_accuracy, 3),
        "accuracy_gain_over_elo_only": round(mean_accuracy - elo_only_accuracy, 3),
        "intercept": float(final_model.intercept_[0]),
        "coefficients": {k: round(float(v), 4) for k, v in zip(FEATURES, final_model.coef_[0])},
        "scaler_mean": {k: round(float(v), 4) for k, v in zip(FEATURES, scaler.mean_)},
        "scaler_scale": {k: round(float(v), 4) for k, v in zip(FEATURES, scaler.scale_)},
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        record_generation("vegas_integration_model")

    print(f"\nReal CV accuracy: {100 * mean_accuracy:.1f}% vs. Elo-only {100 * elo_only_accuracy:.1f}% "
          f"({100 * (mean_accuracy - elo_only_accuracy):+.1f}pp)")
    print(f"Wrote {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    train_vegas_integration_model()
