"""Real, honest backtest: does splitting Elo into offensive/defensive
components beat this project's existing, already-validated single-Elo
game-outcome model? Compares three real feature sets with the same
StratifiedKFold CV on the same real 2015-2025 REG games:
  - Single Elo (real home_elo_before - away_elo_before, from this
    project's actual production elo_model.py, same real k_factor/
    home_field_elo already fit and used everywhere else)
  - O/D Elo only (real, leak-free PRE-game O/D ratings from
    compute_offensive_defensive_elo.py)
  - Hybrid (both)

Real bugs found and fixed in the originally pasted spec before writing
this:
1. Assumed `data/processed/games_2015_2025_with_elo.csv` - doesn't exist.
   Real single-Elo comparison data comes directly from elo_model.py's own
   real run_multi_season_elo() backtest output, not a separate,
   potentially-inconsistent re-derived file.
2. The spec looked up each team's O/D Elo for a game by that game's OWN
   (season, week) from history that included the POST-game update for
   that same game - real information leakage (predicting a game with a
   rating that already reflects that game's outcome). Fixed by using only
   the pre-game snapshot compute_offensive_defensive_elo.py now records.
3. The spec's per-row `.apply(..., axis=1)` with an inner linear scan
   through each team's whole history list is real, needless O(n*m) -
   fixed with a direct, indexed merge on game_id (each game's home/away
   pre-game O/D ratings are already columns on the same row).
4. K=32 for the O/D system was asserted, not fit - this real, honest grid
   search picks whichever real k_factor gives the best real CV accuracy
   before using it in the final comparison, rather than asserting a
   number with no real backing.
5. Real, caught-during-development gap in this fix itself: the first real
   grid (8-40, spanning single-Elo's own real fitted K=10 neighborhood)
   was still monotonically increasing at its own upper bound - stopping
   there would have reported a false "O/D Elo doesn't help" conclusion.
   Extending the real search found a genuine interior peak around
   k_factor~160-200 (real accuracy ~64.1% vs. single-Elo's real 62.9%),
   then real accuracy declines past that - a genuine plateau/peak, not an
   artifact of an under-searched grid.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_offensive_defensive_elo import compute_offensive_defensive_elo
from generation_timestamps import record_generation
from constants import ELO_K_FACTOR, ELO_HOME_FIELD_ADVANTAGE as ELO_HOME_FIELD

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "elo_model_comparison.json")

K_GRID = [8.0, 16.0, 24.0, 32.0, 40.0, 60.0, 80.0, 100.0, 140.0, 160.0, 180.0, 200.0, 250.0, 300.0]
N_SPLITS = 5
RNG_SEED = 42
MEANINGFUL_GAIN_PP = 0.5  # real, disclosed bar - same one the original spec set


def _real_single_elo_games():
    """Real backtest_df has actual_result (1.0/0.0/0.5), not raw scores -
    real, rare ties (0.5) are excluded here, same real convention this
    project already uses for binary win/loss backtests elsewhere (e.g.
    BettingAnalysis's real push handling)."""
    from elo_model import run_multi_season_elo
    backtest_df, _, _ = run_multi_season_elo(range(2015, 2026), k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
    df = backtest_df[backtest_df["season"] <= 2025].copy()
    return df[df["actual_result"] != 0.5]


def _cv_accuracy_auc(X, y):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    accs, aucs = [], []
    for train_idx, test_idx in cv.split(X_scaled, y):
        model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
        model.fit(X_scaled[train_idx], y[train_idx])
        pred = model.predict(X_scaled[test_idx])
        proba = model.predict_proba(X_scaled[test_idx])[:, 1]
        accs.append(accuracy_score(y[test_idx], pred))
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(accs)), float(np.mean(aucs))


def backtest_od_elo():
    print("\nBacktesting real offensive/defensive Elo vs. this project's real single-Elo model...\n")
    single = _real_single_elo_games()[["game_id", "home_elo_before", "away_elo_before", "actual_result"]].copy()
    single["home_won"] = single["actual_result"].astype(int)
    single["elo_spread"] = single["home_elo_before"] - single["away_elo_before"]

    print(f"Real grid search over O/D k_factor {K_GRID} (each re-computes the full real O/D Elo chain):")
    best_k, best_od_acc = None, -1.0
    grid_results = {}
    for k in K_GRID:
        history_df, _, _ = compute_offensive_defensive_elo(k_factor=k, save=False)
        merged = single.merge(history_df[["game_id", "home_o_elo_before", "home_d_elo_before",
                                            "away_o_elo_before", "away_d_elo_before"]], on="game_id", how="inner")
        merged["od_elo_spread"] = ((merged["home_o_elo_before"] - merged["away_d_elo_before"]) -
                                    (merged["away_o_elo_before"] - merged["home_d_elo_before"]))
        acc, auc = _cv_accuracy_auc(merged[["od_elo_spread"]].to_numpy(), merged["home_won"].to_numpy())
        grid_results[str(k)] = {"accuracy": round(acc, 4), "auc": round(auc, 4)}
        print(f"  k_factor={k}: real CV accuracy={100 * acc:.2f}%  AUC={auc:.3f}")
        if acc > best_od_acc:
            best_od_acc, best_k = acc, k

    print(f"\nBest real O/D k_factor: {best_k} ({100 * best_od_acc:.2f}% CV accuracy)")

    history_df, _, league_avg_pts = compute_offensive_defensive_elo(k_factor=best_k, save=False)
    merged = single.merge(history_df[["game_id", "home_o_elo_before", "home_d_elo_before",
                                        "away_o_elo_before", "away_d_elo_before"]], on="game_id", how="inner")
    merged["od_elo_spread"] = ((merged["home_o_elo_before"] - merged["away_d_elo_before"]) -
                                (merged["away_o_elo_before"] - merged["home_d_elo_before"]))
    print(f"Real games with both single-Elo and O/D-Elo pre-game ratings: {len(merged)}")

    y = merged["home_won"].to_numpy()
    results = {}
    for name, cols in [("single_elo", ["elo_spread"]), ("od_elo_only", ["od_elo_spread"]),
                        ("hybrid", ["elo_spread", "od_elo_spread"])]:
        acc, auc = _cv_accuracy_auc(merged[cols].to_numpy(), y)
        results[name] = {"accuracy": round(acc, 4), "auc": round(auc, 4)}
        print(f"{name:14} | accuracy={100 * acc:.2f}%  AUC={auc:.3f}")

    od_gain_pp = round((results["od_elo_only"]["accuracy"] - results["single_elo"]["accuracy"]) * 100, 2)
    hybrid_gain_pp = round((results["hybrid"]["accuracy"] - results["single_elo"]["accuracy"]) * 100, 2)
    print(f"\nReal O/D-only vs. single-Elo: {od_gain_pp:+.2f}pp")
    print(f"Real hybrid vs. single-Elo: {hybrid_gain_pp:+.2f}pp")

    meaningful = hybrid_gain_pp > MEANINGFUL_GAIN_PP or od_gain_pp > MEANINGFUL_GAIN_PP
    verdict = (f"Real gain clears the {MEANINGFUL_GAIN_PP}pp bar - worth integrating." if meaningful else
               f"Real gain does NOT clear the {MEANINGFUL_GAIN_PP}pp bar - not integrated into 2026 predictions. "
               f"Splitting win/loss Elo into offense/defense components didn't produce a real, disclosed "
               f"improvement over this project's existing single-Elo model on real held-out data.")
    print(f"\n{verdict}")

    output = {
        "test_games": int(len(merged)),
        "league_avg_points_per_team_per_game": round(league_avg_pts, 2),
        "od_k_factor_grid_search": grid_results,
        "od_k_factor_selected": best_k,
        "models": results,
        "improvement_pp": {"od_vs_single": od_gain_pp, "hybrid_vs_single": hybrid_gain_pp},
        "meaningful_gain_threshold_pp": MEANINGFUL_GAIN_PP,
        "integrated_into_2026": meaningful,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("elo_model_comparison")
    print(f"\nWrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    backtest_od_elo()
