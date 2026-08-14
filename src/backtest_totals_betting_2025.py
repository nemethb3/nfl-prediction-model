"""Real, honest backtest of an over/under edge-betting strategy on real
2025 games: bet the direction our predicted_total disagrees with the real
Vegas total by more than a threshold. Per this task's own scoping decision
(user confirmed: disclose honestly either way, don't cherry-pick a
flattering threshold) - this refits the point-totals model on real
2015-2024 ONLY and scores real 2025 as a genuine holdout, rather than
reusing the main model's in-sample or same-season predictions, which would
double-dip the same rows used to pick a "best" threshold.

This project already has a directly analogous, disclosed real finding for
SPREADS: GameCard.js's own "Model vs. Vegas" section states real
edge_detection.py found betting on our-model-vs-Vegas disagreements
produced -36% ROI (actively harmful). This script tests the same *shape*
of strategy for TOTALS and reports whatever the real result is - including
if it's flat, negative, or an artifact of a small sample - not a
cherry-picked "best" threshold presented as a validated edge."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from generation_timestamps import record_generation
from train_point_totals_model import FEATURES, _real_elo_spreads_2015_2025

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "totals_betting_backtest_2025.json")

HOLDOUT_SEASON = 2025
TRAIN_SEASONS = range(2015, 2025)  # strictly excludes the real holdout season
THRESHOLDS = np.arange(0.5, 5.5, 0.5)
MIN_BETS_FOR_CONSIDERATION = 10  # real, disclosed minimum before a threshold's win rate is even reported as a candidate


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
    return games


def backtest_totals_betting():
    print(f"\nBacktesting real over/under edge betting on real {HOLDOUT_SEASON} "
          f"(true holdout, model refit on {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} only)...\n")
    games = _real_games_with_features()
    train = games[games["season"].isin(list(TRAIN_SEASONS))]
    holdout = games[games["season"] == HOLDOUT_SEASON].copy()
    print(f"Real train rows: {len(train)} | real holdout ({HOLDOUT_SEASON}) rows: {len(holdout)}")

    scaler = StandardScaler().fit(train[FEATURES].to_numpy())
    model = LinearRegression().fit(scaler.transform(train[FEATURES].to_numpy()), train["total"].to_numpy())
    holdout["predicted_total"] = model.predict(scaler.transform(holdout[FEATURES].to_numpy()))
    holdout["edge"] = holdout["predicted_total"] - holdout["total_line"]
    holdout["actual_over"] = holdout["total"] > holdout["total_line"]

    results_by_threshold = {}
    for threshold in THRESHOLDS:
        bets = holdout[holdout["edge"].abs() > threshold].copy()
        bets["bet_over"] = bets["edge"] > 0
        bets["win"] = np.where(bets["total"] == bets["total_line"], None,
                                bets["bet_over"] == bets["actual_over"])
        decided = bets[bets["win"].notna()]
        wins = int(decided["win"].sum())
        losses = int((~decided["win"].astype(bool)).sum())
        total_bets = wins + losses
        win_rate = wins / total_bets if total_bets > 0 else None
        # Real -110 odds: risk 1 unit to win 100/110 = 0.909 units; a loss costs the full 1 unit
        # staked. (The originally pasted spec's comment - "win $110 on $100 bet" - describes the
        # payout backwards; caught by cross-checking against the real 52.4% break-even win rate
        # this same math has to produce, and fixed before trusting the resulting ROI numbers.)
        units_won = wins * (100 / 110) - losses * 1.0
        roi = (units_won / total_bets) * 100 if total_bets > 0 else None
        results_by_threshold[str(round(float(threshold), 1))] = {
            "threshold": round(float(threshold), 1),
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "units_won": round(float(units_won), 2),
            "roi_pct": round(roi, 2) if roi is not None else None,
        }
        win_str = f"{100 * win_rate:.1f}%" if win_rate is not None else "n/a"
        roi_str = f"{roi:+.1f}%" if roi is not None else "n/a"
        print(f"  edge>{threshold:.1f} pts: {total_bets} real bets, {wins}W-{losses}L "
              f"({win_str} win rate, {roi_str} ROI)")

    candidates = [r for r in results_by_threshold.values() if r["total_bets"] >= MIN_BETS_FOR_CONSIDERATION]
    best = max(candidates, key=lambda r: r["roi_pct"]) if candidates else None

    breakeven_note = (
        "Break-even win rate at real -110 odds is 52.4%. A 'best' threshold chosen by picking the "
        "highest ROI across several thresholds tested on the same one real holdout season is a real "
        "multiple-comparisons risk - it is expected to look better than the true underlying rate even "
        "with zero real edge, especially at these real, small per-threshold sample sizes (n=10-52). "
        "Not a validated strategy or a betting recommendation - see GameCard.js's own disclosed "
        "-36% ROI finding for the analogous real spread-disagreement strategy."
    )
    print(f"\n{breakeven_note}")
    if best:
        print(f"Highest-ROI real threshold tested: {best['threshold']} pts "
              f"({best['total_bets']} bets, {100 * best['win_rate']:.1f}% win rate, {best['roi_pct']:+.1f}% ROI) "
              f"- reported for disclosure, not recommended.")
    else:
        print(f"No real threshold reached the minimum {MIN_BETS_FOR_CONSIDERATION}-bet sample size.")

    output = {
        "strategy": "over_under_edge_betting",
        "methodology": (
            f"Point-totals model refit on real {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} only, scored "
            f"on real {HOLDOUT_SEASON} as a genuine holdout (not reused from the main model). Bet the "
            "direction (OVER/UNDER) our predicted_total disagrees with the real Vegas total_line by "
            "more than a threshold, at real -110 odds. NOT a betting recommendation."
        ),
        "disclosure": breakeven_note,
        "holdout_season": HOLDOUT_SEASON,
        "holdout_games": int(len(holdout)),
        "results_by_threshold": results_by_threshold,
        "highest_roi_threshold_for_disclosure": best,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("totals_betting_backtest_2025")
    print(f"\nWrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    backtest_totals_betting()
