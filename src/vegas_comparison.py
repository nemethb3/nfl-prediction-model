"""Phase 3 Rebuild Task 5 & Phase 3 Redesign Subtask 3: Vegas Comparison & Edge Detection.

Split out of win_projection.py into this file (Master Plan Phase 4 Task 4.2 -
see AUDIT_2026-07-25.md Technical Debt #5 and #8). This was an empty
placeholder stub from the original project scaffolding with real
functionality living elsewhere (win_projection.py) instead - repurposed here
as its actual, correctly-named home rather than deleted, since the name was
already right. No behavior change from the original code.

Season-level Vegas validation (Task 5, moneyline_to_implied_prob/
compute_vegas_implied_wins/validate_model_against_vegas) corrects the
spec's implied-win-probability formula. It used `total_line` (the spec's
"over_under") - a single GAME's combined-score betting line (e.g. 47.5
points) - fed through `0.5 + (over_under - 8.5) / 17`, treating a per-game
point total as if it were a preseason season win total. Those are unrelated
Vegas markets and the arithmetic doesn't correspond to any real quantity
(confirmed by inspecting the actual vegas_with_results_2015_2025.csv columns
before building - there's no season win-total line in this dataset at all).
vegas_with_results_2015_2025.csv DOES have real per-game moneylines
(home_moneyline/away_moneyline, 0 missing for all 272 2025 REG games), which
convert directly to real market-implied win probability per game. Devigging
(normalizing the two sides to sum to 1, removing the bookmaker's built-in
margin) and summing each team's probability across their 17 real games gives
a genuine Vegas-implied season win total.

Game-level Vegas comparison (Subtask 3, vegas_comparison_framework/
identify_edges): load_vegas_lines_all_2026() doesn't exist anywhere - real
2026 lines just live directly on schedules_2026.csv (spread_line/total_line/
moneylines), same as every other season. spread_line sign convention
verified against real historical blowouts before use (not assumed):
positive spread_line = home team favored, matching build_game_prediction_
engine's own expected_spread convention - no sign flip needed.
"""

import os

import numpy as np
import pandas as pd

from backtest import load_2025_actual_results
from epa_to_wins import TARGET_SEASON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")


def moneyline_to_implied_prob(moneyline):
    ml = moneyline.to_numpy()
    with np.errstate(divide="ignore"):  # np.where evaluates both branches over the whole array;
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))  # a +100 ml hits the unused branch's /0


def compute_vegas_implied_wins(vegas, season=TARGET_SEASON):
    games = vegas[(vegas["season"] == season) & (vegas["game_type"] == "REG")].copy()
    games["home_implied_raw"] = moneyline_to_implied_prob(games["home_moneyline"])
    games["away_implied_raw"] = moneyline_to_implied_prob(games["away_moneyline"])
    vig_total = games["home_implied_raw"] + games["away_implied_raw"]
    games["home_win_prob"] = games["home_implied_raw"] / vig_total
    games["away_win_prob"] = games["away_implied_raw"] / vig_total

    home = games[["home_team", "home_win_prob"]].rename(columns={"home_team": "team", "home_win_prob": "win_prob"})
    away = games[["away_team", "away_win_prob"]].rename(columns={"away_team": "team", "away_win_prob": "win_prob"})
    long = pd.concat([home, away], ignore_index=True)

    vegas_wins = long.groupby("team")["win_prob"].sum().rename("vegas_implied_wins").reset_index()
    n_games = long.groupby("team").size().rename("vegas_n_games").reset_index()
    return vegas_wins.merge(n_games, on="team")


def validate_model_against_vegas():
    projections = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{TARGET_SEASON}.csv"))
    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas_wins = compute_vegas_implied_wins(vegas)
    actual = load_2025_actual_results()

    comparison = projections.merge(vegas_wins, on="team", how="inner").merge(
        actual[["team", "actual_wins"]], on="team", how="inner")

    print("=" * 70 + "\nMODEL vs. VEGAS vs. REAL 2025 RESULTS\n" + "=" * 70)

    corr_model_vegas = comparison["projected_wins"].corr(comparison["vegas_implied_wins"])
    corr_model_actual = comparison["projected_wins"].corr(comparison["actual_wins"])
    corr_vegas_actual = comparison["vegas_implied_wins"].corr(comparison["actual_wins"])
    mae_model_actual = np.mean(np.abs(comparison["projected_wins"] - comparison["actual_wins"]))
    mae_vegas_actual = np.mean(np.abs(comparison["vegas_implied_wins"] - comparison["actual_wins"]))

    print(f"\nOur model vs. Vegas (agreement): corr = {corr_model_vegas:+.3f}")
    print(f"\nAccuracy vs. REAL 2025 outcomes (the actual test - who's more right, not who agrees more):")
    print(f"  Our model  : corr = {corr_model_actual:+.3f} | MAE = {mae_model_actual:.2f} wins")
    print(f"  Vegas      : corr = {corr_vegas_actual:+.3f} | MAE = {mae_vegas_actual:.2f} wins")
    print(f"  -> {'Vegas is more accurate' if mae_vegas_actual < mae_model_actual else 'Our model is more accurate'} "
          f"this season (expected - Vegas prices in real-time injury/roster news our preseason-only model can't see)")

    comparison["model_vs_vegas_diff"] = comparison["projected_wins"] - comparison["vegas_implied_wins"]
    print("\nBiggest disagreements (our model vs. Vegas, signed - where we diverge most from the market):")
    disagree = comparison.reindex(comparison["model_vs_vegas_diff"].abs().sort_values(ascending=False).index).head(5)
    for _, row in disagree.iterrows():
        direction = "HIGHER" if row["model_vs_vegas_diff"] > 0 else "LOWER"
        print(f"  {row['team']}: model={row['projected_wins']:.1f} wins, Vegas={row['vegas_implied_wins']:.1f} wins "
              f"({direction} by {abs(row['model_vs_vegas_diff']):.1f}) | actual={row['actual_wins']:.0f} wins")

    print(f"\nCalibration:")
    print(f"  Our model mean={comparison['projected_wins'].mean():.2f} std={comparison['projected_wins'].std():.2f}")
    print(f"  Vegas      mean={comparison['vegas_implied_wins'].mean():.2f} std={comparison['vegas_implied_wins'].std():.2f}")
    print(f"  Real 2025  mean={comparison['actual_wins'].mean():.2f} std={comparison['actual_wins'].std():.2f}")
    print(f"  (Task 4.1 already found our projections are ~3.2x too compressed vs. real outcomes - "
          f"checking here whether Vegas is compressed too, or just us)")

    out_path = os.path.join(os.path.join(PROJECT_ROOT, "data", "diagnostic"), "model_vs_vegas_2025.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    print("=" * 70)

    return comparison, {"corr_model_vegas": corr_model_vegas, "corr_model_actual": corr_model_actual,
                         "corr_vegas_actual": corr_vegas_actual, "mae_model_actual": mae_model_actual,
                         "mae_vegas_actual": mae_vegas_actual}


def vegas_comparison_framework():
    our_predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions_2026_preseason.csv"))
    schedule_2026 = pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    schedule_2026 = schedule_2026[schedule_2026["game_type"] == "REG"]

    vegas_lines = schedule_2026[schedule_2026["spread_line"].notna()][
        ["week", "home_team", "away_team", "spread_line", "total_line", "home_moneyline", "away_moneyline"]]
    print(f"Real Vegas lines currently posted: {len(vegas_lines)}/{len(schedule_2026)} games "
          f"(weeks {int(vegas_lines['week'].min())}-{int(vegas_lines['week'].max())} - "
          f"books haven't published the rest yet, expected this far before the season)")

    comparison = our_predictions.merge(vegas_lines, on=["week", "home_team", "away_team"], how="inner")

    home_raw = moneyline_to_implied_prob(comparison["home_moneyline"])
    away_raw = moneyline_to_implied_prob(comparison["away_moneyline"])
    vig_total = home_raw + away_raw
    comparison["vegas_home_win_prob"] = home_raw / vig_total

    comparison["spread_disagreement"] = comparison["expected_spread"] - comparison["spread_line"]
    comparison["total_disagreement"] = comparison["expected_total"] - comparison["total_line"]
    comparison["win_prob_disagreement"] = comparison["home_win_probability"] - comparison["vegas_home_win_prob"]
    comparison["interpretation"] = np.select(
        [comparison["spread_disagreement"] > 0.5, comparison["spread_disagreement"] < -0.5],
        ["We favor home more", "We favor away more"], default="Agreement")

    print(f"\n{len(comparison)} games compared")
    print(f"Avg spread disagreement: {comparison['spread_disagreement'].mean():+.2f} pts")
    print(f"Avg total disagreement: {comparison['total_disagreement'].mean():+.2f} pts")
    print(f"corr(our spread, Vegas spread): {comparison['expected_spread'].corr(comparison['spread_line']):+.3f}")
    print(f"corr(our win prob, Vegas win prob): {comparison['home_win_probability'].corr(comparison['vegas_home_win_prob']):+.3f}")

    print(f"\nBiggest disagreements (spread, absolute):")
    biggest = comparison.reindex(comparison["spread_disagreement"].abs().sort_values(ascending=False).index).head(5)
    for _, g in biggest.iterrows():
        print(f"  Week {g['week']}: {g['home_team']} vs {g['away_team']} - "
              f"we say {g['expected_spread']:+.1f}, Vegas says {g['spread_line']:+.1f} "
              f"(delta {g['spread_disagreement']:+.1f})")

    out_path = os.path.join(os.path.join(PROJECT_ROOT, "data", "diagnostic"), "vegas_comparison_2026_preseason.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    return comparison


def identify_edges(comparison=None):
    if comparison is None:
        comparison = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "diagnostic", "vegas_comparison_2026_preseason.csv"))

    comparison["confidence"] = (comparison["home_win_probability"] - 0.5).abs()
    comparison["disagreement_size"] = comparison["spread_disagreement"].abs()
    comparison["potential_edge"] = (comparison["confidence"] > 0.15) & (comparison["disagreement_size"] > 1.5)

    edges = comparison[comparison["potential_edge"]].sort_values("disagreement_size", ascending=False)
    print(f"\n{len(edges)} potential edges (confidence>0.15, spread disagreement>1.5 pts) out of {len(comparison)} games:")
    for _, e in edges.head(10).iterrows():
        direction = "we favor home" if e["spread_disagreement"] > 0 else "we favor away"
        print(f"  Week {e['week']}: {e['home_team']} vs {e['away_team']} - {direction}, "
              f"disagreement {e['disagreement_size']:.1f} pts, our win prob {e['home_win_probability']:.1%}")
    return edges
