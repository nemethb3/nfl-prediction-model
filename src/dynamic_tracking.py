"""Dynamic Season-Win Tracking: actual wins-so-far + predicted wins for the
remaining schedule, updated at any checkpoint week.

Corrects 3 things in the original spec before building (see PROGRESS.md /
the corrected-plan message this session for the full rationale):

1. Step 1 ("build weekly_update_pipeline() + game engine wiring") was mostly
   already done - run_weekly_game_prediction_update() (weekly_predictions.py)
   already chains weekly_update_pipeline() -> build_game_prediction_engine()
   restricted to the remaining schedule. This module only adds what was
   actually missing: actual per-team W-L through week N, and combining
   actual + predicted into one total with a CI.

2. The spec's proposed CI ("1.645 * sqrt(sum_of_residual_variances_per_game)"
   using epa_to_wins.py's resid_std_wins ~= 2.0) is a single fixed residual
   from a FULL 17-game season regression - using it flat regardless of how
   many games remain would understate uncertainty early in the season and
   badly overstate it late. Used instead: infer_season_wins_from_game_
   predictions()'s existing per-remaining-game Bernoulli variance
   (p*(1-p) summed over just the games left), which already has the right
   shape (wide early, narrow late) with no rescaling assumption needed.

3. "Compare to Vegas trajectory" isn't buildable - vegas_with_results_2015_
   2025.csv has closing moneylines only, no odds-movement history, so there
   is no real week-by-week Vegas correlation to compute (the spec's own
   table already marks those values "(est)", i.e. invented). Substituted: a
   single fixed Vegas-implied-wins benchmark (same method as ensemble.py's
   Part 1 candidate) computed once, compared against our trajectory.

Also fixes a real performance issue a naive implementation would have:
weekly_update_pipeline() calls estimate_prior_weeks() (an 8-candidate x
3-checkpoint search, each doing a chunked read of the 1.3GB PBP file) unless
prior_weeks is passed explicitly. validate_dynamic_tracking() computes it
ONCE and passes it through every checkpoint call instead of recomputing it
per checkpoint.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")


def _actual_wins_through_week(season, through_week_n):
    """Real per-team W-L record using only games with week <= through_week_n."""
    game_results = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    played = game_results[(game_results["season"] == season) & (game_results["game_type"] == "REG")
                           & (game_results["week"] <= through_week_n)].copy()
    played["home_win"] = (played["home_score"] > played["away_score"]).astype(int)
    played["away_win"] = (played["away_score"] > played["home_score"]).astype(int)

    home = played[["home_team", "home_win"]].rename(columns={"home_team": "team", "home_win": "win"})
    away = played[["away_team", "away_win"]].rename(columns={"away_team": "team", "away_win": "win"})
    long = pd.concat([home, away], ignore_index=True)

    return long.groupby("team")["win"].agg(actual_wins="sum", games_played="count").reset_index()


def run_dynamic_season_win_tracking(season, through_week_n, prior_weeks=None):
    """actual_wins (real, through_week_n) + predicted_wins (remaining schedule,
    via the weekly-updated team strength) = total_projection, with a 90% CI
    built from the predicted portion's real per-game Bernoulli variance."""
    from weekly_predictions import run_weekly_game_prediction_update, estimate_prior_weeks
    from game_predictions import infer_season_wins_from_game_predictions

    if prior_weeks is None:
        prior_weeks = estimate_prior_weeks(season)

    actual = _actual_wins_through_week(season, through_week_n)

    remaining_predictions, _ = run_weekly_game_prediction_update(season, through_week_n, prior_weeks=prior_weeks)
    if len(remaining_predictions):
        predicted = infer_season_wins_from_game_predictions(remaining_predictions)
        predicted = predicted.rename(columns={"expected_wins": "predicted_wins", "num_games": "predicted_games_remaining"})
        predicted = predicted[["team", "predicted_wins", "wins_low_90", "wins_high_90", "predicted_games_remaining"]]
    else:
        predicted = pd.DataFrame(columns=["team", "predicted_wins", "wins_low_90", "wins_high_90", "predicted_games_remaining"])

    out = actual.merge(predicted, on="team", how="left")
    fill_cols = ["predicted_wins", "wins_low_90", "wins_high_90", "predicted_games_remaining"]
    out[fill_cols] = out[fill_cols].fillna(0.0)

    out["total_projection"] = out["actual_wins"] + out["predicted_wins"]
    out["ci_low_90"] = out["actual_wins"] + out["wins_low_90"]
    out["ci_high_90"] = out["actual_wins"] + out["wins_high_90"]
    out["through_week"] = through_week_n
    return out.sort_values("total_projection", ascending=False).reset_index(drop=True)


def validate_dynamic_tracking(season=2025, checkpoint_weeks=(1, 4, 8, 12, 16)):
    """Real validation against the completed 2025 season: runs the tracker
    at each checkpoint, checks MAE/corr/CI-coverage vs. real final wins, and
    compares the trajectory to a single fixed Vegas-implied-wins benchmark
    (no real per-week Vegas trajectory exists in the available data)."""
    from weekly_predictions import estimate_prior_weeks
    from vegas_comparison import compute_vegas_implied_wins

    prior_weeks = estimate_prior_weeks(season)  # computed once, reused for every checkpoint

    actual_final = pd.read_csv(os.path.join(BACKTEST_DIR, "actual_wins_2025.csv")) if season == 2025 else None
    if actual_final is None:
        raise ValueError(f"No real final-outcome ground truth available for season {season}")

    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas_wins = compute_vegas_implied_wins(vegas, season=season)
    vegas_merged = vegas_wins.merge(actual_final[["team", "actual_wins"]], on="team", how="inner")
    vegas_corr = vegas_merged["vegas_implied_wins"].corr(vegas_merged["actual_wins"])
    vegas_mae = float(np.mean(np.abs(vegas_merged["vegas_implied_wins"] - vegas_merged["actual_wins"])))

    print(f"\n{'=' * 70}\nDYNAMIC SEASON-WIN TRACKING VALIDATION (real {season}, prior_weeks={prior_weeks})\n{'=' * 70}")
    print(f"Fixed Vegas benchmark (closing lines, whole season - no real weekly trajectory "
          f"exists in this dataset): corr={vegas_corr:+.3f} MAE={vegas_mae:.2f} wins\n")

    results = []
    per_checkpoint = {}
    for week_n in checkpoint_weeks:
        tracking = run_dynamic_season_win_tracking(season, week_n, prior_weeks=prior_weeks)
        merged = tracking.merge(
            actual_final[["team", "actual_wins"]].rename(columns={"actual_wins": "final_actual_wins"}),
            on="team", how="inner")

        mae = float(np.mean(np.abs(merged["total_projection"] - merged["final_actual_wins"])))
        corr = merged["total_projection"].corr(merged["final_actual_wins"])
        coverage = float(((merged["final_actual_wins"] >= merged["ci_low_90"]) &
                           (merged["final_actual_wins"] <= merged["ci_high_90"])).mean())
        ci_width = float((merged["ci_high_90"] - merged["ci_low_90"]).mean())

        print(f"After week {week_n:>2}: MAE={mae:.2f} corr={corr:+.3f} | "
              f"CI coverage={coverage:.1%} (target 90%, avg width {ci_width:.2f} wins) | "
              f"{'BEATS Vegas' if mae < vegas_mae else 'below Vegas'}")
        results.append({"week": week_n, "mae": mae, "corr": corr, "ci_coverage": coverage, "ci_avg_width": ci_width})
        per_checkpoint[week_n] = merged

    results_df = pd.DataFrame(results)
    print(f"\nFixed Vegas benchmark for reference: MAE={vegas_mae:.2f} corr={vegas_corr:+.3f}")
    print(f"MAE improving monotonically week-over-week: "
          f"{'YES' if results_df['mae'].is_monotonic_decreasing else 'NO'}")

    final_week = max(checkpoint_weeks)
    final_merged = per_checkpoint[final_week].copy()
    final_merged["error"] = final_merged["total_projection"] - final_merged["final_actual_wins"]
    print(f"\nBiggest misses at week {final_week} (|error|, sorted):")
    biggest = final_merged.reindex(final_merged["error"].abs().sort_values(ascending=False).index).head(5)
    for _, row in biggest.iterrows():
        print(f"  {row['team']}: actual_through_wk{final_week}={row['actual_wins']:.0f}, "
              f"projected_total={row['total_projection']:.1f}, final_actual={row['final_actual_wins']:.0f} "
              f"(error {row['error']:+.1f})")

    out_path = os.path.join(PROCESSED_DIR, f"dynamic_tracking_validation_{season}.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    print("=" * 70)

    return results_df, per_checkpoint, {"vegas_mae": vegas_mae, "vegas_corr": vegas_corr, "prior_weeks": prior_weeks}


if __name__ == "__main__":
    validate_dynamic_tracking()
