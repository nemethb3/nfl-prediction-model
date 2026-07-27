"""Component B: Weekly Recalibration Feedback Loop.

Corrects 3 issues found in the spec before building:

1. Function 2 said to call weekly_update_pipeline() to get "updated Elo
   ratings" - that function updates the EPA-based team_strength system
   (Phase 3's shrinkage-blend mechanism), has no connection to elo_model.py,
   and doesn't return anything Elo-shaped. A category error, not a wrong
   file path. The real situation is simpler than the spec assumes: Elo's
   K-factor update rule IS a per-game update rule (that's what an Elo
   rating already does), unlike team_strength/EPA, which needed weekly_
   update_pipeline() specifically because it has no natural sequential
   update of its own. Component A's generate_elo_game_spreads() already
   uses each team's exact real pre-game rating (chained through every prior
   game of the season) for every game - "ratings updated through week N"
   is already sitting in that output; it just needs filtering by week, not
   a new update mechanism. update_elo_with_actual_results() below extracts
   it from the same real chain instead of calling the wrong pipeline.

2. elo_ratings_2025.csv (listed as a data input for "preseason Elo") is, as
   established in Component A, actually Vegas-informed and in-season
   chained - neither purely preseason nor carryover. Not used. Preseason
   ratings come from elo_model.run_multi_season_elo()'s own ratings_at_
   season_start[2025] (real carryover history through 2024, untouched by
   2025).

3. elo_game_predictions_2026.csv (listed as "preseason spreads") is for the
   2026 SEASON, not a preseason baseline for 2025 - not usable here at all.
   Built a genuine 2025 preseason (static, non-updating) baseline directly:
   every 2025 game predicted using each team's single season-starting
   rating, for comparison against the real per-game-updated chain.

One clarification on the spec's own "green flag" list: "90% CIs tighten as
actual data replaces predictions" doesn't apply to per-game spread CIs the
way it applies to season-TOTAL projections (dynamic_tracking.py's win
totals). A single game's outcome uncertainty is a property of that specific
matchup (how close the two teams are), not of how much of the season has
been played - a close game in week 1 and a close game in week 16 have
similar variance. Component A's CI (a flat band from the conversion model's
own historical residual std) correctly reflects this and isn't expected to
shrink over the season; it isn't a bug that it doesn't.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

CHECKPOINT_WEEKS = (1, 4, 8, 12, 16)


def get_actual_results_through_week(season, week_n):
    from elo_game_prediction import _load_game_results
    results = _load_game_results([season])
    return results[results["week"] <= week_n][["game_id", "week", "home_team", "away_team", "point_diff"]].rename(
        columns={"point_diff": "actual_spread"})


def _elo_as_of_week(backtest_df, preseason_ratings, season, week_n):
    """Real per-team Elo as of right after week_n's games - extracted from
    the already-complete, already-causal chain (see module docstring #1)."""
    season_games = backtest_df[backtest_df["season"] == season]
    home_long = season_games[["home_team", "week", "home_elo_after"]].rename(
        columns={"home_team": "team", "home_elo_after": "elo_after"})
    away_long = season_games[["away_team", "week", "away_elo_after"]].rename(
        columns={"away_team": "team", "away_elo_after": "elo_after"})
    long_df = pd.concat([home_long, away_long], ignore_index=True)

    through = long_df[long_df["week"] <= week_n]
    ratings = dict(preseason_ratings)
    if len(through):
        latest = through.sort_values("week").groupby("team").tail(1).set_index("team")["elo_after"]
        ratings.update(latest.to_dict())
    return ratings


def update_elo_with_actual_results(season, week_n, backtest_df=None, preseason_ratings=None):
    """Returns {team: elo_rating} as of right after week_n - real, causal,
    reused from the already-built Elo chain (see module docstring #1), NOT
    weekly_update_pipeline() (wrong system - that's the EPA team_strength
    mechanism)."""
    if backtest_df is None or preseason_ratings is None:
        from elo_model import run_multi_season_elo, TRAIN_SEASONS
        from elo_game_prediction import ELO_K_FACTOR, ELO_HOME_FIELD
        backtest_df, ratings_at_season_start, _ = run_multi_season_elo(
            range(min(TRAIN_SEASONS), season + 1), k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
        preseason_ratings = ratings_at_season_start[season]
    return _elo_as_of_week(backtest_df, preseason_ratings, season, week_n)


def regenerate_spreads_for_remaining_games(season, week_n, updated_elo_ratings, fitted_model):
    from game_predictions import _load_schedule_for_season
    from elo_game_prediction import predict_game_spread_from_elo

    schedule = _load_schedule_for_season(season)
    reg = schedule[(schedule["game_type"] == "REG") & (schedule["week"] > week_n)].copy()
    reg["home_elo"] = reg["home_team"].map(updated_elo_ratings)
    reg["away_elo"] = reg["away_team"].map(updated_elo_ratings)
    reg["predicted_spread"] = predict_game_spread_from_elo(reg["home_elo"], reg["away_elo"], fitted_model)

    band = 1.645 * fitted_model["resid_std"]
    reg["ci_low_90"] = reg["predicted_spread"] - band
    reg["ci_high_90"] = reg["predicted_spread"] + band
    return reg[["game_id", "week", "home_team", "away_team", "home_elo", "away_elo",
                "predicted_spread", "ci_low_90", "ci_high_90"]].reset_index(drop=True)


def compare_preseason_vs_updated_spreads(week_n, preseason_spreads, updated_spreads):
    merged = preseason_spreads[["game_id", "predicted_spread"]].rename(
        columns={"predicted_spread": "preseason_spread"}).merge(
        updated_spreads[["game_id", "predicted_spread"]].rename(columns={"predicted_spread": "updated_spread"}),
        on="game_id", how="inner")
    merged["spread_delta"] = (merged["updated_spread"] - merged["preseason_spread"]).abs()
    merged["week_n"] = week_n
    return merged


def validate_weekly_accuracy(season, week_n, updated_spreads):
    actual = get_actual_results_through_week(season, 18)  # full season, then restrict below
    merged = updated_spreads.merge(actual[["game_id", "actual_spread"]], on="game_id", how="inner")
    if len(merged) == 0:
        return None, None
    corr = merged["predicted_spread"].corr(merged["actual_spread"])
    mae = float(np.mean(np.abs(merged["predicted_spread"] - merged["actual_spread"])))
    return corr, mae


def simulate_full_season_recalibration(season=2025, checkpoint_weeks=CHECKPOINT_WEEKS):
    """One shared Elo chain (see module docstring #1 - Elo's update is
    already causal/sequential, so a single full-season run contains every
    checkpoint; no redundant re-simulation needed) plus one static preseason
    baseline, both computed once and reused across all checkpoints."""
    from elo_model import run_multi_season_elo, TRAIN_SEASONS
    from elo_game_prediction import fit_probability_to_spread_conversion, ELO_K_FACTOR, ELO_HOME_FIELD, \
        predict_game_spread_from_elo
    from game_predictions import _load_schedule_for_season

    fitted_model = fit_probability_to_spread_conversion()
    backtest_df, ratings_at_season_start, _ = run_multi_season_elo(
        range(min(TRAIN_SEASONS), season + 1), k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
    preseason_ratings = ratings_at_season_start[season]

    schedule = _load_schedule_for_season(season)
    reg_all = schedule[schedule["game_type"] == "REG"].copy()
    reg_all["home_elo"] = reg_all["home_team"].map(preseason_ratings)
    reg_all["away_elo"] = reg_all["away_team"].map(preseason_ratings)
    reg_all["predicted_spread"] = predict_game_spread_from_elo(reg_all["home_elo"], reg_all["away_elo"], fitted_model)
    preseason_spreads_full = reg_all[["game_id", "week", "predicted_spread"]].copy()

    print(f"\n{'=' * 70}\nWEEKLY RECALIBRATION SIMULATION (real {season})\n{'=' * 70}")
    rows = []
    saved_updated = {}
    for week_n in checkpoint_weeks:
        updated_ratings = update_elo_with_actual_results(season, week_n, backtest_df, preseason_ratings)
        updated_spreads = regenerate_spreads_for_remaining_games(season, week_n, updated_ratings, fitted_model)
        preseason_remaining = preseason_spreads_full[preseason_spreads_full["week"] > week_n]

        delta_df = compare_preseason_vs_updated_spreads(week_n, preseason_remaining, updated_spreads)
        avg_delta = delta_df["spread_delta"].mean()

        corr, mae = validate_weekly_accuracy(season, week_n, updated_spreads)

        print(f"Week {week_n:>2}: games {week_n + 1}-18 accuracy corr={corr:+.3f} MAE={mae:.2f} pts | "
              f"avg |spread change from preseason|={avg_delta:.2f} pts | n={len(updated_spreads)} games")
        rows.append({"week": week_n, "n_games": len(updated_spreads), "corr": corr, "mae": mae, "avg_spread_delta": avg_delta})
        saved_updated[week_n] = updated_spreads

    trajectory = pd.DataFrame(rows)
    return trajectory, saved_updated, fitted_model


def generate_recalibration_report(season=2025):
    trajectory, saved_updated, fitted_model = simulate_full_season_recalibration(season)

    print(f"\n{'=' * 70}\nWeekly Recalibration Report ({season})\n{'=' * 70}")
    lines = [f"Weekly Recalibration Report ({season})", "=" * 40]
    for _, row in trajectory.iterrows():
        pct_real = row["week"] / 18.0
        lines.append(f"\nWeek {int(row['week'])} ({pct_real:.0%} actual):")
        lines.append(f"  Accuracy (remaining {int(row['n_games'])} games): corr={row['corr']:+.3f}, MAE={row['mae']:.2f}")
        lines.append(f"  Avg |spread change| from preseason: {row['avg_spread_delta']:.2f} pts")

    mae_improving = trajectory["mae"].is_monotonic_decreasing
    corr_improving = trajectory["corr"].is_monotonic_increasing
    delta_growing = trajectory["avg_spread_delta"].is_monotonic_increasing

    lines.append("\nInterpretation:")
    lines.append(f"  - MAE improves monotonically week-by-week: {'YES' if mae_improving else 'NO'}")
    lines.append(f"  - Correlation improves monotonically week-by-week: {'YES' if corr_improving else 'NO'}")
    lines.append(f"  - Spread changes grow monotonically (more real data -> bigger revision): "
                 f"{'YES' if delta_growing else 'NO'}")
    lines.append(f"  - Note: 90% CI width is intentionally FLAT across checkpoints (see module docstring - "
                 f"per-game uncertainty isn't a season-elapsed quantity, unlike season-total CIs)")
    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    trajectory.to_csv(os.path.join(DIAGNOSTIC_DIR, f"weekly_recalibration_trajectory_{season}.csv"),
                       index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "weekly_recalibration_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    for week_n, updated in saved_updated.items():
        updated.to_csv(os.path.join(PROCESSED_DIR, f"weekly_updated_spreads_{season}_week{week_n}.csv"),
                        index=False, encoding="utf-8")
    print(f"\nSaved data/diagnostic/weekly_recalibration_trajectory_{season}.csv, weekly_recalibration_report.txt, "
          f"data/processed/weekly_updated_spreads_{season}_week{{{','.join(str(w) for w in saved_updated)}}}.csv")
    print("=" * 70)

    return trajectory


def prepare_for_live_2026():
    """Documentation for running this live in 2026 - Elo already updates
    naturally as real games are added to game_results_2015_2025.csv (or its
    2026 successor), so live deployment needs no new mechanism: each Monday,
    call update_elo_with_actual_results(2026, current_week) then
    regenerate_spreads_for_remaining_games(...) with that week's real
    completed games included in the underlying game-results data.
    """
    instructions = (
        "LIVE 2026 WEEKLY RECALIBRATION\n"
        "1. After each week's games complete, ensure that week's real "
        "results are appended to the game-results data source Elo reads "
        "from (currently game_results_2015_2025.csv's 2026 successor).\n"
        "2. Call update_elo_with_actual_results(2026, current_week) - this "
        "re-derives every team's real Elo through the games just played; "
        "no separate 'update step' beyond having the real results available.\n"
        "3. Call regenerate_spreads_for_remaining_games(2026, current_week, "
        "updated_ratings, fitted_model) for the upcoming week's games.\n"
        "4. fitted_model (data/processed/elo_probability_to_spread_model.pkl) "
        "does not need refitting weekly - it was fit once on 2015-2023 and "
        "validated on 2024/2025; only the Elo ratings themselves update.\n"
        "Edge cases: bye weeks (a team with no game in a given week keeps "
        "its existing rating - already handled, since _elo_as_of_week only "
        "updates teams that appear in that week's games); a completely new "
        "franchise/relocation would have no real prior-season history and "
        "would need a manual seed value (not currently handled - none exist "
        "in 2015-2026 data, so untested)."
    )
    print(instructions)
    return instructions


if __name__ == "__main__":
    generate_recalibration_report()
    prepare_for_live_2026()
