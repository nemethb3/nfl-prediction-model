"""Phase 3 Redesign Subtasks 1 & 4: Game Prediction Engine, Season Win Inference.

Split out of win_projection.py (Master Plan Phase 4 Task 4.2 - see
AUDIT_2026-07-25.md Technical Debt #5). No behavior change, reorganization only.

Corrects the spec's four asserted constants (EPA_TO_POINTS=3.5,
HOME_FIELD_ADVANTAGE=2.5, league_avg_points=21.0, std_dev=5.0 for win
probability) - it only checked ONE of them (EPA_TO_POINTS) against real
data, and only as an afterthought that was never fed back into the actual
predictions. Instead, one real regression on real historical per-game
results (point_diff ~ epa_diff, pooled across many REG seasons) gives all
four at once: the fitted slope IS the real EPA->points conversion, the
intercept IS the real home-field advantage in points (home teams' real
average edge, not asserted), the residual std IS the real spread for win
probability, and league-average total points is measured directly from
real games instead of assumed. Holdout-validated the same way every other
model in this project is (train on 2016-2023, honest check on 2024).
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")


def fit_game_level_epa_to_points(team_off, team_def, game_results,
                                  train_seasons=range(2016, 2024), holdout_season=2024):
    reg = game_results[game_results["game_type"] == "REG"].copy()

    off = team_off[["team", "season", "off_epa"]]
    defn = team_def[["team", "season", "def_epa_allowed"]]

    reg = reg.merge(off.rename(columns={"team": "home_team", "off_epa": "home_off_epa"}), on=["home_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "home_team", "def_epa_allowed": "home_def_epa"}), on=["home_team", "season"])
    reg = reg.merge(off.rename(columns={"team": "away_team", "off_epa": "away_off_epa"}), on=["away_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "away_team", "def_epa_allowed": "away_def_epa"}), on=["away_team", "season"])

    reg["epa_diff"] = (reg["home_off_epa"] - reg["away_def_epa"]) - (reg["away_off_epa"] - reg["home_def_epa"])
    reg["point_diff"] = reg["home_score"] - reg["away_score"]

    train = reg[reg["season"].isin(train_seasons)]
    slope, intercept = np.polyfit(train["epa_diff"], train["point_diff"], 1)
    train_pred = slope * train["epa_diff"] + intercept
    resid_std = float((train["point_diff"] - train_pred).std())
    train_corr = np.corrcoef(train["epa_diff"], train["point_diff"])[0, 1]

    hold = reg[reg["season"] == holdout_season]
    hold_pred = slope * hold["epa_diff"] + intercept
    hold_mae = float((hold["point_diff"] - hold_pred).abs().mean())
    hold_corr = np.corrcoef(hold["epa_diff"], hold["point_diff"])[0, 1]

    league_avg_total_points = float(train["total"].mean())

    print(f"[game-level EPA->points] train (n={len(train)}, seasons {min(train_seasons)}-{max(train_seasons)}): "
          f"point_diff = {slope:.3f} * epa_diff + {intercept:+.3f} (home-field, points) | "
          f"corr={train_corr:+.3f} | residual std={resid_std:.2f} pts")
    print(f"[game-level EPA->points] {holdout_season} holdout (n={len(hold)}): MAE={hold_mae:.2f} pts | corr={hold_corr:+.3f}")
    print(f"[game-level EPA->points] real league-average total points/game (train seasons): {league_avg_total_points:.2f}")

    return slope, intercept, resid_std, league_avg_total_points


def _load_schedule_for_season(season):
    """schedules_2026.csv is already single-season; every earlier season
    lives in the multi-season schedules_2015_2025.csv and needs filtering."""
    if season == 2026:
        return pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    return schedules[schedules["season"] == season].copy()


def build_game_prediction_engine(season=2026, team_strength=None, schedule=None, fit_params=None,
                                  weeks_after=None, save=True):
    """Generalized (Master Plan Phase 1 Task 1.2) to accept an explicit
    team_strength/schedule instead of always reading team_strength_2026.csv -
    this is what lets the weekly update pipeline chain into this function
    for any season/week rather than only ever producing the 2026 preseason
    file. Defaults reproduce the original Subtask 1 behavior exactly for
    existing callers (season=2026, team_strength_2026.csv on disk).

    weeks_after: if set, restricts to schedule weeks > weeks_after (the
    "remaining games" framing the weekly update pipeline needs) and the
    output filename becomes game_predictions_{season}_after_week{N}.csv
    instead of the preseason name.
    """
    from scipy.stats import norm
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    if team_strength is None:
        team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv"))
    if "offensive_strength_updated" in team_strength.columns:
        # weekly_update_team_strength() keeps the ORIGINAL offensive_strength
        # column alongside the new _updated one - "offensive_strength" is
        # therefore always already present, so a not-in-columns check here
        # never fires and silently uses the stale pre-update value. Drop the
        # stale columns first, unconditionally prefer _updated when present.
        team_strength = team_strength.drop(
            columns=["offensive_strength", "defensive_strength_allowed"], errors="ignore"
        ).rename(columns={
            "offensive_strength_updated": "offensive_strength",
            "defensive_strength_allowed_updated": "defensive_strength_allowed"})

    if schedule is None:
        schedule = _load_schedule_for_season(season)
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    if weeks_after is not None:
        schedule = schedule[schedule["week"] > weeks_after]
    print(f"Loaded {len(team_strength)} teams, {len(schedule)} {season} REG games"
          + (f" (weeks after {weeks_after})" if weeks_after is not None else ""))

    if fit_params is None:
        team_off = compute_team_offense_epa()
        team_def = compute_team_defense_epa()
        game_results = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
        slope, intercept, resid_std, league_avg_total = fit_game_level_epa_to_points(team_off, team_def, game_results)
    else:
        slope, intercept, resid_std, league_avg_total = (
            fit_params["slope"], fit_params["intercept"], fit_params["resid_std"], fit_params["league_avg_total"])

    strength_lookup = team_strength.set_index("team")[["offensive_strength", "defensive_strength_allowed"]]

    rows = []
    missing = set()
    for _, game in schedule.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home not in strength_lookup.index or away not in strength_lookup.index:
            missing.update({home, away} - set(strength_lookup.index))
            continue
        home_off, home_def = strength_lookup.loc[home]
        away_off, away_def = strength_lookup.loc[away]

        epa_diff = (home_off - away_def) - (away_off - home_def)
        point_diff = slope * epa_diff + intercept  # intercept already IS real home-field advantage

        home_pts = league_avg_total / 2 + point_diff / 2
        away_pts = league_avg_total / 2 - point_diff / 2

        z = point_diff / resid_std
        home_win_prob = float(norm.cdf(z))

        rows.append({
            "week": game["week"], "home_team": home, "away_team": away,
            "home_expected_points": round(home_pts, 1), "away_expected_points": round(away_pts, 1),
            "expected_spread": round(point_diff, 2), "expected_total": round(home_pts + away_pts, 1),
            "home_win_probability": round(home_win_prob, 4), "away_win_probability": round(1 - home_win_prob, 4),
        })
    if missing:
        print(f"WARNING: no team_strength row for: {sorted(missing)} - their games skipped")

    predictions_df = pd.DataFrame(rows)
    print(f"\nGenerated predictions for {len(predictions_df)} games")
    if len(predictions_df):
        print(f"Avg spread: {predictions_df['expected_spread'].mean():+.2f} | "
              f"Avg total: {predictions_df['expected_total'].mean():.1f} | "
              f"Spread std: {predictions_df['expected_spread'].std():.2f}")

    if save:
        suffix = "_preseason" if weeks_after is None else f"_after_week{weeks_after}"
        out_path = os.path.join(PROCESSED_DIR, f"game_predictions_{season}{suffix}.csv")
        predictions_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved {out_path}")

    return predictions_df, {"slope": slope, "intercept": intercept, "resid_std": resid_std,
                             "league_avg_total": league_avg_total}


# ---------------------------------------------------------------------------
# Phase 3 Redesign Subtask 4: Season Win Inference from Game Predictions.
#
# Sums win probabilities across a team's real (or predicted) schedule -
# generic over any game_predictions_df with home_team/away_team/
# home_win_probability/away_win_probability columns, so it works for the
# 2026 deliverable and for a real-outcome-validatable season alike (2026
# itself has no played games yet to check against, same limitation as
# every other Phase 3 Redesign subtask - validated on real 2025 instead,
# by generating an equivalent preseason-only 2025 game prediction set via
# the same build_game_prediction_engine() this reuses unchanged).
# ---------------------------------------------------------------------------

def infer_season_wins_from_game_predictions(predictions_df):
    rows = []
    for team in sorted(set(predictions_df["home_team"]) | set(predictions_df["away_team"])):
        home_games = predictions_df[predictions_df["home_team"] == team]["home_win_probability"]
        away_games = predictions_df[predictions_df["away_team"] == team]["away_win_probability"]
        win_probs = pd.concat([home_games, away_games])
        if not len(win_probs):
            continue

        expected_wins = win_probs.sum()
        variance = (win_probs * (1 - win_probs)).sum()  # each game ~ independent Bernoulli
        std_dev = float(np.sqrt(variance))
        wins_low = max(0.0, expected_wins - 1.645 * std_dev)
        wins_high = min(len(win_probs), expected_wins + 1.645 * std_dev)

        rows.append({"team": team, "expected_wins": expected_wins, "std_dev": std_dev,
                     "wins_low_90": wins_low, "wins_high_90": wins_high, "num_games": len(win_probs)})

    return pd.DataFrame(rows).sort_values("expected_wins", ascending=False).reset_index(drop=True)


def run_season_win_inference(season=2026, save=True):
    proj_path = os.path.join(PROCESSED_DIR, f"game_predictions_{season}_preseason.csv")
    predictions_df = pd.read_csv(proj_path)
    season_wins = infer_season_wins_from_game_predictions(predictions_df)

    print(f"\n{'=' * 70}\nSEASON WIN INFERENCE FROM GAME PREDICTIONS ({season})\n{'=' * 70}")
    print(f"Top 10 by expected wins:")
    print(season_wins.head(10)[["team", "expected_wins", "wins_low_90", "wins_high_90", "num_games"]].to_string(index=False))
    print(f"\nBottom 10:")
    print(season_wins.tail(10)[["team", "expected_wins", "wins_low_90", "wins_high_90", "num_games"]].to_string(index=False))
    print(f"\nMean expected wins: {season_wins['expected_wins'].mean():.2f} | std across teams: {season_wins['expected_wins'].std():.2f}")

    if save:
        out_path = os.path.join(PROCESSED_DIR, f"season_wins_from_games_{season}.csv")
        season_wins.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved {out_path}")
    return season_wins


def validate_season_win_inference(season=2025):
    """Real validation (2026 has no outcomes yet): builds an equivalent
    preseason-only game_predictions set for `season` (reusing
    build_game_prediction_engine unchanged), infers season wins from it the
    same way, and compares BOTH to real actual wins AND to the old
    season-level pipeline (win_projections_2025.csv, Task 4.1's direct
    EPA->win_pct regression) - answering "does game-level aggregation beat,
    match, or lose to the season-level approach"."""
    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv"))
    schedule = _load_schedule_for_season(season)
    game_predictions, _ = build_game_prediction_engine(season=season, team_strength=team_strength,
                                                         schedule=schedule, save=False)
    season_wins = infer_season_wins_from_game_predictions(game_predictions)

    actual = pd.read_csv(os.path.join(BACKTEST_DIR, "actual_wins_2025.csv")) if season == 2025 else None
    if actual is None:
        print(f"No real outcomes available for {season} - skipping accuracy check")
        return season_wins, None

    merged = season_wins.merge(actual[["team", "actual_wins"]], on="team", how="inner")
    corr = merged["expected_wins"].corr(merged["actual_wins"])
    mae = np.mean(np.abs(merged["expected_wins"] - merged["actual_wins"]))
    print(f"\n{'=' * 70}\nSEASON WIN INFERENCE VALIDATION (real {season})\n{'=' * 70}")
    print(f"Game-level-aggregated season wins vs. real {season}: corr={corr:+.3f} MAE={mae:.2f} wins")

    old_approach = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{season}.csv"))
    old_merged = old_approach.merge(actual[["team", "actual_wins"]], on="team", how="inner")
    old_corr = old_merged["projected_wins"].corr(old_merged["actual_wins"])
    old_mae = np.mean(np.abs(old_merged["projected_wins"] - old_merged["actual_wins"]))
    print(f"Old season-level pipeline (Task 4.1's direct EPA->win_pct)   : corr={old_corr:+.3f} MAE={old_mae:.2f} wins")
    print(f"{'Game-level aggregation WINS' if mae < old_mae else 'Season-level pipeline WINS'} on MAE")
    print("=" * 70 + "\n")

    return season_wins, {"game_level_corr": corr, "game_level_mae": mae, "old_corr": old_corr, "old_mae": old_mae}
