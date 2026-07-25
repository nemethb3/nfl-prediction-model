"""Coach quality (Phase 2 Task 5.4).

Corrects two problems in the original task spec (see the Task 5.4.1
completion report for the full reasoning):

1. The spec assumed offensive/defensive coordinator identities are available
   via nfl_data_py (`import_coaches`, which doesn't exist) or a manually
   entered external file. Neither is real: OC/DC assignments aren't part of
   nflverse's public data. This module uses HEAD COACH instead - a real,
   already-available signal (schedules_2015_2025.csv's home_coach/
   away_coach, already used elsewhere in this project for the
   coaching_change feature) - as the closest legitimate substitute.

2. The spec's "residual regression" wasn't actually a regression - it
   computed a coach's team-season EPA minus that SAME team's PRIOR-season
   EPA, then called the difference the "coach effect." This project's own
   prior research already shows that comparison is dominated by noise:
   team_strength.py's honest backtest found that naive "assume same as last
   year" defense-EPA prediction scores an average R2 of -0.77 across 6
   holdout years - most of a team's year-over-year EPA swing is mean-
   reversion, not anything persistent. Attributing that swing to "the coach"
   would mostly be attributing noise to a person. This module instead
   builds a real, honestly-backtested predictive model (extending
   team_strength.py's validated Ridge pipeline to a parallel OFFENSE-side
   target) and defines coach effect as actual - MODEL-PREDICTED, which
   already accounts for real mean-reversion.

It also adds a check the original spec didn't include at all: a split-half
reliability check on the resulting coach-effect estimates. If a coach's
residual in their earlier seasons doesn't correlate with their residual in
their later seasons, that's real evidence the "coach effect" signal here is
mostly noise, not a persistent, usable quantity - and this task will say so
honestly rather than assume a real effect exists just because a number can
be computed.
"""

import os

import numpy as np
import pandas as pd

from team_strength import (
    HOLDOUT_SEASON, TRAIN_START, compute_team_defense_epa, train_defense_component_model,
)
from utilities import build_coach_crosswalk, compute_history_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

OFF_PBP_COLS = ["posteam", "season", "week", "play_type", "epa", "season_type"]
TEAM_OFFENSE_FEATURES = [
    "off_epa_last_year", "pass_epa_off_last_year", "rush_epa_off_last_year",
    "qb_epa_last_year", "sack_rate_allowed_last_year",
]
MIN_SEASONS_FOR_COACH_ESTIMATE = 2
MIN_SEASONS_FOR_RELIABILITY_CHECK = 4


def compute_team_offense_epa(pbp_path=None):
    """Team offense EPA per season (combined + pass/rush split), from the
    offense's own plays (posteam) - the offensive mirror of
    team_strength.compute_team_defense_epa (defteam)."""
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    keep_chunks = []
    for chunk in pd.read_csv(path, usecols=OFF_PBP_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[chunk["play_type"].isin(["pass", "run"]) & (chunk["season_type"] == "REG")]
        if len(sub):
            keep_chunks.append(sub)
    reg = pd.concat(keep_chunks, ignore_index=True).dropna(subset=["epa", "posteam"])

    team_off = reg.groupby(["posteam", "season"])["epa"].mean().reset_index(name="off_epa")
    team_pass = reg[reg["play_type"] == "pass"].groupby(["posteam", "season"])["epa"].mean().reset_index(name="pass_epa_off")
    team_rush = reg[reg["play_type"] == "run"].groupby(["posteam", "season"])["epa"].mean().reset_index(name="rush_epa_off")
    team = team_off.merge(team_pass, on=["posteam", "season"]).merge(team_rush, on=["posteam", "season"])
    team = team.rename(columns={"posteam": "team"})
    print(f"[team_offense_epa] {team.shape[0]:,} team-seasons, {int(team['season'].min())}-{int(team['season'].max())}")
    return team


def build_team_offense_features(team_off_epa, team_ol_metrics, qb_by_team_season):
    """Adds lagged offense EPA splits, plus two real supporting leading
    indicators already built in this project: sack_rate_allowed_last_year
    (Task 4.1's OL-quality proxy) and qb_epa_last_year (from
    player_models.team_qb_of_record)."""
    df = team_off_epa.copy()
    for col in ["off_epa", "pass_epa_off", "rush_epa_off"]:
        feats = compute_history_features(df[["team", "season", col]], col, id_col="team")
        df = df.join(feats[[f"{col}_last_year"]])

    df = df.merge(team_ol_metrics[["team", "season", "sack_rate_allowed_last_year"]], on=["team", "season"], how="left")
    df = df.merge(qb_by_team_season[["team", "season", "qb_epa_last_year"]], on=["team", "season"], how="left")
    return df


def build_coach_residuals(off_model, off_df, def_model, def_df, schedules):
    """Per team-season: actual - model-predicted, for both offense and
    defense, joined to the head coach of record for that team-season
    (utilities.build_coach_crosswalk - already-validated real data)."""
    off = off_df.dropna(subset=TEAM_OFFENSE_FEATURES + ["off_epa"]).copy()
    off["predicted_off_epa"] = off_model.predict(off[TEAM_OFFENSE_FEATURES])
    off["off_residual"] = off["off_epa"] - off["predicted_off_epa"]

    from team_strength import TEAM_DEFENSE_FEATURES
    defn = def_df.dropna(subset=TEAM_DEFENSE_FEATURES + ["def_epa_allowed"]).copy()
    defn["predicted_def_epa"] = def_model.predict(defn[TEAM_DEFENSE_FEATURES])
    # Sign flip: for defense, LOWER def_epa_allowed is better, so a good
    # coach's residual (actual - predicted) is NEGATIVE. Flip here so
    # "positive residual = coach helped" holds for both sides of the ball,
    # matching off_residual's convention (positive = better than expected).
    defn["def_residual"] = -(defn["def_epa_allowed"] - defn["predicted_def_epa"])

    coach_crosswalk = build_coach_crosswalk(schedules)
    off_coach = off.merge(coach_crosswalk, on=["team", "season"], how="inner")
    def_coach = defn.merge(coach_crosswalk, on=["team", "season"], how="inner")
    return off_coach, def_coach


def aggregate_coach_impact(off_coach, def_coach, min_seasons=MIN_SEASONS_FOR_COACH_ESTIMATE):
    off_agg = off_coach.groupby("coach").agg(
        off_seasons=("season", "count"), off_residual_avg=("off_residual", "mean")
    ).reset_index()
    def_agg = def_coach.groupby("coach").agg(
        def_seasons=("season", "count"), def_residual_avg=("def_residual", "mean")
    ).reset_index()

    impact = off_agg.merge(def_agg, on="coach", how="outer")
    impact = impact[(impact["off_seasons"].fillna(0) >= min_seasons) | (impact["def_seasons"].fillna(0) >= min_seasons)]
    return impact.sort_values("off_residual_avg", ascending=False).reset_index(drop=True)


def split_half_reliability(coach_df, residual_col, min_seasons=MIN_SEASONS_FOR_RELIABILITY_CHECK):
    """Test-retest style reliability check: for coaches with enough seasons,
    split their team-seasons chronologically into first/second half and
    correlate the average residual across the two halves. A real, persistent
    coach effect should show up as a positive correlation here; near-zero
    means the season-level residual is dominated by noise, not a stable
    coach quality signal."""
    rows = []
    for coach, g in coach_df.groupby("coach"):
        g = g.sort_values("season")
        if len(g) < min_seasons:
            continue
        mid = len(g) // 2
        first_half = g.iloc[:mid][residual_col].mean()
        second_half = g.iloc[mid:][residual_col].mean()
        rows.append({"coach": coach, "n_seasons": len(g), "first_half": first_half, "second_half": second_half})

    if len(rows) < 5:
        print(f"  Only {len(rows)} coaches with {min_seasons}+ seasons - too few for a meaningful reliability check")
        return None

    reliability_df = pd.DataFrame(rows)
    corr = reliability_df["first_half"].corr(reliability_df["second_half"])
    print(f"  Split-half reliability ({residual_col}, n={len(reliability_df)} coaches with {min_seasons}+ seasons): "
          f"corr(first-half avg, second-half avg) = {corr:+.3f}")
    return corr


def validate_coach_impact(impact, off_coach, def_coach):
    print("\n===== COACH IMPACT VALIDATION =====")
    print(f"Coaches with >={MIN_SEASONS_FOR_COACH_ESTIMATE}+ seasons on either side of the ball: {len(impact)}")

    print("\nTop 5 offense (highest residual - team outperformed the model's roster-based expectation):")
    print(impact.dropna(subset=["off_residual_avg"]).nlargest(5, "off_residual_avg")[
        ["coach", "off_seasons", "off_residual_avg"]].to_string(index=False))
    print("\nBottom 5 offense:")
    print(impact.dropna(subset=["off_residual_avg"]).nsmallest(5, "off_residual_avg")[
        ["coach", "off_seasons", "off_residual_avg"]].to_string(index=False))

    print("\nTop 5 defense (residual already sign-flipped so positive = better):")
    print(impact.dropna(subset=["def_residual_avg"]).nlargest(5, "def_residual_avg")[
        ["coach", "def_seasons", "def_residual_avg"]].to_string(index=False))
    print("\nBottom 5 defense:")
    print(impact.dropna(subset=["def_residual_avg"]).nsmallest(5, "def_residual_avg")[
        ["coach", "def_seasons", "def_residual_avg"]].to_string(index=False))

    print("\nThe real question: is this a persistent signal or noise?")
    off_corr = split_half_reliability(off_coach, "off_residual")
    def_corr = split_half_reliability(def_coach, "def_residual")

    print(f"\nMean off_residual_avg: {impact['off_residual_avg'].mean():.4f} | "
          f"Mean def_residual_avg: {impact['def_residual_avg'].mean():.4f} (both should be ~0 - no systematic bias)")
    print("===== END VALIDATION =====\n")
    return {"off_reliability_corr": off_corr, "def_reliability_corr": def_corr}


def run_coach_quality_analysis():
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    team_ol_metrics = pd.read_csv(os.path.join(PROCESSED_DIR, "team_ol_metrics_2015_2025.csv"))

    from player_models import team_qb_of_record
    qb_by_team_season = team_qb_of_record(features_df, season_stats_df)

    print("=" * 60 + "\nBuilding team OFFENSE EPA model (parallel to the existing defense model)\n" + "=" * 60)
    team_off_epa = compute_team_offense_epa()
    off_df = build_team_offense_features(team_off_epa, team_ol_metrics, qb_by_team_season)
    off_model, off_backtest = train_defense_component_model(
        off_df, target_col="off_epa", feature_cols=TEAM_OFFENSE_FEATURES)

    print("\n" + "=" * 60 + "\nBuilding team DEFENSE EPA model (reused rigor, retrained fresh here)\n" + "=" * 60)
    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    from team_strength import build_team_defense_features
    team_def_epa = compute_team_defense_epa()
    def_df = build_team_defense_features(team_def_epa, pass_rush_war_df)
    def_model, def_backtest = train_defense_component_model(def_df, target_col="def_epa_allowed")

    print("\n" + "=" * 60 + "\nComputing coach-level residuals\n" + "=" * 60)
    off_coach, def_coach = build_coach_residuals(off_model, off_df, def_model, def_df, schedules)
    impact = aggregate_coach_impact(off_coach, def_coach)
    reliability = validate_coach_impact(impact, off_coach, def_coach)

    out_path = os.path.join(PROCESSED_DIR, "coach_epa_impact_2015_2024.csv")
    impact.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({len(impact)} coaches)")

    return impact, reliability, off_model, def_model, off_df, def_df


if __name__ == "__main__":
    run_coach_quality_analysis()
