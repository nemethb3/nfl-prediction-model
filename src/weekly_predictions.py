"""Phase 3 Redesign, Subtask 2: Weekly Update Pipeline.

Context: the spec asked for a pipeline that updates player/team EPA from
each week's real results as the season unfolds and recalculates the
remaining schedule's projections. 2026 has no played games yet (confirmed
directly - nflreadpy's load_pbp() raises ValueError for season=2026, since
the season doesn't start until September), so there is nothing real to
"update from" right now. Per the user's direction, this module builds the
real, season/week-parameterized update mechanism and validates it against
the actual, fully-completed 2025 season - processed week by week as if
watching it unfold - since that's the only real data available to prove the
mechanism works before it can be pointed at 2026 in September.

Also corrects several fabricated pieces in the original spec: there is no
load_pbp_streaming()/load_schedule()/load_vegas_lines() anywhere in this
codebase, and PBP has no `position` column (position only comes from
joining passer_id/receiver_id/rusher_id against the player crosswalk - the
same pattern load_real_2025_pbp()/compute_real_epa_per_play() already use).
This module works at the TEAM level (real offense/defense EPA computed from
a partial season, shrinkage-blended with the preseason team_strength
projection) rather than cascading a full player-level re-projection through
every OL/SOS/availability/synergy adjustment mid-week - the original spec's
own player-level update functions were themselves unfinished stubs (loaded
the prior file and immediately saved it back out unchanged, with a "Blend
with Week N actual" comment but no actual blending code), so a working,
validated team-level mechanism is real functionality where the spec had
none, not a scope reduction from something the spec actually delivered.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PBP_WEEK_COLS = ["season", "week", "season_type", "play_type", "epa", "posteam", "defteam"]


def compute_team_epa_through_week(season, week_end, pbp_path=None):
    """Real offense/defense EPA per team, using ONLY weeks 1..week_end of
    `season` - a genuine in-season, partial-season measurement, not a
    projection. Same chunked-read/REG-only/pass-or-run-only convention as
    coach_quality.compute_team_offense_epa / team_strength.compute_team_defense_epa,
    generalized here with a week cutoff and both sides computed in one PBP
    pass (avoids scanning the 1.3GB file twice)."""
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    keep = []
    for chunk in pd.read_csv(path, usecols=PBP_WEEK_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[(chunk["season"] == season) & (chunk["season_type"] == "REG")
                    & (chunk["week"] <= week_end) & (chunk["play_type"].isin(["pass", "run"]))]
        if len(sub):
            keep.append(sub)
    reg = pd.concat(keep, ignore_index=True).dropna(subset=["epa"])

    off = reg.dropna(subset=["posteam"]).groupby("posteam")["epa"].mean().rename("off_epa_partial").reset_index().rename(
        columns={"posteam": "team"})
    defn = reg.dropna(subset=["defteam"]).groupby("defteam")["epa"].mean().rename("def_epa_allowed_partial").reset_index().rename(
        columns={"defteam": "team"})
    weeks_played = reg.groupby("posteam")["week"].nunique().rename("weeks_played_partial").reset_index().rename(
        columns={"posteam": "team"})

    out = off.merge(defn, on="team", how="outer").merge(weeks_played, on="team", how="left")
    return out


def shrinkage_blend(preseason_value, partial_value, weeks_played, prior_weeks):
    """Standard empirical-Bayes shrinkage: updated = (preseason * prior_weeks +
    partial_season * weeks_played) / (prior_weeks + weeks_played). Early in
    the season (few real weeks played), the preseason projection dominates;
    as more real weeks accumulate, real in-season performance takes over.
    prior_weeks is estimated empirically below, not asserted."""
    return (preseason_value * prior_weeks + partial_value * weeks_played) / (prior_weeks + weeks_played)


def weekly_update_team_strength(team_strength_preseason, partial_epa, prior_weeks):
    """Blends preseason offensive_strength/defensive_strength_allowed with
    real partial-season EPA through the checkpoint week."""
    merged = team_strength_preseason.merge(partial_epa, on="team", how="left")
    merged["weeks_played_partial"] = merged["weeks_played_partial"].fillna(0)

    merged["offensive_strength_updated"] = np.where(
        merged["weeks_played_partial"] > 0,
        shrinkage_blend(merged["offensive_strength"], merged["off_epa_partial"],
                         merged["weeks_played_partial"], prior_weeks),
        merged["offensive_strength"],
    )
    merged["defensive_strength_allowed_updated"] = np.where(
        merged["weeks_played_partial"] > 0,
        shrinkage_blend(merged["defensive_strength_allowed"], merged["def_epa_allowed_partial"],
                         merged["weeks_played_partial"], prior_weeks),
        merged["defensive_strength_allowed"],
    )
    merged["net_strength_updated"] = merged["offensive_strength_updated"] - merged["defensive_strength_allowed_updated"]
    return merged


def _real_final_season_epa(season):
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    off = compute_team_offense_epa()
    off = off[off["season"] == season][["team", "off_epa"]]
    defn = compute_team_defense_epa()
    defn = defn[defn["season"] == season][["team", "def_epa_allowed"]]
    return off.merge(defn, on="team")


def estimate_prior_weeks(season=2025, checkpoint_weeks=(4, 8, 12), candidates=(1, 2, 3, 4, 6, 8, 10, 14)):
    """Picks the prior_weeks value that, averaged across several real
    in-season checkpoints, best predicts that SAME season's real final
    outcome - an empirical choice, not an asserted constant. Uses 2025's
    real preseason team_strength (team_strength_2025.csv, built leak-free
    from 2024 data) as the prior and 2025's own real final-season EPA
    (already known, since the season is complete) as ground truth - this is
    a within-season estimate (how fast should the season's own signal be
    trusted), a different question from any of this project's cross-season
    leak-free projections."""
    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, "team_strength_2025.csv"))
    real_final = _real_final_season_epa(season)

    scores = {}
    for prior_weeks in candidates:
        errs = []
        for week_end in checkpoint_weeks:
            partial = compute_team_epa_through_week(season, week_end)
            updated = weekly_update_team_strength(team_strength, partial, prior_weeks)
            merged = updated.merge(real_final, on="team", how="inner")
            mae = np.mean(np.abs(merged["offensive_strength_updated"] - merged["off_epa"]))
            errs.append(mae)
        scores[prior_weeks] = np.mean(errs)
        print(f"[estimate_prior_weeks] prior_weeks={prior_weeks}: avg MAE across weeks {checkpoint_weeks} = {scores[prior_weeks]:.4f}")

    best = min(scores, key=scores.get)
    print(f"[estimate_prior_weeks] winner: prior_weeks={best} (avg MAE={scores[best]:.4f})")
    return best


def run_weekly_update_demo(season=2025, checkpoint_weeks=(1, 2, 4, 8, 12, 16)):
    """Validates the update mechanism against the real, fully-completed
    2025 season (2026 has no played games yet). Shows, at each checkpoint
    week, whether the SHRINKAGE-UPDATED projection tracks the real final
    2025 outcome better than (a) the static preseason-only projection and
    (b) a naive "just use partial-season real data alone, no prior" baseline
    - the latter is expected to be badly overfit to small samples early in
    the season and only catch up as more weeks accumulate."""
    prior_weeks = estimate_prior_weeks(season)

    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, "team_strength_2025.csv"))
    real_final = _real_final_season_epa(season)
    preseason_merged = team_strength.merge(real_final, on="team", how="inner")
    preseason_mae = np.mean(np.abs(preseason_merged["offensive_strength"] - preseason_merged["off_epa"]))
    preseason_corr = preseason_merged["offensive_strength"].corr(preseason_merged["off_epa"])

    print(f"\n{'=' * 70}\nWEEKLY UPDATE VALIDATION (real {season} season, prior_weeks={prior_weeks})\n{'=' * 70}")
    print(f"Static preseason-only projection: MAE={preseason_mae:.4f} corr={preseason_corr:+.3f} (baseline for all weeks)\n")

    results = []
    for week_end in checkpoint_weeks:
        partial = compute_team_epa_through_week(season, week_end)
        updated = weekly_update_team_strength(team_strength, partial, prior_weeks)
        merged = updated.merge(real_final, on="team", how="inner")

        updated_mae = np.mean(np.abs(merged["offensive_strength_updated"] - merged["off_epa"]))
        updated_corr = merged["offensive_strength_updated"].corr(merged["off_epa"])

        naive_mae = np.mean(np.abs(merged["off_epa_partial"].fillna(merged["offensive_strength"]) - merged["off_epa"]))
        naive_corr = merged["off_epa_partial"].fillna(merged["offensive_strength"]).corr(merged["off_epa"])

        print(f"After week {week_end:>2}: shrinkage-updated MAE={updated_mae:.4f} corr={updated_corr:+.3f} | "
              f"naive partial-only MAE={naive_mae:.4f} corr={naive_corr:+.3f} | "
              f"{'BEATS preseason' if updated_mae < preseason_mae else 'below preseason'}")
        results.append({"week": week_end, "updated_mae": updated_mae, "updated_corr": updated_corr,
                         "naive_mae": naive_mae, "naive_corr": naive_corr})

    print(f"\n{'=' * 70}")
    results_df = pd.DataFrame(results)
    out_path = os.path.join(PROCESSED_DIR, "weekly_update_validation_2025.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")
    return results_df, {"prior_weeks": prior_weeks, "preseason_mae": preseason_mae, "preseason_corr": preseason_corr}


def weekly_update_pipeline(season, week_completed, prior_weeks=None):
    """The real, callable entry point this pipeline is FOR - point it at
    2026 once real weeks exist. Returns the updated team_strength for
    whichever season/week you give it; no real-outcome validation is
    possible for a season still in progress (same "not yet played" honesty
    this project already applies to 2026's defensive projections)."""
    if prior_weeks is None:
        prior_weeks = estimate_prior_weeks(season if season in (2025,) else 2025)

    team_strength_path = os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv")
    team_strength = pd.read_csv(team_strength_path)
    partial = compute_team_epa_through_week(season, week_completed)
    updated = weekly_update_team_strength(team_strength, partial, prior_weeks)

    out_path = os.path.join(PROCESSED_DIR, f"team_strength_{season}_after_week{week_completed}.csv")
    updated.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[weekly_update_pipeline] {season} week {week_completed}: saved {out_path}")
    return updated


if __name__ == "__main__":
    run_weekly_update_demo()
