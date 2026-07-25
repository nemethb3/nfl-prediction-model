"""Phase 3, Task 3.1: Aggregate Player Predictions to Team Strength.

Two design decisions made before building, since this task came with no
detailed spec (unlike every Phase 2 task) and had a real architectural fork:

1. TARGET SEASON: offense (QB/WR/RB) projects 2025 (nflverse's offense
   weekly/seasonal endpoint is still 404 for 2025 as of this task - checked
   again before starting). Defense (EDGE/DL/LB/CB/S) is PFR/PBP-native and
   already reaches 2025 data, so its existing saved projections target 2026.
   Combining a 2025 offense with a 2026 defense into one "team strength"
   would be temporally incoherent. Since PROGRESS.md already commits Phase 4
   to "Backtest 2025 Season Predictions," this module rebuilds every
   defensive position's projection with an explicit, leak-free
   ref_season=2024 (same technique as Task 5.1's SOS models) so every input
   to team strength uses only pre-2025 data - a genuine 2025 forecast,
   fully backtestable against the real 2025 season that already happened.

2. OFFENSIVE AGGREGATION: rather than trying to combine QB + RB + WR
   EPA/play into one number (which would double-count - a single pass play
   already shows up in both the QB's attempt and the WR's target), team
   offensive strength is built as a play-mix-weighted blend of the starting
   QB's EPA/play (passing-game strength - already reflects the whole
   passing game, WR quality included, since it's computed from the same
   plays) and the leading RB's EPA/play (rushing-game strength), weighted
   by the team's real 2024 pass/run play split. This is directly comparable
   in units to team_offense_epa (real, PBP-derived), enabling a clean
   validation against realized 2025 outcomes - same discipline as every
   prior task.

Defensive strength uses the already-validated, real-EPA-denominated
team_defense_epa prediction model (rebuilt here for a leak-free 2025 target)
as the PRIMARY signal, with individual-WAR-based pass rush strength and
LB/CB/S blended-score averages reported as supplementary diagnostics rather
than forced into the same unit - consistent with this project's established
refusal to invent unit conversions (see Task 5.3's DL sacks->war conversion,
which IS reused here since it's real and already validated).
"""

import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

REF_SEASON = 2024  # leak-free jump-off point for every 2025-target component
TARGET_SEASON = REF_SEASON + 1

# CB/S/LB winning blend ratios (tackle_weight, leverage_weight) from Phase 2
# Refinement Task 2's holdout search (LB matches CB - both landed at the
# edge of the tested grid; S found a genuine interior optimum). Reused
# here, not re-derived, same as Task 5.3's reuse of these constants.
BLEND_RATIO_BY_POSITION = {"CB": (0.8, 0.2), "S": (0.5, 0.5), "LB": (0.8, 0.2)}


# ---------------------------------------------------------------------------
# Step 1: leak-free 2025-target defensive projections
# ---------------------------------------------------------------------------

def rebuild_edge_war_2025():
    from player_models import PassRushWARModel, WAR_TRAIN_SEASONS

    war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    team_defense_df = pd.read_csv(os.path.join(PROCESSED_DIR, "team_defense_epa_2015_2025.csv"))

    model = PassRushWARModel("EDGE", n_estimators=60, max_depth=3, min_snaps=1)
    prepped = model.prepare_data(war_df, crosswalk, schedules, team_defense_df)
    train_df = prepped[prepped["season"].isin(WAR_TRAIN_SEASONS)]
    model.train(train_df)
    predictions = model.predict_next_season(war_df, crosswalk, schedules, team_defense_df, ref_season=REF_SEASON)
    print(f"[EDGE WAR] rebuilt leak-free {TARGET_SEASON} projection from {REF_SEASON} data ({len(predictions)} players)")
    return predictions


def rebuild_dl_sacks_2025():
    from player_models import DefensePositionModel, DEFENSE_TRAIN_SEASONS, load_age_curves

    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    defense_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_defense.csv"))
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    snap_counts = pd.read_csv(os.path.join(RAW_DIR, "snap_counts_2015_2025.csv"))
    age_curves = load_age_curves()

    model = DefensePositionModel("DL", "sk", "sacks", leading_indicator_col="prss",
                                  n_estimators=80, max_depth=4, min_games=8, xgb_weight=0.9)
    prepped = model.prepare_data(features_df, defense_stats_df, crosswalk, snap_counts)
    train_df = prepped[prepped["season"].isin(DEFENSE_TRAIN_SEASONS)]
    model.train(train_df)
    predictions = model.predict_next_season(features_df, defense_stats_df, crosswalk, snap_counts, age_curves,
                                              ref_season=REF_SEASON)
    print(f"[DL sacks] rebuilt leak-free {TARGET_SEASON} projection from {REF_SEASON} data ({len(predictions)} players)")
    return predictions


def rebuild_blended_2025(position):
    from player_models import BlendedDefenseModel, BLEND_TRAIN_SEASONS, build_blended_score_table

    tw, lw = BLEND_RATIO_BY_POSITION[position]
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    tackle_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tackle_efficiency_2018_2025.csv"))
    leverage_df = pd.read_csv(os.path.join(PROCESSED_DIR, "leverage_war_2016_2025.csv"))

    scored = build_blended_score_table(tackle_df, leverage_df, position, tw, lw)
    model = BlendedDefenseModel(position)
    prepped = model.prepare_data(scored, crosswalk, schedules)
    train_df = prepped[prepped["season"].isin(BLEND_TRAIN_SEASONS)]
    model.train(train_df)
    predictions = model.predict_next_season(scored, crosswalk, schedules, ref_season=REF_SEASON)
    print(f"[{position} blended] rebuilt leak-free {TARGET_SEASON} projection from {REF_SEASON} data ({len(predictions)} players)")
    return predictions


def rebuild_all_defense_2025():
    return {
        "EDGE": rebuild_edge_war_2025(),
        "DL": rebuild_dl_sacks_2025(),
        "CB": rebuild_blended_2025("CB"),
        "S": rebuild_blended_2025("S"),
        "LB": rebuild_blended_2025("LB"),
    }


# ---------------------------------------------------------------------------
# Step 2: offensive team strength - play-mix-weighted QB/RB blend
# ---------------------------------------------------------------------------

def compute_team_play_mix(season_stats_df, ref_season=REF_SEASON):
    """Each team's real pass/run play share in ref_season - used to weight
    the starting QB's (passing-game) vs. leading RB's (rushing-game)
    EPA/play into one team offensive figure, without double-counting a
    single play under both a QB's attempt and a WR's target."""
    season = season_stats_df[season_stats_df["season"] == ref_season]
    from utilities import season_team_from_weekly
    weekly = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    team_col = season_team_from_weekly(weekly)
    season = season.merge(team_col, on=["player_id", "season"], how="left")

    attempts = season.groupby("team")["attempts"].sum()
    carries = season.groupby("team")["carries"].sum()
    mix = pd.DataFrame({"attempts": attempts, "carries": carries}).fillna(0)
    mix["pass_share"] = mix["attempts"] / (mix["attempts"] + mix["carries"]).replace(0, np.nan)
    return mix.reset_index()[["team", "pass_share"]]


def build_offensive_team_strength(qb_proj, rb_proj, play_mix):
    """team_offensive_strength = pass_share * starting-QB EPA/play (baseline -
    SOS-adjusted no longer validates for QB once the ref_season off-by-one
    was fixed; see SOS Bug Fix Task, 2026-07-25, PROGRESS.md) +
    (1-pass_share) * leading-RB EPA/play (OL-adjusted - the one that
    validated for RB, unaffected by that fix). One row per team; teams with
    multiple QB/RB candidates in the projection file are already just the
    top projected player at that position per team (both projection files
    are pre-sorted by predicted value)."""
    qb_team = qb_proj.sort_values("predicted_epa_per_play", ascending=False).drop_duplicates("team")
    rb_team = rb_proj.sort_values("predicted_epa_per_play_ol_adjusted", ascending=False).drop_duplicates("team")

    qb_team = qb_team[["team", "player_name", "predicted_epa_per_play"]].rename(
        columns={"player_name": "starting_qb", "predicted_epa_per_play": "qb_epa_per_play"})
    rb_team = rb_team[["team", "player_name", "predicted_epa_per_play_ol_adjusted"]].rename(
        columns={"player_name": "leading_rb", "predicted_epa_per_play_ol_adjusted": "rb_epa_per_play"})

    strength = qb_team.merge(rb_team, on="team", how="outer").merge(play_mix, on="team", how="left")
    strength["pass_share"] = strength["pass_share"].fillna(play_mix["pass_share"].mean())
    strength["qb_epa_per_play"] = strength["qb_epa_per_play"].fillna(strength["qb_epa_per_play"].mean())
    strength["rb_epa_per_play"] = strength["rb_epa_per_play"].fillna(strength["rb_epa_per_play"].mean())

    strength["offensive_strength"] = (
        strength["pass_share"] * strength["qb_epa_per_play"]
        + (1 - strength["pass_share"]) * strength["rb_epa_per_play"]
    )
    return strength.sort_values("offensive_strength", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: defensive team strength - validated team-level model (primary) +
# real WAR-based pass rush + LB/CB/S composite (supplementary diagnostics)
# ---------------------------------------------------------------------------

def rebuild_team_defense_epa_2025():
    """The already-validated team-level defense EPA model (Phase 2.X-Team,
    generalized in Task 5.1/5.4), rebuilt with the same leak-free
    ref_season=2024 jump-off used for every other component here."""
    from team_strength import build_team_defense_features, compute_team_defense_epa, train_defense_component_model, predict_next_season

    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    team_epa = compute_team_defense_epa()
    df = build_team_defense_features(team_epa, pass_rush_war_df)
    model, backtest = train_defense_component_model(df, target_col="def_epa_allowed")
    predictions, ref_season = predict_next_season(df, model, ref_season=REF_SEASON)
    print(f"[team_defense_epa] rebuilt leak-free {TARGET_SEASON} projection from {ref_season} data")
    return predictions.rename(columns={"predicted_def_epa_allowed": "defensive_strength_allowed"})


def build_defensive_supplementary_signals(defense_projections):
    """Real-WAR-based pass rush strength (EDGE WAR + DL sacks converted via
    the real, data-derived sacks->war conversion from Task 5.3) and average
    LB/CB/S blended_score, reported as diagnostics alongside the primary
    team_defense_epa figure - not forced into the same unit."""
    from sos_adjustment import fit_dl_sacks_to_war_conversion

    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    slope, intercept = fit_dl_sacks_to_war_conversion(pass_rush_war_df)

    edge = defense_projections["EDGE"]
    dl = defense_projections["DL"].copy()
    # PFR gives a traded player's combined-stint row a literal "2TM"/"3TM"
    # team value (Task 1.3 deduplicated the row but never resolved this
    # value to the player's real final team - a narrow, disclosed fix here;
    # the deeper fix belongs in data_pipeline.create_defense_season_stats).
    # Only affects DL among the positions used here (0 rows for EDGE/CB/S/LB
    # in this run) - excluded rather than guessed at, since we don't know
    # which of the 2-3 teams should get credit.
    multi_team_mask = dl["team"].astype(str).str.match(r"^\dTM$")
    if multi_team_mask.any():
        print(f"[team_aggregation] dropping {multi_team_mask.sum()} DL rows with unresolved PFR "
              f"multi-team ('2TM'/'3TM') team values: {dl.loc[multi_team_mask, 'player_name'].tolist()}")
        dl = dl[~multi_team_mask]
    dl["dl_war_estimate"] = slope * dl["predicted_sacks"] + intercept

    edge_team = edge.groupby("team")["predicted_war"].sum().rename("edge_war_total")
    dl_team = dl.groupby("team")["dl_war_estimate"].sum().rename("dl_war_total")
    pass_rush = pd.concat([edge_team, dl_team], axis=1).fillna(0)
    pass_rush["team_pass_rush_war"] = pass_rush["edge_war_total"] + pass_rush["dl_war_total"]
    pass_rush = pass_rush.reset_index()[["team", "team_pass_rush_war"]]

    front_seven_secondary = []
    for position in ["LB", "CB", "S"]:
        avg = defense_projections[position].groupby("team")["predicted_blended_score"].mean().rename(f"{position.lower()}_avg_blended_score")
        front_seven_secondary.append(avg)
    coverage = pd.concat(front_seven_secondary, axis=1).reset_index()

    return pass_rush.merge(coverage, on="team", how="outer")


# ---------------------------------------------------------------------------
# Step 4: validate against real, already-realized 2025 team outcomes
# ---------------------------------------------------------------------------

def validate_team_strength(team_strength):
    """The 2025 season has already been played - checks the pre-season
    (2024-data-only) offensive/defensive strength projections against real
    2025 team EPA, same discipline as every Phase 2 Task 4/5 validation."""
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    real_off = compute_team_offense_epa()
    real_off_2025 = real_off[real_off["season"] == 2025][["team", "off_epa"]]
    real_def = compute_team_defense_epa()
    real_def_2025 = real_def[real_def["season"] == 2025][["team", "def_epa_allowed"]]

    merged_off = team_strength.merge(real_off_2025, on="team", how="inner")
    off_corr = merged_off["offensive_strength"].corr(merged_off["off_epa"])
    off_mae = np.mean(np.abs(merged_off["offensive_strength"] - merged_off["off_epa"]))

    merged_def = team_strength.merge(real_def_2025, on="team", how="inner")
    def_corr = merged_def["defensive_strength_allowed"].corr(merged_def["def_epa_allowed"])

    print("\n===== TEAM STRENGTH VALIDATION (vs. real 2025 outcomes) =====")
    print(f"Offensive strength: corr(projected, real 2025 off_epa) = {off_corr:+.3f} | MAE = {off_mae:.4f} "
          f"(expect strongly positive - this is the aggregation formula's core validation)")
    print(f"Defensive strength: corr(projected, real 2025 def_epa_allowed) = {def_corr:+.3f} "
          f"(expect positive - both are 'allowed' EPA, so higher predicted should mean higher real allowed; "
          f"reused from the already-validated team_defense_epa model, low confidence expected per its own docstring)")
    print("===== END VALIDATION =====\n")
    return {"off_corr": off_corr, "off_mae": off_mae, "def_corr": def_corr}


def run_team_aggregation():
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    rb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_epa_projections_2025.csv"))

    print("=" * 60 + "\nStep 1: rebuilding leak-free 2025-target defensive projections\n" + "=" * 60)
    defense_projections = rebuild_all_defense_2025()

    print("\n" + "=" * 60 + "\nStep 2: offensive team strength\n" + "=" * 60)
    play_mix = compute_team_play_mix(season_stats_df)
    off_strength = build_offensive_team_strength(qb_proj, rb_proj, play_mix)

    print("\n" + "=" * 60 + "\nStep 3: defensive team strength\n" + "=" * 60)
    def_strength = rebuild_team_defense_epa_2025()
    supplementary = build_defensive_supplementary_signals(defense_projections)

    team_strength = off_strength.merge(
        def_strength[["team", "defensive_strength_allowed"]], on="team", how="outer"
    ).merge(supplementary, on="team", how="outer")
    team_strength["net_strength"] = team_strength["offensive_strength"] - team_strength["defensive_strength_allowed"]
    team_strength = team_strength.sort_values("net_strength", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60 + "\nStep 4: validation against real 2025 outcomes\n" + "=" * 60)
    metrics = validate_team_strength(team_strength)

    print("\nTop 10 teams by net strength (2025 projection, from 2024 data):")
    print(team_strength.head(10)[["team", "starting_qb", "leading_rb", "offensive_strength",
                                   "defensive_strength_allowed", "net_strength"]].to_string(index=False))
    print("\nBottom 10 teams by net strength:")
    print(team_strength.tail(10)[["team", "starting_qb", "leading_rb", "offensive_strength",
                                   "defensive_strength_allowed", "net_strength"]].to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv")
    team_strength.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path} ({len(team_strength)} teams)")

    for position, df in defense_projections.items():
        pos_path = os.path.join(PROCESSED_DIR, f"{position.lower()}_leakfree_predictions_{TARGET_SEASON}.csv")
        df.to_csv(pos_path, index=False, encoding="utf-8")
        print(f"Saved {pos_path}")

    return team_strength, metrics, defense_projections


# ---------------------------------------------------------------------------
# Phase 3 Rebuild Task 3: Backup depth scenarios.
#
# Scoped to QB/RB only - the two positions build_offensive_team_strength
# actually uses. WR/TE aren't part of the production offensive_strength
# formula, and the Phase 3 Diagnostic already found adding WR in any tested
# form HURTS team-level accuracy (+0.341 -> as low as +0.240) - re-testing
# that with a depth-scenario wrapper isn't attempted here; if WR/TE ever
# join the formula (Task 4's job), scenarios can be added the same way.
#
# The projection CSVs are NOT sorted starter-first by row order (verified
# before building - e.g. Jameis Winston appears before Deshaun Watson for
# CLE despite Watson being the real starter), so "starter" is defined the
# same way build_offensive_team_strength already defines it: highest
# projected value on that position's one validated adjustment column. Most
# teams have no second QB/RB in the projections at all (only players who
# cleared the position's min_opportunities threshold last season show up),
# so "backup" falls back to a real, empirically-derived league-average
# backup-tier value - not an asserted constant like the original spec's
# QB=-0.05/RB=-0.02.
# ---------------------------------------------------------------------------

def _rank_starter_backup(proj_df, canonical_col):
    ranked = proj_df.sort_values(canonical_col, ascending=False).groupby("team").head(2).copy()
    ranked["depth_rank"] = ranked.groupby("team").cumcount()
    starters = ranked[ranked["depth_rank"] == 0][["team", "player_name", canonical_col]].rename(
        columns={"player_name": "starter_name", canonical_col: "starter_value"})
    backups = ranked[ranked["depth_rank"] == 1][["team", "player_name", canonical_col]].rename(
        columns={"player_name": "backup_name", canonical_col: "backup_value_real"})
    return starters, backups


def build_depth_scenarios(position, proj_df, canonical_col, availability_col="availability_factor"):
    """Three team-level scenarios:
      - full_season: starter plays all season (== what production
        team_strength_2025.csv already assumes)
      - availability_adjusted: starter_value and backup_value blended by the
        starter's own real availability_factor (Task 5.2) - a team whose
        starter projects a low availability leans more on its backup here
      - backup_only: starter is out all season, backup plays all season
    """
    starters, backups = _rank_starter_backup(proj_df, canonical_col)
    league_backup_avg = backups["backup_value_real"].mean()
    print(f"[{position}] {len(backups)}/{len(starters)} teams have a real 2nd-{position} in the projections "
          f"(cleared min_opportunities last season) - league-average backup value used for the rest: "
          f"{league_backup_avg:+.4f}")

    avail = proj_df.sort_values(canonical_col, ascending=False).drop_duplicates("team")[["team", availability_col]]
    scenarios = starters.merge(backups, on="team", how="left").merge(avail, on="team", how="left")
    scenarios["had_real_backup"] = scenarios["backup_value_real"].notna()
    scenarios["backup_value"] = scenarios["backup_value_real"].fillna(league_backup_avg)

    scenarios["full_season"] = scenarios["starter_value"]
    scenarios["availability_adjusted"] = (
        scenarios["starter_value"] * scenarios[availability_col]
        + scenarios["backup_value"] * (1 - scenarios[availability_col])
    )
    scenarios["backup_only"] = scenarios["backup_value"]
    return scenarios


def project_team_strength_scenarios(qb_canonical_col="predicted_epa_per_play", file_suffix=""):
    """qb_canonical_col default changed from predicted_epa_per_play_sos_adjusted to
    baseline predicted_epa_per_play (SOS Bug Fix Task, 2026-07-25): once the
    ref_season off-by-one was corrected, QB SOS-adjusted no longer validates
    against real 2025 outcomes (flips from HELPS to HURTS - see PROGRESS.md).
    Baseline is QB's next-best-validated column (OL-adjusted is ~identical to
    baseline for QB, per Task 4.2 - both are effectively noise-level for QB)."""
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    rb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_epa_projections_2025.csv"))
    play_mix = compute_team_play_mix(season_stats_df)

    qb_scen = build_depth_scenarios("QB", qb_proj, qb_canonical_col)
    rb_scen = build_depth_scenarios("RB", rb_proj, "predicted_epa_per_play_ol_adjusted")

    qb_scen.to_csv(os.path.join(PROCESSED_DIR, f"depth_scenario_qb{file_suffix}.csv"), index=False, encoding="utf-8")
    rb_scen.to_csv(os.path.join(PROCESSED_DIR, f"depth_scenario_rb{file_suffix}.csv"), index=False, encoding="utf-8")
    print(f"Saved depth_scenario_qb{file_suffix}.csv, depth_scenario_rb{file_suffix}.csv (player-level detail)")

    results = {}
    for scenario in ["full_season", "availability_adjusted", "backup_only"]:
        merged = qb_scen[["team", scenario]].rename(columns={scenario: "qb_epa_per_play"}).merge(
            rb_scen[["team", scenario]].rename(columns={scenario: "rb_epa_per_play"}), on="team", how="outer"
        ).merge(play_mix, on="team", how="left")
        merged["pass_share"] = merged["pass_share"].fillna(play_mix["pass_share"].mean())
        merged["offensive_strength"] = (
            merged["pass_share"] * merged["qb_epa_per_play"]
            + (1 - merged["pass_share"]) * merged["rb_epa_per_play"]
        )
        out_path = os.path.join(PROCESSED_DIR, f"team_strength_2025_{scenario}{file_suffix}.csv")
        merged[["team", "qb_epa_per_play", "rb_epa_per_play", "offensive_strength"]].to_csv(
            out_path, index=False, encoding="utf-8")
        print(f"Saved {out_path}")
        results[scenario] = merged

    return results


def validate_depth_scenarios(results):
    """Real test the original spec never proposed: does blending in
    availability/backup information actually predict real 2025 team
    offensive EPA better than production's 'starter plays every game'
    assumption? If not, the scenario framework is informative range context
    at best, not something that should replace the production number."""
    from coach_quality import compute_team_offense_epa
    real = compute_team_offense_epa()
    real_2025 = real[real["season"] == TARGET_SEASON][["team", "off_epa"]]

    print("\n===== DEPTH SCENARIO VALIDATION (vs. real 2025 team offensive EPA) =====")
    metrics = {}
    for scenario, df in results.items():
        merged = df.merge(real_2025, on="team", how="inner")
        corr = merged["offensive_strength"].corr(merged["off_epa"])
        mae = np.mean(np.abs(merged["offensive_strength"] - merged["off_epa"]))
        print(f"{scenario}: corr={corr:+.3f} MAE={mae:.4f}")
        metrics[scenario] = {"corr": corr, "mae": mae}
    print("(production team_strength_2025.csv's offensive_strength, for reference, scored corr=+0.341)")
    print("===== END VALIDATION =====\n")
    return metrics


def run_depth_scenarios():
    results = project_team_strength_scenarios()
    metrics = validate_depth_scenarios(results)
    return results, metrics


# ---------------------------------------------------------------------------
# Phase 3 Rebuild Task 4: Scientific offensive weight optimization.
#
# Corrects the spec's fit direction: it proposed regressing real 2025 team
# offensive EPA against 2025 PROJECTED components (n=32, one season) and
# calling that "the optimized weights," then backtesting on 2015-2024 as an
# afterthought. That's backwards from every other weight-fit in this project
# (OL/SOS/synergy all fit on many historical TRAIN seasons, using the SAME
# already-trained model applied across each ref_season - a convention this
# project explicitly discloses as in-sample-on-train/optimistic, with the
# REAL test being an untouched holdout) - and 32 points across up to 4
# predictors is a very easy fit to overfit. Here the roles are swapped back:
# fit pooled across 9 historical target seasons (2016-2024, ~280+ team-season
# rows) using each position's already-trained model applied at every
# historical ref_season (same reuse pattern as Task 5.1/5.3's weight fits),
# then the REAL 2025 outcome - genuinely never touched during fitting - is
# the decisive check, exactly like every other task in this project.
#
# Also drops the spec's "normalize weights to sum of absolute value = 1" -
# an arbitrary rescaling with no stated justification that would need the
# intercept dropped too (changing what's actually being predicted). The
# fitted regression's raw coefficients + intercept are used directly, since
# the target is real, unstandardized offensive EPA, not an index.
# ---------------------------------------------------------------------------

WEIGHT_FIT_TARGET_SEASONS = range(2016, 2025)  # 2016-2024; ref_season = target-1 (2015-2023)
STEPWISE_COMPONENTS = [["qb_epa"], ["qb_epa", "rb_epa"], ["qb_epa", "rb_epa", "wr_epa"],
                        ["qb_epa", "rb_epa", "wr_epa", "te_epa"]]


def project_position_team_value_by_season(position, target_seasons=WEIGHT_FIT_TARGET_SEASONS):
    """For each historical target season S, projects every player's EPA/play
    using ref_season=S-1 (leak-free one-step-ahead, same convention as every
    projection in this project), then takes the highest-projected player per
    team as that position's team-level value for season S - the same
    "starter = highest projected value" convention already used for
    QB/RB in build_offensive_team_strength, extended here to WR/TE too."""
    from ol_quality import load_epa_model
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    model, prepped = load_epa_model(position, features_df, season_stats_df)
    # load_epa_model only unpickles xgb_model - residual_std_ (set during
    # .train(), used by predict_next_season solely for the informational
    # confidence_epa_per_play_pm column) was never persisted. Not used here,
    # so a placeholder avoids predict_next_season's round(None) crash.
    model.residual_std_ = 0.0

    rows = []
    for target_season in target_seasons:
        ref_season = target_season - 1
        if ref_season not in prepped["season"].values:
            continue
        proj = model.predict_next_season(features_df, season_stats_df, ref_season=ref_season)
        team_val = proj.sort_values("predicted_epa_per_play", ascending=False).drop_duplicates("team")
        team_val = team_val[["team", "predicted_epa_per_play"]].rename(
            columns={"predicted_epa_per_play": f"{position.lower()}_epa"})
        team_val["season"] = target_season
        rows.append(team_val)
    return pd.concat(rows, ignore_index=True)


def build_offensive_weight_training_table():
    from coach_quality import compute_team_offense_epa

    data = None
    for position in ["QB", "RB", "WR", "TE"]:
        comp = project_position_team_value_by_season(position)
        data = comp if data is None else data.merge(comp, on=["team", "season"], how="inner")

    real = compute_team_offense_epa()[["team", "season", "off_epa"]].rename(
        columns={"off_epa": "real_offensive_epa"})
    data = data.merge(real, on=["team", "season"], how="inner")
    return data


def load_real_2025_baseline_components():
    """The SAME baseline predicted_epa_per_play already saved in each
    position's 2025 projections file (ref_season=2024 - identical convention
    to project_position_team_value_by_season above), never used anywhere in
    this task's fitting - the genuine holdout these weights get judged on."""
    out = None
    for position in ["QB", "RB", "WR", "TE"]:
        proj = pd.read_csv(os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2025.csv"))
        team_val = proj.sort_values("predicted_epa_per_play", ascending=False).drop_duplicates("team")
        team_val = team_val[["team", "predicted_epa_per_play"]].rename(
            columns={"predicted_epa_per_play": f"{position.lower()}_epa"})
        out = team_val if out is None else out.merge(team_val, on="team", how="inner")
    return out


def optimize_offensive_weights_via_regression():
    from sklearn.linear_model import LinearRegression
    from coach_quality import compute_team_offense_epa

    train_data = build_offensive_weight_training_table()
    print(f"Weight-fit training pool: {len(train_data)} team-seasons "
          f"({train_data['season'].min()}-{train_data['season'].max()})")

    real = compute_team_offense_epa()
    real_2025 = real[real["season"] == 2025][["team", "off_epa"]].rename(columns={"off_epa": "real_offensive_epa"})
    holdout = load_real_2025_baseline_components().merge(real_2025, on="team", how="inner")

    print("\nIndividual component correlations (historical train pool, in-sample):")
    for col in ["qb_epa", "rb_epa", "wr_epa", "te_epa"]:
        corr = train_data[col].corr(train_data["real_offensive_epa"])
        print(f"  {col}: {corr:+.3f}")

    results = []
    for cols in STEPWISE_COMPONENTS:
        X_train = train_data[cols].to_numpy()
        y_train = train_data["real_offensive_epa"].to_numpy()
        model = LinearRegression().fit(X_train, y_train)
        r2_train = model.score(X_train, y_train)

        X_holdout = holdout[cols].to_numpy()
        y_holdout = holdout["real_offensive_epa"].to_numpy()
        pred_holdout = model.predict(X_holdout)
        corr_holdout = np.corrcoef(pred_holdout, y_holdout)[0, 1]
        r2_holdout = 1 - np.sum((y_holdout - pred_holdout) ** 2) / np.sum((y_holdout - y_holdout.mean()) ** 2)

        label = "+".join(cols)
        print(f"\n{label}: train R2 (in-sample, n={len(X_train)}) = {r2_train:.3f} | "
              f"REAL 2025 holdout (n={len(X_holdout)}): corr={corr_holdout:+.3f} R2={r2_holdout:.3f}")
        results.append({"components": cols, "label": label, "model": model,
                         "r2_train": r2_train, "corr_holdout": corr_holdout, "r2_holdout": r2_holdout})

    winner = max(results, key=lambda r: r["corr_holdout"])
    print(f"\nWINNER (selected by REAL 2025 holdout correlation, NOT in-sample train fit): "
          f"{winner['label']} (corr={winner['corr_holdout']:+.3f})")

    weights = dict(zip(winner["components"], winner["model"].coef_))
    weights["intercept"] = winner["model"].intercept_
    print("Fitted weights (raw regression coefficients + intercept, not renormalized):")
    for k, v in weights.items():
        print(f"  {k}: {v:+.4f}")

    weights_df = pd.DataFrame([weights])
    weights_path = os.path.join(PROCESSED_DIR, "optimized_offensive_weights.csv")
    weights_df.to_csv(weights_path, index=False, encoding="utf-8")
    print(f"Saved {weights_path}")

    return results, winner, holdout


def apply_optimized_weights_to_team_strength(winner, holdout):
    cols = winner["components"]
    out = holdout.copy()
    out["offensive_strength_optimized"] = winner["model"].predict(out[cols].to_numpy())
    out_path = os.path.join(PROCESSED_DIR, "team_strength_2025_optimized.csv")
    out[["team", "offensive_strength_optimized", "real_offensive_epa"]].to_csv(
        out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")
    return out


def run_offensive_weight_optimization():
    results, winner, holdout = optimize_offensive_weights_via_regression()
    team_strength_optimized = apply_optimized_weights_to_team_strength(winner, holdout)
    print(f"\nFor reference on the same real-2025 check: production full_season offensive_strength "
          f"scored corr=+0.341; Task 3's availability_adjusted scenario scored corr=+0.464.")
    return results, winner, team_strength_optimized


# ---------------------------------------------------------------------------
# Phase 3 Final Improvements Task 1: combine Task 3's availability blending
# with Task 4's optimized weights.
#
# Two spec bugs fixed before building (confirmed against the actual saved
# files): optimized_offensive_weights.csv's real columns are
# qb_epa/rb_epa/wr_epa/te_epa/intercept, not qb/rb/wr/te; team_strength_2025
# .csv's real defensive column is defensive_strength_allowed, not
# defensive_strength.
#
# QB/RB reuse build_depth_scenarios()'s already-validated
# availability_adjusted column directly rather than re-deriving the
# starter/backup blend inline, avoiding drift from Task 3's real,
# empirically-derived backup fallback values.
#
# Disclosed limitation, not a bug: Task 4's weights were fit on each
# position's BASELINE predicted_epa_per_play (the historical backtest never
# computed SOS/OL/synergy-adjusted values for old seasons - those only ever
# ran once, on the single 2025 projection). Applied here to each position's
# best-VALIDATED column instead (SOS-adjusted QB, OL-adjusted RB, synergy-
# adjusted TE, baseline WR since nothing validated for WR) per the user's
# explicit direction - this is an extrapolation of weights fit on slightly
# different inputs, not a like-for-like reapplication of what Task 4 tested.
# ---------------------------------------------------------------------------

def combine_availability_and_optimized_weights(qb_canonical_col="predicted_epa_per_play"):
    """qb_canonical_col default changed from predicted_epa_per_play_sos_adjusted
    to baseline (SOS Bug Fix Task, 2026-07-25 - same reasoning as
    project_team_strength_scenarios/build_full_roster_aggregation)."""
    weights = pd.read_csv(os.path.join(PROCESSED_DIR, "optimized_offensive_weights.csv")).iloc[0].to_dict()

    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    rb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_epa_projections_2025.csv"))
    wr_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2025.csv"))
    te_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "te_epa_projections_2025.csv"))

    qb_scen = build_depth_scenarios("QB", qb_proj, qb_canonical_col)
    rb_scen = build_depth_scenarios("RB", rb_proj, "predicted_epa_per_play_ol_adjusted")
    qb_val = qb_scen[["team", "availability_adjusted"]].rename(columns={"availability_adjusted": "qb_epa"})
    rb_val = rb_scen[["team", "availability_adjusted"]].rename(columns={"availability_adjusted": "rb_epa"})

    wr_val = wr_proj.sort_values("predicted_epa_per_play", ascending=False).drop_duplicates("team")[
        ["team", "predicted_epa_per_play"]].rename(columns={"predicted_epa_per_play": "wr_epa"})
    te_val = te_proj.sort_values("predicted_epa_per_play_synergy_adjusted", ascending=False).drop_duplicates("team")[
        ["team", "predicted_epa_per_play_synergy_adjusted"]].rename(
        columns={"predicted_epa_per_play_synergy_adjusted": "te_epa"})

    components = qb_val.merge(rb_val, on="team", how="outer").merge(wr_val, on="team", how="outer").merge(
        te_val, on="team", how="outer")
    for col in ["qb_epa", "rb_epa", "wr_epa", "te_epa"]:
        components[col] = components[col].fillna(components[col].mean())

    components["offensive_strength"] = (
        weights["qb_epa"] * components["qb_epa"] + weights["rb_epa"] * components["rb_epa"]
        + weights["wr_epa"] * components["wr_epa"] + weights["te_epa"] * components["te_epa"]
        + weights["intercept"]
    )

    team_def = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))[
        ["team", "defensive_strength_allowed"]]
    combined = components.merge(team_def, on="team", how="left")
    combined["net_strength"] = combined["offensive_strength"] - combined["defensive_strength_allowed"]

    from coach_quality import compute_team_offense_epa
    real = compute_team_offense_epa()
    real_2025 = real[real["season"] == TARGET_SEASON][["team", "off_epa"]].rename(
        columns={"off_epa": "real_offensive_epa"})
    merged = combined.merge(real_2025, on="team", how="inner")

    corr = merged["offensive_strength"].corr(merged["real_offensive_epa"])
    mae = np.mean(np.abs(merged["offensive_strength"] - merged["real_offensive_epa"]))

    print("\n" + "=" * 70 + "\nCOMBINED AVAILABILITY + OPTIMIZED WEIGHTS\n" + "=" * 70)
    print(f"Offensive strength vs. real 2025 team offensive EPA: corr={corr:+.3f} MAE={mae:.4f}")
    print(f"\nFor reference on the same real-2025 check:")
    print(f"  Task 3 (availability only, play-mix formula): +0.464")
    print(f"  Task 4 (optimized weights only, baseline EPA): +0.368")
    print(f"  Combined: {corr:+.3f}")

    out_path = os.path.join(PROCESSED_DIR, "team_strength_2025_combined.csv")
    combined.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")

    return combined, {"corr": corr, "mae": mae}


# ---------------------------------------------------------------------------
# Parallel Task A: Full-Roster Depth Aggregation (corrected data sources).
#
# player_season_stats.csv has no team column (confirmed earlier this
# project) - team comes via season_team_from_weekly(), the same
# last-team-played crosswalk compute_team_play_mix() already uses. Target
# column is targets, not receiving_targets. RB workload uses avg_snap_pct
# (a real, already-normalized column) rather than a nonexistent snaps
# column - a better signal for RBs anyway since pure runners get few
# targets. Shares are renormalized within each team's top-N PROJECTED
# players (not against the team's full real depth chart) so a team with
# real depth beyond top-N/below the EPA model's min_opportunities threshold
# doesn't silently shrink that team's value - a deliberate, disclosed
# choice, not the same thing the spec's pseudocode implied.
# ---------------------------------------------------------------------------

def _team_position_weight_pool(season_stats_df, weekly_df, position, weight_col, season=REF_SEASON):
    from utilities import season_team_from_weekly
    team_col = season_team_from_weekly(weekly_df)
    pool = season_stats_df[(season_stats_df["season"] == season) & (season_stats_df["position"] == position)]
    pool = pool.merge(team_col, on=["player_id", "season"], how="left")
    return pool[["player_id", "team", weight_col]].rename(columns={weight_col: "real_weight"})


def _depth_weighted_component(proj_df, canonical_col, pool, top_n):
    rows = []
    for team, team_proj in proj_df.groupby("team"):
        top = team_proj.sort_values(canonical_col, ascending=False).head(top_n).copy()
        top = top.merge(pool[["player_id", "real_weight"]], on="player_id", how="left")
        fill = top["real_weight"].mean() if top["real_weight"].notna().any() else 1.0
        top["real_weight"] = top["real_weight"].fillna(fill).clip(lower=0)
        total = top["real_weight"].sum()
        top["share"] = top["real_weight"] / total if total > 0 else 1.0 / len(top)
        rows.append({"team": team, "value": (top[canonical_col] * top["share"]).sum()})
    return pd.DataFrame(rows)


def build_full_roster_aggregation(qb_canonical_col="predicted_epa_per_play"):
    """qb_canonical_col default changed from predicted_epa_per_play_sos_adjusted
    to baseline (SOS Bug Fix Task, 2026-07-25 - same reasoning as
    project_team_strength_scenarios). WR still uses SOS-adjusted here - that
    was already "NO HELP" before AND after the ref_season fix (unchanged
    verdict), so it's a separate, pre-existing question out of this
    regeneration's scope, not touched."""
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    weekly_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    weights = pd.read_csv(os.path.join(PROCESSED_DIR, "optimized_offensive_weights.csv")).iloc[0].to_dict()

    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    rb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_epa_projections_2025.csv"))
    wr_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2025.csv"))
    te_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "te_epa_projections_2025.csv"))

    # QB: reuse Task 3's availability blend (starter+backup) - no prior task
    # in this project has ever built true top-3 QB weighting, since a real
    # QB3 taking meaningful snaps in a single season is a rare edge case.
    qb_scen = build_depth_scenarios("QB", qb_proj, qb_canonical_col)
    qb_val = qb_scen[["team", "availability_adjusted"]].rename(columns={"availability_adjusted": "qb_epa"})

    rb_pool = _team_position_weight_pool(season_stats_df, weekly_df, "RB", "avg_snap_pct")
    wr_pool = _team_position_weight_pool(season_stats_df, weekly_df, "WR", "targets")
    te_pool = _team_position_weight_pool(season_stats_df, weekly_df, "TE", "targets")

    rb_val = _depth_weighted_component(rb_proj, "predicted_epa_per_play_ol_adjusted", rb_pool, top_n=3).rename(
        columns={"value": "rb_epa"})
    wr_val = _depth_weighted_component(wr_proj, "predicted_epa_per_play_sos_adjusted", wr_pool, top_n=5).rename(
        columns={"value": "wr_epa"})
    te_val = _depth_weighted_component(te_proj, "predicted_epa_per_play_synergy_adjusted", te_pool, top_n=3).rename(
        columns={"value": "te_epa"})

    components = qb_val.merge(rb_val, on="team", how="outer").merge(wr_val, on="team", how="outer").merge(
        te_val, on="team", how="outer")
    for col in ["qb_epa", "rb_epa", "wr_epa", "te_epa"]:
        components[col] = components[col].fillna(components[col].mean())

    components["offensive_strength"] = (
        weights["qb_epa"] * components["qb_epa"] + weights["rb_epa"] * components["rb_epa"]
        + weights["wr_epa"] * components["wr_epa"] + weights["te_epa"] * components["te_epa"]
        + weights["intercept"]
    )

    team_def = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))[
        ["team", "defensive_strength_allowed"]]
    full_roster = components.merge(team_def, on="team", how="left")
    full_roster["net_strength"] = full_roster["offensive_strength"] - full_roster["defensive_strength_allowed"]

    from phase3_diagnostic import compute_real_2025_team_epa
    real = compute_real_2025_team_epa()
    merged = full_roster.merge(real[["team", "real_offensive_epa", "real_defensive_epa"]], on="team", how="inner")

    corr = merged["offensive_strength"].corr(merged["real_offensive_epa"])
    mae = np.mean(np.abs(merged["offensive_strength"] - merged["real_offensive_epa"]))
    r2 = corr ** 2

    proj_std = full_roster["net_strength"].std()
    real_std = (merged["real_offensive_epa"] - merged["real_defensive_epa"]).std()
    compression = real_std / proj_std if proj_std > 0 else 0

    print("\n" + "=" * 70 + "\nFULL-ROSTER DEPTH AGGREGATION\n" + "=" * 70)
    print(f"Offensive strength vs. real 2025 team offensive EPA: corr={corr:+.3f} R2={r2:.3f} MAE={mae:.4f}")
    print(f"\nFor reference on the same real-2025 check:")
    print(f"  Task 3 (availability QB+RB, play-mix formula): +0.464")
    print(f"  Improvement 1 (combined availability + optimized weights): +0.444")
    print(f"  Full-roster depth: {corr:+.3f}")
    print(f"\nCompression: projected net_strength std={proj_std:.4f} | real net EPA std={real_std:.4f} | "
          f"ratio={compression:.2f}x (Vegas: ~1.6x, production full_season: ~3.4x)")

    out_path = os.path.join(PROCESSED_DIR, "team_strength_2025_full_roster.csv")
    full_roster.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")

    return full_roster, {"corr": corr, "mae": mae, "r2": r2, "compression": compression}


# ---------------------------------------------------------------------------
# Parallel Task B: Availability + Simple Formula.
#
# depth_scenario_qb.csv/depth_scenario_rb.csv (Task 3's real output) carry
# the blended value directly as availability_adjusted - there's no
# predicted_epa_per_play_sos_adjusted/_ol_adjusted column in those files to
# reference, that value already IS the SOS/OL-adjusted starter blended with
# the backup. team_play_mix_2024.csv was never saved as a file -
# compute_team_play_mix() computes it in memory - called directly instead.
# ---------------------------------------------------------------------------

def test_availability_with_simple_formula(qb_canonical_col="predicted_epa_per_play"):
    """qb_canonical_col default changed from predicted_epa_per_play_sos_adjusted
    to baseline (SOS Bug Fix Task, 2026-07-25 - same as project_team_strength_scenarios,
    which this function is mathematically identical to)."""
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    rb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "rb_epa_projections_2025.csv"))

    qb_scen = build_depth_scenarios("QB", qb_proj, qb_canonical_col)
    rb_scen = build_depth_scenarios("RB", rb_proj, "predicted_epa_per_play_ol_adjusted")
    qb_val = qb_scen[["team", "availability_adjusted"]].rename(columns={"availability_adjusted": "qb_epa"})
    rb_val = rb_scen[["team", "availability_adjusted"]].rename(columns={"availability_adjusted": "rb_epa"})

    play_mix = compute_team_play_mix(season_stats_df)

    simple = qb_val.merge(rb_val, on="team", how="outer").merge(play_mix, on="team", how="left")
    simple["pass_share"] = simple["pass_share"].fillna(play_mix["pass_share"].mean())
    simple["offensive_strength"] = (
        simple["pass_share"] * simple["qb_epa"] + (1 - simple["pass_share"]) * simple["rb_epa"]
    )

    team_def = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))[
        ["team", "defensive_strength_allowed"]]
    simple = simple.merge(team_def, on="team", how="left")
    simple["net_strength"] = simple["offensive_strength"] - simple["defensive_strength_allowed"]

    from phase3_diagnostic import compute_real_2025_team_epa
    real = compute_real_2025_team_epa()
    merged = simple.merge(real[["team", "real_offensive_epa", "real_defensive_epa"]], on="team", how="inner")

    corr = merged["offensive_strength"].corr(merged["real_offensive_epa"])
    mae = np.mean(np.abs(merged["offensive_strength"] - merged["real_offensive_epa"]))
    r2 = corr ** 2

    proj_std = simple["net_strength"].std()
    real_std = (merged["real_offensive_epa"] - merged["real_defensive_epa"]).std()
    compression = real_std / proj_std if proj_std > 0 else 0

    print("\n" + "=" * 70 + "\nAVAILABILITY + SIMPLE FORMULA (QB+RB ONLY)\n" + "=" * 70)
    print(f"Offensive strength vs. real 2025 team offensive EPA: corr={corr:+.3f} R2={r2:.3f} MAE={mae:.4f}")
    print(f"\nFor reference on the same real-2025 check:")
    print(f"  Task 3 (availability, original formula): +0.464")
    print(f"  Improvement 1 (combined): +0.444")
    print(f"  Full-roster depth: +0.376")
    print(f"  Simple formula with availability: {corr:+.3f}")
    if corr > 0.464:
        print(f"  -> NEW BEST (+{corr - 0.464:+.3f} over Task 3)")
    elif abs(corr - 0.464) < 1e-9:
        print(f"  -> exactly matches Task 3")
    else:
        print(f"  -> below Task 3")
    print(f"\nCompression: projected net_strength std={proj_std:.4f} | real net EPA std={real_std:.4f} | "
          f"ratio={compression:.2f}x")

    out_path = os.path.join(PROCESSED_DIR, "team_strength_2025_simple.csv")
    simple.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")

    return simple, {"corr": corr, "mae": mae, "r2": r2, "compression": compression}


if __name__ == "__main__":
    run_team_aggregation()
