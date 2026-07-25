"""Phase 3 Diagnostic 1: Individual Player Projection Accuracy.

Fixes made before running this (see the completion report for the full
reasoning):

1. No defensive projection file (EDGE/DL/CB/S/LB) had a `player_id` column -
   only `player_name`. The spec's `merge(..., on='player_id')` would have
   silently failed for every defensive position. Fixed at the source:
   added `player_id` to PassRushWARModel/DefensePositionModel/
   BlendedDefenseModel's predict_next_season() outputs (same minimal,
   additive fix already made to OffenseEpaModel in Phase 4 Task 4.2), and
   regenerated all five leak-free 2025 projection files.

2. The spec assumed every defensive position has a "_blended_projections_
   2026.csv" file. EDGE (WAR-based) and DL (sacks-based) were never
   "blended," and using the *2026* files at all would reintroduce the exact
   temporal mismatch Task 3.1 was built to avoid. This uses the leak-free
   *2025*-target files from Task 3.1 (`{pos}_leakfree_predictions_2025.csv`)
   throughout, matching the season being validated against.

3. Ground truth for EDGE/DL/CB/S/LB isn't a fresh PBP computation - it's
   already real, already-computed data: EDGE/DL's real 2025 `war` sits in
   pass_rush_war_2015_2025.csv; CB/S/LB's real 2025 blended_score is
   reconstructed via build_blended_score_table() (same function used to
   train the models), filtered to season 2025 and standardized on the same
   basis the models were trained on.

4. TE is confirmed completely missing, not just "using QB implicitly" as
   the spec speculated - grepped player_models.py: no TEModel class, no
   TE wrapper function, anywhere. utilities.PRIMARY_METRIC does map TE to
   receiving_yards (so an age curve exists), but no position model was ever
   trained for it in Phase 2. Reported as N/A, not estimated.
"""

import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

TARGET_SEASON = 2025

# Which prediction column(s) exist per offense position, worth checking all
# of - Task 4.2/5.1 already validated which one is "best" per position, but
# this diagnostic re-checks all of them against real 2025 data directly.
OFFENSE_PRED_COLS = {
    "QB": ["predicted_epa_per_play", "predicted_epa_per_play_ol_adjusted", "predicted_epa_per_play_sos_adjusted"],
    "WR": ["predicted_epa_per_play", "predicted_epa_per_play_sos_adjusted"],
    "RB": ["predicted_epa_per_play", "predicted_epa_per_play_ol_adjusted", "predicted_epa_per_play_sos_adjusted"],
}
DEFENSE_PRED_COL = {"EDGE": "predicted_war", "DL": "predicted_sacks",
                     "CB": "predicted_blended_score", "S": "predicted_blended_score", "LB": "predicted_blended_score"}
BLEND_RATIO_BY_POSITION = {"CB": (0.8, 0.2), "S": (0.5, 0.5), "LB": (0.8, 0.2)}


def load_real_2025_offense_epa():
    from ol_quality import load_real_2025_pbp, compute_real_epa_per_play
    pbp_2025 = load_real_2025_pbp()
    return {pos: compute_real_epa_per_play(pos, pbp_2025) for pos in ["QB", "WR", "RB"]}


def load_real_2025_defense_ground_truth():
    """Real (not projected) 2025 values for EDGE/DL/CB/S/LB - all already
    computed elsewhere in this project from real PBP/PFR data."""
    pass_rush_war = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    real_2025_pass_rush = pass_rush_war[pass_rush_war["season"] == TARGET_SEASON]

    truth = {
        "EDGE": real_2025_pass_rush[real_2025_pass_rush["position"] == "EDGE"][["player_id", "war"]].rename(columns={"war": "real_value"}),
        "DL": real_2025_pass_rush[real_2025_pass_rush["position"] == "DL"][["player_id", "sacks_from_pbp"]].rename(columns={"sacks_from_pbp": "real_value"}),
    }

    from player_models import build_blended_score_table
    tackle_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tackle_efficiency_2018_2025.csv"))
    leverage_df = pd.read_csv(os.path.join(PROCESSED_DIR, "leverage_war_2016_2025.csv"))
    for position in ["CB", "S", "LB"]:
        tw, lw = BLEND_RATIO_BY_POSITION[position]
        scored = build_blended_score_table(tackle_df, leverage_df, position, tw, lw)
        real_2025 = scored[scored["season"] == TARGET_SEASON][["player_id", "blended_score"]].rename(
            columns={"blended_score": "real_value"})
        truth[position] = real_2025

    return truth


def _corr_r2_mae(pred, actual):
    valid = (~pred.isna()) & (~actual.isna())
    pred, actual = pred[valid], actual[valid]
    if len(pred) < 5:
        return None
    corr = pred.corr(actual)
    mae = (pred - actual).abs().mean()
    return {"n": len(pred), "correlation": corr, "r2": corr ** 2, "mae": mae,
            "projected_mean": pred.mean(), "real_mean": actual.mean(),
            "projected_std": pred.std(), "real_std": actual.std()}


def validate_individual_player_projections():
    rows = []
    real_offense = load_real_2025_offense_epa()
    real_defense = load_real_2025_defense_ground_truth()

    print("=" * 70 + "\nOFFENSE (real 2025 EPA/play, all candidate prediction columns)\n" + "=" * 70)
    for position, pred_cols in OFFENSE_PRED_COLS.items():
        proj_path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2025.csv")
        proj = pd.read_csv(proj_path)
        real = real_offense[position]
        merged = proj.merge(real[["player_id", "real_2025_epa_per_play"]], on="player_id", how="inner")

        for pred_col in pred_cols:
            if pred_col not in merged.columns:
                continue
            metrics = _corr_r2_mae(merged[pred_col], merged["real_2025_epa_per_play"])
            if metrics is None:
                continue
            print(f"\n{position} [{pred_col}]: n={metrics['n']} corr={metrics['correlation']:+.3f} "
                  f"R2={metrics['r2']:.3f} MAE={metrics['mae']:.4f}")
            print(f"  projected mean={metrics['projected_mean']:+.4f} std={metrics['projected_std']:.4f} | "
                  f"real mean={metrics['real_mean']:+.4f} std={metrics['real_std']:.4f}")
            rows.append({"position": position, "pred_col": pred_col, **metrics})

    print("\n" + "=" * 70 + "\nTE\n" + "=" * 70)
    print("NO MODEL BUILT - confirmed via grep of player_models.py, no TEModel class or wrapper exists. "
          "utilities.PRIMARY_METRIC maps TE->receiving_yards (an age curve exists) but Phase 2 never trained "
          "a position model for it. Reported as a real gap, not estimated.")
    rows.append({"position": "TE", "pred_col": None, "n": 0, "correlation": None, "r2": None, "mae": None,
                 "projected_mean": None, "real_mean": None, "projected_std": None, "real_std": None})

    print("\n" + "=" * 70 + "\nDEFENSE (real 2025 war/sacks/blended_score, leak-free 2025-target projections)\n" + "=" * 70)
    for position, pred_col in DEFENSE_PRED_COL.items():
        proj_path = os.path.join(PROCESSED_DIR, f"{position.lower()}_leakfree_predictions_{TARGET_SEASON}.csv")
        proj = pd.read_csv(proj_path)
        real = real_defense[position]
        merged = proj.merge(real, on="player_id", how="inner")
        metrics = _corr_r2_mae(merged[pred_col], merged["real_value"])
        if metrics is None:
            print(f"\n{position}: insufficient matched players")
            continue
        print(f"\n{position} [{pred_col}]: n={metrics['n']} corr={metrics['correlation']:+.3f} "
              f"R2={metrics['r2']:.3f} MAE={metrics['mae']:.4f}")
        print(f"  projected mean={metrics['projected_mean']:+.4f} std={metrics['projected_std']:.4f} | "
              f"real mean={metrics['real_mean']:+.4f} std={metrics['real_std']:.4f}")
        rows.append({"position": position, "pred_col": pred_col, **metrics})

    accuracy_df = pd.DataFrame(rows)
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    out_path = os.path.join(DIAGNOSTIC_DIR, "individual_position_accuracy_2025.csv")
    accuracy_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    return accuracy_df


def identify_unused_assets(accuracy_df):
    print("\n\n" + "=" * 70 + "\nUNUSED ASSETS INVENTORY\n" + "=" * 70)

    wr_rows = accuracy_df[accuracy_df["position"] == "WR"]
    best_wr = wr_rows.loc[wr_rows["correlation"].abs().idxmax()] if len(wr_rows.dropna(subset=["correlation"])) else None
    print(f"\n1. WR projections (not a direct input to team offensive strength - Task 3.1 used QB+RB only):")
    if best_wr is not None:
        print(f"   Best individual WR column ({best_wr['pred_col']}): corr={best_wr['correlation']:+.3f}, "
              f"R2={best_wr['r2']:.3f}, n={best_wr['n']:.0f}")
        print(f"   {'Real signal - worth testing as a team-strength input' if abs(best_wr['correlation']) > 0.15 else 'Weak/noisy on its own'}")

    print(f"\n2. TE projections: NOT BUILT (see above) - a real, structural gap, not a deliberate exclusion.")

    qb_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "qb_epa_projections_2025.csv"))
    print(f"\n3. Availability factor (exists on every offense projection file, not used in Task 3.1's aggregation):")
    print(f"   QB availability_factor range: {qb_proj['availability_factor'].min():.2f} to {qb_proj['availability_factor'].max():.2f} "
          f"(mean {qb_proj['availability_factor'].mean():.2f})")
    print(f"   Task 3.1 picked a single starting QB/leading RB by raw prior-season opportunities, with no "
          f"injury-risk discount and no backup blended in.")

    print(f"\n4. Snap share / opportunity share: Task 3.1 used a single starter per position (100% implicit share) "
          f"rather than a depth-weighted average - real committees (RB) and rotational packages aren't reflected.")

    sos_summary = accuracy_df[accuracy_df["position"].isin(["QB", "WR", "RB"])]
    print(f"\n5. Schedule strength (SOS) - already tested in Task 5.1: QB SOS-adjusted validated (real 2025 "
          f"improvement), WR SOS-adjusted was flat/no help, RB SOS-adjusted actively hurt real accuracy. "
          f"Task 3.1 used QB=SOS-adjusted correctly; RB/WR correctly did NOT use it.")

    print(f"\n6. OL adjustment - already tested in Task 4.2: real improvement for RB only (QB/WR were noise-level). "
          f"Task 3.1 used RB=OL-adjusted correctly; QB/WR correctly did NOT use it.")

    print("=" * 70 + "\n")


def run_diagnostic_1():
    accuracy_df = validate_individual_player_projections()
    identify_unused_assets(accuracy_df)
    return accuracy_df



# ---------------------------------------------------------------------------
# Diagnostic 2: Aggregation-Level Signal Loss
# ---------------------------------------------------------------------------

def compute_real_2025_team_epa():
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    off = compute_team_offense_epa()
    defn = compute_team_defense_epa()
    real = off.merge(defn, on=["team", "season"])
    real = real[real["season"] == TARGET_SEASON].copy()
    real["real_net_epa"] = real["off_epa"] - real["def_epa_allowed"]
    return real.rename(columns={"off_epa": "real_offensive_epa", "def_epa_allowed": "real_defensive_epa"})


def trace_aggregation_signal_loss(real=None):
    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))
    real = compute_real_2025_team_epa() if real is None else real
    merged = team_strength.merge(real, on="team")

    off_corr = merged["offensive_strength"].corr(merged["real_offensive_epa"])
    def_corr = merged["defensive_strength_allowed"].corr(merged["real_defensive_epa"])
    net_corr = merged["net_strength"].corr(merged["real_net_epa"])

    print("\n" + "=" * 70 + "\nLAYER 1: AGGREGATION QUALITY (Phase 3's team_strength_2025.csv vs. real 2025)\n" + "=" * 70)
    print(f"Offensive strength -> real offensive EPA: corr = {off_corr:+.3f}")
    print(f"Defensive strength allowed -> real defensive EPA: corr = {def_corr:+.3f}")
    print(f"Net strength -> real net EPA: corr = {net_corr:+.3f}")
    print(f"  {'RED - weak' if off_corr < 0.40 else 'reasonable'} offense | {'RED - weak' if def_corr < 0.40 else 'reasonable'} defense")

    print(f"\nLAYER 2: EPA -> WINS CONVERSION (Task 4.1, already validated)")
    print(f"  Backtest R2 = 0.747 (corr ~= 0.864) on REAL historical epa_diff - this layer works well on its own.")

    implied_win_corr = net_corr * 0.864
    print(f"\nCOMPOUND EFFECT:")
    print(f"  net_strength corr with real epa ({net_corr:+.3f}) x conversion corr (~0.864) "
          f"implies a compound win correlation of ~{implied_win_corr:+.3f}")
    print(f"  Actual Phase 4 result: 0.224")
    print(f"  {'Diagnosis: Layer 1 (aggregation) is the bottleneck, not the EPA->wins conversion' if net_corr < 0.40 else 'Diagnosis: loss is happening somewhere else'}")
    return {"off_corr": off_corr, "def_corr": def_corr, "net_corr": net_corr, "implied_win_corr": implied_win_corr}


def measure_component_contributions(real=None):
    """Diagnostic 1 found EDGE (corr+0.60) and LB (+0.56) are the strongest
    individual player signals in the entire pipeline - stronger than the
    top-down team_defense_epa model Task 3.1 actually used as the primary
    defensive_strength_allowed signal (+0.33). This checks whether that
    individual-level strength survives being aggregated to the team level,
    and whether a bottom-up composite built from it would have beaten the
    top-down model Task 3.1 chose."""
    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))
    real = compute_real_2025_team_epa() if real is None else real
    merged = team_strength.merge(real, on="team")
    merged["real_defensive_quality"] = -merged["real_defensive_epa"]  # flip so higher = better, matching offensive_strength's orientation

    print("\n" + "=" * 70 + "\nDEFENSIVE COMPONENT CONTRIBUTIONS (each vs. real 2025 defensive quality)\n" + "=" * 70)
    components = {
        "team_pass_rush_war (EDGE WAR + DL sacks->war, team sum)": "team_pass_rush_war",
        "cb_avg_blended_score (team avg)": "cb_avg_blended_score",
        "s_avg_blended_score (team avg)": "s_avg_blended_score",
        "lb_avg_blended_score (team avg)": "lb_avg_blended_score",
        "defensive_strength_allowed (CURRENT PRIMARY - top-down Ridge model)": "defensive_strength_allowed",
    }
    corrs = {}
    for label, col in components.items():
        sign = -1 if col == "defensive_strength_allowed" else 1  # defensive_strength_allowed: lower=better, flip for consistent orientation
        corr = (sign * merged[col]).corr(merged["real_defensive_quality"])
        corrs[col] = corr
        print(f"  {label}: corr = {corr:+.3f}")

    # Bottom-up composite: standardize each real defensive component and average -
    # same standardization-not-invented-conversion discipline used throughout this project.
    z_cols = []
    for col in ["team_pass_rush_war", "cb_avg_blended_score", "s_avg_blended_score", "lb_avg_blended_score"]:
        z_col = f"z_{col}"
        merged[z_col] = (merged[col] - merged[col].mean()) / merged[col].std()
        z_cols.append(z_col)
    merged["bottom_up_composite"] = merged[z_cols].mean(axis=1)
    composite_corr = merged["bottom_up_composite"].corr(merged["real_defensive_quality"])

    # EDGE+LB only (the two individually-strongest signals from Diagnostic 1)
    merged["edge_lb_composite"] = merged[["z_team_pass_rush_war", "z_lb_avg_blended_score"]].mean(axis=1)
    edge_lb_corr = merged["edge_lb_composite"].corr(merged["real_defensive_quality"])

    print(f"\nBOTTOM-UP composite (all 4 components, standardized average): corr = {composite_corr:+.3f}")
    print(f"BOTTOM-UP composite (EDGE+LB only - the 2 strongest individual signals): corr = {edge_lb_corr:+.3f}")
    print(f"CURRENT top-down model (what Task 3.1 actually used): corr = {corrs['defensive_strength_allowed']:+.3f}")
    winner = max([("bottom-up (all 4)", composite_corr), ("bottom-up (EDGE+LB)", edge_lb_corr),
                  ("top-down model", corrs["defensive_strength_allowed"])], key=lambda t: t[1])
    print(f"WINNER: {winner[0]} (corr={winner[1]:+.3f})")
    print("=" * 70 + "\n")
    return {"component_corrs": corrs, "bottom_up_all4": composite_corr, "bottom_up_edge_lb": edge_lb_corr}


def test_structural_alternatives(real=None):
    """Tests concrete alternatives to Task 3.1's offensive aggregation
    formula against real 2025 data - not speculation, each one computed."""
    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))
    real = compute_real_2025_team_epa() if real is None else real
    merged = team_strength.merge(real, on="team")
    current_corr_off = merged["offensive_strength"].corr(merged["real_offensive_epa"])

    print("\n" + "=" * 70 + "\nOFFENSIVE STRUCTURAL ALTERNATIVES (vs. real 2025 offensive EPA)\n" + "=" * 70)
    print(f"CURRENT (play-mix-weighted QB + RB): corr = {current_corr_off:+.3f}")

    wr_proj = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2025.csv"))
    wr_by_team = wr_proj.groupby("team")["predicted_epa_per_play"].mean().rename("wr_team_avg")
    merged = merged.merge(wr_by_team, on="team", how="left")

    # Alternative 1: simple 3-way average (QB + RB + WR)
    alt1 = (merged["qb_epa_per_play"] + merged["rb_epa_per_play"] + merged["wr_team_avg"]) / 3
    alt1_corr = alt1.corr(merged["real_offensive_epa"])
    print(f"ALT 1 (simple QB+RB+WR average, no play-mix weight): corr = {alt1_corr:+.3f} "
          f"({alt1_corr - current_corr_off:+.3f} vs. current)")

    # Alternative 2: play-mix-weighted QB+RB, but blend in WR as a 3rd weighted component
    # (WR gets a fixed 30% weight on the passing share, splitting passing credit between QB and WR)
    alt2 = merged["pass_share"] * (0.7 * merged["qb_epa_per_play"] + 0.3 * merged["wr_team_avg"]) + \
        (1 - merged["pass_share"]) * merged["rb_epa_per_play"]
    alt2_corr = alt2.corr(merged["real_offensive_epa"])
    print(f"ALT 2 (play-mix QB/RB, with WR blended 30% into the passing share): corr = {alt2_corr:+.3f} "
          f"({alt2_corr - current_corr_off:+.3f} vs. current)")

    # Alternative 3: QB alone (does RB actually help, or is it diluting the QB signal?)
    alt3_corr = merged["qb_epa_per_play"].corr(merged["real_offensive_epa"])
    print(f"ALT 3 (QB alone, no RB blend at all): corr = {alt3_corr:+.3f} ({alt3_corr - current_corr_off:+.3f} vs. current)")

    alternatives = {"current": current_corr_off, "alt1_simple_avg": alt1_corr,
                     "alt2_wr_blended_in": alt2_corr, "alt3_qb_only": alt3_corr}
    best = max(alternatives.items(), key=lambda t: t[1])
    print(f"\nWINNER: {best[0]} (corr={best[1]:+.3f})")
    print("=" * 70 + "\n")
    return alternatives


def run_diagnostic_2():
    real = compute_real_2025_team_epa()  # scanned once, shared across all three checks below
    layer_trace = trace_aggregation_signal_loss(real=real)
    component_analysis = measure_component_contributions(real=real)
    structural_alternatives = test_structural_alternatives(real=real)
    return layer_trace, component_analysis, structural_alternatives


if __name__ == "__main__":
    run_diagnostic_1()
    run_diagnostic_2()
