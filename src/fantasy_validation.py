"""Fantasy Direction Validation: does this project's existing EPA-based
player projections predict real fantasy output?

Corrects 4 issues found in the spec before building:

1. Steps 1 & 2 ("parse PBP into fantasy points from scratch", "estimate
   usage, EPA x usage x asserted conversion factor") both reinvent things
   that already exist, real, in this codebase: data/processed/player_
   weekly_stats.csv already has real per-player-per-week fantasy_points_ppr
   for 2015-2025 (built earlier via nflreadpy), and data/processed/
   {qb,rb,wr,te}_epa_projections_2025.csv already ARE this project's real,
   leak-free, prior-season-based preseason player projections (predicted_
   epa_per_play_*, opportunities_prior_season, expected_games_2025) -
   exactly the Step-2 deliverable, built months earlier in this project.
   Reused directly instead of rebuilt.

2. Spot-checked the existing fantasy_points_ppr column against the spec's
   literal scoring formula on 3 real week-1 2025 QBs before trusting it:
   every term matched except passing TDs, which the file scores at 4 points
   (the far more common ESPN/Yahoo/Sleeper default), not the spec's 6. Per
   the user's explicit choice, uses the existing column (4pt passing TD) as-
   is rather than rebuilding a custom scorer to match the spec's less-common
   6pt convention.

3. QB uses predicted_epa_per_play_OL_ADJUSTED, not _SOS_ADJUSTED, for
   projected score - per this project's own established finding (Production
   Update task, this session) that QB's SOS adjustment HURTS accuracy after
   the ref_season bug fix. RB/WR/TE use the SOS-adjusted column (no such
   reversal was found for them).

4. projected_volume = opportunities_prior_season * (expected_games_2025/17)
   - a disclosed simplification: opportunities_prior_season is a RAW 2024
   total (not per-game-normalized), so a player who missed games in 2024
   gets a proportionally lower projected volume than a full-season player
   with the same per-game usage rate. Correcting this would need
   reconstructing 2024 games-played per player, out of scope for this
   quick-test pass.

MAE is reported from an IN-SAMPLE linear fit (projected_score -> actual
season points), fit and evaluated on the same 2025 season, since only one
season of ready-made preseason projection files exists in this project (a
genuine out-of-sample MAE would need re-running the full position-model
pipeline for prior seasons - out of scope here). Correlation - the spec's
own primary decision metric - is NOT compromised by this: projected_score
is a real, leak-free preseason quantity with no mechanical relationship to
2025's realized outcomes, so the correlation check is a genuine test.

Defense/IDP is explicitly skipped (the spec marks it optional/"if time");
none of this project's existing infrastructure covers defensive fantasy
scoring (tackles/sacks/INTs from PBP), and building it fresh doesn't fit a
quick-test budget.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FANTASY_DIR = os.path.join(PROJECT_ROOT, "data", "fantasy")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

POSITIONS = ["QB", "RB", "WR", "TE"]
QB_EPA_COL = "predicted_epa_per_play_ol_adjusted"
OTHER_EPA_COL = "predicted_epa_per_play_sos_adjusted"

# Spec's stated interpretation thresholds (Step 4 / Success Criteria)
GREEN_CORR, GREEN_TOP10 = 0.40, 0.50
RED_CORR, RED_TOP10 = 0.20, 0.30


def extract_actual_fantasy_points_2025(league_scoring="ppr"):
    """Real per-player-per-week actual fantasy points, 2025 REG season only
    - a filter/rename of the already-computed player_weekly_stats.csv, not a
    rebuild from PBP (see module docstring #1)."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    reg = pws[(pws["season"] == 2025) & (pws["season_type"] == "REG")].copy()

    if league_scoring == "ppr":
        reg["actual_fantasy_pts"] = reg["fantasy_points_ppr"]
    elif league_scoring == "half_ppr":
        reg["actual_fantasy_pts"] = reg["fantasy_points"] + 0.5 * reg["receptions"].fillna(0)
    elif league_scoring == "non_ppr":
        reg["actual_fantasy_pts"] = reg["fantasy_points"]
    else:
        raise ValueError(f"Unknown league_scoring: {league_scoring}")

    out = reg[["player_id", "player_display_name", "position", "week", "recent_team", "actual_fantasy_pts"]].rename(
        columns={"player_display_name": "player_name", "recent_team": "team"})

    os.makedirs(FANTASY_DIR, exist_ok=True)
    out.to_csv(os.path.join(FANTASY_DIR, "actual_fantasy_points_2025_by_week.csv"), index=False, encoding="utf-8")
    return out


def get_actual_season_totals_2025(league_scoring="ppr"):
    weekly = extract_actual_fantasy_points_2025(league_scoring)
    return weekly.groupby(["player_id", "player_name", "position"])["actual_fantasy_pts"].sum().reset_index().rename(
        columns={"actual_fantasy_pts": "actual_season_fantasy_pts"})


def project_fantasy_points_from_epa(position):
    """Real, leak-free preseason projected score for `position`, using this
    project's existing {position}_epa_projections_2025.csv (see module
    docstring #1, #3, #4). projected_score isn't literally calibrated to
    fantasy-point units yet - that calibration happens in validate_fantasy_
    correlation_by_position() via an in-sample linear fit (docstring note on
    MAE)."""
    path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2025.csv")
    df = pd.read_csv(path)
    epa_col = QB_EPA_COL if position.upper() == "QB" else OTHER_EPA_COL

    df = df.copy()
    df["projected_volume"] = df["opportunities_prior_season"] * (df["expected_games_2025"] / 17.0)
    df["projected_score"] = df[epa_col] * df["projected_volume"]

    out_path = os.path.join(FANTASY_DIR, f"projected_fantasy_score_2025_{position.lower()}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df[["player_id", "player_name", "team", "projected_score", "projected_volume"]]


def validate_fantasy_correlation_by_position(season=2025, league_scoring="ppr"):
    """Per spec Step 3: corr, MAE (in-sample-calibrated, see module
    docstring), top-10 pick accuracy, sample size, by position."""
    actual = get_actual_season_totals_2025(league_scoring)

    results = []
    merged_by_pos = {}
    for position in POSITIONS:
        proj = project_fantasy_points_from_epa(position)
        act_pos = actual[actual["position"] == position]
        merged = proj.merge(act_pos[["player_id", "actual_season_fantasy_pts"]], on="player_id", how="inner")
        merged = merged[merged["projected_volume"] > 0].reset_index(drop=True)

        if len(merged) < 5:
            print(f"[{position}] too few matched players ({len(merged)}) - skipped")
            continue

        corr = merged["projected_score"].corr(merged["actual_season_fantasy_pts"])
        slope, intercept = np.polyfit(merged["projected_score"], merged["actual_season_fantasy_pts"], 1)
        merged["predicted_pts"] = slope * merged["projected_score"] + intercept
        mae = float(np.mean(np.abs(merged["predicted_pts"] - merged["actual_season_fantasy_pts"])))

        top10_proj = set(merged.nlargest(10, "projected_score")["player_id"])
        top10_actual = set(merged.nlargest(10, "actual_season_fantasy_pts")["player_id"])
        top10_pct = len(top10_proj & top10_actual) / 10.0

        results.append({"position": position, "n": len(merged), "corr": corr, "mae": mae, "top10_pct": top10_pct})
        merged_by_pos[position] = merged

    results_df = pd.DataFrame(results)
    return results_df, merged_by_pos


def validate_week_by_week_trends(merged_by_pos, checkpoint_weeks=(4, 8, 12, 16, 18)):
    """Correlation of the STATIC preseason projected_score against
    cumulative real actual points through week N - does a fixed preseason
    signal track the season better as more real games accumulate? (Same
    style of check as dynamic_tracking.py, applied at the player level.)"""
    weekly = extract_actual_fantasy_points_2025()
    rows = []
    for position, merged in merged_by_pos.items():
        for wk in checkpoint_weeks:
            cum = weekly[(weekly["week"] <= wk) & (weekly["position"] == position)].groupby(
                "player_id")["actual_fantasy_pts"].sum().reset_index().rename(columns={"actual_fantasy_pts": "cum_actual"})
            m = merged[["player_id", "projected_score"]].merge(cum, on="player_id", how="inner")
            if len(m) < 5:
                continue
            rows.append({"position": position, "week": wk, "corr": m["projected_score"].corr(m["cum_actual"]), "n": len(m)})
    return pd.DataFrame(rows)


def identify_fantasy_breakouts_and_busts(merged, position, threshold_stddev=1.5):
    """Players whose real actual points diverge from the in-sample-
    calibrated prediction by > threshold_stddev residual std devs."""
    residual = merged["actual_season_fantasy_pts"] - merged["predicted_pts"]
    std = residual.std()
    flagged = merged.copy()
    flagged["residual"] = residual
    flagged["flag"] = np.where(residual > threshold_stddev * std, "BREAKOUT",
                                np.where(residual < -threshold_stddev * std, "BUST", "-"))
    return flagged[flagged["flag"] != "-"].sort_values("residual", ascending=False)


def generate_fantasy_validation_report(season=2025, league_scoring="ppr"):
    results_df, merged_by_pos = validate_fantasy_correlation_by_position(season, league_scoring)

    print(f"\n{'=' * 70}\nFANTASY DIRECTION VALIDATION (real {season}, {league_scoring})\n{'=' * 70}")
    print(results_df.to_string(index=False))

    trend_df = validate_week_by_week_trends(merged_by_pos)
    print(f"\nWeek-by-week trend (static preseason projection vs. cumulative actual):")
    print(trend_df.to_string(index=False))

    print(f"\nBreakouts/busts (|residual| > 1.5 std):")
    all_flags = []
    for position, merged in merged_by_pos.items():
        flagged = identify_fantasy_breakouts_and_busts(merged, position)
        all_flags.append(flagged.assign(position=position))
        if len(flagged):
            print(f"  {position}: {(flagged['flag'] == 'BREAKOUT').sum()} breakouts, "
                  f"{(flagged['flag'] == 'BUST').sum()} busts")
            for _, row in flagged.head(3).iterrows():
                print(f"    {row['flag']}: {row['player_name']} projected={row['predicted_pts']:.1f} "
                      f"actual={row['actual_season_fantasy_pts']:.1f} ({row['residual']:+.1f})")
    flags_df = pd.concat(all_flags, ignore_index=True) if all_flags else pd.DataFrame()

    n_strong = int((results_df["corr"] > GREEN_CORR).sum())
    n_top10_strong = int((results_df["top10_pct"] > GREEN_TOP10).sum())
    n_weak = int((results_df["corr"] < RED_CORR).sum())
    n_very_weak = int((results_df["corr"] < 0.15).sum())

    viable = (n_strong >= 2) and (n_top10_strong >= 1) and (n_very_weak == 0)
    shelve = (n_weak >= 3) or ((results_df["top10_pct"] < 0.40).all())

    lines = []
    lines.append("=" * 70)
    lines.append(f"FANTASY DIRECTION VALIDATION REPORT (real {season}, {league_scoring})")
    lines.append("=" * 70)
    lines.append(results_df.to_string(index=False))
    lines.append("")
    lines.append(f"Positions with corr > {GREEN_CORR}: {n_strong}/4 | "
                 f"Positions with top-10 accuracy > {GREEN_TOP10:.0%}: {n_top10_strong}/4 | "
                 f"Positions with corr < 0.15 (near-noise): {n_very_weak}/4")
    if viable:
        recommendation = "PURSUE - fantasy direction is viable per the spec's own success criteria."
    elif shelve:
        recommendation = "SHELVE - fantasy direction does not meet the spec's viability bar."
    else:
        recommendation = "MIXED - meets neither the pursue nor shelve bar cleanly; see per-position detail before deciding."
    lines.append(f"\nRECOMMENDATION: {recommendation}")
    report_text = "\n".join(lines)
    print(f"\n{report_text}")

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(DIAGNOSTIC_DIR, "fantasy_validation_2025.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "fantasy_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved data/diagnostic/fantasy_validation_2025.csv, fantasy_validation_report.txt")

    return {"results": results_df, "trend": trend_df, "flags": flags_df, "recommendation": recommendation}


if __name__ == "__main__":
    generate_fantasy_validation_report()
