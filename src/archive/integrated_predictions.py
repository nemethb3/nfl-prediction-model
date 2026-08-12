"""Integration & Baseline Validation: wires the real, validated findings
from Phases 1-4 into three "final" production pipelines, and backtests
each against real 2025 to establish new baseline numbers.

Adoption decisions (see PROGRESS.md for the full evidence behind each):

ADOPTED:
- Game spreads: Vegas primary (Component C - beats Elo at every real
  checkpoint tested, no exception) + the real matchup-EPA adjustment
  (Component 2.3 - +0.010 corr / -0.09 MAE, real fitted coefficient).
  Elo (+ the real, marginal QB-Elo blend, Component 2.2) as fallback only
  for games with no real posted Vegas line.
- Season win projections: switched from the old EPA/weekly_update_pipeline
  mechanism (Dynamic Tracking) to real carryover Elo (Component A beats
  EPA for season wins too) + real actual-wins-so-far, reusing Component
  B's already-built "freeze Elo at week N, project remaining games"
  mechanism instead of re-deriving one.
- Fantasy: RB and QB/TE switch to volume-only (Components 1.2/3.3 -
  decisive real wins for both). WR is UNCHANGED - it's the one position
  where the combined EPA+volume formula (+0.591) already beats volume-
  alone (+0.554), so switching would be a real regression, not an
  improvement.

NOT ADOPTED (tested, real, and rejected - not simply omitted):
- Injury adjustments (Component 2.1 - real null result, marginally worse
  in backtest)
- Rest adjustments (Component 3.2 - real null result, zero measurable
  effect on corr/MAE)
- Edge detection as a betting signal (Component 4.1 - real -36% ROI,
  decisively negative)
- K=40 momentum weighting (Component 3.1 - a genuine, never-resolved
  corr-up/MAE-down trade-off) - K=10 (Component 1's original) kept as the
  default rather than silently picking a side of an open trade-off.
"""

import os

import numpy as np
import pandas as pd

from constants import MATCHUP_FITTED_COEFFICIENT

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")


# ---------------------------------------------------------------------------
# Game-level spreads: Vegas primary + matchup adjustment, Elo fallback
# ---------------------------------------------------------------------------

def generate_integrated_game_predictions(season, fitted_model=None):
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads
    from vegas_integration_optimized import extract_vegas_lines
    from matchup_features import (extract_player_epa_by_position, build_defense_epa_by_position,
                                    extract_game_matchups, calculate_matchup_epa_edges,
                                    apply_matchup_adjustment_to_game_spread)

    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()
    elo_preds = generate_elo_game_spreads(season, fitted_model).rename(columns={"predicted_spread": "elo_spread"})
    vegas = extract_vegas_lines(season)[["game_id", "vegas_spread"]]

    merged = elo_preds.merge(vegas, on="game_id", how="left")
    merged["has_vegas_line"] = merged["vegas_spread"].notna()
    merged["base_spread"] = np.where(merged["has_vegas_line"], merged["vegas_spread"], merged["elo_spread"])
    merged["base_source"] = np.where(merged["has_vegas_line"], "vegas", "elo_fallback")

    off_epa = extract_player_epa_by_position(season)
    def_epa = build_defense_epa_by_position(season)
    net_rows = []
    for week in sorted(merged["week"].unique()):
        matchups = extract_game_matchups(season, week, off_epa)
        if len(matchups) == 0:
            continue
        edges = calculate_matchup_epa_edges(matchups, off_epa, def_epa)
        net_rows.append(edges.groupby(["game_id", "off_team"])["edge_epa"].sum().reset_index())
    net = pd.concat(net_rows, ignore_index=True) if net_rows else pd.DataFrame(columns=["game_id", "off_team", "edge_epa"])

    merged = merged.merge(net.rename(columns={"off_team": "home_team", "edge_epa": "home_net_edge"}),
                           on=["game_id", "home_team"], how="left")
    merged = merged.merge(net.rename(columns={"off_team": "away_team", "edge_epa": "away_net_edge"}),
                           on=["game_id", "away_team"], how="left")
    merged[["home_net_edge", "away_net_edge"]] = merged[["home_net_edge", "away_net_edge"]].fillna(0.0)
    merged["net_edge_diff"] = merged["home_net_edge"] - merged["away_net_edge"]
    merged["final_spread"] = apply_matchup_adjustment_to_game_spread(
        merged["base_spread"], merged["net_edge_diff"], MATCHUP_FITTED_COEFFICIENT)

    return merged[["game_id", "season", "week", "home_team", "away_team", "base_source", "base_spread",
                   "net_edge_diff", "final_spread"]]


# ---------------------------------------------------------------------------
# Season win projections: real actual wins + Elo-based remaining games
# ---------------------------------------------------------------------------

def generate_integrated_season_projections(season, through_week):
    from weekly_recalibration import update_elo_with_actual_results, regenerate_spreads_for_remaining_games
    from elo_game_prediction import fit_probability_to_spread_conversion, calculate_win_probability_from_elo, _load_game_results

    fitted_model = fit_probability_to_spread_conversion()
    updated_ratings = update_elo_with_actual_results(season, through_week)
    remaining = regenerate_spreads_for_remaining_games(season, through_week, updated_ratings, fitted_model)

    played = _load_game_results([season])
    played = played[played["week"] <= through_week].copy()
    played["home_win_val"] = np.select([played["point_diff"] > 0, played["point_diff"] < 0], [1.0, 0.0], default=0.5)
    played["away_win_val"] = 1.0 - played["home_win_val"]
    actual_wins = played.groupby("home_team")["home_win_val"].sum().add(
        played.groupby("away_team")["away_win_val"].sum(), fill_value=0.0)

    remaining = remaining.copy()
    remaining["home_win_prob"] = calculate_win_probability_from_elo(remaining["home_elo"], remaining["away_elo"])
    remaining["away_win_prob"] = 1.0 - remaining["home_win_prob"]
    home = remaining[["home_team", "home_win_prob"]].rename(columns={"home_team": "team", "home_win_prob": "win_prob"})
    away = remaining[["away_team", "away_win_prob"]].rename(columns={"away_team": "team", "away_win_prob": "win_prob"})
    remaining_win_probs = pd.concat([home, away], ignore_index=True).groupby("team")["win_prob"].agg(
        predicted_wins="sum", n_remaining="count")

    all_teams = sorted(set(actual_wins.index) | set(remaining_win_probs.index))
    out = pd.DataFrame({"team": all_teams})
    out["actual_wins"] = out["team"].map(actual_wins).fillna(0.0)
    out["predicted_wins_remaining"] = out["team"].map(remaining_win_probs["predicted_wins"]).fillna(0.0)
    out["total_projection"] = out["actual_wins"] + out["predicted_wins_remaining"]
    return out.sort_values("total_projection", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fantasy: RB/QB/TE volume-only, WR unchanged
# ---------------------------------------------------------------------------

def generate_integrated_fantasy_projections():
    from fantasy_rb_formula import project_rb_fantasy_points, load_rb_actual_fantasy_2025
    from fantasy_formula_improvements import _build_common, _real_ppr
    from fantasy_validation import project_fantasy_points_from_epa, get_actual_season_totals_2025

    out = {}

    rb_proj = project_rb_fantasy_points()
    rb_actual = load_rb_actual_fantasy_2025()
    rb_merged = rb_proj.merge(rb_actual, on=["player_id", "week"], how="inner")
    out["RB"] = {"corr": rb_merged["projected_fantasy_pts"].corr(rb_merged["actual_fantasy_pts"]), "n": len(rb_merged)}

    for position in ["QB", "TE"]:
        _, trailing_vol, _, _, actual = _build_common(position)
        trailing_vol = trailing_vol.copy()
        trailing_vol["projected_fantasy_pts"] = _real_ppr(position, trailing_vol)
        merged = trailing_vol.merge(actual, on=["player_id", "week"], how="inner")
        out[position] = {"corr": merged["projected_fantasy_pts"].corr(merged["actual_fantasy_pts"]), "n": len(merged)}

    wr_proj = project_fantasy_points_from_epa("WR")
    wr_actual = get_actual_season_totals_2025()
    wr_actual = wr_actual[wr_actual["position"] == "WR"]
    wr_merged = wr_proj.merge(wr_actual[["player_id", "actual_season_fantasy_pts"]], on="player_id", how="inner")
    out["WR"] = {"corr": wr_merged["projected_score"].corr(wr_merged["actual_season_fantasy_pts"]), "n": len(wr_merged)}

    return out


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

def run_full_2025_backtest():
    from elo_game_prediction import _load_game_results

    print(f"\n{'=' * 70}\nFULL INTEGRATED 2025 BACKTEST\n{'=' * 70}")

    # 1. Game spreads
    game_preds = generate_integrated_game_predictions(2025)
    actual = _load_game_results([2025])[["game_id", "point_diff"]]
    gm = game_preds.merge(actual, on="game_id", how="inner")
    game_corr = gm["final_spread"].corr(gm["point_diff"])
    game_mae = float(np.mean(np.abs(gm["final_spread"] - gm["point_diff"])))
    n_vegas = int((gm["base_source"] == "vegas").sum())
    print(f"\n1. GAME SPREADS (n={len(gm)}, {n_vegas} using real Vegas base): "
          f"corr={game_corr:+.3f} MAE={game_mae:.2f}")
    print(f"   vs. old baselines: Elo alone corr=+0.385/MAE=10.36 | Vegas alone corr=+0.504/MAE=9.72 "
          f"| matchup-adjusted-only corr=+0.399/MAE=10.49")

    # 2. Season win projections (checkpoint weeks, matching established convention)
    print(f"\n2. SEASON WIN PROJECTIONS (real actual wins-so-far + Elo-projected remaining):")
    actual_final = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "backtest", "actual_wins_2025.csv"))
    season_rows = []
    for week in (1, 4, 8, 12, 16):
        proj = generate_integrated_season_projections(2025, week)
        merged = proj.merge(actual_final[["team", "actual_wins"]].rename(columns={"actual_wins": "final_actual_wins"}),
                             on="team", how="inner")
        corr = merged["total_projection"].corr(merged["final_actual_wins"])
        mae = float(np.mean(np.abs(merged["total_projection"] - merged["final_actual_wins"])))
        print(f"   Week {week:>2}: corr={corr:+.3f} MAE={mae:.2f}")
        season_rows.append({"week": week, "corr": corr, "mae": mae})
    print(f"   vs. old baseline (EPA/weekly_update_pipeline-based Dynamic Tracking): "
          f"corr trajectory +0.069 (wk1) -> +0.974 (wk16, but poorly calibrated CIs)")

    # 3. Fantasy
    fantasy_results = generate_integrated_fantasy_projections()
    print(f"\n3. FANTASY PROJECTIONS (real 2025):")
    old_baseline = {"QB": 0.435, "RB": -0.504, "WR": 0.591, "TE": 0.436}
    for pos, r in fantasy_results.items():
        print(f"   {pos}: corr={r['corr']:+.3f} (n={r['n']}) | old baseline: {old_baseline[pos]:+.3f} | "
              f"delta {r['corr'] - old_baseline[pos]:+.3f}")

    print(f"\n{'=' * 70}")

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    game_preds.to_csv(os.path.join(PROCESSED_DIR, "integrated_game_predictions_2025.csv"), index=False, encoding="utf-8")
    pd.DataFrame(season_rows).to_csv(os.path.join(DIAGNOSTIC_DIR, "integrated_season_backtest_2025.csv"),
                                       index=False, encoding="utf-8")
    print(f"Saved data/processed/integrated_game_predictions_2025.csv, "
          f"data/diagnostic/integrated_season_backtest_2025.csv")

    return {"game_corr": game_corr, "game_mae": game_mae, "n_vegas": n_vegas,
            "season_trajectory": season_rows, "fantasy": fantasy_results}


if __name__ == "__main__":
    run_full_2025_backtest()
