"""Phase 4: WR Dynamic Projections Backtest.

Verification-driven fixes from the pasted spec (Q&A decisions, 2026-07-30):

1. Real leave-one-out (LOO) baseline, not the existing leaky one. Checked
   `_wr_projections()` (generate_fantasy_dashboard_data.py) directly before
   building this: the shipped WR `projected_ppr` is fit via
   `season_projected_pts = slope * projected_score + intercept`, where
   slope/intercept come from an in-sample regression against each real WR
   player's own real FULL-SEASON actual total - the same total that
   includes whatever week is later being scored against it. Every one of
   the pasted spec's Approaches 1-4 just rescales that same leaky number,
   so a "winner" among them wouldn't reflect real predictive skill for a
   future season. Fixed here: for every real WR player-week, the
   population EPA->season-points regression is refit with that specific
   week's own real actual points excluded from that player's season total
   before projecting it - so no row's projection ever had access to the
   result it's being scored against. `expected_games_2025` (the per-game
   divisor) is already a real, leak-free PRESEASON estimate, so it's reused
   unchanged.

2. Real field-name bug fixed: the spec's code uses `player_name`
   throughout, but the real field in fantasy_rankings_2025.json is `name`
   (verified before writing this) - would have raised a KeyError on the
   first DataFrame slice.

3. `vegas_spread` is genuinely absent from fantasy_rankings_2025.json
   (verified) - Approach 4 (Game-Script Adjusted) always hits the spec's
   own real fallback branch and is a no-op identical to the baseline. Kept
   as specified (not an error), just disclosed.

4. `compute_metrics()`'s `tier_accuracy_pct` was dead code in the pasted
   spec - hardcoded to always return 0.0 (the spec's own comment admits
   it's "a placeholder"). Replaced with a real, well-defined metric: the
   real green-tier hit rate (% of |actual-projected| within this project's
   real, live-verified green-tier boundary for WR, +-3.2 PPR - see
   FantasyRankings.js's ACCURACY_TERCILE_RANGES, confirmed against a live
   re-run of _accuracy_tier_thresholds() in the prior task this session).
"""

import json
from generation_timestamps import record_generation
import os

import numpy as np
import pandas as pd
from scipy.special import expit

from fantasy_validation import extract_actual_fantasy_points_2025, project_fantasy_points_from_epa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
FANTASY_RANKINGS_PATH = os.path.join(FRONTEND_DATA_DIR, "fantasy_rankings_2025.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "wr_dynamic_backtest_results_2025.json")

WR_GREEN_TIER_PPR = 3.2  # real, verified boundary (see module docstring #4)
CURRENT_SHIPPED_CORR = 0.400  # the existing (leaky) dashboard's real correlation, for context only


def _wr_epa_and_season_actuals():
    proj = project_fantasy_points_from_epa("WR")
    raw = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2025.csv"))
    proj = proj.merge(raw[["player_id", "expected_games_2025"]], on="player_id", how="left")

    actual = extract_actual_fantasy_points_2025()
    actual_wr = actual[actual["position"] == "WR"].copy()
    actual_season = actual_wr.groupby("player_id")["actual_fantasy_pts"].sum().reset_index().rename(
        columns={"actual_fantasy_pts": "actual_season_fantasy_pts"})

    merged = proj.merge(actual_season, on="player_id", how="inner")
    merged = merged[merged["projected_volume"] > 0].reset_index(drop=True)
    merged = merged.dropna(subset=["expected_games_2025"])
    merged = merged[merged["expected_games_2025"] > 0].reset_index(drop=True)
    if len(merged) < 5:
        raise RuntimeError("Too few matched real WR players to fit a calibration - check upstream data.")
    return merged, actual_wr


def _leave_one_out_wr_baseline():
    """Real, leak-free per-(player, week) WR baseline - see module
    docstring #1. Returns a DataFrame keyed by (player_id, week) with
    projected_ppr_loo."""
    merged, actual_wr = _wr_epa_and_season_actuals()

    x_all = merged["projected_score"].to_numpy()
    y_all = merged["actual_season_fantasy_pts"].to_numpy()
    player_ids = merged["player_id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    expected_games = dict(zip(merged["player_id"], merged["expected_games_2025"]))
    projected_score = dict(zip(merged["player_id"], merged["projected_score"]))

    wr_weeks = actual_wr[actual_wr["player_id"].isin(id_to_idx)][["player_id", "week", "actual_fantasy_pts"]]

    rows = []
    for player_id, week, actual_this_week in wr_weeks.itertuples(index=False):
        idx = id_to_idx[player_id]
        y_loo = y_all.copy()
        y_loo[idx] = y_all[idx] - actual_this_week  # real leave-one-out: exclude this week's own result
        slope, intercept = np.polyfit(x_all, y_loo, 1)
        season_projected_pts_loo = slope * projected_score[player_id] + intercept
        per_game_loo = season_projected_pts_loo / expected_games[player_id]
        rows.append({"player_id": player_id, "week": int(week), "projected_ppr_loo": per_game_loo})

    return pd.DataFrame(rows)


def load_data():
    """Real WR game records for the backtest: leak-free LOO baseline
    (projected_ppr, replacing the shipped leaky number) merged with the
    real per-week context fields already exported to the dashboard
    (opponent, opponent_defense_rank_vs_position, recent_form, actual_ppr)."""
    with open(FANTASY_RANKINGS_PATH, encoding="utf-8") as f:
        fantasy_data = json.load(f)
    fantasy_df = pd.DataFrame(fantasy_data)
    wr_df = fantasy_df[(fantasy_df["position"] == "WR") & (fantasy_df["actual_ppr"].notna())].copy()
    wr_df["player_id"] = wr_df["id"].str.replace(r"_w\d+$", "", regex=True)

    loo = _leave_one_out_wr_baseline()
    wr_df = wr_df.merge(loo, on=["player_id", "week"], how="inner")
    wr_df["projected_ppr"] = wr_df["projected_ppr_loo"]  # LOO baseline replaces the shipped leaky value
    wr_df = wr_df.rename(columns={"name": "player_name"})
    return wr_df


def approach_1_volume_only(df):
    results = {
        "approach": "Volume-Only",
        "variation": "leave_one_out_baseline",
        "predictions": df["projected_ppr"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }
    return results


def approach_2_matchup_adjusted_linear(df, adjustment_pct):
    def adjust_projection(row):
        if pd.isna(row["opponent_defense_rank_vs_position"]):
            return row["projected_ppr"]
        rank = row["opponent_defense_rank_vs_position"]
        percentile = (rank - 1) / 31
        adjustment = (percentile - 0.5) * 2 * adjustment_pct
        return row["projected_ppr"] * (1 + adjustment)

    df_adjusted = df.copy()
    df_adjusted["adjusted_projection"] = df_adjusted.apply(adjust_projection, axis=1)
    return {
        "approach": "Matchup-Adjusted (Linear)",
        "variation": f"linear_{int(adjustment_pct*100)}pct",
        "predictions": df_adjusted["adjusted_projection"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }


def approach_2_matchup_adjusted_sigmoid(df, scale=0.15):
    def adjust_projection(row):
        if pd.isna(row["opponent_defense_rank_vs_position"]):
            return row["projected_ppr"]
        rank = row["opponent_defense_rank_vs_position"]
        x = (rank - 16.5) / 16.5
        sigmoid_val = expit(x / scale)
        adjustment = (sigmoid_val - 0.5) * 2
        return row["projected_ppr"] * (1 + adjustment * scale)

    df_adjusted = df.copy()
    df_adjusted["adjusted_projection"] = df_adjusted.apply(adjust_projection, axis=1)
    return {
        "approach": "Matchup-Adjusted (Sigmoid)",
        "variation": f"sigmoid_scale_{scale}",
        "predictions": df_adjusted["adjusted_projection"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }


def approach_2_matchup_adjusted_tiered(df):
    def adjust_projection(row):
        if pd.isna(row["opponent_defense_rank_vs_position"]):
            return row["projected_ppr"]
        rank = row["opponent_defense_rank_vs_position"]
        if rank <= 10:
            return row["projected_ppr"] * 0.85
        elif rank <= 22:
            return row["projected_ppr"] * 0.95
        return row["projected_ppr"] * 1.15

    df_adjusted = df.copy()
    df_adjusted["adjusted_projection"] = df_adjusted.apply(adjust_projection, axis=1)
    return {
        "approach": "Matchup-Adjusted (Tiered)",
        "variation": "tiered_buckets",
        "predictions": df_adjusted["adjusted_projection"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }


def _matchup_adjusted_series(df, method, param):
    if method == "linear":
        return df.apply(
            lambda row: row["projected_ppr"] * (1 + ((row["opponent_defense_rank_vs_position"] - 1) / 31 - 0.5) * 2 * param)
            if pd.notna(row["opponent_defense_rank_vs_position"]) else row["projected_ppr"],
            axis=1,
        ).to_numpy()
    if method == "sigmoid":
        return df.apply(
            lambda row: row["projected_ppr"] * (
                1 + (expit(((row["opponent_defense_rank_vs_position"] - 16.5) / 16.5) / param) - 0.5) * 2 * param
            ) if pd.notna(row["opponent_defense_rank_vs_position"]) else row["projected_ppr"],
            axis=1,
        ).to_numpy()
    if method == "tiered":
        return df.apply(
            lambda row: row["projected_ppr"] * (
                0.85 if pd.notna(row["opponent_defense_rank_vs_position"]) and row["opponent_defense_rank_vs_position"] <= 10
                else 0.95 if pd.notna(row["opponent_defense_rank_vs_position"]) and row["opponent_defense_rank_vs_position"] <= 22
                else 1.15 if pd.notna(row["opponent_defense_rank_vs_position"])
                else 1.0
            ),
            axis=1,
        ).to_numpy()
    raise ValueError(f"Unknown matchup method: {method}")


def approach_3_hybrid(df, volume_weight, matchup_weight, matchup_method="linear", matchup_param=0.10):
    volume_proj = df["projected_ppr"].to_numpy()
    matchup_proj = _matchup_adjusted_series(df, matchup_method, matchup_param)
    hybrid_proj = volume_weight * volume_proj + matchup_weight * matchup_proj
    return {
        "approach": "Hybrid",
        "variation": f"{int(volume_weight*100)}v_{int(matchup_weight*100)}m_{matchup_method}",
        "predictions": hybrid_proj.tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }


def approach_4_game_script_adjusted(df):
    if "vegas_spread" not in df.columns:
        return {
            "approach": "Game-Script Adjusted",
            "variation": "no_vegas_spread_data",
            "predictions": df["projected_ppr"].tolist(),
            "actuals": df["actual_ppr"].tolist(),
            "note": "vegas_spread is not present in fantasy_rankings_2025.json (verified before running) - "
                    "falls back to the same leave-one-out volume baseline as Approach 1, unchanged.",
        }

    def adjust_projection(row):
        if pd.isna(row["vegas_spread"]):
            return row["projected_ppr"]
        spread_adj = max(0.7, min(1.3, 1.0 - abs(row["vegas_spread"]) / 30))
        return row["projected_ppr"] * spread_adj

    df_adjusted = df.copy()
    df_adjusted["adjusted_projection"] = df_adjusted.apply(adjust_projection, axis=1)
    return {
        "approach": "Game-Script Adjusted",
        "variation": "spread_margin_based",
        "predictions": df_adjusted["adjusted_projection"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
    }


def approach_5_recent_form(df):
    def get_recent_avg(row):
        form = row["recent_form"]
        if isinstance(form, list) and len(form) > 0:
            valid = [x for x in form if pd.notna(x)]
            if valid:
                return float(np.mean(valid))
        return row["projected_ppr"]  # week-1-style fallback: real leave-one-out baseline, not the old leaky one

    df_recent = df.copy()
    df_recent["adjusted_projection"] = df_recent.apply(get_recent_avg, axis=1)
    return {
        "approach": "Recent Form",
        "variation": "trailing_4week_avg",
        "predictions": df_recent["adjusted_projection"].tolist(),
        "actuals": df["actual_ppr"].tolist(),
        "note": "Real, already leak-free (trailing weeks strictly before the target week) independent of the "
                "LOO baseline fix - falls back to the LOO baseline only when no real recent_form exists yet.",
    }


def compute_metrics(predictions, actuals):
    predictions = np.array(predictions, dtype=float)
    actuals = np.array(actuals, dtype=float)

    correlation = np.corrcoef(predictions, actuals)[0, 1]
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    green_tier_rate = float(np.mean(np.abs(predictions - actuals) <= WR_GREEN_TIER_PPR) * 100)

    return {
        "correlation": round(float(correlation), 4) if not np.isnan(correlation) else 0.0,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "green_tier_rate_pct": round(green_tier_rate, 1),
    }


def run_all_approaches(df):
    all_results = [approach_1_volume_only(df)]

    for pct in [0.05, 0.10, 0.15, 0.20]:
        all_results.append(approach_2_matchup_adjusted_linear(df, pct))
    for scale in [0.10, 0.15, 0.20]:
        all_results.append(approach_2_matchup_adjusted_sigmoid(df, scale))
    all_results.append(approach_2_matchup_adjusted_tiered(df))

    for vol_w in [0.5, 0.6, 0.7, 0.8]:
        match_w = round(1.0 - vol_w, 2)
        all_results.append(approach_3_hybrid(df, vol_w, match_w, "linear", 0.10))
    for vol_w in [0.5, 0.6, 0.7, 0.8]:
        match_w = round(1.0 - vol_w, 2)
        all_results.append(approach_3_hybrid(df, vol_w, match_w, "sigmoid", 0.15))
    for vol_w in [0.5, 0.6, 0.7, 0.8]:
        match_w = round(1.0 - vol_w, 2)
        all_results.append(approach_3_hybrid(df, vol_w, match_w, "tiered", 0.15))

    all_results.append(approach_4_game_script_adjusted(df))
    all_results.append(approach_5_recent_form(df))

    ranked_results = []
    for result in all_results:
        metrics = compute_metrics(result["predictions"], result["actuals"])
        ranked_results.append({
            "approach": result["approach"],
            "variation": result["variation"],
            "correlation": metrics["correlation"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "green_tier_rate_pct": metrics["green_tier_rate_pct"],
            "note": result.get("note", ""),
        })

    ranked_results.sort(key=lambda x: (-x["correlation"], x["mae"]))
    for i, result in enumerate(ranked_results):
        result["rank"] = i + 1
    return ranked_results


if __name__ == "__main__":
    wr_df = load_data()
    print(f"Loaded {len(wr_df)} real WR game records (leave-one-out baseline)")

    ranked_results = run_all_approaches(wr_df)
    baseline_result = next(r for r in ranked_results if r["variation"] == "leave_one_out_baseline")

    output = {
        "wr_records_tested": len(wr_df),
        "methodology_note": (
            "All predictions use a real leave-one-out (LOO) WR baseline, not the dashboard's shipped "
            "static season-total projection: each player-week's own real actual result is excluded from "
            "that player's season total before refitting the EPA->points calibration used to project it, "
            "so no row's projection ever saw the result it's being scored against. See module docstring."
        ),
        "all_results": ranked_results,
        "winner": ranked_results[0],
        "top_5": ranked_results[:5],
        "leave_one_out_baseline": {
            "approach": "Volume-Only (leave-one-out)",
            "correlation": baseline_result["correlation"],
            "note": "The real, leak-free reference point for judging improvement in this backtest.",
        },
        "currently_shipped_dashboard": {
            "approach": "Volume-Only (shipped, in-sample-calibrated)",
            "correlation": CURRENT_SHIPPED_CORR,
            "note": (
                "Not directly comparable to the results above - this is the existing dashboard's real "
                "correlation using the leaky (non-LOO) season-total baseline, shown for context only."
            ),
        },
    }

    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("wr_dynamic_backtest_results_2025")

    print("\n=== WR DYNAMIC PROJECTIONS BACKTEST (leak-free LOO baseline) ===")
    print(f"Leave-one-out Volume-Only baseline correlation: {baseline_result['correlation']}")
    print(f"Currently shipped (leaky) dashboard correlation: {CURRENT_SHIPPED_CORR}")
    print(f"\nWinner: {ranked_results[0]['approach']} ({ranked_results[0]['variation']})")
    print(f"Correlation: {ranked_results[0]['correlation']}")
    print(f"MAE: {ranked_results[0]['mae']} PPR")
    print(f"RMSE: {ranked_results[0]['rmse']} PPR")
    print("\nTop 5:")
    for r in ranked_results[:5]:
        print(f"  {r['rank']}. {r['approach']} ({r['variation']}) - corr {r['correlation']}, mae {r['mae']}, "
              f"green-tier {r['green_tier_rate_pct']}%")

    print(f"\nResults exported to: {OUTPUT_PATH}")
