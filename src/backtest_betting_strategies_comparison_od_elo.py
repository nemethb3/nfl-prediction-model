"""Real comparison of moneyline/ATS/totals betting strategies driven by
single-Elo vs. real O/D Elo, on real held-out games with real Vegas odds.

Real, serious problems found and fixed in the originally pasted spec before
writing this:

1. Assumed pre-existing files that don't exist: `data/processed/game_
   prediction_model_single_elo.json`, `game_prediction_model_od_elo.json`,
   `elo_by_week_2015_2025.json`, `team_elo_history_offensive_defensive_
   2015_2025.json` (the real file is a .csv, not .json - same exact bug
   already flagged and fixed once this session). `games['season_type']`
   doesn't exist either (real column is `game_type`).
2. Win probabilities were a fabricated, unfit `1 / (1 + exp(-spread/15))`
   formula for BOTH models, instead of each model's own real, already-fit
   probability model (elo_game_prediction.calculate_win_probability_from_
   elo for single-Elo; compute_offensive_defensive_elo.od_elo_win_
   probability for O/D Elo - both reused directly here).
3. Vegas spread/total fell back to `single_elo_spread * 0.98` (a circular
   proxy derived from one of the two models being compared - the same bug
   pattern already flagged and fixed twice this session) and a flat 42.5
   for every game when no real line was posted. Real Vegas lines exist for
   every real REG game 2015-2025 in data/backtest/vegas_with_results_
   2015_2025.csv (spread_line, total_line, real per-game moneylines and
   spread/total odds) - used directly, no proxy needed.
4. The ATS strategy's "correct" check compared the bet side against
   `home_won` (straight-up winner) - that's a moneyline check, not a real
   ATS cover check, so it would have silently mislabeled every ATS result.
   Real fix: reuses betting_backtest.py's already-validated `_settle_ats`,
   which checks the actual margin against the real posted spread.
5. Flat -110 assumed for every bet (moneyline, ATS, and totals) - real
   per-game odds already exist for all three in vegas_with_results_
   2015_2025.csv and are used directly via betting_backtest.py's real
   `payout_for_stake`.
6. Totals had no real O/D-Elo-aware model at all - `od_elo_total` was
   invented from `od_elo_spread` via the same fabricated `42.5 + spread/15
   * 2.5` shape as the single-Elo side, with no statistical fit. This
   project's real point-totals model (train_point_totals_model.py) only
   uses single-Elo features. Built a real, parallel O/D-Elo totals model
   here (home_o_elo/home_d_elo/away_o_elo/away_d_elo + week_norm +
   season_norm, same real linear-regression methodology) so the totals
   comparison is a genuine model-vs-model test, not a fabricated-formula
   dressed up as one.
7. The spec's "backtest 2015-2025" would have been in-sample for any model
   fit on the same span (no real train/holdout split was ever defined -
   the assumed prediction-model JSON files were treated as already fit).
   Real fix: same real train/holdout convention already established this
   session (train 2015-2023, holdout 2024 as the primary real result, 2025
   for context) - both models' probability/spread/totals components are
   fit ONLY on train seasons, then genuinely evaluated out-of-sample.

Reuses this project's already-validated real settlement code directly
(betting_backtest.py's moneyline_to_implied_probability, payout_for_stake,
should_bet, get_bet_direction, _settle_moneyline, _settle_ats,
_our_system_direction) rather than re-implementing betting math that could
silently diverge from the already-checked version.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from generation_timestamps import record_generation
from elo_game_prediction import (
    fit_probability_to_spread_conversion,
    generate_elo_game_spreads,
    calculate_win_probability_from_elo,
)
from compute_offensive_defensive_elo import fit_od_elo_model, generate_od_elo_game_spreads
from betting_backtest import payout_for_stake, _settle_moneyline, _settle_ats, _our_system_direction

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "elo_model_comparison.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "betting_strategies_comparison_od_elo.json")

TRAIN_SEASONS = range(2015, 2024)
HOLDOUT_SEASON = 2024
CONTEXT_SEASON = 2025
ALL_SEASONS = range(2015, 2026)

TOTALS_THRESHOLDS = [1.0, 2.0, 3.0]
SINGLE_TOTALS_FEATURES = ["elo_sum", "elo_diff", "week_norm", "season_norm"]
OD_TOTALS_FEATURES = ["home_o_elo", "home_d_elo", "away_o_elo", "away_d_elo", "week_norm", "season_norm"]


def _real_vegas_games():
    v = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    v = v[v["game_type"] == "REG"].copy()
    v["point_diff"] = v["home_score"] - v["away_score"]
    v["actual_winner"] = np.where(
        v["home_score"] > v["away_score"], v["home_team"],
        np.where(v["away_score"] > v["home_score"], v["away_team"], "TIE"),
    )
    return v


def _all_single_elo_spreads(fitted_model):
    frames = [generate_elo_game_spreads(s, fitted_model) for s in ALL_SEASONS]
    return pd.concat(frames, ignore_index=True)[["game_id", "home_elo", "away_elo", "predicted_spread"]]


def _all_od_elo_spreads(fitted_od):
    frames = [generate_od_elo_game_spreads(s, fitted_od) for s in ALL_SEASONS]
    return pd.concat(frames, ignore_index=True)[
        ["game_id", "home_o_elo", "home_d_elo", "away_o_elo", "away_d_elo", "win_prob_home", "predicted_spread"]
    ]


def _normalize_week_season(df, week_bounds, season_bounds):
    week_min, week_max = week_bounds
    season_min, season_max = season_bounds
    df["week_norm"] = (df["week"] - week_min) / (week_max - week_min)
    df["season_norm"] = (df["season"] - season_min) / (season_max - season_min)
    return df


def _fit_totals_model(train_df, features):
    scaler = StandardScaler().fit(train_df[features].to_numpy())
    model = LinearRegression().fit(scaler.transform(train_df[features].to_numpy()), train_df["total"].to_numpy())
    return scaler, model


def _predict_totals(df, features, scaler, model):
    return model.predict(scaler.transform(df[features].to_numpy()))


def _season_summary(bets):
    if not bets:
        return {"total_bets": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0.0,
                "pnl_units": 0.0, "roi_pct": 0.0}
    df = pd.DataFrame(bets)
    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    pushes = int((df["result"] == "push").sum())
    decided = wins + losses
    pnl_units = float(df["pnl_units"].sum())
    return {
        "total_bets": int(len(df)), "wins": wins, "losses": losses, "pushes": pushes,
        "win_pct": round(wins / decided * 100, 1) if decided else 0.0,
        "pnl_units": round(pnl_units, 3),
        "roi_pct": round(pnl_units / decided * 100, 2) if decided else 0.0,
    }


def _moneyline_and_ats_bets(games_df):
    """Real bets for one model's real predictions, using betting_backtest.py's
    already-validated `_our_system_direction` (should-bet gate on real
    moneyline-implied probability, real sign-safe favorite/underdog
    direction) settled two real ways."""
    moneyline_bets, ats_bets = [], []
    for _, game in games_df.iterrows():
        side = _our_system_direction(game)
        if side is None:
            continue
        result, odds_used, pnl = _settle_moneyline(side, game)
        moneyline_bets.append({"result": result, "pnl_units": round(float(pnl), 3)})
        result, odds_used, pnl = _settle_ats(side, game)
        ats_bets.append({"result": result, "pnl_units": round(float(pnl), 3)})
    return moneyline_bets, ats_bets


def _totals_bets(df, threshold):
    bets = df[df["edge"].abs() > threshold].copy()
    bets_out = []
    for _, row in bets.iterrows():
        bet_over = row["edge"] > 0
        odds = row["over_odds"] if bet_over else row["under_odds"]
        if row["total"] == row["total_line"]:
            bets_out.append({"result": "push", "pnl_units": 0.0})
            continue
        actual_over = row["total"] > row["total_line"]
        if bet_over == actual_over:
            bets_out.append({"result": "win", "pnl_units": round(float(payout_for_stake(odds)), 3)})
        else:
            bets_out.append({"result": "loss", "pnl_units": -1.0})
    return bets_out


def backtest_betting_strategies_comparison():
    print("\nBacktesting moneyline/ATS/totals strategies: single-Elo vs. real O/D Elo...\n")
    print(f"Real train/holdout convention matching this project's existing O/D Elo validation: "
          f"train {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}, holdout {HOLDOUT_SEASON} (primary), "
          f"{CONTEXT_SEASON} for context.\n")

    vegas = _real_vegas_games()

    # --- Single-Elo: real probability/spread model, fit train-only ---
    single_fitted = fit_probability_to_spread_conversion(train_seasons=TRAIN_SEASONS)
    single_spreads = _all_single_elo_spreads(single_fitted)

    # --- O/D Elo: real probability/spread model, fit train-only, real k ---
    with open(COMPARISON_PATH, encoding="utf-8") as f:
        best_k = json.load(f)["od_k_factor_selected"]
    od_fitted = fit_od_elo_model(train_seasons=TRAIN_SEASONS, k_factor=best_k)
    od_spreads = _all_od_elo_spreads(od_fitted)

    # --- Real totals models, fit train-only ---
    single_feat = vegas.merge(single_spreads, on="game_id", how="inner")
    single_feat["elo_sum"] = single_feat["home_elo"] + single_feat["away_elo"]
    single_feat["elo_diff"] = (single_feat["home_elo"] - single_feat["away_elo"]).abs()
    week_bounds = (single_feat["week"].min(), single_feat["week"].max())
    season_bounds = (single_feat["season"].min(), single_feat["season"].max())
    single_feat = _normalize_week_season(single_feat, week_bounds, season_bounds)

    od_feat = vegas.merge(od_spreads, on="game_id", how="inner")
    od_feat = _normalize_week_season(od_feat, week_bounds, season_bounds)

    single_train = single_feat[single_feat["season"].isin(list(TRAIN_SEASONS)) & single_feat["total_line"].notna()]
    od_train = od_feat[od_feat["season"].isin(list(TRAIN_SEASONS)) & od_feat["total_line"].notna()]
    single_scaler, single_totals_model = _fit_totals_model(single_train, SINGLE_TOTALS_FEATURES)
    od_scaler, od_totals_model = _fit_totals_model(od_train, OD_TOTALS_FEATURES)

    results = {"single_elo": {"moneyline": {}, "ats": {}, "totals": {}},
               "od_elo": {"moneyline": {}, "ats": {}, "totals": {}}}

    for season_label, season in [("holdout_2024", HOLDOUT_SEASON), ("context_2025", CONTEXT_SEASON)]:
        print(f"\n{'=' * 70}\n{season_label.upper()}\n{'=' * 70}")

        # ---- Moneyline + ATS ----
        se_season = single_feat[single_feat["season"] == season].copy()
        se_season["win_prob_home"] = calculate_win_probability_from_elo(se_season["home_elo"], se_season["away_elo"])
        se_season["win_prob_away"] = 1.0 - se_season["win_prob_home"]
        se_season["our_spread"] = se_season["predicted_spread"]
        se_season["vegas_spread"] = se_season["spread_line"]

        od_season = od_feat[od_feat["season"] == season].copy()
        od_season["win_prob_away"] = 1.0 - od_season["win_prob_home"]
        od_season["our_spread"] = od_season["predicted_spread"]
        od_season["vegas_spread"] = od_season["spread_line"]

        se_ml_bets, se_ats_bets = _moneyline_and_ats_bets(se_season)
        od_ml_bets, od_ats_bets = _moneyline_and_ats_bets(od_season)

        results["single_elo"]["moneyline"][season_label] = _season_summary(se_ml_bets)
        results["od_elo"]["moneyline"][season_label] = _season_summary(od_ml_bets)
        results["single_elo"]["ats"][season_label] = _season_summary(se_ats_bets)
        results["od_elo"]["ats"][season_label] = _season_summary(od_ats_bets)

        print(f"Moneyline | single-Elo: {results['single_elo']['moneyline'][season_label]}")
        print(f"Moneyline | O/D Elo:    {results['od_elo']['moneyline'][season_label]}")
        print(f"ATS       | single-Elo: {results['single_elo']['ats'][season_label]}")
        print(f"ATS       | O/D Elo:    {results['od_elo']['ats'][season_label]}")

        # ---- Totals ----
        se_totals = se_season[se_season["total_line"].notna()].copy()
        se_totals["predicted_total"] = _predict_totals(se_totals, SINGLE_TOTALS_FEATURES, single_scaler, single_totals_model)
        se_totals["edge"] = se_totals["predicted_total"] - se_totals["total_line"]

        od_totals = od_season[od_season["total_line"].notna()].copy()
        od_totals["predicted_total"] = _predict_totals(od_totals, OD_TOTALS_FEATURES, od_scaler, od_totals_model)
        od_totals["edge"] = od_totals["predicted_total"] - od_totals["total_line"]

        results["single_elo"]["totals"][season_label] = {}
        results["od_elo"]["totals"][season_label] = {}
        for threshold in TOTALS_THRESHOLDS:
            se_bets = _totals_bets(se_totals, threshold)
            od_bets = _totals_bets(od_totals, threshold)
            results["single_elo"]["totals"][season_label][str(threshold)] = _season_summary(se_bets)
            results["od_elo"]["totals"][season_label][str(threshold)] = _season_summary(od_bets)
            print(f"Totals (edge>{threshold}) | single-Elo: {results['single_elo']['totals'][season_label][str(threshold)]}")
            print(f"Totals (edge>{threshold}) | O/D Elo:    {results['od_elo']['totals'][season_label][str(threshold)]}")

    output = {
        "methodology": (
            "Real train/holdout split (train 2015-2023, holdout 2024 as the primary real result, "
            "2025 for context) - all probability/spread/totals model components refit on train "
            "seasons only for both arms, then genuinely evaluated out-of-sample. Real Vegas odds "
            "(moneyline, spread odds, total odds) from data/backtest/vegas_with_results_2015_2025.csv "
            "used for every settlement - no synthetic odds. Moneyline/ATS direction and settlement "
            "reuse betting_backtest.py's already-validated real logic unchanged. Totals uses two real, "
            "separately-fit linear regression models (single-Elo: elo_sum/elo_diff features; O/D Elo: "
            "home/away offensive and defensive Elo features directly) - not a shared formula."
        ),
        "od_k_factor": best_k,
        "results": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("betting_strategies_comparison_od_elo")
    print(f"\nWrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    backtest_betting_strategies_comparison()
