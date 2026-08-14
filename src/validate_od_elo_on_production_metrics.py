"""Real validation of O/D Elo against this project's actual production
spread-generation methodology - spread MAE (vs. real actual point margin,
not a fabricated Vegas proxy), real 90% CI coverage, and Brier score.

Real, serious problems found and fixed in the originally pasted spec
before writing this (not just wrong file paths this time):

1. The spec's `games['vegas_spread'] = games.get('spread_line',
   games['elo_spread'] * 0.95)` - `game_results_2015_2025.csv` has no
   spread_line column, so pandas' DataFrame.get() silently falls through
   to the WHOLE fallback Series: real "vegas_spread" would have just been
   elo_spread*0.95, a circular proxy derived from one of the two models
   being compared. Real Vegas lines come from
   data/backtest/vegas_with_results_2015_2025.csv - used here, but only
   as a secondary, clearly-labeled "do we agree with the market" context
   metric, not the primary accuracy measure (Vegas isn't training-label
   ground truth; real actual point margin is).
2. The spec's real "Spread MAE" computation (`mean_absolute_error(
   spreads_test, spreads_vegas_test)`) never referenced the model's own
   prediction at all - it would have produced an identical number for
   both the Single-Elo and O/D-Elo runs regardless of which is better.
   Real fix: MAE is computed between each model's own real predicted
   spread and the real actual point margin.
3. The spec's CI half-width used an asserted, unfit "3.5 points per Elo
   point" conversion and a 1.96 z-score (95% CI) mislabeled as a 90% CI
   (this project's own real code elsewhere correctly uses 1.645 for 90%).
   Real fix: reuses this project's actual real methodology instead -
   elo_game_prediction.py's real fit_probability_to_spread_conversion
   (candidate linear/logit/normal forms, picked by real train MAE, with a
   real fitted resid_std for the CI band) for the single-Elo arm, and a
   parallel, equally-real version here for O/D Elo (same real candidate-
   form comparison, same real 1.645 * resid_std 90% band), fit on the
   exact same real train/holdout split (train 2015-2023, holdout 2024,
   2025 for context) as the production single-Elo model, for a genuinely
   fair, apples-to-apples comparison.

The O/D-Elo fitting/prediction logic (fit_od_elo_model,
od_elo_win_probability, predict_od_spread) now lives in
compute_offensive_defensive_elo.py, shared with generate_od_elo_game_
spreads() (used by the real production pipeline after the pipeline swap)
- this script no longer keeps its own private copy, avoiding the two
implementations silently drifting apart."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from generation_timestamps import record_generation
from compute_offensive_defensive_elo import compute_offensive_defensive_elo, fit_od_elo_model, \
    od_elo_win_probability, predict_od_spread

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "elo_model_comparison.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "od_elo_production_validation.json")

TRAIN_SEASONS = range(2015, 2024)
HOLDOUT_SEASON = 2024
CONTEXT_SEASON = 2025
CI_Z_90 = 1.645  # real 90% CI z-score, same real constant elo_game_prediction.py uses


def _real_point_diffs(seasons):
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    games = games[(games["season"].isin(list(seasons))) & (games["game_type"] == "REG")].copy()
    games["point_diff"] = games["home_score"] - games["away_score"]
    return games[["game_id", "season", "point_diff"]]


def _real_vegas_spreads(seasons):
    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas = vegas[(vegas["season"].isin(list(seasons))) & (vegas["game_type"] == "REG")
                  & vegas["spread_line"].notna()]
    return vegas[["game_id", "spread_line"]]


def _real_single_elo_arm():
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads, \
        calculate_win_probability_from_elo
    fitted_spread_model = fit_probability_to_spread_conversion(train_seasons=TRAIN_SEASONS)

    def _score(season):
        preds = generate_elo_game_spreads(season, fitted_spread_model)
        preds["win_prob"] = calculate_win_probability_from_elo(preds["home_elo"], preds["away_elo"])
        actual = _real_point_diffs([season])
        merged = preds.merge(actual, on="game_id", how="inner")
        return merged

    return _score, fitted_spread_model["resid_std"], fitted_spread_model["form"]


def _real_od_elo_arm(k_factor):
    fitted = fit_od_elo_model(train_seasons=TRAIN_SEASONS, k_factor=k_factor)
    history_df, _, _ = compute_offensive_defensive_elo(k_factor=k_factor, save=False)
    history_df["od_elo_spread"] = ((history_df["home_o_elo_before"] - history_df["away_d_elo_before"]) -
                                    (history_df["away_o_elo_before"] - history_df["home_d_elo_before"]))

    def _score(season):
        season_pd = _real_point_diffs([season])
        merged = history_df.merge(season_pd, on="game_id", how="inner")
        merged["win_prob"] = od_elo_win_probability(merged["od_elo_spread"].to_numpy(), fitted)
        merged["predicted_spread"] = predict_od_spread(merged["win_prob"].to_numpy(), fitted["spread_model"])
        return merged

    return _score, fitted["spread_model"]["resid_std"], fitted["spread_model"]


def _real_metrics(merged, resid_std):
    band = CI_Z_90 * resid_std
    ci_low = merged["predicted_spread"] - band
    ci_high = merged["predicted_spread"] + band
    mae = float(np.mean(np.abs(merged["predicted_spread"] - merged["point_diff"])))
    coverage = float(((merged["point_diff"] >= ci_low) & (merged["point_diff"] <= ci_high)).mean())
    decided = merged[merged["point_diff"] != 0]
    actual_result = (decided["point_diff"] > 0).astype(int)
    brier = float(brier_score_loss(actual_result, np.clip(decided["win_prob"], 0.01, 0.99)))
    return {"n_games": int(len(merged)), "spread_mae": round(mae, 3), "ci_coverage_90": round(coverage, 4),
            "brier_score": round(brier, 4)}


def validate_od_elo_on_production_metrics():
    print("\nValidating real O/D Elo against this project's real production spread methodology...\n")
    with open(COMPARISON_PATH, encoding="utf-8") as f:
        comparison = json.load(f)
    best_k = comparison["od_k_factor_selected"]

    single_score_fn, single_resid_std, single_form = _real_single_elo_arm()
    od_score_fn, od_resid_std, od_fitted = _real_od_elo_arm(best_k)
    print(f"Real single-Elo spread conversion (reused from elo_game_prediction.py's own real fit): "
          f"form={single_form}, resid_std={single_resid_std:.2f}")
    print(f"Real O/D-Elo spread conversion: form={od_fitted['form']}, resid_std={od_resid_std:.2f}")

    results = {}
    for season_label, season in [("holdout_2024", HOLDOUT_SEASON), ("context_2025", CONTEXT_SEASON)]:
        single_merged = single_score_fn(season)
        single_merged["season"] = season
        od_merged = od_score_fn(season)

        results.setdefault(season_label, {})
        results[season_label]["single_elo"] = _real_metrics(single_merged, single_resid_std)
        results[season_label]["od_elo"] = _real_metrics(od_merged, od_resid_std)

        # Real, secondary, clearly-labeled context: agreement with the real Vegas line (not ground truth)
        vegas = _real_vegas_spreads([season])
        if len(vegas):
            single_vs_vegas = single_merged.merge(vegas, on="game_id", how="inner")
            od_vs_vegas = od_merged.merge(vegas, on="game_id", how="inner")
            results[season_label]["single_elo"]["mae_vs_real_vegas_line"] = round(float(
                np.mean(np.abs(single_vs_vegas["predicted_spread"] - single_vs_vegas["spread_line"]))), 3)
            results[season_label]["od_elo"]["mae_vs_real_vegas_line"] = round(float(
                np.mean(np.abs(od_vs_vegas["predicted_spread"] - od_vs_vegas["spread_line"]))), 3)

        print(f"\n{season_label}:")
        for model in ["single_elo", "od_elo"]:
            m = results[season_label][model]
            print(f"  {model:10} | MAE(vs actual)={m['spread_mae']:.2f}  "
                  f"90% CI coverage={100 * m['ci_coverage_90']:.1f}%  Brier={m['brier_score']:.4f}"
                  + (f"  MAE(vs Vegas)={m.get('mae_vs_real_vegas_line')}" if "mae_vs_real_vegas_line" in m else ""))

    # Real decision rule, evaluated on the real primary holdout (2024) - 2025 shown for context only
    h = results["holdout_2024"]
    mae_win = h["od_elo"]["spread_mae"] < h["single_elo"]["spread_mae"]
    brier_win = h["od_elo"]["brier_score"] < h["single_elo"]["brier_score"]
    coverage_ok = abs(h["od_elo"]["ci_coverage_90"] - 0.90) <= abs(h["single_elo"]["ci_coverage_90"] - 0.90) + 0.05

    if mae_win and brier_win and coverage_ok:
        decision = "SWAP_PIPELINE"
        verdict = "O/D Elo real-beats single-Elo on spread MAE and Brier score, with comparable CI calibration."
    elif (mae_win or brier_win) and coverage_ok:
        decision = "HYBRID_INTEGRATION"
        verdict = "O/D Elo wins on some but not all real production metrics - selective integration only."
    else:
        decision = "KEEP_SINGLE_ELO"
        verdict = "O/D Elo does not real-beat single-Elo on the metrics the production spread model is validated on."
    print(f"\nDecision: {decision}\n{verdict}")

    output = {
        "methodology": (
            "Real train/holdout split matching elo_game_prediction.py's own production convention "
            f"(train {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}, holdout {HOLDOUT_SEASON}, "
            f"{CONTEXT_SEASON} for context). Spread MAE and CI coverage measured against real actual "
            "point margins (ground truth), not the Vegas line - Vegas agreement is reported separately "
            "as secondary context only."
        ),
        "od_k_factor": best_k,
        "results": results,
        "decision": decision,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("od_elo_production_validation")
    print(f"\nWrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    validate_od_elo_on_production_metrics()
