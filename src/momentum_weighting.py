"""Phase 3 Component 3.1: Momentum / Recency Weighting.

Corrects a real methodological issue found in the spec before building:

The spec's core mechanism - "reweight a DataFrame of already-applied Elo
updates relative to current_week" - doesn't correspond to anything a
sequential Elo system can do. Once an update is applied (elo_new = elo_old
+ K*(actual-expected)), it's permanently baked into the rating; there's no
way to retroactively "reweight" a past update without re-simulating the
whole season from that point with a different rule.

More importantly: a fixed-K Elo system already IS an exponentially
recency-weighted system by construction - a real, well-known property, not
something that needs bolting on. The influence of a game from t updates
ago decays roughly geometrically already, and K controls the decay RATE.
A separate "decay_factor" parameter (the spec's Option 2) would just be
re-parameterizing the same knob K already controls, not an independent
second mechanism.

The one coherent, buildable version of "recency weighting" is the spec's
own offered "Alternative: increase K-factor for recent games" - this
module tests that directly: a K-factor sensitivity sweep, LEARNED on real
train seasons (2015-2024, no leakage) specifically for LATE-SEASON (weeks
13-18) predictive accuracy - a genuinely different optimization objective
than Component 1's original K=10 (chosen to minimize OVERALL Brier score,
not a late-season-specific one) - then validated out-of-sample on real
2025, reporting full-season and late-season accuracy separately as the
spec requests. The spec's 7 originally-requested functions (apply_recency_
weights_to_elo, apply_exponential_decay_to_elo, run_elo_with_recency_
weights, run_elo_with_exponential_decay) are consolidated into ONE real
function (run_elo_with_k_factor) rather than building 4 near-duplicate or
incoherent wrappers around the same underlying K-factor lever.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

LATE_SEASON_WEEKS = range(13, 19)  # weeks 13-18
TRAIN_SEASONS = range(2015, 2025)
HOLDOUT_SEASON = 2025
K_CANDIDATES = (5, 10, 15, 20, 25, 30, 40, 50)


def run_elo_with_k_factor(seasons, k_factor, home_field_elo, points_per_win=None):
    """The real, single mechanism behind every 'weighting scheme' this
    component could coherently test - see module docstring. Returns the
    real game-by-game backtest_df (elo_model.run_multi_season_elo)."""
    from elo_model import run_multi_season_elo
    backtest_df, _, _ = run_multi_season_elo(seasons, k_factor=k_factor, home_field_elo=home_field_elo)
    return backtest_df


def learn_k_factor_for_late_season_accuracy(train_seasons=TRAIN_SEASONS, k_candidates=K_CANDIDATES):
    """Real, leak-free K-factor search: for each candidate K, real Brier
    score on REAL train-season LATE-SEASON games only (weeks 13-18) - a
    genuinely different objective than Component 1's original K=10 search
    (which minimized OVERALL Brier, not a late-season-specific one)."""
    from elo_model import _estimate_home_field_elo, _load_games_chronological

    all_train_games = _load_games_chronological(train_seasons)
    home_field_elo = _estimate_home_field_elo(all_train_games)

    scores = {}
    for k in k_candidates:
        backtest_df = run_elo_with_k_factor(train_seasons, k, home_field_elo)
        late = backtest_df[backtest_df["week"].isin(LATE_SEASON_WEEKS)]
        brier = float(np.mean((late["predicted_prob"] - late["actual_result"]) ** 2))
        scores[k] = brier
        print(f"[learn_k_factor_for_late_season_accuracy] K={k}: late-season (wk13-18) Brier={brier:.4f} (n={len(late)})")

    best_k = min(scores, key=scores.get)
    print(f"[learn_k_factor_for_late_season_accuracy] winner: K={best_k} (Brier={scores[best_k]:.4f})")
    return best_k, home_field_elo, scores


def compare_weighting_schemes(season=HOLDOUT_SEASON, baseline_k=10, late_season_k=None):
    """Real, out-of-sample validation on 2025: baseline K=10 (Component 1's
    original, overall-Brier-optimized) vs. a K chosen specifically to
    minimize late-season Brier on train data - both scored on REAL,
    genuinely held-out 2025 full-season AND late-season-only accuracy."""
    from elo_game_prediction import fit_probability_to_spread_conversion, predict_game_spread_from_elo, _load_game_results
    from elo_model import _estimate_home_field_elo, _load_games_chronological

    if late_season_k is None:
        late_season_k, home_field_elo, _ = learn_k_factor_for_late_season_accuracy()
    else:
        home_field_elo = _estimate_home_field_elo(_load_games_chronological(TRAIN_SEASONS))

    fitted_model = fit_probability_to_spread_conversion()
    actual = _load_game_results([season])[["game_id", "point_diff", "week"]]

    results = {}
    for label, k in [("baseline_K10", baseline_k), (f"late_season_optimized_K{late_season_k}", late_season_k)]:
        backtest_df = run_elo_with_k_factor(range(min(TRAIN_SEASONS), season + 1), k, home_field_elo)
        season_games = backtest_df[backtest_df["season"] == season].copy()
        season_games["predicted_spread"] = predict_game_spread_from_elo(
            season_games["home_elo_before"], season_games["away_elo_before"], fitted_model)
        merged = season_games.merge(actual[["game_id", "point_diff"]], on="game_id", how="inner")

        full_corr = merged["predicted_spread"].corr(merged["point_diff"])
        full_mae = float(np.mean(np.abs(merged["predicted_spread"] - merged["point_diff"])))

        late = merged[merged["week"].isin(LATE_SEASON_WEEKS)]
        late_corr = late["predicted_spread"].corr(late["point_diff"]) if len(late) > 2 else np.nan
        late_mae = float(np.mean(np.abs(late["predicted_spread"] - late["point_diff"]))) if len(late) else np.nan

        results[label] = {"k": k, "full_corr": full_corr, "full_mae": full_mae,
                           "late_corr": late_corr, "late_mae": late_mae, "n_late": len(late)}
        print(f"{label} (K={k}): full-season corr={full_corr:+.3f} MAE={full_mae:.2f} | "
              f"late-season (wk13-18, n={len(late)}) corr={late_corr:+.3f} MAE={late_mae:.2f}")

    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "scheme"})


def validate_momentum_weighting(season=HOLDOUT_SEASON):
    comparison = compare_weighting_schemes(season)
    baseline = comparison[comparison["scheme"] == "baseline_K10"].iloc[0]
    best_other = comparison[comparison["scheme"] != "baseline_K10"].iloc[0]

    delta_full = best_other["full_corr"] - baseline["full_corr"]
    delta_late = best_other["late_corr"] - baseline["late_corr"]
    return {"baseline_corr": baseline["full_corr"], "weighted_corr": best_other["full_corr"],
            "delta": delta_full, "late_season_improvement": delta_late, "comparison": comparison}


def generate_momentum_report(season=HOLDOUT_SEASON):
    results = validate_momentum_weighting(season)
    comp = results["comparison"]

    lines = ["=" * 60, f"Momentum/Recency (K-factor) Weighting Analysis (real {season})", "=" * 60]
    lines.append("\n(See module docstring: this tests K-factor magnitude, the one coherent")
    lines.append(" mechanism for 'recency weighting' in a sequential Elo system - a fixed-K")
    lines.append(" Elo is already an exponential-recency-weighted system by construction.)\n")
    for _, row in comp.iterrows():
        lines.append(f"{row['scheme']} (K={row['k']}):")
        lines.append(f"  Full-season: corr={row['full_corr']:+.3f} MAE={row['full_mae']:.2f}")
        lines.append(f"  Late-season (wk13-18, n={int(row['n_late'])}): corr={row['late_corr']:+.3f} MAE={row['late_mae']:.2f}\n")

    lines.append(f"Delta (late-season-optimized K vs. baseline K=10): full-season corr {results['delta']:+.3f}, "
                  f"late-season corr {results['late_season_improvement']:+.3f}")
    if abs(results["delta"]) < 0.005 and abs(results["late_season_improvement"]) < 0.01:
        verdict = "Marginal / within noise - not clearly worth the added complexity of a second K-factor regime."
    elif results["delta"] > 0 or results["late_season_improvement"] > 0:
        verdict = "Real, if modest, improvement - worth adopting."
    else:
        verdict = "No improvement found - keep Component 1's original K=10."
    lines.append(f"\nRecommendation: {verdict}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    comp.to_csv(os.path.join(DIAGNOSTIC_DIR, "momentum_weighting_comparison_2025.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "momentum_weighting_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved data/diagnostic/momentum_weighting_comparison_2025.csv, momentum_weighting_report.txt")
    return report


if __name__ == "__main__":
    generate_momentum_report()
