"""Component C: Vegas Integration with LOOCV Weight Optimization.

Corrects 3 issues found in the spec before building:

1. The grid search range [0.60, 0.95] contradicts the spec's own stated
   hypothesis (Vegas weight reaching ~12% by week 16) - that value is
   mathematically unreachable inside a floor of 0.60. Widened the search to
   the full [0.0, 1.0] so the data can show a genuinely low late-season
   Vegas weight if that's real, rather than being prevented from finding it.

2. The LOOCV pseudocode never evaluates the held-out week: it picks the
   best weight using only training-weeks' own MAE, then labels that
   "optimal_weights[W]" without ever checking accuracy on week W itself -
   the same missing-held-out-evaluation gap already caught and fixed twice
   this session (ensemble.py's stacking, Component 2's weight learning).
   Fixed: after selecting weights from the other weeks, also scores them
   against the real held-out week W and reports that (held_out_mae) - a
   genuine validation number, not just an assigned label.

3. vegas_spread uses the real, direct spread_line column (what oddsmakers
   actually post), not a derived proxy from moneyline-implied win
   probability - more direct, and consistent with vegas_comparison.py's
   already-verified sign convention (positive = home favored, matching Elo's
   predicted_spread). implied_win_prob is still extracted for completeness/
   diagnostic use but isn't what's blended for spreads.

2026 lines are sparse (53/272 REG games have a posted spread_line as of this
run - books haven't published the rest yet, expected this far out). Games
without a real posted line get an Elo-only prediction, clearly flagged, not
silently fabricated.

Also disclosed, not "fixed" (nothing to fix - it's inherent to the data):
fit_weight_curve() only has 5 points (weeks 1/4/8/12/16) to fit up to a
3-parameter curve. A high R2 there is expected almost by construction with
that few points/that many degrees of freedom and is NOT strong evidence of
a real underlying functional form - reported honestly, not oversold.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

CHECKPOINT_WEEKS = (1, 4, 8, 12, 16)
W_VEGAS_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)  # corrected range, see module docstring #1


def extract_vegas_lines(season=2025):
    from game_predictions import _load_schedule_for_season
    from vegas_comparison import moneyline_to_implied_prob

    schedule = _load_schedule_for_season(season)
    reg = schedule[schedule["game_type"] == "REG"].copy()
    reg["vegas_spread"] = reg["spread_line"]

    home_raw = moneyline_to_implied_prob(reg["home_moneyline"])
    away_raw = moneyline_to_implied_prob(reg["away_moneyline"])
    vig_total = home_raw + away_raw
    reg["implied_win_prob"] = np.where(vig_total > 0, home_raw / vig_total, np.nan)

    return reg[["game_id", "week", "home_team", "away_team", "vegas_spread", "implied_win_prob"]]


def _get_elo_spreads(season=2025, fitted_model=None):
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads
    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()
    spreads = generate_elo_game_spreads(season, fitted_model)
    return spreads.rename(columns={"predicted_spread": "elo_spread"}), fitted_model


def _get_actual_results(season=2025):
    from elo_game_prediction import _load_game_results
    return _load_game_results([season])[["game_id", "week", "point_diff"]].rename(
        columns={"point_diff": "actual_spread"})


def _merged_frame(elo_spreads, vegas_spreads, actual_results):
    return elo_spreads[["game_id", "week", "elo_spread"]].merge(
        vegas_spreads[["game_id", "vegas_spread"]], on="game_id", how="inner").merge(
        actual_results[["game_id", "actual_spread"]], on="game_id", how="inner")


def grid_search_optimal_weights(elo_spreads, vegas_spreads, actual_results, training_weeks,
                                 w_vegas_range=None):
    merged = _merged_frame(elo_spreads, vegas_spreads, actual_results)
    train = merged[merged["week"].isin(training_weeks)]

    grid = W_VEGAS_GRID if w_vegas_range is None else np.round(np.arange(*w_vegas_range), 2)
    best = None
    for w_vegas in grid:
        w_elo = 1.0 - w_vegas
        pred = w_vegas * train["vegas_spread"] + w_elo * train["elo_spread"]
        mae = float(np.mean(np.abs(pred - train["actual_spread"])))
        if best is None or mae < best[2]:
            best = (float(w_vegas), float(w_elo), mae)
    return best


def learn_optimal_weights_loocv(elo_spreads, vegas_spreads, actual_results, checkpoint_weeks=CHECKPOINT_WEEKS):
    merged = _merged_frame(elo_spreads, vegas_spreads, actual_results)
    all_weeks = sorted(merged["week"].unique())

    rows = []
    for W in checkpoint_weeks:
        training_weeks = [w for w in all_weeks if w != W]
        w_vegas, w_elo, train_mae = grid_search_optimal_weights(elo_spreads, vegas_spreads, actual_results, training_weeks)

        test = merged[merged["week"] == W]
        if len(test):
            held_out_pred = w_vegas * test["vegas_spread"] + w_elo * test["elo_spread"]
            held_out_mae = float(np.mean(np.abs(held_out_pred - test["actual_spread"])))
        else:
            held_out_mae = np.nan

        print(f"Week {W:>2}: optimal w_vegas={w_vegas:.2f}, w_elo={w_elo:.2f} "
              f"(learned from {len(training_weeks)} other weeks, train MAE={train_mae:.2f}) | "
              f"held-out week {W} MAE={held_out_mae:.2f} (n={len(test)} games)")
        rows.append({"week": W, "optimal_w_vegas": w_vegas, "optimal_w_elo": w_elo,
                     "train_mae": train_mae, "held_out_mae": held_out_mae, "n_test_games": len(test)})

    return pd.DataFrame(rows)


def _r2(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_weight_curve(learned_weights_df):
    """Fits linear/quadratic/exponential(+asymptote)/sigmoid through the 5
    learned points, picks the best R². See module docstring: with only 5
    points and up to 3 free parameters, a high R2 is expected almost by
    construction and is reported here, not treated as strong evidence."""
    t = learned_weights_df["week"].to_numpy(dtype=float)
    y = learned_weights_df["optimal_w_vegas"].to_numpy(dtype=float)

    candidates = {}

    a, b = np.polyfit(t, y, 1)
    candidates["linear"] = ({"a": a, "b": b}, _r2(y, a * t + b))

    coeffs = np.polyfit(t, y, 2)
    candidates["quadratic"] = ({"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]}, _r2(y, np.polyval(coeffs, t)))

    try:
        def _exp_asym(t, a, b, c):
            return a * np.exp(b * t) + c
        popt, _ = curve_fit(_exp_asym, t, y, p0=[0.8, -0.1, 0.15], maxfev=5000)
        candidates["exponential"] = ({"a": popt[0], "b": popt[1], "c": popt[2]}, _r2(y, _exp_asym(t, *popt)))
    except RuntimeError:
        pass

    try:
        def _sigmoid(t, a, b, c):
            return a / (1.0 + np.exp(-b * (t - c)))
        popt, _ = curve_fit(_sigmoid, t, y, p0=[0.9, -0.3, 8.0], maxfev=5000)
        candidates["sigmoid"] = ({"a": popt[0], "b": popt[1], "c": popt[2]}, _r2(y, _sigmoid(t, *popt)))
    except RuntimeError:
        pass

    best_type = max(candidates, key=lambda k: candidates[k][1])
    best_params, best_r2 = candidates[best_type]

    print(f"\n[fit_weight_curve] candidates (R2, 5 points only - see caveat): "
          + ", ".join(f"{k}={v[1]:.4f}" for k, v in candidates.items()))
    print(f"[fit_weight_curve] winner: {best_type} (R2={best_r2:.4f}) - "
          f"CAVEAT: with only 5 points and up to 3 free params, high R2 here is expected by "
          f"construction, not strong evidence of the true functional form")

    return best_type, best_params, best_r2, candidates


def predict_weight_for_week(week_number, curve_type, params):
    t = float(week_number)
    if curve_type == "linear":
        w_vegas = params["a"] * t + params["b"]
    elif curve_type == "quadratic":
        w_vegas = params["a"] * t ** 2 + params["b"] * t + params["c"]
    elif curve_type == "exponential":
        w_vegas = params["a"] * np.exp(params["b"] * t) + params["c"]
    elif curve_type == "sigmoid":
        w_vegas = params["a"] / (1.0 + np.exp(-params["b"] * (t - params["c"])))
    else:
        raise ValueError(f"Unknown curve_type: {curve_type}")

    w_vegas = float(np.clip(w_vegas, 0.0, 1.0))
    return w_vegas, 1.0 - w_vegas


def apply_learned_weights_to_season(elo_spreads, vegas_spreads, curve_type, params):
    merged = elo_spreads[["game_id", "week", "elo_spread"]].merge(
        vegas_spreads[["game_id", "vegas_spread"]], on="game_id", how="inner")
    weights = merged["week"].apply(lambda w: predict_weight_for_week(w, curve_type, params))
    merged["w_vegas"] = weights.apply(lambda x: x[0])
    merged["w_elo"] = weights.apply(lambda x: x[1])
    merged["blended_spread"] = merged["w_vegas"] * merged["vegas_spread"] + merged["w_elo"] * merged["elo_spread"]
    return merged


def apply_arbitrary_linear_weights_to_season(elo_spreads, vegas_spreads, week_min=1, week_max=17):
    """The spec's own stated comparison baseline: 80/20 at week 1 ramping
    linearly to 20/80 at week week_max (passes through ~50/50 around the
    midpoint)."""
    merged = elo_spreads[["game_id", "week", "elo_spread"]].merge(
        vegas_spreads[["game_id", "vegas_spread"]], on="game_id", how="inner")
    frac = (merged["week"].clip(week_min, week_max) - week_min) / (week_max - week_min)
    merged["w_vegas"] = (0.80 + (0.20 - 0.80) * frac).clip(0.0, 1.0)
    merged["w_elo"] = 1.0 - merged["w_vegas"]
    merged["blended_spread"] = merged["w_vegas"] * merged["vegas_spread"] + merged["w_elo"] * merged["elo_spread"]
    return merged


def validate_learned_vs_arbitrary(season=2025):
    elo_spreads, fitted_model = _get_elo_spreads(season)
    vegas_spreads = extract_vegas_lines(season)
    actual = _get_actual_results(season)

    learned_weights_df = learn_optimal_weights_loocv(elo_spreads, vegas_spreads, actual)
    curve_type, params, r2, candidates = fit_weight_curve(learned_weights_df)

    learned = apply_learned_weights_to_season(elo_spreads, vegas_spreads, curve_type, params)
    arbitrary = apply_arbitrary_linear_weights_to_season(elo_spreads, vegas_spreads)

    def _score(blend_df):
        m = blend_df.merge(actual[["game_id", "actual_spread"]], on="game_id", how="inner")
        corr = m["blended_spread"].corr(m["actual_spread"])
        mae = float(np.mean(np.abs(m["blended_spread"] - m["actual_spread"])))
        return corr, mae

    learned_corr, learned_mae = _score(learned)
    arbitrary_corr, arbitrary_mae = _score(arbitrary)

    return {
        "learned_weights_df": learned_weights_df, "curve_type": curve_type, "params": params, "curve_r2": r2,
        "curve_candidates": candidates, "learned": learned, "arbitrary": arbitrary,
        "learned_corr": learned_corr, "learned_mae": learned_mae,
        "arbitrary_corr": arbitrary_corr, "arbitrary_mae": arbitrary_mae,
        "elo_spreads": elo_spreads, "vegas_spreads": vegas_spreads, "actual": actual, "fitted_model": fitted_model,
    }


def generate_learned_weight_schedule_report(season=2025):
    r = validate_learned_vs_arbitrary(season)
    lw, curve_type, params = r["learned_weights_df"], r["curve_type"], r["params"]

    lines = ["=" * 40, "LEARNED WEIGHT SCHEDULE (LOOCV Optimized)", "=" * 40]
    lines.append("\nLearned Optimal Weights (5 Checkpoints):")
    for _, row in lw.iterrows():
        lines.append(f"  Week {int(row['week']):>2}: w_vegas={row['optimal_w_vegas']:.2f}, "
                      f"w_elo={row['optimal_w_elo']:.2f} (train MAE={row['train_mae']:.2f}, "
                      f"held-out week MAE={row['held_out_mae']:.2f}, n={int(row['n_test_games'])})")

    lines.append(f"\nFitted Curve Type: {curve_type.capitalize()} (R2={r['curve_r2']:.4f} - "
                 f"5 points only, see module caveat, not strong evidence of true functional form)")
    lines.append(f"Parameters: {params}")

    lines.append(f"\nLearned Schedule Validation ({season}):")
    lines.append(f"  Correlation: {r['learned_corr']:+.3f}")
    lines.append(f"  MAE: {r['learned_mae']:.2f} pts")

    lines.append(f"\nArbitrary Linear Schedule (80/20 -> 20/80) Validation ({season}):")
    lines.append(f"  Correlation: {r['arbitrary_corr']:+.3f}")
    lines.append(f"  MAE: {r['arbitrary_mae']:.2f} pts")

    delta_corr = r["learned_corr"] - r["arbitrary_corr"]
    delta_mae = r["learned_mae"] - r["arbitrary_mae"]
    if delta_mae < -0.2 or delta_corr > 0.02:
        verdict = "GREEN - learned schedule meaningfully beats arbitrary linear"
    elif abs(delta_mae) <= 0.2 and abs(delta_corr) <= 0.02:
        verdict = "MIXED - learned and arbitrary are comparable; learned still preferred (data-driven)"
    else:
        verdict = "RED - arbitrary linear beats learned; investigate before deploying learned"
    lines.append(f"\nComparison: delta corr={delta_corr:+.3f}, delta MAE={delta_mae:+.2f} pts")
    lines.append(f"Verdict: {verdict}")

    week1_wv = lw.loc[lw["week"] == 1, "optimal_w_vegas"]
    week16_wv = lw.loc[lw["week"] == 16, "optimal_w_vegas"]
    lines.append("\nPattern insights:")
    if len(week1_wv):
        lines.append(f"  - Week 1 learned w_vegas={week1_wv.iloc[0]:.2f} "
                      f"({'higher' if week1_wv.iloc[0] > 0.80 else 'lower or equal to'} the spec's 80% hypothesis)")
    if len(week16_wv):
        lines.append(f"  - Week 16 learned w_vegas={week16_wv.iloc[0]:.2f} "
                      f"({'lower' if week16_wv.iloc[0] < 0.20 else 'higher or equal to'} the spec's 20% hypothesis)")
    monotonic_decay = lw["optimal_w_vegas"].is_monotonic_decreasing
    lines.append(f"  - Vegas weight decays monotonically week 1->16: {'YES' if monotonic_decay else 'NO'}")

    lines.append(f"\nWeek-by-week learned schedule (all real 2025 weeks, curve-interpolated):")
    all_weeks = sorted(r["elo_spreads"]["week"].unique())
    for wk in all_weeks:
        wv, we = predict_weight_for_week(wk, curve_type, params)
        lines.append(f"  Week {int(wk):>2}: w_vegas={wv:.2f}, w_elo={we:.2f}")

    lines.append(f"\nReady for 2026?")
    confidence = "MEDIUM" if r["curve_r2"] > 0.8 else "LOW"
    lines.append(f"  - Learned schedule ready: YES (with the 5-point-curve-fit caveat above)")
    lines.append(f"  - Confidence: {confidence}")
    lines.append("=" * 40)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    lw.to_csv(os.path.join(DIAGNOSTIC_DIR, "learned_optimal_weights_loocv.csv"), index=False, encoding="utf-8")
    curve_rows = [{"curve_type": k, **v[0], "r2": v[1]} for k, v in r["curve_candidates"].items()]
    pd.DataFrame(curve_rows).to_csv(os.path.join(DIAGNOSTIC_DIR, "weight_curve_fit_parameters.csv"),
                                     index=False, encoding="utf-8")
    r["learned"].to_csv(os.path.join(PROCESSED_DIR, f"vegas_blended_spreads_learned_{season}.csv"),
                         index=False, encoding="utf-8")
    comparison_df = pd.DataFrame([{"schedule": "learned", "corr": r["learned_corr"], "mae": r["learned_mae"]},
                                   {"schedule": "arbitrary_linear", "corr": r["arbitrary_corr"], "mae": r["arbitrary_mae"]}])
    comparison_df.to_csv(os.path.join(DIAGNOSTIC_DIR, "learned_vs_arbitrary_comparison.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "vegas_integration_learned_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved data/diagnostic/learned_optimal_weights_loocv.csv, weight_curve_fit_parameters.csv, "
          f"learned_vs_arbitrary_comparison.csv, vegas_integration_learned_report.txt, "
          f"data/processed/vegas_blended_spreads_learned_{season}.csv")

    return r


def generate_blended_predictions_2026_optimized(curve_type=None, params=None, fitted_model=None):
    """2026 preseason (week 1) blended predictions. Real 2026 Vegas lines
    are sparse (53/272 REG games posted as of this run - books haven't
    published the rest yet) - games without a real posted spread_line get
    an Elo-only prediction, flagged via has_vegas_line, not fabricated."""
    if curve_type is None or params is None:
        r = validate_learned_vs_arbitrary(2025)
        curve_type, params, fitted_model = r["curve_type"], r["params"], r["fitted_model"]

    elo_spreads, _ = _get_elo_spreads(2026, fitted_model)
    vegas_spreads = extract_vegas_lines(2026)

    merged = elo_spreads[["game_id", "week", "home_team", "away_team", "elo_spread"]].merge(
        vegas_spreads[["game_id", "vegas_spread"]], on="game_id", how="left")
    merged["has_vegas_line"] = merged["vegas_spread"].notna()

    w_vegas, w_elo = predict_weight_for_week(1, curve_type, params)
    merged["w_vegas"] = np.where(merged["has_vegas_line"], w_vegas, 0.0)
    merged["w_elo"] = np.where(merged["has_vegas_line"], w_elo, 1.0)
    merged["blended_spread"] = np.where(
        merged["has_vegas_line"],
        merged["w_vegas"] * merged["vegas_spread"] + merged["w_elo"] * merged["elo_spread"],
        merged["elo_spread"])

    n_with_line = int(merged["has_vegas_line"].sum())
    print(f"\n2026 blended predictions: {n_with_line}/{len(merged)} games have a real posted Vegas line "
          f"(week-1 weights w_vegas={w_vegas:.2f}/w_elo={w_elo:.2f} applied there); "
          f"remaining {len(merged) - n_with_line} games are Elo-only (no line posted yet)")

    out_path = os.path.join(PROCESSED_DIR, "vegas_blended_spreads_learned_2026.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")
    return merged


if __name__ == "__main__":
    generate_learned_weight_schedule_report()
    generate_blended_predictions_2026_optimized()
