"""Phase 4, Task 4.2: Compare Win Projections vs. Actual 2025 Results.

Split out of win_projection.py (Master Plan Phase 4 Task 4.2 - see
AUDIT_2026-07-25.md Technical Debt #5). No behavior change, reorganization only.
"""

import os

import numpy as np
import pandas as pd

from epa_to_wins import compute_real_win_pct, TARGET_SEASON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")


def load_2025_actual_results():
    """Real 2025 win totals, reusing compute_real_win_pct (the fixed
    win-counting logic - result is a point differential, not +-1) rather
    than the spec's original buggy version."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    wl = compute_real_win_pct(schedules)
    actual = wl[wl["season"] == TARGET_SEASON][["team", "wins", "games", "win_pct"]].rename(
        columns={"wins": "actual_wins", "win_pct": "actual_win_pct"})

    print(f"\n{TARGET_SEASON} Actual Results:")
    print(actual.sort_values("actual_wins", ascending=False).to_string(index=False))

    os.makedirs(BACKTEST_DIR, exist_ok=True)
    out_path = os.path.join(BACKTEST_DIR, f"actual_wins_{TARGET_SEASON}.csv")
    actual.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")
    return actual


def compare_projections_vs_actual(projections, actual):
    comparison = projections.merge(actual, on="team")
    comparison["residual"] = comparison["actual_wins"] - comparison["projected_wins"]
    comparison["abs_error"] = comparison["residual"].abs()

    mae = comparison["abs_error"].mean()
    rmse = np.sqrt((comparison["residual"] ** 2).mean())
    correlation = comparison["projected_wins"].corr(comparison["actual_wins"])
    r2 = correlation ** 2
    bias = comparison["residual"].mean()

    print(f"\n===== {TARGET_SEASON} BACKTEST RESULTS =====")
    print(f"MAE: {mae:.2f} wins")
    print(f"RMSE: {rmse:.2f} wins")
    print(f"Correlation: {correlation:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Bias (mean residual): {bias:+.2f} wins ({'model overestimated on average' if bias < 0 else 'model underestimated on average' if bias > 0 else 'no systematic bias'})")

    print(f"\nBiggest overestimates (model too high - actual < projected):")
    print(comparison.nsmallest(5, "residual")[["team", "projected_wins", "actual_wins", "residual"]].to_string(index=False))
    print(f"\nBiggest underestimates (model too low - actual > projected):")
    print(comparison.nlargest(5, "residual")[["team", "projected_wins", "actual_wins", "residual"]].to_string(index=False))
    print("===== END BACKTEST RESULTS =====\n")

    out_path = os.path.join(BACKTEST_DIR, f"win_projections_vs_actual_{TARGET_SEASON}.csv")
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

    return comparison, {"mae": mae, "rmse": rmse, "correlation": correlation, "r2": r2, "bias": bias}


def analyze_compression_effect(comparison):
    """Directly follows up on Task 4.1's flagged finding: Phase 3's
    projected epa_diff (and therefore win spread) was ~3.2x more compressed
    than real historical team-season variance. Checks whether that shows up
    here as a systematic pattern (best teams under-projected, worst teams
    over-projected) rather than random noise."""
    real_std = comparison["actual_wins"].std()
    proj_std = comparison["projected_wins"].std()
    print("\n===== COMPRESSION EFFECT CHECK (following up on Task 4.1's flagged finding) =====")
    print(f"Real {TARGET_SEASON} win std dev: {real_std:.2f} | Projected win std dev: {proj_std:.2f} | "
          f"ratio: {real_std / proj_std:.2f}x (Task 4.1 predicted ~3.2x based on epa_diff variance)")

    proj_deviation = comparison["projected_wins"] - comparison["projected_wins"].mean()
    corr_check = comparison["residual"].corr(proj_deviation)
    print(f"corr(residual, projected deviation from mean) = {corr_check:+.3f}")
    print("(positive = confirms the compression pattern: teams projected above-average tend to have ADDITIONAL "
          "positive residual too [actual even better than projected], and teams projected below-average tend to "
          "underperform further - i.e. the model doesn't spread teams out enough. Near-zero or negative would mean "
          "the compression concern didn't actually show up as a systematic error this season.)")
    print("===== END COMPRESSION CHECK =====\n")
    return {"real_std": real_std, "proj_std": proj_std, "compression_ratio": real_std / proj_std, "corr_check": corr_check}


def run_backtest_comparison():
    projections = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{TARGET_SEASON}.csv"))
    actual = load_2025_actual_results()
    comparison, metrics = compare_projections_vs_actual(projections, actual)
    compression = analyze_compression_effect(comparison)
    return comparison, metrics, compression


if __name__ == "__main__":
    run_backtest_comparison()
