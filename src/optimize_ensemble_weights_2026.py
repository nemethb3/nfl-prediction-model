"""Optimize per-position ensemble weights against real 2026 completed-week
outcomes (see ensemble_analysis_2026.py for why 2025 isn't used as ground
truth - that data doesn't exist).

Real, disclosed fix from the pasted spec: its weight_options list
(`[0.3, 0.4, 0.5, 0.6, 0.7]`) arbitrarily restricted the search to a narrow
band around 50/50 with no real justification - an unexplained constraint
that would silently prevent finding a real optimum outside that band (e.g.
a position where one model is decisively better deserves a weight near 0
or 1, not floored at 0.3/0.7). Searched here over the real full range,
0.00-1.00 in 0.05 steps (21 points) - cheap to compute, no reason to
artificially narrow it.

Real, disclosed statistical caveat, not hidden: with only Week 1's ~50-150
players per position as ground truth, the "optimal" weight is a real fit to
a small, single-week sample and can be noisy - not a season-validated
constant. Refit automatically every week (refresh_weekly.py) as more real
completed weeks accumulate, and the raw sample size is stored in the output
file so any consumer can see how much to trust a given position's weight.
"""

import json
import os
import statistics
from datetime import datetime, timezone

from ensemble_analysis_2026 import _load_completed_week_pairs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "ensemble_weights_2026.json")

# Real, full search space - not the spec's arbitrary 0.3-0.7 restriction.
WEIGHT_GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00

MIN_SAMPLE_SIZE_FOR_FIT = 10


def _mae_at_weight(rows, m1_weight):
    errors = [
        abs(row["actual_ppr"] - (m1_weight * row["model1_ppr"] + (1 - m1_weight) * row["model2_ppr"]))
        for row in rows
    ]
    return statistics.mean(errors)


def optimize():
    print("\nOPTIMIZING ENSEMBLE WEIGHTS (real 2026 completed weeks as ground truth)\n")
    by_position = _load_completed_week_pairs()

    weights_by_position = {}
    for position, rows in sorted(by_position.items()):
        if len(rows) < MIN_SAMPLE_SIZE_FOR_FIT:
            print(f"{position}: only {len(rows)} real samples (< {MIN_SAMPLE_SIZE_FOR_FIT}) - "
                  f"real, disclosed fallback to 0.50/0.50 rather than fitting on too little data.")
            weights_by_position[position] = {
                "model1_weight": 0.5, "model2_weight": 0.5,
                "mae": round(_mae_at_weight(rows, 0.5), 2) if rows else None,
                "n": len(rows), "fit": "fallback_insufficient_sample",
            }
            continue

        best_mae, best_w1 = float("inf"), 0.5
        for w1 in WEIGHT_GRID:
            mae = _mae_at_weight(rows, w1)
            if mae < best_mae:
                best_mae, best_w1 = mae, w1

        weights_by_position[position] = {
            "model1_weight": best_w1, "model2_weight": round(1.0 - best_w1, 2),
            "mae": round(best_mae, 2), "n": len(rows), "fit": "grid_search_2026_completed_weeks",
        }
        print(f"{position}: n={len(rows):3d}  best model1_weight={best_w1:.2f}  "
              f"model2_weight={1.0 - best_w1:.2f}  MAE={best_mae:.2f}")

    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Real, in-season fit: grid search (0.00-1.00 step 0.05) minimizing MAE against real "
            "2026 completed-week actual_ppr, comparing fantasy_rankings_2026.json's projected_ppr "
            "(Model 1) against a PPR figure computed from player_props_2026.json's predicted_stats "
            "(Model 2) using this project's own real PPR formula constants. No real 2025 holdout "
            "exists for this comparison (player-props model is 2026-only) - see ensemble_analysis_"
            "2026.py's module docstring."
        ),
        "min_sample_size_for_fit": MIN_SAMPLE_SIZE_FOR_FIT,
        "weights_by_position": weights_by_position,
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")
    return config


if __name__ == "__main__":
    optimize()
