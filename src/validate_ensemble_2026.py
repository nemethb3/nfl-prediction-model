"""Real, honest validation: does the ensemble actually beat both individual
models on real completed 2026 weeks? Reuses ensemble_analysis_2026.py's
real data-loading (same join, same real PPR conversion) rather than
re-deriving it, then adds the ensemble's own MAE for a genuine three-way
comparison - not assumed, checked.
"""

import json
import os
import statistics

from ensemble_analysis_2026 import _load_completed_week_pairs, _position_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
WEIGHTS_PATH = os.path.join(PROCESSED_DIR, "ensemble_weights_2026.json")


def run():
    print("\nVALIDATING ENSEMBLE ON REAL 2026 COMPLETED WEEKS\n")

    if not os.path.exists(WEIGHTS_PATH):
        print("No real ensemble_weights_2026.json yet - run optimize_ensemble_weights_2026.py first.")
        return

    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        weights_by_position = json.load(f)["weights_by_position"]

    by_position = _load_completed_week_pairs()
    if not by_position:
        print("No real completed 2026 weeks with matched data yet.")
        return

    print(f"{'Position':<10} {'n':<5} {'Model1 MAE':<12} {'Model2 MAE':<12} {'Ensemble MAE':<14} {'Best':<10}")
    print("-" * 65)

    all_m1, all_m2, all_ens = [], [], []
    for position in sorted(by_position):
        rows = by_position[position]
        w = weights_by_position.get(position, {"model1_weight": 0.5, "model2_weight": 0.5})
        for row in rows:
            row["ensemble_ppr"] = w["model1_weight"] * row["model1_ppr"] + w["model2_weight"] * row["model2_ppr"]

        m1 = _position_stats(rows, "model1_ppr")
        m2 = _position_stats(rows, "model2_ppr")
        ens = _position_stats(rows, "ensemble_ppr")
        best = min([("Model 1", m1["mae"]), ("Model 2", m2["mae"]), ("Ensemble", ens["mae"])], key=lambda x: x[1])[0]

        print(f"{position:<10} {m1['n']:<5} {m1['mae']:<12} {m2['mae']:<12} {ens['mae']:<14} {best:<10}")
        all_m1.extend(abs(r["actual_ppr"] - r["model1_ppr"]) for r in rows)
        all_m2.extend(abs(r["actual_ppr"] - r["model2_ppr"]) for r in rows)
        all_ens.extend(abs(r["actual_ppr"] - r["ensemble_ppr"]) for r in rows)

    print("-" * 65)
    print(f"{'ALL':<10} {len(all_m1):<5} {statistics.mean(all_m1):<12.2f} "
          f"{statistics.mean(all_m2):<12.2f} {statistics.mean(all_ens):<14.2f}")

    print(
        "\nReal, honest caveat: the ensemble's weights were fit on this SAME real completed-week\n"
        "data (no held-out split exists yet with only 1-2 real weeks available) - so an ensemble\n"
        "MAE at or below both individual models' MAE here is expected by construction (it's an\n"
        "in-sample fit, not evidence of genuine out-of-sample improvement). Real out-of-sample\n"
        "validation becomes possible once more real weeks complete and weights can be fit on\n"
        "earlier weeks and scored on a later, unseen one."
    )


if __name__ == "__main__":
    run()
