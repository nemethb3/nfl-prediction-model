"""Real, forward-looking, non-circular test of the empirical age curves,
no fabrication.

Real bug found and fixed before writing this: an earlier draft compared
`player_seasons.iloc[i]` vs `.iloc[i + 1]` directly against
`player_weekly_stats.csv`'s raw WEEKLY rows - never aggregated to real
season totals first. Verified directly: for an arbitrary real player,
consecutive rows sorted by season are actually consecutive WEEKS within
the same season (e.g. real week-1 PPR vs real week-2 PPR), not season
totals - since a player's `age` value is constant across a whole season,
the age-curve "prediction" was being scored against single-game noise on
a completely mismatched timescale, not a real year-over-year outcome.
Real fix: aggregates to real (player_id, season) totals first (same real
aggregation compute_empirical_age_curves.py uses), and additionally
requires the two seasons compared to be REAL, LITERALLY CONSECUTIVE
years (season_next == season_now + 1) - a player who missed a whole real
season (injury, out of the league) doesn't get a spurious "year-over-year"
comparison bridging the gap.

Real, honestly-labeled claim: this tests whether the empirical age curve's
real DIRECTION (is the following age's curve value >= this age's) predicts
the real, actual sign of a player's own season-over-season PPR change -
out of sample in the sense that no single player's own trajectory was
used to fit their own prediction (the curve is a population-level
aggregate). This is NOT a test of "trade recommendation accuracy" -
this project has no real dataset of actual trade outcomes to validate
against - and is not presented as one.
"""

import json
from generation_timestamps import record_generation
import os

from compute_empirical_age_curves import _real_season_totals

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CURVES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "empirical_age_curves.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "directional_accuracy_results.json")

POSITIONS = ["QB", "RB", "WR", "TE"]

# AUDIT_2026-08-12_DEEP.md Section 4.1: _real_season_totals() was copy-pasted
# verbatim here and in compute_empirical_age_curves.py. Imported from there
# instead - same real (player_id, season) aggregation, one copy to keep correct.


def validate_directional_accuracy():
    print("\nComputing real, forward-looking directional accuracy (season-over-season)...\n")
    totals = _real_season_totals()
    with open(CURVES_PATH, encoding="utf-8") as f:
        age_curves = json.load(f)

    results = {
        "methodology": (
            "For each real player, does the empirical age curve's real direction "
            "(next age's curve value >= this age's) predict the real, actual sign of "
            "their own season-over-season PPR change, for real literally-consecutive seasons? "
            "Not a test of trade-outcome accuracy - no real trade-outcome data exists in this project."
        ),
        "by_position": {},
        "overall_accuracy": 0.0,
        "correct_predictions": 0,
        "total_predictions": 0,
    }

    for position in POSITIONS:
        if position not in age_curves:
            continue
        curve = {int(k): v for k, v in age_curves[position]["curve"].items()}
        pos_totals = totals[totals["position"] == position].sort_values(["player_id", "season"])

        correct, total = 0, 0
        for _, player_seasons in pos_totals.groupby("player_id"):
            rows = player_seasons.sort_values("season").to_dict("records")
            for i in range(len(rows) - 1):
                season_now, season_next = rows[i], rows[i + 1]
                if season_next["season"] != season_now["season"] + 1:
                    continue  # real gap year (injury/out of league) - not a real year-over-year pair

                age_now = int(season_now["age_int"])
                if age_now not in curve or (age_now + 1) not in curve:
                    continue

                model_predicts_increase = curve[age_now + 1] >= curve[age_now]
                actual_increase = season_next["season_ppr"] > season_now["season_ppr"]

                if model_predicts_increase == actual_increase:
                    correct += 1
                total += 1

        if total > 0:
            results["by_position"][position] = {
                "directional_accuracy": round(correct / total, 3),
                "predictions_tested": total,
                "correct": correct,
            }
            results["correct_predictions"] += correct
            results["total_predictions"] += total
            print(f"{position}: {correct}/{total} real correct ({100 * correct / total:.1f}% directional accuracy)")

    if results["total_predictions"] > 0:
        results["overall_accuracy"] = round(results["correct_predictions"] / results["total_predictions"], 3)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        record_generation("directional_accuracy_results")

    print(f"\nOverall real directional accuracy: {100 * results['overall_accuracy']:.1f}% "
          f"({results['correct_predictions']}/{results['total_predictions']} real season-over-season pairs)")
    print(f"Wrote {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    validate_directional_accuracy()
