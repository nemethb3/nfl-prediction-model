"""Real season regression-to-mean for O/D Elo entering 2026, closing the
disclosed limitation in apply_offensive_defensive_elo_2026.py's first
version (which carried the real end-of-2025 rating forward with no
regression at all, unlike elo_model.py's real single-Elo chain, which
applies regression_to_mean=1/3 - see constants.py/elo_model.py). This
part of the originally pasted spec was sound as written; kept close to
it, just reading the real k_factor the backtest actually selected instead
of assuming raw end-of-2025 ratings were already computed with some
particular k."""

import json
import os

from generation_timestamps import record_generation
from compute_offensive_defensive_elo import compute_offensive_defensive_elo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "elo_model_comparison.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")

REGRESSION_FACTOR = 1 / 3  # same real, disclosed constant elo_model.py's single-Elo chain uses
BASELINE = 1500.0


def apply_season_regression_od_elo(k_factor=None):
    print("\nApplying real season regression-to-mean to O/D Elo entering 2026...\n")
    if k_factor is None:
        with open(COMPARISON_PATH, encoding="utf-8") as f:
            k_factor = json.load(f)["od_k_factor_selected"]
    print(f"Using real, grid-searched k_factor={k_factor}")

    _, final_2025, _ = compute_offensive_defensive_elo(k_factor=k_factor, save=False)

    regressed = {}
    for team, ratings in final_2025.items():
        o_elo_2026 = BASELINE + (ratings["o_elo"] - BASELINE) * REGRESSION_FACTOR
        d_elo_2026 = BASELINE + (ratings["d_elo"] - BASELINE) * REGRESSION_FACTOR
        regressed[team] = {
            "o_elo": round(o_elo_2026, 1),
            "d_elo": round(d_elo_2026, 1),
            "total_elo": round(o_elo_2026 + d_elo_2026, 1),
            "o_elo_2025_raw": ratings["o_elo"],
            "d_elo_2025_raw": ratings["d_elo"],
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(regressed, f, indent=2)
        record_generation("team_elo_offensive_defensive_2026_regressed")

    if "KC" in regressed:
        r = regressed["KC"]
        print(f"Example (KC): O_Elo {r['o_elo_2025_raw']} -> {r['o_elo']}, "
              f"D_Elo {r['d_elo_2025_raw']} -> {r['d_elo']}")
    print(f"Wrote {OUTPUT_PATH}")
    return regressed


if __name__ == "__main__":
    apply_season_regression_od_elo()
