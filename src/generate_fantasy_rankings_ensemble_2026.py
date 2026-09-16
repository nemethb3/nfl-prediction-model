"""Generate real 2026 ensemble player projections - a new, separate,
disclosed file (frontend/src/data/ensemble_projections_2026.json), not an
in-place overwrite of fantasy_rankings_2026.json's projected_ppr.

Per user decision (AskUserQuestion, 2026-09-16): the spec's approach
(replace projected_ppr everywhere in place) was rejected. Both real
component models' outputs stay exactly as they are today - projected_ppr
(Model 1, trailing-rate) is untouched, player_props_2026.json's
predicted_stats (Model 2, opponent-adjusted) is untouched. This file adds
a third, clearly-labeled real number (ensemble_ppr) alongside them, for
every (fantasy_rankings row, matching player_props row) pair, using the
real, currently-fitted weights from optimize_ensemble_weights_2026.py.

This also directly addresses the real UI gap the prior diagnostic task
found: FantasyRankings.js shows projected_ppr and predicted_stats on the
same card with nothing connecting them. Wiring this file in gives the
frontend a real, disclosed third figure to show alongside both, with a
methodology note - not a silent replacement of either.

Regenerated fresh every run (safe - this file has no additive state of its
own; it's purely derived from the two real source files each time), unlike
fantasy_rankings_2026.json/games_2026.json, which hold real, additive
actual_* results that must never be wiped.
"""

import json
import os
from datetime import datetime, timezone

from diagnose_stats_ppr_mismatch import _comparison_ppr_from_predicted_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RANKINGS_PATH = os.path.join(FRONTEND_DATA_DIR, "fantasy_rankings_2026.json")
PROPS_PATH = os.path.join(FRONTEND_DATA_DIR, "player_props_2026.json")
WEIGHTS_PATH = os.path.join(PROCESSED_DIR, "ensemble_weights_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "ensemble_projections_2026.json")

FALLBACK_WEIGHT = {"model1_weight": 0.5, "model2_weight": 0.5, "n": 0, "fit": "no_weights_file_yet"}


def generate():
    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = json.load(f)
    with open(PROPS_PATH, encoding="utf-8") as f:
        props = json.load(f)
    props_by_id = {p["id"]: p for p in props}

    weights_config = {}
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, encoding="utf-8") as f:
            weights_config = json.load(f).get("weights_by_position", {})
    else:
        print("No real ensemble_weights_2026.json yet - falling back to 0.50/0.50 for every position "
              "(run optimize_ensemble_weights_2026.py once real completed-week data exists).")

    records = []
    n_no_props = 0
    for r in rankings:
        p = props_by_id.get(r["id"])
        if p is None:
            n_no_props += 1
            continue

        m2_ppr = round(_comparison_ppr_from_predicted_stats(r["position"], p["predicted_stats"]), 1)
        w = weights_config.get(r["position"], FALLBACK_WEIGHT)
        ensemble_ppr = round(w["model1_weight"] * r["projected_ppr"] + w["model2_weight"] * m2_ppr, 1)

        records.append({
            "id": r["id"], "week": r["week"], "position": r["position"],
            "team": r["team"], "name": r["name"],
            "model1_ppr": r["projected_ppr"],
            "model2_ppr": m2_ppr,
            "ensemble_ppr": ensemble_ppr,
            "model1_weight": w["model1_weight"], "model2_weight": w["model2_weight"],
            "weight_fit_sample_size": w.get("n", 0),
            "weight_fit_method": w.get("fit", "unknown"),
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_note": (
            "Real, disclosed ensemble of two independently-fitted real models - not a replacement "
            "for either. model1_ppr is fantasy_rankings_2026.json's own real projected_ppr "
            "(trailing-rate/EPA-static, unchanged). model2_ppr is computed here from player_props_"
            "2026.json's real predicted_stats using this project's own real PPR formula constants "
            "(same *_tds_prob-as-expected-count-proxy convention used elsewhere in this project). "
            "ensemble_ppr blends them using per-position weights fit against real 2026 completed-week "
            "outcomes (see ensemble_weights_2026.json's own methodology + sample size per position - "
            "weights fit on very few real weeks so far are real but noisy, disclosed via "
            "weight_fit_sample_size on every record, not hidden)."
        ),
        "n_matched": len(records),
        "n_unmatched_no_props_row": n_no_props,
        "players": records,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(records)} real ensemble projections -> {OUTPUT_PATH}")
    print(f"  {n_no_props} fantasy_rankings rows had no matching player_props row (excluded)")
    return output


if __name__ == "__main__":
    generate()
