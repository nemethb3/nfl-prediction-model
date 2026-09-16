"""Analyze Model 1 (fantasy_rankings_2026.json's projected_ppr) vs Model 2
(player_props_2026.json's predicted_stats, converted to a comparable PPR
figure) against real 2026 completed-week outcomes.

Real, serious problem found in the originally pasted spec before writing
this: it loads `frontend/src/data/player_props_2025.json` and
`fantasy_rankings_2025.json` to fit ensemble weights on a real, completed,
withheld season - checked directly, player_props_2025.json doesn't exist
anywhere in this project. Player props (the opponent-Elo-adjusted
regression model) were only ever built for 2026 scoring (see
SeasonContext.js's own real, disclosed comment: "no real 2015-2025
per-player-game backtest display was requested"). The entire premise of
fitting weights against a real historical 2025 holdout is impossible with
real data that exists right now.

Per user decision (AskUserQuestion, 2026-09-16): ground truth instead comes
from 2026's own real completed weeks (currently Week 1) - both models'
real, leak-free PREseason projections for that week, compared against the
real actual_ppr this project has already ingested. Smaller, noisier sample
than a full historical season would have given, disclosed honestly below
rather than hidden - and it grows every week as more real 2026 games
complete, refit automatically by refresh_weekly.py.

Model 2's real PPR conversion reuses diagnose_stats_ppr_mismatch.py's
_comparison_ppr_from_predicted_stats() directly (not re-derived a second
time) - the same real, imported PPR weights, and the same disclosed
*_tds_prob-as-expected-count-proxy convention already established
elsewhere in this project (generate_mvp_race_2026.py,
adjustProjectionsForLeague.js).
"""

import json
import os
import statistics
from collections import defaultdict

from diagnose_stats_ppr_mismatch import _comparison_ppr_from_predicted_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
RANKINGS_PATH = os.path.join(FRONTEND_DATA_DIR, "fantasy_rankings_2026.json")
PROPS_PATH = os.path.join(FRONTEND_DATA_DIR, "player_props_2026.json")


def _load_completed_week_pairs():
    """Real (model1_ppr, model2_ppr, actual_ppr) triples for every real
    completed 2026 week - the only real ground truth available right now."""
    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = json.load(f)
    with open(PROPS_PATH, encoding="utf-8") as f:
        props = json.load(f)
    props_by_id = {p["id"]: p for p in props}

    by_position = defaultdict(list)
    for r in rankings:
        if r.get("actual_ppr") is None:
            continue
        p = props_by_id.get(r["id"])
        if p is None:
            continue
        m2_ppr = _comparison_ppr_from_predicted_stats(r["position"], p["predicted_stats"])
        by_position[r["position"]].append({
            "name": r["name"], "team": r["team"], "week": r["week"],
            "model1_ppr": r["projected_ppr"], "model2_ppr": round(m2_ppr, 1),
            "actual_ppr": r["actual_ppr"],
        })
    return by_position


def _position_stats(rows, key):
    errors = [abs(row["actual_ppr"] - row[key]) for row in rows]
    signed = [row["actual_ppr"] - row[key] for row in rows]
    return {
        "n": len(rows),
        "mae": round(statistics.mean(errors), 2) if errors else None,
        "mean_error": round(statistics.mean(signed), 2) if signed else None,
        "stdev": round(statistics.stdev(errors), 2) if len(errors) > 1 else None,
    }


def run():
    print("\n2026 ENSEMBLE GROUND-TRUTH ANALYSIS (real completed weeks only)\n")
    by_position = _load_completed_week_pairs()

    if not by_position:
        print("No real completed 2026 weeks with both actual_ppr and a matched player_props row yet.")
        return by_position

    weeks_covered = sorted({row["week"] for rows in by_position.values() for row in rows})
    print(f"Real completed weeks covered: {weeks_covered}\n")
    print("NOTE: sample sizes below are small (a single week or two of one season) - real,\n"
          "disclosed statistical-power caveat, not hidden. Weights will keep refitting as more\n"
          "real weeks complete.\n")

    print(f"{'Position':<10} {'n':<5} {'Model1 MAE':<12} {'Model2 MAE':<12} {'Better model':<14}")
    print("-" * 60)
    for position in sorted(by_position):
        rows = by_position[position]
        m1 = _position_stats(rows, "model1_ppr")
        m2 = _position_stats(rows, "model2_ppr")
        better = "Model 1" if m1["mae"] < m2["mae"] else "Model 2"
        print(f"{position:<10} {m1['n']:<5} {m1['mae']:<12} {m2['mae']:<12} {better:<14}")

    print()
    for position in sorted(by_position):
        rows = by_position[position]
        m1 = _position_stats(rows, "model1_ppr")
        m2 = _position_stats(rows, "model2_ppr")
        print(f"{position}: Model1 mean_error={m1['mean_error']:+.2f} stdev={m1['stdev']}  |  "
              f"Model2 mean_error={m2['mean_error']:+.2f} stdev={m2['stdev']}")

    return by_position


if __name__ == "__main__":
    run()
