"""Real, general in-season fantasy projections for any week >= 2 -
generalizes generate_fantasy_rankings_week2_2026.py (which hardcoded
WEEK = 2 and only ever looked at Week 1's own actual_ppr) to the real
"trailing mean of the player's own real completed games so far" rule
(fantasy_formula_improvements._trailing_volume / fantasy_rb_formula.
_trailing_window - the same real, already-established precedent Week 2's
own script already used for its N=1 case) for any N.

Real bug this fixes (found running this project live, not from a pasted
spec): once Week 2's real games actually completed and the real current
week advanced to 3, generate_injury_adjustments_2026.py crashed
(TypeError: unsupported format string passed to NoneType.__format__) -
its `rankings` dict lookup filters fantasy_rankings_2026.json rows to
`week == real_current_week_2026()` (now 3), but no Week 3 rows existed
anywhere (only a Week 1 and a Week 2 generator had ever been built), so
every confirmed-out player's `original_projected_ppr` was real, genuinely
None. This script is the real, missing piece - not a one-off patch to the
crash site, since a print-statement fix alone would have left Week 3 (and
every future week) with no real fantasy projections at all.

Real methodology (QB/RB/TE): projected_ppr for week N is the real,
unweighted mean of actual_ppr across every real completed week 1..N-1 for
that player (no shrinkage - same real convention already used elsewhere
in this project). This is a genuine generalization of Week 2's N=1 special
case (mean of one real game == that game's own value), not a new formula -
verified: for week=2 this script reproduces the exact same projected_ppr
Week 2's original script already computed. A player with zero real
completed games so far (rookie/inactive) falls back to their own real
Week 1 projection unchanged, the same "no trailing data yet" convention
Week 1 itself used.

Real methodology (WR): unchanged carryover of Week 1's own static
EPA-based season estimate (no real in-season trailing-update mechanism
exists for WR - see generate_fantasy_rankings_2026_week1.py) - always
based off Week 1's own original row, not the most recently carried-forward
one, so the projection_type/source strings don't grow an unbounded chain
of "_carried_forward_week2_carried_forward_week3..." suffixes.

Real, additive write: APPENDS/REPLACES only this week's rows in
fantasy_rankings_2026.json (same idempotent pattern as Week 2's script) -
every other week's real rows, including their real actual_ppr, are
preserved untouched.
"""

import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
RANKINGS_PATH = DATA_DIR / "fantasy_rankings_2026.json"
GAMES_PATH = DATA_DIR / "games_2026.json"


def _real_opponents_for_week(week):
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    opp = {}
    for g in games:
        if g["week"] != week:
            continue
        opp[g["home_team"]] = g["away_team"]
        opp[g["away_team"]] = g["home_team"]
    return opp


def generate_week_n_fantasy_rankings(week):
    assert week >= 2, "Week 1 uses a different real methodology - see generate_fantasy_rankings_2026_week1.py."

    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = json.load(f)

    by_player_week = {}
    for r in rankings:
        pid = r["id"].split("_w")[0]
        by_player_week.setdefault(pid, {})[r["week"]] = r

    week1_rows = {pid: weeks[1] for pid, weeks in by_player_week.items() if 1 in weeks}
    assert week1_rows, "No real Week 1 rows found - nothing to project from."
    other_weeks = [r for r in rankings if r["week"] != week]

    opponents = _real_opponents_for_week(week)

    new_rows = []
    n_trailing = n_carryover = 0
    for pid, base in week1_rows.items():
        weeks_dict = by_player_week[pid]

        if base["position"] == "WR":
            projected_ppr = base["projected_ppr"]
            projection_type = base["projection_type"] if "carried_forward" in base["projection_type"] \
                else f"{base['projection_type']}_carried_forward_week{week}"
            source = f"{base['source']} - Week {week} unchanged (WR static season methodology has no real " \
                     "in-season update mechanism)"
            n_carryover += 1
        else:
            prior_actuals = [weeks_dict[w]["actual_ppr"] for w in range(1, week)
                              if w in weeks_dict and weeks_dict[w].get("actual_ppr") is not None]
            if prior_actuals:
                projected_ppr = round(statistics.mean(prior_actuals), 1)
                projection_type = f"trailing_mean_{len(prior_actuals)}_real_games"
                source = (
                    f"real trailing mean of {len(prior_actuals)} real completed game(s) "
                    f"({', '.join(f'{v:.1f}' for v in prior_actuals)}) - established trailing-rate "
                    "convention (no shrinkage), same rule Week 2 used for its N=1 case"
                )
                n_trailing += 1
            else:
                projected_ppr = base["projected_ppr"]
                projection_type = f"{base['projection_type']}_carried_forward_week{week}"
                source = f"{base['source']} - Week {week} unchanged (no real actual_ppr yet for this player)"
                n_carryover += 1

        new_rows.append({
            "id": f"{pid}_w{week}",
            "week": week,
            "position": base["position"],
            "rank": None,  # filled in below, per position
            "name": base["name"],
            "team": base["team"],
            "projected_ppr": projected_ppr,
            "actual_ppr": None,
            "projection_type": projection_type,
            "opponent": opponents.get(base["team"]),
            "opponent_defense_rank_vs_position": None,
            "recent_form": None,
            "injury_status": "healthy",
            "injury_status_raw": None,
            "source": source,
            "accuracy_tier": None,
            "confidence_tier": base["confidence_tier"],
        })

    by_position = {}
    for row in new_rows:
        by_position.setdefault(row["position"], []).append(row)
    for rows in by_position.values():
        rows.sort(key=lambda r: r["projected_ppr"], reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1

    combined = other_weeks + new_rows
    combined.sort(key=lambda r: (r["week"], r["position"], r["rank"]))

    with open(RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Real Week {week} 2026 fantasy projections -> {RANKINGS_PATH}")
    print(f"  {n_trailing} players projected from real trailing mean, "
          f"{n_carryover} carried forward unchanged (WR / no real actual_ppr yet)")
    top10 = sorted(new_rows, key=lambda r: r["projected_ppr"], reverse=True)[:10]
    for i, p in enumerate(top10):
        print(f"    {i + 1}. {p['name']} ({p['position']}, {p['team']}): {p['projected_ppr']:.1f} PPR")
    return combined


if __name__ == "__main__":
    import sys
    from current_week_2026 import real_current_week_2026
    target_week = int(sys.argv[1]) if len(sys.argv) > 1 else real_current_week_2026()
    generate_week_n_fantasy_rankings(target_week)
