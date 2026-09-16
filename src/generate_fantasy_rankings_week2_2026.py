"""Real Week 2 2026 fantasy projections - the first real per-week-in-
season generator this project has (see refresh_weekly.py's own long-
standing disclosed gap: "there is no real per-week generator for Week 2+
yet").

Real, deliberate design (see the "Week 2 Fantasy Projections" task
decision, 2026-09-16, for the fuller writeup of why a pasted spec's
version was rejected): the spec's generator (a) read a fabricated
frontend/src/data/team_elo_2026.json that doesn't exist, (b) used
`player.get('player_name')` on fantasy_rankings_2026.json rows, whose
real field is `name`, (c) applied an asserted "50% of the Week 1 beat/
miss delta + opponent-defense-Elo/100*0.5" adjustment with no real
derivation, and (d) OVERWROTE fantasy_rankings_2026.json entirely with
Week 2 rows - destroying every real Week 1 actual_ppr this project spent
real effort ingesting (ingest_completed_results_2026.py), which the
frontend's FantasyRankings.js "Game Result vs. Projection" section
depends on.

Real fix, built from this project's own already-established precedent
(fantasy_formula_improvements.py's _trailing_volume / fantasy_rb_
formula.py's _trailing_window - the exact real "week 1 falls back to
prior season, week 2+ uses the trailing mean of the player's own games
so far" rule already used for 2025's real per-week generation, extended
one real year and one real week forward):

  - QB/RB/TE: for a player with a real Week 1 actual_ppr, Week 2's
    projection IS that real Week 1 actual_ppr. This is mathematically
    identical (not an approximation) to re-deriving from real Week 1 box-
    score volume stats through the position's own real PPR formula and
    then trailing-averaging over N=1 games - both this project's real PPR
    formulas (fantasy_formula_improvements._real_ppr, fantasy_rb_
    formula._real_ppr) are linear in their volume-stat inputs, so "trailing
    mean of 1 real game's formula output" == "formula applied to the
    trailing mean of 1 real game's inputs". A player with no real Week 1
    actual_ppr (rookie, inactive, not yet in the real box score) falls
    back to their own Week 1 projection unchanged - the same real
    "no trailing data yet" convention Week 1 itself used for 2025.
  - WR: this project's real WR methodology (wr_epa_volume_formula_2026,
    see generate_fantasy_rankings_2026_week1.py) is a static, EPA-based
    SEASON estimate with no real in-season trailing-update mechanism at
    all - Week 2 WR projections are real, disclosed carryovers of Week 1's
    own value, not silently "updated" with an invented formula.
  - Real, disclosed limitation NOT silently patched: a player zeroed in
    Week 1 for a confirmed-out injury (injury_adjustments_2026.json) who
    is healthy again for Week 2 will show an artificially low trailing-
    rate projection here - no real model exists in this project for "how
    much does a returning player's true rate differ from their one
    injured game," so none was invented (same real philosophy as
    generate_injury_adjustments_2026.py's own module docstring).

Real, additive write: APPENDS Week 2 rows to the SAME fantasy_rankings_
2026.json array (idempotent - replaces any existing week==2 rows rather
than duplicating them) instead of overwriting the file, preserving every
real Week 1 row (and its real actual_ppr) - FantasyRankings.js already
filters players by `p.week === selectedWeek`, so this is the shape the
frontend already expects, not a new one.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
RANKINGS_PATH = DATA_DIR / "fantasy_rankings_2026.json"
GAMES_PATH = DATA_DIR / "games_2026.json"

WEEK = 2


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


def generate_week2_fantasy_rankings():
    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = json.load(f)

    week1_rows = [r for r in rankings if r["week"] == 1]
    other_weeks = [r for r in rankings if r["week"] != 1 and r["week"] != WEEK]
    assert week1_rows, "No real Week 1 rows found - nothing to project Week 2 from."

    opponents = _real_opponents_for_week(WEEK)

    week2_rows = []
    n_trailing = n_carryover = 0
    for r in week1_rows:
        pid = r["id"].split("_w")[0]
        use_trailing = r["position"] != "WR" and r.get("actual_ppr") is not None
        if use_trailing:
            projected_ppr = round(float(r["actual_ppr"]), 1)
            projection_type = "trailing_week1_actual"
            source = (
                f"real Week 1 actual PPR ({r['actual_ppr']:.1f}) used directly as the Week 2 "
                "trailing-rate estimate (N=1 game) - mathematically identical to re-deriving from "
                "real Week 1 box-score volume stats through this position's real, linear PPR "
                "formula (see module docstring)"
            )
            n_trailing += 1
        else:
            projected_ppr = r["projected_ppr"]
            projection_type = r["projection_type"] + "_carried_forward_week2"
            reason = "WR static season methodology has no real in-season update mechanism" \
                if r["position"] == "WR" else "no real Week 1 actual_ppr for this player (no box-score row)"
            source = f"{r['source']} - Week 2 unchanged from Week 1 ({reason})"
            n_carryover += 1

        week2_rows.append({
            "id": f"{pid}_w{WEEK}",
            "week": WEEK,
            "position": r["position"],
            "rank": None,  # filled in below, per position
            "name": r["name"],
            "team": r["team"],
            "projected_ppr": projected_ppr,
            "actual_ppr": None,
            "projection_type": projection_type,
            "opponent": opponents.get(r["team"]),
            "opponent_defense_rank_vs_position": None,
            "recent_form": None,
            "injury_status": "healthy",
            "injury_status_raw": None,
            "source": source,
            "accuracy_tier": None,
            "confidence_tier": r["confidence_tier"],
        })

    by_position = {}
    for row in week2_rows:
        by_position.setdefault(row["position"], []).append(row)
    for pos, rows in by_position.items():
        rows.sort(key=lambda r: r["projected_ppr"], reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1

    combined = week1_rows + other_weeks + week2_rows
    combined.sort(key=lambda r: (r["week"], r["position"], r["rank"]))

    with open(RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Real Week 2 2026 fantasy projections -> {RANKINGS_PATH}")
    print(f"  {n_trailing} players projected from real Week 1 trailing rate, "
          f"{n_carryover} carried forward unchanged (WR / no real Week 1 data)")
    top10 = sorted(week2_rows, key=lambda r: r["projected_ppr"], reverse=True)[:10]
    for i, p in enumerate(top10):
        print(f"    {i + 1}. {p['name']} ({p['position']}, {p['team']}): {p['projected_ppr']:.1f} PPR")
    return combined


if __name__ == "__main__":
    generate_week2_fantasy_rankings()
