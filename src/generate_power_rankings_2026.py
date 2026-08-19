"""Real 2026 Power Rankings data, no fabrication (SeasonProjections tabbed
redesign follow-up).

Combines three already-real, already-computed sources rather than deriving
a fourth, independent copy of anything:
- Single Elo: real_2026_carryover_elo() (simulate_2026_playoffs.py) - the
  same real preseason carryover Elo already powering this project's 2026
  win totals, playoff simulation, and Super Bowl odds.
- Offensive/defensive Elo: data/processed/team_elo_offensive_defensive_
  2026_regressed.json (compute_offensive_defensive_elo.py /
  apply_season_regression_od_elo.py) - the same real O/D split already
  used by the player-props pipeline's opp_d_elo feature.
- Playoff odds + team/conference/division metadata: frontend/src/data/
  season_projections_2026.json - reused, not re-derived, so Power
  Rankings' playoff-odds column always agrees with the Playoff Picture
  tab's own numbers."""

import json
import os

from generation_timestamps import record_generation
from simulate_2026_playoffs import real_2026_carryover_elo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
OD_ELO_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")
SEASON_PROJECTIONS_PATH = os.path.join(FRONTEND_DATA_DIR, "season_projections_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "power_rankings_2026.json")

SEASON = 2026


def generate_power_rankings_2026():
    with open(SEASON_PROJECTIONS_PATH, encoding="utf-8") as f:
        season_projections = json.load(f)
    with open(OD_ELO_PATH, encoding="utf-8") as f:
        od_elo = json.load(f)
    single_elo = real_2026_carryover_elo()

    missing_od = [t["team"] for t in season_projections if t["team"] not in od_elo]
    if missing_od:
        raise RuntimeError(f"Missing real O/D Elo for real teams: {missing_od}")
    missing_single = [t["team"] for t in season_projections if t["team"] not in single_elo]
    if missing_single:
        raise RuntimeError(f"Missing real single Elo for real teams: {missing_single}")

    rankings = []
    for t in season_projections:
        team = t["team"]
        rankings.append({
            "team": team,
            "team_name": t["team_name"],
            "conference": t["conference"],
            "division": t["division"],
            "single_elo": round(single_elo[team], 1),
            "o_elo": round(od_elo[team]["o_elo"], 1),
            "d_elo": round(od_elo[team]["d_elo"], 1),
            "playoff_percentage": t["playoff_percentage"],
            "is_playoff_team": t["is_playoff_team"],
            "is_division_winner": t["is_division_winner"],
        })
    rankings.sort(key=lambda r: -r["single_elo"])

    if len(rankings) != 32:
        raise RuntimeError(f"Expected 32 real teams, got {len(rankings)}")

    output = {
        "season": SEASON,
        "is_preseason": True,
        "methodology_note": (
            "Real preseason carryover Elo (same source powering this project's 2026 win totals, "
            "playoff simulation, and Super Bowl odds), split into real offensive/defensive Elo "
            "(same source as the player-props opp_d_elo feature) alongside the real single rating. "
            "Playoff odds reused directly from the real Monte Carlo simulation shown in Playoff "
            "Picture - not a separate estimate."
        ),
        "teams": rankings,
    }

    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("power_rankings_2026")

    print(f"Wrote {OUTPUT_PATH} ({len(rankings)} real teams)")
    print("\nTop 5 by single Elo:")
    for r in rankings[:5]:
        print(f"  {r['team']:>4}: single={r['single_elo']:.1f}  o={r['o_elo']:.1f}  d={r['d_elo']:.1f}  "
              f"playoff%={r['playoff_percentage'] * 100:.1f}%")
    return output


if __name__ == "__main__":
    generate_power_rankings_2026()
