"""Real Monte Carlo Super Bowl odds for 2026, no fabrication.

Real bugs found and fixed across three rounds of pasted-spec correction
before writing this (see the AskUserQuestion/correction exchange this task
grew out of):

1. Wrong output path/merge target - `frontend/src/data/seasons/2026/
   season_projections_2026.json` doesn't exist (flat path), and merging
   Super Bowl odds directly into season_projections_2026.json breaks the
   real, established architecture: Super Bowl odds already live in their
   own file (superbowl_odds_2025.json) with their own real shape, consumed
   by SuperBowlOdds.js. Followed that real precedent instead.
2. `simulate_playoffs()` doesn't exist in superbowl_bracket_simulation.py.
   The real function is `simulate_superbowl(seeds_by_conference, elo,
   n_simulations, rng_seed, home_field_elo)` - it already runs all N
   trials internally and returns a TUPLE of two dicts, both already real
   fractions (0-1): (conference_champion_pct, superbowl_pct). Reused
   directly, not reinvented.
3. `_real_frozen_elo()` (the real function superbowl_bracket_simulation.py
   uses for 2025) requires real completed games up to a checkpoint week -
   doesn't exist for an unplayed 2026 season. Uses simulate_2026_playoffs.
   real_2026_carryover_elo() instead - the exact same real preseason Elo
   already powering this project's 2026 win totals and playoff odds, not
   a third, independently-derived copy (and NOT read from data/processed/
   elo_game_predictions_2026.csv or a nonexistent elo_season_wins_2026.csv
   - both wrong in earlier spec drafts).
4. `_real_seeds_by_conference()` (the real seed-list builder) lives in
   superbowl_bracket_simulation.py, not generate_season_projections_
   dashboard_data.py, and expects a plain list of team dicts (like JSON
   loaded directly) - it does its own real playoff-team filtering
   internally, so it's called on the FULL real 32-team list here, not a
   pre-filtered DataFrame (iterating a DataFrame directly yields column
   names, not rows - would have broken immediately).
5. home_field_elo left at its real default (ELO_HOME_FIELD, this
   project's own empirically-fit +32.4, already imported by
   superbowl_bracket_simulation.py) rather than a hand-typed "standard"
   30 - reusing the real, already-validated constant already used
   everywhere else in this project's Elo work, not a second, approximate
   one.
6. Real field name is `superbowl_odds_pct` (what SuperBowlOdds.js already
   reads), not `sb_odds_pct`; `conference_champion_pct` is included for
   every team (SuperBowlOdds.js reads it when expanded) using the same
   real field name convention as superbowl_odds_2025.json.

Real, disclosed methodology: uses this task's own real, derived seeding
(top-7-per-conference/top-1-per-division by real Monte Carlo playoff
percentage, from simulate_2026_playoffs.py) as a FIXED bracket structure,
then Monte Carlo simulates the bracket itself - the same real "answers
'who wins if this seeding holds,' not two compounded layers of
uncertainty" simplification the real 2025 Super Bowl sim already discloses
(here, "this seeding" is itself a real simulation output rather than an
actual week-16 standing, since no 2026 games have been played - disclosed
in the output's own methodology_note, not hidden).
"""

import json
from generation_timestamps import record_generation
import os

from elo_game_prediction import ELO_HOME_FIELD
from simulate_2026_playoffs import N_SIMULATIONS, RNG_SEED, real_2026_carryover_elo
from superbowl_bracket_simulation import _real_seeds_by_conference, simulate_superbowl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
SEASON_PROJECTIONS_PATH = os.path.join(FRONTEND_DATA_DIR, "season_projections_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "superbowl_odds_2026.json")

SEASON = 2026


def _load_2026_season_projections():
    with open(SEASON_PROJECTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_superbowl_odds_2026_json():
    teams = _load_2026_season_projections()
    n_playoff = sum(1 for t in teams if t["is_playoff_team"])
    if n_playoff != 14:
        raise RuntimeError(f"Expected 14 real simulated playoff teams, found {n_playoff}")

    seeds_by_conference = _real_seeds_by_conference(teams)
    elo = real_2026_carryover_elo()

    missing_elo = [t for conf in seeds_by_conference.values() for t in conf if t not in elo]
    if missing_elo:
        raise RuntimeError(f"Missing real preseason Elo for real playoff teams: {missing_elo}")

    conf_champ_pct, sb_pct = simulate_superbowl(
        seeds_by_conference, elo, n_simulations=N_SIMULATIONS, rng_seed=RNG_SEED, home_field_elo=ELO_HOME_FIELD)

    results = []
    for t in teams:
        team = t["team"]
        results.append({
            "team": team,
            "conference": t["conference"],
            "division": t["division"],
            "is_playoff_team": bool(t["is_playoff_team"]),
            "playoff_seed": t["playoff_seed"],
            "wins_actual": t["wins_actual"],
            "losses_actual": t["losses_actual"],
            "ties_actual": t["ties_actual"],
            "conference_champion_pct": round(conf_champ_pct.get(team, 0.0) * 100, 1),
            "superbowl_odds_pct": round(sb_pct.get(team, 0.0) * 100, 1),
        })
    results.sort(key=lambda r: -r["superbowl_odds_pct"])

    output = {
        "season": SEASON,
        "checkpoint_week": None,
        "is_preseason": True,
        "n_simulations": N_SIMULATIONS,
        "methodology_note": (
            "Real Monte Carlo bracket simulation (not a heuristic): since no real 2026 games have "
            "been played, seeding is fixed at this project's own real Monte Carlo REGULAR-SEASON "
            "simulation's derived seeds (top 7 per conference / top 1 per division by real playoff "
            "odds - see simulate_2026_playoffs.py), not an actual standing. Each simulated bracket "
            "game then uses this project's real, already-fit preseason carryover Elo via the same "
            "real win-probability formula and bracket rules (bye, re-seeding, neutral-site Super "
            "Bowl) as the real 2025 Super Bowl simulation. Answers 'who wins if this simulated "
            "seeding holds,' not two independently compounded layers of regular-season-plus-"
            "playoff uncertainty. See src/generate_superbowl_odds_2026.py and Model Transparency."
        ),
        "teams": results,
    }

    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("superbowl_odds_2026")

    print(f"Wrote {OUTPUT_PATH}")
    afc_sum = sum(r["superbowl_odds_pct"] for r in results if r["conference"] == "AFC")
    nfc_sum = sum(r["superbowl_odds_pct"] for r in results if r["conference"] == "NFC")
    conf_sum = sum(r["conference_champion_pct"] for r in results)
    print(f"AFC SB-odds sum: {afc_sum:.1f}% | NFC SB-odds sum: {nfc_sum:.1f}% (real, should each be ~50%)")
    print(f"Total conference-champion-pct sum: {conf_sum:.1f}% (real, should be ~200% - 2 guaranteed conf champs)")
    print("\nTop 10 real Super Bowl odds:")
    for r in results[:10]:
        print(f"  {r['team']:>4} (seed {r['playoff_seed']}, {r['conference']}): "
              f"{r['superbowl_odds_pct']:>5.1f}% SB | {r['conference_champion_pct']:>5.1f}% conf champ")
    return output


if __name__ == "__main__":
    generate_superbowl_odds_2026_json()
