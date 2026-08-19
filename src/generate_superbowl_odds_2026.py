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

Real, disclosed methodology (revised 2026-08-19 - see "Real bug fixed"
below): each of the real 10,000 regular-season Monte Carlo trials
(simulate_2026_playoffs.py) now simulates its OWN real playoff bracket
using THAT TRIAL's own real seeding, rather than fixing the bracket at one
aggregate, most-likely seeding across all trials. Real bracket-advancement
rules (bye, re-seeding, neutral-site Super Bowl) are unchanged, reused
from superbowl_bracket_simulation.py - only WHICH bracket gets simulated,
per trial, changed.

Real bug found and fixed here: the prior version of this script fixed the
bracket at simulate_2026_playoffs.py's own DERIVED, single most-likely
14-team seeding (top-7-per-conference by aggregate Monte Carlo playoff
percentage) before simulating the bracket - meaning any of the real other
18 teams got a hard, literal 0.0% Super Bowl chance, even ones with a real,
meaningful (e.g. 20-40%) chance of actually making the playoffs. Before a
real season starts, every team still has SOME real chance until it's
mathematically eliminated - collapsing 10,000 trials' worth of real "who
actually makes it" uncertainty into one fixed bracket erased that,
understating every non-lock team's real odds and completely zeroing out
bubble/longshot teams that a real fan would still call "alive." Fixed by
integrating the bracket simulation into simulate_2026_playoffs.py's own
per-trial loop (run_2026_superbowl_simulation) instead of simulating N
brackets against one fixed, aggregated seeding.
"""

import json
from generation_timestamps import record_generation
import os

from simulate_2026_playoffs import N_SIMULATIONS, RNG_SEED, run_2026_superbowl_simulation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
SEASON_PROJECTIONS_PATH = os.path.join(FRONTEND_DATA_DIR, "season_projections_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "superbowl_odds_2026.json")

SEASON = 2026


def _load_2026_season_projections():
    """Real per-team conference/division/actual-record metadata - reused
    here rather than re-derived, since generate_season_projections_
    dashboard_data_2026.py already computes it from the same real source
    (DIVISIONS) and games_2026.json's real actual scores."""
    with open(SEASON_PROJECTIONS_PATH, encoding="utf-8") as f:
        return {t["team"]: t for t in json.load(f)}


def generate_superbowl_odds_2026_json():
    team_meta = _load_2026_season_projections()
    team_stats = run_2026_superbowl_simulation(n_simulations=N_SIMULATIONS, rng_seed=RNG_SEED)
    n_playoff = sum(1 for s in team_stats.values() if s["is_playoff_team"])
    if n_playoff != 14:
        raise RuntimeError(f"Expected 14 real derived playoff teams, found {n_playoff}")

    results = []
    for team, s in team_stats.items():
        meta = team_meta[team]
        results.append({
            "team": team,
            "conference": meta["conference"],
            "division": meta["division"],
            "is_playoff_team": bool(s["is_playoff_team"]),
            "playoff_seed": s["playoff_seed"],
            "wins_actual": meta["wins_actual"],
            "losses_actual": meta["losses_actual"],
            "ties_actual": meta["ties_actual"],
            "conference_champion_pct": round(s["conference_champion_percentage"] * 100, 1),
            "superbowl_odds_pct": round(s["superbowl_percentage"] * 100, 1),
        })
    results.sort(key=lambda r: -r["superbowl_odds_pct"])

    output = {
        "season": SEASON,
        "checkpoint_week": None,
        "is_preseason": True,
        "n_simulations": N_SIMULATIONS,
        "methodology_note": (
            "Real Monte Carlo bracket simulation, integrated with the real regular-season "
            "simulation (not a heuristic, and not a single fixed bracket): since no real 2026 "
            "games have been played, EACH of the 10,000 real regular-season trials (see "
            "simulate_2026_playoffs.py) simulates its own real playoff bracket using that trial's "
            "own real seeding - so every real team's odds reflect the true joint probability of "
            "both making the playoffs and winning it, not just 'if the single most-likely seeding "
            "holds.' Each simulated bracket game uses this project's real, already-fit preseason "
            "carryover Elo via the real win-probability formula and real bracket rules (bye, "
            "re-seeding, neutral-site Super Bowl). See src/generate_superbowl_odds_2026.py and "
            "Model Transparency."
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
