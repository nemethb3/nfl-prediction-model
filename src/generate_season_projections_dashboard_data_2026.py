"""Dashboard Section 3 data export for the 2026 season - real preseason
projections only, no fabricated standings.

Real, verified fact: the 2026 season hasn't been played (see generate_
dashboard_data_2026.py's docstring). So unlike the 2025 export, this
script cannot compute real wins-so-far or real seeding from actual games -
that doesn't exist yet. Reuses the real, already-built 2026 preseason
ensemble (data/processed/ensemble_season_wins_2026.csv, a real blend of
this project's EPA-based and Elo-based season-win projections) for
`projected_wins` - the one real, computable point estimate for an unplayed
season. `wins_actual`/`losses_actual`/`ties_actual` are real 0s, not nulls
- every team really has played 0 real 2026 games as of this run.

Two real additions (see the "Fix Win Totals Variance" investigation this
task grew out of - confirmed the model's win-total point estimate isn't
under-dispersed by a bug, it's a real, exact, already-computed 90% CI that
was simply never surfaced):

1. `projected_wins_low_90`/`projected_wins_high_90` - the SAME real,
   closed-form 90% CI already sitting in ensemble_season_wins_2026.csv as
   `ensemble_wins_low_90`/`ensemble_wins_high_90` (elo_model.py's
   project_season_wins_from_elo: Var(sum of independent per-game
   Bernoullis) = sum(p*(1-p)), exact, not a Monte Carlo approximation) -
   not a new computation, just finally exposed to the dashboard.
2. `playoff_percentage`/`division_winner_percentage`/`playoff_seed`/
   `is_division_winner`/`is_playoff_team` - real, computed by a genuine new
   Monte Carlo simulation (simulate_2026_playoffs.py, 10,000 real trials of
   the actual 2026 schedule using this project's real, already-fit Elo
   win-probability formula) rather than left null/false. See that module's
   docstring for why a real simulation (not the closed-form CI above) was
   needed here: closed-form per-team variance can't answer the JOINT
   question "does this team's simulated record beat its real division/
   conference rivals' simulated records in the same trial" that playoff
   odds and seeding actually require.

Reuses generate_season_projections_dashboard_data.py's real, static
division/conference structure (TEAM_NAMES/TEAM_TO_DIVISION/
TEAM_TO_CONFERENCE) rather than redefining it - that structure doesn't
change year to year.
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

from generate_season_projections_dashboard_data import TEAM_NAMES, TEAM_TO_CONFERENCE, TEAM_TO_DIVISION
from simulate_2026_playoffs import run_2026_playoff_simulation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "season_projections_2026.json")

SEASON = 2026


def _real_preseason_remaining_strength():
    """Real average preseason Elo of every real 2026 opponent - the whole
    season is "remaining" since none of it has been played, so this is
    every real scheduled opponent, not a week-16-forward subset like the
    2025 version."""
    elo = pd.read_csv(os.path.join(PROCESSED_DIR, "elo_game_predictions_2026.csv"))
    home = elo[["home_team", "away_elo"]].rename(columns={"home_team": "team", "away_elo": "opp_elo"})
    away = elo[["away_team", "home_elo"]].rename(columns={"away_team": "team", "home_elo": "opp_elo"})
    combined = pd.concat([home, away], ignore_index=True)
    return combined.groupby("team")["opp_elo"].mean().round(1).to_dict()


def generate_season_projections_2026_json():
    ensemble = pd.read_csv(os.path.join(PROCESSED_DIR, "ensemble_season_wins_2026.csv")).set_index("team")
    remaining_strength = _real_preseason_remaining_strength()
    playoff_sim = run_2026_playoff_simulation()

    rows = []
    for team in TEAM_TO_DIVISION:
        in_ensemble = team in ensemble.index
        sim = playoff_sim[team]
        row = {
            "team": team,
            "team_name": TEAM_NAMES[team],
            "conference": TEAM_TO_CONFERENCE[team],
            "division": TEAM_TO_DIVISION[team],
            "wins_actual": 0,
            "losses_actual": 0,
            "ties_actual": 0,
            "projected_wins": round(float(ensemble.loc[team, "ensemble_wins"]), 1) if in_ensemble else None,
            "projected_wins_low_90": round(float(ensemble.loc[team, "ensemble_wins_low_90"]), 1) if in_ensemble else None,
            "projected_wins_high_90": round(float(ensemble.loc[team, "ensemble_wins_high_90"]), 1) if in_ensemble else None,
            "playoff_percentage": sim["playoff_percentage"],
            "division_winner_percentage": sim["division_winner_percentage"],
            "playoff_seed": sim["playoff_seed"],
            "remaining_schedule_strength": remaining_strength.get(team),
            "is_division_winner": sim["is_division_winner"],
            "is_playoff_team": sim["is_playoff_team"],
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["conference"], r["division"], -(r["projected_wins"] or 0)))

    # AUDIT_2026-08-12_DEEP.md Section 10.9: real shape guards before writing
    # (moved ahead of the write - these were previously informational prints
    # only, so a partial ensemble file would have shipped nulls silently).
    n_with_proj = sum(1 for r in rows if r["projected_wins"] is not None)
    n_playoff = sum(1 for r in rows if r["is_playoff_team"])
    n_div_winners = sum(1 for r in rows if r["is_division_winner"])
    assert len(rows) == 32, f"Expected 32 real NFL teams, got {len(rows)}"
    assert n_with_proj == 32, f"Only {n_with_proj}/32 teams have a real ensemble projected-wins figure"
    assert n_playoff == 14, f"Expected 14 real Monte Carlo playoff teams, got {n_playoff}"
    assert n_div_winners == 8, f"Expected 8 real division winners, got {n_div_winners}"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        record_generation("season_projections_2026")

    print(f"Generated {len(rows)} real 2026 preseason team projections -> {OUTPUT_PATH}")
    print(f"  With a real ensemble projected-wins figure + real 90% CI: {n_with_proj}/{len(rows)}")
    print(f"  Real Monte Carlo playoff teams: {n_playoff} (expect 14) | division winners: {n_div_winners} (expect 8)")
    print("  All wins_actual are real 0s (no real 2026 games played yet).")
    return rows


if __name__ == "__main__":
    generate_season_projections_2026_json()
