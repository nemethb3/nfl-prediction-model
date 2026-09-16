"""Real, in-season 2026 Weekly Summary - reuses generate_weekly_summary_
dashboard_data.py's compute_weekly_summary() verbatim (it was already
written generically: it only fills in this_week/top_performers for weeks
whose games have real actual_home_score, and always fills in next_week/
season_context off whatever real data exists - it needed zero logic
changes to run on an in-progress season, only different real input files).

The one real, disclosed adaptation: the playoff_race_note. The 2025
version's text ("Week-16 snapshot") describes that season's own real
generation convention (a fixed week-16 checkpoint, since 2025 is complete
and doesn't need to be regenerated). 2026's season_projections_2026.json is
regenerated fresh every real week via refresh_weekly.py's in-season
recalibration (see recalibrate_2026_elo.py) - a different real fact, so it
gets its own accurate note instead of reusing the 2025 wording.
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

from generate_weekly_summary_dashboard_data import compute_weekly_summary

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "weekly_summary_2026.json")

PLAYOFF_RACE_NOTE_2026 = (
    "Real, current playoff-odds projection (Monte Carlo simulation, recalculated every week off "
    "the latest in-season Elo recalibration) - not a fixed checkpoint like the completed 2025 "
    "season's Week-16 snapshot."
)


def _load_json_as_df(filename):
    with open(os.path.join(FRONTEND_DATA_DIR, filename), encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def generate_weekly_summary_2026_json():
    games_df = _load_json_as_df("games_2026.json")
    fantasy_df = _load_json_as_df("fantasy_rankings_2026.json")
    season_projections_df = _load_json_as_df("season_projections_2026.json")

    completed_weeks = games_df[games_df["actual_home_score"].notna()]["week"].unique()
    current_week = int(max(completed_weeks)) if len(completed_weeks) > 0 else 1

    summary = compute_weekly_summary(
        games_df, fantasy_df, season_projections_df, current_week,
        playoff_race_note=PLAYOFF_RACE_NOTE_2026,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        record_generation("weekly_summary_2026")

    print(f"Generated real 2026 weekly summary -> {OUTPUT_PATH}")
    print(f"current_week (most recent completed): {current_week}")
    print(f"weeks with data: {len(summary['weeks'])}")
    return summary


if __name__ == "__main__":
    generate_weekly_summary_2026_json()
