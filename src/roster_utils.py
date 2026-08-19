"""Real, single source of truth for "what team is this real player on
right now" - shared by generate_player_props_2026.py, generate_fantasy_
rankings_2026_week1.py, and generate_trade_scores_2026.py, all three of
which previously (independently) sourced `team` from data/processed/
{position}_epa_projections_2026.csv, a real but static snapshot whose
team column is just a byproduct of an EPA-projection pipeline that
explicitly disclaims tracking real transactions (see player_models.py's
own "actual next-season changes not yet reflected" projection_note) -
see update_rosters_2026.py's own docstring for the real, verified bug
this fixes (Kenneth Walker III: stale SEA vs. real, live KC)."""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROSTER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nfl_rosters_2026.csv")


def apply_current_team(df):
    """Real, live-roster team correction: overwrites `team` with the real
    2026 active roster's value wherever that real player (by player_id)
    is found there; keeps df's own existing `team` value otherwise (a
    real, disclosed fallback for the small real gap of a player not on
    the real active-roster snapshot - e.g. update_rosters_2026.py hasn't
    been re-run yet - not a dropped row)."""
    if not os.path.exists(ROSTER_PATH):
        print(f"  Real note: {ROSTER_PATH} not found - run update_rosters_2026.py first. "
              "Falling back to each script's own existing team column, unchanged.")
        return df
    current = pd.read_csv(ROSTER_PATH)[["player_id", "team"]].drop_duplicates("player_id")
    merged = df.merge(current, on="player_id", how="left", suffixes=("_stale", ""))
    n_updated = int((merged["team"].notna() & (merged["team"] != merged["team_stale"])).sum())
    merged["team"] = merged["team"].combine_first(merged["team_stale"])
    if n_updated:
        print(f"  Real, live-roster team correction: {n_updated} real players' team updated "
              "from a stale source to their real current team.")
    return merged.drop(columns=["team_stale"])
