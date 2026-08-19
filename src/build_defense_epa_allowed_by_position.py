"""Caches real, prior-season defensive EPA/play allowed by position (QB/RB/
WR/TE), for use as a player-props feature (Fantasy Model Overhaul Phase 1,
Part 1/3 - real "does this defense allow more production to this position"
signal).

Reuses matchup_features.py's already-validated, already-used-in-production
build_defense_epa_by_position_multi_season() (real target_position resolved
via the real player_id -> position crosswalk on receiver_id/rusher_id - see
that module's own docstring for why a literal "position vs team allowed
yards/TDs" table as originally specced doesn't exist and isn't rebuilt here)
rather than re-deriving a second, parallel signal. That function does ONE
chunked pass over the 1.3GB real pbp_2015_2025.csv for all 2015-2025 seasons
at once - this script just runs it once and caches the season-level result,
so build_player_props_signals.py (training) and generate_player_props_2026.py
(2026 scoring) both read a small CSV instead of each re-parsing the full pbp
file.

Real, disclosed limitation carried over from matchup_features.py: for QB,
"target_position" only resolves on rushing plays (receiver_id is never a QB
on a real pass play), so QB's real allowed-EPA figure here reflects defense
quality against QB SCRAMBLES/designed runs only, not against QB passing
overall - a real, narrower signal than the RB/WR/TE versions, not silently
equivalent to "pass defense allowed to QBs." Kept anyway (real, honestly
measured) rather than dropped, and the before/after model comparison this
task produces will show directly whether it helps QB's real models.

Real PRIOR-SEASON convention (same as build_player_props_signals.py's team
pace/red-zone rate): weekly EPA-allowed-per-play is averaged up to a
team-season figure, then lagged +1 season. This avoids the in-season
week-1 cold-start problem a trailing within-season version would have, and
generalizes directly to 2026 scoring (season=2026 resolves to each team's
real 2025 rate) with no separate logic - the same real reason
build_player_props_signals.py already uses this convention for pace/RZ."""

import os

import pandas as pd

from matchup_features import build_defense_epa_by_position_multi_season

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "defense_epa_allowed_by_position_2015_2025.csv")


def build_and_cache():
    print("Computing real defensive EPA/play allowed by position, 2015-2025 "
          "(one chunked pass over pbp_2015_2025.csv)...")
    weekly = build_defense_epa_by_position_multi_season(range(2015, 2026))

    season_avg = weekly.groupby(["team", "season", "position"])["epa_allowed_per_play"].mean().reset_index()

    lagged = season_avg.copy()
    lagged["season"] = lagged["season"] + 1
    lagged = lagged.rename(columns={"epa_allowed_per_play": "opp_epa_allowed_vs_position_prior_season"})

    lagged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUTPUT_PATH} ({len(lagged)} team-season-position rows, "
          f"real seasons {int(season_avg['season'].min())}-{int(season_avg['season'].max())} "
          f"lagged to apply in {int(lagged['season'].min())}-{int(lagged['season'].max())})")
    return lagged


if __name__ == "__main__":
    build_and_cache()
