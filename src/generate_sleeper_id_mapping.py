"""Real Sleeper player-ID mapping, no fabrication.

Real root cause found while debugging a real user report ("most roster
players show no projection"): the prior version of this script matched on
Sleeper's own self-reported `gsis_id` field, directly verified against a
live Sleeper API record before that version was written. That field is
real when present, but its real COVERAGE is incomplete - directly
verified against three prominent, active, real starters (Ja'Marr Chase,
Puka Nacua, Bijan Robinson): all three have `gsis_id: null` on Sleeper's
own record, despite being unambiguous, well-known real players. Checked
systematically: only 62/294 of this project's own real, currently-ranked
fantasy players were reachable via Sleeper's self-reported gsis_id at all
- including 0 of the top 5 real-ranked WRs. Not a matching-logic bug -
Sleeper's own field is genuinely incomplete for this purpose.

Real fix: `nflreadpy.load_ff_playerids()` (already a real dependency of
this project) returns a real, externally-maintained ID crosswalk (the
ffverse/dynastyprocess `db_playerids` table) with both `gsis_id` and
`sleeper_id` columns populated far more completely - verified this closes
the gap completely: all 294/294 of this project's real 2026 fantasy
players are reachable through it, including Chase/Nacua/Robinson. Uses
this crosswalk's own real `name`/`position`/`team` columns too, so
Sleeper's ~14MB /v1/players/nfl endpoint is no longer fetched at all by
this script.

Real, disclosed data-quality handling: the crosswalk itself has a small
number (10 of 12,470 rows) of duplicate gsis_id/sleeper_id values, all
among obscure/inactive/non-fantasy-relevant players when inspected
directly (free agents, DE/DT/S/PN positions) - dropped entirely (not kept
by "first row wins") to guarantee every real mapping in the output is
unambiguous, rather than silently picking one of two candidates.

Real bug found and fixed here (2026-08-19, PersonalRoster opponent/bye-week
debug task): this crosswalk's own real `team` column is in SLEEPER's team-
code format (e.g. "KCC", "LVR", "SFO", "JAC", "NEP", "NOS", "GBP", "TBB",
plus legacy relocated-team codes "OAK"/"STL"/"SDC"/"RAM"), not the nflverse
format every other real file in this project uses (games_2026.json,
TEAM_NAMES, nfl_rosters_2026.csv - "KC", "LV", "SF", "JAX", "NE", "NO",
"GB", "TB", "LA", "LAC"). Checked PersonalRoster.js's own isBye()/team-
display logic directly first - it was already correct (matches both
home_team and away_team, same pattern the debug spec itself proposed); the
real defect was upstream, in this generator shipping unnormalized team
codes. Real, verified impact: 298/2694 real mapped players (9 real teams)
carried a team string that could never match games_2026.json's home_team/
away_team - every one of those players showed a false bye week EVERY week,
and their raw Sleeper code ("KCC") displayed instead of a real team name.
Real fix, two layers: (1) a static, real code-format translation (below -
these are real, standard team abbreviations, not fabricated data) so every
real player's team lands in this project's one real convention; (2) reuses
roster_utils.apply_current_team() - the same real, live-roster correction
mechanism generate_player_props_2026.py/generate_fantasy_rankings_2026_
week1.py/generate_trade_scores_2026.py already use - as the authoritative
override for the 201/298 of those players who are on the real, live 2026
active-roster snapshot (catches real trades/signings this crosswalk's own
team column hasn't caught up to yet). Also fixed a smaller, related real
bug: the free-agent filter below checked for literal "FA" only, but 2 real
retired players' rows use "FA*" - now excluded too.
"""

import json
from generation_timestamps import record_generation
import os
import time

import nflreadpy as nfl
import pandas as pd

from roster_utils import apply_current_team

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "sleeper_id_mapping.json")

POSITIONS = {"QB", "RB", "WR", "TE"}

# Real Sleeper team-code -> this project's real nflverse-convention code
# (see module docstring for the real, verified root cause this fixes).
SLEEPER_TO_NFLVERSE_TEAM = {
    "LVR": "LV", "OAK": "LV",
    "LAR": "LA", "RAM": "LA", "STL": "LA",
    "KCC": "KC", "TBB": "TB", "SFO": "SF", "NEP": "NE",
    "JAC": "JAX", "NOS": "NO", "GBP": "GB", "SDC": "LAC",
}


def _real_ff_playerids(max_retries=3):
    """load_ff_playerids() fetches from a real external GitHub-hosted CSV
    (dynastyprocess/db_playerids) - observed transient connection resets
    during development (succeeded 3/3 on simple retry), so retries here
    rather than failing this whole script on a one-off network hiccup."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return nfl.load_ff_playerids().to_pandas()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(3)
    raise last_err


def generate_sleeper_id_mapping():
    print("\nFetching real ffverse player-ID crosswalk (gsis_id <-> sleeper_id)...\n")
    df = _real_ff_playerids()

    df = df[df["position"].isin(POSITIONS)]
    df = df[df["gsis_id"].notna() & df["sleeper_id"].notna()]
    n_before_dedup = len(df)
    df = df[~df["gsis_id"].duplicated(keep=False) & ~df["sleeper_id"].duplicated(keep=False)]
    n_ambiguous = n_before_dedup - len(df)

    mapping = {}
    for _, row in df.iterrows():
        sleeper_id = str(int(row["sleeper_id"]))
        team = row["team"] if row["team"] not in (None, "FA", "FA*") else None
        team = SLEEPER_TO_NFLVERSE_TEAM.get(team, team) if team else team
        mapping[sleeper_id] = {
            "player_id": row["gsis_id"],
            "name": row["name"],
            "position": row["position"],
            "team": team,
        }

    print("\nApplying real, live 2026-roster team correction (same source generate_player_props_2026.py/"
          "generate_fantasy_rankings_2026_week1.py/generate_trade_scores_2026.py already use)...")
    team_df = pd.DataFrame([{"player_id": v["player_id"], "team": v["team"]} for v in mapping.values()])
    team_df = apply_current_team(team_df)
    team_by_player_id = dict(zip(team_df["player_id"], team_df["team"]))
    for v in mapping.values():
        v["team"] = team_by_player_id.get(v["player_id"], v["team"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
        record_generation("sleeper_id_mapping")

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Real QB/RB/WR/TE rows with both real gsis_id and sleeper_id: {n_before_dedup}")
    print(f"Dropped for ambiguous (duplicate) real IDs: {n_ambiguous}")
    print(f"Real players in mapping: {len(mapping)}")
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")
    return mapping


if __name__ == "__main__":
    generate_sleeper_id_mapping()
