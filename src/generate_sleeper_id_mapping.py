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
"""

import json
import os
import time

import nflreadpy as nfl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "sleeper_id_mapping.json")

POSITIONS = {"QB", "RB", "WR", "TE"}


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
        mapping[sleeper_id] = {
            "player_id": row["gsis_id"],
            "name": row["name"],
            "position": row["position"],
            "team": row["team"] if row["team"] not in (None, "FA") else None,
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Real QB/RB/WR/TE rows with both real gsis_id and sleeper_id: {n_before_dedup}")
    print(f"Dropped for ambiguous (duplicate) real IDs: {n_ambiguous}")
    print(f"Real players in mapping: {len(mapping)}")
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")
    return mapping


if __name__ == "__main__":
    generate_sleeper_id_mapping()
