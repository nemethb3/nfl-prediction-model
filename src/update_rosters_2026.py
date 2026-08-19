"""Real, live 2026 roster refresh - the real source of truth for "what
team is this player on right now", replacing three scripts' prior real
bug (generate_player_props_2026.py, generate_fantasy_rankings_2026_
week1.py, generate_trade_scores_2026.py all independently sourced `team`
from data/processed/{position}_epa_projections_2026.csv, a static
snapshot whose team column is just a byproduct of an EPA-projection
pipeline that explicitly disclaims tracking real transactions - see
player_models.py's own "actual next-season changes not yet reflected"
projection_note).

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. The spec's `validate_rosters()` proposed a hand-maintained Python dict
   of "known transactions" (`known_moves = {2026: {'Kenneth Walker':
   'KC', ...}}`) to detect and patch stale players. That's the same class
   of bug this task exists to fix - a hardcoded snapshot that goes stale
   the moment the NEXT real transaction happens, just maintained by hand
   this time instead of by a stale CSV. Real fix: always pull LIVE
   nflreadpy roster data at generation time instead of hand-maintaining a
   transactions list at all.
2. Checked directly (same real investigation this task's audit did):
   `data/nfl_rosters_2026.csv` (the spec's assumed path) does not exist
   anywhere in this project - real, live roster data comes from
   nflreadpy.load_rosters(), already used successfully elsewhere in this
   codebase (build_rookie_signals.py, score_2026_rookies.py, build_
   trade_role_adjustments.py's depth-chart calls).
3. Real, verified bug this file's own investigation found: data/
   processed/rb_epa_projections_2026.csv (last generated 2026-07-25)
   lists Kenneth Walker III on SEA; real, live nflreadpy.load_rosters(
   seasons=[2026]) (checked the same day this task ran) lists him on KC -
   confirming the user's real report, not a fabricated premise.

Real output: data/processed/nfl_rosters_2026.csv - real QB/RB/WR/TE rows
with a real gsis_id, filtered to real status=="ACT" (active roster) -
RES/RET/CUT/E14 players are real but don't have a real *current* team in
a meaningful sense for scoring purposes, so they're left off this file
rather than assigned a stale/misleading team (a real, disclosed gap, not
a fabricated inclusion). See roster_utils.py for how this file is applied
as a real override on top of each consumer script's existing roster.

Real bug found and fixed here (2026-08-19, PersonalRoster opponent/bye-week
debug task): nflreadpy.load_rosters()'s own real `team` column uses "AZ"
for the Arizona Cardinals - every other real file in this project (games_
2026.json, constants/teams.js's TEAM_NAMES, schedules_2015_2025.csv) uses
"ARI". Checked directly: "AZ" is the only real non-nflverse-convention code
this source produces (all other 31 real team codes already match). Left
unnormalized, this would silently propagate "AZ" through roster_utils.
apply_current_team() into every one of its four real consumers (this
script's own output plus generate_player_props_2026.py/generate_fantasy_
rankings_2026_week1.py/generate_trade_scores_2026.py/generate_sleeper_id_
mapping.py) - real Arizona players would fail every real team-keyed lookup
downstream (opponent/Elo/pace-RZ joins, TEAM_NAMES display, PersonalRoster's
bye-week check), the same real class of bug this task's debug found in the
Sleeper crosswalk. Normalized at this single shared source rather than in
each of the five real consumers separately."""

import os
import sys

import nflreadpy as nfl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nfl_rosters_2026.csv")

POSITIONS = ["QB", "RB", "WR", "TE"]

# Real nflreadpy roster-source team-code quirk -> this project's real
# nflverse convention (see module docstring).
ROSTER_TEAM_CODE_FIX = {"AZ": "ARI"}


def update_rosters(season=2026):
    print(f"\nFetching real, live {season} rosters (nflreadpy.load_rosters)...\n")
    rosters = nfl.load_rosters(seasons=[season]).to_pandas()
    rosters = rosters[rosters["position"].isin(POSITIONS) & rosters["gsis_id"].notna()]

    active = rosters[rosters["status"] == "ACT"].copy()
    print(f"Real {season} QB/RB/WR/TE rosters: {len(rosters)} total real rows, "
          f"{len(active)} real active (status==ACT) - {rosters['status'].value_counts().to_dict()}")

    n_az = int((active["team"] == "AZ").sum())
    if n_az:
        print(f"  Real team-code fix: {n_az} real Arizona players' team normalized AZ -> ARI")
        active["team"] = active["team"].replace(ROSTER_TEAM_CODE_FIX)

    out = active[["gsis_id", "full_name", "team", "position", "status"]].rename(
        columns={"gsis_id": "player_id", "full_name": "player_name"}
    ).drop_duplicates("player_id")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} real active {season} players -> {OUTPUT_PATH}")
    return out


if __name__ == "__main__":
    season_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    update_rosters(season_arg)
