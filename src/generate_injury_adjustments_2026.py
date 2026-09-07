"""Real Week 1 injury-adjustment layer - a new, separate, disclosed file
the frontend reads alongside the model's own output, rather than
mutating player_props_2026.json/fantasy_rankings_2026.json in place.

Real, deliberate scope decisions made after finding serious problems in
the originally pasted spec (see DECISIONS_LOG.md for the fuller writeup;
short version below):

1. The spec's adjustment code targeted field names that don't exist on
   the real files (`predicted_ppr`/`predicted_yards`/`predicted_tds` on
   player_props_2026.json rows - the real predictions live nested under
   `predicted_stats`; a `status`/`role` field on fantasy_rankings rows -
   neither exists). As written, it would have silently added new, unused
   fields to those files without changing anything the app actually
   displays - a real no-op dressed up as a fix.
2. No real, fit model exists anywhere in this project for "how much does
   a backup's projection increase when the starter is ruled out" - the
   spec's 1.5x boost was asserted, not derived. Real, verified backup-role
   data DOES exist (trade_role_adjustments.json's real role_multiplier),
   but it reflects the PRE-injury snap split, not a post-injury
   redistribution - using it to justify a bump here would still be a
   fabrication, just wearing real data as a disguise. Per user decision:
   confirmed-out players are zeroed in this adjustment layer; backups are
   NOT boosted.
3. Same reasoning extends to "Questionable" players: reducing their
   projection by an asserted 40-60% has no real basis either. Questionable
   players are flagged here (so they're visible) but their numeric
   projection is left alone - a real status flag, not a fabricated haircut.
4. Game-level predictions (games_2026.json spreads/win probabilities) are
   NOT adjusted for individual player injuries - this project has no real,
   fit model for "point value of one RB" at the team-Elo level. Disclosed
   here as a real, honest gap rather than an invented adjustment.
5. data/locked_predictions/ (the Week 1 quality-check task's real,
   permanent pre-game snapshot) is NOT touched - overwriting it would
   destroy the one real record of what the model predicted before injury
   news, which is the entire point of it existing.

Real output: frontend/src/data/injury_adjustments_2026.json - wired into
SeasonContext.js (2026 only) like every other real 2026 data file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from get_espn_injuries import ESPNInjuriesFetcher, FLAGGED_STATUSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANKINGS_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "fantasy_rankings_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "injury_adjustments_2026.json"

CONFIRMED_OUT_STATUSES = {"Out", "Injured Reserve"}


def generate_injury_adjustments():
    fetcher = ESPNInjuriesFetcher()
    matched, unmatched = fetcher.match_to_rankings()

    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = {(p["name"], p["team"]): p for p in json.load(f)}

    players = []
    for name, row in matched.items():
        rank_row = rankings.get((name, row["team"]))
        players.append({
            "name": name,
            "team": row["team"],
            "position": row["position"],
            "status": row["status"],
            "confirmed_out": row["status"] in CONFIRMED_OUT_STATUSES,
            "body_part": row["body_part"],
            "detail": row["detail"],
            "return_date": row["return_date"],
            "source_note": row["short_comment"],
            "original_projected_ppr": rank_row["projected_ppr"] if rank_row else None,
            # Real, disclosed: confirmed-out players' effective projection is 0 (a real fact -
            # they won't play); Questionable players' original projection is left as-is (no real
            # model exists to haircut it - see module docstring).
            "adjusted_projected_ppr": 0.0 if row["status"] in CONFIRMED_OUT_STATUSES
            else (rank_row["projected_ppr"] if rank_row else None),
        })

    players.sort(key=lambda p: (not p["confirmed_out"], -(p["original_projected_ppr"] or 0)))

    output = {
        "week": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ESPN live injury report (site.api.espn.com/.../nfl/injuries) - see get_espn_injuries.py",
        "methodology_note": (
            "Real, disclosed scope: confirmed Out/Injured Reserve players are shown with "
            "adjusted_projected_ppr=0.0 (a real fact - they won't play). Questionable players are "
            "flagged but NOT numerically adjusted - no real, fit model exists in this project for "
            "how much a game-status designation should reduce a projection, so none was invented. "
            "Backup players are NOT boosted for the same reason - no real model exists for how much "
            "workload/points a backup absorbs when a starter is out. This is a real, separate "
            "overlay - it does not modify player_props_2026.json, fantasy_rankings_2026.json, "
            "games_2026.json, or data/locked_predictions/."
        ),
        "confirmed_out_count": sum(1 for p in players if p["confirmed_out"]),
        "questionable_count": sum(1 for p in players if not p["confirmed_out"]),
        "unmatched_flagged_count": len(unmatched),
        "players": players,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Real injury adjustments -> {OUTPUT_PATH}")
    print(f"  Confirmed Out/IR (fantasy-relevant): {output['confirmed_out_count']}")
    print(f"  Questionable (fantasy-relevant, flagged only): {output['questionable_count']}")
    for p in players:
        if p["confirmed_out"]:
            print(f"    OUT  {p['name']} ({p['team']}, {p['position']}): was {p['original_projected_ppr']:.1f} PPR -> 0.0")
    return output


if __name__ == "__main__":
    generate_injury_adjustments()
