"""Real "Bounce-Back Candidate" flags - a new, separate, disclosed file
(frontend/src/data/bounceback_warnings_2026.json), not a mutation of
fantasy_rankings_2026.json in place (same real precedent as ensemble_
projections_2026.json/injury_adjustments_2026.json this session).

Real, serious problem found in the originally pasted spec before writing
this: its mark_bounce_back_candidates() reads `player.get('actual_ppr_
week1', 0)` / `player.get('projected_ppr_week1', 0)` off the SAME row -
verified directly, no such fields exist anywhere on fantasy_rankings_2026.
json (each row is per-week; a Week 2 row has no suffixed reference back to
that player's own Week 1 row at all). As written, both would always
default to 0, `week1_projected > 0` would always be False, and NO player
would ever be flagged - the same silent-no-op fabrication pattern already
found twice this session. Fixed to join each week-N row to that SAME
player's own week-(N-1) row by player_id.

Real, corrected example (the spec's own worked example was simply wrong -
checked directly): Josh Allen's real Week 1 was proj=23.0/actual=35.7 - he
OVERperformed by +12.7, not underperformed. He's the real poster child for
the OPPOSITE problem (found in the earlier stats<->PPR diagnostic task),
not a bounce-back candidate at all.

Real, disclosed nuance the spec didn't address: a real actual_ppr of
exactly 0.0 is ambiguous - it could mean a real bad game while playing, OR
the player didn't play at all (injury/inactive/bye), which have very
different implications for "due to bounce back." fantasy_rankings_2026.
json's own injury_status field is hardcoded "healthy" for everyone at
Week-1 generation time (real 2026 injury data doesn't cover preseason -
see generate_fantasy_rankings_2026_week1.py's own docstring), so it can't
be used to disambiguate after the fact. Flagged with a distinct, more
cautious reason string rather than asserting a confident "bounce-back"
read on a game the player may not have actually played.

Real threshold used exactly as specified in the task (underperformed
Week-(N-1) projection by more than 25%) - not silently changed, but also
not asserted as a validated cutoff; disclosed as the requested convention.
"""

import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
RANKINGS_PATH = os.path.join(FRONTEND_DATA_DIR, "fantasy_rankings_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "bounceback_warnings_2026.json")

UNDERPERFORMANCE_THRESHOLD_PCT = 0.75  # actual < projected * 0.75 => flagged


def _player_id(row_id):
    # Real id format: "{player_id}_w{week}" (established this session,
    # e.g. generate_player_props_2026.py) - strip the week suffix.
    return row_id.rsplit("_w", 1)[0]


def generate():
    with open(RANKINGS_PATH, encoding="utf-8") as f:
        rankings = json.load(f)

    by_player_week = {}
    for r in rankings:
        by_player_week[(_player_id(r["id"]), r["week"])] = r

    warnings = {}
    for r in rankings:
        pid, week = _player_id(r["id"]), r["week"]
        if week < 2:
            continue
        prev = by_player_week.get((pid, week - 1))
        if prev is None or prev.get("actual_ppr") is None or not prev.get("projected_ppr"):
            continue

        prev_proj, prev_actual = prev["projected_ppr"], prev["actual_ppr"]
        if prev_actual >= prev_proj * UNDERPERFORMANCE_THRESHOLD_PCT:
            continue

        underperformance = round(prev_proj - prev_actual, 1)
        if prev_actual == 0.0:
            reason = (
                f"Recorded 0.0 PPR in Week {week - 1} (projected {prev_proj:.1f}) - real, disclosed "
                "ambiguity: this could be a true zero-stat game OR the player not actually playing "
                "(injury/inactive/bye), which this data can't distinguish after the fact. Treat this "
                "flag with extra caution."
            )
        else:
            reason = f"Underperformed Week {week - 1} projection by {underperformance:.1f} PPR " \
                      f"({prev_proj:.1f} projected vs. {prev_actual:.1f} actual)."

        warnings[r["id"]] = {
            "id": r["id"], "name": r["name"], "position": r["position"], "team": r["team"], "week": week,
            "flagged": True,
            "reason": reason,
            "prior_week": week - 1, "prior_week_projected": prev_proj, "prior_week_actual": prev_actual,
            "underperformance_ppr": underperformance,
            "ambiguous_zero_actual": prev_actual == 0.0,
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_note": (
            f"Real, disclosed heuristic: flags a player whose prior-week real actual_ppr fell below "
            f"{int(UNDERPERFORMANCE_THRESHOLD_PCT * 100)}% of that week's own real projected_ppr (the "
            "threshold requested for this feature, not a separately validated cutoff). A real actual_ppr "
            "of exactly 0.0 is flagged with a more cautious reason - it may mean the player didn't play "
            "at all rather than underperformed while playing (fantasy_rankings_2026.json's own "
            "injury_status is hardcoded 'healthy' at Week 1 generation time, so it can't disambiguate "
            "this after the fact - see module docstring)."
        ),
        "threshold_pct": UNDERPERFORMANCE_THRESHOLD_PCT,
        "n_flagged": len(warnings),
        "players": list(warnings.values()),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(warnings)} real bounce-back warnings -> {OUTPUT_PATH}")
    for w in sorted(warnings.values(), key=lambda x: x["underperformance_ppr"], reverse=True)[:10]:
        print(f"  {w['name']:20s} {w['position']:3s} {w['team']:3s} wk{w['week']}: {w['reason']}")
    return output


if __name__ == "__main__":
    generate()
