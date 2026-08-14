"""Orchestrates the real games_2026.json producer/enrichment chain in the
one order that's actually correct, closing the real, disclosed gap found in
the Comprehensive System Audit task: nothing enforced that order in code.

Real, narrow scope - only the two scripts that actually read/write
games_2026.json in a way that creates a data-loss hazard:

1. generate_dashboard_data_2026.py - writes games_2026.json FRESH, no
   merge (see that script's own module docstring). Running it alone after
   step 2 has already run would silently wipe the point-totals fields
   step 2 added.
2. apply_point_totals_2026.py - reads the existing games_2026.json,
   enriches it with predicted_total_value/predicted_total_diff/
   predicted_total_direction/vegas_total, writes it back. Must run AFTER
   step 1, every time step 1 runs.

Real fabrication caught in the originally pasted spec before writing this:
it named a third step, `generate_rookie_scores_2026.py` - doesn't exist
(the real script is score_2026_rookies.py) - and assumed it reads/writes
games_2026.json. Checked: it doesn't reference games_2026.json at all, so
it isn't part of this real hazard and isn't included here. Other real 2026
scripts (generate_fantasy_rankings_2026_week1.py, compute_injury_
consistency_scores_2026.py, generate_season_projections_dashboard_data_
2026.py, generate_superbowl_odds_2026.py, generate_trade_scores_2026.py,
simulate_2026_playoffs.py) are independent of this specific ordering
concern - generate_fantasy_rankings_2026_week1.py does read games_2026.json,
but only for week/home_team/away_team (schedule lookup), never the spread/
totals/O-D fields this hazard is about, so it isn't included either."""

import subprocess
import sys
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "games_2026.json")

# (script, human-readable purpose, real fields this step is responsible for)
PIPELINE = [
    (
        "generate_dashboard_data_2026.py",
        "Core 2026 predictions: O/D Elo spread, CI, win probability, single-Elo comparison fields",
        ["our_spread", "win_prob_home", "ci_low_90", "ci_high_90", "home_o_elo", "home_d_elo",
         "away_o_elo", "away_d_elo", "single_elo_spread"],
    ),
    (
        "apply_point_totals_2026.py",
        "Point-totals enrichment (predicted_total_value + Vegas total diff where posted)",
        ["predicted_total_value"],
    ),
]


def _run_step(script):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", script)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def _validate_fields(required_fields):
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    if len(games) != 272:
        print(f"  FAILED: expected 272 real 2026 REG games, got {len(games)}")
        return False
    missing = {field: sum(1 for g in games if g.get(field) is None) for field in required_fields}
    ok = all(n == 0 for n in missing.values())
    for field, n in missing.items():
        print(f"  {'OK' if n == 0 else 'FAILED'}: {field} ({272 - n}/272 populated)")
    return ok


def orchestrate_2026_pipeline():
    print("\n" + "=" * 70)
    print("2026 GAMES.JSON PIPELINE (real, narrow scope - see module docstring)")
    print("=" * 70)

    for script, purpose, required_fields in PIPELINE:
        print(f"\n[{script}]\n  {purpose}")
        if not _run_step(script):
            print(f"\nFAILED at {script} - stopping (not running later steps against a broken state).")
            return False
        if not _validate_fields(required_fields):
            print(f"\nFAILED validation after {script} - stopping.")
            return False

    print("\n" + "=" * 70)
    print("Pipeline complete: games_2026.json has real O/D Elo predictions,")
    print("single-Elo comparison fields, and point-totals enrichment, 272/272 games.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sys.exit(0 if orchestrate_2026_pipeline() else 1)
