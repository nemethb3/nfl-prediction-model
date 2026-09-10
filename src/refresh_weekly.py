"""Real weekly refresh pipeline - re-runs the real, already-existing 2026
generation scripts in the real order their own dependencies require.

Real, serious problems found and fixed vs. the originally pasted spec
before writing this:

1. `update_rosters.py`, `calculate_season_projections_2026.py`,
   `league_settings.py`, `lineup_optimizer.py` were assumed - only the
   first exists, under a different real name (update_rosters_2026.py);
   the season-projections script is really
   generate_season_projections_dashboard_data_2026.py; league settings/
   lineup optimization are real FEATURES but real client-side JS
   (frontend/src/utils/), not Python - there is nothing to call here.
2. Real, load-bearing gap: generate_fantasy_rankings_2026_week1.py is
   hardcoded to Week 1 by name and design (this project is preseason -
   0 real 2026 games played as of when this was written - see
   constants/seasons.js) - there is no real per-week generator for Week
   2+ yet. Calling it under a "Week 3 refresh" label would silently
   re-emit the same Week 1 numbers relabeled as current. Fixed: this
   script only ever refreshes fantasy rankings when week == 1, and
   raises a clear, real error for any other week rather than doing that
   silently.
3. The spec's step list included build_player_props_signals.py/
   train_player_props_models.py/train_td_logistic_models.py - checked
   directly: those TRAIN real models on fixed real 2015-2025 historical
   data. Re-running them weekly would refit against the exact same
   historical data every time (2026's own real in-season results aren't
   part of that training window) - a real no-op that wastes time, not a
   refresh. Only the real SCORING step (generate_player_props_2026.py,
   which re-reads current 2026 rosters/opponents against the
   already-trained models) belongs in a weekly refresh. Same real
   reasoning excludes score_2026_rookies.py (draft-time-only signals -
   draft round/combine data doesn't change week to week).
4. Steps call `sys.executable`, not a bare `python` - this machine's
   PATH `python` resolves to a stale conda install missing nflreadpy/etc
   (verified this session) - sys.executable guarantees the same real
   interpreter this script itself is running under.
5. Final verification reuses audit_project.py's real ProjectAudit
   frontend-data checks instead of re-implementing separate, possibly
   inconsistent ones.

Real order (each step's real, documented input dependency, not an
arbitrary sequence):
  1. update_rosters_2026.py - real live roster pull; must run before
     every consumer below that reads roster_utils.apply_current_team().
  2. espn_odds_orchestrate.py - independent, real Vegas odds.
  3. orchestrate_2026_pipeline.py - real games_2026.json Elo/point-totals
     pipeline (already self-validates - see that script's own docstring).
  4. generate_player_props_2026.py - real per-player-week stat scoring,
     picks up any roster changes from step 1.
  5. generate_fantasy_rankings_2026_week1.py - Week 1 only (see #2 above).
  6. generate_season_projections_dashboard_data_2026.py
  7. generate_superbowl_odds_2026.py - real, seeded from step 6's output.
  8. generate_power_rankings_2026.py
  9. generate_trade_scores_2026.py - real, depends on step 4's output.
  10. generate_mvp_race_2026.py - real, depends on steps 4 and 6.
  11. generate_injury_adjustments_2026.py - real, live ESPN injury overlay
      (added after a real launch-day report: "confirmed out" badges
      weren't propagating into projections everywhere they mattered -
      lineup optimizer, trade analyzer, personal roster). Depends on
      step 5's real fantasy_rankings_2026.json output for the real
      original_projected_ppr comparison it reports.

Real results-ingestion step (added 2026-09-10, once the 2026 season
actually started): ingest_completed_results_2026.py runs FIRST, before
the week-1-only guard below and before any generator - it fills the real,
already-existing-but-null result fields in games_2026.json /
fantasy_rankings_2026.json for every game nflreadpy now has a real final
score for. It is safe for any week (a real no-op when nothing new has
completed), does not depend on the week-1 generators, and does NOT flip
the season-wide SEASON_HAS_RESULTS[2026] flag (see that script's own
docstring for why).
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from audit_project import ProjectAudit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT / "data" / "refresh_logs"


class WeeklyRefresh:
    def __init__(self, week: int):
        self.week = week
        self.timestamp = datetime.now().isoformat()
        self.log = []

    def run(self):
        print(f"\nStarting real weekly refresh for Week {self.week}...")
        print(f"   Timestamp: {self.timestamp}\n")

        # Real completed-game results first - safe for any week, independent
        # of the week-1-only generators, a no-op when nothing new has
        # finished. Runs even when the week guard below refuses the rest.
        self.step("Ingesting completed game results + player box scores",
                  ["ingest_completed_results_2026.py"])

        if self.week != 1:
            print(
                f"REFUSING to run: generate_fantasy_rankings_2026_week1.py only ever "
                f"produces real Week 1 data - there is no real Week {self.week} generator yet "
                "(this project is preseason as of when this script was written). Running it "
                "under a different week label would silently mislabel stale data as current. "
                "See this file's own module docstring for what's needed before this can support "
                "week > 1.\n"
            )
            self.log.append({"step": "week guard", "status": "REFUSED", "time": datetime.now().isoformat()})
            self.log_refresh()
            sys.exit(1)

        self.step("Updating live rosters", ["update_rosters_2026.py"])
        self.step("Collecting ESPN odds", ["espn_odds_orchestrate.py"])
        self.step("Games pipeline (Elo spread/win-prob + point-totals)", ["orchestrate_2026_pipeline.py"])
        self.step("Scoring player props", ["generate_player_props_2026.py"])
        self.step("Regenerating fantasy rankings (Week 1)", ["generate_fantasy_rankings_2026_week1.py"])
        self.step("Refreshing ESPN injury adjustments", ["generate_injury_adjustments_2026.py"])
        self.step("Regenerating season projections", ["generate_season_projections_dashboard_data_2026.py"])
        self.step("Regenerating Super Bowl odds", ["generate_superbowl_odds_2026.py"])
        self.step("Regenerating Power Rankings", ["generate_power_rankings_2026.py"])
        self.step("Regenerating trade scores", ["generate_trade_scores_2026.py"])
        self.step("Regenerating MVP race", ["generate_mvp_race_2026.py"])
        self.step("Verifying data", self.verify_data)

        self.log_refresh()
        self.print_summary()

    def step(self, name, target):
        print(f"  {name}...")
        try:
            if callable(target):
                target()
            else:
                script = target[0]
                result = subprocess.run(
                    [sys.executable, str(SRC_DIR / script)],
                    cwd=SRC_DIR, capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            self.log.append({"step": name, "status": "OK", "time": datetime.now().isoformat()})
            print("     Done")
        except Exception as e:
            self.log.append({"step": name, "status": "ERROR", "error": str(e), "time": datetime.now().isoformat()})
            print(f"     ERROR: {e}")

    def verify_data(self):
        audit = ProjectAudit()
        audit.check_frontend_data()
        if audit.errors:
            raise RuntimeError("; ".join(audit.errors))

    def log_refresh(self):
        metadata = {
            "week": self.week,
            "timestamp": self.timestamp,
            "completed_at": datetime.now().isoformat(),
            "steps": self.log,
        }
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"week_{self.week}_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nLog written: {log_file}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("WEEKLY REFRESH SUMMARY")
        print("=" * 60)
        print(f"Week: {self.week}")
        ok = sum(1 for s in self.log if s["status"] == "OK")
        errors = sum(1 for s in self.log if s["status"] == "ERROR")
        print(f"  {ok} successful")
        if errors:
            print(f"  {errors} errors")
            for s in self.log:
                if s["status"] == "ERROR":
                    print(f"    - {s['step']}: {s['error']}")
        print("=" * 60)
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    week_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    WeeklyRefresh(week_arg).run()
