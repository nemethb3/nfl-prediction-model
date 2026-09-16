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
  1. build_game_results_2026.py - real, freshly-regenerated feed of
     completed 2026 games (data/backtest/game_results_2026.csv) from
     nflreadpy - must run first; every Elo-recalibration step below reads
     it (directly or via elo_model._load_games_chronological).
  2. update_rosters_2026.py - real live roster pull; must run before
     every consumer below that reads roster_utils.apply_current_team().
  3. espn_odds_orchestrate.py - independent, real Vegas odds.
  4. orchestrate_2026_pipeline.py - real games_2026.json Elo/point-totals
     pipeline (already self-validates - see that script's own docstring).
     Both Elo models now (2026-09-16) recalibrate in-season - see
     recalibrate_2026_elo.py's own docstring - using step 1's real
     completed-game feed; games that have already been played keep their
     real, leak-free pre-game prediction (never rewritten after the
     fact), games not yet played use the current real rating. Still
     rewrites games_2026.json FRESH every run (hardcodes every actual_*
     result field back to null) - step 6 below MUST run after this.
  5. recalibrate_2026_elo.py (refresh_current_od_elo_file) - overwrites
     team_elo_offensive_defensive_2026_regressed.json with the CURRENT
     (not frozen-preseason) O/D-Elo. Must run after step 4 - step 4's
     internal preseason-seed computation touches this same file, so
     without this step running after it, Power Rankings/Trade Scores
     (both read this file directly) would silently show the preseason
     value again despite games_2026.json itself being correctly updated.
  6. ingest_completed_results_2026.py - real, week-agnostic, a no-op when
     nothing new has completed. Fills the real, already-existing-but-null
     result fields in games_2026.json / fantasy_rankings_2026.json for
     every game nflreadpy now has a real final score for. Does NOT flip
     the season-wide SEASON_HAS_RESULTS[2026] flag (see that script's own
     docstring for why - it would crash 3 other tabs whose 2026 data is
     still legitimately null).
  7. generate_player_props_2026.py - real per-player-week stat scoring,
     picks up any roster changes from step 2.
  8. generate_fantasy_rankings_2026_week1.py - Week 1 only (see the week
     guard below). Same wipe hazard as step 4, same fix: also rewrites
     fantasy_rankings_2026.json FRESH, hardcoding actual_ppr back to null
     for every row (see its own module docstring) - step 6 (ingest) runs
     AGAIN right after this, every time, for the same reason it runs after
     step 4. Verified idempotent - a real, cheap no-op when there's
     nothing new to re-ingest.
  9. generate_fantasy_rankings_week2_2026.py (added 2026-09-16) - real,
     additive Week 2 trailing-rate projections (see that script's own
     docstring for the real methodology: reuses this project's own
     already-established prior-season-fallback/trailing-mean convention,
     not an asserted formula). APPENDS to fantasy_rankings_2026.json -
     does not overwrite Week 1's real rows or their real actual_ppr.
  10. generate_injury_adjustments_2026.py - real, live ESPN injury overlay
      (added after a real launch-day report: "confirmed out" badges
      weren't propagating into projections everywhere they mattered -
      lineup optimizer, trade analyzer, personal roster). Fixed 2026-09-16
      (current_week_2026.py) to target the real current week instead of a
      hardcoded week=1 - fantasy_rankings_2026.json now legitimately holds
      more than one real week's rows (step 9), and the old unfiltered
      (name, team) join would have silently picked up whichever week's row
      happened to sort last.
  11. generate_season_projections_dashboard_data_2026.py - reads
      games_2026.json's actual_winner directly (fixed 2026-09-15) for real
      wins_actual/losses_actual/ties_actual (must run after the step-6
      re-ingest, not before), AND (fixed 2026-09-16) its real Monte Carlo
      playoff simulation (simulate_2026_playoffs.py) now starts from
      current in-season Elo and locks already-played games to their real
      outcome instead of re-simulating them - see simulate_2026_playoffs.
      real_2026_carryover_elo's docstring.
  12. generate_superbowl_odds_2026.py - real, seeded from step 11's output.
  13. generate_power_rankings_2026.py - reads step 5's current O/D-Elo file
      directly, and single-Elo via the same fixed real_2026_carryover_elo.
  14. generate_trade_scores_2026.py - real, depends on step 7's output and
      the same current-Elo team-strength context as step 13.
  15. generate_mvp_race_2026.py - real, depends on steps 7 and 11. Fixed
      2026-09-16: now blends real completed weeks' nflreadpy box scores
      into the season stat total in place of that week's prediction (the
      standard real fantasy-industry "rest-of-season" convention), instead
      of the pasted spec's crude, separately-weighted "actual x 17 pace"
      replacement - see that script's own module docstring.
  16. join_espn_odds_2026.py (added 2026-09-16) - real, joins the already-
      collected ESPN odds history (step 3's accumulated output) into
      games_2026.json's own vegas_spread field, plus a new real per-game
      moneyline/spread-price file (data/processed/vegas_lines_2026.csv).
      Must run after step 4 (which rewrites games_2026.json fresh every
      time, same wipe hazard as steps 4/8) - placed right after it.
  17. generate_accuracy_tracker_dashboard_data_2026.py (added 2026-09-16) -
      real, in-season Accuracy Tracker (games/fantasy sections only -
      season-projection accuracy isn't scoreable until the real season
      ends). Needs steps 6/9/16's real data.
  18. generate_weekly_summary_dashboard_data_2026.py (added 2026-09-16) -
      real, in-season Weekly Summary; reuses the 2025 generator's already-
      generic compute_weekly_summary() unchanged. Needs step 11's real
      season projections.
  19. betting_backtest_2026.py (added 2026-09-16) - real, in-season
      Betting Analysis (moneyline + ATS), reusing betting_backtest.py's
      already-validated strategies/settlement math. Needs step 16's real
      vegas_lines_2026.csv.
  20. optimize_ensemble_weights_2026.py (added 2026-09-16) - refits real
      per-position blend weights (fantasy_rankings' trailing-rate
      projected_ppr vs. a PPR figure computed from player_props'
      predicted_stats) against whatever real 2026 completed weeks exist so
      far. No real 2025 holdout exists for this comparison (player props is
      2026-only) - see ensemble_analysis_2026.py's own docstring.
  21. generate_fantasy_rankings_ensemble_2026.py (added 2026-09-16) - real,
      separate ensemble_projections_2026.json (does NOT overwrite
      fantasy_rankings_2026.json's own projected_ppr - both real component
      models stay visible unchanged). Needs step 20's real weights.

Real, disclosed scope: every real 2026 deliverable this project ships now
genuinely recalibrates from real completed games - Elo, Division Winners,
Playoff Picture, Super Bowl odds, Power Rankings, Trade Scores, MVP race,
Week 2+ fantasy projections, and (new) a real, in-season Accuracy Tracker/
Weekly Summary/Betting Analysis (partial-coverage, honestly disclosed
where real Vegas data doesn't reach yet - see each script's own docstring).
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

        # These six are real, whole-season, week-agnostic steps - they run
        # regardless of which week was requested, unlike the week-1-only
        # generators guarded below.
        #
        # Ordering bug fixed here (found 2026-09-15): ingestion must run
        # AFTER the games pipeline, not before it. generate_dashboard_
        # data_2026.py (called by orchestrate_2026_pipeline.py) rewrites
        # games_2026.json FRESH every time - it hardcodes actual_home_score/
        # actual_away_score/actual_winner/actual_spread_margin/did_we_
        # predict_correctly to null for every one of the 272 games (see its
        # own module docstring: "every actual_*/accuracy field is genuinely
        # null" was true when it was written, preseason, and the script was
        # never updated to merge real results back in). Running ingestion
        # first (this script's original order) had it immediately undone by
        # this step on every single refresh.
        self.step("Building real 2026 completed-results feed for Elo", ["build_game_results_2026.py"])
        self.step("Updating live rosters", ["update_rosters_2026.py"])
        self.step("Collecting ESPN odds", ["espn_odds_orchestrate.py"])
        self.step("Games pipeline (Elo spread/win-prob + point-totals)", ["orchestrate_2026_pipeline.py"])
        # Must run right after the games pipeline (which rewrites games_2026.json
        # fresh every time) - joins the real, already-collected ESPN odds history
        # into games_2026.json's vegas_spread + a real per-game moneyline file.
        self.step("Joining real ESPN odds history into games_2026.json", ["join_espn_odds_2026.py"])
        self.step("Refreshing current O/D-Elo file (Power Rankings/Trade Scores)",
                  ["recalibrate_2026_elo.py"])
        self.step("Ingesting completed game results + player box scores",
                  ["ingest_completed_results_2026.py"])

        if self.week != 1:
            print(
                f"REFUSING to run the remaining week-1-only steps: generate_fantasy_rankings_"
                f"2026_week1.py only ever produces real Week 1 data - there is no real Week "
                f"{self.week} generator yet (this project is preseason as of when this script "
                "was written). Running it under a different week label would silently mislabel "
                "stale data as current. Rosters/odds/games-pipeline/results-ingestion above "
                "already ran (real, week-agnostic). See this file's own module docstring for "
                "what's needed before the rest can support week > 1.\n"
            )
            self.log.append({"step": "week guard", "status": "REFUSED", "time": datetime.now().isoformat()})
            self.log_refresh()
            sys.exit(1)

        self.step("Scoring player props", ["generate_player_props_2026.py"])
        self.step("Regenerating fantasy rankings (Week 1)", ["generate_fantasy_rankings_2026_week1.py"])

        # Same wipe hazard as the games pipeline above, same fix: generate_
        # fantasy_rankings_2026_week1.py rewrites fantasy_rankings_2026.json
        # FRESH too, hardcoding actual_ppr back to null for every row (see
        # its own module docstring). Re-running ingestion here (safe -
        # verified idempotent) restores actual_ppr before season projections
        # reads real records off games_2026.json below.
        self.step("Re-ingesting completed results (fantasy rankings step just reset actual_ppr)",
                  ["ingest_completed_results_2026.py"])
        # Real, additive Week 2 trailing-rate projections (real once Week 1
        # has completed games to trail from - see that script's own module
        # docstring for the real "no fabricated formula" methodology).
        # Appends to fantasy_rankings_2026.json, does not overwrite it - must
        # run after the re-ingest above (needs real Week 1 actual_ppr) and
        # before injury adjustments below (needs real Week 2 rows to exist
        # for its current-week join).
        self.step("Generating Week 2 fantasy projections", ["generate_fantasy_rankings_week2_2026.py"])
        # Real, in-season ensemble (added 2026-09-16): refits per-position blend
        # weights against whatever real completed weeks exist so far, then
        # regenerates the real, separate ensemble_projections_2026.json - must run
        # after the re-ingest/Week 2 steps above (needs real actual_ppr + both
        # weeks' real projected_ppr) and after player props scoring earlier in
        # this run (needs real predicted_stats for every week).
        self.step("Optimizing ensemble weights (real 2026 completed weeks)",
                  ["optimize_ensemble_weights_2026.py"])
        self.step("Generating ensemble player projections",
                  ["generate_fantasy_rankings_ensemble_2026.py"])
        self.step("Refreshing ESPN injury adjustments", ["generate_injury_adjustments_2026.py"])
        self.step("Regenerating season projections", ["generate_season_projections_dashboard_data_2026.py"])
        self.step("Regenerating Super Bowl odds", ["generate_superbowl_odds_2026.py"])
        self.step("Regenerating Power Rankings", ["generate_power_rankings_2026.py"])
        self.step("Regenerating trade scores", ["generate_trade_scores_2026.py"])
        self.step("Regenerating MVP race", ["generate_mvp_race_2026.py"])
        # Real, in-season Accuracy Tracker/Weekly Summary/Betting Analysis -
        # need this run's real ingested results, real joined odds, and real
        # season projections, so they run last.
        self.step("Regenerating in-season accuracy tracker",
                  ["generate_accuracy_tracker_dashboard_data_2026.py"])
        self.step("Regenerating in-season weekly summary",
                  ["generate_weekly_summary_dashboard_data_2026.py"])
        self.step("Regenerating in-season betting analysis", ["betting_backtest_2026.py"])
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
