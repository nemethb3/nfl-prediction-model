"""Comprehensive Week 1 final pre-game check - runs the real roster/
injury/prediction/tracking checks above and locks a real snapshot.

Real corrections vs. the originally pasted spec: subprocess calls use
sys.executable (not a bare `python` - this machine's PATH `python`
resolves to a stale conda env missing nflreadpy/etc, verified in an
earlier task this session), paths are Path(__file__)-based (robust to
cwd), and field names match the real schemas verified directly (see
verify_week1_rosters.py/verify_week1_injuries.py's own docstrings for
the specific real corrections made there).
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_espn_injuries import ESPNInjuriesFetcher  # noqa: E402
from verify_week1_injuries import CRITICAL_PLAYERS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"


class Week1FinalCheck:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.checks_passed = 0
        self.checks_failed = 0
        self.critical_flags = []

    def run(self):
        print("\n" + "=" * 60)
        print("WEEK 1 FINAL PRE-GAME QUALITY CHECK")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 60 + "\n")

        print("1) ROSTER VERIFICATION")
        self._run_script("verify_week1_rosters.py")

        print("\n2) INJURY DATA CHECK (real ESPN report + this project's own known gap)")
        self._run_script("verify_week1_injuries.py")
        self.check_critical_flags()

        print("\n3) DATA INTEGRITY CHECK")
        self.check_data_integrity()

        print("\n4) PREDICTION SANITY CHECK")
        self.validate_predictions()

        print("\n5) TRACKING INFRASTRUCTURE")
        self.verify_tracking_setup()

        print("\n6) LOCKING PREDICTIONS")
        self._run_script("lock_week1_predictions.py")

        self.print_final_summary()

    def _run_script(self, name):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / name)],
            cwd=PROJECT_ROOT / "src", capture_output=False,
        )
        if result.returncode != 0:
            self.checks_failed += 1

    def check_critical_flags(self):
        # Real, direct re-check here rather than trusting the sub-script's
        # exit code/stdout - verify_week1_injuries.py exits 0 on a
        # successful RUN even when it found real flags to review (a real
        # flag is informational, not a script failure), so this project's
        # own final summary would otherwise say "no blocking issues" right
        # above real, un-surfaced Mahomes/McCaffrey-type flags. Caught and
        # fixed live this task - see the git history for the before/after.
        try:
            matched, _ = ESPNInjuriesFetcher().match_to_rankings()
            self.critical_flags = [
                (name, row) for name, row in matched.items() if name in CRITICAL_PLAYERS
            ]
        except Exception as e:
            print(f"    Could not re-check critical flags: {e}")

    def check_data_integrity(self):
        required_files = {
            "games": DATA_DIR / "games_2026.json",
            "players": DATA_DIR / "player_props_2026.json",
            "rankings": DATA_DIR / "fantasy_rankings_2026.json",
            "rosters": PROJECT_ROOT / "data" / "processed" / "nfl_rosters_2026.csv",
            "sleeper_mapping": DATA_DIR / "sleeper_id_mapping.json",
            "power_rankings": DATA_DIR / "power_rankings_2026.json",
            "sb_odds": DATA_DIR / "superbowl_odds_2026.json",
        }
        for name, path in required_files.items():
            if path.exists():
                print(f"    OK  {name}: {path.stat().st_size:,} bytes")
                self.checks_passed += 1
            else:
                print(f"    XX  {name}: MISSING")
                self.checks_failed += 1

    def validate_predictions(self):
        games_path = DATA_DIR / "games_2026.json"
        if not games_path.exists():
            print("    XX  games_2026.json missing")
            self.checks_failed += 1
            return
        try:
            with open(games_path, encoding="utf-8") as f:
                games = json.load(f)
            week1_games = [g for g in games if g.get("week") == 1]
            valid = sum(
                1 for g in week1_games
                if all(f in g for f in ["home_team", "away_team", "our_spread", "win_prob_home"])
                and -30 <= g["our_spread"] <= 30 and 0 <= g["win_prob_home"] <= 1
            )
            if valid == len(week1_games) and week1_games:
                print(f"    OK  All {len(week1_games)} real Week 1 game predictions pass sanity bounds")
                self.checks_passed += 1
            else:
                print(f"    !!  {valid}/{len(week1_games)} Week 1 predictions pass sanity bounds")
                self.checks_failed += 1
        except Exception as e:
            print(f"    XX  Error validating predictions: {e}")
            self.checks_failed += 1

    def verify_tracking_setup(self):
        tracking_dir = PROJECT_ROOT / "data" / "season_tracking" / "week_1"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        print(f"    OK  Tracking structure ready: {tracking_dir}")
        self.checks_passed += 1

    def print_final_summary(self):
        print("\n" + "=" * 60)
        print("FINAL READINESS ASSESSMENT")
        print("=" * 60)
        print(f"\nChecks passed: {self.checks_passed}")
        print(f"Checks failed: {self.checks_failed}")

        if self.critical_flags:
            print(f"\n{len(self.critical_flags)} REAL INJURY FLAG(S) ON YOUR CRITICAL-PLAYER LIST:")
            for name, row in self.critical_flags:
                print(f"  !!  {name} ({row['team']}): {row['status']} - {row['body_part']}, return {row['return_date']}")
            print("  Verify these manually before kickoff - this is real ESPN data, not a guess,")
            print("  but 'this project checked' isn't the same as 'confirmed clear to play'.")
        else:
            print("\nNo real ESPN flags on your critical-player list.")

        if self.checks_failed == 0:
            print("\nNo blocking issues found by the automated checks above.")
            print("Real, disclosed gap (not a blocker, but read it): this project's OWN generated")
            print("files still can't carry real injury_status (nflreadpy has no 2026 support) - the")
            print("real signal above comes from ESPN's live injury report instead.")
            print("\nGames start: September 9, 2026")
            print("Tracking begins: September 16, 2026 (after Monday night)")
        else:
            print(f"\n{self.checks_failed} issue(s) detected above - resolve before relying on this.")
        print("=" * 60)


if __name__ == "__main__":
    Week1FinalCheck().run()
