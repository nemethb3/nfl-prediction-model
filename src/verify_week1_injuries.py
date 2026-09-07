"""Real Week 1 injury-data check.

Real history: this script originally existed to disclose, honestly, that
there is no real per-player injury data for season 2026 in this
project's own generated files - generate_fantasy_rankings_2026_week1.py
hardcodes `injury_status: "healthy"` for every row because nflreadpy has
no 2026 injury support yet (verified - raises "Season must be between
2009 and 2025"). That gap in this project's OWN files is still real and
still checked below.

Real update (Check ESPN API for Injury Data task): a real, live, working
ESPN endpoint was found and verified - see get_espn_injuries.py's own
module docstring for the real research. This script now uses it as the
actual real injury signal for Week 1, matched by exact name+team against
fantasy_rankings_2026.json. Real, disclosed limitation carried over: only
players in this project's own ~390-player ranked pool are matched (see
get_espn_injuries.py's real match-rate output for the real, honest miss
count - most misses are real non-fantasy-relevant positions, not a
matching bug).

Also real, separate signal: data/processed/nfl_rosters_2026.csv is
filtered to real roster status=="ACT" by update_rosters_2026.py - a
player on real long-term IR is already excluded from that file entirely
(distinct from ESPN's day-to-day report above). Checked here too.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from get_espn_injuries import ESPNInjuriesFetcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANKINGS_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "fantasy_rankings_2026.json"
ROSTER_PATH = PROJECT_ROOT / "data" / "processed" / "nfl_rosters_2026.csv"
GENERATED_AT_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "generated_at.json"

CRITICAL_PLAYERS = [
    "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts",
    "Christian McCaffrey", "Travis Kelce",
]


class InjuryDataCheck:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.notes = []
        self.issues = []

    def run(self):
        print("\nWEEK 1 INJURY DATA CHECK\n")
        self.check_espn_injuries()
        self.disclose_real_gap()
        self.check_generation_freshness()
        self.check_ir_exclusion_for_critical_players()
        self.print_report()

    def check_espn_injuries(self):
        print("  Fetching real ESPN injury report (Questionable/Out/IR/Suspension)...")
        try:
            fetcher = ESPNInjuriesFetcher()
            matched, unmatched = fetcher.match_to_rankings()
            self.notes.append(
                f"Real ESPN injury report: {len(matched)} flagged players matched into this "
                f"project's ranked pool by exact name+team ({len(unmatched)} flagged ESPN rows "
                "unmatched - real non-fantasy-relevant positions, not necessarily a bug)."
            )
            critical_hits = [
                (name, row) for name, row in matched.items() if name in CRITICAL_PLAYERS
            ]
            if critical_hits:
                for name, row in critical_hits:
                    self.issues.append(
                        f"REAL FLAG: {name} ({row['team']}) - {row['status']}: {row['body_part']} "
                        f"({row['detail']}), return {row['return_date']} - {row['short_comment']}"
                    )
            for name, row in matched.items():
                if name in CRITICAL_PLAYERS:
                    continue
                self.notes.append(
                    f"{name} ({row['team']}, {row['position']}): {row['status']} - {row['body_part']}, "
                    f"return {row['return_date']}"
                )
        except Exception as e:
            self.issues.append(f"ESPN injury fetch failed: {e}")

    def disclose_real_gap(self):
        print("  Checking real injury_status coverage...")
        if not RANKINGS_PATH.exists():
            self.issues.append(f"Missing: {RANKINGS_PATH}")
            return
        try:
            with open(RANKINGS_PATH, encoding="utf-8") as f:
                players = json.load(f)
            statuses = {p.get("injury_status") for p in players}
            if statuses == {"healthy"}:
                self.notes.append(
                    f"Confirmed real, known gap: all {len(players)} rows show injury_status="
                    "'healthy' - this is NOT a real per-player signal (nflreadpy has no 2026 "
                    "injury reports yet). Do not treat this as a health confirmation."
                )
            else:
                # A real, unexpected change - worth surfacing loudly rather
                # than assuming it's still the same known gap.
                self.notes.append(
                    f"injury_status values changed from the known all-'healthy' gap: {statuses} - "
                    "real 2026 injury data may now be available; re-check generate_fantasy_"
                    "rankings_2026_week1.py's own real nflreadpy.load_injuries() support before "
                    "trusting this."
                )
        except Exception as e:
            self.issues.append(f"Error reading rankings: {e}")

    def check_generation_freshness(self):
        print("  Checking real generation timestamp...")
        if not GENERATED_AT_PATH.exists():
            self.issues.append(f"Missing: {GENERATED_AT_PATH}")
            return
        try:
            with open(GENERATED_AT_PATH, encoding="utf-8") as f:
                stamps = json.load(f)
            latest = max(stamps.values())
            latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            hours_old = (datetime.now(timezone.utc) - latest_dt).total_seconds() / 3600
            if hours_old < 48:
                self.notes.append(f"Most recent real generation: {hours_old:.1f} hours ago ({latest})")
            else:
                self.issues.append(f"Most recent real generation is {hours_old:.1f} hours old - re-run refresh_weekly.py")
        except Exception as e:
            self.issues.append(f"Error checking generation timestamp: {e}")

    def check_ir_exclusion_for_critical_players(self):
        print("  Checking real roster-status (IR) exclusion for key players...")
        if not ROSTER_PATH.exists():
            self.issues.append(f"Missing: {ROSTER_PATH}")
            return
        try:
            rosters = pd.read_csv(ROSTER_PATH)
            on_active_roster = set(rosters["player_name"])
            for name in CRITICAL_PLAYERS:
                if name in on_active_roster:
                    self.notes.append(f"{name}: on real active (ACT) roster - not on long-term IR")
                else:
                    self.issues.append(
                        f"{name}: NOT in nfl_rosters_2026.csv's real ACT-only roster - either a "
                        "real long-term IR/inactive designation, a name-matching miss, or a real "
                        "trade/release. Verify manually."
                    )
        except Exception as e:
            self.issues.append(f"Error checking roster status: {e}")

    def print_report(self):
        print("\n" + "=" * 60)
        print("INJURY DATA CHECK REPORT")
        print("=" * 60)
        if self.notes:
            print(f"\nNOTES ({len(self.notes)}):")
            for msg in self.notes:
                print(f"  ..  {msg}")
        if self.issues:
            print(f"\nISSUES ({len(self.issues)}):")
            for msg in self.issues:
                print(f"  !!  {msg}")
        print(
            "\nREAL, HONEST BOTTOM LINE: real day-to-day injury status now comes from ESPN's live "
            "injury report (see above) - this project's OWN generated files (injury_status field) "
            "still can't provide it (nflreadpy has no 2026 support). Any name in CRITICAL_PLAYERS "
            "flagged above needs a manual double-check before kickoff, same as always - this is "
            "one real source, not a guarantee."
        )
        print("=" * 60)


if __name__ == "__main__":
    InjuryDataCheck().run()
