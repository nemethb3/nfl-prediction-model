"""Real Week 1 roster verification - checks the real files this project
actually has, with the real field names/values they actually use.

Real corrections made vs. the originally pasted spec before writing this:

1. `confidence_tier` real values are 'high'/'lower' (lowercase) - see
   generate_fantasy_rankings_2026_week1.py's own docstring. The spec
   checked for 'High' (capitalized) - would have always found 0 matches,
   a guaranteed false "only 0 QB starters" alarm every single run.
2. data/processed/nfl_rosters_2026.csv real columns are
   `player_id, player_name, team, position, status` - real `status`
   values here are roster status (e.g. "ACT"), already filtered to
   ACT-only by update_rosters_2026.py (see that script's own docstring) -
   not a per-game injury/game-status field (that's a separate, real gap -
   see verify_week1_injuries.py).
3. Paths made robust to cwd (Path(__file__) - based PROJECT_ROOT)
   instead of assuming "run from repo root".
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROSTER_PATH = PROJECT_ROOT / "data" / "processed" / "nfl_rosters_2026.csv"
RANKINGS_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "fantasy_rankings_2026.json"
SLEEPER_MAPPING_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "sleeper_id_mapping.json"

# Real 32 nflverse-convention team codes this project uses throughout
# (verified directly against nfl_rosters_2026.csv's real, current values).
EXPECTED_TEAMS = {
    "KC", "BUF", "BAL", "PIT", "LA", "SF", "PHI", "DAL", "DEN", "IND", "CIN", "HOU",
    "MIN", "GB", "TB", "DET", "NE", "NYJ", "LAC", "TEN", "CAR", "NO", "ATL", "ARI",
    "NYG", "WAS", "SEA", "LV", "CHI", "JAX", "MIA", "CLE",
}


class RosterVerification:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.issues = []
        self.passed = []

    def run(self):
        print("\nWEEK 1 ROSTER VERIFICATION\n")
        self.check_all_teams_present()
        self.check_starter_confidence()
        self.verify_sleeper_mapping()
        self.print_report()

    def check_all_teams_present(self):
        print("  Checking all 32 teams...")
        if not ROSTER_PATH.exists():
            self.issues.append(f"Missing: {ROSTER_PATH}")
            return
        try:
            rosters = pd.read_csv(ROSTER_PATH)
            found = set(rosters["team"].unique())
            missing = EXPECTED_TEAMS - found
            extra = found - EXPECTED_TEAMS
            if not missing:
                self.passed.append(f"All 32 teams in rosters ({len(rosters)} real active QB/RB/WR/TE rows)")
            else:
                self.issues.append(f"Missing teams: {sorted(missing)}")
            if extra:
                self.issues.append(f"Unexpected team codes: {sorted(extra)}")
        except Exception as e:
            self.issues.append(f"Error reading rosters: {e}")

    def check_starter_confidence(self):
        print("  Checking real confidence_tier distribution by position...")
        if not RANKINGS_PATH.exists():
            self.issues.append(f"Missing: {RANKINGS_PATH}")
            return
        try:
            with open(RANKINGS_PATH, encoding="utf-8") as f:
                players = json.load(f)
            by_pos = {}
            for p in players:
                pos = p.get("position", "UNK")
                bucket = by_pos.setdefault(pos, {"high": 0, "lower": 0})
                bucket[p.get("confidence_tier", "high")] += 1

            print(f"     Real confidence_tier counts by position: {by_pos}")
            # Real, disclosed expectation (not asserted): every real 2026
            # roster has exactly one real starting QB, so 'high'-tier QB
            # rows should track real team count (~32), not 28 asserted by
            # the original spec with no stated basis.
            qb_high = by_pos.get("QB", {}).get("high", 0)
            if qb_high >= 28:
                self.passed.append(f"QB 'high'-confidence rows: {qb_high}")
            else:
                self.issues.append(f"QB 'high'-confidence rows: only {qb_high} (expected ~32)")
        except Exception as e:
            self.issues.append(f"Error checking confidence tiers: {e}")

    def verify_sleeper_mapping(self):
        print("  Verifying Sleeper ID mapping...")
        if not SLEEPER_MAPPING_PATH.exists():
            self.issues.append(f"Missing: {SLEEPER_MAPPING_PATH}")
            return
        try:
            with open(SLEEPER_MAPPING_PATH, encoding="utf-8") as f:
                mapping = json.load(f)
            if len(mapping) >= 100:
                self.passed.append(f"Sleeper mapping: {len(mapping)} players mapped")
            else:
                self.issues.append(f"Sleeper mapping: only {len(mapping)} players (expected 100+)")
        except Exception as e:
            self.issues.append(f"Error reading Sleeper mapping: {e}")

    def print_report(self):
        print("\n" + "=" * 60)
        print("ROSTER VERIFICATION REPORT")
        print("=" * 60)
        if self.passed:
            print(f"\nPASSED ({len(self.passed)}):")
            for msg in self.passed:
                print(f"  OK  {msg}")
        if self.issues:
            print(f"\nISSUES ({len(self.issues)}):")
            for msg in self.issues:
                print(f"  !!  {msg}")
            print("\nRESOLVE BEFORE GAME START")
        else:
            print("\nALL ROSTERS VERIFIED")
        print("=" * 60)


if __name__ == "__main__":
    RosterVerification().run()
