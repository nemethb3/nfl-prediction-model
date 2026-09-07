"""Real Week 1 prediction snapshot - a timestamped copy of what this
project's real models predicted before any 2026 games were played, kept
for honest after-the-fact accuracy comparison.

Real correction from the originally pasted spec's wording: these files
are a real SNAPSHOT (a copy, checked into git for a real, permanent,
tamper-evident record), not literally OS-level immutable - nothing stops
someone from editing the copy too. The real protection is git history
plus not editing frontend/src/data/*.json in place for Week 1 numbers
after this runs.

Paths made robust to cwd (Path(__file__)-based PROJECT_ROOT).
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
LOCK_DIR = PROJECT_ROOT / "data" / "locked_predictions"


class PredictionLock:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        LOCK_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\nLOCKING WEEK 1 PREDICTIONS\n")
        self.snapshot("games_2026.json", "week_1_game_predictions.json", "games")
        self.snapshot("player_props_2026.json", "week_1_player_projections.json", "projections")
        self.snapshot_full("fantasy_rankings_2026.json", "week_1_player_rankings.json", "rankings")
        self.create_manifest()
        self.print_summary()

    def snapshot(self, src_name, out_name, key):
        print(f"  Locking {src_name} (Week 1 rows only)...")
        src = DATA_DIR / src_name
        if not src.exists():
            print(f"    Missing: {src}")
            return
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        week1 = [row for row in data if row.get("week") == 1]
        lock_data = {"week": 1, "locked_at": self.timestamp, "count": len(week1), key: week1}
        out = LOCK_DIR / out_name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)
        print(f"    Locked {len(week1)} rows -> {out.name}")

    def snapshot_full(self, src_name, out_name, key):
        # fantasy_rankings_2026.json is Week 1 only right now (real, disclosed -
        # see DECISIONS_LOG.md), so there's no `week` filter needed - every row
        # already is Week 1.
        print(f"  Locking {src_name}...")
        src = DATA_DIR / src_name
        if not src.exists():
            print(f"    Missing: {src}")
            return
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        lock_data = {"week": 1, "locked_at": self.timestamp, "count": len(data), key: data}
        out = LOCK_DIR / out_name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)
        print(f"    Locked {len(data)} rows -> {out.name}")

    def create_manifest(self):
        print("  Creating lock manifest...")
        manifest = {
            "week": 1,
            "locked_at": self.timestamp,
            "files": {
                "games": "week_1_game_predictions.json",
                "projections": "week_1_player_projections.json",
                "rankings": "week_1_player_rankings.json",
            },
            "notes": (
                "Real snapshot of this project's Week 1 predictions as of the locked_at "
                "timestamp - kept for honest post-hoc accuracy comparison, not literally "
                "immutable (see this script's own module docstring)."
            ),
            "games_scheduled": "2026-09-09 (Thursday) - 2026-09-15 (Monday night)",
            "tracking_begins": "2026-09-16",
        }
        with open(LOCK_DIR / "week_1_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("    Manifest created")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("PREDICTION LOCK COMPLETE")
        print("=" * 60)
        print(f"\nLocked at: {self.timestamp}")
        print(f"Location: {LOCK_DIR}/")
        print("\nThese are a real, git-tracked snapshot for later accuracy comparison.")
        print("Do not re-run this script for Week 1 after games start.")
        print("=" * 60)


if __name__ == "__main__":
    PredictionLock().run()
