"""Real Week N (N >= 2) prediction snapshot - generalizes lock_week2_
predictions.py (same real convention: flat files directly under
data/locked_predictions/, week-filtered rows, a manifest, plus the
ensemble_projections_2026.json snapshot added for Week 2) the same way
generate_fantasy_rankings_weekN_2026.py generalized the old Week-2-only
fantasy generator once Week 3 actually arrived with no generator for it.

lock_week1_predictions.py is kept as-is (Week 1 has its own extra manifest
fields - games_scheduled/tracking_begins - and no ensemble file existed
yet when it ran); this script replaces lock_week2_predictions.py for every
week from here on. Targets real_current_week_2026() by default, same CLI
override convention (`python lock_weekN_predictions.py <week>`) as the
weekN fantasy-rankings generator.

Real, disclosed reminder carried over from lock_week2_predictions.py's own
docstring: this is a git-tracked SNAPSHOT COPY for honest post-hoc
comparison, not a gate that blocks the live frontend/src/data/*.json files
from continuing to update additively as the locked week is actually
played.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
LOCK_DIR = PROJECT_ROOT / "data" / "locked_predictions"


class PredictionLock:
    def __init__(self, week: int):
        self.week = week
        self.timestamp = datetime.now().isoformat()
        LOCK_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"\nLOCKING WEEK {self.week} PREDICTIONS\n")
        self.snapshot("games_2026.json", f"week_{self.week}_game_predictions.json", "games")
        self.snapshot("player_props_2026.json", f"week_{self.week}_player_projections.json", "projections")
        self.snapshot("fantasy_rankings_2026.json", f"week_{self.week}_player_rankings.json", "rankings")
        self.snapshot("ensemble_projections_2026.json", f"week_{self.week}_ensemble_projections.json",
                       "ensemble", nested_key="players")
        self.create_manifest()
        self.print_summary()

    def snapshot(self, src_name, out_name, key, nested_key=None):
        print(f"  Locking {src_name} (Week {self.week} rows only)...")
        src = DATA_DIR / src_name
        if not src.exists():
            print(f"    Missing: {src}")
            return
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        rows = data[nested_key] if nested_key else data
        week_rows = [row for row in rows if row.get("week") == self.week]
        lock_data = {"week": self.week, "locked_at": self.timestamp, "count": len(week_rows), key: week_rows}
        out = LOCK_DIR / out_name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)
        print(f"    Locked {len(week_rows)} rows -> {out.name}")

    def create_manifest(self):
        print("  Creating lock manifest...")
        manifest = {
            "week": self.week,
            "locked_at": self.timestamp,
            "files": {
                "games": f"week_{self.week}_game_predictions.json",
                "projections": f"week_{self.week}_player_projections.json",
                "rankings": f"week_{self.week}_player_rankings.json",
                "ensemble": f"week_{self.week}_ensemble_projections.json",
            },
            "notes": (
                f"Real snapshot of this project's Week {self.week} predictions as of the "
                "locked_at timestamp - kept for honest post-hoc accuracy comparison, not "
                "literally immutable (see this script's own module docstring, same "
                "convention as lock_week1_predictions.py/lock_week2_predictions.py)."
            ),
        }
        with open(LOCK_DIR / f"week_{self.week}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("    Manifest created")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("PREDICTION LOCK COMPLETE")
        print("=" * 60)
        print(f"\nLocked at: {self.timestamp}")
        print(f"Location: {LOCK_DIR}/")
        print("\nThese are a real, git-tracked snapshot for later accuracy comparison.")
        print(f"Do not re-run this script for Week {self.week} after games start.")
        print("=" * 60)


if __name__ == "__main__":
    from current_week_2026 import real_current_week_2026
    target_week = int(sys.argv[1]) if len(sys.argv) > 1 else real_current_week_2026()
    PredictionLock(target_week).run()
