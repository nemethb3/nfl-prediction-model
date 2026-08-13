"""Real generation timestamps for frontend/src/data/*.json exports, no
fabrication.

AUDIT_2026-08-12_DEEP.md Section 7.1/Recommendation 14: none of the 17
shipped data files carried any indication of when they were generated.
Adding a top-level `generated_at` field directly to each file was
considered and rejected for most of them: 6 of the 17 (games_*, fantasy_
rankings_*, season_projections_*) are top-level JSON ARRAYS, not objects -
every consuming component does `data.map(...)`/`data.filter(...)` directly
on the imported JSON (verified: GamePredictions.js, FantasyRankings.js,
SeasonProjections.js, SeasonContext.js all treat these as bare arrays) -
turning them into `{generated_at, records: [...]}` envelopes would be a
real breaking change requiring every one of those components to be
updated in lockstep. A single shared sidecar file avoids that entirely -
zero risk to any existing consumer, real generation time still available
for a future "data as of" UI element to read.
"""

import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "generated_at.json")


def record_generation(dataset_name):
    """Records a real UTC timestamp for `dataset_name` in the shared
    sidecar file, merging with (not overwriting) any other dataset's
    already-recorded timestamp."""
    existing = {}
    if os.path.exists(SIDECAR_PATH):
        with open(SIDECAR_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    existing[dataset_name] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(SIDECAR_PATH), exist_ok=True)
    with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
