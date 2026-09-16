"""Real current-2026-week helper - the first week with any game whose
real outcome isn't in yet (actual_winner still null), mirroring frontend/
src/utils/getCurrentWeek.js exactly (same real completion check
GameCard.js already uses: actual_home_score/actual_winner !== null, NOT
a `game_completed` field - no such field exists anywhere in
games_2026.json). Kept in one shared place so any real 2026 generator
that needs to know "what week is it right now" (e.g. generate_injury_
adjustments_2026.py, which used to hardcode week=1 forever) can't drift
from the frontend's own notion of the current week.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAMES_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "games_2026.json"


def real_current_week_2026():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    weeks = sorted({g["week"] for g in games})
    for week in weeks:
        week_games = [g for g in games if g["week"] == week]
        if any(g.get("actual_winner") is None for g in week_games):
            return week
    return weeks[-1] if weeks else 1


if __name__ == "__main__":
    print(real_current_week_2026())
