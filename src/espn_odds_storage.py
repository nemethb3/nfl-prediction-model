"""Simple, real JSON-based storage for ESPN odds collection runs - no
database, matching this project's existing convention of JSON artifacts
under data/ (see data/processed/, data/predictions/). Upgrade to a real
DB only if the accumulated history file becomes large enough to need one
- not pre-built here since that need doesn't exist yet.
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = PROJECT_ROOT / "data" / "espn_odds"
DAILY_DIR = ODDS_DIR / "daily"
HISTORY_PATH = ODDS_DIR / "history.json"


def save_daily_snapshot(games_odds, run_date=None):
    """Writes one file per collection run: data/espn_odds/daily/YYYY-MM-DD.json
    (real, current-day view - overwritten if run more than once same day,
    not appended, since it's a snapshot of "odds as of this run")."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")
    path = DAILY_DIR / f"{run_date}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(games_odds, f, indent=2)
    return path


def append_history(games_odds):
    """Appends this run's real records to the accumulated real history
    log (data/espn_odds/history.json) - deduped on (game_id, sportsbook,
    fetched date) so re-running collection multiple times in one day
    doesn't pile up near-identical rows; a real, distinct line-movement
    row IS kept if the same book's numbers actually changed since the
    last recorded row that day."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)

    existing_keys = {
        (r.get("game_id"), r.get("sportsbook"), r.get("fetched_at", "")[:10],
         r.get("home_spread_current"), r.get("away_spread_current"),
         r.get("home_moneyline"), r.get("away_moneyline"), r.get("total"))
        for r in history
    }

    appended = 0
    for record in games_odds:
        key = (record.get("game_id"), record.get("sportsbook"), record.get("fetched_at", "")[:10],
               record.get("home_spread_current"), record.get("away_spread_current"),
               record.get("home_moneyline"), record.get("away_moneyline"), record.get("total"))
        if key in existing_keys:
            continue
        history.append(record)
        existing_keys.add(key)
        appended += 1

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return appended, len(history)
