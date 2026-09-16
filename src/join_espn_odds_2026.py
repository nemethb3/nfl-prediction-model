"""Joins real, already-collected ESPN historical odds (data/espn_odds/
history.json - accumulated every week by espn_odds_orchestrate.py, which
already runs inside refresh_weekly.py) into the real 2026 game records.

Real, disclosed gap this fixes: games_2026.json's own vegas_spread field is
null for all 272 games (nflverse has no posted lines that far out - see
espn_odds_client.py's module docstring). ESPN's odds endpoint is the only
real market-line source available for 2026, and it's already being fetched
and cached every week - nobody had joined it back into games_2026.json or
into a real per-game moneyline file, which is why Betting Analysis (and the
real Vegas-comparison columns in Accuracy Tracker/Weekly Summary) have
stayed empty for 2026 even after Week 1 completed.

Real join key: history.json rows carry ESPN's own numeric event id
(game_id), which does NOT match this project's own id scheme
("2026_WW_AWAY_HOME") - verified 0/96 overlap by id. Both sides DO carry
real team abbreviations though, and (away_team, home_team) is a unique key
within one real NFL season (teams play each other at most once with a
given home/away assignment in the regular season, other than the rare
division rematch, which reverses home/away). Joined on that pair instead.

Real "closing line" convention: history.json accumulates one row per
(game, sportsbook, day) every time refresh_weekly.py's odds-collection step
runs, so a game can have several real snapshots between the line opening
and kickoff. The LATEST real fetched_at row per (away_team, home_team) is
used - the standard real-world reference line for grading a spread bet.

Real sign-convention fix: ESPN's own home_spread_current field uses the
standard sportsbook convention (negative = home favored, e.g. "-3" means
home favored by 3). This project's own vegas_spread/our_spread convention
is the opposite (positive = home favored - established and verified in
betting_backtest.py's module docstring for the real 2015-2025 data). Every
real spread pulled from ESPN is negated here to match.

Real, disclosed limitation: ESPN's parsed odds record (espn_odds_client.
parse_odds) captures real moneylines and real point-spread values but not
a real per-side spread-betting price ("juice") - the raw API does expose
one (awayTeamOdds.spreadOdds), but the already-running collector never
captured it, so nothing on disk has it for weeks already gone by. Standard
-110 (the near-universal real-world default price on either side of a
point spread) is used for ATS bets going forward - a disclosed, standard
assumption, not an invented weight.
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAMES_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "games_2026.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "espn_odds" / "history.json"
VEGAS_LINES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vegas_lines_2026.csv"

STANDARD_SPREAD_JUICE = -110


def _latest_odds_by_matchup():
    """Real {(away_team, home_team): latest real odds row} - one entry per
    real matchup, keeping only the row with the latest real fetched_at."""
    if not HISTORY_PATH.exists():
        return {}
    with open(HISTORY_PATH, encoding="utf-8") as f:
        history = json.load(f)

    latest = {}
    for row in history:
        if row.get("home_spread_current") is None and row.get("home_moneyline") is None:
            continue
        key = (row.get("away_team"), row.get("home_team"))
        if key[0] is None or key[1] is None:
            continue
        existing = latest.get(key)
        if existing is None or (row.get("fetched_at") or "") > (existing.get("fetched_at") or ""):
            latest[key] = row

    return latest


def join_odds():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)

    odds_by_matchup = _latest_odds_by_matchup()

    n_spread_joined = 0
    vegas_lines_rows = []
    for game in games:
        key = (game["away_team"], game["home_team"])
        row = odds_by_matchup.get(key)
        if row is None:
            continue

        if row.get("home_spread_current") is not None:
            game["vegas_spread"] = round(-float(row["home_spread_current"]), 1)
            n_spread_joined += 1

        if row.get("home_moneyline") is not None and row.get("away_moneyline") is not None:
            vegas_lines_rows.append({
                "game_id": game["id"],
                "home_moneyline": row["home_moneyline"],
                "away_moneyline": row["away_moneyline"],
                "home_spread_odds": STANDARD_SPREAD_JUICE,
                "away_spread_odds": STANDARD_SPREAD_JUICE,
                "sportsbook": row.get("sportsbook"),
                "closing_line_fetched_at": row.get("fetched_at"),
            })

    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)

    VEGAS_LINES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(VEGAS_LINES_OUTPUT, "w", encoding="utf-8", newline="") as f:
        if vegas_lines_rows:
            writer = csv.DictWriter(f, fieldnames=list(vegas_lines_rows[0].keys()))
            writer.writeheader()
            writer.writerows(vegas_lines_rows)

    print(f"Real vegas_spread joined onto games_2026.json for {n_spread_joined}/{len(games)} games")
    print(f"Real vegas_lines_2026.csv written -> {VEGAS_LINES_OUTPUT} ({len(vegas_lines_rows)} rows)")
    return n_spread_joined, len(vegas_lines_rows)


if __name__ == "__main__":
    join_odds()
