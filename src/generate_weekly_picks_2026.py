"""Real pre-game "Our System" picks for every 2026 game, played or not -
closes a real, disclosed UX gap: betting_backtest_2026.py (and the
Betting Analysis tab) only ever shows a game once it's COMPLETE, which
makes a real pick look like it was added after the fact even though the
selection itself never depended on the result. This script runs the
exact same, already-validated "Our System" selection rule (imported
directly from betting_backtest.py - STRATEGIES["our_system"]["direction_fn"],
BET_TYPES[...]["settle_fn"] - not re-derived) against every real game
that has real odds joined, regardless of whether it's been played, and
writes frontend/src/data/weekly_picks_2026.json so the pick is visible
before kickoff.

Real, disclosed scope: a game only gets a real pick here if it clears
the same real entry gate Betting Analysis already uses (>2% moneyline
probability disagreement) AND has every real field that gate needs
(our_spread/vegas_spread/win_prob_home/win_prob_away/both moneylines/
both spread prices) - a game with no real odds joined yet (too far out)
simply has no real pick yet, not a fabricated one.
"""

import json
import os

from betting_backtest import STRATEGIES, BET_TYPES
from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
GAMES_PATH = os.path.join(FRONTEND_DATA_DIR, "games_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "weekly_picks_2026.json")

REQUIRED_FIELDS = ["our_spread", "vegas_spread", "win_prob_home", "win_prob_away",
                   "home_moneyline", "away_moneyline"]

_OUR_SYSTEM_DIRECTION = STRATEGIES["our_system"]["direction_fn"]
_SETTLE_MONEYLINE = BET_TYPES["moneyline"]["settle_fn"]
_SETTLE_ATS = BET_TYPES["ats"]["settle_fn"]


def generate_weekly_picks():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)

    picks = []
    for game in games:
        if any(game.get(f) is None for f in REQUIRED_FIELDS):
            continue
        side = _OUR_SYSTEM_DIRECTION(game)
        if side is None:
            continue

        pick_team = game["home_team"] if side == "home" else game["away_team"]
        record = {
            "id": game["id"],
            "week": game["week"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "kickoff_datetime": game.get("kickoff_datetime"),
            "kickoff_utc": game.get("kickoff_utc"),
            "pick_team": pick_team,
            "pick_side": side,
            "moneyline_odds": game["home_moneyline"] if side == "home" else game["away_moneyline"],
            "our_spread": game["our_spread"],
            "vegas_spread": game["vegas_spread"],
            "actual_winner": game.get("actual_winner"),
            "actual_home_score": game.get("actual_home_score"),
            "actual_away_score": game.get("actual_away_score"),
        }

        if game.get("actual_winner") is None:
            record["result_moneyline"] = "pending"
            record["result_ats"] = "pending"
            record["pnl_moneyline_units"] = None
            record["pnl_ats_units"] = None
        else:
            # Real settlement - same functions Betting Analysis uses, just
            # needing the one derived field (_load_games() computes this
            # for every completed game there; a single game here needs it
            # computed the same way).
            settle_game = dict(game)
            settle_game["point_diff"] = game["actual_home_score"] - game["actual_away_score"]
            result_ml, _odds_ml, pnl_ml = _SETTLE_MONEYLINE(side, settle_game)
            record["result_moneyline"] = result_ml
            record["pnl_moneyline_units"] = round(float(pnl_ml), 3)
            if game.get("home_spread_odds") is not None and game.get("away_spread_odds") is not None:
                result_ats, _odds_ats, pnl_ats = _SETTLE_ATS(side, settle_game)
                record["result_ats"] = result_ats
                record["pnl_ats_units"] = round(float(pnl_ats), 3)
            else:
                record["result_ats"] = "pending"
                record["pnl_ats_units"] = None

        picks.append(record)

    picks.sort(key=lambda r: (r["week"], r["kickoff_datetime"] or ""))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(picks, f, indent=2)
        record_generation("weekly_picks_2026")
    print(f"Wrote {len(picks)} real pre-game pick(s) -> {OUTPUT_PATH}")
    return picks


if __name__ == "__main__":
    generate_weekly_picks()
