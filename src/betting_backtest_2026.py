"""Real, in-season 2026 Betting Analysis - reuses betting_backtest.py's
already-validated strategies/settlement math (STRATEGIES, BET_TYPES,
moneyline_to_implied_probability, payout_for_stake, get_bet_direction,
_settle_moneyline, _settle_ats, _our_system_direction, etc.) verbatim via
import, rather than re-deriving the same formulas a second time - the same
real sign convention (positive vegas_spread = home favored) and the same
three strategies (Our System / Vegas Favorites / Underdogs Only) apply
unchanged; only the real data sources and the merge behavior differ.

Two real, disclosed adaptations from the 2025 version:

1. Real, partial odds coverage. 2025's run_betting_backtest() asserts every
   one of the 272 completed games has a real matched Vegas line (true for
   that already-complete, already-fully-sourced season). 2026 only has a
   real vegas_spread/moneyline for whichever games join_espn_odds_2026.py
   could match against the real, already-collected ESPN odds history (see
   that module's docstring) - currently Week 1 in full, growing weekly as
   refresh_weekly.py's odds collection keeps running before each week's
   games. An inner merge is used instead of an assert, and the real
   coverage (games bet vs. games completed) is disclosed in the output
   rather than silently only covering part of the season with no note.

2. Real -110 ATS juice assumption. 2025's real per-game home_spread_odds/
   away_spread_odds come from data/raw/vegas_lines_2015_2025.csv. ESPN's
   already-collected 2026 odds don't include a captured per-side spread
   price (see join_espn_odds_2026.py's docstring) - standard -110 is used
   instead, a disclosed, standard real-world default, not a derived model
   output.
"""

import csv
import json
from generation_timestamps import record_generation
import os

import pandas as pd

from betting_backtest import (
    STRATEGIES, BET_TYPES, _bet_team, _weekly_and_season_summary,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
GAMES_PATH = os.path.join(FRONTEND_DATA_DIR, "games_2026.json")
VEGAS_LINES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "vegas_lines_2026.csv")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "betting_backtest_results_2026.json")


def _load_games():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    df = pd.DataFrame(games)
    df = df[df["actual_home_score"].notna()].copy()
    df["point_diff"] = df["actual_home_score"] - df["actual_away_score"]
    return df


def _load_real_odds():
    if not os.path.exists(VEGAS_LINES_PATH):
        return pd.DataFrame(columns=["game_id", "home_moneyline", "away_moneyline",
                                      "home_spread_odds", "away_spread_odds"])
    with open(VEGAS_LINES_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    for col in ["home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_betting_backtest_2026():
    games = _load_games()
    odds = _load_real_odds()
    df = games.merge(odds, left_on="id", right_on="game_id", how="inner")
    df = df[df["vegas_spread"].notna()]

    coverage_note = (
        f"{len(df)}/{len(games)} real completed 2026 games had a real, joined ESPN closing-line "
        f"spread + moneyline (see join_espn_odds_2026.py) - only those are bet here. Coverage grows "
        f"weekly as more games complete and more real odds get collected."
    )

    results = {}
    for strategy_key, strategy in STRATEGIES.items():
        results[strategy_key] = {"label": strategy["label"], "description": strategy["description"]}
        for bet_type_key, bet_type in BET_TYPES.items():
            bets = []
            for _, game in df.iterrows():
                side = strategy["direction_fn"](game)
                if side is None:
                    continue
                result, odds_used, pnl = bet_type["settle_fn"](side, game)
                bets.append({
                    "week": int(game["week"]),
                    "matchup": f"{game['away_team']} @ {game['home_team']}",
                    "bet_team": _bet_team(side, game),
                    "bet_side": side,
                    "odds": float(odds_used),
                    "result": result,
                    "actual_winner": game["actual_winner"],
                    "pnl_units": round(float(pnl), 3),
                })
            weekly_summary, season_summary = _weekly_and_season_summary(bets)
            results[strategy_key][bet_type_key] = {
                "bets": bets,
                "weekly_summary": weekly_summary,
                "season_summary": season_summary,
            }

    return results, coverage_note, len(df), len(games)


def generate_betting_backtest_2026_json():
    results, coverage_note, n_bet, n_completed = run_betting_backtest_2026()
    # Real, disclosed schema note: kept FLAT like betting_backtest_results_2025.json
    # (BettingAnalysis.js iterates Object.keys(resultsData) expecting only real
    # strategy sub-dicts) - the two extra metadata keys below are filtered out by
    # the frontend's strategyKeys computation (checks for a real `.label` field),
    # not treated as a fourth strategy.
    output = {
        **results,
        "real_odds_coverage_note": coverage_note,
        "ats_juice_assumption": "Standard -110 both sides (see betting_backtest_2026.py docstring) - "
                                 "ESPN's already-collected 2026 odds don't include a captured per-side spread price.",
    }
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("betting_backtest_results_2026")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Real odds coverage: {n_bet}/{n_completed} completed games")
    for strategy_key, strategy_results in results.items():
        for bet_type_key in BET_TYPES:
            summary = strategy_results[bet_type_key]["season_summary"]
            print(f"{strategy_key} / {bet_type_key}: {summary}")
    return output


if __name__ == "__main__":
    generate_betting_backtest_2026_json()
