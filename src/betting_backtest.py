"""Betting Analysis Backtest: three strategies (Our System / Vegas Favorites
/ Underdogs Only) on all 272 real, completed 2025 games, settled both as
moneyline and against-the-spread (ATS) bets.

Verification-driven changes from the pasted spec (Q&A decisions, 2026-07-30):

1. Real odds, not synthetic ones. The spec's estimate_moneyline_from_spread()
   invents a moneyline from the spread's magnitude alone (e.g. a real -140
   favorite would get priced at -250). This project's own raw data already
   has real odds for every 2025 game - data/raw/vegas_lines_2015_2025.csv
   has home_moneyline/away_moneyline/home_spread_odds/away_spread_odds for
   all 272 games used here (matched 272/272 by game_id, verified before
   writing this). Those are used directly instead.

2. Real sign convention. The spec's moneyline formula assumed "positive
   spread = underdog." This project's real, verified convention (284/284
   real 2025 moneylines, established in the Section 1 task) is the
   opposite: positive vegas_spread/our_spread means the HOME team is
   favored. get_bet_direction() below uses the real convention. (The
   synthetic formula itself is dropped per point 1, but the direction logic
   built on top of it still needed the same sign fix.)

3. "Our System" uses the spec's requested heuristic (bet the favorite when
   our spread is more extreme than Vegas's, the underdog when less extreme)
   as its own strategy - explicitly requested as separate from, not merged
   with, edge_detection.py's already-validated disagreement+confidence ATS
   logic (real -36% ROI finding). Three games out of 272 have our_spread and
   vegas_spread favoring opposite sides (a real, rare case where the model
   flips who it thinks is favored vs. Vegas) - "more/less extreme" isn't a
   coherent comparison there, so those are treated as no-bet, matching the
   spec's own "shouldn't happen with real data" comment.

4. "Our" probability, for the should-bet gate, is win_prob_home straight
   from games_2025.json - this project's real, already-fitted win-
   probability model output (the Vegas-fit backtest winner, see
   win_probability_backtest.py) - rather than re-deriving a second synthetic
   probability from our_spread via a vig formula, which would have
   reintroduced the exact fabrication problem removed in point 1.

5. Real edge cases handled: one real 2025 tie (2025_04_GB_DAL, 40-40) is a
   push on moneyline bets (no "TIE" moneyline was offered) and settled
   normally on ATS (the formula naturally handles a 0 point_diff). One real
   exact ATS push exists (2025_12_PIT_CHI, margin 3 vs. a 3.0 line). Pushes
   are excluded from win/loss counts and from the ROI denominator (no stake
   was actually at risk) but are still counted and reported.
"""

import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
GAMES_PATH = os.path.join(FRONTEND_DATA_DIR, "games_2025.json")
VEGAS_ODDS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "vegas_lines_2015_2025.csv")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "betting_backtest_results_2025.json")

SHOULD_BET_MIN_DIFF = 0.02


def moneyline_to_implied_probability(moneyline):
    """Standard American-odds implied probability (not de-vigged - the real
    quoted odds already include the book's vig, as they do on a real bet
    slip)."""
    if moneyline == 0:
        return 0.5
    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    return 100 / (moneyline + 100)


def payout_for_stake(moneyline, stake=1.0):
    """Real American-odds profit on a winning bet of `stake` units."""
    if moneyline >= 0:
        return stake * moneyline / 100.0
    return stake * 100.0 / abs(moneyline)


def should_bet(our_prob, vegas_prob, min_diff=SHOULD_BET_MIN_DIFF):
    return abs(our_prob - vegas_prob) > min_diff


def get_bet_direction(our_spread, vegas_spread):
    """Real sign convention: positive spread = HOME team favored (see module
    docstring point 2). Returns 'home', 'away', or None (no bet)."""
    if our_spread == vegas_spread:
        return None

    our_home_favored = our_spread > 0
    vegas_home_favored = vegas_spread > 0
    if our_home_favored != vegas_home_favored:
        return None  # model flips the favored side vs. Vegas - rare (3/272), not a clean extremity comparison

    favorite_side = "home" if our_home_favored else "away"
    underdog_side = "away" if our_home_favored else "home"
    return favorite_side if abs(our_spread) > abs(vegas_spread) else underdog_side


def _load_games():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    df = pd.DataFrame(games)
    df["point_diff"] = df["actual_home_score"] - df["actual_away_score"]
    return df


def _load_real_odds():
    odds = pd.read_csv(VEGAS_ODDS_PATH)
    odds = odds[odds["season"] == 2025][
        ["game_id", "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds"]
    ]
    return odds


def _settle_moneyline(side, game):
    moneyline = game["home_moneyline"] if side == "home" else game["away_moneyline"]
    if game["actual_winner"] == "TIE":
        return "push", moneyline, 0.0
    bet_team_won = (side == "home" and game["actual_winner"] == game["home_team"]) or (
        side == "away" and game["actual_winner"] == game["away_team"]
    )
    if bet_team_won:
        return "win", moneyline, payout_for_stake(moneyline)
    return "loss", moneyline, -1.0


def _settle_ats(side, game):
    spread_odds = game["home_spread_odds"] if side == "home" else game["away_spread_odds"]
    home_margin_vs_spread = game["point_diff"] - game["vegas_spread"]
    margin = home_margin_vs_spread if side == "home" else -home_margin_vs_spread
    if margin > 0:
        return "win", spread_odds, payout_for_stake(spread_odds)
    if margin == 0:
        return "push", spread_odds, 0.0
    return "loss", spread_odds, -1.0


def _bet_team(side, game):
    return game["home_team"] if side == "home" else game["away_team"]


def _our_system_direction(game):
    our_prob_home = game["win_prob_home"]
    our_prob_away = game["win_prob_away"]
    vegas_prob_home = moneyline_to_implied_probability(game["home_moneyline"])
    vegas_prob_away = moneyline_to_implied_probability(game["away_moneyline"])

    if not (should_bet(our_prob_home, vegas_prob_home) or should_bet(our_prob_away, vegas_prob_away)):
        return None
    return get_bet_direction(game["our_spread"], game["vegas_spread"])


def _vegas_favorite_direction(game):
    return "home" if game["vegas_spread"] > 0 else "away"


def _underdog_direction(game):
    return "away" if game["vegas_spread"] > 0 else "home"


STRATEGIES = {
    "our_system": {
        "label": "Our System",
        "description": (
            "Bet when our model's win probability and Vegas's implied probability differ by "
            "more than 2%; direction is the favorite when our spread is more extreme than "
            "Vegas's, the underdog when less extreme."
        ),
        "direction_fn": _our_system_direction,
    },
    "vegas_favorites": {
        "label": "Vegas Favorites",
        "description": "Bet the Vegas favorite in every game.",
        "direction_fn": _vegas_favorite_direction,
    },
    "underdogs_only": {
        "label": "Underdogs Only",
        "description": "Bet the underdog in every game.",
        "direction_fn": _underdog_direction,
    },
}

BET_TYPES = {
    "moneyline": {"label": "Moneyline (straight-up)", "settle_fn": _settle_moneyline},
    "ats": {"label": "Against the Spread", "settle_fn": _settle_ats},
}


def _weekly_and_season_summary(bets):
    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        return {}, {"total_bets": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0.0, "pnl_units": 0.0, "roi_pct": 0.0}

    weekly_summary = {}
    for week, wk in bets_df.groupby("week"):
        wins = int((wk["result"] == "win").sum())
        losses = int((wk["result"] == "loss").sum())
        pushes = int((wk["result"] == "push").sum())
        decided = wins + losses
        weekly_summary[int(week)] = {
            "total_bets": int(len(wk)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": round(wins / decided * 100, 1) if decided else 0.0,
            "pnl_units": round(float(wk["pnl_units"].sum()), 3),
        }

    wins = int((bets_df["result"] == "win").sum())
    losses = int((bets_df["result"] == "loss").sum())
    pushes = int((bets_df["result"] == "push").sum())
    decided = wins + losses
    pnl_units = float(bets_df["pnl_units"].sum())
    season_summary = {
        "total_bets": int(len(bets_df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": round(wins / decided * 100, 1) if decided else 0.0,
        "pnl_units": round(pnl_units, 3),
        "roi_pct": round(pnl_units / decided * 100, 2) if decided else 0.0,
    }
    return weekly_summary, season_summary


def run_betting_backtest():
    games = _load_games()
    odds = _load_real_odds()
    df = games.merge(odds, left_on="id", right_on="game_id", how="inner")
    assert len(df) == len(games), (
        f"real odds merge dropped games: {len(games)} games -> {len(df)} after merge"
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
                bets.append(
                    {
                        "week": int(game["week"]),
                        "matchup": f"{game['away_team']} @ {game['home_team']}",
                        "bet_team": _bet_team(side, game),
                        "bet_side": side,
                        "odds": float(odds_used),
                        "result": result,
                        "actual_winner": game["actual_winner"],
                        "pnl_units": round(float(pnl), 3),
                    }
                )
            weekly_summary, season_summary = _weekly_and_season_summary(bets)
            results[strategy_key][bet_type_key] = {
                "bets": bets,
                "weekly_summary": weekly_summary,
                "season_summary": season_summary,
            }

    return results


def generate_betting_backtest_json():
    results = run_betting_backtest()
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    for strategy_key, strategy_results in results.items():
        for bet_type_key in BET_TYPES:
            summary = strategy_results[bet_type_key]["season_summary"]
            print(f"{strategy_key} / {bet_type_key}: {summary}")
    return results


if __name__ == "__main__":
    generate_betting_backtest_json()
