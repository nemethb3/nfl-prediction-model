"""Shared per-game option building - used identically by backtest.py (on
real historical walk-forward games) and generate_juke_picks.py (on this
week's real live games), so the live picks generator is provably running
the exact strategy the backtest measured, not a reimplementation that
could silently drift from it.
"""

from juke.forecasting import model_probability
from juke.our_system import our_system_pick
from juke.pricing import get_market_price
from juke.valuation import best_option_per_game, expected_value_per_credit


def build_game_options(event, sigma, weights, elo_ensemble=None):
    """Real candidate options (one per market x side with real data
    available) for a single event, each carrying its own real p_model/
    price/p_final/decimal_odds/ev/edge - then collapsed to at most one
    per game (valuation.best_option_per_game) since moneyline and spread
    on the same team are highly correlated bets on the same outcome.

    Each option also carries confirmed_by_our_system: whether this
    project's already-live-validated "Our System" heuristic (see
    our_system.py) independently picks the SAME side for this game -
    real, disclosed cross-check signal, per the user's instruction to
    build on what's already shown positive ROI."""
    options = []
    our_system_side = our_system_pick(event)
    for market in ("moneyline", "spread"):
        w = weights[market]
        for side in ("home", "away"):
            p_model = model_probability(event, market, side, sigma, elo_ensemble)
            price_info = get_market_price(event, market, side)
            if p_model is None or price_info is None:
                continue
            price = price_info["price"]
            p_final = w * p_model + (1 - w) * price
            decimal_odds = price_info["decimal_odds"]
            ev = expected_value_per_credit(p_final, decimal_odds)
            options.append({
                "game_id": event["game_id"], "week": event["week"], "market": market, "side": side,
                "selection": event["home_team"] if side == "home" else event["away_team"],
                "p_model": p_model, "price": price, "p_final": p_final,
                "decimal_odds": decimal_odds, "ev": ev, "edge": p_final - price,
                "confirmed_by_our_system": our_system_side == side,
                "price_info": price_info, "event": event,
            })
    return best_option_per_game(options)
