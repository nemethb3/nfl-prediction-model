"""Market prices behind one interface, get_market_price(event, market,
side) - so a later real Polymarket feed can replace the body of this
function without touching forecasting/valuation/staking at all.

Real proxy in use now (per the task's own instruction): ESPN's own
moneyline/spread odds (already collected by espn_odds_orchestrate.py,
joined onto games_2026.json by join_espn_odds_2026.py - both real,
already-running infrastructure, not new for this task) stand in for a
real Polymarket price. The de-vigged (no-vig) implied probability is
treated as the assumed Polymarket price, per the task's own instruction -
Polymarket's real fee structure is a flat per-market fee on a binary
share, closer to a no-vig probability than to a vigged American-odds
probability.

`event` is one row from games_2026.json (or an equivalent historical row
with the same field names - see calibration.py/backtest.py, which build
one from data/backtest/vegas_with_results_2015_2025.csv for the walk-
forward backtest). Real fields read here: home_moneyline, away_moneyline,
vegas_spread, home_spread_odds, away_spread_odds, used_default_spread_juice,
vegas_line_source, vegas_spread_observed_at.
"""

from betting_backtest import moneyline_to_implied_probability
from juke.config import POLYMARKET_FEE, SPREAD_PRICE_DEFAULT


def no_vig_probabilities(odds_a, odds_b):
    """Real two-sided de-vig: convert both sides' American odds to vigged
    implied probability, then normalize them to sum to 1. Returns
    (prob_a, prob_b)."""
    p_a = moneyline_to_implied_probability(odds_a)
    p_b = moneyline_to_implied_probability(odds_b)
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


def price_to_decimal_odds(price, fee=POLYMARKET_FEE):
    """Real no-vig price (0-1) -> decimal odds, with a real fee applied to
    the WINNINGS portion only (the stake itself is always returned on a
    win - a flat fee on total payout would make a price near 1.0 produce
    decimal odds below 1.0, which isn't a real possible price)."""
    price = min(max(float(price), 1e-6), 1 - 1e-6)
    raw_decimal = 1.0 / price
    return 1.0 + (raw_decimal - 1.0) * (1.0 - fee)


def get_market_price(event, market, side):
    """Real market price for one side of one market on one event.

    market: "moneyline" | "spread". side: "home" | "away".
    Returns None if the real data needed isn't present (e.g. no real
    odds collected yet for this game) - callers must handle that, not
    assume a price always exists.
    """
    if market == "moneyline":
        home_ml, away_ml = event.get("home_moneyline"), event.get("away_moneyline")
        if home_ml is None or away_ml is None:
            return None
        p_home, p_away = no_vig_probabilities(home_ml, away_ml)
        price = p_home if side == "home" else p_away
        american = home_ml if side == "home" else away_ml
        return {
            "price": price,
            "american": american,
            "decimal_odds": price_to_decimal_odds(price),
            "line": None,
            "source": event.get("vegas_line_source"),
            "observed_at": event.get("vegas_spread_observed_at"),
            "used_default_spread_juice": False,
        }

    if market == "spread":
        line = event.get("vegas_spread")
        if line is None:
            return None
        home_odds = event.get("home_spread_odds")
        away_odds = event.get("away_spread_odds")
        used_default = event.get("used_default_spread_juice", False)
        if home_odds is None or away_odds is None:
            home_odds = away_odds = SPREAD_PRICE_DEFAULT
            used_default = True
        p_home, p_away = no_vig_probabilities(home_odds, away_odds)
        price = p_home if side == "home" else p_away
        american = home_odds if side == "home" else away_odds
        # Real sign bug fixed here (2026-10-20): sportsbook convention is
        # negative = favorite. Our stored vegas_spread is home-perspective
        # with the OPPOSITE sign (positive = home favored) - so the home
        # side's own sportsbook-convention line is -vegas_spread (a home
        # favorite, vegas_spread > 0, must show negative), and the away
        # side's is +vegas_spread. The old `line if side=="home" else
        # -line` had this backwards - a favored side was shown with a
        # positive line, directly contradicting this project's own
        # declared "spread_convention": "sportsbook (negative = favorite)"
        # header. Never affected EV/edge/staking (those use `price`/
        # `decimal_odds`, not this field) - purely a display-field bug,
        # but a real one for anyone (or anything, e.g. the weekly results
        # tracker) reading the exported line at face value.
        side_line = -line if side == "home" else line
        return {
            "price": price,
            "american": american,
            "decimal_odds": price_to_decimal_odds(price),
            "line": side_line,
            "source": event.get("vegas_line_source"),
            "observed_at": event.get("vegas_spread_observed_at"),
            "used_default_spread_juice": bool(used_default),
        }

    raise ValueError(f"Unknown market: {market!r}")
