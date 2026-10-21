"""Real cross-check against betting_backtest.py's already-validated "Our
System" heuristic - the SAME strategy that's showing real positive ROI
in the live Betting Analysis dashboard for 2026 so far (10.2% moneyline,
5.5% ATS on 40 real bets as of this writing). Not re-derived here -
betting_backtest.py's own should_bet/get_bet_direction are imported and
called directly, so this can never silently drift from what the
dashboard is actually reporting.

Per the user's own instruction ("we have a positive ROI... so let's keep
building on it"): this is used as an additional, disclosed CONFIRMATION
signal alongside the juke engine's own EV/Kelly selection, not a
replacement for it - see engine.py for how it's combined, and REPORT.md
for whether it measurably helped in the walk-forward backtest.
"""

from betting_backtest import get_bet_direction, moneyline_to_implied_probability, should_bet


def our_system_pick(event):
    """Real replication of betting_backtest._our_system_direction, using
    this event's own real win_prob_home/win_prob_away/home_moneyline/
    away_moneyline/our_spread/vegas_spread. Returns the side ('home' |
    'away') Our System would bet, or None if its own real entry gate
    (>2% moneyline probability disagreement) isn't met, or its spread-
    extremity direction logic doesn't resolve (our_spread/vegas_spread
    disagree on which side is favored - a real, rare case)."""
    home_ml, away_ml = event.get("home_moneyline"), event.get("away_moneyline")
    our_spread, vegas_spread = event.get("our_spread"), event.get("vegas_spread")
    win_prob_home = event.get("win_prob_home")
    if None in (home_ml, away_ml, our_spread, vegas_spread, win_prob_home):
        return None

    vegas_prob_home = moneyline_to_implied_probability(home_ml)
    vegas_prob_away = moneyline_to_implied_probability(away_ml)
    win_prob_away = 1.0 - win_prob_home

    if not (should_bet(win_prob_home, vegas_prob_home) or should_bet(win_prob_away, vegas_prob_away)):
        return None
    return get_bet_direction(our_spread, vegas_spread)
