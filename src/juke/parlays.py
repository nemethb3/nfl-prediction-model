"""Real parlay construction - combines 2+ single-leg options from
DIFFERENT games into one combined bet, under a disclosed, standard
simplifying assumption: legs from different games are independent (no
real correlation data between e.g. divisional-weather or early-slate/
late-slate effects is modeled here - a real, disclosed limitation, not a
hidden one).

Parlay probability = product of leg p_finals. Parlay decimal odds =
product of leg decimal odds (the standard, correct way odds compose
across independent legs - no additional vig is layered on here, unlike a
real sportsbook parlay, since our prices are already no-vig). Parlay EV
per credit uses the exact same formula as a single bet, on the combined
probability/odds.
"""

from itertools import combinations

from juke.valuation import expected_value_per_credit


def build_parlay(legs):
    """legs: list of option dicts (each with p_final, decimal_odds,
    game_id). Real, disclosed requirement: every leg must be from a
    DIFFERENT real game - a parlay combining two bets on the same game's
    outcome isn't independent and isn't offered as a parlay leg pair by
    any real book."""
    game_ids = {leg["game_id"] for leg in legs}
    if len(game_ids) != len(legs):
        raise ValueError("Real parlay legs must each be from a different game.")
    p_final = 1.0
    decimal_odds = 1.0
    for leg in legs:
        p_final *= leg["p_final"]
        decimal_odds *= leg["decimal_odds"]
    ev = expected_value_per_credit(p_final, decimal_odds)
    return {
        "legs": legs,
        "game_ids": sorted(game_ids),
        "p_final": p_final,
        "decimal_odds": decimal_odds,
        "ev": ev,
    }


def best_parlays(options, max_legs=3, top_n=5):
    """Real, exhaustive-within-reason search: every 2-leg and 3-leg
    combination of the given (already at-most-one-per-game) options,
    ranked by EV per credit - capped to the top `top_n` so this stays
    cheap even with a full week's worth of candidates (15 games -> 105
    two-leg combos, 455 three-leg combos; still trivial)."""
    results = []
    for n_legs in range(2, max_legs + 1):
        for combo in combinations(options, n_legs):
            results.append(build_parlay(list(combo)))
    results.sort(key=lambda p: p["ev"], reverse=True)
    return results[:top_n]
