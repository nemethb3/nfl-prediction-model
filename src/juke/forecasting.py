"""Real model probabilities - this project's own already-fitted O/D-Elo
outputs (win_prob_home/win_prob_away, our_spread), not a new model.

Sign convention (verified against betting_backtest.py/join_espn_odds_
2026.py, same convention this whole project uses): positive our_spread/
vegas_spread = HOME team favored by that many points. Every conversion
below is unit-tested against that convention (see
tests/test_juke_picks.py).
"""

from scipy.stats import norm


def _ensemble_win_prob_home(event, elo_ensemble):
    od_p = event.get("win_prob_home")
    single_p = event.get("single_elo_win_prob_home")
    if od_p is None:
        return None
    if single_p is None or elo_ensemble is None:
        return od_p  # real, disclosed fallback - single-Elo not available for this event
    e = elo_ensemble["e_moneyline"]
    return e * od_p + (1 - e) * single_p


def _ensemble_predicted_margin(event, elo_ensemble):
    od_margin = event.get("our_spread")
    single_margin = event.get("single_elo_spread")
    if od_margin is None:
        return None
    if single_margin is None or elo_ensemble is None:
        return od_margin
    e = elo_ensemble["e_spread"]
    return e * od_margin + (1 - e) * single_margin


def model_probability(event, market, side, sigma, elo_ensemble=None):
    """Real model-implied probability of `side` winning (moneyline) or
    covering (spread) this event. sigma is the real, walk-forward
    out-of-sample residual std of (actual_margin - predicted_margin) -
    always passed in from calibration.py's fitted output, never
    hardcoded here.

    elo_ensemble (optional, from calibration.fit_elo_ensemble): when
    given, blends O/D-Elo with this project's OTHER already-fitted game
    model (single-Elo) using a real, data-driven weight, instead of
    using O/D-Elo alone - "use as much of the models we've already built
    as possible," per the user's own instruction. Falls back to pure
    O/D-Elo when the event has no real single_elo_* fields (or no
    ensemble weights were given) - never raises."""
    if market == "moneyline":
        p_home = _ensemble_win_prob_home(event, elo_ensemble)
        if p_home is None:
            return None
        return p_home if side == "home" else 1.0 - p_home

    if market == "spread":
        predicted_margin = _ensemble_predicted_margin(event, elo_ensemble)
        line = event.get("vegas_spread")
        if predicted_margin is None or line is None:
            return None
        # P(home margin > line) under actual_margin ~ Normal(predicted_margin, sigma)
        p_home_covers = float(norm.cdf((predicted_margin - line) / sigma))
        return p_home_covers if side == "home" else 1.0 - p_home_covers

    raise ValueError(f"Unknown market: {market!r}")


def forecast_value_and_measure(event, market):
    """Real (forecast_value, forecast_measures) pair for the output JSON -
    the model's own predicted margin (sportsbook convention, negative =
    favorite) for spread picks, or its win probability for moneyline
    picks."""
    if market == "moneyline":
        return event.get("win_prob_home"), "win probability"
    if market == "spread":
        our_spread = event.get("our_spread")
        sportsbook_margin = -our_spread if our_spread is not None else None
        return sportsbook_margin, "predicted margin"
    raise ValueError(f"Unknown market: {market!r}")
