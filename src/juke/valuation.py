"""Edge / EV per credit, and the real "at most one position per game"
rule (moneyline and spread on the same team are highly correlated bets
on the same underlying game outcome - see generate_juke_picks.py's own
module docstring for why only the highest-EV option per game survives).
"""

from datetime import datetime, timezone

from run_provenance import parse_any_iso_to_utc


def expected_value_per_credit(p_final, decimal_odds):
    """Real EV per credit staked: p * (payout_per_credit_won) - (1-p) * (credit lost)."""
    b = decimal_odds - 1.0
    return p_final * b - (1.0 - p_final)


def event_kickoff_utc(event):
    """Real, best-available UTC kickoff for an exclude-if-already-started
    check: prefers the real ESPN-sourced kickoff_utc (tz-aware), falls
    back to the naive kickoff_datetime (assumed local system time, same
    real convention run_provenance.parse_any_iso_to_utc already uses
    elsewhere in this project) only if the real one isn't available yet."""
    for key in ("kickoff_utc", "kickoff_datetime"):
        parsed = parse_any_iso_to_utc(event.get(key))
        if parsed is not None:
            return parsed
    return None


def has_kicked_off(event, now=None):
    now = now or datetime.now(timezone.utc)
    kickoff = event_kickoff_utc(event)
    return kickoff is not None and kickoff <= now


def best_option_per_game(options):
    """Real "at most one position per game" selection: given a list of
    candidate-option dicts (each with a real 'game_id' and 'ev' key),
    keep only the single highest-EV option for each real game_id."""
    best = {}
    for opt in options:
        key = opt["game_id"]
        if key not in best or opt["ev"] > best[key]["ev"]:
            best[key] = opt
    return list(best.values())
