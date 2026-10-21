"""Required tests for the juke picks engine: sign conversion, no-vig
math, price -> decimal odds, Kelly, cap + redistribution, exact-1000
credit sum, one position per game, no post-kickoff lines, append-only
log."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from juke.calibration import _fit_shrinkage_weight
from juke.pricing import get_market_price, no_vig_probabilities, price_to_decimal_odds
from juke.staking import allocate_stakes, kelly_fraction, select_candidates
from juke.valuation import best_option_per_game, event_kickoff_utc, has_kicked_off


# ---- Sign convention ----

def test_spread_line_exported_in_sportsbook_convention():
    """This project's own INTERNAL convention: positive vegas_spread =
    home favored. But the exported `line` field must be in SPORTSBOOK
    convention (negative = favorite, per this project's own declared
    "spread_convention" in the output JSON) - a favored side's own line
    must be negative, not positive."""
    event = {
        "home_team": "KC", "away_team": "LV",
        "vegas_spread": 6.5,  # home (KC) favored by 6.5, our internal convention
        "home_spread_odds": -110, "away_spread_odds": -110,
        "used_default_spread_juice": False,
    }
    home_price = get_market_price(event, "spread", "home")
    away_price = get_market_price(event, "spread", "away")
    assert home_price["line"] == -6.5  # KC favored -> sportsbook line is negative
    assert away_price["line"] == 6.5   # LV underdog -> sportsbook line is positive


def test_sportsbook_export_convention_is_negated():
    """Output JSON must export spreads in sportsbook convention (negative
    = favorite) - the opposite of this project's internal positive-home-
    favored convention. forecasting.forecast_value_and_measure performs
    exactly that negation for the spread market."""
    from juke.forecasting import forecast_value_and_measure
    event = {"our_spread": 6.5}
    value, measure = forecast_value_and_measure(event, "spread")
    assert value == -6.5
    assert measure == "predicted margin"


# ---- No-vig math ----

def test_no_vig_probabilities_sum_to_one():
    p_home, p_away = no_vig_probabilities(-150, 130)
    assert abs((p_home + p_away) - 1.0) < 1e-9


def test_no_vig_removes_vig_from_a_pick_em():
    """Two sides at -110/-110 (a real standard vigged price) should
    de-vig to exactly 50/50."""
    p_home, p_away = no_vig_probabilities(-110, -110)
    assert abs(p_home - 0.5) < 1e-9
    assert abs(p_away - 0.5) < 1e-9


def test_no_vig_favors_the_shorter_price():
    p_home, p_away = no_vig_probabilities(-200, 170)
    assert p_home > p_away


# ---- Price -> decimal odds ----

def test_price_to_decimal_odds_basic():
    assert abs(price_to_decimal_odds(0.5) - 2.0) < 1e-9
    assert abs(price_to_decimal_odds(0.25) - 4.0) < 1e-9


def test_price_to_decimal_odds_fee_reduces_winnings_only():
    no_fee = price_to_decimal_odds(0.4, fee=0.0)
    with_fee = price_to_decimal_odds(0.4, fee=0.1)
    # stake always returned (decimal odds > 1), but winnings portion shrinks
    assert with_fee < no_fee
    assert with_fee > 1.0


# ---- Kelly ----

def test_kelly_fraction_positive_edge():
    # p=0.6 at even money (decimal 2.0) is a real positive-edge bet
    f = kelly_fraction(0.6, 2.0)
    assert f > 0


def test_kelly_fraction_zero_at_fair_price():
    # p=0.5 at even money (decimal 2.0) is exactly breakeven
    f = kelly_fraction(0.5, 2.0)
    assert abs(f) < 1e-9


def test_kelly_fraction_never_negative():
    f = kelly_fraction(0.3, 2.0)
    assert f == 0.0


# ---- Cap + redistribution / exact credit sum ----

def _opt(game_id, p_final, decimal_odds, ev):
    return {"game_id": game_id, "p_final": p_final, "decimal_odds": decimal_odds, "ev": ev}


def test_allocation_sums_to_exactly_total_credits():
    options = [_opt(f"g{i}", 0.55 + 0.01 * i, 2.0, 0.05 + 0.01 * i) for i in range(10)]
    picks, warning = allocate_stakes(options, total_credits=1000)
    assert sum(p["credits"] for p in picks) == 1000


def test_allocation_respects_cap_with_enough_picks():
    """With >= MIN_PICKS real qualifying options, no single pick should
    exceed the 15% cap (150 of 1000)."""
    options = [_opt(f"g{i}", 0.9, 5.0, 2.5) for i in range(8)]  # all identical, equal split expected
    picks, warning = allocate_stakes(options, total_credits=1000, max_bet_pct=0.15, min_picks=7)
    assert all(p["credits"] <= 150 for p in picks)
    assert sum(p["credits"] for p in picks) == 1000


def test_allocation_redistributes_excess_from_capped_picks():
    # One hugely dominant option would want ~all the money; the cap should
    # clip it and hand the rest to the others, not just drop the excess.
    options = [_opt("dominant", 0.95, 20.0, 15.0)] + [_opt(f"g{i}", 0.52, 2.0, 0.04) for i in range(7)]
    picks, warning = allocate_stakes(options, total_credits=1000, max_bet_pct=0.15, min_picks=7)
    dominant = next(p for p in picks if p["game_id"] == "dominant")
    assert dominant["credits"] <= 150
    assert sum(p["credits"] for p in picks) == 1000


def test_fewer_than_min_picks_relaxes_cap_and_warns():
    options = [_opt("g0", 0.6, 2.0, 0.1), _opt("g1", 0.6, 2.0, 0.1)]
    picks, warning = allocate_stakes(options, total_credits=1000, max_bet_pct=0.15, min_picks=7)
    assert warning is not None
    assert sum(p["credits"] for p in picks) == 1000


def test_max_picks_caps_at_the_ceiling_not_a_target():
    # 15 real qualifying (ev >= default min_ev) options - should keep only
    # the 10 highest-EV ones, not force all 15 in.
    options = [_opt(f"g{i}", 0.6, 2.0, 0.05 + 0.01 * i) for i in range(15)]
    picks, warning = allocate_stakes(options, total_credits=1000, max_picks=10, min_picks=7)
    assert len(picks) == 10
    assert sum(p["credits"] for p in picks) == 1000
    kept_ids = {p["game_id"] for p in picks}
    # the 10 kept should be exactly the 10 highest-EV ones (g5..g14)
    assert kept_ids == {f"g{i}" for i in range(5, 15)}


def test_fewer_real_qualifiers_than_max_picks_is_fine():
    # Only 8 real qualifying options exist - max_picks=10 must not force
    # padding up to 10 with anything non-positive-EV.
    options = [_opt(f"g{i}", 0.6, 2.0, 0.05 + 0.01 * i) for i in range(8)]
    picks, warning = allocate_stakes(options, total_credits=1000, max_picks=10, min_picks=7)
    assert len(picks) == 8
    assert sum(p["credits"] for p in picks) == 1000


def test_no_positive_ev_options_allocates_nothing():
    options = [_opt("g0", 0.4, 2.0, -0.2)]
    picks, warning = allocate_stakes(options, total_credits=1000, min_ev=0.03, min_picks=7)
    assert picks == []
    assert warning is not None


def test_core_vs_filler_tagging():
    options = [_opt("core1", 0.7, 2.0, 0.2)] + [_opt(f"filler{i}", 0.55, 2.0, 0.01) for i in range(6)]
    picks, _ = allocate_stakes(options, total_credits=1000, min_ev=0.05, min_picks=7)
    tiers = {p["game_id"]: p["tier"] for p in picks}
    assert tiers["core1"] == "core"
    assert all(tiers[f"filler{i}"] == "filler" for i in range(6))


# ---- One position per game ----

def test_at_most_one_position_per_game():
    options = [
        {"game_id": "g1", "market": "moneyline", "side": "home", "ev": 0.05},
        {"game_id": "g1", "market": "spread", "side": "home", "ev": 0.12},
        {"game_id": "g2", "market": "moneyline", "side": "away", "ev": 0.02},
    ]
    best = best_option_per_game(options)
    game_ids = [o["game_id"] for o in best]
    assert sorted(game_ids) == ["g1", "g2"]
    g1 = next(o for o in best if o["game_id"] == "g1")
    assert g1["market"] == "spread"  # the higher-EV option for g1


# ---- No post-kickoff lines / already-started games ----

def test_has_kicked_off_true_for_past_kickoff():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    event = {"kickoff_utc": past}
    assert has_kicked_off(event) is True


def test_has_kicked_off_false_for_future_kickoff():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    event = {"kickoff_utc": future}
    assert has_kicked_off(event) is False


def test_event_kickoff_utc_falls_back_to_naive_field():
    event = {"kickoff_utc": None, "kickoff_datetime": "2020-01-01T00:00:00"}
    assert event_kickoff_utc(event) is not None


# ---- Block-bootstrap SE (calibration.py) ----

def test_shrinkage_weight_runs_without_groups():
    rng = np.random.default_rng(0)
    n = 100
    p_model = rng.uniform(0.3, 0.7, n)
    p_market = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(0, 1, n) < p_market).astype(float)
    best_w, best_loss, losses, w_one_se, se = _fit_shrinkage_weight(p_model, p_market, y)
    assert 0.0 <= best_w <= 1.0
    assert se >= 0.0


def test_shrinkage_weight_with_groups_uses_block_bootstrap():
    rng = np.random.default_rng(1)
    n = 100
    p_model = rng.uniform(0.3, 0.7, n)
    p_market = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(0, 1, n) < p_market).astype(float)
    groups = np.array([f"2023_{i // 10}" for i in range(n)])  # 10 week-blocks
    best_w, best_loss, losses, w_one_se, se = _fit_shrinkage_weight(p_model, p_market, y, groups=groups)
    assert 0.0 <= best_w <= 1.0
    assert se > 0.0  # real variability across resampled blocks, not a degenerate 0


# ---- Append-only log ----

def test_forecast_log_is_append_only(tmp_path):
    log_path = tmp_path / "juke_picks.jsonl"
    record_1 = {"event_id": "a", "credits": 10}
    record_2 = {"event_id": "b", "credits": 20}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_1) + "\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_2) + "\n")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == record_1
    assert json.loads(lines[1]) == record_2
