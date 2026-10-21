"""Real config for the juke weekly picks competition - every constant the
pasted spec named, in one place. Nothing here is fit from data (the
things that ARE fit - sigma, the market-shrinkage weight w - live in
calibration.py's own output, not as hardcoded numbers here)."""

TOTAL_CREDITS = 1000
ALLOWED_MARKETS = ["moneyline", "spread"]
MIN_EV = 0.03
KELLY_FRACTION = 0.25
MAX_BET_PCT = 0.15
MIN_PICKS = 7
MAX_PICKS = 10  # a real ceiling, not a target - fewer real qualifying picks is fine
POLYMARKET_FEE = 0.0  # reduces payout; set once a real fee schedule is known
SPREAD_PRICE_DEFAULT = -110  # only used when a real per-side spread price is missing

# Real production O/D-Elo k-factor (generate_dashboard_data_2026.OD_K_FACTOR) -
# reused here, not re-grid-searched, so the walk-forward calibration fits
# the exact same model that actually produces our_spread/win_prob_home.
OD_K_FACTOR = 180.0

SPORTSBOOK_CONVENTION_NOTE = "sportsbook (negative = favorite)"
SETTLEMENT_NOTE = "polymarket market odds"
PRICE_SOURCE_NOTE = "ESPN no-vig proxy for Polymarket"
