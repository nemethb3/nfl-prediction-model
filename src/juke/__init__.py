"""Juke paper-trading weekly picks engine.

Separate modules on purpose (see generate_juke_picks.py's own module
docstring for the real reason): forecasting / pricing / valuation /
staking / output, each with a narrow interface, so a live/in-game
forecaster or a real Polymarket price feed can be swapped in later
without touching the others.
"""
