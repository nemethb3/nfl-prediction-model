"""Shared constants used across multiple src/ modules.

Audit finding (AUDIT_2026-07-25.md, Technical Debt #3): BLEND_RATIO_BY_POSITION
was copy-pasted into team_aggregation.py, sos_adjustment.py, and
phase3_diagnostic.py independently, and had already diverged - sos_adjustment.py's
copy was missing the "LB" entry (harmless today since that module's only
consumer only loops CB/S, but a real KeyError risk the moment anyone extends
it to LB). Centralized here so there is exactly one copy to keep correct.
"""

# CB/S/LB winning blend ratios (tackle_weight, leverage_weight) from Phase 2
# Refinement Task 2's holdout search (LB matches CB - both landed at the edge
# of the tested grid; S found a genuine interior optimum).
BLEND_RATIO_BY_POSITION = {"CB": (0.8, 0.2), "S": (0.5, 0.5), "LB": (0.8, 0.2)}
