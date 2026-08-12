# src/archive/

Early-phase experiment scripts, moved here 2026-07-30 (Comprehensive Fix
task, following the 2026-07-30 audit). None of these are imported by any
live pipeline script (confirmed by an import-graph sweep across all of
`src/` during the audit) - the real, validated approaches these scripts
explored either lost a real backtest to a simpler method (see
`momentum_weighting.py`/`rest_tracking.py`, whose real "tested, not
integrated" findings are still cited in the dashboard's Model Transparency
section) or were superseded by later, consolidated scripts as this
project's pipeline evolved through its "Phase" development history.

Kept, not deleted: several of these document real, disclosed "tested and
rejected" findings this project has consistently valued keeping visible
rather than quietly dropping.

Contents:
- `momentum_weighting.py` - Phase 3 Component 3.1, recency weighting
- `rest_tracking.py` - Phase 3 Component 3.2, rest-day tracking
- `ensemble.py` - season-level wins + game-level spreads ensemble
- `ensemble_model.py` - Phase A Component 2, EPA-Elo ensemble
- `dynamic_tracking.py` - dynamic season-win tracking
- `weekly_tracking.py` - Phase 1 Component 1.1, weekly tracking infra
- `playoff_probability.py` - Phase 1 Component 1.3, playoff probability calc
- `player_impact.py` - Wins Above Replacement (WAR) player impact
- `integrated_predictions.py` - integration & baseline validation
- `qb_elo_model.py` - Phase 2 Component 2.2, QB-specific Elo model

To run any of these standalone: `python src/archive/<script_name>.py`
(each may need its own real data inputs - see individual docstrings).
