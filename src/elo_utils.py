"""Shared Elo carryover computation, no fabrication.

Extracted after the real AUDIT_2026-08-12_DEEP.md Section 2.1 Elo bug fix
left near-duplicate logic in `elo_game_prediction.py` (the `season > 2025`
branch of `generate_elo_game_spreads`) and `simulate_2026_playoffs.py`
(`real_2026_carryover_elo`) - both compute the same real thing (a team's
carryover Elo entering a future, unplayed season) the same way, just for
different callers. Centralized here so there's exactly one copy to keep
correct, matching this project's established `constants.py` precedent for
exactly this kind of duplication.

Lives in its own module (not `constants.py`, which holds data, not
functions) specifically so both `elo_game_prediction.py` and `simulate_
2026_playoffs.py` can import it without a circular dependency -
`simulate_2026_playoffs.py` already imports FROM `elo_game_prediction.py`
(`ELO_HOME_FIELD`, `calculate_win_probability_from_elo`), so this module
must not import from either of them.
"""


def compute_season_start_elo(earliest_season, target_season, k_factor, home_field_elo):
    """Real carryover Elo entering `target_season` - the correct rating
    chained through every real game from `earliest_season` up through
    `target_season - 1`, with `run_multi_season_elo`'s standard 1/3
    season-boundary regression applied at the `target_season` boundary
    too.

    The real bug this fixes (AUDIT_2026-08-12_DEEP.md Section 2.1):
    `run_multi_season_elo`'s `ratings_at_season_start` dict entry for a
    given season is captured AT THE START of that season's loop
    iteration - so `ratings_at_season_start[target_season]` is only
    populated (and only regressed for the target_season boundary) if
    `target_season` itself is included in the `seasons` range passed in.
    Passing `range(earliest_season, target_season)` (excluding
    target_season) - the original bug, and also what the first draft of
    this shared utility proposed - silently returns target_season-1's
    rating instead.

    Callers pass their own already-defined `earliest_season`/`k_factor`/
    `home_field_elo` explicitly rather than this module importing them
    itself, to avoid assuming which module's copy of those constants is
    canonical (elo_game_prediction.py defines ELO_EARLIEST_SEASON
    locally; simulate_2026_playoffs.py derives the same real value from
    elo_model.TRAIN_SEASONS - both resolve to 2015, verified, but neither
    is "more canonical" than the other)."""
    from elo_model import run_multi_season_elo

    _, ratings_at_season_start, _ = run_multi_season_elo(
        range(earliest_season, target_season + 1), k_factor=k_factor, home_field_elo=home_field_elo)
    return ratings_at_season_start[target_season]
