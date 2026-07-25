"""win_projection.py - backward-compatible re-export shim.

This module used to contain 5 unrelated jobs in one ~700-line file (season
win projection, backtest comparison, the game prediction engine, Vegas
comparison/edge detection, and season win inference from game predictions -
see AUDIT_2026-07-25.md Technical Debt #5). Split (Master Plan Phase 4 Task
4.2, no behavior change) into:

  - epa_to_wins.py        - Task 4.1's EPA->win_pct conversion
  - backtest.py            - Task 4.2's backtest vs. actual results
  - game_predictions.py    - Phase 3 Redesign Subtasks 1 & 4 (game engine,
                              season win inference from game predictions)
  - vegas_comparison.py    - Task 5 & Subtask 3 (season + game-level Vegas
                              comparison, edge detection)

Everything is re-exported here so existing callers (e.g.
weekly_predictions.py's `from win_projection import build_game_prediction_
engine`) keep working unchanged.
"""

from epa_to_wins import (
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, BACKTEST_DIR, MODELS_DIR,
    TRAIN_SEASONS, BACKTEST_YEARS, TARGET_SEASON,
    compute_real_win_pct, build_historical_epa_wins_dataset, backtest_epa_to_wins_model,
    fit_final_model, project_season_wins, validate_win_projections, run_win_projection,
)
from backtest import (
    load_2025_actual_results, compare_projections_vs_actual, analyze_compression_effect,
    run_backtest_comparison,
)
from game_predictions import (
    fit_game_level_epa_to_points, build_game_prediction_engine,
    infer_season_wins_from_game_predictions, run_season_win_inference, validate_season_win_inference,
)
from vegas_comparison import (
    moneyline_to_implied_prob, compute_vegas_implied_wins, validate_model_against_vegas,
    vegas_comparison_framework, identify_edges,
)

if __name__ == "__main__":
    run_win_projection()
    run_backtest_comparison()
