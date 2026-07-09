# NFL Win Prediction Model

A player-level bottom-up model for predicting NFL team win totals and weekly
game outcomes, benchmarked against Vegas lines to surface betting edges.

## Overview

The model works bottom-up:

1. Collect historical play-by-play, snap count, and player stat data (2015-2025).
2. Clean and standardize player identities, positions, and metadata.
3. Train position-level models (QB, WR, RB, EDGE, ...) to project 2026 player output.
4. Aggregate player projections into team offensive/defensive strength.
5. Convert team strength into win-total and game-by-game win probability models.
6. Compare model output to Vegas lines to flag edges.
7. Backtest the full pipeline on the 2024/2025 seasons to validate edge quality.
8. Serve results via a Streamlit dashboard, refreshed weekly during the season.

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data Sources

- **Play-by-play, snap counts, player stats**: [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (nflverse data)
- **Vegas lines**: best-effort historical lines (see `src/data_pipeline.py` / vegas module notes on source + proxy quality)
- **Game results**: `nfl_data_py.import_schedules` / `import_games`

## Project Structure

```
data/
  raw/         # Unmodified downloads
  processed/   # Cleaned, aggregated, model-ready data
  backtest/    # Historical results used for validation
src/           # Pipeline, feature engineering, models
models/        # Trained model artifacts (.pkl)
notebooks/     # Exploration / validation notebooks
dashboard.py   # Streamlit app
backtest.py    # Historical validation harness
automated_pipeline.py  # Weekly in-season refresh
```

## Timeline

Built in phases (see `PROGRESS.md` for current status):

1. Data pipeline & feature engineering
2. Tier 1 position models (QB, WR, RB, EDGE)
3. Team strength aggregation & win prediction models
4. Vegas backtesting & validation
5. Automation for weekly in-season refresh
6. Dashboard & polish
