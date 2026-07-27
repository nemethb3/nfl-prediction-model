# NFL Win Prediction Model

A dual-track model for NFL season win totals and weekly game outcomes:
a detailed, player-level EPA aggregation pipeline, and a simpler Elo rating
system built alongside it — both benchmarked against real historical
outcomes and real Vegas lines. See `PROGRESS.md` for the full, chronological
build log this README summarizes.

## Key Findings (real, backtested results — see PROGRESS.md for full detail)

1. **Elo beats the EPA pipeline, at both levels tested.**
   - Season wins (real 2025): carryover Elo corr=+0.316, MAE=2.74 vs. EPA corr=+0.216, MAE=2.88.
   - Game spreads (real 2024 holdout): Elo corr=+0.393, MAE=10.21 vs. EPA corr=+0.255, MAE=10.82. Replicated on real 2025 (corr=+0.385, MAE=10.36).
   - "Elo" here means *carryover* Elo — real win/loss history only, no Vegas signal anywhere. A Vegas-informed Elo variant also exists internally but is circular (it imports Vegas's own number as a starting rating) and isn't a fair comparison point.

2. **Vegas beats everything, at every point tested — including late season.** A LOOCV search over Vegas/Elo blend weights at 5 checkpoints (weeks 1, 4, 8, 12, 16) independently chose **100% Vegas, 0% Elo at every single checkpoint**, contradicting the original hypothesis that Elo should take over as the season progresses. Real Vegas spread accuracy (2025): corr=+0.504, MAE=9.72 (game-level); corr=+0.798, MAE=1.78 (season win totals — a different market, kept as a separate figure, see `src/constants.py`).

3. **Simple weighted blending never beats using the single strongest signal alone.** This pattern showed up independently four separate times: an EPA-candidates ensemble collapsed to Vegas alone; an EPA+Elo season-win ensemble collapsed to pure Elo (LOOCV selected 0% EPA weight in all 32 leave-one-out folds); the Vegas/Elo game-spread blend collapsed to pure Vegas at every checkpoint. Related, correlated signals don't cancel error the way true ensembling assumes.

4. **Weekly recalibration works, but modestly, and the honest number is lower than a naive check suggests.** Freezing Elo at week N and projecting all remaining games (the real, live-deployment scenario) improves correlation monotonically through the season: +0.248 (week 1) → +0.412 (week 16, real 2025). This is meaningfully lower than a look-ahead-chained per-game accuracy figure (+0.385) — the difference matters and is documented in `PROGRESS.md` (Component B).

5. **Fantasy viability is position-dependent, not uniform.** Real 2025 PPR correlation: WR +0.591 (strong), QB +0.435 and TE +0.436 (borderline), **RB −0.504 (broken by the specified EPA×volume formula)** — raw prior-season opportunity volume *alone* would have scored +0.667 for RB, meaningfully better than the specified formula. Not evidence fantasy-from-EPA is hopeless; evidence the RB formula specifically needs rework.

## Architecture

Two related but independent prediction pipelines share the same raw data:

```
nflreadpy / nfl_data_py (play-by-play, weekly stats, schedules, Vegas lines)
    |
    +-- EPA / player pipeline (original, pre-Elo):
    |     data_pipeline.py -> player_models.py / team_aggregation.py / sos_adjustment.py
    |     -> team_strength_{season}.csv -> epa_to_wins.py / game_predictions.py
    |     -> season win + game spread projections (the EPA baseline cited above)
    |     -> fantasy_validation.py (fantasy rankings use THIS pipeline's player
    |        projection files, independent of Elo)
    |
    +-- Elo pipeline (this session, beats EPA at both levels):
          elo_model.py (team Elo ratings, carryover + Vegas-informed variants)
          -> elo_game_prediction.py (Elo -> game spreads, real prob->spread fit)
          -> weekly_recalibration.py (in-season Elo updates, real per-game chain)
          -> vegas_integration_optimized.py (learned Vegas/Elo blend weights -
             found blending doesn't help; Vegas wins outright)
```

`src/constants.py` centralizes the shared Elo hyperparameters and all
EPA/Elo/Vegas baseline numbers cited above — see it directly for the
authoritative, current values rather than trusting any number restated here
to stay current.

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data Sources

- **Play-by-play, snap counts, player stats**: [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (nflverse data); `nflreadpy` additionally used where `nfl_data_py`'s 2025-season endpoints were unavailable (returns polars DataFrames, converted to pandas).
- **Vegas lines**: real historical spreads/moneylines/totals from the nflverse schedules data (`schedules_2015_2025.csv`, `schedules_2026.csv`) — not a proxy.
- **Game results**: same schedules data (`home_score`/`away_score`) plus `data/backtest/game_results_2015_2025.csv`.

## Running the Model

```bash
# EPA / player pipeline
python src/data_pipeline.py          # raw data -> player-level EPA features
python src/epa_to_wins.py            # team strength -> season win projections
python src/game_predictions.py       # team strength -> game spreads

# Elo pipeline (currently the more accurate of the two)
python src/elo_model.py              # build + validate Elo ratings
python src/elo_game_prediction.py    # Elo -> game spreads, validate vs. EPA
python src/weekly_recalibration.py   # simulate in-season weekly Elo updates
python src/vegas_integration_optimized.py  # learn Vegas/Elo blend weights

# Fantasy
python src/fantasy_validation.py     # validate EPA projections against real fantasy output
```

## Project Structure

```
data/
  raw/         # Unmodified downloads (see data/README.md)
  processed/   # Cleaned, aggregated, model-ready data + predictions
  backtest/    # Historical results used for validation
  diagnostic/  # Validation/accuracy outputs
  fantasy/     # Fantasy-specific outputs
src/           # Pipeline, feature engineering, both prediction tracks
models/        # Trained model artifacts (.pkl)
```

`dashboard.py`, `backtest.py` (root), and `automated_pipeline.py` were empty
scaffolding from the initial project setup and have been removed (see
`AUDIT_2026-07-27.md`) — none had ever been implemented. They'll be
recreated when that work is actually scoped (see Next Steps).

## Known Limitations

- **Compression**: team-strength/win projections from the EPA pipeline are measured at roughly 3.3x too narrow vs. real season-to-season variance. Confirmed this cannot be fixed by simple rescaling (proven this session — rescaling amplifies noise faster than signal once real correlation is this weak).
- **Confidence interval calibration is uneven, not uniformly good.** A percentile-across-correlated-candidates CI method was found badly miscalibrated (12.5% actual coverage vs. 90% target) and abandoned. Direct residual-std bands from real regression fits (used in the Elo game-spread work) calibrate well (~89-90% coverage). Season-total tracking CIs self-correct over the season but start poorly calibrated early (~59-63% coverage through week 8).
- **Fantasy RB formula is currently broken** (see Key Finding 5) — needs a different formula before use, not currently fixed.
- No injury/availability-severity model yet.
- No QB-specific Elo yet.
- No unit tests anywhere in the codebase — validation has been via direct backtesting against real historical outcomes throughout, not unit tests.

## Next Steps

- [ ] Fix RB fantasy formula (raw opportunity volume outperforms the current EPA x volume approach)
- [ ] Injury/availability severity model
- [ ] QB-specific Elo
- [ ] Matchup-specific features
- [ ] Weekly tracking & publication infrastructure (predictions vs. results, logged weekly)
- [ ] Playoff probability calculator
- [ ] Dashboard / API (not started — see Known Limitations)

## Technical Notes

- Language: Python 3.11+
- Core libraries: pandas, numpy, scikit-learn, scipy (see `requirements.txt`)
- Validation convention throughout: fit/derive on one real historical range, validate against a genuinely separate real holdout (commonly 2015-2023/2024 train vs. 2024/2025 holdout) — constants and coefficients are derived by regression/search on real data rather than asserted, per this project's standing convention.
