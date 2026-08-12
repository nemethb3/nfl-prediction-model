# NFL Prediction & Fantasy Model

A validated NFL prediction system built on real 2015-2025 data: game spread
predictions, season win projections, fantasy player rankings, and playoff
probabilities — each backtested against real historical outcomes, with
every finding (positive, negative, or null) reported honestly. See
`PROGRESS.md` for the full, chronological build log this README summarizes.

## What's Built

- **Game spreads**: Vegas lines when available (they beat everything else
  tested), with a small real matchup-EPA adjustment; Elo (+ a marginal
  QB-Elo blend) as a fallback for games with no posted line.
- **Season win projections**: real actual wins-to-date + Elo-projected
  remaining games — dramatically more accurate than the original EPA-based
  approach.
- **Fantasy rankings**: RB/QB/TE on validated volume-only formulas; WR on
  its original (still-best) combined EPA+volume formula.
- **Playoff probability**: Monte Carlo simulation (10,000 runs/checkpoint)
  over the real 7-team-per-conference NFL structure.
- **Weekly tracking**: a SQLite database logging predictions, results, and
  accuracy over time, validated end-to-end on real 2025 data.

## Current Performance (real 2025 backtests)

### Game spreads (n=272 real games)
| | Correlation | MAE |
|---|---|---|
| Integrated (Vegas + matchup adj., or Elo fallback) | +0.500 | 9.69 |
| Vegas alone | +0.504 | 9.72 |

Essentially a wash vs. Vegas alone — the matchup adjustment was validated
as a fix for *Elo's* errors specifically (+0.010 corr over Elo alone) and
doesn't clearly transfer to correcting Vegas's already-better predictions.

### Season win projections (real actual wins + Elo-projected remainder)
| Week | New (Elo-based) | Old (EPA-based) | Improvement |
|---|---|---|---|
| 1 | +0.300 | +0.069 | +0.231 |
| 4 | +0.594 | +0.275 | +0.319 |
| 8 | +0.736 | +0.297 | +0.439 |
| 12 | +0.885 | +0.369 | +0.516 |
| 16 | +0.977 | +0.412 | +0.565 |

The single most decisive improvement in the whole project — the gap grows
through the season before converging near 1.0 once real wins dominate.

### Fantasy (real 2025, PPR)
| Position | Correlation | Note |
|---|---|---|
| RB | +0.651 | was −0.504 on the original EPA×volume formula |
| TE | +0.543 | was +0.436 |
| WR | +0.591 | unchanged — the one position where EPA+volume already beats volume alone (this is the original season-total-vs-season-total validation number; the live dashboard's real per-game-week WR correlation is a separate, later-tracked figure - +0.442 as of the 2026-07-30 Phase 4 fix, see DASHBOARD_DATA_GAPS.md) |
| QB | +0.447 | was +0.435 (the most modest gain of the four) |

### Playoff probability
Real, correctly-scoped simulation: 7 playoff spots per 16-team conference
(not a pooled top-8), every remaining game simulated individually per trial
(preserving shared-opponent correlation, not independent per-team draws).
Real 2025 spot-checks: odds spread widens appropriately through the season
(std dev 0.18 → 0.47, week 1 → 16) and individual team trajectories move
sensibly with real record changes.

## Key Findings

1. **Elo beats EPA**, at both season-win and game-spread level (real
   backtests, both years tested).
2. **Vegas beats everything, at every checkpoint tested — including late
   season.** A real LOOCV search over Vegas/Elo blend weights chose 100%
   Vegas at every one of 5 checkpoints (weeks 1, 4, 8, 12, 16), contradicting
   the original hypothesis that Elo should take over as the season
   progresses.
3. **Simple weighted blending consistently loses to using the single
   strongest signal alone.** This exact pattern recurred independently at
   least four times: an EPA-candidates ensemble collapsed to Vegas; an
   EPA+Elo season-win ensemble collapsed to pure Elo; the Vegas/Elo
   game-spread blend collapsed to pure Vegas; the game-spread matchup
   adjustment helps against Elo but washes out against Vegas.
4. **Volume beats EPA×volume for fantasy at RB, QB, and TE** — WR is the
   sole exception, where the combined formula already wins. The RB fix in
   particular was dramatic (−0.504 → +0.651).
5. **Several "obvious" adjustments tested real and came back null, and were
   correctly NOT integrated rather than kept anyway:** injury-severity
   adjustments (team-level EPA effect too heterogeneous to average
   reliably, confirmed after fixing a real duplicate-counting bug), rest-day
   adjustments (zero measurable effect on real backtest accuracy), and
   betting on our own disagreements with Vegas (real backtest ROI: −36%,
   i.e. actively harmful, not just unhelpful).

## Architecture

```
nflreadpy / nfl_data_py (play-by-play, weekly stats, schedules, Vegas lines, injuries)
    |
    +-- EPA / player pipeline (original):
    |     data_pipeline.py -> player_models.py / team_aggregation.py / sos_adjustment.py
    |     -> epa_to_wins.py / game_predictions.py (season wins + game spreads, EPA baseline)
    |
    +-- Elo pipeline (beats EPA at both levels):
    |     elo_model.py (team Elo, carryover + Vegas-informed variants)
    |     -> elo_game_prediction.py (Elo -> game spreads)
    |     -> weekly_recalibration.py (in-season Elo updates)
    |     -> qb_elo_model.py (QB-specific Elo, marginal blend addition)
    |
    +-- matchup_features.py (real team-position defensive EPA edges)
    |
    +-- integrated_predictions.py (final pipeline: Vegas-primary game
    |     spreads + matchup adjustment, Elo-based season projections)
    |
    +-- fantasy_rb_formula.py / fantasy_formula_improvements.py (validated
    |     volume-only RB/QB/TE formulas) + fantasy_validation.py (WR,
    |     unchanged - see Known Gaps below for RB/QB/TE wiring status)
    |
    +-- playoff_probability.py (Monte Carlo, real conference structure)
    +-- weekly_tracking.py (SQLite prediction/accuracy log)
```

`vegas_integration_optimized.py` (LOOCV blend-weight search), `injury_model.py`,
`rest_tracking.py`, `momentum_weighting.py`, and `edge_detection.py` are all
real, complete, validated modules whose *findings* were negative/null/
mixed and were deliberately **not** wired into the pipelines above — see
Key Findings above and PROGRESS.md for the evidence behind each.

`src/constants.py` centralizes the shared Elo hyperparameters and all
EPA/Elo/Vegas/matchup baseline numbers cited above — check it directly for
the current, authoritative values rather than trusting any number restated
here to stay current.

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data Sources

- **Play-by-play, weekly stats, injuries**: [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) / `nflreadpy` (nflverse data) — `nflreadpy` used where `nfl_data_py`'s 2025-season endpoints were unavailable (returns polars, converted to pandas), and for real weekly injury reports.
- **Vegas lines**: real historical spreads/moneylines/totals from the nflverse schedules data — one closing-line snapshot per game (no intraday/opening-line history exists in this data source).
- **Game results**: same schedules data plus `data/backtest/game_results_2015_2025.csv`.

## Running the Model

```bash
# Elo pipeline (the more accurate of the two prediction tracks)
python src/elo_model.py                    # build + validate team Elo
python src/elo_game_prediction.py          # Elo -> game spreads, vs. EPA
python src/weekly_recalibration.py         # in-season Elo updates
python src/qb_elo_model.py                 # QB-specific Elo + blend

# Final, integrated pipeline (Vegas-primary spreads, Elo season projections, fantasy)
python src/integrated_predictions.py
# Real outputs (verified, not aspirational):
#   data/processed/integrated_game_predictions_2025.csv
#   data/diagnostic/integrated_season_backtest_2025.csv
# Fantasy correlations are computed and printed by this script but not yet
# persisted to their own CSV - see Known Gaps.

# Playoff odds
python src/playoff_probability.py
# Outputs: data/processed/playoff_probability_{week}.csv (per checkpoint week)

# Weekly tracking demo (real 2025 backtest through the full pipeline)
python src/weekly_tracking.py
# No CLI flags exist yet - this runs a fixed demo across checkpoint weeks.
# Outputs: data/tracking.db, data/predictions/week_{N}_{predictions,results}.csv
```

### Known Gaps (real, disclosed - not silently patched over)

- **`fantasy_validation.py` still runs the ORIGINAL EPA×volume formula**,
  not the validated volume-only RB/QB/TE formulas. Those live in
  `fantasy_rb_formula.py` / `fantasy_formula_improvements.py` and are
  exercised together (with real, correct numbers) inside
  `integrated_predictions.generate_integrated_fantasy_projections()` - but
  that function only prints/returns results, it doesn't write a combined
  rankings CSV yet. Running `python src/fantasy_validation.py` directly
  will reproduce the old, inferior numbers (e.g. RB ≈ −0.5), not the
  validated ones - don't rely on it for current fantasy output.
- `integrated_predictions.py` does not yet call `playoff_probability.py` -
  playoff odds are a real, working, separately-run pipeline, not yet part
  of the single integrated entrypoint.
- `weekly_tracking.py` has no command-line interface - live/weekly use
  currently means calling its functions directly (`save_weekly_predictions`,
  `log_weekly_results`, etc.), not a CLI.

## Project Structure

```
data/
  raw/         # Unmodified downloads (see data/README.md)
  processed/   # Cleaned, aggregated, model-ready data + predictions
  backtest/    # Historical results used for validation
  diagnostic/  # Validation/accuracy outputs
  fantasy/     # Fantasy-specific outputs
  predictions/ # Real weekly tracking outputs (Component 1.1)
src/           # Pipeline, feature engineering, both prediction tracks
src/archive/   # Early-phase experiments, not in the live pipeline (see src/archive/README.md)
models/        # Trained model artifacts (.pkl)
frontend/      # React dashboard (see "Multi-Season Dashboard Support" below)
```

## Multi-Season Dashboard Support

The dashboard's season selector switches between real, independent datasets - it
doesn't reinterpret one dataset two ways:

- **2025**: a real, fully-completed season. Every section (games, fantasy, season
  projections, accuracy tracker, weekly summary, betting analysis, model
  transparency) has real, validated data.
- **2026**: the real season hadn't been played as of this writing (real 2026-09-09
  opener). Only what's genuinely computable pre-season is shown - the real
  schedule with real preseason model predictions (rolled-forward Elo, since no
  real Vegas lines exist for games this far out) in Section 1, and real preseason
  ensemble win projections in Section 3. Fantasy Rankings, Accuracy Tracker,
  Weekly Summary, and Betting Analysis are hidden for 2026 with a real "not
  available yet" message rather than shown empty or fabricated - none of them
  have real completed games to compute from yet.

### Adding real data for a new season once it's playable

1. Confirm real schedule data exists (`data/raw/schedules_{year}.csv`).
2. Preseason-only, before any games are played: run
   `python src/generate_dashboard_data_{year}.py` and
   `python src/generate_season_projections_dashboard_data_{year}.py`
   (see the 2026 versions for the real pattern - preseason Elo, no fabricated
   results).
3. Once real games start completing: extend `generate_dashboard_data.py`,
   `generate_fantasy_dashboard_data.py`, `generate_accuracy_tracker_dashboard_
   data.py`, `generate_weekly_summary_dashboard_data.py`, and `betting_
   backtest.py` to accept that season - each currently assumes 2025 is the only
   completed season on record.
4. Add the year to `frontend/src/constants/seasons.js`'s `AVAILABLE_SEASONS`
   and `SEASON_HAS_RESULTS`, and wire its real JSON exports into
   `frontend/src/context/SeasonContext.js`.

## Known Limitations

- **Compression**: EPA-pipeline team-strength/win projections are ~3.3x too
  narrow vs. real variance; proven this can't be fixed by simple rescaling.
  This is also *why* several real "confidence-based" features underperform
  their naive expectation — e.g. edge detection's confidence never exceeds
  ~45% in real 2025 data because Elo's own win probabilities are compressed
  toward 50%.
- **CI calibration is uneven, not uniformly good.** Percentile-across-
  correlated-candidates CIs were found badly miscalibrated (12.5% actual
  coverage vs. 90% target) and abandoned. Direct residual-std bands
  (game-spread work) calibrate well (~89-90%). Season-total tracking CIs
  self-correct over the season but start poorly calibrated early.
- **Vegas beats the model, full stop, at every checkpoint tested.** This
  isn't a preseason-only gap that closes as the season progresses - the
  real LOOCV search never found a point where blending in Elo helped.
- **No unit tests** - validation is via direct backtesting against real
  historical outcomes throughout, including honest reporting of null and
  negative results (see Key Findings), not via a unit test suite.
- See "Known Gaps" above for real, disclosed wiring gaps between validated
  findings and the production entrypoints.

## Next Steps

- [ ] Wire the validated volume-only RB/QB/TE fantasy formulas into `fantasy_validation.py`'s actual production path (currently only reachable via `integrated_predictions.py`)
- [ ] Persist `integrated_predictions.py`'s fantasy results to a combined CSV
- [ ] Wire `playoff_probability.py` into the single integrated entrypoint
- [ ] Add a real CLI to `weekly_tracking.py` for live weekly use
- [ ] Dashboard / API (not started)

## Technical Notes

- Language: Python 3.11+
- Core libraries: pandas, numpy, scikit-learn, scipy (see `requirements.txt`)
- Validation convention throughout: fit/derive on one real historical range,
  validate against a genuinely separate real holdout (commonly 2015-2023/24
  train vs. 2024/2025 holdout), with LOOCV wherever a weight or threshold is
  learned from the same data it's evaluated against - constants and
  coefficients are derived by regression/search on real data, never
  asserted, per this project's standing convention.

## Project History

- **Session 1 (2026-07-25)**: built the EPA model, identified compression
  and weakness issues.
- **Session 2 (2026-07-27)**: built Elo (beats EPA), validated vs. Vegas
  (Vegas wins), explored fantasy, built and validated Phases 1-4
  (weekly tracking, RB fantasy fix, playoff odds, injury/QB-Elo/matchup
  models, momentum/rest/QB-TE-fantasy refinements, edge detection, line
  movement - skipped, no data), integrated the validated findings, and
  audited/cleaned up twice.

See `PROGRESS.md` for the full, task-by-task session notes behind every
number in this document.
