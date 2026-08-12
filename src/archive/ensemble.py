"""Ensemble Approach: season-level wins + game-level spreads.

Corrects several issues found in the spec before building (see completion
report for full reasoning):

1. Two "season-level" candidates (Task 3's play-mix formula, Phase 2
   Candidate 2's R2-weighted aggregation) were listed with OFFENSE-ONLY
   correlations against real offensive EPA - neither was ever run through
   the EPA->wins conversion, so neither was actually a win prediction.
   Fixed here by pairing each with the existing defensive Ridge model to
   get net_strength, then running that through the ALREADY-FITTED
   epa_to_wins model (models/epa_to_wins.pkl) - genuine, comparable win
   predictions, not a re-fit.

2. "Vegas spreads: corr=+1.0 (by definition)" is the wrong quantity (Vegas
   correlating with itself). The real, relevant number - Vegas spread_line
   vs. real 2024 game outcomes - was computed directly: corr=+0.501,
   MAE=9.61 pts, genuinely better than the game engine's own holdout
   (+0.255/10.82 pts).

3. "Simple EPA difference: corr=+0.20 (estimated)" was a guess. Computed
   for real: same-season real EPA gives +0.603, but that's leaked (you
   can't know a season's real EPA before predicting that season's games).
   The genuine, leak-free version here uses PRIOR-season real EPA.

4. "Weekly-updated predictions: widening variance" undersold the real,
   already-measured result (Master Plan Phase 1 Task 1.2): weekly-updated
   MAE was WORSE than static preseason at every checkpoint tested. Also
   structurally excluded from the game-level ensemble here since it isn't
   a single static candidate - each game's prediction depends on which
   week you ask, unlike every other candidate.

5. Honesty note on sample size: unlike this project's other backtests
   (100s of team-seasons), several of these candidates only exist for a
   single season (2025 season-level, 2024 game-level) since they're new
   constructs built for this task. Stacking in particular is fit via
   leave-one-out cross-validation rather than a plain in-sample fit, to
   avoid a fit-and-evaluate-on-the-same-32-rows result that would trivially
   "win" by construction.
"""

import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# PART 1: Season-level wins ensemble
# ---------------------------------------------------------------------------

def _offense_formula_to_wins(offensive_strength_df, season=2025):
    """Pairs an offense-only formula's offensive_strength with the existing
    production defensive_strength_allowed (top-down Ridge model - unchanged
    across every offensive variant tried this project) and runs the
    resulting net_strength through the already-fitted epa_to_wins model.

    Bug found while testing: the win model's intercept was calibrated
    specifically to PRODUCTION's offensive_strength scale (play-mix QB+RB).
    An alternative formula with a different mean/scale (e.g. Candidate 2's
    R2-weighted blend, which weights WR/TE more heavily) shifts the whole
    output off-center when fed through that same fixed intercept - a first
    pass of this produced a 12.2-win average, nowhere near the ~8.5 every
    other candidate correctly centers on. Fixed by rescaling the alternative
    formula's offensive_strength to match production's mean/std before
    pairing - preserves the alternative formula's relative team-to-team
    pattern (the actual thing being tested) while correcting the scale
    mismatch, rather than silently shipping a biased result."""
    prod = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv"))
    prod_def = prod[["team", "defensive_strength_allowed"]]

    merged = offensive_strength_df.merge(prod_def, on="team", how="inner")
    if not np.isclose(merged["offensive_strength"].mean(), prod["offensive_strength"].mean(), atol=1e-9):
        merged["offensive_strength"] = (
            (merged["offensive_strength"] - merged["offensive_strength"].mean()) / merged["offensive_strength"].std()
            * prod["offensive_strength"].std() + prod["offensive_strength"].mean()
        )
    merged["net_strength"] = merged["offensive_strength"] - merged["defensive_strength_allowed"]

    with open(os.path.join(MODELS_DIR, "epa_to_wins.pkl"), "rb") as f:
        model = pickle.load(f)
    win_pct = np.clip(model["slope"] * merged["net_strength"] + model["intercept"], 0.0, 1.0)
    merged["projected_wins"] = win_pct * 17
    return merged.set_index("team")["projected_wins"]


def build_season_wins_candidates(season=2025):
    from vegas_comparison import compute_vegas_implied_wins
    from game_predictions import build_game_prediction_engine, infer_season_wins_from_game_predictions, \
        _load_schedule_for_season
    from team_aggregation import build_offensive_weight_training_table, load_real_2025_baseline_components
    from phase3_diagnostic import compute_real_2025_team_epa

    candidates = {}

    prod = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{season}.csv"))
    candidates["epa_wins_model"] = prod.set_index("team")["projected_wins"]

    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas_wins = compute_vegas_implied_wins(vegas, season=season)
    candidates["vegas_implied_wins"] = vegas_wins.set_index("team")["vegas_implied_wins"]

    task3_off = pd.read_csv(os.path.join(
        PROCESSED_DIR, f"team_strength_{season}_availability_adjusted_corrected.csv"))[["team", "offensive_strength"]]
    candidates["task3_formula_wins"] = _offense_formula_to_wins(task3_off, season)

    # Candidate 2 (R2-weighted aggregation) - recompute per-team offensive_strength
    # (never saved to a file, only returned as a summary dict in Phase 2)
    train_data = build_offensive_weight_training_table()
    holdout = load_real_2025_baseline_components()
    real_2025 = compute_real_2025_team_epa()[["team", "real_offensive_epa"]]
    holdout = holdout.merge(real_2025, on="team", how="inner")
    r2_by_pos = {col: train_data[col].corr(train_data["real_offensive_epa"]) ** 2
                 for col in ["qb_epa", "rb_epa", "wr_epa", "te_epa"]}
    total_r2 = sum(r2_by_pos.values())
    weights = {k: v / total_r2 for k, v in r2_by_pos.items()}
    holdout["offensive_strength"] = sum(weights[c] * holdout[c] for c in ["qb_epa", "rb_epa", "wr_epa", "te_epa"])
    candidates["candidate2_r2_weighted_wins"] = _offense_formula_to_wins(holdout[["team", "offensive_strength"]], season)

    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv"))
    schedule = _load_schedule_for_season(season)
    game_pred, _ = build_game_prediction_engine(season=season, team_strength=team_strength, schedule=schedule, save=False)
    game_wins = infer_season_wins_from_game_predictions(game_pred)
    candidates["game_level_wins"] = game_wins.set_index("team")["expected_wins"]

    candidates["naive_baseline"] = pd.Series(8.5, index=candidates["epa_wins_model"].index)

    df = pd.DataFrame(candidates)
    if df.isnull().any().any():
        # candidate2_r2_weighted_wins drops DEN (DEN's TE never cleared the
        # min_opportunities threshold - the same known gap from Task 4's
        # original build). Same fillna-with-column-mean convention used
        # throughout this project for missing-team situations, rather than
        # dropping the team from the whole ensemble over one candidate's gap.
        missing = df[df.isnull().any(axis=1)].index.tolist()
        print(f"[build_season_wins_candidates] filling missing candidate values with column mean for: {missing}")
        df = df.fillna(df.mean())
    print(f"\nSeason-level candidates built for {season} ({len(df)} teams, {len(df.columns)} candidates):")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    return df


def _individual_candidate_accuracy(candidates_df, actual):
    """Each candidate's own MAE/corr against real outcomes - needed both to
    report standalone accuracy and to build accuracy-weighted combining rules."""
    merged = candidates_df.merge(actual, left_index=True, right_on="team")
    rows = []
    for col in candidates_df.columns:
        mae = np.mean(np.abs(merged[col] - merged["actual"]))
        corr = merged[col].corr(merged["actual"])
        rows.append({"candidate": col, "mae": mae, "corr": corr})
    return pd.DataFrame(rows).set_index("candidate")


def combine_simple_average(candidates_df, accuracy=None):
    return candidates_df.mean(axis=1)


def combine_weighted_average(candidates_df, accuracy):
    """Weight inversely proportional to each candidate's own real-outcome
    MAE (lower error -> higher weight), normalized to sum to 1."""
    inv_mae = 1.0 / accuracy["mae"]
    weights = inv_mae / inv_mae.sum()
    return sum(candidates_df[c] * weights[c] for c in candidates_df.columns)


def combine_median(candidates_df, accuracy=None):
    return candidates_df.median(axis=1)


def combine_trimmed_mean(candidates_df, accuracy=None):
    def _trim(row):
        vals = row.sort_values()
        return vals.iloc[1:-1].mean() if len(vals) > 2 else vals.mean()
    return candidates_df.apply(_trim, axis=1)


def combine_vegas_anchored(candidates_df, accuracy=None, vegas_weight=0.5):
    others = candidates_df.drop(columns=["vegas_implied_wins"]).mean(axis=1)
    return vegas_weight * candidates_df["vegas_implied_wins"] + (1 - vegas_weight) * others


def combine_stacking_loocv(candidates_df, actual):
    """Leave-one-out linear stacking: with only 32 real team-rows and up to
    6 predictors, an in-sample fit would trivially "win" by construction
    (fitting an OLS on the exact data it's then scored against). LOOCV
    fits on 31 teams and predicts the held-out 32nd, repeated for every
    team - a genuinely out-of-sample stacking prediction, though still a
    small-sample result worth treating cautiously (see completion report)."""
    from sklearn.linear_model import LinearRegression

    merged = candidates_df.merge(actual, left_index=True, right_on="team").set_index("team")
    X_cols = list(candidates_df.columns)
    preds = {}
    for team in merged.index:
        train = merged.drop(index=team)
        model = LinearRegression().fit(train[X_cols], train["actual"])
        preds[team] = model.predict(merged.loc[[team], X_cols])[0]
    return pd.Series(preds)


COMBINING_RULES = {
    "simple_average": combine_simple_average,
    "weighted_average": combine_weighted_average,
    "median": combine_median,
    "trimmed_mean": combine_trimmed_mean,
    "vegas_anchored": combine_vegas_anchored,
}


def compute_calibration(candidates_df, actual, confidence=0.90):
    """Percentile-method 90% CI (5th/95th percentile across ensemble member
    predictions per team) and coverage check - does the real outcome
    actually fall inside the interval ~90% of the time?"""
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    lo = candidates_df.quantile(lo_q, axis=1)
    hi = candidates_df.quantile(hi_q, axis=1)
    merged = pd.DataFrame({"lo": lo, "hi": hi}).merge(actual.set_index("team"), left_index=True, right_index=True)
    covered = (merged["actual"] >= merged["lo"]) & (merged["actual"] <= merged["hi"])
    coverage = covered.mean()
    avg_width = (merged["hi"] - merged["lo"]).mean()
    return {"coverage": coverage, "target_coverage": confidence, "avg_width": avg_width}


def backtest_season_ensemble(season=2025):
    candidates_df = build_season_wins_candidates(season)
    actual = pd.read_csv(os.path.join(BACKTEST_DIR, f"actual_wins_{season}.csv"))[["team", "actual_wins"]].rename(
        columns={"actual_wins": "actual"})

    print(f"\n{'=' * 70}\nINDIVIDUAL CANDIDATE ACCURACY (real {season})\n{'=' * 70}")
    accuracy = _individual_candidate_accuracy(candidates_df, actual)
    print(accuracy.round(3))

    print(f"\n{'=' * 70}\nCOMBINING RULES (real {season})\n{'=' * 70}")
    results = []
    combined_series = {}
    for name, fn in COMBINING_RULES.items():
        combined = fn(candidates_df, accuracy)
        combined_series[name] = combined
        merged = pd.DataFrame({"pred": combined}).merge(actual.set_index("team"), left_index=True, right_index=True)
        mae = np.mean(np.abs(merged["pred"] - merged["actual"]))
        corr = merged["pred"].corr(merged["actual"])
        print(f"{name:<20} MAE={mae:.3f}  corr={corr:+.3f}")
        results.append({"rule": name, "mae": mae, "corr": corr})

    stacked = combine_stacking_loocv(candidates_df, actual)
    combined_series["stacking_loocv"] = stacked
    merged = pd.DataFrame({"pred": stacked}).merge(actual.set_index("team"), left_index=True, right_index=True)
    mae_s = np.mean(np.abs(merged["pred"] - merged["actual"]))
    corr_s = merged["pred"].corr(merged["actual"])
    print(f"{'stacking_loocv':<20} MAE={mae_s:.3f}  corr={corr_s:+.3f}")
    results.append({"rule": "stacking_loocv", "mae": mae_s, "corr": corr_s})

    results_df = pd.DataFrame(results).sort_values("mae")
    best_rule = results_df.iloc[0]["rule"]
    print(f"\nBest by MAE: {best_rule} (MAE={results_df.iloc[0]['mae']:.3f})")
    print(f"For reference: best SINGLE candidate was "
          f"{accuracy['mae'].idxmin()} (MAE={accuracy['mae'].min():.3f})")

    calibration = compute_calibration(candidates_df, actual)
    print(f"\n90% CI coverage (percentile method, across all {len(candidates_df.columns)} raw candidates): "
          f"{calibration['coverage']:.1%} (target 90%) | avg width={calibration['avg_width']:.2f} wins")
    print("=" * 70 + "\n")

    out_path = os.path.join(PROCESSED_DIR, f"ensemble_season_wins_backtest_{season}.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

    return candidates_df, accuracy, results_df, combined_series, calibration
