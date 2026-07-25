"""Team-level defense EPA prediction (Phase 2.X-Team, Task 2).

Rather than decomposing team defense EPA down to individual CB/S/LB with
invented allocation weights (the original WAR spec's Task 2.X.3), this
predicts team-level defense EPA allowed directly - a real, well-defined
target computable straight from PBP for every team-season 2015-2025.

Honestly backtested across 6 independent holdout years (2019-2024), not just
one: team defensive EPA is famously non-sticky year-over-year in the NFL
(~0.28 correlation with its own prior-season value - much lower than
offense). "Assume this season looks like last season" scores an average R2
of -0.77 across those 6 years - actively worse than just predicting the
league average (-0.09). The model here (lagged EPA splits + pass-rush WAR +
sacks, standardized Ridge) scores -0.086 on average: a real, consistent
improvement over the naive-lag baseline, but landing close to simply
predicting the league average - not a strong model in absolute terms.

Personnel-continuity features (returning-snap share by position group -
front seven / LB / secondary) were also tested and didn't move the average
at all (also -0.086), so they were left out: they'd add three more
parameters to a 309-row dataset for no measured benefit, pure overfitting
risk. This was a deliberate, tested decision, not an oversight.

Bottom line carried into Phase 3: treat next-season team defensive EPA
projections as low-confidence relative to offensive projections, and don't
expect this component to be a strong predictor on its own - pass-rush WAR
(Task 1, on real individual-level data) is the more trustworthy defensive
signal available.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utilities import compute_history_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

PBP_COLS = ["season", "week", "defteam", "play_type", "epa", "season_type", "sack"]
TEAM_DEFENSE_FEATURES = [
    "def_epa_allowed_last_year", "pass_epa_allowed_last_year", "rush_epa_allowed_last_year",
    "team_pass_rush_war_last_year", "team_sacks_last_year",
]
TRAIN_START = 2016  # earliest season with a valid lag (needs a 2015 prior season)
HOLDOUT_SEASON = 2024
BACKTEST_YEARS = range(2019, 2025)  # honest multi-year backtest, not just the single 2024 holdout
RIDGE_ALPHA = 10  # fixed, not tuned per holdout - see module docstring


def compute_team_defense_epa(pbp_path=None):
    """Team defense EPA allowed per season - the real target - plus its
    pass/rush split and raw sack count, all from PBP scrimmage plays
    (pass/run only, regular season). Streams the 1.3GB PBP file in chunks,
    same memory-safety reason as player_impact.py's pass-rush WAR (this
    machine has hit MemoryError loading the full file before)."""
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    keep_chunks = []
    for chunk in pd.read_csv(path, usecols=PBP_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[chunk["play_type"].isin(["pass", "run"]) & (chunk["season_type"] == "REG")]
        if len(sub):
            keep_chunks.append(sub)
    reg = pd.concat(keep_chunks, ignore_index=True)
    reg = reg.dropna(subset=["epa", "defteam"])

    team_def = reg.groupby(["defteam", "season"])["epa"].mean().reset_index(name="def_epa_allowed")
    team_pass = reg[reg["play_type"] == "pass"].groupby(["defteam", "season"])["epa"].mean().reset_index(name="pass_epa_allowed")
    team_rush = reg[reg["play_type"] == "run"].groupby(["defteam", "season"])["epa"].mean().reset_index(name="rush_epa_allowed")
    team_sacks = reg.groupby(["defteam", "season"])["sack"].sum().reset_index(name="team_sacks")

    team = team_def.merge(team_pass, on=["defteam", "season"]).merge(team_rush, on=["defteam", "season"]).merge(
        team_sacks, on=["defteam", "season"]
    )
    team = team.rename(columns={"defteam": "team"})
    print(f"[team_defense_epa] {team.shape[0]:,} team-seasons, {int(team['season'].min())}-{int(team['season'].max())}")
    return team


def build_team_defense_features(team_epa, pass_rush_war_df):
    """Adds lagged (leak-free) versions of every raw column, plus team-level
    pass-rush WAR aggregated from Task 1's per-player output."""
    team_war = pass_rush_war_df.groupby(["team", "season"])["war"].sum().reset_index(name="team_pass_rush_war")
    df = team_epa.merge(team_war, on=["team", "season"], how="left")
    df["team_pass_rush_war"] = df["team_pass_rush_war"].fillna(0)

    for col in ["def_epa_allowed", "pass_epa_allowed", "rush_epa_allowed", "team_pass_rush_war", "team_sacks"]:
        feats = compute_history_features(df[["team", "season", col]], col, id_col="team")
        df = df.join(feats[[f"{col}_last_year"]])

    return df


def _r2(actual, pred):
    return 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)


def backtest_team_defense_model(df, years=BACKTEST_YEARS, target_col="def_epa_allowed", feature_cols=None):
    """Honest multi-year backtest: trains fresh on all prior seasons for each
    holdout year (not just a single 2024 split), comparing the model against
    two naive baselines. This is what caught that the original single-year
    (2024) result was misleadingly optimistic relative to the 6-year average,
    and confirmed personnel-continuity features didn't actually help once
    averaged across years rather than eyeballed on one.

    target_col/feature_cols: generalized (Phase 2 Task 5.1, extended in
    Task 5.4) so the same honest backtest can be reused for the
    pass/rush-allowed splits and for a parallel team OFFENSE model - same
    rigor, different target/features. Defaults preserve the original
    combined defense-model behavior exactly."""
    feature_cols = feature_cols or TEAM_DEFENSE_FEATURES
    sub = df.dropna(subset=feature_cols + [target_col])
    rows = []
    for holdout_year in years:
        train = sub[sub["season"] < holdout_year]
        hold = sub[sub["season"] == holdout_year]
        if len(hold) == 0 or len(train) == 0:
            continue
        actual = hold[target_col].to_numpy()

        naive_lag = hold[f"{target_col}_last_year"].to_numpy()
        r2_naive_lag = _r2(actual, naive_lag)

        league_mean = np.full_like(actual, train[target_col].mean())
        r2_league_mean = _r2(actual, league_mean)

        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(hold[feature_cols])
        r2_model = _r2(actual, pred)
        mae_model = np.mean(np.abs(pred - actual))

        rows.append({
            "holdout_year": holdout_year, "n_train": len(train), "n_holdout": len(hold),
            "r2_naive_lag": r2_naive_lag, "r2_league_mean": r2_league_mean,
            "r2_model": r2_model, "mae_model": mae_model,
        })

    results = pd.DataFrame(rows)
    print(f"\n===== TEAM {target_col.upper()} BACKTEST (independent holdout years) =====")
    print(results.to_string(index=False))
    print(f"\nAverage R2: naive_lag={results['r2_naive_lag'].mean():.3f} | "
          f"league_mean={results['r2_league_mean'].mean():.3f} | "
          f"model={results['r2_model'].mean():.3f}")
    print("===== END BACKTEST =====\n")
    return results


def validate_final_holdout(df, model, holdout_season=HOLDOUT_SEASON, target_col="def_epa_allowed", feature_cols=None):
    """Detailed look at the single most recent holdout (2024): best/worst
    predicted defenses, spot sanity checks."""
    feature_cols = feature_cols or TEAM_DEFENSE_FEATURES
    sub = df.dropna(subset=feature_cols + [target_col])
    hold = sub[sub["season"] == holdout_season].copy()
    hold["predicted"] = model.predict(hold[feature_cols])
    hold["error"] = hold["predicted"] - hold[target_col]

    print(f"\n===== {holdout_season} HOLDOUT DETAIL ({target_col}) =====")
    print("Actual best 5 (lowest):")
    print(hold.nsmallest(5, target_col)[["team", target_col, "predicted"]].to_string(index=False))
    print("\nActual worst 5:")
    print(hold.nlargest(5, target_col)[["team", target_col, "predicted"]].to_string(index=False))
    print(f"\nMAE: {hold['error'].abs().mean():.4f} | R2: {_r2(hold[target_col].to_numpy(), hold['predicted'].to_numpy()):.3f}")
    print("===== END HOLDOUT DETAIL =====\n")


def predict_next_season(df, model, ref_season=None, target_col="def_epa_allowed", out_col="predicted_def_epa_allowed",
                         feature_map=None):
    """Projects next season's team defense EPA allowed, using ref_season
    (auto-detected as the latest season present, defaults to 2025 here since
    this is PBP-native and not capped by PFR's 2018-2025 coverage) as the
    jump-off state.

    feature_map: {feature_col_name: current_season_source_col_name},
    generalized (Task 5.4) so this can build the projection frame for a
    non-defense feature set too. Defaults to the original defense mapping."""
    if ref_season is None:
        ref_season = int(df["season"].max())
    current = df[df["season"] == ref_season].copy()

    feature_map = feature_map or {
        "def_epa_allowed_last_year": "def_epa_allowed",
        "pass_epa_allowed_last_year": "pass_epa_allowed",
        "rush_epa_allowed_last_year": "rush_epa_allowed",
        "team_pass_rush_war_last_year": "team_pass_rush_war",
        "team_sacks_last_year": "team_sacks",
    }
    proj_features = pd.DataFrame({feat_col: current[src_col] for feat_col, src_col in feature_map.items()})
    predicted = model.predict(proj_features)

    out = pd.DataFrame({
        "team": current["team"],
        out_col: predicted,
    }).sort_values(out_col).reset_index(drop=True)  # ascending: best defense first
    out["projection_note"] = (
        f"projected from {ref_season} data; model's realistic skill is modest "
        f"(see backtest - roughly on par with predicting league-average defense)"
    )
    return out, ref_season


def train_defense_component_model(df, target_col, train_start=TRAIN_START, holdout_season=HOLDOUT_SEASON,
                                   feature_cols=None):
    """Fits + honestly backtests a Ridge model for an arbitrary team-season
    target column (def_epa_allowed, pass_epa_allowed, rush_epa_allowed, or -
    as of Task 5.4 - an offense-side target), reusing the exact same
    pipeline/backtest rigor as the primary combined defense model. Returns
    the fitted final model (trained on train_start..holdout_season-1) plus
    its backtest results."""
    feature_cols = feature_cols or TEAM_DEFENSE_FEATURES
    backtest_results = backtest_team_defense_model(df, target_col=target_col, feature_cols=feature_cols)

    train = df[(df["season"] >= train_start) & (df["season"] <= holdout_season - 1)].dropna(
        subset=feature_cols + [target_col])
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[feature_cols], train[target_col])
    print(f"[{target_col}] final model trained on {len(train)} team-seasons ({train_start}-{holdout_season - 1})")

    validate_final_holdout(df, model, holdout_season=holdout_season, target_col=target_col, feature_cols=feature_cols)
    return model, backtest_results


def run_team_defense_model():
    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    team_epa = compute_team_defense_epa()
    df = build_team_defense_features(team_epa, pass_rush_war_df)

    backtest_team_defense_model(df)

    train = df[(df["season"] >= TRAIN_START) & (df["season"] <= HOLDOUT_SEASON - 1)].dropna(subset=TEAM_DEFENSE_FEATURES)
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[TEAM_DEFENSE_FEATURES], train["def_epa_allowed"])
    print(f"[team_defense] final model trained on {len(train)} team-seasons ({TRAIN_START}-{HOLDOUT_SEASON - 1})")

    validate_final_holdout(df, model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "team_defense_epa.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {model_path}")

    epa_path = os.path.join(PROCESSED_DIR, "team_defense_epa_2015_2025.csv")
    df.to_csv(epa_path, index=False, encoding="utf-8")
    print(f"Saved {epa_path} ({df.shape[0]:,} rows)")

    predictions, ref_season = predict_next_season(df, model)
    target_season = ref_season + 1
    pred_path = os.path.join(PROCESSED_DIR, f"team_defense_epa_predictions_{target_season}.csv")
    predictions.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\n[team_defense] projecting season {target_season} from {ref_season} data")
    print(f"Saved {pred_path} ({len(predictions)} teams)")
    print("Best 10 projected defenses:")
    print(predictions.head(10).to_string(index=False))

    return model, df, predictions


if __name__ == "__main__":
    run_team_defense_model()
