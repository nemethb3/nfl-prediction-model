"""Phase 4, Task 4.1: Convert Team Strength to Season Win Projections.

Split out of win_projection.py (Master Plan Phase 4 Task 4.2 - that file had
grown to cover 5 unrelated jobs; see AUDIT_2026-07-25.md Technical Debt #5).
No behavior change from the original module, only reorganization.

Corrects two problems in the original task spec (see the Task 4.1 completion
report for the full reasoning, including the numbers behind each):

1. The spec's win-counting logic assumed schedules_2015_2025.csv's `result`
   column is 1/-1 (home win / away win). It isn't - `result` is the actual
   home-minus-away point differential (e.g. 4.0, -3.0, 25.0). Checked before
   writing any of this: a `result == 1` filter would only catch one-point
   games, silently dropping almost the entire season. Real win logic here
   uses `result > 0` (home win) / `result < 0` (away win).

2. The spec's EPA->wins conversion (season_epa_total = epa_diff * 1200,
   wins_vs_average = season_epa_total * 0.127) was checked against this
   project's own real historical data before use: it implies a slope of
   win_pct ~= 8.97 * epa_diff, but the REAL empirical relationship (fit on
   all 320 real 2015-2024 team-seasons, using this project's own already-
   computed real offense/defense EPA and real win records) is
   win_pct ~= 1.41 * epa_diff + 0.50, with a strong real correlation
   (+0.87). The spec's constant would have been about 6.4x too steep -
   nearly every team's win range in this project's actual EPA scale would
   have clipped to 0 or 17 wins instead of spreading realistically across
   the standings. This module fits the real relationship instead of
   asserting one, same discipline used for every other constant in this
   project (OL/SOS/synergy weights, sacks->war conversion, etc.), with an
   honest multi-year backtest and a confidence band derived from the
   model's own real residual spread rather than an asserted +-1.7 games.
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

TRAIN_SEASONS = range(2015, 2025)  # all real historical seasons - 2025 held out entirely,
                                    # never used to fit the conversion (see module docstring)
BACKTEST_YEARS = range(2019, 2025)  # honest multi-year backtest, same convention as team_strength.py
TARGET_SEASON = 2025


def compute_real_win_pct(schedules):
    """Real win percentage per team-season, from actual game results.
    result > 0 = home win, result < 0 = away win (result == 0 ties don't
    occur in the NFL post-2015 without OT resolution, but ties do exist -
    a tie counts as half a win for both teams, the standard convention)."""
    reg = schedules[schedules["game_type"] == "REG"].dropna(subset=["home_score", "away_score"])

    home = reg[["season", "home_team", "result"]].rename(columns={"home_team": "team"})
    home["win_credit"] = np.select([home["result"] > 0, home["result"] < 0], [1.0, 0.0], default=0.5)
    away = reg[["season", "away_team", "result"]].rename(columns={"away_team": "team"})
    away["win_credit"] = np.select([away["result"] < 0, away["result"] > 0], [1.0, 0.0], default=0.5)

    games = pd.concat([home[["season", "team", "win_credit"]], away[["season", "team", "win_credit"]]])
    wl = games.groupby(["team", "season"]).agg(wins=("win_credit", "sum"), games=("win_credit", "count")).reset_index()
    wl["win_pct"] = wl["wins"] / wl["games"]
    return wl


def build_historical_epa_wins_dataset():
    """Real (epa_diff, win_pct) pairs for every team-season - the training
    data for the conversion model. epa_diff = real offense EPA/play minus
    real defense EPA/play allowed, both already-computed real quantities
    (coach_quality.compute_team_offense_epa / team_strength.compute_team_defense_epa)."""
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    off = compute_team_offense_epa()
    defn = compute_team_defense_epa()
    epa = off.merge(defn, on=["team", "season"])
    epa["epa_diff"] = epa["off_epa"] - epa["def_epa_allowed"]

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    wl = compute_real_win_pct(schedules)

    return epa.merge(wl, on=["team", "season"])[["team", "season", "epa_diff", "win_pct", "wins", "games"]]


def _r2(actual, pred):
    return 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)


def backtest_epa_to_wins_model(df, years=BACKTEST_YEARS):
    """Honest multi-year backtest, same convention used throughout this
    project: train fresh on all real seasons before the holdout year,
    compare against the naive 'everyone is a .500 team' baseline."""
    rows = []
    for holdout_year in years:
        train = df[df["season"] < holdout_year]
        hold = df[df["season"] == holdout_year]
        if len(hold) == 0 or len(train) == 0:
            continue
        actual = hold["win_pct"].to_numpy()

        naive = np.full(len(hold), 0.5)
        r2_naive = _r2(actual, naive)

        slope, intercept = np.polyfit(train["epa_diff"], train["win_pct"], 1)
        pred = slope * hold["epa_diff"].to_numpy() + intercept
        r2_model = _r2(actual, pred)
        mae_model = np.mean(np.abs(pred - actual)) * hold["games"].mean()  # in wins, not win_pct

        rows.append({"holdout_year": holdout_year, "n_train": len(train), "n_holdout": len(hold),
                      "slope": slope, "intercept": intercept,
                      "r2_naive": r2_naive, "r2_model": r2_model, "mae_wins": mae_model})

    results = pd.DataFrame(rows)
    print("\n===== EPA -> WIN_PCT BACKTEST (independent holdout years) =====")
    print(results.to_string(index=False))
    print(f"\nAverage R2: naive(.500)={results['r2_naive'].mean():.3f} | model={results['r2_model'].mean():.3f}")
    print(f"Average MAE: {results['mae_wins'].mean():.2f} wins")
    print("===== END BACKTEST =====\n")
    return results


def fit_final_model(df, train_seasons=TRAIN_SEASONS):
    """Final conversion model, fit on every real historical season
    (2015-2024) - 2025's real outcomes are never used here, only its
    PROJECTED epa_diff gets fed through this already-fitted model later."""
    train = df[df["season"].isin(train_seasons)]
    slope, intercept = np.polyfit(train["epa_diff"], train["win_pct"], 1)
    pred = slope * train["epa_diff"].to_numpy() + intercept
    resid = train["win_pct"].to_numpy() - pred
    resid_std_wins = float(resid.std() * 17)  # residual spread in wins (2025 is a 17-game season)
    print(f"[epa_to_wins] final model: win_pct = {slope:.4f} * epa_diff + {intercept:.4f} "
          f"(trained on {len(train)} team-seasons, {min(train_seasons)}-{max(train_seasons)})")
    print(f"[epa_to_wins] residual std (in-sample): {resid_std_wins:.2f} wins - used for the confidence band")
    return {"slope": slope, "intercept": intercept, "resid_std_wins": resid_std_wins}


def project_season_wins(team_strength, model, target_season=TARGET_SEASON, games=17):
    """Applies the fitted conversion to team_strength_2025.csv's net_strength
    (already exactly offensive_strength - defensive_strength_allowed)."""
    df = team_strength.copy()
    df["epa_diff"] = df["net_strength"]
    df["win_pct"] = np.clip(model["slope"] * df["epa_diff"] + model["intercept"], 0.0, 1.0)
    df["projected_wins"] = df["win_pct"] * games

    band = model["resid_std_wins"]
    df["projected_wins_low"] = np.clip(df["projected_wins"] - band, 0, games)
    df["projected_wins_high"] = np.clip(df["projected_wins"] + band, 0, games)

    return df.sort_values("projected_wins", ascending=False).reset_index(drop=True)


def validate_win_projections(projections, games=17):
    print("\n===== WIN PROJECTION SANITY CHECKS =====")
    top, bottom = projections.iloc[0], projections.iloc[-1]
    spread = top["projected_wins"] - bottom["projected_wins"]
    mean_wins = projections["projected_wins"].mean()
    std_wins = projections["projected_wins"].std()

    print(f"Top team: {top['team']} - {top['projected_wins']:.1f} wins "
          f"[{top['projected_wins_low']:.1f}, {top['projected_wins_high']:.1f}]")
    print(f"Bottom team: {bottom['team']} - {bottom['projected_wins']:.1f} wins "
          f"[{bottom['projected_wins_low']:.1f}, {bottom['projected_wins_high']:.1f}]")
    print(f"Spread: {spread:.1f} wins | Mean: {mean_wins:.1f} wins | Std dev: {std_wins:.1f} wins")
    print(f"(real NFL 2015-2024 seasons: mean win spread top-to-bottom is typically ~10-13 wins per season, "
          f"real mean is exactly {games / 2:.1f} by construction of a zero-sum league)")
    print("===== END SANITY CHECKS =====\n")
    return {"spread": spread, "mean": mean_wins, "std": std_wins}


def run_win_projection():
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    historical = build_historical_epa_wins_dataset()
    historical_pre2025 = historical[historical["season"] < 2025]

    backtest_epa_to_wins_model(historical_pre2025)
    model = fit_final_model(historical_pre2025)

    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{TARGET_SEASON}.csv"))
    projections = project_season_wins(team_strength, model, target_season=TARGET_SEASON)
    validate_win_projections(projections)

    print(f"\n{TARGET_SEASON} Season Win Projections (all {len(projections)} teams):")
    print(projections[["team", "epa_diff", "win_pct", "projected_wins",
                        "projected_wins_low", "projected_wins_high"]].to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, f"win_projections_{TARGET_SEASON}.csv")
    projections.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")

    model_path = os.path.join(MODELS_DIR, "epa_to_wins.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {model_path}")

    return projections, model


if __name__ == "__main__":
    run_win_projection()
