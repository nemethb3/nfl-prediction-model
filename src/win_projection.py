"""Phase 4, Task 4.1: Convert Team Strength to Season Win Projections.

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



# ---------------------------------------------------------------------------
# Task 4.2: compare projections vs. actual 2025 results
# ---------------------------------------------------------------------------

def load_2025_actual_results():
    """Real 2025 win totals, reusing compute_real_win_pct (the fixed
    win-counting logic - result is a point differential, not +-1) rather
    than the spec's original buggy version."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    wl = compute_real_win_pct(schedules)
    actual = wl[wl["season"] == TARGET_SEASON][["team", "wins", "games", "win_pct"]].rename(
        columns={"wins": "actual_wins", "win_pct": "actual_win_pct"})

    print(f"\n{TARGET_SEASON} Actual Results:")
    print(actual.sort_values("actual_wins", ascending=False).to_string(index=False))

    os.makedirs(BACKTEST_DIR, exist_ok=True)
    out_path = os.path.join(BACKTEST_DIR, f"actual_wins_{TARGET_SEASON}.csv")
    actual.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")
    return actual


def compare_projections_vs_actual(projections, actual):
    comparison = projections.merge(actual, on="team")
    comparison["residual"] = comparison["actual_wins"] - comparison["projected_wins"]
    comparison["abs_error"] = comparison["residual"].abs()

    mae = comparison["abs_error"].mean()
    rmse = np.sqrt((comparison["residual"] ** 2).mean())
    correlation = comparison["projected_wins"].corr(comparison["actual_wins"])
    r2 = correlation ** 2
    bias = comparison["residual"].mean()

    print(f"\n===== {TARGET_SEASON} BACKTEST RESULTS =====")
    print(f"MAE: {mae:.2f} wins")
    print(f"RMSE: {rmse:.2f} wins")
    print(f"Correlation: {correlation:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Bias (mean residual): {bias:+.2f} wins ({'model overestimated on average' if bias < 0 else 'model underestimated on average' if bias > 0 else 'no systematic bias'})")

    print(f"\nBiggest overestimates (model too high - actual < projected):")
    print(comparison.nsmallest(5, "residual")[["team", "projected_wins", "actual_wins", "residual"]].to_string(index=False))
    print(f"\nBiggest underestimates (model too low - actual > projected):")
    print(comparison.nlargest(5, "residual")[["team", "projected_wins", "actual_wins", "residual"]].to_string(index=False))
    print("===== END BACKTEST RESULTS =====\n")

    out_path = os.path.join(BACKTEST_DIR, f"win_projections_vs_actual_{TARGET_SEASON}.csv")
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

    return comparison, {"mae": mae, "rmse": rmse, "correlation": correlation, "r2": r2, "bias": bias}


def analyze_compression_effect(comparison):
    """Directly follows up on Task 4.1's flagged finding: Phase 3's
    projected epa_diff (and therefore win spread) was ~3.2x more compressed
    than real historical team-season variance. Checks whether that shows up
    here as a systematic pattern (best teams under-projected, worst teams
    over-projected) rather than random noise."""
    real_std = comparison["actual_wins"].std()
    proj_std = comparison["projected_wins"].std()
    print("\n===== COMPRESSION EFFECT CHECK (following up on Task 4.1's flagged finding) =====")
    print(f"Real {TARGET_SEASON} win std dev: {real_std:.2f} | Projected win std dev: {proj_std:.2f} | "
          f"ratio: {real_std / proj_std:.2f}x (Task 4.1 predicted ~3.2x based on epa_diff variance)")

    proj_deviation = comparison["projected_wins"] - comparison["projected_wins"].mean()
    corr_check = comparison["residual"].corr(proj_deviation)
    print(f"corr(residual, projected deviation from mean) = {corr_check:+.3f}")
    print("(positive = confirms the compression pattern: teams projected above-average tend to have ADDITIONAL "
          "positive residual too [actual even better than projected], and teams projected below-average tend to "
          "underperform further - i.e. the model doesn't spread teams out enough. Near-zero or negative would mean "
          "the compression concern didn't actually show up as a systematic error this season.)")
    print("===== END COMPRESSION CHECK =====\n")
    return {"real_std": real_std, "proj_std": proj_std, "compression_ratio": real_std / proj_std, "corr_check": corr_check}


def run_backtest_comparison():
    projections = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{TARGET_SEASON}.csv"))
    actual = load_2025_actual_results()
    comparison, metrics = compare_projections_vs_actual(projections, actual)
    compression = analyze_compression_effect(comparison)
    return comparison, metrics, compression


# ---------------------------------------------------------------------------
# Phase 3 Rebuild Task 5: Validate against Vegas.
#
# Corrects the spec's implied-win-probability formula. It used
# `total_line` (the spec's "over_under") - a single GAME's combined-score
# betting line (e.g. 47.5 points) - fed through
# `0.5 + (over_under - 8.5) / 17`, treating a per-game point total as if it
# were a preseason season win total. Those are unrelated Vegas markets and
# the arithmetic doesn't correspond to any real quantity (confirmed by
# inspecting the actual vegas_with_results_2015_2025.csv columns before
# building - there's no season win-total line in this dataset at all).
#
# vegas_with_results_2015_2025.csv DOES have real per-game moneylines
# (home_moneyline/away_moneyline, 0 missing for all 272 2025 REG games),
# which convert directly to real market-implied win probability per game.
# Devigging (normalizing the two sides to sum to 1, removing the
# bookmaker's built-in margin) and summing each team's probability across
# their 17 real games gives a genuine Vegas-implied season win total - the
# actual, correct benchmark this task needs, built from real market data
# instead of an invented formula.
# ---------------------------------------------------------------------------

def moneyline_to_implied_prob(moneyline):
    ml = moneyline.to_numpy()
    with np.errstate(divide="ignore"):  # np.where evaluates both branches over the whole array;
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))  # a +100 ml hits the unused branch's /0


def compute_vegas_implied_wins(vegas, season=TARGET_SEASON):
    games = vegas[(vegas["season"] == season) & (vegas["game_type"] == "REG")].copy()
    games["home_implied_raw"] = moneyline_to_implied_prob(games["home_moneyline"])
    games["away_implied_raw"] = moneyline_to_implied_prob(games["away_moneyline"])
    vig_total = games["home_implied_raw"] + games["away_implied_raw"]
    games["home_win_prob"] = games["home_implied_raw"] / vig_total
    games["away_win_prob"] = games["away_implied_raw"] / vig_total

    home = games[["home_team", "home_win_prob"]].rename(columns={"home_team": "team", "home_win_prob": "win_prob"})
    away = games[["away_team", "away_win_prob"]].rename(columns={"away_team": "team", "away_win_prob": "win_prob"})
    long = pd.concat([home, away], ignore_index=True)

    vegas_wins = long.groupby("team")["win_prob"].sum().rename("vegas_implied_wins").reset_index()
    n_games = long.groupby("team").size().rename("vegas_n_games").reset_index()
    return vegas_wins.merge(n_games, on="team")


def validate_model_against_vegas():
    projections = pd.read_csv(os.path.join(PROCESSED_DIR, f"win_projections_{TARGET_SEASON}.csv"))
    vegas = pd.read_csv(os.path.join(BACKTEST_DIR, "vegas_with_results_2015_2025.csv"))
    vegas_wins = compute_vegas_implied_wins(vegas)
    actual = load_2025_actual_results()

    comparison = projections.merge(vegas_wins, on="team", how="inner").merge(
        actual[["team", "actual_wins"]], on="team", how="inner")

    print("=" * 70 + "\nMODEL vs. VEGAS vs. REAL 2025 RESULTS\n" + "=" * 70)

    corr_model_vegas = comparison["projected_wins"].corr(comparison["vegas_implied_wins"])
    corr_model_actual = comparison["projected_wins"].corr(comparison["actual_wins"])
    corr_vegas_actual = comparison["vegas_implied_wins"].corr(comparison["actual_wins"])
    mae_model_actual = np.mean(np.abs(comparison["projected_wins"] - comparison["actual_wins"]))
    mae_vegas_actual = np.mean(np.abs(comparison["vegas_implied_wins"] - comparison["actual_wins"]))

    print(f"\nOur model vs. Vegas (agreement): corr = {corr_model_vegas:+.3f}")
    print(f"\nAccuracy vs. REAL 2025 outcomes (the actual test - who's more right, not who agrees more):")
    print(f"  Our model  : corr = {corr_model_actual:+.3f} | MAE = {mae_model_actual:.2f} wins")
    print(f"  Vegas      : corr = {corr_vegas_actual:+.3f} | MAE = {mae_vegas_actual:.2f} wins")
    print(f"  -> {'Vegas is more accurate' if mae_vegas_actual < mae_model_actual else 'Our model is more accurate'} "
          f"this season (expected - Vegas prices in real-time injury/roster news our preseason-only model can't see)")

    comparison["model_vs_vegas_diff"] = comparison["projected_wins"] - comparison["vegas_implied_wins"]
    print("\nBiggest disagreements (our model vs. Vegas, signed - where we diverge most from the market):")
    disagree = comparison.reindex(comparison["model_vs_vegas_diff"].abs().sort_values(ascending=False).index).head(5)
    for _, row in disagree.iterrows():
        direction = "HIGHER" if row["model_vs_vegas_diff"] > 0 else "LOWER"
        print(f"  {row['team']}: model={row['projected_wins']:.1f} wins, Vegas={row['vegas_implied_wins']:.1f} wins "
              f"({direction} by {abs(row['model_vs_vegas_diff']):.1f}) | actual={row['actual_wins']:.0f} wins")

    print(f"\nCalibration:")
    print(f"  Our model mean={comparison['projected_wins'].mean():.2f} std={comparison['projected_wins'].std():.2f}")
    print(f"  Vegas      mean={comparison['vegas_implied_wins'].mean():.2f} std={comparison['vegas_implied_wins'].std():.2f}")
    print(f"  Real 2025  mean={comparison['actual_wins'].mean():.2f} std={comparison['actual_wins'].std():.2f}")
    print(f"  (Task 4.1 already found our projections are ~3.2x too compressed vs. real outcomes - "
          f"checking here whether Vegas is compressed too, or just us)")

    out_path = os.path.join(os.path.join(PROJECT_ROOT, "data", "diagnostic"), "model_vs_vegas_2025.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    print("=" * 70)

    return comparison, {"corr_model_vegas": corr_model_vegas, "corr_model_actual": corr_model_actual,
                         "corr_vegas_actual": corr_vegas_actual, "mae_model_actual": mae_model_actual,
                         "mae_vegas_actual": mae_vegas_actual}


if __name__ == "__main__":
    run_win_projection()
    run_backtest_comparison()


# ---------------------------------------------------------------------------
# Phase 3 Redesign Subtask 1: Game Prediction Engine.
#
# Corrects the spec's four asserted constants (EPA_TO_POINTS=3.5,
# HOME_FIELD_ADVANTAGE=2.5, league_avg_points=21.0, std_dev=5.0 for win
# probability) - it only checked ONE of them (EPA_TO_POINTS) against real
# data, and only as an afterthought that was never fed back into the actual
# predictions. Instead, one real regression on real historical per-game
# results (point_diff ~ epa_diff, pooled across many REG seasons) gives all
# four at once: the fitted slope IS the real EPA->points conversion, the
# intercept IS the real home-field advantage in points (home teams' real
# average edge, not asserted), the residual std IS the real spread for win
# probability, and league-average total points is measured directly from
# real games instead of assumed. Holdout-validated the same way every other
# model in this project is (train on 2016-2023, honest check on 2024).
# ---------------------------------------------------------------------------

def fit_game_level_epa_to_points(team_off, team_def, game_results,
                                  train_seasons=range(2016, 2024), holdout_season=2024):
    reg = game_results[game_results["game_type"] == "REG"].copy()

    off = team_off[["team", "season", "off_epa"]]
    defn = team_def[["team", "season", "def_epa_allowed"]]

    reg = reg.merge(off.rename(columns={"team": "home_team", "off_epa": "home_off_epa"}), on=["home_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "home_team", "def_epa_allowed": "home_def_epa"}), on=["home_team", "season"])
    reg = reg.merge(off.rename(columns={"team": "away_team", "off_epa": "away_off_epa"}), on=["away_team", "season"])
    reg = reg.merge(defn.rename(columns={"team": "away_team", "def_epa_allowed": "away_def_epa"}), on=["away_team", "season"])

    reg["epa_diff"] = (reg["home_off_epa"] - reg["away_def_epa"]) - (reg["away_off_epa"] - reg["home_def_epa"])
    reg["point_diff"] = reg["home_score"] - reg["away_score"]

    train = reg[reg["season"].isin(train_seasons)]
    slope, intercept = np.polyfit(train["epa_diff"], train["point_diff"], 1)
    train_pred = slope * train["epa_diff"] + intercept
    resid_std = float((train["point_diff"] - train_pred).std())
    train_corr = np.corrcoef(train["epa_diff"], train["point_diff"])[0, 1]

    hold = reg[reg["season"] == holdout_season]
    hold_pred = slope * hold["epa_diff"] + intercept
    hold_mae = float((hold["point_diff"] - hold_pred).abs().mean())
    hold_corr = np.corrcoef(hold["epa_diff"], hold["point_diff"])[0, 1]

    league_avg_total_points = float(train["total"].mean())

    print(f"[game-level EPA->points] train (n={len(train)}, seasons {min(train_seasons)}-{max(train_seasons)}): "
          f"point_diff = {slope:.3f} * epa_diff + {intercept:+.3f} (home-field, points) | "
          f"corr={train_corr:+.3f} | residual std={resid_std:.2f} pts")
    print(f"[game-level EPA->points] {holdout_season} holdout (n={len(hold)}): MAE={hold_mae:.2f} pts | corr={hold_corr:+.3f}")
    print(f"[game-level EPA->points] real league-average total points/game (train seasons): {league_avg_total_points:.2f}")

    return slope, intercept, resid_std, league_avg_total_points


def build_game_prediction_engine():
    from scipy.stats import norm
    from coach_quality import compute_team_offense_epa
    from team_strength import compute_team_defense_epa

    team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, "team_strength_2026.csv"))
    schedule_2026 = pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    schedule_2026 = schedule_2026[schedule_2026["game_type"] == "REG"].copy()
    print(f"Loaded {len(team_strength)} teams, {len(schedule_2026)} 2026 REG games")

    team_off = compute_team_offense_epa()
    team_def = compute_team_defense_epa()
    game_results = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    slope, intercept, resid_std, league_avg_total = fit_game_level_epa_to_points(team_off, team_def, game_results)

    strength_lookup = team_strength.set_index("team")[["offensive_strength", "defensive_strength_allowed"]]

    rows = []
    missing = set()
    for _, game in schedule_2026.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home not in strength_lookup.index or away not in strength_lookup.index:
            missing.update({home, away} - set(strength_lookup.index))
            continue
        home_off, home_def = strength_lookup.loc[home]
        away_off, away_def = strength_lookup.loc[away]

        epa_diff = (home_off - away_def) - (away_off - home_def)
        point_diff = slope * epa_diff + intercept  # intercept already IS real home-field advantage

        home_pts = league_avg_total / 2 + point_diff / 2
        away_pts = league_avg_total / 2 - point_diff / 2

        z = point_diff / resid_std
        home_win_prob = float(norm.cdf(z))

        rows.append({
            "week": game["week"], "home_team": home, "away_team": away,
            "home_expected_points": round(home_pts, 1), "away_expected_points": round(away_pts, 1),
            "expected_spread": round(point_diff, 2), "expected_total": round(home_pts + away_pts, 1),
            "home_win_probability": round(home_win_prob, 4), "away_win_probability": round(1 - home_win_prob, 4),
        })
    if missing:
        print(f"WARNING: no team_strength_2026 row for: {sorted(missing)} - their games skipped")

    predictions_df = pd.DataFrame(rows)
    print(f"\nGenerated predictions for {len(predictions_df)} games")
    print(f"Avg spread: {predictions_df['expected_spread'].mean():+.2f} | "
          f"Avg total: {predictions_df['expected_total'].mean():.1f} | "
          f"Spread std: {predictions_df['expected_spread'].std():.2f}")

    out_path = os.path.join(PROCESSED_DIR, "game_predictions_2026_preseason.csv")
    predictions_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

    return predictions_df, {"slope": slope, "intercept": intercept, "resid_std": resid_std,
                             "league_avg_total": league_avg_total}


# ---------------------------------------------------------------------------
# Phase 3 Redesign Subtask 3: Vegas Comparison & Edge Detection.
#
# load_vegas_lines_all_2026() doesn't exist anywhere - real 2026 lines just
# live directly on schedules_2026.csv (spread_line/total_line/moneylines),
# same as every other season. Checked before building: books have only
# posted real lines through about week 3-4 so far (53/272 games) - not a
# bug, that's genuinely how far out sportsbooks publish lines in real life
# as of today (2026-07-25, season starts September). So this compares our
# predictions to REAL, currently-live market lines for the games that have
# them, rather than needing a historical simulation like Subtask 2 did.
#
# The spec's implied_win_prob formula (0.5 + (over_under-8.5)/17) is the
# exact same bug already found and fixed in the earlier "Validate Against
# Vegas" task - confuses a game point total with a season win total.
# Reused moneyline_to_implied_prob() (already built, already devigged)
# instead of repeating that mistake.
#
# spread_line sign convention verified against real historical blowouts
# before use (not assumed): positive spread_line = home team favored,
# matching build_game_prediction_engine's own expected_spread convention -
# no sign flip needed.
# ---------------------------------------------------------------------------

def vegas_comparison_framework():
    our_predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions_2026_preseason.csv"))
    schedule_2026 = pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    schedule_2026 = schedule_2026[schedule_2026["game_type"] == "REG"]

    vegas_lines = schedule_2026[schedule_2026["spread_line"].notna()][
        ["week", "home_team", "away_team", "spread_line", "total_line", "home_moneyline", "away_moneyline"]]
    print(f"Real Vegas lines currently posted: {len(vegas_lines)}/{len(schedule_2026)} games "
          f"(weeks {int(vegas_lines['week'].min())}-{int(vegas_lines['week'].max())} - "
          f"books haven't published the rest yet, expected this far before the season)")

    comparison = our_predictions.merge(vegas_lines, on=["week", "home_team", "away_team"], how="inner")

    home_raw = moneyline_to_implied_prob(comparison["home_moneyline"])
    away_raw = moneyline_to_implied_prob(comparison["away_moneyline"])
    vig_total = home_raw + away_raw
    comparison["vegas_home_win_prob"] = home_raw / vig_total

    comparison["spread_disagreement"] = comparison["expected_spread"] - comparison["spread_line"]
    comparison["total_disagreement"] = comparison["expected_total"] - comparison["total_line"]
    comparison["win_prob_disagreement"] = comparison["home_win_probability"] - comparison["vegas_home_win_prob"]
    comparison["interpretation"] = np.select(
        [comparison["spread_disagreement"] > 0.5, comparison["spread_disagreement"] < -0.5],
        ["We favor home more", "We favor away more"], default="Agreement")

    print(f"\n{len(comparison)} games compared")
    print(f"Avg spread disagreement: {comparison['spread_disagreement'].mean():+.2f} pts")
    print(f"Avg total disagreement: {comparison['total_disagreement'].mean():+.2f} pts")
    print(f"corr(our spread, Vegas spread): {comparison['expected_spread'].corr(comparison['spread_line']):+.3f}")
    print(f"corr(our win prob, Vegas win prob): {comparison['home_win_probability'].corr(comparison['vegas_home_win_prob']):+.3f}")

    print(f"\nBiggest disagreements (spread, absolute):")
    biggest = comparison.reindex(comparison["spread_disagreement"].abs().sort_values(ascending=False).index).head(5)
    for _, g in biggest.iterrows():
        print(f"  Week {g['week']}: {g['home_team']} vs {g['away_team']} - "
              f"we say {g['expected_spread']:+.1f}, Vegas says {g['spread_line']:+.1f} "
              f"(delta {g['spread_disagreement']:+.1f})")

    out_path = os.path.join(os.path.join(PROJECT_ROOT, "data", "diagnostic"), "vegas_comparison_2026_preseason.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    return comparison


def identify_edges(comparison=None):
    if comparison is None:
        comparison = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "diagnostic", "vegas_comparison_2026_preseason.csv"))

    comparison["confidence"] = (comparison["home_win_probability"] - 0.5).abs()
    comparison["disagreement_size"] = comparison["spread_disagreement"].abs()
    comparison["potential_edge"] = (comparison["confidence"] > 0.15) & (comparison["disagreement_size"] > 1.5)

    edges = comparison[comparison["potential_edge"]].sort_values("disagreement_size", ascending=False)
    print(f"\n{len(edges)} potential edges (confidence>0.15, spread disagreement>1.5 pts) out of {len(comparison)} games:")
    for _, e in edges.head(10).iterrows():
        direction = "we favor home" if e["spread_disagreement"] > 0 else "we favor away"
        print(f"  Week {e['week']}: {e['home_team']} vs {e['away_team']} - {direction}, "
              f"disagreement {e['disagreement_size']:.1f} pts, our win prob {e['home_win_probability']:.1%}")
    return edges
