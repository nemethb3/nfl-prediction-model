"""Real offensive/defensive Elo split: separate O_Elo (scoring ability) and
D_Elo (scoring prevention) per team, updated from real game-by-game scoring
margins relative to a real, opponent-adjusted expectation - not reusing
elo_model.py's win/loss Elo chain (that tracks who wins, not by how much
each side scores), but built following the same real, causal,
leak-free-pre-game-snapshot discipline that module already established.

Real bugs found and fixed in the originally pasted spec before writing
this:
1. Assumed `data/processed/games_2015_2025.csv` with a `season_type`
   column - doesn't exist. Real file is
   data/backtest/game_results_2015_2025.csv (game_type, home_score,
   away_score, total).
2. BASELINE_SCORE was a dead parameter: the spec computed
   home_expected/away_expected as BASELINE*(o/1500)*(1500/d), then
   immediately renormalized both to sum to a fixed 42 -
   BASELINE_SCORE cancels out of that ratio algebraically (verified by
   hand), so its real value never affected anything.
3. That same renormalize-to-a-fixed-total step is a deeper, real design
   flaw, not just a cosmetic one: it forces EVERY game's expected combined
   score to exactly 42, regardless of the two teams' real O/D ratings -
   structurally throwing away the real game-level scoring-environment
   signal (blowout-prone vs. defensive-struggle matchups) that a real O/D
   split is supposed to capture in the first place. Fixed here by
   computing each side's expected score independently from a real
   opponent ratio, with no artificial shared-total normalization, and
   using a real, empirically-measured league-average (22.79 real points/
   team/REG-game, 2015-2025) instead of the spec's asserted, unverified 21.
4. K=32 was asserted, not fit. This module accepts k_factor as a real
   parameter instead - see backtest_offensive_defensive_elo.py's real,
   disclosed grid search over k_factor before picking one.
5. Real per-team-week history records BOTH the pre-game and post-game
   O/D Elo - the originally pasted spec only recorded post-game values,
   which the backtest script then looked up BY THAT SAME GAME's
   (season, week) to "predict" that exact game - a real information-
   leakage bug (using the outcome-updated rating to predict the outcome
   that produced it). Fixed by keeping both, and having the backtest use
   only the pre-game snapshot, the same real leak-free discipline this
   project's Elo carryover and point-in-time signals already use
   elsewhere.
"""

import json
import os

import numpy as np
import pandas as pd

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FINAL_RATINGS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "team_elo_offensive_defensive_2025.json")
HISTORY_PATH = os.path.join(PROCESSED_DIR, "team_elo_history_offensive_defensive_2015_2025.json")

STARTING_ELO = 1500.0
EARLIEST_SEASON, LATEST_SEASON = 2015, 2025


def _real_league_avg_points(games):
    return float(pd.concat([games["home_score"], games["away_score"]]).mean())


def compute_offensive_defensive_elo(k_factor=24.0, save=True):
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    games = games[games["game_type"] == "REG"].copy()
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    league_avg_pts = _real_league_avg_points(games)

    o_elo, d_elo = {}, {}

    def get(d, team):
        return d.get(team, STARTING_ELO)

    rows = []
    for _, g in games.iterrows():
        season, week = int(g["season"]), int(g["week"])
        home, away = g["home_team"], g["away_team"]
        home_score, away_score = float(g["home_score"]), float(g["away_score"])

        home_o_before, home_d_before = get(o_elo, home), get(d_elo, home)
        away_o_before, away_d_before = get(o_elo, away), get(d_elo, away)

        # Real, opponent-adjusted expectation - no fixed-total renormalization
        # (see module docstring #3): each side's expected score is a real
        # function of its own offense vs. the real opponent's defense only.
        home_expected = league_avg_pts * (home_o_before / away_d_before)
        away_expected = league_avg_pts * (away_o_before / home_d_before)

        home_o_move = k_factor * (home_score - home_expected) / league_avg_pts
        away_o_move = k_factor * (away_score - away_expected) / league_avg_pts
        home_d_move = -k_factor * (away_score - away_expected) / league_avg_pts
        away_d_move = -k_factor * (home_score - home_expected) / league_avg_pts

        home_o_after, home_d_after = home_o_before + home_o_move, home_d_before + home_d_move
        away_o_after, away_d_after = away_o_before + away_o_move, away_d_before + away_d_move

        o_elo[home], d_elo[home] = home_o_after, home_d_after
        o_elo[away], d_elo[away] = away_o_after, away_d_after

        rows.append({
            "game_id": g["game_id"], "season": season, "week": week,
            "home_team": home, "away_team": away,
            "home_o_elo_before": round(home_o_before, 2), "home_d_elo_before": round(home_d_before, 2),
            "away_o_elo_before": round(away_o_before, 2), "away_d_elo_before": round(away_d_before, 2),
            "home_o_elo_after": round(home_o_after, 2), "home_d_elo_after": round(home_d_after, 2),
            "away_o_elo_after": round(away_o_after, 2), "away_d_elo_after": round(away_d_after, 2),
        })

    history_df = pd.DataFrame(rows)

    final_ratings = {}
    for team in sorted(set(o_elo) | set(d_elo)):
        final_ratings[team] = {
            "o_elo": round(get(o_elo, team), 1),
            "d_elo": round(get(d_elo, team), 1),
            "total_elo": round(get(o_elo, team) + get(d_elo, team), 1),
        }

    if save:
        os.makedirs(os.path.dirname(FINAL_RATINGS_PATH), exist_ok=True)
        with open(FINAL_RATINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(final_ratings, f, indent=2)
            record_generation("team_elo_offensive_defensive_2025")
        history_df.to_csv(HISTORY_PATH.replace(".json", ".csv"), index=False)
        print(f"Real league-average points/team/REG-game (2015-2025): {league_avg_pts:.2f}")
        print(f"Wrote {FINAL_RATINGS_PATH}")
        print(f"Wrote {HISTORY_PATH.replace('.json', '.csv')}")

    return history_df, final_ratings, league_avg_pts


def _fit_prob_to_spread(p, point_diff):
    """Real candidate-form comparison (linear/logit/normal, picked by real
    train MAE) - the exact same real methodology
    elo_game_prediction.fit_probability_to_spread_conversion uses,
    generalized here to take any real win-probability array so both the
    single-Elo and O/D-Elo systems are fit and validated identically."""
    from scipy.stats import norm as _norm
    p = np.clip(p, 0.01, 0.99)
    candidates = {}
    a, b = np.polyfit(p - 0.5, point_diff, 1)
    candidates["linear"] = (a, b, np.mean(np.abs((a * (p - 0.5) + b) - point_diff)))
    logit_p = np.log(p / (1 - p))
    a, b = np.polyfit(logit_p, point_diff, 1)
    candidates["logit"] = (a, b, np.mean(np.abs((a * logit_p + b) - point_diff)))
    probit_p = _norm.ppf(p)
    a, b = np.polyfit(probit_p, point_diff, 1)
    candidates["normal"] = (a, b, np.mean(np.abs((a * probit_p + b) - point_diff)))
    best_form = min(candidates, key=lambda k: candidates[k][2])
    a, b, train_mae = candidates[best_form]
    if best_form == "linear":
        pred = a * (p - 0.5) + b
    elif best_form == "logit":
        pred = a * logit_p + b
    else:
        pred = a * probit_p + b
    resid_std = float(np.std(point_diff - pred))
    return {"form": best_form, "a": float(a), "b": float(b), "resid_std": resid_std, "train_mae": float(train_mae)}


def predict_od_spread(p, fitted):
    from scipy.stats import norm as _norm
    p = np.clip(p, 0.01, 0.99)
    if fitted["form"] == "linear":
        return fitted["a"] * (p - 0.5) + fitted["b"]
    elif fitted["form"] == "logit":
        return fitted["a"] * np.log(p / (1 - p)) + fitted["b"]
    return fitted["a"] * _norm.ppf(p) + fitted["b"]


def fit_od_elo_model(train_seasons=range(2015, 2024), k_factor=180.0):
    """Real, complete O/D Elo prediction model: a win-probability logistic
    regression on od_elo_spread, plus a real probability->point-spread
    conversion fit the same way (and on the same real train_seasons
    window) as elo_game_prediction.fit_probability_to_spread_conversion -
    for a genuinely fair, apples-to-apples comparison/replacement.
    train_seasons defaults to 2015-2023, matching that real production
    function's own default (2024/2025 held out, same real double-holdout
    discipline the rest of this project already uses)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    history_df, _, _ = compute_offensive_defensive_elo(k_factor=k_factor, save=False)
    history_df["od_elo_spread"] = ((history_df["home_o_elo_before"] - history_df["away_d_elo_before"]) -
                                    (history_df["away_o_elo_before"] - history_df["home_d_elo_before"]))

    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    games = games[games["game_type"] == "REG"].copy()
    games["point_diff"] = games["home_score"] - games["away_score"]
    train_pd = games[games["season"].isin(list(train_seasons))][["game_id", "point_diff"]]

    train = history_df.merge(train_pd, on="game_id", how="inner")
    train = train[train["point_diff"] != 0]  # real, rare ties excluded from the probability fit
    train["home_won"] = (train["point_diff"] > 0).astype(int)

    scaler = StandardScaler().fit(train[["od_elo_spread"]].to_numpy())
    prob_model = LogisticRegression(max_iter=1000, random_state=42)
    prob_model.fit(scaler.transform(train[["od_elo_spread"]].to_numpy()), train["home_won"].to_numpy())

    train_win_prob = prob_model.predict_proba(scaler.transform(train[["od_elo_spread"]].to_numpy()))[:, 1]
    spread_model = _fit_prob_to_spread(train_win_prob, train["point_diff"].to_numpy())

    return {"k_factor": k_factor, "scaler_mean": float(scaler.mean_[0]), "scaler_scale": float(scaler.scale_[0]),
            "prob_intercept": float(prob_model.intercept_[0]), "prob_coef": float(prob_model.coef_[0][0]),
            "spread_model": spread_model}


def od_elo_win_probability(od_elo_spread, fitted):
    z = fitted["prob_intercept"] + fitted["prob_coef"] * (
        (np.asarray(od_elo_spread, dtype=float) - fitted["scaler_mean"]) / fitted["scaler_scale"])
    return 1.0 / (1.0 + np.exp(-z))


def generate_od_elo_game_spreads(season, fitted):
    """Real per-game O/D Elo predictions for `season`, in the same shape
    elo_game_prediction.generate_elo_game_spreads uses (game_id, week,
    home/away team, ratings, predicted_spread, ci_low_90/ci_high_90) -
    real, leak-free pre-game ratings for season<=2025 (from this module's
    real history), real regressed carryover for season>2025 (2026's
    preseason snapshot, same real convention generate_elo_game_spreads
    already uses for single-Elo)."""
    resid_std = fitted["spread_model"]["resid_std"]
    band = 1.645 * resid_std  # real 90% CI z-score, same real constant used everywhere else in this project

    if season <= 2025:
        history_df, _, _ = compute_offensive_defensive_elo(k_factor=fitted["k_factor"], save=False)
        rows = history_df[history_df["season"] == season].copy()
        rows["od_elo_spread"] = ((rows["home_o_elo_before"] - rows["away_d_elo_before"]) -
                                  (rows["away_o_elo_before"] - rows["home_d_elo_before"]))
        rows = rows.rename(columns={"home_o_elo_before": "home_o_elo", "home_d_elo_before": "home_d_elo",
                                     "away_o_elo_before": "away_o_elo", "away_d_elo_before": "away_d_elo"})
    else:
        from apply_season_regression_od_elo import apply_season_regression_od_elo
        from game_predictions import _load_schedule_for_season
        regressed = apply_season_regression_od_elo(k_factor=fitted["k_factor"])
        schedule = _load_schedule_for_season(season)
        reg = schedule[schedule["game_type"] == "REG"].copy()
        reg["home_o_elo"] = reg["home_team"].map(lambda t: regressed.get(t, {}).get("o_elo"))
        reg["home_d_elo"] = reg["home_team"].map(lambda t: regressed.get(t, {}).get("d_elo"))
        reg["away_o_elo"] = reg["away_team"].map(lambda t: regressed.get(t, {}).get("o_elo"))
        reg["away_d_elo"] = reg["away_team"].map(lambda t: regressed.get(t, {}).get("d_elo"))
        reg["od_elo_spread"] = (reg["home_o_elo"] - reg["away_d_elo"]) - (reg["away_o_elo"] - reg["home_d_elo"])
        reg["game_id"] = reg["home_team"] + "_" + reg["away_team"] + "_" + reg["week"].astype(str)
        reg["season"] = season
        rows = reg[["game_id", "season", "week", "home_team", "away_team",
                     "home_o_elo", "home_d_elo", "away_o_elo", "away_d_elo", "od_elo_spread"]].copy()

    rows["win_prob_home"] = od_elo_win_probability(rows["od_elo_spread"].to_numpy(), fitted)
    rows["predicted_spread"] = predict_od_spread(rows["win_prob_home"].to_numpy(), fitted["spread_model"])
    rows["ci_low_90"] = rows["predicted_spread"] - band
    rows["ci_high_90"] = rows["predicted_spread"] + band
    return rows[["game_id", "season", "week", "home_team", "away_team", "home_o_elo", "home_d_elo",
                 "away_o_elo", "away_d_elo", "win_prob_home", "predicted_spread",
                 "ci_low_90", "ci_high_90"]].reset_index(drop=True)


if __name__ == "__main__":
    _, ratings, _ = compute_offensive_defensive_elo()
    print(f"\nComputed real O/D Elo for {len(ratings)} teams")
    if "KC" in ratings:
        print(f"Example (KC): {ratings['KC']}")
