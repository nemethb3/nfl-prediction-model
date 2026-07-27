"""Phase 2 Component 2.2: QB-Specific Elo Model.

Corrects 3 issues found in the spec before building:

1. "opponent_qb_elo" as the comparison baseline for expected performance is
   a real football-logic error - a QB's performance depends on the
   opposing DEFENSE, not the opposing team's own QB rating (they never
   play against each other in a way that makes their ratings directly
   comparable). update_qb_elo()'s parameter is renamed opponent_team_elo
   and uses the opposing TEAM's real, already-validated Elo (elo_model.py's
   multi-season chain) instead - it already reflects real defensive
   quality (imperfectly, same as everywhere else team Elo is used in this
   project, but far more defensibly than an opposing QB's own rating).

2. "weights learned from backtest (try different ratios, pick best)" would
   fit the blend weight on the same 2025 data then validated against - the
   identical leakage class already caught and fixed twice this session
   (Component 2's EPA-Elo blend, Component C's Vegas-Elo blend). Fixed the
   same way: leave-one-week-out CV.

3. performance_score is built from the REAL QB EPA/play distribution
   (verified before building: mean=0.035, std=0.362 across 5994 real QB
   game-weeks with >=10 attempts, player_weekly_stats.csv), computed leak-
   free from TRAIN seasons only (2015-2024, excluding the 2025 holdout) -
   not an arbitrary 0-1 scale.

Disclosed simplification (not hidden): the probability->spread conversion
used to score blended ratings REUSES elo_game_prediction.py's already-
fitted team-Elo conversion (a=72.596, b=-1.641) applied to the blended
rating difference, rather than refitting a new conversion specifically for
blended ratings from scratch. Since the blend is a weighted average of two
quantities on the same nominal Elo scale (both centered near 1500), this is
a reasonable, scope-appropriate reuse - not re-verified against a from-
scratch refit, which would be the more rigorous (but much more expensive)
alternative.

"Starter" is the real QB with the most real attempts for that team that
week (a standard, verifiable proxy, minimum QB_MIN_ATTEMPTS_AS_STARTER=5) -
lets a backup taking over register immediately, the actual point of this
component per its own stated PURPOSE.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

QB_ELO_K_FACTOR = 10  # consistent with team Elo, per spec's own instruction
QB_MIN_ATTEMPTS_AS_STARTER = 5
QB_REGRESSION_TO_MEAN = 1.0 / 3.0  # same season-boundary constant as elo_model.py's team Elo
TRAIN_SEASONS = range(2015, 2025)
HOLDOUT_SEASON = 2025


def _qb_epa_distribution(seasons=TRAIN_SEASONS):
    """Real, leak-free (train-seasons-only) QB EPA/play mean+std, used to
    normalize performance_score - see module docstring #3."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    qb = pws[(pws["position"] == "QB") & (pws["season"].isin(list(seasons))) & (pws["season_type"] == "REG")
             & (pws["attempts"] >= QB_MIN_ATTEMPTS_AS_STARTER)].copy()
    qb["epa_per_play"] = qb["passing_epa"] / qb["attempts"]
    return float(qb["epa_per_play"].mean()), float(qb["epa_per_play"].std())


def calculate_qb_performance_metric(epa_per_play, mean, std):
    """Logistic transform of the real EPA/play z-score -> 0-1 performance
    score (0.5 = exactly league-average that game)."""
    z = (epa_per_play - mean) / std
    return 1.0 / (1.0 + np.exp(-z))


def update_qb_elo(qb_elo_old, performance_score, opponent_team_elo, k_factor=QB_ELO_K_FACTOR):
    """opponent_team_elo, not opponent_qb_elo - see module docstring #1."""
    expected = 1.0 / (1.0 + 10.0 ** (-(qb_elo_old - opponent_team_elo) / 400.0))
    return qb_elo_old + k_factor * (performance_score - expected)


def _team_elo_before_lookup(team_elo_backtest_df):
    home = team_elo_backtest_df[["home_team", "season", "week", "home_elo_before"]].rename(
        columns={"home_team": "team", "home_elo_before": "elo_before"})
    away = team_elo_backtest_df[["away_team", "season", "week", "away_elo_before"]].rename(
        columns={"away_team": "team", "away_elo_before": "elo_before"})
    long_df = pd.concat([home, away], ignore_index=True)
    return {(r.team, r.season, r.week): r.elo_before for r in long_df.itertuples()}


def initialize_qb_elo_ratings(season=HOLDOUT_SEASON, base_elo=1500.0):
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    qb = pws[(pws["position"] == "QB") & (pws["season"] == season) & (pws["season_type"] == "REG")
             & (pws["attempts"] >= QB_MIN_ATTEMPTS_AS_STARTER)]
    starters = qb.groupby("player_id").agg(qb_name=("player_display_name", "first"), team=("recent_team", "last"),
                                            games_played=("week", "nunique")).reset_index()
    starters["elo"] = base_elo
    return starters.rename(columns={"player_id": "qb_id"})


def run_multi_season_qb_elo(seasons, team_elo_backtest_df, k_factor=QB_ELO_K_FACTOR):
    """Continuous real QB Elo across multiple seasons, updated each real
    game against the opposing TEAM's real Elo (see module docstring #1),
    with the same season-boundary regression-to-mean elo_model.py's team
    Elo uses."""
    mean, std = _qb_epa_distribution([s for s in seasons if s in TRAIN_SEASONS] or TRAIN_SEASONS)

    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    qb = pws[(pws["position"] == "QB") & (pws["season"].isin(list(seasons))) & (pws["season_type"] == "REG")
             & (pws["attempts"] >= QB_MIN_ATTEMPTS_AS_STARTER)].copy()
    qb["epa_per_play"] = qb["passing_epa"] / qb["attempts"]
    starters = qb.loc[qb.groupby(["recent_team", "season", "week"])["attempts"].idxmax()].sort_values(["season", "week"])

    team_elo_lookup = _team_elo_before_lookup(team_elo_backtest_df)

    current = {}
    rows = []
    for season in sorted(set(seasons)):
        for pid in list(current.keys()):
            current[pid] = current[pid] + QB_REGRESSION_TO_MEAN * (1500.0 - current[pid])

        for _, row in starters[starters["season"] == season].iterrows():
            pid, team, week, opp = row["player_id"], row["recent_team"], row["week"], row["opponent_team"]
            if pid not in current:
                current[pid] = 1500.0
            opp_elo = team_elo_lookup.get((opp, season, week))
            if opp_elo is None:
                continue

            perf = calculate_qb_performance_metric(row["epa_per_play"], mean, std)
            elo_before = current[pid]
            current[pid] = update_qb_elo(elo_before, perf, opp_elo, k_factor)

            rows.append({"player_id": pid, "player_name": row["player_display_name"], "team": team,
                         "season": season, "week": week, "epa_per_play": row["epa_per_play"],
                         "performance_score": perf, "elo_before": elo_before, "elo_after": current[pid]})

    return pd.DataFrame(rows), dict(current)


def blend_team_and_qb_elo(team_elo, qb_elo, team_weight=0.6):
    return team_weight * team_elo + (1.0 - team_weight) * qb_elo


def get_effective_team_strength_for_game(team, week, season, team_elo_lookup, qb_elo_before_lookup, team_weight=0.6):
    team_elo = team_elo_lookup.get((team, season, week), 1500.0)
    qb_elo = qb_elo_before_lookup.get((team, season, week), 1500.0)  # real fallback: neutral if no real starter record
    return blend_team_and_qb_elo(team_elo, qb_elo, team_weight)


def _score_blended_games(team_weight, games, qb_before_lookup, fitted_model):
    from elo_game_prediction import calculate_win_probability_from_elo, predict_game_spread_from_elo
    g = games.copy()
    g["home_qb"] = g.apply(lambda r: qb_before_lookup.get((r["home_team"], r["season"], r["week"]), 1500.0), axis=1)
    g["away_qb"] = g.apply(lambda r: qb_before_lookup.get((r["away_team"], r["season"], r["week"]), 1500.0), axis=1)
    g["home_blended"] = blend_team_and_qb_elo(g["home_elo_before"], g["home_qb"], team_weight)
    g["away_blended"] = blend_team_and_qb_elo(g["away_elo_before"], g["away_qb"], team_weight)
    g["predicted_spread"] = predict_game_spread_from_elo(g["home_blended"], g["away_blended"], fitted_model)
    return g


def validate_qb_elo_backtest(season=HOLDOUT_SEASON, checkpoint_weeks=(1, 4, 8, 12, 16)):
    from elo_model import run_multi_season_elo, TRAIN_SEASONS as TEAM_TRAIN_SEASONS
    from elo_game_prediction import ELO_K_FACTOR, ELO_HOME_FIELD, fit_probability_to_spread_conversion, _load_game_results

    team_backtest_df, _, _ = run_multi_season_elo(range(min(TEAM_TRAIN_SEASONS), season + 1),
                                                    k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
    fitted_model = fit_probability_to_spread_conversion()

    qb_events, _ = run_multi_season_qb_elo(range(min(TEAM_TRAIN_SEASONS), season + 1), team_backtest_df)
    qb_before_lookup = {(r.team, r.season, r.week): r.elo_before for r in qb_events.itertuples()}

    games = team_backtest_df[team_backtest_df["season"] == season].copy()
    actual = _load_game_results([season])[["game_id", "point_diff"]]
    games = games.merge(actual, on="game_id", how="inner")

    weight_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    all_weeks = sorted(games["week"].unique())

    loocv_rows = []
    for W in checkpoint_weeks:
        train_weeks = [w for w in all_weeks if w != W]
        best = None
        for tw in weight_grid:
            scored = _score_blended_games(tw, games[games["week"].isin(train_weeks)], qb_before_lookup, fitted_model)
            mae = float(np.mean(np.abs(scored["predicted_spread"] - scored["point_diff"])))
            if best is None or mae < best[1]:
                best = (tw, mae)
        chosen_weight, train_mae = best

        held_out = _score_blended_games(chosen_weight, games[games["week"] == W], qb_before_lookup, fitted_model)
        held_out_mae = float(np.mean(np.abs(held_out["predicted_spread"] - held_out["point_diff"]))) if len(held_out) else np.nan
        print(f"Week {W:>2}: LOOCV chosen team_weight={chosen_weight:.2f} (train MAE={train_mae:.2f}) | "
              f"held-out week {W} MAE={held_out_mae:.2f} (n={len(held_out)})")
        loocv_rows.append({"week": W, "team_weight": chosen_weight, "train_mae": train_mae,
                           "held_out_mae": held_out_mae, "n": len(held_out)})
    loocv_df = pd.DataFrame(loocv_rows)
    avg_weight = float(loocv_df["team_weight"].mean())

    team_only = _score_blended_games(1.0, games, qb_before_lookup, fitted_model)
    blended_full = _score_blended_games(avg_weight, games, qb_before_lookup, fitted_model)

    team_corr = team_only["predicted_spread"].corr(team_only["point_diff"])
    team_mae = float(np.mean(np.abs(team_only["predicted_spread"] - team_only["point_diff"])))
    blend_corr = blended_full["predicted_spread"].corr(blended_full["point_diff"])
    blend_mae = float(np.mean(np.abs(blended_full["predicted_spread"] - blended_full["point_diff"])))

    print(f"\n{'=' * 60}\nQB ELO BACKTEST (real {season}, LOOCV mean team_weight={avg_weight:.2f})\n{'=' * 60}")
    print(f"Team Elo alone (team_weight=1.0): corr={team_corr:+.3f} MAE={team_mae:.2f}")
    print(f"Team+QB blend (team_weight={avg_weight:.2f}): corr={blend_corr:+.3f} MAE={blend_mae:.2f}")
    print(f"Delta: corr {blend_corr - team_corr:+.3f} | MAE {blend_mae - team_mae:+.2f} "
          f"({'blend better' if blend_mae < team_mae else 'team alone better'})")
    print("=" * 60)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    loocv_df.to_csv(os.path.join(DIAGNOSTIC_DIR, "qb_elo_validation_2025.csv"), index=False, encoding="utf-8")

    return {"loocv": loocv_df, "avg_weight": avg_weight, "team_elo_alone_corr": team_corr,
            "team_elo_alone_mae": team_mae, "blended_corr": blend_corr, "blended_mae": blend_mae,
            "qb_events": qb_events}


def generate_qb_elo_ratings_report(season=HOLDOUT_SEASON, results=None):
    if results is None:
        results = validate_qb_elo_backtest(season)
    qb_events = results["qb_events"]
    season_events = qb_events[qb_events["season"] == season]
    final_ratings = season_events.sort_values("week").groupby("player_id").tail(1)[
        ["player_name", "team", "elo_after"]].sort_values("elo_after", ascending=False)

    movement = season_events.groupby("player_id").agg(
        player_name=("player_name", "first"), start_elo=("elo_before", "first"), end_elo=("elo_after", "last"))
    movement["change"] = movement["end_elo"] - movement["start_elo"]

    lines = ["=" * 60, f"QB ELO RATINGS REPORT (real {season})", "=" * 60]
    lines.append("\nTop 5 QBs by end-of-season Elo:")
    for _, r in final_ratings.head(5).iterrows():
        lines.append(f"  {r['player_name']} ({r['team']}): {r['elo_after']:.0f}")
    lines.append("\nBottom 5 QBs by end-of-season Elo:")
    for _, r in final_ratings.tail(5).iterrows():
        lines.append(f"  {r['player_name']} ({r['team']}): {r['elo_after']:.0f}")

    lines.append("\nBiggest risers:")
    for _, r in movement.sort_values("change", ascending=False).head(3).iterrows():
        lines.append(f"  {r['player_name']}: {r['change']:+.0f} Elo ({r['start_elo']:.0f} -> {r['end_elo']:.0f})")
    lines.append("Biggest fallers:")
    for _, r in movement.sort_values("change").head(3).iterrows():
        lines.append(f"  {r['player_name']}: {r['change']:+.0f} Elo ({r['start_elo']:.0f} -> {r['end_elo']:.0f})")

    lines.append(f"\nLOOCV blend weight (mean across checkpoints): team={results['avg_weight']:.2f}, "
                  f"qb={1 - results['avg_weight']:.2f}")
    lines.append(f"Team Elo alone: corr={results['team_elo_alone_corr']:+.3f} MAE={results['team_elo_alone_mae']:.2f}")
    lines.append(f"Team+QB blend: corr={results['blended_corr']:+.3f} MAE={results['blended_mae']:.2f}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    final_ratings.to_csv(os.path.join(PROCESSED_DIR, f"qb_elo_ratings_{season}.csv"), index=False, encoding="utf-8")
    season_events.to_csv(os.path.join(PROCESSED_DIR, f"qb_elo_by_week_{season}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "qb_elo_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    results = validate_qb_elo_backtest()
    generate_qb_elo_ratings_report(results=results)
