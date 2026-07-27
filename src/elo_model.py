"""Phase A Component 1: Elo Rating Model.

Corrects 4 issues found in the spec before building:

1. "vs. EPA baseline corr=+0.306" was the wrong metric. Traced +0.306 in
   PROGRESS.md: it's `offensive_strength`'s correlation to real OFFENSIVE EPA
   from the mid-session SOS-bug-fix entry - a component-level number, not a
   season-win-projection accuracy figure - and it was itself later superseded
   to +0.327 by the baseline-QB production update. The real, current EPA
   win-projection baseline (validated against actual 2025 outcomes, most
   recently in ensemble.py) is corr=+0.216, MAE=2.88 - the spec's MAE was
   already right, only the correlation got swapped in from an unrelated
   metric. Validated against the real number below.

2. schedules_2025.csv doesn't exist as a standalone file (only
   schedules_2015_2025.csv and schedules_2026.csv do) - uses
   game_predictions._load_schedule_for_season() instead, which conveniently
   already carries home_moneyline/away_moneyline/spread_line directly, so no
   separate Vegas file is needed here.

3. "Derive from Vegas win totals (like nfelo does)" - no such market exists
   in this dataset (already established in vegas_comparison.py: only
   per-game moneylines, no preseason season-win-total line). Substituted the
   already-validated compute_vegas_implied_wins() (devigged per-game
   moneyline sum -> season win total) as the Vegas signal, converted to an
   Elo rating via a points-per-win scale that's GRID-SEARCHED alongside
   K-factor on real 2015-2024 games (minimizing Brier score), not asserted.

4. The spec never addresses cross-season Elo carryover. Resolved by resetting
   every season fresh from its own Vegas-informed prior (initialize_elo_
   ratings(season)) rather than carrying raw Elo across seasons with an
   invented, unvalidated regression-to-mean constant - each season's Vegas
   line already encodes that season's real personnel changes.

One addition beyond the spec, disclosed rather than silently added: home-
field advantage. The spec's expected-win-probability formula has no home
term at all, which would systematically underrate home teams (real 2015-2024
home win rate is meaningfully above 50%, confirmed below). Fit empirically
from real games (_estimate_home_field_elo), not asserted.

One important caveat about the results (see validate_elo_backtest): a
Vegas-informed PRESEASON Elo that's never updated will closely track Vegas's
own accuracy almost by construction, since it imports Vegas's number as the
starting rating - "beats EPA" in that mode mostly means "Vegas beats EPA"
again (already established in ensemble.py), not that Elo-the-methodology is
good. The neutral-start (no Vegas signal) variant is the actual, non-
circular test of whether win/loss history alone (Elo's real, independent
signal) beats EPA's play-by-play signal - both are reported and clearly
labeled below.
"""

import os

import numpy as np
import pandas as pd

from constants import (
    EPA_BASELINE_SEASON_CORR_2025 as EPA_BASELINE_CORR,
    EPA_BASELINE_SEASON_MAE_2025 as EPA_BASELINE_MAE,
    VEGAS_BASELINE_SEASON_CORR_2025 as VEGAS_BASELINE_CORR,
    VEGAS_BASELINE_SEASON_MAE_2025 as VEGAS_BASELINE_MAE,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

TRAIN_SEASONS = range(2015, 2025)  # 2015-2024, real 2025 held out for honest validation
HOLDOUT_SEASON = 2025
LEAGUE_AVG_WINS = 8.5  # half of a 17-game season, by construction of a zero-sum league

# EPA / Vegas baselines this Elo model is judged against - measured from real
# 2025 outcomes (see ensemble.py's backtest, the most recent authoritative
# measurement, NOT the spec's stale +0.306 figure - see module docstring #1).
# Values now centralized in constants.py (AUDIT_2026-07-27.md) - imported above.


def _load_games_chronological(seasons):
    """Real REG-season games for `seasons`, ordered (season, week, game_id) so
    Elo updates process games in the order they actually happened. home_result
    is 1.0/0.0/0.5 (win/loss/tie) - ties are real and rare in this data (10
    across 2015-2025), not something that can be silently dropped."""
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    games = games[(games["season"].isin(list(seasons))) & (games["game_type"] == "REG")].copy()
    games["home_result"] = np.select(
        [games["home_score"] > games["away_score"], games["home_score"] < games["away_score"]],
        [1.0, 0.0], default=0.5)
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _estimate_home_field_elo(train_games):
    """Empirical home-field Elo offset: solves for the Elo-point gap that
    reproduces the real observed home win rate in the logistic formula, i.e.
    offset = 400 * log10(p / (1 - p)) where p = real home win rate."""
    home_win_rate = float(np.clip(train_games["home_result"].mean(), 0.01, 0.99))
    return 400.0 * np.log10(home_win_rate / (1.0 - home_win_rate))


def calculate_expected_win_probability(elo_home, elo_away, home_field_elo=0.0):
    """P(home win) = 1 / (1 + 10^(-(elo_home + home_field_elo - elo_away)/400))."""
    diff = (elo_home + home_field_elo) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def update_elo(elo_old, k_factor, actual_result, expected_probability):
    """actual_result: 1.0 win, 0.0 loss, 0.5 tie."""
    return elo_old + k_factor * (actual_result - expected_probability)


def initialize_elo_ratings(season, use_vegas_signals=True, points_per_win=25.0):
    """Team Elo ratings at the start of `season`. use_vegas_signals=True
    (default, per spec) centers each team at 1500 + (vegas_implied_wins -
    8.5) * points_per_win, using the real per-game devigged-moneyline season
    win total (compute_vegas_implied_wins) since no preseason win-total
    market exists in this data (see module docstring #3). points_per_win is
    grid-searched by learn_elo_hyperparameters(), not asserted here."""
    from game_predictions import _load_schedule_for_season
    from vegas_comparison import compute_vegas_implied_wins

    schedule = _load_schedule_for_season(season)
    reg = schedule[schedule["game_type"] == "REG"]
    teams = sorted(set(reg["home_team"]) | set(reg["away_team"]))

    if use_vegas_signals:
        vegas_wins = compute_vegas_implied_wins(schedule, season=season).set_index("team")["vegas_implied_wins"]
        elo_rating = 1500.0 + (vegas_wins.reindex(teams) - LEAGUE_AVG_WINS) * points_per_win
        elo_rating = elo_rating.values
    else:
        elo_rating = np.full(len(teams), 1500.0)

    return pd.DataFrame({"team": teams, "elo_rating": elo_rating, "elo_uncertainty": 1.0})


def run_elo_season(season, initial_ratings=None, k_factor=25.0, home_field_elo=0.0, points_per_win=25.0):
    """Processes every real REG game of `season` in chronological order,
    updating both teams' Elo after each. Returns (game-by-game backtest_df,
    final_elo_df)."""
    games = _load_games_chronological([season])
    if initial_ratings is None:
        initial_ratings = initialize_elo_ratings(season, use_vegas_signals=True, points_per_win=points_per_win)
    current = initial_ratings.set_index("team")["elo_rating"].to_dict()

    rows = []
    for g in games.itertuples():
        home, away = g.home_team, g.away_team
        elo_home_before, elo_away_before = current[home], current[away]
        expected = calculate_expected_win_probability(elo_home_before, elo_away_before, home_field_elo)
        actual = g.home_result

        elo_home_after = update_elo(elo_home_before, k_factor, actual, expected)
        elo_away_after = update_elo(elo_away_before, k_factor, 1.0 - actual, 1.0 - expected)
        current[home], current[away] = elo_home_after, elo_away_after

        rows.append({
            "game_id": g.game_id, "season": season, "week": g.week,
            "home_team": home, "away_team": away,
            "home_elo_before": elo_home_before, "away_elo_before": elo_away_before,
            "home_elo_after": elo_home_after, "away_elo_after": elo_away_after,
            "predicted_prob": expected, "actual_result": actual,
        })

    backtest_df = pd.DataFrame(rows)
    final_elo = pd.DataFrame({"team": list(current.keys()), "elo_rating": list(current.values())})
    return backtest_df, final_elo


def project_season_wins_from_elo(team_elos, schedule_df, home_field_elo=0.0):
    """Given a team_elos snapshot (any point in time) and a schedule (whole
    season or remaining games), sums per-game win probability into a season
    win projection with a 90% CI from real per-game Bernoulli variance (same
    convention as game_predictions.infer_season_wins_from_game_predictions)."""
    elo_lookup = team_elos.set_index("team")["elo_rating"]
    reg = schedule_df[schedule_df["game_type"] == "REG"] if "game_type" in schedule_df.columns else schedule_df

    rows = []
    for team in sorted(elo_lookup.index):
        home_games = reg[reg["home_team"] == team]
        away_games = reg[reg["away_team"] == team]

        win_probs = []
        for g in home_games.itertuples():
            win_probs.append(calculate_expected_win_probability(elo_lookup[team], elo_lookup[g.away_team], home_field_elo))
        for g in away_games.itertuples():
            win_probs.append(1.0 - calculate_expected_win_probability(elo_lookup[g.home_team], elo_lookup[team], home_field_elo))

        if not win_probs:
            continue
        win_probs = np.array(win_probs)
        projected_wins = float(win_probs.sum())
        variance = float((win_probs * (1 - win_probs)).sum())
        std_dev = float(np.sqrt(variance))

        rows.append({
            "team": team, "elo_rating": float(elo_lookup[team]), "projected_wins": projected_wins,
            "projected_wins_low_90": max(0.0, projected_wins - 1.645 * std_dev),
            "projected_wins_high_90": min(len(win_probs), projected_wins + 1.645 * std_dev),
            "num_games": len(win_probs),
        })

    return pd.DataFrame(rows).sort_values("projected_wins", ascending=False).reset_index(drop=True)


def run_multi_season_elo(seasons, k_factor=25.0, home_field_elo=0.0, regression_to_mean=1.0 / 3.0):
    """Continuous Elo across multiple seasons with NO Vegas signal anywhere -
    a genuinely independent win/loss-history-only rating (unlike a single-
    season neutral-1500-for-everyone reset, which carries zero information
    and trivially produces zero correlation - a real flaw caught in this
    module's own first validation pass, see validate_elo_backtest's
    docstring). All teams start at 1500 in the earliest season; between
    seasons every team's rating regresses `regression_to_mean` of the way
    back to 1500 - the standard convention used by public NFL Elo systems
    (e.g. 538's original ~1/3 regression). This fraction is ASSUMED, not
    fit - we lack a pre-2015 season to validate the very first reset
    against, and tuning it properly is future work, disclosed as a known
    limitation rather than silently asserted as exact.

    Returns (game-by-game backtest_df, {season: {team: preseason_elo}},
    final current ratings dict)."""
    current = {}
    all_rows = []
    ratings_at_season_start = {}
    for season in seasons:
        games = _load_games_chronological([season])
        teams_this_season = sorted(set(games["home_team"]) | set(games["away_team"]))
        for team in teams_this_season:
            if team not in current:
                current[team] = 1500.0
            else:
                current[team] = current[team] + regression_to_mean * (1500.0 - current[team])
        ratings_at_season_start[season] = dict(current)

        for g in games.itertuples():
            home, away = g.home_team, g.away_team
            elo_home_before, elo_away_before = current[home], current[away]
            expected = calculate_expected_win_probability(elo_home_before, elo_away_before, home_field_elo)
            actual = g.home_result
            elo_home_after = update_elo(elo_home_before, k_factor, actual, expected)
            elo_away_after = update_elo(elo_away_before, k_factor, 1.0 - actual, 1.0 - expected)
            current[home], current[away] = elo_home_after, elo_away_after
            all_rows.append({
                "game_id": g.game_id, "season": season, "week": g.week,
                "home_team": home, "away_team": away,
                "home_elo_before": elo_home_before, "away_elo_before": elo_away_before,
                "home_elo_after": elo_home_after, "away_elo_after": elo_away_after,
                "predicted_prob": expected, "actual_result": actual,
            })

    return pd.DataFrame(all_rows), ratings_at_season_start, current


def learn_elo_hyperparameters(train_seasons=TRAIN_SEASONS,
                               k_candidates=(10, 20, 30, 40, 50),
                               ppw_candidates=(15, 25, 35, 45)):
    """Grid-searches (K-factor, points_per_win) on real 2015-2024 games,
    minimizing Brier score (mean squared error of predicted vs. actual game
    outcome) - the direct, real target for a probabilistic rating system,
    not asserted from the spec's 'typical range 20-50'. home_field_elo is
    fit once from the same real games (not part of the grid - it's a
    measured constant, not a tunable one)."""
    train_games_by_season = {s: _load_games_chronological([s]) for s in train_seasons}
    all_train_games = _load_games_chronological(train_seasons)
    home_field_elo = _estimate_home_field_elo(all_train_games)
    print(f"[learn_elo_hyperparameters] empirical home-field Elo offset: {home_field_elo:+.1f} "
          f"(real {min(train_seasons)}-{max(train_seasons)} home win rate: {all_train_games['home_result'].mean():.3f})")

    best = None
    for k in k_candidates:
        for ppw in ppw_candidates:
            sq_errors = []
            for season in train_seasons:
                games = train_games_by_season[season]
                initial = initialize_elo_ratings(season, use_vegas_signals=True, points_per_win=ppw)
                current = initial.set_index("team")["elo_rating"].to_dict()
                for g in games.itertuples():
                    exp = calculate_expected_win_probability(current[g.home_team], current[g.away_team], home_field_elo)
                    actual = g.home_result
                    sq_errors.append((exp - actual) ** 2)
                    current[g.home_team] = update_elo(current[g.home_team], k, actual, exp)
                    current[g.away_team] = update_elo(current[g.away_team], k, 1.0 - actual, 1.0 - exp)
            brier = float(np.mean(sq_errors))
            if best is None or brier < best["brier"]:
                best = {"k_factor": k, "points_per_win": ppw, "brier": brier}

    print(f"[learn_elo_hyperparameters] winner: K={best['k_factor']}, points_per_win={best['points_per_win']} "
          f"(Brier={best['brier']:.4f} across {min(train_seasons)}-{max(train_seasons)})")
    return best["k_factor"], best["points_per_win"], home_field_elo, best["brier"]


def validate_elo_backtest(season=HOLDOUT_SEASON):
    """Real, honest validation against the completed 2025 season. Reports
    THREE things, not one, because a single headline number here would be
    misleading (see module docstring's circularity caveat):
    (a) Vegas-informed preseason Elo vs. actual - the spec's literal ask,
        but expected to closely track Vegas itself;
    (b) carryover Elo (2015-2024 win/loss history only, NO Vegas signal
        anywhere, regressed 1/3 toward the mean each season boundary) vs.
        actual - the real, non-circular test of Elo's own signal vs. EPA's.
        NOTE: an earlier version of this validation reset every team to a
        flat neutral 1500 for the whole 2025 season with no updates at all -
        that's zero information in, so it trivially produced ~zero
        correlation. That was a flaw in the test's construction, not a
        finding about Elo, caught before reporting it - carryover Elo
        (real accumulated win/loss history) is the actual non-circular test;
    (c) in-season-updating Elo's real-time game-level accuracy (Brier score),
        a diagnostic on the mechanism itself, independent of (a)/(b)."""
    from game_predictions import _load_schedule_for_season

    k_factor, points_per_win, home_field_elo, train_brier = learn_elo_hyperparameters()

    schedule = _load_schedule_for_season(season)
    reg_schedule = schedule[schedule["game_type"] == "REG"]
    actual = pd.read_csv(os.path.join(BACKTEST_DIR, "actual_wins_2025.csv"))

    print(f"\n{'=' * 70}\nELO MODEL VALIDATION (real {season}, K={k_factor}, "
          f"points_per_win={points_per_win}, home_field_elo={home_field_elo:+.1f})\n{'=' * 70}")

    # (a) Vegas-informed preseason Elo
    vegas_elo = initialize_elo_ratings(season, use_vegas_signals=True, points_per_win=points_per_win)
    vegas_proj = project_season_wins_from_elo(vegas_elo, reg_schedule, home_field_elo)
    vegas_merged = vegas_proj.merge(actual[["team", "actual_wins"]], on="team", how="inner")
    vegas_corr = vegas_merged["projected_wins"].corr(vegas_merged["actual_wins"])
    vegas_mae = float(np.mean(np.abs(vegas_merged["projected_wins"] - vegas_merged["actual_wins"])))
    print(f"\n(a) Vegas-informed preseason Elo: corr={vegas_corr:+.3f} MAE={vegas_mae:.2f} wins "
          f"[CAUTION: imports Vegas's own number as the starting rating - a strong result here "
          f"mostly re-confirms Vegas beats EPA (already known), not that Elo-the-method is good]")

    # (b) carryover Elo (real 2015-2024 win/loss history, no Vegas anywhere) - the real, non-circular test
    _, ratings_at_season_start, _ = run_multi_season_elo(
        range(min(TRAIN_SEASONS), season + 1), k_factor=k_factor, home_field_elo=home_field_elo)
    carryover_elo = pd.DataFrame(list(ratings_at_season_start[season].items()), columns=["team", "elo_rating"])
    carryover_proj = project_season_wins_from_elo(carryover_elo, reg_schedule, home_field_elo)
    carryover_merged = carryover_proj.merge(actual[["team", "actual_wins"]], on="team", how="inner")
    carryover_corr = carryover_merged["projected_wins"].corr(carryover_merged["actual_wins"])
    carryover_mae = float(np.mean(np.abs(carryover_merged["projected_wins"] - carryover_merged["actual_wins"])))
    print(f"(b) Carryover Elo (real win/loss history, no Vegas): corr={carryover_corr:+.3f} MAE={carryover_mae:.2f} wins "
          f"[the real test: pure win/loss-history signal vs. EPA's play-by-play signal]")

    print(f"\nBaselines for context: EPA model corr={EPA_BASELINE_CORR:+.3f} MAE={EPA_BASELINE_MAE:.2f} | "
          f"Vegas corr={VEGAS_BASELINE_CORR:+.3f} MAE={VEGAS_BASELINE_MAE:.2f}")
    print(f"(b) carryover Elo {'BEATS' if carryover_mae < EPA_BASELINE_MAE else 'does NOT beat'} the real EPA baseline on MAE")

    # (c) in-season updating Elo's real-time game-level accuracy
    backtest_df, final_elo = run_elo_season(season, k_factor=k_factor, home_field_elo=home_field_elo,
                                             points_per_win=points_per_win)
    game_brier = float(np.mean((backtest_df["predicted_prob"] - backtest_df["actual_result"]) ** 2))
    game_corr = backtest_df["predicted_prob"].corr(backtest_df["actual_result"])
    print(f"\n(c) In-season game-level accuracy: Brier={game_brier:.4f} (train Brier was {train_brier:.4f}) | "
          f"corr(predicted_prob, actual_result)={game_corr:+.3f}")

    # Save outputs
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    home_long = backtest_df[["home_team", "week", "home_elo_after"]].rename(
        columns={"home_team": "team", "home_elo_after": "elo_after"})
    away_long = backtest_df[["away_team", "week", "away_elo_after"]].rename(
        columns={"away_team": "team", "away_elo_after": "elo_after"})
    elo_by_week = pd.concat([home_long, away_long], ignore_index=True).sort_values(["team", "week"])
    elo_by_week.to_csv(os.path.join(PROCESSED_DIR, f"elo_ratings_{season}.csv"), index=False, encoding="utf-8")

    season_wins_out = vegas_proj.rename(columns={
        "elo_rating": "elo_rating_vegas_informed", "projected_wins": "projected_wins_vegas_informed",
        "projected_wins_low_90": "wins_low_90_vegas_informed", "projected_wins_high_90": "wins_high_90_vegas_informed",
    }).merge(carryover_proj.rename(columns={
        "elo_rating": "elo_rating_carryover", "projected_wins": "projected_wins_carryover",
        "projected_wins_low_90": "wins_low_90_carryover", "projected_wins_high_90": "wins_high_90_carryover",
    })[["team", "elo_rating_carryover", "projected_wins_carryover", "wins_low_90_carryover", "wins_high_90_carryover"]],
        on="team").merge(actual[["team", "actual_wins"]], on="team")
    season_wins_out.to_csv(os.path.join(PROCESSED_DIR, f"elo_season_wins_{season}.csv"), index=False, encoding="utf-8")
    backtest_df.to_csv(os.path.join(DIAGNOSTIC_DIR, f"elo_backtest_{season}.csv"), index=False, encoding="utf-8")

    print(f"\nSaved data/processed/elo_ratings_{season}.csv, elo_season_wins_{season}.csv, "
          f"data/diagnostic/elo_backtest_{season}.csv")
    print("=" * 70)

    return {
        "k_factor": k_factor, "points_per_win": points_per_win, "home_field_elo": home_field_elo,
        "vegas_informed_corr": vegas_corr, "vegas_informed_mae": vegas_mae,
        "carryover_corr": carryover_corr, "carryover_mae": carryover_mae,
        "game_level_brier": game_brier, "game_level_corr": game_corr,
    }


if __name__ == "__main__":
    validate_elo_backtest()
