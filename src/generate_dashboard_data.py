"""Dashboard Section 1 data export: real 2025 predictions + real results, no fabrication.

Fields come from integrated_game_predictions_2025.csv (our model's output)
and schedules_2015_2025.csv (kickoff date/time, real final scores, QB
names, and the full real 2015-2025 game log used for recent-form/head-to-
head). win_prob_home/win_prob_away come from win_probability_backtest.py's
real, backtested winner (a logistic regression fit on real Vegas spread ->
real outcome) - see that module for the full comparison against an Elo-
based model and an asserted-constant heuristic. See DASHBOARD_DATA_GAPS.md
for what's still missing (no per-player confidence, no ATS-strategy
recommendation - that was tested separately and found harmful).

Sign convention (matches data_pipeline.py's documented convention and
verified here against real 2025 moneylines, 284/284 agreement): positive
spread = home team favored; negative = away team favored.
actual_spread_margin is signed the same way (home_score - away_score).

matchup_quality is bucketed via EMPIRICAL terciles of the real 272-game
net_edge_diff distribution (computed live from the data itself, not an
asserted +-1.0 threshold) - a real, if crude, categorization method.
Labeled from the HOME team's perspective (positive net_edge_diff = a real
matchup edge favoring the home team).

team_recent_form and head_to_head are both computed leak-free: only real
games with a real gameday STRICTLY BEFORE this game's own gameday are used,
crossing season boundaries into the full real 2015-2025 log where a team's
current season doesn't yet have enough games (same leak-free convention as
elo_model.py's carryover and every trailing-window formula this project has
built).
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

from win_probability_backtest import _load_2024_training_data, train_vegas_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "games_2025.json")

RECENT_FORM_N = 5
HEAD_TO_HEAD_MAX = 10  # most recent N real meetings shown, not just a count


def _full_real_game_log():
    """Every real REG+POST game 2015-2025 with a decided (non-null) score,
    long-format (one row per team per game) for recent-form lookups."""
    sched = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    sched = sched.dropna(subset=["home_score", "away_score", "gameday"]).copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"])

    home = sched.rename(columns={"home_team": "team", "away_team": "opponent",
                                  "home_score": "team_score", "away_score": "opp_score"})
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent",
                                  "away_score": "team_score", "home_score": "opp_score"})
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["result"] = pd.Series(
        pd.NA, index=long_df.index, dtype="object")
    long_df.loc[long_df["team_score"] > long_df["opp_score"], "result"] = "W"
    long_df.loc[long_df["team_score"] < long_df["opp_score"], "result"] = "L"
    long_df.loc[long_df["team_score"] == long_df["opp_score"], "result"] = "T"
    return long_df.sort_values("gameday")


def _team_recent_form(team, before_date, game_log):
    prior = game_log[(game_log["team"] == team) & (game_log["gameday"] < before_date)]
    prior = prior.sort_values("gameday").tail(RECENT_FORM_N)
    return prior["result"].tolist()


def _head_to_head(home_team, away_team, before_date, sched_wide):
    prior = sched_wide[
        (sched_wide["gameday"] < before_date) &
        (((sched_wide["home_team"] == home_team) & (sched_wide["away_team"] == away_team)) |
         ((sched_wide["home_team"] == away_team) & (sched_wide["away_team"] == home_team)))
    ].sort_values("gameday", ascending=False).head(HEAD_TO_HEAD_MAX)

    home_wins = int(((prior["home_team"] == home_team) & (prior["home_score"] > prior["away_score"])).sum() +
                     ((prior["away_team"] == home_team) & (prior["away_score"] > prior["home_score"])).sum())
    away_wins = int(((prior["home_team"] == away_team) & (prior["home_score"] > prior["away_score"])).sum() +
                     ((prior["away_team"] == away_team) & (prior["away_score"] > prior["home_score"])).sum())
    ties = int(len(prior) - home_wins - away_wins)
    return {"meetings_considered": int(len(prior)), "home_team_wins": home_wins,
            "away_team_wins": away_wins, "ties": ties}


def _elo_lookup(season=2025):
    """Real per-team, per-week Elo (elo_ratings_2025.csv's `elo_after`,
    i.e. POST-that-week's-game). For a leak-free "entering this week"
    rating, lag by one real week (use week W-1's elo_after for week W's
    game) - week 1 has no real prior entry and is left null rather than
    guessed at an initial value."""
    elo = pd.read_csv(os.path.join(PROCESSED_DIR, f"elo_ratings_{season}.csv"))
    return {(r["team"], int(r["week"])): float(r["elo_after"]) for _, r in elo.iterrows()}


def _win_probability_model():
    """The real backtest winner (win_probability_backtest.py): a logistic
    regression fit on real 2024 Vegas spread_line -> real home-win outcome
    (Brier 0.2512 on the real 2025 weeks 13-17 holdout, beating both an
    asserted-constant heuristic at 0.3149 and an Elo-based model at
    0.2874). Refit live here each run rather than hardcoding the
    coefficients, so it can't go stale - same convention as elo_game_
    prediction.fit_probability_to_spread_conversion(). Applied only to
    real `vegas_spread` (what it was validated on), not `our_spread`
    (which includes a matchup adjustment the model was never tested
    against)."""
    train = _load_2024_training_data()
    return train_vegas_model(train)


def _matchup_quality_label(net_edge_diff, q33, q67):
    if pd.isna(net_edge_diff):
        return None
    if net_edge_diff > q67:
        return "favorable_home"
    if net_edge_diff < q33:
        return "favorable_away"
    return "neutral"


def generate_games_json(season=2025):
    pred = pd.read_csv(os.path.join(PROCESSED_DIR, "integrated_game_predictions_2025.csv"))
    sched_full = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    sched_full["gameday"] = pd.to_datetime(sched_full["gameday"])
    sched = sched_full[sched_full["season"] == season]

    merged = pred.merge(
        sched[["game_id", "gameday", "gametime", "weekday", "home_score", "away_score",
               "home_qb_name", "away_qb_name"]],
        on="game_id", how="left")

    # Empirical terciles of the real net_edge_diff distribution for THIS
    # season's 272 games - not an asserted threshold.
    q33, q67 = merged["net_edge_diff"].quantile([0.33, 0.67])

    game_log = _full_real_game_log()
    elo_lookup = _elo_lookup(season)
    win_prob_model = _win_probability_model()

    games = []
    for _, r in merged.iterrows():
        has_result = pd.notna(r["home_score"]) and pd.notna(r["away_score"])
        home_score = int(r["home_score"]) if has_result else None
        away_score = int(r["away_score"]) if has_result else None

        actual_winner = None
        actual_spread_margin = None
        did_we_predict_correctly = None
        if has_result:
            actual_spread_margin = home_score - away_score
            if home_score > away_score:
                actual_winner = r["home_team"]
            elif away_score > home_score:
                actual_winner = r["away_team"]
            else:
                actual_winner = "TIE"

            if actual_winner == "TIE" or r["final_spread"] == 0:
                did_we_predict_correctly = None
            else:
                predicted_home_win = r["final_spread"] > 0
                did_we_predict_correctly = bool(predicted_home_win == (actual_winner == r["home_team"]))

        kickoff_datetime = None
        if pd.notna(r["gameday"]) and pd.notna(r["gametime"]):
            kickoff_datetime = f"{r['gameday'].date()}T{r['gametime']}:00"

        game_date = r["gameday"]
        home_form = _team_recent_form(r["home_team"], game_date, game_log) if pd.notna(game_date) else []
        away_form = _team_recent_form(r["away_team"], game_date, game_log) if pd.notna(game_date) else []
        h2h = _head_to_head(r["home_team"], r["away_team"], game_date, sched_full) if pd.notna(game_date) else None

        week = int(r["week"])
        home_elo = elo_lookup.get((r["home_team"], week - 1)) if week > 1 else None
        away_elo = elo_lookup.get((r["away_team"], week - 1)) if week > 1 else None

        win_prob_home = None
        if r["base_source"] == "vegas":
            win_prob_home = float(win_prob_model.predict_proba([[r["base_spread"]]])[0][1])

        games.append({
            "id": r["game_id"],
            "week": int(r["week"]),
            "weekday": r["weekday"] if pd.notna(r["weekday"]) else None,
            "kickoff_datetime": kickoff_datetime,
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_qb_name": r["home_qb_name"] if pd.notna(r["home_qb_name"]) else None,
            "away_qb_name": r["away_qb_name"] if pd.notna(r["away_qb_name"]) else None,
            "home_elo": round(home_elo, 1) if home_elo is not None else None,
            "away_elo": round(away_elo, 1) if away_elo is not None else None,
            "our_spread": float(r["final_spread"]),
            "vegas_spread": float(r["base_spread"]) if r["base_source"] == "vegas" else None,
            "win_prob_home": round(win_prob_home, 4) if win_prob_home is not None else None,
            "win_prob_away": round(1.0 - win_prob_home, 4) if win_prob_home is not None else None,
            "base_source": r["base_source"],
            "net_edge_diff": float(r["net_edge_diff"]) if pd.notna(r["net_edge_diff"]) else None,
            "matchup_quality": _matchup_quality_label(r["net_edge_diff"], q33, q67),
            "home_recent_form": home_form,
            "away_recent_form": away_form,
            "head_to_head": h2h,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_winner": actual_winner,
            "actual_spread_margin": actual_spread_margin,
            "did_we_predict_correctly": did_we_predict_correctly,
        })

    games.sort(key=lambda g: (g["week"], g["kickoff_datetime"] or ""))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
        record_generation("games_2025")

    n_with_results = sum(1 for g in games if g["actual_winner"] is not None)
    n_correct = sum(1 for g in games if g["did_we_predict_correctly"] is True)
    n_incorrect = sum(1 for g in games if g["did_we_predict_correctly"] is False)
    print(f"Generated {len(games)} games -> {OUTPUT_PATH}")
    print(f"  With real results: {n_with_results} | correct: {n_correct} | incorrect: {n_incorrect} "
          f"| push/tie (unscored): {n_with_results - n_correct - n_incorrect}")
    print(f"  matchup_quality empirical thresholds: favorable_away < {q33:.2f} < neutral < {q67:.2f} < favorable_home")
    return games


if __name__ == "__main__":
    generate_games_json()
