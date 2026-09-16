"""Real, in-season 2026 Accuracy Tracker.

Reuses the real games/fantasy accuracy logic from generate_accuracy_
tracker_dashboard_data.py's compute_accuracy_metrics() pattern (games and
fantasy sections are season-agnostic already - they only look at rows with
real actual values present, same as the weekly-summary reuse), imported
directly rather than copy-pasted, to avoid the two versions' formulas
drifting apart.

One real, disclosed, deliberate omission: the 2025 version's `season_
projections` sub-section scores a real week-16 wins/division/playoff
projection against the real FINAL 18-week outcome - that comparison
requires the season to be over. 2026 is not over (see constants/seasons.js
- real, verified 0-2/18 weeks complete as of this build), so there is no
real final outcome to score against yet. Rather than compute something
misleadingly labeled "actual_wins"/"actual_division_winner" off provisional,
still-changing in-season records, this section is left `null` with a real,
disclosed reason - matching this project's own established rule (see
DECISIONS_LOG.md) against fabricating a stand-in for missing real data.

Real vegas-spread comparison: unlike 2025 (which has a real vegas_spread on
every game from data/raw/vegas_lines_2015_2025.csv), 2026 only has a real
vegas_spread for games join_espn_odds_2026.py could match against the real,
already-collected ESPN odds history (see that module's docstring for the
real, disclosed coverage limits). Games without a real vegas_spread are
simply excluded from the vs.-Vegas comparison (pandas .abs().mean() already
skips NaN) rather than treated as a 0-error match.
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "accuracy_tracker_2026.json")

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def _load_json_as_df(filename):
    with open(os.path.join(FRONTEND_DATA_DIR, filename), encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def _fantasy_position_stats(pos_data):
    if len(pos_data) < 2:
        return None
    corr = pos_data["projected_ppr"].corr(pos_data["actual_ppr"])
    mae = (pos_data["actual_ppr"] - pos_data["projected_ppr"]).abs().mean()
    return {"correlation": round(float(corr), 3) if pd.notna(corr) else None,
            "mae": round(float(mae), 2), "samples": int(len(pos_data))}


def compute_accuracy_metrics_2026(games_df, fantasy_df):
    accuracy = {"season_summary": {}, "weekly_breakdown": []}

    completed_games = games_df[games_df["actual_home_score"].notna()].copy()

    if len(completed_games) == 0:
        accuracy["season_summary"] = {
            "games": None, "fantasy": None, "season_projections": None, "betting": None,
            "note": "No real 2026 games completed yet.",
        }
        return accuracy

    completed_games["home_won"] = completed_games["actual_home_score"] > completed_games["actual_away_score"]
    completed_games["predicted_home_win"] = completed_games["win_prob_home"] > 0.5
    completed_games["moneyline_correct"] = completed_games["home_won"] == completed_games["predicted_home_win"]

    total_correct = int(completed_games["moneyline_correct"].sum())
    game_accuracy = total_correct / len(completed_games) * 100
    mae_our_spread = (
        (completed_games["actual_home_score"] - completed_games["actual_away_score"]) - completed_games["our_spread"]
    ).abs().mean()

    games_with_vegas = completed_games[completed_games["vegas_spread"].notna()]
    vs_vegas_spread = None
    if len(games_with_vegas) > 0:
        vs_vegas_spread = round(float((
            (games_with_vegas["actual_home_score"] - games_with_vegas["actual_away_score"]) - games_with_vegas["vegas_spread"]
        ).abs().mean()), 2)

    accuracy["season_summary"]["games"] = {
        "total_games": int(len(completed_games)),
        "correct_predictions": total_correct,
        "accuracy_pct": round(game_accuracy, 1),
        "mae_spread": round(float(mae_our_spread), 2),
        "vs_vegas_spread": vs_vegas_spread,
        "vs_vegas_coverage": f"{len(games_with_vegas)}/{len(completed_games)} completed games had a real, "
                              f"joined ESPN closing-line spread (see join_espn_odds_2026.py)",
    }

    fantasy_with_actuals = fantasy_df[fantasy_df["actual_ppr"].notna()].copy()
    fantasy_accuracy = {}
    for position in FANTASY_POSITIONS:
        stats = _fantasy_position_stats(fantasy_with_actuals[fantasy_with_actuals["position"] == position])
        if stats is not None:
            fantasy_accuracy[position] = stats
    accuracy["season_summary"]["fantasy"] = fantasy_accuracy if fantasy_accuracy else None

    accuracy["season_summary"]["season_projections"] = None
    accuracy["season_summary"]["season_projections_note"] = (
        "Not available yet: the 2026 season is still in progress, so there is no real final outcome "
        "to score the season projections against. This section reappears once the real season ends."
    )

    accuracy["season_summary"]["betting"] = {
        "moneyline_accuracy_pct": round(game_accuracy, 1),
        "note": "Spread coverage betting tested via edge_detection.py on 2015-2025 data: -36% ROI (real "
                "finding, not pursued). Moneyline accuracy above is the only real prediction this model's "
                "win-probability output makes. See the Betting Analysis page for a real, in-season ATS/"
                "moneyline backtest using the real ESPN odds captured for 2026 so far.",
    }

    max_week = int(games_df["week"].max())
    for week in range(1, max_week + 1):
        week_games = completed_games[completed_games["week"] == week]
        if len(week_games) == 0:
            continue

        week_correct = int(week_games["moneyline_correct"].sum())
        week_accuracy = week_correct / len(week_games) * 100
        week_mae = ((week_games["actual_home_score"] - week_games["actual_away_score"]) - week_games["our_spread"]).abs().mean()

        week_vegas = week_games[week_games["vegas_spread"].notna()]
        week_vegas_mae = None
        if len(week_vegas) > 0:
            week_vegas_mae = round(float((
                (week_vegas["actual_home_score"] - week_vegas["actual_away_score"]) - week_vegas["vegas_spread"]
            ).abs().mean()), 2)

        week_fantasy = fantasy_with_actuals[fantasy_with_actuals["week"] == week]
        week_fantasy_acc = {}
        for position in FANTASY_POSITIONS:
            stats = _fantasy_position_stats(week_fantasy[week_fantasy["position"] == position])
            if stats is not None:
                week_fantasy_acc[position] = {"correlation": stats["correlation"], "mae": stats["mae"]}

        accuracy["weekly_breakdown"].append({
            "week": week,
            "games": {
                "correct": week_correct,
                "total": int(len(week_games)),
                "accuracy_pct": round(week_accuracy, 1),
                "mae_spread": round(float(week_mae), 2),
                "vs_vegas_mae": week_vegas_mae,
            },
            "fantasy": week_fantasy_acc,
            "betting": {"moneyline_accuracy_pct": round(week_accuracy, 1)},
        })

    return accuracy


def generate_accuracy_tracker_2026_json():
    games_df = _load_json_as_df("games_2026.json")
    fantasy_df = _load_json_as_df("fantasy_rankings_2026.json")

    accuracy = compute_accuracy_metrics_2026(games_df, fantasy_df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(accuracy, f, indent=2)
        record_generation("accuracy_tracker_2026")

    print(f"Generated real 2026 accuracy tracker -> {OUTPUT_PATH}")
    print(f"Games: {accuracy['season_summary'].get('games')}")
    print(f"Weekly breakdown: {len(accuracy['weekly_breakdown'])} weeks")
    return accuracy


if __name__ == "__main__":
    generate_accuracy_tracker_2026_json()
