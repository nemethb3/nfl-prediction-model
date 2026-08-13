"""Dashboard Section 4 data export: real accuracy metrics, no fabrication.

Reads from this project's own real dashboard exports rather than raw
pipeline files (games_2025.json, fantasy_rankings_2025.json,
season_projections_2025.json), since those already have the derived real
fields (win_prob_home, actual_ppr, etc.) that don't exist in the raw
processed CSVs - the same file-existence mistake caught and fixed in
earlier "generate_*_dashboard_data.py" scripts this session.

final_standings (real final wins/division-winner/playoff-team, from ALL 18
real weeks) reuses generate_season_projections_dashboard_data.py's real,
already-verified _real_records_through_week()/_compute_seeds() functions
with checkpoint_week=19 (covers every real week 1-18) rather than
reimplementing the same seeding logic a second time - the week-16
`season_projections_2025.json` snapshot is then scored against this real
final outcome.

Spread-coverage "accuracy" is deliberately NOT computed here - it isn't a
real prediction this project's model makes (no "which side covers" output
exists), and the one real backtest of a coverage-betting strategy
(edge_detection.py) found -36% ROI. Cited directly instead of inventing a
new, less rigorous metric that would look precise but measure nothing real.
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

from generate_season_projections_dashboard_data import (
    TEAM_TO_DIVISION, _real_records_through_week, _compute_seeds,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "accuracy_tracker_2025.json")

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def _load_json_as_df(filename):
    with open(os.path.join(FRONTEND_DATA_DIR, filename), encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def compute_final_standings():
    """Real final standings (all 18 real 2025 weeks) - reuses the same
    real seeding logic generate_season_projections_dashboard_data.py
    already validated, just run through the full season instead of a
    week-16 cutoff."""
    records = _real_records_through_week(checkpoint_week=19)
    seeds_afc = _compute_seeds(records, "AFC")
    seeds_nfc = _compute_seeds(records, "NFC")
    all_seeds = {**seeds_afc, **seeds_nfc}
    division_winners = {t for t, s in all_seeds.items() if s <= 4}

    rows = []
    for team in TEAM_TO_DIVISION:
        rows.append({
            "team": team,
            "actual_wins": records[team]["wins"],
            "actual_division_winner": team in division_winners,
            "actual_playoff_team": team in all_seeds,
        })
    return pd.DataFrame(rows)


def _fantasy_position_stats(pos_data):
    if len(pos_data) < 2:
        return None
    corr = pos_data["projected_ppr"].corr(pos_data["actual_ppr"])
    mae = (pos_data["actual_ppr"] - pos_data["projected_ppr"]).abs().mean()
    return {"correlation": round(float(corr), 3), "mae": round(float(mae), 2), "samples": int(len(pos_data))}


def compute_accuracy_metrics(games_df, fantasy_df, season_projections_df, final_standings_df):
    accuracy = {"season_summary": {}, "weekly_breakdown": []}

    completed_games = games_df[games_df["actual_home_score"].notna()].copy()
    completed_games["home_won"] = completed_games["actual_home_score"] > completed_games["actual_away_score"]
    completed_games["predicted_home_win"] = completed_games["win_prob_home"] > 0.5
    completed_games["moneyline_correct"] = completed_games["home_won"] == completed_games["predicted_home_win"]

    total_correct = int(completed_games["moneyline_correct"].sum())
    game_accuracy = total_correct / len(completed_games) * 100
    mae_our_spread = (
        (completed_games["actual_home_score"] - completed_games["actual_away_score"]) - completed_games["our_spread"]
    ).abs().mean()
    mae_vegas_spread = (
        (completed_games["actual_home_score"] - completed_games["actual_away_score"]) - completed_games["vegas_spread"]
    ).abs().mean()

    accuracy["season_summary"]["games"] = {
        "total_games": int(len(completed_games)),
        "correct_predictions": total_correct,
        "accuracy_pct": round(game_accuracy, 1),
        "mae_spread": round(float(mae_our_spread), 2),
        "vs_vegas_spread": round(float(mae_vegas_spread), 2),
    }

    fantasy_with_actuals = fantasy_df[fantasy_df["actual_ppr"].notna()].copy()
    fantasy_accuracy = {}
    for position in FANTASY_POSITIONS:
        stats = _fantasy_position_stats(fantasy_with_actuals[fantasy_with_actuals["position"] == position])
        if stats is not None:
            fantasy_accuracy[position] = stats
    accuracy["season_summary"]["fantasy"] = fantasy_accuracy

    projections_scored = season_projections_df.merge(
        final_standings_df[["team", "actual_wins", "actual_division_winner", "actual_playoff_team"]],
        on="team", how="left")
    div_correct = int((projections_scored["is_division_winner"] & projections_scored["actual_division_winner"]).sum())
    playoff_correct = int((projections_scored["is_playoff_team"] & projections_scored["actual_playoff_team"]).sum())
    proj_with_wins = projections_scored.dropna(subset=["projected_wins"])
    wins_error = (proj_with_wins["projected_wins"] - proj_with_wins["actual_wins"]).abs().mean()

    accuracy["season_summary"]["season_projections"] = {
        "teams": int(len(projections_scored)),
        "correct_division_winners": div_correct,
        "correct_playoff_teams": playoff_correct,
        "avg_wins_error": round(float(wins_error), 2),
    }

    accuracy["season_summary"]["betting"] = {
        "moneyline_accuracy_pct": round(game_accuracy, 1),
        "note": "Spread coverage betting tested via edge_detection.py: -36% ROI (real finding, not pursued). "
                "Moneyline accuracy above is the only real prediction this model's win-probability output makes.",
    }

    for week in range(1, 19):
        week_games = completed_games[completed_games["week"] == week]
        if len(week_games) == 0:
            continue

        week_correct = int(week_games["moneyline_correct"].sum())
        week_accuracy = week_correct / len(week_games) * 100
        week_mae = ((week_games["actual_home_score"] - week_games["actual_away_score"]) - week_games["our_spread"]).abs().mean()
        week_vegas_mae = ((week_games["actual_home_score"] - week_games["actual_away_score"]) - week_games["vegas_spread"]).abs().mean()

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
                "vs_vegas_mae": round(float(week_vegas_mae), 2),
            },
            "fantasy": week_fantasy_acc,
            "betting": {"moneyline_accuracy_pct": round(week_accuracy, 1)},
        })

    return accuracy


def generate_accuracy_tracker_json():
    games_df = _load_json_as_df("games_2025.json")
    fantasy_df = _load_json_as_df("fantasy_rankings_2025.json")
    season_projections_df = _load_json_as_df("season_projections_2025.json")
    final_standings_df = compute_final_standings()

    accuracy = compute_accuracy_metrics(games_df, fantasy_df, season_projections_df, final_standings_df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(accuracy, f, indent=2)
        record_generation("accuracy_tracker_2025")

    print(f"Generated accuracy tracker -> {OUTPUT_PATH}")
    print(f"Games: {accuracy['season_summary']['games']}")
    print(f"Fantasy: {accuracy['season_summary']['fantasy']}")
    print(f"Season projections: {accuracy['season_summary']['season_projections']}")
    print(f"Weekly breakdown: {len(accuracy['weekly_breakdown'])} weeks")
    return accuracy


if __name__ == "__main__":
    generate_accuracy_tracker_json()
