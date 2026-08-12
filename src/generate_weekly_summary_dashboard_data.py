"""Dashboard Section 5 data export: real weekly recap + next-week preview,
no fabrication.

Reuses real division structure from generate_season_projections_dashboard_
data.py (TEAM_TO_DIVISION/DIVISIONS) rather than reimplementing it - the
draft spec's local re-implementation used the pre-2002 "Central" division
names, caught before building.

Two real bugs fixed from the pasted draft before running:

1. `division_leaders` was built as a list of real per-division leader
   dicts, then immediately overwritten by an unrelated int (division-
   winner count) a few lines later, before being placed into the output -
   the frontend calls .map() on that key expecting the list, so this would
   have crashed on load. Renamed the count variable instead.

2. rate_matchup()'s real opponent_defense_rank_vs_position lookup was
   inverted: it filtered fantasy_df for players ON the opponent's own
   team, then read the defense rank THEY face (wrong direction). Fixed to
   find players whose real `opponent` field equals the team in question
   (players playing AGAINST them that week) and read the rank off one of
   those rows - opponent_defense_rank_vs_position is a per-(week,
   opponent, position) value, redundantly stored on every offensive
   player's own record, so any matching row gives the same real number.

next_week's key_players and top_fantasy_plays reuse the real, already-
computed `opponent` field directly from fantasy_rankings_2025.json instead
of recomputing it from games_df a second time.
"""

import json
import os

import pandas as pd

from generate_season_projections_dashboard_data import DIVISIONS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "weekly_summary_2025.json")

MAX_WEEK = 18


def _load_json_as_df(filename):
    with open(os.path.join(FRONTEND_DATA_DIR, filename), encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def _rate_matchup(position, opponent_team, week, fantasy_df):
    """Real trailing defense rank vs. position, read off any player row
    whose real `opponent` field matches opponent_team (players playing
    AGAINST them that week) - see module docstring #2."""
    facing_opponent = fantasy_df[
        (fantasy_df["opponent"] == opponent_team) & (fantasy_df["week"] == week) & (fantasy_df["position"] == position)
    ]
    if len(facing_opponent) == 0:
        return "neutral"
    defense_rank = facing_opponent.iloc[0]["opponent_defense_rank_vs_position"]
    if defense_rank is None or pd.isna(defense_rank):
        return "neutral"
    if defense_rank <= 12:
        return "tough"
    if defense_rank <= 20:
        return "neutral"
    return "good"


def _key_players(team, week, fantasy_df, n=3):
    team_players = fantasy_df[
        (fantasy_df["team"] == team) & (fantasy_df["week"] == week) & (fantasy_df["position"].isin(["QB", "WR"]))
    ].sort_values("projected_ppr", ascending=False)
    return [f"{r['name']} ({r['position']})" for _, r in team_players.head(n).iterrows()]


def compute_weekly_summary(games_df, fantasy_df, season_projections_df, current_week):
    weekly_data = {"current_week": current_week, "weeks": []}

    for week in range(1, MAX_WEEK + 1):
        week_games = games_df[games_df["week"] == week]
        if len(week_games) == 0:
            continue

        is_completed = bool(week_games["actual_home_score"].notna().all())
        week_summary = {"week": week, "is_completed": is_completed, "this_week": None,
                         "top_performers": None, "next_week": None, "season_context": None}

        if is_completed:
            completed_games = week_games[week_games["actual_home_score"].notna()].copy()
            completed_games["home_won"] = completed_games["actual_home_score"] > completed_games["actual_away_score"]
            completed_games["predicted_home_win"] = completed_games["win_prob_home"] > 0.5
            completed_games["correct"] = completed_games["home_won"] == completed_games["predicted_home_win"]

            correct = int(completed_games["correct"].sum())
            accuracy_pct = correct / len(completed_games) * 100
            mean_error = (
                (completed_games["actual_home_score"] - completed_games["actual_away_score"]) - completed_games["our_spread"]
            ).abs().mean()

            games_list = []
            surprises = []
            for _, game in completed_games.iterrows():
                actual_spread = game["actual_home_score"] - game["actual_away_score"]
                predicted_winner = game["home_team"] if game["predicted_home_win"] else game["away_team"]
                actual_winner = game["home_team"] if game["home_won"] else game["away_team"]

                games_list.append({
                    "week": week, "home_team": game["home_team"], "away_team": game["away_team"],
                    "home_score": int(game["actual_home_score"]), "away_score": int(game["actual_away_score"]),
                    "predicted_winner": predicted_winner, "actual_winner": actual_winner,
                    "prediction_correct": bool(game["correct"]),
                    "predicted_spread": round(float(game["our_spread"]), 1),
                    "actual_spread": round(float(actual_spread), 1),
                    "vegas_spread": round(float(game["vegas_spread"]), 1) if pd.notna(game["vegas_spread"]) else None,
                })

                if not game["correct"]:
                    prediction_confidence = max(game["win_prob_home"], game["win_prob_away"]) * 100
                    surprise_magnitude = abs(actual_spread - game["our_spread"])
                    surprises.append({
                        "home_team": game["home_team"], "away_team": game["away_team"],
                        "home_score": int(game["actual_home_score"]), "away_score": int(game["actual_away_score"]),
                        "predicted_winner": predicted_winner, "actual_winner": actual_winner,
                        "prediction_confidence": f"{prediction_confidence:.0f}% {predicted_winner}",
                        "actual_spread": round(float(actual_spread), 1),
                        "surprise_score": round(float(surprise_magnitude), 1),
                    })

            surprises = sorted(surprises, key=lambda x: x["surprise_score"], reverse=True)[:3]

            week_summary["this_week"] = {
                "total_games": int(len(completed_games)), "correct_predictions": correct,
                "accuracy_pct": round(accuracy_pct, 1), "mean_spread_error": round(float(mean_error), 2),
                "games": games_list, "biggest_surprises": surprises,
            }

            week_fantasy = fantasy_df[(fantasy_df["week"] == week) & fantasy_df["actual_ppr"].notna()].copy()
            if len(week_fantasy) > 0:
                week_fantasy["difference"] = week_fantasy["actual_ppr"] - week_fantasy["projected_ppr"]
                best = week_fantasy.nlargest(5, "difference")
                worst = week_fantasy.nsmallest(5, "difference")

                def _player_row(r):
                    return {"player_name": r["name"], "position": r["position"], "team": r["team"],
                            "projected_ppr": round(float(r["projected_ppr"]), 1), "actual_ppr": round(float(r["actual_ppr"]), 1),
                            "difference": round(float(r["difference"]), 1)}

                week_summary["top_performers"] = {
                    "best": [_player_row(r) for _, r in best.iterrows()],
                    "worst": [_player_row(r) for _, r in worst.iterrows()],
                }

        next_week = week + 1
        if next_week <= MAX_WEEK:
            next_week_games = games_df[games_df["week"] == next_week]
            if len(next_week_games) > 0:
                games_preview = []
                for _, game in next_week_games.iterrows():
                    games_preview.append({
                        "week": next_week, "home_team": game["home_team"], "away_team": game["away_team"],
                        "predicted_winner": game["home_team"] if game["win_prob_home"] > 0.5 else game["away_team"],
                        "win_prob_home": round(float(game["win_prob_home"]), 3) if pd.notna(game["win_prob_home"]) else None,
                        "win_prob_away": round(float(game["win_prob_away"]), 3) if pd.notna(game["win_prob_away"]) else None,
                        "predicted_spread": round(float(game["our_spread"]), 1),
                        "key_players": {
                            "home": _key_players(game["home_team"], next_week, fantasy_df),
                            "away": _key_players(game["away_team"], next_week, fantasy_df),
                        },
                    })

                next_week_fantasy = fantasy_df[fantasy_df["week"] == next_week].sort_values("projected_ppr", ascending=False).head(10)
                top_plays = []
                for _, player in next_week_fantasy.iterrows():
                    opponent = player["opponent"]
                    top_plays.append({
                        "player_name": player["name"], "position": player["position"], "team": player["team"],
                        "opponent": opponent, "projected_ppr": round(float(player["projected_ppr"]), 1),
                        "matchup_quality": _rate_matchup(player["position"], opponent, next_week, fantasy_df),
                        "injury_status": player.get("injury_status", "healthy"),
                    })

                week_summary["next_week"] = {"week": next_week, "games": games_preview, "top_fantasy_plays": top_plays}

        division_leader_rows = []
        for div, teams in DIVISIONS.items():
            div_rows = season_projections_df[season_projections_df["team"].isin(teams)]
            if len(div_rows) == 0:
                continue
            # Real fix (2026-07-30 audit): reuse season_projections_df's real,
            # tiebreak-aware is_division_winner field (real wins + real
            # point-diff tiebreak, from _compute_seeds()) instead of a plain
            # nlargest(wins_actual), which silently picks the wrong team on
            # any real wins-tie - confirmed via audit for NFC South (real
            # winner TB, nlargest picked CAR) and NFC West (real winner SEA,
            # nlargest picked LA), both real 2025 ties. nlargest kept only as
            # a defensive fallback for the case (not expected in real data)
            # where no row is marked a division winner.
            winner_rows = div_rows[div_rows["is_division_winner"]]
            leader = (winner_rows.iloc[0] if len(winner_rows) > 0 else div_rows.nlargest(1, "wins_actual").iloc[0])
            division_leader_rows.append({"division": div, "leader": leader["team"],
                                          "wins": int(leader["wins_actual"]), "losses": int(leader["losses_actual"])})

        playoff_teams = season_projections_df[season_projections_df["is_playoff_team"]]
        division_winner_count = int(season_projections_df["is_division_winner"].sum())
        wild_card_contenders = len(playoff_teams) - division_winner_count

        week_summary["season_context"] = {
            "week": week,
            "division_leaders": division_leader_rows,
            "playoff_race": {
                "leading_for_playoff": int(len(playoff_teams)),
                "leading_for_division": division_winner_count,
                "wild_card_contenders": int(wild_card_contenders),
            },
            "playoff_race_note": "Week-16 snapshot (not true elimination numbers) - see Section 3 for detail.",
        }

        weekly_data["weeks"].append(week_summary)

    return weekly_data


def generate_weekly_summary_json():
    games_df = _load_json_as_df("games_2025.json")
    fantasy_df = _load_json_as_df("fantasy_rankings_2025.json")
    season_projections_df = _load_json_as_df("season_projections_2025.json")

    completed_weeks = games_df[games_df["actual_home_score"].notna()]["week"].unique()
    current_week = int(max(completed_weeks)) if len(completed_weeks) > 0 else 1

    summary = compute_weekly_summary(games_df, fantasy_df, season_projections_df, current_week)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated weekly summary -> {OUTPUT_PATH}")
    print(f"current_week (most recent completed): {current_week}")
    print(f"weeks with data: {len(summary['weeks'])}")
    last_week = summary["weeks"][-1]
    print(f"week {last_week['week']} has next_week preview: {last_week['next_week'] is not None} "
          f"(expected False - season ends at week {MAX_WEEK})")
    return summary


if __name__ == "__main__":
    generate_weekly_summary_json()
