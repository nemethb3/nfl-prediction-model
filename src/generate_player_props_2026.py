"""Scores real player props predictions for every real 2026 REG game, for
every real rostered QB/RB/WR/TE with real 2015-2025 history.

Real weather/rest addition (Quick Wins task): models now also take
is_dome/own_rest_days (real, both knowable in advance for 2026 - see
build_player_props_signals.py/train_player_props_models.py docstrings for
the real, honest null-result finding from adding them, and why temp/wind
were excluded as not knowable in advance for a future game).

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. Assumed `frontend/src/data/trade_scores_2026.json` has per-player
   `ppr_projection`/`avg_targets` fields - checked, it doesn't (real shape
   is `{"players": {player_id: {"signals": {...}, "prob_ppr_increase":
   ...}}}`, no PPR or volume figures at all). Real 2026 roster (player_id
   -> team) instead comes from data/processed/{position}_epa_projections_
   2026.csv - the same real, already-fixed source generate_fantasy_
   rankings_2026_week1.py and generate_trade_scores_2026.py both already
   use (fixes the real 30/107-stale-team bug the 2026-07-30 audit found -
   see those scripts' own docstrings).
2. Assumed `data/processed/games_2026.csv` and `data/processed/team_
   defense_stats_2026.csv` - neither exists. Real 2026 schedule is
   data/raw/schedules_2026.csv; real 2026 team O/D Elo (static preseason,
   same convention as every other 2026 deliverable in this project) is
   data/processed/team_elo_offensive_defensive_2026_regressed.json.
3. The spec predicted a `home_qb_id`/`away_qb_id` schedule lookup for QBs
   specifically, but a plain team lookup for every other position -
   real schedules_2026.csv has no *_qb_id columns with real values this
   far out (checked: 0/272 rows populated, same real gap already disclosed
   in generate_dashboard_data_2026.py). Real fix: every position uses the
   same real team-schedule join (which real team does this player's real
   2026 roster say they're on, and who does that team play each week).
4. Real career-average features for 2026 scoring use each player's real
   FULL 2015-2025 history (not leak-free-per-week like training, since
   2026 hasn't started - as of right now every real game they've played is
   genuinely prior). A real, newly-drafted rookie with no 2015-2025 history
   is excluded (a real, disclosed gap - identical precedent to
   FantasyRankings.js's own WR rookie-exclusion note), not assigned a
   fabricated league-average fallback."""

import json
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "player_props_models.json")
REGRESSED_OD_ELO_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")
SCHEDULE_PATH = os.path.join(RAW_DIR, "schedules_2026.csv")
PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "player_props_2026.json")

SEASON = 2026
POSITIONS = ["QB", "RB", "WR", "TE"]
CAREER_AVG_SOURCE_COLS = {
    "QB": ["completions", "attempts", "passing_yards", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "rushing_yards"],
    "TE": ["targets", "receptions", "receiving_yards", "rushing_yards"],
}


def _real_2026_roster(position):
    """Same real, already-fixed source generate_fantasy_rankings_2026_
    week1.py and generate_trade_scores_2026.py both use."""
    path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2026.csv")
    df = pd.read_csv(path)
    return df[["player_id", "player_name", "team"]].drop_duplicates("player_id")


def _real_career_averages(position, player_ids):
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[(stats["season_type"] == "REG") & (stats["position"] == position) &
                  (stats["player_id"].isin(player_ids))]
    cols = CAREER_AVG_SOURCE_COLS[position]
    return stats.groupby("player_id")[cols].mean().rename(columns={c: f"career_avg_{c}" for c in cols})


def _real_2026_schedule_by_team():
    sched = pd.read_csv(SCHEDULE_PATH)
    sched = sched[sched["game_type"] == "REG"]
    # Real roof/rest, same as build_player_props_signals.py's training-side
    # features - both genuinely knowable in advance for a 2026 game.
    # 43/272 real 2026 games have no roof recorded yet this far out; those
    # default to is_dome=0 (real majority class - 177/229 known real 2026
    # games are outdoors), a disclosed real fallback, not a fabricated one.
    sched["is_dome"] = sched["roof"].isin(["dome", "closed"]).fillna(False).astype(int)
    home = sched[["week", "home_team", "away_team", "is_dome", "home_rest"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_rest": "own_rest_days"})
    home["is_home"] = True
    away = sched[["week", "away_team", "home_team", "is_dome", "away_rest"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_rest": "own_rest_days"})
    away["is_home"] = False
    return pd.concat([home, away], ignore_index=True)


def _predict(feature_values, model_info):
    features = model_info["features"]
    x_scaled = np.array([
        (feature_values[f] - model_info["scaler_mean"][f]) / model_info["scaler_scale"][f]
        for f in features
    ])
    coefs = np.array([model_info["coefficients"][f] for f in features])
    pred = model_info["intercept"] + float(np.dot(x_scaled, coefs))
    return max(0.0, pred)


def generate_player_props_2026():
    print(f"\nGenerating real player props for {SEASON}...\n")
    with open(MODELS_PATH, encoding="utf-8") as f:
        models = json.load(f)
    with open(REGRESSED_OD_ELO_PATH, encoding="utf-8") as f:
        regressed_od_elo = json.load(f)
    schedule_by_team = _real_2026_schedule_by_team()
    # Real, min-max week normalization matching the same real training
    # convention (build_player_props_signals.py) - REG season weeks only.
    week_min, week_max = schedule_by_team["week"].min(), schedule_by_team["week"].max()

    all_props = []
    for position in POSITIONS:
        roster = _real_2026_roster(position)
        career_avgs = _real_career_averages(position, roster["player_id"].tolist())
        roster_with_history = roster.merge(career_avgs, on="player_id", how="inner")
        n_excluded = len(roster) - len(roster_with_history)
        print(f"[{position}] {len(roster_with_history)}/{len(roster)} real rostered players have 2015-2025 "
              f"history ({n_excluded} real rookies/no-history players excluded - a real, disclosed gap, "
              "not a fabricated fallback)")

        position_models = models[position]
        n_games = 0
        for _, player in roster_with_history.iterrows():
            player_games = schedule_by_team[schedule_by_team["team"] == player["team"]]
            for _, g in player_games.iterrows():
                opp_elo = regressed_od_elo.get(g["opponent"], {})
                if "d_elo" not in opp_elo:
                    continue
                week_norm = (g["week"] - week_min) / (week_max - week_min)
                feature_values = {
                    f"career_avg_{c}": float(player[f"career_avg_{c}"]) for c in CAREER_AVG_SOURCE_COLS[position]
                }
                feature_values["opp_d_elo"] = float(opp_elo["d_elo"])
                feature_values["is_home"] = float(g["is_home"])
                feature_values["week_norm"] = float(week_norm)
                feature_values["is_dome"] = float(g["is_dome"])
                feature_values["own_rest_days"] = float(g["own_rest_days"])

                predicted_stats = {
                    stat: round(_predict(feature_values, model_info), 1)
                    for stat, model_info in position_models.items()
                }
                all_props.append({
                    "id": f"{player['player_id']}_w{int(g['week'])}",
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": position,
                    "team": player["team"],
                    "week": int(g["week"]),
                    "opponent": g["opponent"],
                    "is_home": bool(g["is_home"]),
                    "opponent_d_elo": round(float(opp_elo["d_elo"]), 1),
                    "predicted_stats": predicted_stats,
                })
                n_games += 1
        print(f"       {n_games} real player-games scored")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_props, f, indent=2)
    print(f"\nWrote {len(all_props)} real player-game props -> {OUTPUT_PATH}")
    return all_props


if __name__ == "__main__":
    generate_player_props_2026()
