"""Applies the real, trained point-totals regression to all 272 real 2026
games. Real, honest finding from training (point_totals_model.json):
Elo/week/season carry essentially zero real signal for predicting combined
score (OOF R^2=0.005, directional accuracy flat at ~50% across every real
edge threshold tested, 0.5-5.0 points) - there is no real threshold where
confidence is actually elevated. Consistent with that real finding, this
script does NOT produce a `predicted_total_alert` flag - firing one would
fabricate confidence the real backtest doesn't support. It exports the raw
`predicted_total_value` as weak, informational context only.

Real Vegas totals only exist for real 2026 weeks 1-4 (53/272 games,
verified via data/raw/schedules_2026.csv's real total_line column - lines
post progressively as each week approaches, same real pattern already
documented for spread_line in vegas_integration_optimized.py). Games
without a real posted total still get a real predicted_total_value (the
model needs no Vegas input), but vegas_total/predicted_total_diff/
predicted_total_direction are left null for them - no fabrication."""

import json
import os

import numpy as np
import pandas as pd

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "point_totals_model.json")
GAMES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "games_2026.json")

CURRENT_SEASON = 2026


def _real_2026_vegas_totals():
    schedule = pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    schedule = schedule[schedule["total_line"].notna()]
    return schedule.set_index("game_id")["total_line"].to_dict()


def apply_point_totals_2026():
    print(f"\nApplying real point-totals model to {CURRENT_SEASON} games...\n")
    with open(MODEL_PATH, encoding="utf-8") as f:
        model_info = json.load(f)
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)

    week_min, week_max = model_info["week_range"]
    season_min, season_max = model_info["season_range"]
    coefs, mean, scale = model_info["coefficients"], model_info["scaler_mean"], model_info["scaler_scale"]
    vegas_totals = _real_2026_vegas_totals()

    n_scored, n_with_vegas = 0, 0
    for game in games:
        home_elo, away_elo = game.get("home_elo"), game.get("away_elo")
        if home_elo is None or away_elo is None:
            game["predicted_total_value"] = None
            game["vegas_total"] = None
            game["predicted_total_diff"] = None
            game["predicted_total_direction"] = None
            continue

        raw_features = {
            "elo_sum": home_elo + away_elo,
            "elo_diff": abs(home_elo - away_elo),
            "week_norm": (game["week"] - week_min) / (week_max - week_min),
            "season_norm": (CURRENT_SEASON - season_min) / (season_max - season_min),
        }
        z = model_info["intercept"]
        for feature, value in raw_features.items():
            z += coefs[feature] * ((value - mean[feature]) / scale[feature])
        predicted_total = round(float(z), 1)
        game["predicted_total_value"] = predicted_total
        n_scored += 1

        vegas_total = vegas_totals.get(game["id"])
        if vegas_total is not None:
            game["vegas_total"] = float(vegas_total)
            game["predicted_total_diff"] = round(predicted_total - vegas_total, 1)
            game["predicted_total_direction"] = "OVER" if predicted_total > vegas_total else "UNDER"
            n_with_vegas += 1
        else:
            game["vegas_total"] = None
            game["predicted_total_diff"] = None
            game["predicted_total_direction"] = None

    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
        record_generation("games_2026_point_totals")

    print(f"Real predicted_total_value computed for {n_scored}/{len(games)} games "
          f"(remaining {len(games) - n_scored} lack an Elo rating)")
    print(f"Real vegas_total attached for {n_with_vegas}/{len(games)} games "
          f"(only real 2026 weeks with a posted total_line so far)")
    print(f"No predicted_total_alert field emitted - the trained model found no real edge-size "
          f"threshold with elevated directional accuracy (see point_totals_model.json)")
    print(f"Wrote {GAMES_PATH}")
    return games


if __name__ == "__main__":
    apply_point_totals_2026()
