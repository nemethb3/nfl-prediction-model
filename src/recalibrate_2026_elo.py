"""Real in-season O/D-Elo recalibration for 2026 - the piece with no
existing infrastructure at all (unlike single-Elo, which already had a
validated update mechanism - see elo_game_prediction.generate_elo_game_
spreads's season>2025 branch, weekly_recalibration.py).

Real, deliberate design (see the "In-Season Dynamic Recalibration" task
decision, 2026-09-16, for the fuller writeup of why a pasted spec's
generic win/loss K=20 formula was rejected in favor of this):

compute_offensive_defensive_elo.py's own chain has NO season-boundary
regression logic at all (that's bolted on separately, apply_season_
regression_od_elo.py, specifically for the 2025->2026 preseason
boundary) - and it hardcodes reading ONLY game_results_2015_2025.csv, so
it can't see real 2026 games even if asked. Rather than touch that
function (used by 2 other real callers - fit_od_elo_model,
backtest_offensive_defensive_elo.py - that must keep computing the real
2015-2025 historical chain exactly as before), this is its own short,
separate, additive 2026-only chain:

  1. Seed: the real, already-existing preseason regression
     (apply_season_regression_od_elo() - unchanged, still the real
     "entering 2026" baseline).
  2. Update: the EXACT real per-game O/D-Elo formula copied verbatim from
     compute_offensive_defensive_elo.py (same real opponent-adjusted
     expectation, same real league_avg_pts constant, same real k_factor -
     not re-derived or approximated), applied to real completed 2026
     games only (build_game_results_2026.py's output).

Returns per-game before/after history (so callers needing a specific
already-played game's real, leak-free pre-game rating can get it,
mirroring elo_game_prediction.py's single-Elo hybrid) AND the final
"current" rating per team (as of the latest real completed game) for
callers that just want current team strength (Power Rankings, Trade
Scores, the playoff/Super Bowl simulation).

Before any 2026 game completes, current == the pure preseason seed
(the update loop below is simply a no-op over zero rows) - verified
byte-for-byte, so this is a strict extension of the existing preseason
behavior, not a replacement of it.
"""

import json
import os

import pandas as pd

from apply_season_regression_od_elo import apply_season_regression_od_elo
from compute_offensive_defensive_elo import compute_offensive_defensive_elo
from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "elo_model_comparison.json")
GAMES_2026_PATH = os.path.join(BACKTEST_DIR, "game_results_2026.csv")
OD_ELO_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")


def _real_od_k_factor():
    with open(COMPARISON_PATH, encoding="utf-8") as f:
        return json.load(f)["od_k_factor_selected"]


def compute_od_elo_2026(k_factor=None):
    """Real (seed, current, history_df) for 2026 O/D-Elo:
    - seed: {team: {o_elo, d_elo, total_elo}} - the real preseason regression.
    - current: {team: {o_elo, d_elo, total_elo}} - as of the latest real
      completed game (== seed, unchanged, if none have completed yet).
    - history_df: one real row per completed 2026 game, with each team's
      real leak-free o_elo/d_elo BEFORE and AFTER that specific game -
      needed by generate_od_elo_game_spreads's per-game hybrid so an
      already-played game's own real pre-game rating never gets
      overwritten by a later week's information."""
    if k_factor is None:
        k_factor = _real_od_k_factor()

    seed = apply_season_regression_od_elo(k_factor=k_factor)
    _, _, league_avg_pts = compute_offensive_defensive_elo(k_factor=k_factor, save=False)

    o_elo = {t: v["o_elo"] for t, v in seed.items()}
    d_elo = {t: v["d_elo"] for t, v in seed.items()}

    rows = []
    if os.path.exists(GAMES_2026_PATH):
        games = pd.read_csv(GAMES_2026_PATH)
        games = games[games["game_type"] == "REG"].sort_values(["week", "game_id"]).reset_index(drop=True)
        for _, g in games.iterrows():
            week = int(g["week"])
            home, away = g["home_team"], g["away_team"]
            home_score, away_score = float(g["home_score"]), float(g["away_score"])

            home_o_before, home_d_before = o_elo[home], d_elo[home]
            away_o_before, away_d_before = o_elo[away], d_elo[away]

            # Real, opponent-adjusted expectation - identical formula to
            # compute_offensive_defensive_elo.py (reused verbatim).
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
                "game_id": g["game_id"], "week": week, "home_team": home, "away_team": away,
                "home_o_elo_before": round(home_o_before, 2), "home_d_elo_before": round(home_d_before, 2),
                "away_o_elo_before": round(away_o_before, 2), "away_d_elo_before": round(away_d_before, 2),
                "home_o_elo_after": round(home_o_after, 2), "home_d_elo_after": round(home_d_after, 2),
                "away_o_elo_after": round(away_o_after, 2), "away_d_elo_after": round(away_d_after, 2),
            })

    current = {t: {"o_elo": round(o_elo[t], 1), "d_elo": round(d_elo[t], 1),
                    "total_elo": round(o_elo[t] + d_elo[t], 1)} for t in seed}
    history_df = pd.DataFrame(rows)
    return seed, current, history_df


def refresh_current_od_elo_file(k_factor=None):
    """Overwrites team_elo_offensive_defensive_2026_regressed.json with
    the CURRENT (not frozen-preseason) O/D-Elo - the same file Power
    Rankings and Trade Scores already read directly, so they pick up real
    in-season movement with no changes on their end. A real, strict
    extension: identical content to the pure preseason file when no 2026
    game has completed yet."""
    _, current, _ = compute_od_elo_2026(k_factor)
    os.makedirs(os.path.dirname(OD_ELO_OUTPUT_PATH), exist_ok=True)
    with open(OD_ELO_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
        record_generation("team_elo_offensive_defensive_2026_regressed")
    print(f"Real current 2026 O/D-Elo -> {OD_ELO_OUTPUT_PATH}")
    if "KC" in current:
        print(f"  Example (KC): O_Elo {current['KC']['o_elo']}, D_Elo {current['KC']['d_elo']}")
    return current


if __name__ == "__main__":
    refresh_current_od_elo_file()
