"""Dashboard Section 1 data export for the 2026 season - real preseason
predictions only, no fabricated results.

Real, verified fact before writing this: the 2026 NFL season has not been
played yet (real schedule in data/raw/schedules_2026.csv runs 2026-09-09 to
2027-01-10; zero games have a real score as of this run). So unlike
generate_dashboard_data.py (2025, a fully-completed season), this script
can only ever populate real PRE-GAME fields - every actual_*/accuracy field
is genuinely null, not a bug.

Reuses this project's already-built, already-validated 2026 preseason
infrastructure rather than inventing anything new:
- our_spread/home_elo/away_elo come from elo_game_prediction.py's real
  generate_elo_game_spreads(2026, ...) - real preseason Elo (2015-2025
  chained ratings, regressed one-third toward 1500 for the season
  boundary), already the documented convention for season > 2025 in that
  function, not something invented for this task.
- No real vegas_spread exists for 2026 (verified: data/raw/vegas_lines_
  2015_2025.csv only has seasons 2015-2025; data/processed/vegas_blended_
  spreads_learned_2026.csv independently confirms has_vegas_line=False for
  all 272 real 2026 games) - base_source is "elo" for every game, the same
  honest fallback this project already uses when no vegas line exists.
- win_prob_home/away use this project's real calculate_win_probability_
  from_elo() directly on the real preseason Elo ratings above - NOT the
  2025 script's Vegas-fit win_probability_backtest.py model, since that
  model was validated specifically on real vegas_spread, which doesn't
  exist here. Elo-based win probability is this project's own real,
  already-backtested second-best candidate (Brier 0.2874), not a new,
  unvalidated formula.
- net_edge_diff/matchup_quality are left null: that adjustment needs real
  in-season EPA stats which don't exist before Week 1 is played - a real,
  disclosed gap, not fabricated with a preseason stand-in.
- home_recent_form/away_recent_form/head_to_head reuse generate_dashboard_
  data.py's real, season-agnostic helpers unchanged - they only need real
  PRIOR games (2015-2025), which fully exist regardless of 2026's own
  games not having happened yet.
- home_qb_name/away_qb_name are null: schedules_2026.csv genuinely has no
  real starter data populated yet this far ahead of the season (verified:
  0/272 rows).
"""

import json
import os

import pandas as pd

from elo_game_prediction import calculate_win_probability_from_elo, fit_probability_to_spread_conversion, \
    generate_elo_game_spreads
from generate_dashboard_data import _full_real_game_log, _head_to_head, _team_recent_form

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "games_2026.json")

SEASON = 2026


def generate_games_2026_json():
    sched = pd.read_csv(os.path.join(RAW_DIR, "schedules_2026.csv"))
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched_full = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    sched_full["gameday"] = pd.to_datetime(sched_full["gameday"])

    fitted_model = fit_probability_to_spread_conversion()
    elo_spreads = generate_elo_game_spreads(SEASON, fitted_model)
    elo_by_matchup = {(r["home_team"], r["away_team"], int(r["week"])): r for _, r in elo_spreads.iterrows()}

    game_log = _full_real_game_log()

    games = []
    for _, r in sched.iterrows():
        week = int(r["week"])
        home, away = r["home_team"], r["away_team"]
        elo_row = elo_by_matchup.get((home, away, week))

        home_elo = away_elo = our_spread = win_prob_home = win_prob_away = None
        if elo_row is not None:
            home_elo, away_elo = float(elo_row["home_elo"]), float(elo_row["away_elo"])
            our_spread = float(elo_row["predicted_spread"])
            win_prob_home = float(calculate_win_probability_from_elo(home_elo, away_elo))
            win_prob_away = 1.0 - win_prob_home

        kickoff_datetime = None
        if pd.notna(r["gameday"]) and pd.notna(r["gametime"]):
            kickoff_datetime = f"{r['gameday'].date()}T{r['gametime']}:00"

        game_date = r["gameday"]
        home_form = _team_recent_form(home, game_date, game_log) if pd.notna(game_date) else []
        away_form = _team_recent_form(away, game_date, game_log) if pd.notna(game_date) else []
        h2h = _head_to_head(home, away, game_date, sched_full) if pd.notna(game_date) else None

        games.append({
            "id": f"{SEASON}_{week:02d}_{away}_{home}",
            "week": week,
            "weekday": r["weekday"] if pd.notna(r["weekday"]) else None,
            "kickoff_datetime": kickoff_datetime,
            "home_team": home,
            "away_team": away,
            "home_qb_name": None,
            "away_qb_name": None,
            "home_elo": round(home_elo, 1) if home_elo is not None else None,
            "away_elo": round(away_elo, 1) if away_elo is not None else None,
            "our_spread": round(our_spread, 2) if our_spread is not None else None,
            "vegas_spread": None,
            "win_prob_home": round(win_prob_home, 4) if win_prob_home is not None else None,
            "win_prob_away": round(win_prob_away, 4) if win_prob_away is not None else None,
            "base_source": "elo" if elo_row is not None else None,
            "net_edge_diff": None,
            "matchup_quality": None,
            "home_recent_form": home_form,
            "away_recent_form": away_form,
            "head_to_head": h2h,
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_winner": None,
            "actual_spread_margin": None,
            "did_we_predict_correctly": None,
        })

    games.sort(key=lambda g: (g["week"], g["kickoff_datetime"] or ""))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)

    n_with_elo = sum(1 for g in games if g["our_spread"] is not None)
    print(f"Generated {len(games)} real 2026 preseason games -> {OUTPUT_PATH}")
    print(f"  With a real preseason Elo spread: {n_with_elo}/{len(games)}")
    print("  All actual_*/accuracy fields are null - the real 2026 season has not been played yet.")
    return games


if __name__ == "__main__":
    generate_games_2026_json()
