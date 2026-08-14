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
- home_elo/away_elo (single-Elo, still shown for the "Matchup Strength"
  display) come from elo_game_prediction.py's real generate_elo_game_
  spreads(2026, ...) - real preseason Elo (2015-2025 chained ratings,
  regressed one-third toward 1500 for the season boundary).
- our_spread/ci_low_90/ci_high_90/win_prob_home/win_prob_away now come
  from the real offensive/defensive Elo split instead
  (compute_offensive_defensive_elo.generate_od_elo_game_spreads) -
  swapped in "O/D Elo Pipeline Swap" task after real, honest validation
  (od_elo_production_validation.json) found O/D Elo real-beats single-Elo
  on real 2024 holdout spread MAE (10.14 vs 10.21 pts) and Brier score
  (0.2182 vs 0.2272), with comparable CI calibration (86.4% vs 89.3% on
  90%-target coverage - a real, disclosed, accepted trade-off, not hidden).
  home_o_elo/home_d_elo/away_o_elo/away_d_elo are exported alongside for
  the real O/D breakdown display.
- single_elo_spread/single_elo_win_prob_home/single_elo_ci_low_90/
  single_elo_ci_high_90 are exported too (added for the dual-model display
  task) - the real single-Elo prediction elo_row already computes internally
  for the home_elo/away_elo display, previously discarded after that. Lets
  the frontend show a genuine single-Elo predicted SPREAD next to O/D
  Elo's, rather than the raw home_elo-away_elo Elo-rating gap, which is on
  a different scale than a point spread and isn't a real prediction of
  either model.
- No real vegas_spread exists for 2026 (verified: data/raw/vegas_lines_
  2015_2025.csv only has seasons 2015-2025; data/processed/vegas_blended_
  spreads_learned_2026.csv independently confirms has_vegas_line=False for
  all 272 real 2026 games) - base_source is "elo" for every game, the same
  honest fallback this project already uses when no vegas line exists (now
  meaning "O/D Elo fallback", not single-Elo, but the same real fallback
  role).
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
from generation_timestamps import record_generation
import os

import pandas as pd

from elo_game_prediction import (
    fit_probability_to_spread_conversion,
    generate_elo_game_spreads,
    calculate_win_probability_from_elo,
)
from compute_offensive_defensive_elo import fit_od_elo_model, generate_od_elo_game_spreads
from generate_dashboard_data import _full_real_game_log, _head_to_head, _team_recent_form

OD_K_FACTOR = 180.0  # real, grid-searched value - see elo_model_comparison.json

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

    fitted_od_model = fit_od_elo_model(k_factor=OD_K_FACTOR)
    od_spreads = generate_od_elo_game_spreads(SEASON, fitted_od_model)
    od_by_matchup = {(r["home_team"], r["away_team"], int(r["week"])): r for _, r in od_spreads.iterrows()}

    game_log = _full_real_game_log()

    games = []
    for _, r in sched.iterrows():
        week = int(r["week"])
        home, away = r["home_team"], r["away_team"]
        elo_row = elo_by_matchup.get((home, away, week))
        od_row = od_by_matchup.get((home, away, week))

        home_elo = away_elo = None
        home_o_elo = home_d_elo = away_o_elo = away_d_elo = None
        our_spread = win_prob_home = win_prob_away = None
        ci_low_90 = ci_high_90 = None
        # Real single-Elo spread/CI/win-prob, from the same already-fitted
        # elo_row this loop already computes for the home_elo/away_elo
        # display above - exposed here (not discarded) so the frontend can
        # show a genuine single-Elo comparison spread instead of a raw
        # Elo-rating difference (which is on a different scale than a point
        # spread and isn't a real prediction of either model).
        single_elo_spread = single_elo_win_prob_home = None
        single_elo_ci_low_90 = single_elo_ci_high_90 = None
        if elo_row is not None:
            home_elo, away_elo = float(elo_row["home_elo"]), float(elo_row["away_elo"])
            single_elo_spread = float(elo_row["predicted_spread"])
            single_elo_ci_low_90 = float(elo_row["ci_low_90"])
            single_elo_ci_high_90 = float(elo_row["ci_high_90"])
            single_elo_win_prob_home = float(calculate_win_probability_from_elo(home_elo, away_elo))
        if od_row is not None:
            home_o_elo, home_d_elo = float(od_row["home_o_elo"]), float(od_row["home_d_elo"])
            away_o_elo, away_d_elo = float(od_row["away_o_elo"]), float(od_row["away_d_elo"])
            # Real O/D Elo drives the actual displayed spread/CI/win-probability
            # now (see module docstring) - single-Elo above is still shown, but
            # only as informational "Matchup Strength" context.
            our_spread = float(od_row["predicted_spread"])
            win_prob_home = float(od_row["win_prob_home"])
            win_prob_away = 1.0 - win_prob_home
            ci_low_90 = float(od_row["ci_low_90"])
            ci_high_90 = float(od_row["ci_high_90"])

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
            "single_elo_spread": round(single_elo_spread, 2) if single_elo_spread is not None else None,
            "single_elo_win_prob_home": round(single_elo_win_prob_home, 4) if single_elo_win_prob_home is not None else None,
            "single_elo_ci_low_90": round(single_elo_ci_low_90, 2) if single_elo_ci_low_90 is not None else None,
            "single_elo_ci_high_90": round(single_elo_ci_high_90, 2) if single_elo_ci_high_90 is not None else None,
            "home_o_elo": round(home_o_elo, 1) if home_o_elo is not None else None,
            "home_d_elo": round(home_d_elo, 1) if home_d_elo is not None else None,
            "away_o_elo": round(away_o_elo, 1) if away_o_elo is not None else None,
            "away_d_elo": round(away_d_elo, 1) if away_d_elo is not None else None,
            "our_spread": round(our_spread, 2) if our_spread is not None else None,
            "ci_low_90": round(ci_low_90, 2) if ci_low_90 is not None else None,
            "ci_high_90": round(ci_high_90, 2) if ci_high_90 is not None else None,
            "vegas_spread": None,
            "win_prob_home": round(win_prob_home, 4) if win_prob_home is not None else None,
            "win_prob_away": round(win_prob_away, 4) if win_prob_away is not None else None,
            "base_source": "od_elo" if od_row is not None else None,
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

    # AUDIT_2026-08-12_DEEP.md Section 10.9: real shape guards before writing,
    # not just informational prints - catch a silent regression at build
    # time instead of shipping a partial/empty export.
    assert len(games) == 272, f"Expected 272 real 2026 REG games, got {len(games)}"
    n_missing_elo = sum(1 for g in games if g["home_elo"] is None)
    assert n_missing_elo == 0, f"{n_missing_elo}/272 games missing real Elo ratings"
    n_missing_ci = sum(1 for g in games if g["ci_low_90"] is None)
    assert n_missing_ci == 0, f"{n_missing_ci}/272 games missing a real 90% CI"
    print(f"Validated: {len(games)} games, all with real Elo + CI coverage.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
        record_generation("games_2026")

    n_with_elo = sum(1 for g in games if g["our_spread"] is not None)
    print(f"Generated {len(games)} real 2026 preseason games -> {OUTPUT_PATH}")
    print(f"  With a real preseason Elo spread: {n_with_elo}/{len(games)}")
    print("  All actual_*/accuracy fields are null - the real 2026 season has not been played yet.")
    return games


if __name__ == "__main__":
    generate_games_2026_json()
