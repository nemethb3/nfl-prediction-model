"""Dashboard Section 3 data export: real season win projections + real
playoff odds, no fabrication.

Corrects real issues found in the pasted spec before building:

1. `current_standings_2025.csv` and `playoff_simulations_2025.csv` don't
   exist. Real playoff odds already exist in
   `data/diagnostic/playoff_odds_trajectory_2025.csv` (real Monte Carlo
   simulation output from playoff_probability.py's earlier work), but only
   at 5 checkpoint weeks (1, 4, 8, 12, 16) - there's no "current" week
   concept for this project's data, since it's a fully-completed 2025
   season backtest, not a live feed. Uses the week-16 checkpoint (closest
   to season-end, most confident real numbers) as the single snapshot this
   table shows. Real win/loss records are derived directly from
   schedules_2015_2025.csv (games with week < 16), the same source and
   convention the trajectory file's own `actual_wins_through_week` column
   uses (cross-checked: KC's real 6 wins through week<16 matches the file
   exactly).

2. `superbowl_percentage` doesn't exist anywhere in this project, real or
   otherwise - playoff_probability.py only ever modeled "makes the
   playoffs," never simulated a path through the bracket to a champion.
   Per explicit instruction, omitted entirely rather than fabricated -
   see DASHBOARD_DATA_GAPS.md.

3. The pasted compute_playoff_seed() has two real bugs: it uses a post-
   sort_values DataFrame's stale row-index as if it were a 0-based rank
   (`.index[0]` returns the ORIGINAL row label, not a rank), and despite
   its own docstring claiming "1-4: division winners, 5-7: wildcards," the
   code never actually checks division standings - it just flatly ranks
   the whole conference by wins. Real seeding logic implemented instead:
   the 4 real division leaders (by real wins-through-week-16, tiebroken by
   real point differential) get seeds 1-4 sorted by wins, the next-best 3
   non-division-winners in the conference get wildcard seeds 5-7.

4. `remaining_schedule_strength` (average Elo of remaining opponents) uses
   each opponent's real Elo rating as of the week-16 checkpoint
   (elo_ratings_2025.csv's real elo_after, week 15 - the same lagged,
   leak-free convention used everywhere else in this project) as a proxy
   for that opponent's strength in the remaining (week > 16) games - a
   disclosed simplification (their Elo could still move before those games
   are actually played), not a fabrication - the rating itself is real.

Division/conference mapping is real, standard NFL structure (unchanged
since the 2002 realignment plus the LV/WAS renames already used elsewhere
in this project) - public reference data, not a modeled quantity, same
category as team names/colors.
"""

import json
from generation_timestamps import record_generation
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "season_projections_2025.json")

SEASON = 2025
CHECKPOINT_WEEK = 16  # closest available real checkpoint to season-end (odds file has 1,4,8,12,16)

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SF", "SEA"],
}
TEAM_TO_DIVISION = {team: div for div, teams in DIVISIONS.items() for team in teams}
TEAM_TO_CONFERENCE = {team: div.split(" ")[0] for team, div in TEAM_TO_DIVISION.items()}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _real_records_through_week(checkpoint_week=CHECKPOINT_WEEK, season=SEASON):
    """Real wins/losses/ties/point-diff for every team, from real games
    with week < checkpoint_week (leak-free, matches the trajectory file's
    own actual_wins_through_week convention - cross-checked)."""
    sched = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    s = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & (sched["week"] < checkpoint_week)]
    s = s.dropna(subset=["home_score", "away_score"])

    records = {team: {"wins": 0, "losses": 0, "ties": 0, "point_diff": 0} for team in TEAM_TO_DIVISION}
    for _, g in s.iterrows():
        margin = g["home_score"] - g["away_score"]
        for team, is_home in [(g["home_team"], True), (g["away_team"], False)]:
            if team not in records:
                continue
            team_margin = margin if is_home else -margin
            records[team]["point_diff"] += team_margin
            if team_margin > 0:
                records[team]["wins"] += 1
            elif team_margin < 0:
                records[team]["losses"] += 1
            else:
                records[team]["ties"] += 1
    return records


def _real_remaining_schedule_strength(checkpoint_week=CHECKPOINT_WEEK, season=SEASON):
    """Real average Elo of each team's remaining (week >= checkpoint_week)
    opponents, using each opponent's real Elo as of the checkpoint (lagged
    one real week, same convention as generate_dashboard_data.py)."""
    sched = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    s = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & (sched["week"] >= checkpoint_week)]

    elo = pd.read_csv(os.path.join(PROCESSED_DIR, f"elo_ratings_{season}.csv"))
    elo_at_checkpoint = elo[elo["week"] == checkpoint_week - 1].set_index("team")["elo_after"].to_dict()

    opponent_elos = {team: [] for team in TEAM_TO_DIVISION}
    for _, g in s.iterrows():
        home, away = g["home_team"], g["away_team"]
        if home in opponent_elos and away in elo_at_checkpoint:
            opponent_elos[home].append(elo_at_checkpoint[away])
        if away in opponent_elos and home in elo_at_checkpoint:
            opponent_elos[away].append(elo_at_checkpoint[home])

    league_avg_elo = sum(elo_at_checkpoint.values()) / len(elo_at_checkpoint)
    return {team: (round(sum(v) / len(v)) if v else round(league_avg_elo)) for team, v in opponent_elos.items()}


def _compute_seeds(records, conference):
    """Real seeding: division leaders (seeds 1-4, sorted by real wins) then
    the next-best 3 non-division-winners by real wins (seeds 5-7).
    Tiebreak: real point differential (a real, simplified stand-in for the
    NFL's full multi-step tiebreaker procedure - disclosed, not hidden)."""
    conf_divisions = [d for d in DIVISIONS if d.startswith(conference)]
    division_leaders = []
    for div in conf_divisions:
        teams = DIVISIONS[div]
        leader = max(teams, key=lambda t: (records[t]["wins"], records[t]["point_diff"]))
        division_leaders.append(leader)
    division_leaders.sort(key=lambda t: (records[t]["wins"], records[t]["point_diff"]), reverse=True)

    conf_teams = [t for t in TEAM_TO_CONFERENCE if TEAM_TO_CONFERENCE[t] == conference]
    wildcard_pool = [t for t in conf_teams if t not in division_leaders]
    wildcard_pool.sort(key=lambda t: (records[t]["wins"], records[t]["point_diff"]), reverse=True)
    wildcards = wildcard_pool[:3]

    seeds = {}
    for i, team in enumerate(division_leaders):
        seeds[team] = i + 1
    for i, team in enumerate(wildcards):
        seeds[team] = i + 5
    return seeds


def generate_season_projections_json():
    traj = pd.read_csv(os.path.join(DIAGNOSTIC_DIR, "playoff_odds_trajectory_2025.csv"))
    checkpoint = traj[traj["week"] == CHECKPOINT_WEEK].set_index("team")

    records = _real_records_through_week()
    remaining_strength = _real_remaining_schedule_strength()
    seeds_afc = _compute_seeds(records, "AFC")
    seeds_nfc = _compute_seeds(records, "NFC")
    all_seeds = {**seeds_afc, **seeds_nfc}
    division_winners = {t for t, s in all_seeds.items() if s <= 4}

    rows = []
    for team in TEAM_TO_DIVISION:
        rec = records[team]
        row = {
            "team": team,
            "team_name": TEAM_NAMES[team],
            "conference": TEAM_TO_CONFERENCE[team],
            "division": TEAM_TO_DIVISION[team],
            "wins_actual": rec["wins"],
            "losses_actual": rec["losses"],
            "ties_actual": rec["ties"],
            "projected_wins": None,
            "playoff_percentage": None,
            "playoff_seed": all_seeds.get(team),
            "remaining_schedule_strength": remaining_strength.get(team),
            "is_division_winner": team in division_winners,
            "is_playoff_team": team in all_seeds,
        }
        if team in checkpoint.index:
            row["projected_wins"] = round(float(checkpoint.loc[team, "projected_final_wins"]), 1)
            row["playoff_percentage"] = round(float(checkpoint.loc[team, "playoff_odds_pct"]), 4)
        rows.append(row)

    rows.sort(key=lambda r: (r["conference"], r["division"], -r["wins_actual"]))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        record_generation("season_projections_2025")

    n_playoff = sum(1 for r in rows if r["is_playoff_team"])
    n_div_winners = sum(1 for r in rows if r["is_division_winner"])
    print(f"Generated {len(rows)} team projections (checkpoint: real week {CHECKPOINT_WEEK}) -> {OUTPUT_PATH}")
    print(f"Playoff teams: {n_playoff} (expect 14) | Division winners: {n_div_winners} (expect 8)")
    return rows


if __name__ == "__main__":
    generate_season_projections_json()
