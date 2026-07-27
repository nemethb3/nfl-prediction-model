"""Phase 1 Component 1.3: Playoff Probability Calculator.

Corrects 2 issues found in the spec before building:

1. "8 of 16 teams" doesn't match the real NFL playoff structure. The real
   format (since 2020) is 7 teams per 16-team CONFERENCE (14 total across
   the league), not a pooled top-8 of either 16 or 32. TEAM_CONFERENCE below
   uses the real division/conference structure (stable since 2002, safe for
   this project's 2015-2026 data) with this dataset's REAL team codes,
   verified directly (the Rams are "LA" in this data, not "LAR" - checked
   before hardcoding the mapping to avoid a silent join failure).

2. "Simulate N times (draw from binomial)" per team independently ignores a
   real correlation: two teams that play EACH OTHER in the remaining
   schedule share a single coin flip within a simulation trial - one team's
   win is the other's loss in that same trial, not two independent draws.
   monte_carlo_playoff_odds() simulates every real remaining GAME across the
   whole league per trial instead, so shared-opponent correlation (which
   matters most for close divisional/conference races) is preserved.

Reuses weekly_recalibration.py's already-built "freeze Elo at week N,
get real actual wins + real remaining schedule" mechanism (Component B)
rather than rebuilding it - this component's genuinely new contribution is
the Monte Carlo playoff layer on top of that, not re-deriving win
projections from scratch.

Real, disclosed simplification (per the spec's own "can add tiebreaker
logic later" hedge): this ranks each conference's 16 teams purely by
simulated total wins and takes the top 7 - it does NOT model the real
division-winner-guaranteed-a-spot rule or any real tiebreaker (head-to-head,
division record, strength of victory, etc.). This can misclassify a
mediocre division winner as "eliminated" when they'd actually be guaranteed
a spot, and vice versa for a strong non-division-winner. A real limitation,
not silently glossed over - see generate_playoff_probability_report()'s
printed caveat.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

TEAM_CONFERENCE = {
    "BUF": "AFC", "MIA": "AFC", "NE": "AFC", "NYJ": "AFC",
    "BAL": "AFC", "CIN": "AFC", "CLE": "AFC", "PIT": "AFC",
    "HOU": "AFC", "IND": "AFC", "JAX": "AFC", "TEN": "AFC",
    "DEN": "AFC", "KC": "AFC", "LV": "AFC", "LAC": "AFC",
    "DAL": "NFC", "NYG": "NFC", "PHI": "NFC", "WAS": "NFC",
    "CHI": "NFC", "DET": "NFC", "GB": "NFC", "MIN": "NFC",
    "ATL": "NFC", "CAR": "NFC", "NO": "NFC", "TB": "NFC",
    "ARI": "NFC", "LA": "NFC", "SF": "NFC", "SEA": "NFC",
}
PLAYOFF_SPOTS_PER_CONFERENCE = 7


def load_playoff_structure():
    return {"conferences": ["AFC", "NFC"], "spots_per_conference": PLAYOFF_SPOTS_PER_CONFERENCE,
            "team_conference": dict(TEAM_CONFERENCE),
            "note": "Division-winner guarantee and real tiebreakers not modeled - see module docstring."}


def _elo_snapshot_and_remaining_games(season, week_n):
    """Real actual wins-so-far (from completed games through week_n) + the
    real remaining schedule with each game's real win probability, using
    Elo frozen at week_n (weekly_recalibration.py's Component B convention)."""
    from weekly_recalibration import update_elo_with_actual_results
    from elo_game_prediction import calculate_win_probability_from_elo, _load_game_results, ELO_HOME_FIELD
    from game_predictions import _load_schedule_for_season

    updated_ratings = update_elo_with_actual_results(season, week_n)

    played = _load_game_results([season])
    played = played[played["week"] <= week_n].copy()
    played["home_win_val"] = np.select([played["point_diff"] > 0, played["point_diff"] < 0], [1.0, 0.0], default=0.5)
    played["away_win_val"] = 1.0 - played["home_win_val"]
    home_wins = played.groupby("home_team")["home_win_val"].sum()
    away_wins = played.groupby("away_team")["away_win_val"].sum()
    wins_series = home_wins.add(away_wins, fill_value=0.0)
    actual_wins = {t: float(wins_series.get(t, 0.0)) for t in TEAM_CONFERENCE}

    schedule = _load_schedule_for_season(season)
    remaining = schedule[(schedule["game_type"] == "REG") & (schedule["week"] > week_n)].copy()
    remaining["home_win_prob"] = calculate_win_probability_from_elo(
        remaining["home_team"].map(updated_ratings), remaining["away_team"].map(updated_ratings), ELO_HOME_FIELD)
    return actual_wins, remaining[["week", "home_team", "away_team", "home_win_prob"]].reset_index(drop=True)


def get_team_season_projection(team, season, week_n, actual_wins=None, remaining=None):
    """Convenience single-team accessor. For all 32 teams (the real use
    case), calculate_playoff_probability_by_week() is far more efficient -
    one shared Elo/schedule fetch, not 32 repeated ones."""
    if actual_wins is None or remaining is None:
        actual_wins, remaining = _elo_snapshot_and_remaining_games(season, week_n)
    team_games = remaining[(remaining["home_team"] == team) | (remaining["away_team"] == team)]
    win_probs = np.where(team_games["home_team"] == team, team_games["home_win_prob"], 1 - team_games["home_win_prob"])
    return actual_wins[team] + float(win_probs.sum())


def get_remaining_schedule(team, season, week_n, remaining=None):
    if remaining is None:
        _, remaining = _elo_snapshot_and_remaining_games(season, week_n)
    return remaining[(remaining["home_team"] == team) | (remaining["away_team"] == team)].reset_index(drop=True)


def monte_carlo_playoff_odds(actual_wins, remaining_games, n_simulations=10000, seed=42):
    """Simulates every real remaining game across the league per trial (see
    module docstring #2), ranks each conference's 16 teams by simulated
    total wins, top 7 make the playoffs. Odds necessarily sum to EXACTLY 7.0
    per conference (not 'approximately 8') by construction - every trial
    credits exactly 7 of the 16 teams in each conference."""
    rng = np.random.default_rng(seed)
    teams = list(actual_wins.keys())
    team_idx = {t: i for i, t in enumerate(teams)}
    base_wins = np.array([actual_wins[t] for t in teams])

    home_idx = remaining_games["home_team"].map(team_idx).to_numpy()
    away_idx = remaining_games["away_team"].map(team_idx).to_numpy()
    home_prob = remaining_games["home_win_prob"].to_numpy()
    n_games = len(remaining_games)

    conferences = {conf: [i for i, t in enumerate(teams) if TEAM_CONFERENCE[t] == conf]
                   for conf in ("AFC", "NFC")}

    made_playoffs_count = np.zeros(len(teams))
    final_wins_accum = np.zeros(len(teams))

    for _ in range(n_simulations):
        if n_games:
            home_wins_draw = rng.random(n_games) < home_prob
            sim_wins = base_wins.copy()
            np.add.at(sim_wins, home_idx[home_wins_draw], 1.0)
            np.add.at(sim_wins, away_idx[~home_wins_draw], 1.0)
        else:
            sim_wins = base_wins.copy()
        final_wins_accum += sim_wins

        tiebreak_noise = rng.random(len(teams)) * 1e-6  # real tiebreakers not modeled - random tiebreak only
        for idxs in conferences.values():
            conf_wins = sim_wins[idxs] + tiebreak_noise[idxs]
            order = np.argsort(-conf_wins)
            for o in order[:PLAYOFF_SPOTS_PER_CONFERENCE]:
                made_playoffs_count[idxs[o]] += 1

    playoff_pct = made_playoffs_count / n_simulations
    avg_final_wins = final_wins_accum / n_simulations
    se = np.sqrt(np.clip(playoff_pct * (1 - playoff_pct), 0, None) / n_simulations)
    ci_low = np.clip(playoff_pct - 1.645 * se, 0, 1)
    ci_high = np.clip(playoff_pct + 1.645 * se, 0, 1)

    return pd.DataFrame({"team": teams, "conference": [TEAM_CONFERENCE[t] for t in teams],
                          "playoff_odds_pct": playoff_pct, "playoff_odds_90ci_low": ci_low,
                          "playoff_odds_90ci_high": ci_high, "projected_final_wins": avg_final_wins})


def calculate_playoff_probability_by_week(season, week_n, n_simulations=10000):
    actual_wins, remaining = _elo_snapshot_and_remaining_games(season, week_n)
    odds = monte_carlo_playoff_odds(actual_wins, remaining, n_simulations=n_simulations)
    odds["actual_wins_through_week"] = odds["team"].map(actual_wins)
    return odds.sort_values(["conference", "playoff_odds_pct"], ascending=[True, False]).reset_index(drop=True)


def generate_playoff_probability_report(season, week_n, odds=None):
    if odds is None:
        odds = calculate_playoff_probability_by_week(season, week_n)

    lines = [f"Playoff Odds - {season}, through week {week_n}", "=" * 50,
             "(Real 7-of-16 conference format; division-winner guarantee and real",
             " tiebreakers NOT modeled - see src/playoff_probability.py docstring)"]
    for conf in ["AFC", "NFC"]:
        sub = odds[odds["conference"] == conf].sort_values("playoff_odds_pct", ascending=False)
        lines.append(f"\n{conf} (odds sum to {sub['playoff_odds_pct'].sum():.2f} of {PLAYOFF_SPOTS_PER_CONFERENCE} real spots, "
                      f"exact by construction):")
        for _, r in sub.iterrows():
            flag = " <1%" if r["playoff_odds_pct"] < 0.01 else f"{r['playoff_odds_pct']:>5.1%}"
            lines.append(f"  {r['team']:>4}: {flag}  (proj {r['projected_final_wins']:.1f} wins, "
                          f"actual {r['actual_wins_through_week']:.1f} so far)")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    with open(os.path.join(DIAGNOSTIC_DIR, f"playoff_probability_report_{week_n}.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def export_playoff_odds_csv(season, week_n, odds=None, output_path=None):
    if odds is None:
        odds = calculate_playoff_probability_by_week(season, week_n)
    if output_path is None:
        output_path = os.path.join(PROCESSED_DIR, f"playoff_probability_{week_n}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    odds.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def calculate_cumulative_playoff_odds(season, checkpoint_weeks=(1, 4, 8, 12, 16)):
    rows = []
    for wk in checkpoint_weeks:
        odds = calculate_playoff_probability_by_week(season, wk)
        odds["week"] = wk
        rows.append(odds)
        print(f"Week {wk:>2}: simulated ({odds['conference'].eq('AFC').sum()} AFC / "
              f"{odds['conference'].eq('NFC').sum()} NFC teams)")
    trajectory = pd.concat(rows, ignore_index=True)
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    trajectory.to_csv(os.path.join(DIAGNOSTIC_DIR, f"playoff_odds_trajectory_{season}.csv"), index=False, encoding="utf-8")
    return trajectory


if __name__ == "__main__":
    traj = calculate_cumulative_playoff_odds(2025)
    generate_playoff_probability_report(2025, 4, odds=traj[traj["week"] == 4])
