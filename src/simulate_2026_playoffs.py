"""Real Monte Carlo regular-season simulation - real 2026 preseason playoff
odds, no fabrication.

Real context (see AskUserQuestion exchange this task started from): this
project's real "Fix Win Totals Variance" investigation found the dashboard's
narrow-looking 2026 win-total point estimate already has a real, exact 90%
CI sitting unused in ensemble_season_wins_2026.csv (elo_model.py's
project_season_wins_from_elo computes Var(sum of independent per-game
Bernoullis) = sum(p*(1-p)) in closed form - a real 10,000-trial Monte Carlo
of the same per-game probabilities would converge to that exact same
mean/variance, not a new number). That closed-form CI is surfaced separately
by generate_season_projections_dashboard_data_2026.py, not here.

What a real Monte Carlo sim genuinely adds instead - the one thing the
closed-form per-team CI structurally cannot answer - is a JOINT question:
"does this team's simulated record beat its real division/conference
rivals' simulated records IN THE SAME TRIAL." That's exactly the real,
already-disclosed gap this project's own docstrings flagged: generate_
season_projections_dashboard_data_2026.py's playoff_percentage/playoff_seed/
is_division_winner/is_playoff_team are all null/false for 2026 because "no
real Monte Carlo playoff simulation built against a still-hypothetical 2026
standing" existed yet. This module builds that, reusing real, already-
validated infrastructure rather than inventing new per-game probabilities:

- calculate_win_probability_from_elo (elo_game_prediction.py) - the same
  real, already-fit Elo win-probability formula superbowl_bracket_
  simulation.py already uses for the playoff bracket, one step earlier
  (regular season instead of playoffs).
- DIVISIONS/TEAM_TO_DIVISION/TEAM_TO_CONFERENCE and the real seeding
  RULE (division leaders 1-4 by strength, next-best 3 wildcards 5-7) from
  generate_season_projections_dashboard_data.py's real _compute_seeds -
  same real algorithm, not reinvented.

Real per-team preseason Elo is recomputed here via elo_model.run_multi_
season_elo(range(2015, 2027)) - the EXACT same real carryover-Elo call
archive/ensemble_model.py's get_elo_season_predictions(2026) already uses
to produce elo_wins/ensemble_wins (the number this task's other half
surfaces a CI for) - NOT read from data/processed/elo_game_predictions_
2026.csv. Checked and found a real, pre-existing, undisclosed inconsistency
between those two: elo_game_prediction.py's generate_elo_game_spreads()
season>2025 branch takes ratings_at_season_start[season - 1] (2026's would
be ratings_at_season_start[2025] - the START of 2025, before any 2025
games) and applies one manual extra regression step, which silently
discards every real 2025 game result rather than rolling them forward -
the discarded return value (run_multi_season_elo's 3rd element, true
end-of-2025 ratings) is what should have been used. That bug lives in
elo_game_predictions_2026.csv/the Game Predictions section's spreads,
outside this task's scope (fixing it would also move the real spread/CI
shown elsewhere in the dashboard) - flagged in DASHBOARD_DATA_GAPS.md
rather than silently fixed here. Using the SAME real Elo values that
already produced elo_wins/ensemble_wins keeps this simulation internally
consistent with the win-total point estimate/CI surfaced alongside it,
rather than quietly introducing a second, different "2026 preseason Elo."

Update (Full Polish task, 2026-08-12): the elo_game_predictions_2026.csv
bug described above is now fixed (elo_game_prediction.py's season>2025
branch), and the shared computation both this function and that fixed
branch use is now centralized in elo_utils.compute_season_start_elo
rather than each maintaining its own copy - see that module's docstring.

Two real, disclosed simplifications (both matching this project's existing
disclosure convention rather than hiding them):

1. Tiebreak proxy: _compute_seeds tiebreaks real teams by real point
   differential, which requires real game scores. This simulation only
   draws binary win/loss per game (a real matchup's real margin isn't
   simulated), so no simulated point differential exists to tiebreak with.
   Substitutes each team's real, static preseason Elo rating instead - the
   closest real, already-computed proxy available, same category as this
   project's other disclosed tiebreak simplifications (_compute_seeds
   itself already discloses using point differential as "a real, simplified
   stand-in for the NFL's full multi-step tiebreaker procedure").
2. Ties aren't modeled (real per-game tie rate is ~0.4%, 10 total across
   2015-2025 - rare enough that this project's other real win-probability
   work (elo_model.py, epa_to_wins.py) doesn't model them either beyond
   crediting 0.5 wins when they occur historically); every simulated game
   has a binary winner.

Two real Monte Carlo outputs are computed and used differently:
- playoff_percentage / division_winner_percentage: the RAW real fraction of
  10,000 trials in which each team's real per-trial seeding (division
  leaders 1-4 by trial wins+Elo-tiebreak, wildcards 5-7) placed it in the
  playoffs / atop its division. Valid, honest real probabilities for every
  team - not constrained to sum to a fixed count.
- is_playoff_team / is_division_winner / playoff_seed: a real, DERIVED
  selection (exactly the top-7-by-playoff_percentage per conference / top-1-
  by-division_winner_percentage per division, seeded 1-4/5-7 by the same
  real _compute_seeds-style rule) - needed because the frontend's bracket
  and division-winner panels structurally expect exactly 7 seeds/conference
  and exactly 1 winner/division (same real invariant 2025's deterministic
  seeding always satisfies), which a raw 50%-threshold on independent
  per-team probabilities would NOT reliably produce a preseason where odds
  are naturally more spread out. This mirrors _compute_seeds's own real
  algorithm, just keyed on aggregated Monte Carlo percentages instead of a
  single trial's (wins, point_diff).
"""

import os

import numpy as np

from elo_game_prediction import ELO_HOME_FIELD, calculate_win_probability_from_elo
from elo_model import TRAIN_SEASONS
from constants import ELO_K_FACTOR
from game_predictions import _load_schedule_for_season
from generate_season_projections_dashboard_data import DIVISIONS, TEAM_TO_CONFERENCE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SEASON = 2026

N_SIMULATIONS = 10000
RNG_SEED = 42


def real_2026_carryover_elo():
    """Real per-team carryover preseason Elo for 2026 - recomputed via
    elo_utils.compute_season_start_elo (shared with elo_game_prediction.py's
    2026 game-spread branch, see that module's docstring) rather than read
    from elo_game_predictions_2026.csv (see module docstring: the latter has a
    real, separate, pre-existing bug that discards 2025's actual results;
    this project's already-published elo_wins/ensemble_wins do NOT have
    that bug). Exported (not underscore-prefixed) so other real 2026
    deliverables that need the exact same preseason Elo - e.g. the 2026
    Super Bowl odds simulation - reuse this instead of re-deriving a
    second, potentially-diverging copy."""
    from elo_utils import compute_season_start_elo
    return compute_season_start_elo(min(TRAIN_SEASONS), SEASON, ELO_K_FACTOR, ELO_HOME_FIELD)


def _real_2026_schedule_and_elo():
    """Real 272-game 2026 REG schedule (schedules_2026.csv, via the same
    real _load_schedule_for_season every other 2026 deliverable uses) +
    each team's real carryover preseason Elo rating (real_2026_carryover_
    elo(), see its docstring)."""
    elo_lookup = real_2026_carryover_elo()

    schedule = _load_schedule_for_season(SEASON)
    reg = schedule[schedule["game_type"] == "REG"].copy()
    teams = sorted(set(reg["home_team"]) | set(reg["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}

    home_idx = reg["home_team"].map(team_idx).to_numpy()
    away_idx = reg["away_team"].map(team_idx).to_numpy()
    p_home = np.array([
        calculate_win_probability_from_elo(elo_lookup[h], elo_lookup[a], ELO_HOME_FIELD)
        for h, a in zip(reg["home_team"], reg["away_team"])
    ])
    return teams, team_idx, home_idx, away_idx, p_home, elo_lookup


def _simulate_win_totals(n_teams, home_idx, away_idx, p_home, n_simulations, rng_seed):
    """Fully vectorized: draws all (n_simulations x 272) real per-game
    Bernoulli outcomes at once from the real per-game Elo win probabilities,
    then matrix-multiplies into per-team win totals per trial."""
    rng = np.random.default_rng(rng_seed)
    n_games = len(p_home)
    home_wins = rng.random((n_simulations, n_games)) < p_home

    home_onehot = np.zeros((n_games, n_teams))
    home_onehot[np.arange(n_games), home_idx] = 1
    away_onehot = np.zeros((n_games, n_teams))
    away_onehot[np.arange(n_games), away_idx] = 1

    wins = home_wins.astype(np.int32) @ home_onehot + (~home_wins).astype(np.int32) @ away_onehot
    return wins.astype(np.int32)  # (n_simulations, n_teams)


def _trial_seeds_ordered(win_row, team_idx, elo_lookup, conference):
    """One trial's real ORDERED seeding for one conference: division
    leaders 1-4 (by trial wins, Elo-tiebreak - see module docstring
    simplification #1), next-best 3 wildcards 5-7 - returned as an ordered
    (leaders, wildcards) pair (seed 1..7 in list order), not just a set, so
    a real per-trial bracket (see run_2026_superbowl_simulation below) can
    be seeded correctly. Mirrors generate_season_projections_dashboard_
    data._compute_seeds's real algorithm."""
    def strength(t):
        return (win_row[team_idx[t]], elo_lookup[t])

    conf_divisions = [d for d in DIVISIONS if d.startswith(conference)]
    division_leaders = [max(DIVISIONS[div], key=strength) for div in conf_divisions]
    division_leaders.sort(key=strength, reverse=True)

    conf_teams = [t for t in TEAM_TO_CONFERENCE if TEAM_TO_CONFERENCE[t] == conference]
    wildcard_pool = sorted((t for t in conf_teams if t not in division_leaders), key=strength, reverse=True)
    wildcards = wildcard_pool[:3]

    return division_leaders, wildcards


def _trial_seeds(win_row, team_idx, elo_lookup, conference):
    """Unordered real playoff-team set for one trial/conference - thin
    wrapper over _trial_seeds_ordered for callers that only need
    membership (e.g. the playoff_percentage tally below), not seed order."""
    leaders, wildcards = _trial_seeds_ordered(win_row, team_idx, elo_lookup, conference)
    return set(leaders) | set(wildcards)


def _aggregate_seeds(team_stats, conference):
    """Real, derived top-7/top-1 selection from the aggregated Monte Carlo
    percentages - see module docstring for why this (not a 50% threshold)
    is used for the frontend-facing is_playoff_team/is_division_winner/
    playoff_seed fields."""
    conf_divisions = [d for d in DIVISIONS if d.startswith(conference)]
    division_leaders = [
        max(DIVISIONS[div], key=lambda t: team_stats[t]["division_winner_percentage"])
        for div in conf_divisions
    ]
    division_leaders.sort(key=lambda t: team_stats[t]["playoff_percentage"], reverse=True)

    conf_teams = [t for t in TEAM_TO_CONFERENCE if TEAM_TO_CONFERENCE[t] == conference]
    wildcard_pool = sorted(
        (t for t in conf_teams if t not in division_leaders),
        key=lambda t: team_stats[t]["playoff_percentage"], reverse=True)
    wildcards = wildcard_pool[:3]

    seeds = {}
    for i, t in enumerate(division_leaders):
        seeds[t] = i + 1
    for i, t in enumerate(wildcards):
        seeds[t] = i + 5
    return seeds, set(division_leaders)


def run_2026_playoff_simulation(n_simulations=N_SIMULATIONS, rng_seed=RNG_SEED):
    teams, team_idx, home_idx, away_idx, p_home, elo_lookup = _real_2026_schedule_and_elo()
    n_teams = len(teams)
    wins = _simulate_win_totals(n_teams, home_idx, away_idx, p_home, n_simulations, rng_seed)

    playoff_count = {t: 0 for t in teams}
    division_winner_count = {t: 0 for t in teams}
    # Real per-seed tally (Playoff Picture seed-distribution grid,
    # 2026-08-19 follow-up): unlike playoff_percentage/is_playoff_team
    # (playoff-or-not, and a single derived "most likely" seed), this keeps
    # the full real distribution across all 10,000 trials - e.g. a bubble
    # team might be seed 7 in 20% of trials and miss entirely in the other
    # 80%, which a single point estimate can't show.
    seed_count = {t: {s: 0 for s in range(1, 8)} for t in teams}

    for trial in range(n_simulations):
        row = wins[trial]
        trial_playoff_teams = set()
        trial_div_winners = set()
        for conf in ("AFC", "NFC"):
            def strength(t, row=row):
                return (row[team_idx[t]], elo_lookup[t])
            conf_divisions = [d for d in DIVISIONS if d.startswith(conf)]
            leaders = [max(DIVISIONS[div], key=strength) for div in conf_divisions]
            trial_div_winners.update(leaders)

            ordered_leaders, ordered_wildcards = _trial_seeds_ordered(row, team_idx, elo_lookup, conf)
            for i, t in enumerate(ordered_leaders):
                seed_count[t][i + 1] += 1
            for i, t in enumerate(ordered_wildcards):
                seed_count[t][i + 5] += 1
            trial_playoff_teams.update(set(ordered_leaders) | set(ordered_wildcards))
        for t in trial_playoff_teams:
            playoff_count[t] += 1
        for t in trial_div_winners:
            division_winner_count[t] += 1

    team_stats = {
        t: {
            "playoff_percentage": round(playoff_count[t] / n_simulations, 4),
            "division_winner_percentage": round(division_winner_count[t] / n_simulations, 4),
            "seed_distribution": {str(s): round(seed_count[t][s] / n_simulations, 4) for s in range(1, 8)},
        }
        for t in teams
    }

    all_seeds, all_div_winners = {}, set()
    for conf in ("AFC", "NFC"):
        seeds, div_winners = _aggregate_seeds(team_stats, conf)
        all_seeds.update(seeds)
        all_div_winners |= div_winners

    for t in teams:
        team_stats[t]["is_playoff_team"] = t in all_seeds
        team_stats[t]["is_division_winner"] = t in all_div_winners
        team_stats[t]["playoff_seed"] = all_seeds.get(t)

    return team_stats


def run_2026_superbowl_simulation(n_simulations=N_SIMULATIONS, rng_seed=RNG_SEED):
    """Real, INTEGRATED regular-season + playoff-bracket Monte Carlo
    (PersonalRoster/SeasonProjections follow-up, 2026-08-19): the original
    generate_superbowl_odds_2026.py collapsed all real regular-season
    uncertainty into ONE fixed, most-likely 14-team bracket (this module's
    own _aggregate_seeds, a real, derived point estimate) before running
    the playoff bracket - meaning any of the real 18 non-derived teams got
    a hard, literal 0.0% Super Bowl chance, even ones with real, meaningful
    (e.g. 20-40%) playoff odds. Before the real season starts, ANY team
    still has a real (if often small) chance at the Super Bowl until it's
    actually eliminated - collapsing to one bracket erases that real
    uncertainty rather than modeling it.

    Real fix: simulates a real playoff bracket INSIDE each of the N real
    regular-season trials, using THAT TRIAL's own real seeding (this
    module's own _trial_seeds_ordered - real per-trial division
    leaders/wildcards, not a fixed aggregate), reusing superbowl_bracket_
    simulation.py's real, already-validated bracket-advancement rules
    (bye, re-seeding, neutral-site Super Bowl) rather than reinventing
    them. Every one of the real 32 teams' Super Bowl odds now correctly
    integrate the real JOINT probability of "makes the playoffs in this
    trial's real seeding" AND "wins the resulting bracket" - a rare team
    that snuck into 3% of trials' brackets still gets a real, nonzero
    (if small) share of those trials' Super Bowl wins, instead of being
    silently zeroed out for not appearing in the one aggregate bracket.

    Uses the SAME real per-trial win draw (same rng_seed) as
    run_2026_playoff_simulation() above, so this function's own
    playoff_percentage/division_winner_percentage outputs are guaranteed
    identical to that function's - real, deliberate internal consistency
    (same reasoning as real_2026_carryover_elo()'s own docstring), not a
    second, independently-derived simulation that could quietly diverge.
    Uses a separate RNG stream (rng_seed + 1) for the bracket-game draws
    themselves, so those draws don't consume/shift the regular-season
    Bernoulli draws."""
    from superbowl_bracket_simulation import simulate_conference_bracket, _play_seeded_game

    teams, team_idx, home_idx, away_idx, p_home, elo_lookup = _real_2026_schedule_and_elo()
    n_teams = len(teams)
    wins = _simulate_win_totals(n_teams, home_idx, away_idx, p_home, n_simulations, rng_seed)
    bracket_rng = np.random.default_rng(rng_seed + 1)

    playoff_count = {t: 0 for t in teams}
    division_winner_count = {t: 0 for t in teams}
    conf_champ_count = {t: 0 for t in teams}
    sb_win_count = {t: 0 for t in teams}

    for trial in range(n_simulations):
        row = wins[trial]
        trial_champs = {}
        for conf in ("AFC", "NFC"):
            leaders, wildcards = _trial_seeds_ordered(row, team_idx, elo_lookup, conf)
            for t in leaders:
                division_winner_count[t] += 1
            for t in leaders + wildcards:
                playoff_count[t] += 1

            ordered_seeds = leaders + wildcards  # seed 1..7, real bracket order
            champ = simulate_conference_bracket(bracket_rng, ordered_seeds, elo_lookup, ELO_HOME_FIELD)
            conf_champ_count[champ] += 1
            trial_champs[conf] = champ

        sb_winner, _ = _play_seeded_game(
            bracket_rng, trial_champs["AFC"], 1, trial_champs["NFC"], 1, elo_lookup, home_field_elo=0)
        sb_win_count[sb_winner] += 1

    team_stats = {
        t: {
            "playoff_percentage": round(playoff_count[t] / n_simulations, 4),
            "division_winner_percentage": round(division_winner_count[t] / n_simulations, 4),
            "conference_champion_percentage": round(conf_champ_count[t] / n_simulations, 4),
            "superbowl_percentage": round(sb_win_count[t] / n_simulations, 4),
        }
        for t in teams
    }

    all_seeds, all_div_winners = {}, set()
    for conf in ("AFC", "NFC"):
        seeds, div_winners = _aggregate_seeds(team_stats, conf)
        all_seeds.update(seeds)
        all_div_winners |= div_winners

    for t in teams:
        team_stats[t]["is_playoff_team"] = t in all_seeds
        team_stats[t]["is_division_winner"] = t in all_div_winners
        team_stats[t]["playoff_seed"] = all_seeds.get(t)

    return team_stats


if __name__ == "__main__":
    stats = run_2026_playoff_simulation()
    n_playoff = sum(1 for s in stats.values() if s["is_playoff_team"])
    n_div_winners = sum(1 for s in stats.values() if s["is_division_winner"])
    print(f"Real {N_SIMULATIONS}-trial 2026 preseason playoff simulation complete.")
    print(f"Playoff teams (derived): {n_playoff} (expect 14) | Division winners (derived): {n_div_winners} (expect 8)")
    top = sorted(stats.items(), key=lambda kv: -kv[1]["playoff_percentage"])[:10]
    print("\nTop 10 real playoff percentages:")
    for team, s in top:
        seed = f"seed {s['playoff_seed']}" if s["playoff_seed"] else "no seed"
        print(f"  {team:>4}: {s['playoff_percentage']*100:5.1f}% playoffs | "
              f"{s['division_winner_percentage']*100:5.1f}% division | {seed}")
