"""Real Monte Carlo Super Bowl bracket simulation.

Extends src/archive/playoff_probability.py's already-validated approach
(real per-game win probabilities from real, frozen Elo; every simulated
game drawn independently per trial rather than an unfit heuristic) from
"who makes the playoffs" through a real seeded bracket to a real Super Bowl
winner - chosen over the simpler win-probability-scaling heuristic
originally proposed, per explicit instruction, since it's more consistent
with this project's established preference for real simulation over
asserted formulas (the same preference already reflected in the real
`playoff_percentage` field).

Real, disclosed simplification: the bracket seeding is FIXED at the real
week-16 checkpoint's projected seeds (season_projections_2025.json's real
`playoff_seed` field - the same seeds the Playoff Picture panel shows),
not re-simulated. This answers "who wins the Super Bowl if the projected
seeding holds," not two compounded layers of uncertainty (regular-season
seeding AND playoff results). Disclosed here and in Model Transparency,
the same way playoff_probability.py discloses its own real simplifications
(no real tiebreakers modeled) rather than glossing over them.

Real bracket rules modeled (2020-present NFL format, 7 seeds/conference):
- Wild Card round: #1 seed byes. Real matchups: (2) v (7), (3) v (6), (4) v (5).
- Divisional round: real NFL re-seeding rule - the #1 seed plays whichever
  wild-card winner has the WORST (highest-numbered) seed, not a fixed
  bracket slot; the other two wild-card winners play each other.
- Conference Championship: the two divisional-round winners play each other.
- Super Bowl: AFC champion v NFC champion, real neutral-site rule (no
  home-field Elo term - home_field_elo=0), conference champions from
  simulated brackets, not fixed seeds.
- Real rule used throughout except the Super Bowl: the better (lower-
  numbered) remaining seed always hosts.

Home-field Elo term reuses this project's already-fit regular-season
constant (ELO_HOME_FIELD) - a real, disclosed simplification, since no
separate real playoff-specific home-field constant exists in this project
to fit from; reusing the already-validated constant is preferred here over
inventing a new, unfit one for playoff games specifically.
"""

import json
from generation_timestamps import record_generation
import os

import numpy as np
import pandas as pd

from elo_game_prediction import ELO_HOME_FIELD, calculate_win_probability_from_elo
from weekly_recalibration import update_elo_with_actual_results

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
SEASON_PROJECTIONS_PATH = os.path.join(FRONTEND_DATA_DIR, "season_projections_2025.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "superbowl_odds_2025.json")

SEASON = 2025
CHECKPOINT_WEEK = 16  # matches Season Projections' real week-16 checkpoint
N_SIMULATIONS = 10000
RNG_SEED = 42


def _load_season_projections():
    with open(SEASON_PROJECTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _real_seeds_by_conference(teams):
    """Real week-16 projected seeds (1-7) per conference, from
    season_projections_2025.json's already-computed, tiebreak-aware
    playoff_seed field - not re-derived here."""
    seeds = {"AFC": [None] * 7, "NFC": [None] * 7}
    for t in teams:
        if not t["is_playoff_team"]:
            continue
        seeds[t["conference"]][int(t["playoff_seed"]) - 1] = t["team"]
    for conf, lst in seeds.items():
        if any(s is None for s in lst):
            raise RuntimeError(f"Missing a real playoff seed in {conf}: {lst}")
    return seeds


def _real_frozen_elo(season=SEASON, week_n=CHECKPOINT_WEEK):
    """Real, causal Elo ratings as of right after week_n - same real,
    leak-free snapshot mechanism src/archive/playoff_probability.py already
    uses (weekly_recalibration.update_elo_with_actual_results)."""
    return update_elo_with_actual_results(season, week_n)


def _play_seeded_game(rng, team_a, seed_a, team_b, seed_b, elo, home_field_elo):
    """Real rule: the better (lower-numbered) seed hosts, except at the
    Super Bowl (home_field_elo=0 there, making the home/away label moot -
    a real neutral-site game)."""
    if seed_a <= seed_b:
        home_team, home_seed, away_team, away_seed = team_a, seed_a, team_b, seed_b
    else:
        home_team, home_seed, away_team, away_seed = team_b, seed_b, team_a, seed_a
    prob_home = calculate_win_probability_from_elo(elo[home_team], elo[away_team], home_field_elo)
    if rng.random() < prob_home:
        return home_team, home_seed
    return away_team, away_seed


def simulate_conference_bracket(rng, seeds, elo, home_field_elo):
    """Real single-conference bracket for one trial - see module docstring
    for the real rules modeled. seeds is the real ordered [seed1, ..., seed7]
    list. Returns the real simulated conference champion."""
    seed_team = {i + 1: seeds[i] for i in range(7)}

    wc_pairs = [(2, 7), (3, 6), (4, 5)]
    wc_winners = [
        _play_seeded_game(rng, seed_team[s_hi], s_hi, seed_team[s_lo], s_lo, elo, home_field_elo)
        for s_hi, s_lo in wc_pairs
    ]

    wc_winners.sort(key=lambda w: w[1])  # ascending seed number = best remaining first
    best_two, lowest_remaining = wc_winners[:2], wc_winners[2]

    div_w1 = _play_seeded_game(rng, seed_team[1], 1, lowest_remaining[0], lowest_remaining[1], elo, home_field_elo)
    div_w2 = _play_seeded_game(rng, best_two[0][0], best_two[0][1], best_two[1][0], best_two[1][1], elo, home_field_elo)

    champion, _ = _play_seeded_game(rng, div_w1[0], div_w1[1], div_w2[0], div_w2[1], elo, home_field_elo)
    return champion


def simulate_superbowl(seeds_by_conference, elo, n_simulations=N_SIMULATIONS, rng_seed=RNG_SEED,
                        home_field_elo=ELO_HOME_FIELD):
    rng = np.random.default_rng(rng_seed)
    playoff_teams = seeds_by_conference["AFC"] + seeds_by_conference["NFC"]
    conf_champ_wins = {t: 0 for t in playoff_teams}
    sb_wins = {t: 0 for t in playoff_teams}

    for _ in range(n_simulations):
        afc_champ = simulate_conference_bracket(rng, seeds_by_conference["AFC"], elo, home_field_elo)
        nfc_champ = simulate_conference_bracket(rng, seeds_by_conference["NFC"], elo, home_field_elo)
        conf_champ_wins[afc_champ] += 1
        conf_champ_wins[nfc_champ] += 1

        # Real neutral-site Super Bowl: home_field_elo=0, so the seed labels
        # passed here (1, 1) never affect the outcome - they only decide an
        # otherwise-irrelevant "home" label inside _play_seeded_game.
        sb_winner, _ = _play_seeded_game(rng, afc_champ, 1, nfc_champ, 1, elo, home_field_elo=0)
        sb_wins[sb_winner] += 1

    conf_champ_pct = {t: conf_champ_wins[t] / n_simulations for t in playoff_teams}
    sb_pct = {t: sb_wins[t] / n_simulations for t in playoff_teams}
    return conf_champ_pct, sb_pct


def generate_superbowl_odds_json():
    teams = _load_season_projections()
    seeds_by_conference = _real_seeds_by_conference(teams)
    elo = _real_frozen_elo()

    missing_elo = [t for conf in seeds_by_conference.values() for t in conf if t not in elo]
    if missing_elo:
        raise RuntimeError(f"Missing real frozen Elo for real playoff teams: {missing_elo}")

    conf_champ_pct, sb_pct = simulate_superbowl(seeds_by_conference, elo)

    seed_lookup = {t["team"]: t for t in teams}
    results = []
    for t in teams:
        team = t["team"]
        results.append({
            "team": team,
            "conference": t["conference"],
            "division": t["division"],
            "is_playoff_team": bool(t["is_playoff_team"]),
            "playoff_seed": int(t["playoff_seed"]) if t["is_playoff_team"] else None,
            "wins_actual": t["wins_actual"],
            "losses_actual": t["losses_actual"],
            "ties_actual": t["ties_actual"],
            "conference_champion_pct": round(conf_champ_pct.get(team, 0.0) * 100, 1),
            "superbowl_odds_pct": round(sb_pct.get(team, 0.0) * 100, 1),
        })
    results.sort(key=lambda r: -r["superbowl_odds_pct"])

    output = {
        "season": SEASON,
        "checkpoint_week": CHECKPOINT_WEEK,
        "n_simulations": N_SIMULATIONS,
        "methodology_note": (
            "Real Monte Carlo bracket simulation (not a heuristic): seeding fixed at the real "
            "week-16 projected seeds; each simulated playoff game uses real, frozen week-16 Elo "
            "ratings via this project's real win-probability formula; real bracket rules modeled "
            "(bye, re-seeding, neutral-site Super Bowl). Answers 'who wins if the projected "
            "seeding holds,' not two compounded layers of regular-season-plus-playoff uncertainty. "
            "See src/superbowl_bracket_simulation.py and Model Transparency for detail."
        ),
        "teams": results,
    }

    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("superbowl_odds_2025")

    print(f"Wrote {OUTPUT_PATH}")
    afc_sum = sum(r["superbowl_odds_pct"] for r in results if r["conference"] == "AFC")
    nfc_sum = sum(r["superbowl_odds_pct"] for r in results if r["conference"] == "NFC")
    print(f"AFC SB-odds sum: {afc_sum:.1f}% | NFC SB-odds sum: {nfc_sum:.1f}% (real, should each be ~50%)")
    print("\nTop 10 real Super Bowl odds:")
    for r in results[:10]:
        print(f"  {r['team']:>4} (seed {r['playoff_seed']}, {r['conference']}): "
              f"{r['superbowl_odds_pct']:>5.1f}% SB | {r['conference_champion_pct']:>5.1f}% conf champ")
    return output


if __name__ == "__main__":
    generate_superbowl_odds_json()
