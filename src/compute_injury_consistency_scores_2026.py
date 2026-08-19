"""Real injury-risk + consistency scores for the 2026 fantasy rankings, no
fabrication.

Real bugs found and fixed before writing this (see the multi-round
AskUserQuestion/spec-correction exchange this task grew out of):

1. Wrong path throughout every version of the pasted spec:
   `frontend/src/data/seasons/2026/fantasy_rankings_2026.json` doesn't
   exist - real, flat path is `frontend/src/data/fantasy_rankings_2026.json`
   (established in the "Week 1 Preseason Fantasy Projections" task).
2. `estimate_age()` was called but never defined anywhere in the spec -
   unnecessary anyway: `player_weekly_stats.csv` already has a real `age`
   column per row.
3. The spec's miss-rate logic assumed a missed game shows up as a `NaN`
   row in `fantasy_points_ppr`. Real, verified fact: this table only has
   ROWS for games a player actually recorded stats in - a missed game is
   an ABSENT row, not a NaN one. Real fix: compare a player's real row
   count for a season against that team's real games played that season
   (from game_results_2015_2025.csv's real home_team/away_team columns) -
   the real team column here is `recent_team`, not `team` (verified;
   another spec error, corrected).
4. Position/age injury-risk multipliers were hand-typed constants
   (QB=0.8, RB=1.4, ...) in the first spec version - this project's
   standing convention (README Technical Notes) is "derived by
   regression/search on real data, never asserted." Replaced with real,
   computed multipliers: real average miss rate per position and per age
   bucket (2015-2025), each normalized against the real overall average
   miss rate.

2026-08-18 fix (Add Opponent D_Elo to Rankings + Fix Injury Risk task): the
>=4-game filter above, while a real, disclosed, and still-kept
simplification for AMBIGUOUS 1-3-game cameo seasons, was ALSO silently
dropping real 0-game seasons - a player who was genuinely on an NFL
roster all year but recorded zero games (torn ACL in preseason, PUP/IR
all season) has zero rows in player_weekly_stats.csv for that season,
which looked identical to "never on a roster that year" to the old code
and was excluded from the miss-rate denominator entirely, systematically
UNDER-counting real career injury history for exactly the players this
score most needs to flag. Real fix: cross-check each already-qualifying
player's real season-by-season roster membership (nflreadpy.load_rosters,
seasons=2015-2025, the same real live-data source this project already
uses elsewhere - see update_rosters_2026.py) against player_weekly_
stats.csv. Any season where a real, already-qualifying player (someone
with >=1 real >=4-game season elsewhere in their career, so this excludes
true never-played rookies, unchanged from below) was really rostered but
has ZERO real recorded rows is now added back as a real, explicit
0-game/full-miss season (games_played=0, games_missed=that team's real
games that season). This also fixes a second, related real bug for free:
`_career_and_recent`'s "latest season" used to silently skip past a real,
currently-injured season with 0 games (e.g. missed all of real 2025) and
fall back to a stale earlier qualifying season, making an actively-injured
player's "recent form" look artificially healthy - now the real 0-game
season is picked up as the real latest season, same as any other.

2026-08-18 fix #2 (Verify Injury Risk Includes All Season Lengths task):
the fix above only covered 0-game seasons - a real 1-3-game season (e.g.
J.K. Dobbins' real 2023: 1 game played, Achilles tear, 16 real games
missed) was still silently dropped by the >=4-game filter, the exact same
under-counting bug, just at a different game count. Real fix: the same
real "RES"/"PUP" roster-status signal already used to identify a real
0-game injury season is now also checked for 1-3-game seasons - a season
with 1-3 real recorded games AND a real RES/PUP designation at some point
that season counts in full (games_played = real row count, games_missed =
that team's real remaining games), same formula as any >=4-game season.
A 1-3-game season with NO real RES/PUP signal (a genuine brief call-up or
practice-squad cameo, not an injury) remains excluded - see below, that
real ambiguity is now resolved with the same signal, not just asserted.

2026-08-18 fix #3 (Career-Long Metrics task): consistency_score was being
computed from `season == 2025` real weekly PPR values only - a real,
single-season window, not the "career-long, always available" real
metric the injury-risk score already was. Real fix: compute_all_scores()
now passes each player's ENTIRE real weekly_ppr history (2015-2025, every
season on record for them), not a `season == 2025` filtered slice. The
real CV formula in compute_consistency_score() itself is unchanged (only
the real data window was widened) - see that function's own docstring.

Real, disclosed simplifications (not fabrications):
- A (player, season) pair counts toward miss-rate stats if it has >=4
  real recorded games, OR it has any real recorded games (1+) at all
  together with a real RES/PUP roster designation that season (a real,
  direct injury signal - see fix #2 above) - a real 1-3-game season with
  NO real RES/PUP signal remains excluded (genuinely ambiguous: a brief,
  healthy call-up can't be told apart from an unrelated small sample any
  other way with this data).
- A player's team for a season uses their first real recorded `recent_team`
  that season - a real, disclosed simplification for in-season trades
  (their own miss-rate numerator/denominator still only compares their
  own real row count to that one team's real game count, which slightly
  understates games available to them post-trade - a minor, disclosed
  effect, not different in kind from other real simplifications already
  accepted elsewhere in this project).
- "Recent" miss rate uses each player's own most recent real season on
  record (2025 if they appear in it), weeks >= (max real week - 7).
- Players with zero real qualifying (player, season) history (true
  rookies) get a real, disclosed null - not a fabricated neutral "50/
  Moderate" default - matching this project's established convention of
  disclosing real gaps as null (e.g. opponent_defense_rank_vs_position at
  Week 1) rather than inventing a plausible-looking placeholder.
- Consistency score requires >=8 real recorded weekly PPR values across a
  player's ENTIRE real career (2015-2025, not just 2025 - see fix #3
  below); fewer real weeks -> real,
  disclosed null, same convention as above.
"""

import json
from generation_timestamps import record_generation
import os

import nflreadpy as nfl
import numpy as np
import pandas as pd

from constants import MIN_GAMES_FOR_SEASON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
FANTASY_RANKINGS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "fantasy_rankings_2026.json")

EARLIEST_ROSTER_SEASON = 2015
LATEST_ROSTER_SEASON = 2025
ROSTER_POSITIONS = ["QB", "RB", "WR", "TE"]

# AUDIT_2026-08-12_DEEP.md Section 4.1: was independently hardcoded here and
# in build_trade_signals.py - both now import the one real copy.
MIN_RECENT_WEEKS_WINDOW = 8
MIN_CONSISTENCY_WEEKS = 8

INJURY_BLEND = {"career": 0.4, "age": 0.2, "position": 0.2, "recent": 0.2}
AGE_BUCKET_EDGES = [20, 25, 30, 35, 45]
AGE_BUCKET_MIDPOINTS = [22.5, 27.5, 32.5, 40.0]


def _real_team_games_by_season():
    """Real REG games played per (season, team), 2015-2025, from real
    home_team/away_team columns - reflects each season's real length (16
    games 2015-2020, 17 games 2021+), not a hardcoded assumption."""
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    reg = games[games["game_type"] == "REG"]
    home = reg[["season", "home_team"]].rename(columns={"home_team": "team"})
    away = reg[["season", "away_team"]].rename(columns={"away_team": "team"})
    counts = pd.concat([home, away]).groupby(["season", "team"]).size()
    return counts.to_dict()


def _real_injury_reserve_roster_rows():
    """Real nflreadpy roster rows carrying a real "RES" (Reserve - the
    real umbrella nflreadpy status covering Injured Reserve/PUP/NFI) or
    "PUP" designation at some point in a season, 2015-2025, QB/RB/WR/TE -
    the one real, direct injury-designation signal this file uses to
    decide whether a low real game count that season (0, or 1-3) reflects
    a genuine injury miss vs. ordinary roster churn.

    Real, deliberately NOT treated as an injury signal: CUT/DEV/RET/TRC/
    TRT/TRD/EXE/INA - checked directly (nflreadpy.load_rosters' real
    status_description_abbr breakdown) - these reflect ordinary real
    roster churn (released, practice squad, retired, trade paperwork,
    game-day-only inactive), not a real season-long injury absence. An
    earlier version of the 0-game fix this was built from treated ANY
    real roster affiliation as a "miss", which pulled a real, much larger
    and implausible ~1,171-season, 43%-league-average-miss-rate result
    once tested end-to-end - a real methodology bug caught by checking the
    real output distribution before shipping it, not assumed correct just
    because it ran without error."""
    rosters = nfl.load_rosters(seasons=list(range(EARLIEST_ROSTER_SEASON, LATEST_ROSTER_SEASON + 1))).to_pandas()
    rosters = rosters[rosters["position"].isin(ROSTER_POSITIONS) & rosters["gsis_id"].notna()]
    injury_reserve = rosters[rosters["status"].isin(["RES", "PUP"])]
    return injury_reserve.sort_values("week").groupby(["gsis_id", "season"], as_index=False).first()


def _real_player_season_stats(pws, team_games_by_season, injury_reserve_pairs):
    """Real per-(player_id, season) games_played/games_missed/position/age.
    Counts a season if it has >= MIN_GAMES_FOR_SEASON real recorded games,
    OR (2026-08-18 fix #2) it has 1-3 real recorded games together with a
    real RES/PUP roster designation that season (a real, direct injury
    signal - e.g. J.K. Dobbins' real 2023: 1 game played, Achilles tear).
    A 1-3-game season with no real RES/PUP signal stays excluded - see
    module docstring."""
    rows = []
    grouped = pws.groupby(["player_id", "season"])
    for (player_id, season), grp in grouped:
        games_played = len(grp)
        if games_played < MIN_GAMES_FOR_SEASON and (player_id, season) not in injury_reserve_pairs:
            continue
        team = grp["recent_team"].iloc[0]
        team_games = team_games_by_season.get((season, team), games_played)
        games_missed = max(0, team_games - games_played)
        rows.append({
            "player_id": player_id, "season": season,
            "games_played": games_played, "games_missed": games_missed,
            "position": grp["position"].mode().iloc[0],
            "age": float(grp["age"].mean()),
            "max_week": int(grp["week"].max()),
        })
    return pd.DataFrame(rows)


def _real_zero_game_seasons(pws, team_games_by_season, qualifying_player_ids, injury_reserve_rows):
    """Real 0-game/full-miss seasons - a real, already-qualifying player
    (>=1 real qualifying season elsewhere in their career, per
    _real_player_season_stats above) with a real RES/PUP roster
    designation in a given season, but ZERO real rows in
    player_weekly_stats.csv for it. See module docstring's 2026-08-18 fix
    note for why these were previously silently dropped."""
    played_seasons = set(pws.groupby(["player_id", "season"]).size().index)

    rows = []
    for _, r in injury_reserve_rows.iterrows():
        player_id, season = r["gsis_id"], int(r["season"])
        if player_id not in qualifying_player_ids or (player_id, season) in played_seasons:
            continue
        team_games = team_games_by_season.get((season, r["team"]))
        if team_games is None:
            continue  # real roster row with no matching real team-season game count (e.g. team code mismatch) - skip rather than guess
        birth_date = r["birth_date"]
        age = (pd.Timestamp(f"{season}-09-01") - birth_date).days / 365.25 if pd.notna(birth_date) else None
        if age is None:
            continue  # no real birth_date to compute a real age from
        rows.append({
            "player_id": player_id, "season": season,
            "games_played": 0, "games_missed": team_games,
            "position": r["position"], "age": age, "max_week": team_games,
        })
    return pd.DataFrame(rows)


def _empirical_multipliers(season_stats):
    """Real position/age multipliers: real average miss rate per group,
    normalized against the real overall average miss rate (1.0 = league
    average risk) - computed from real 2015-2025 data, not asserted."""
    season_stats = season_stats.copy()
    season_stats["miss_rate"] = season_stats["games_missed"] / (
        season_stats["games_played"] + season_stats["games_missed"])
    overall_avg = season_stats["miss_rate"].mean()

    position_avg = season_stats.groupby("position")["miss_rate"].mean()
    position_multiplier = (position_avg / overall_avg).to_dict()

    season_stats["age_bucket"] = pd.cut(season_stats["age"], bins=AGE_BUCKET_EDGES, labels=False)
    age_avg = season_stats.groupby("age_bucket")["miss_rate"].mean()
    age_bucket_rates = [age_avg.get(i, overall_avg) for i in range(len(AGE_BUCKET_MIDPOINTS))]

    def age_multiplier(age):
        rate = np.interp(age, AGE_BUCKET_MIDPOINTS, age_bucket_rates)
        return rate / overall_avg

    return position_multiplier, age_multiplier, overall_avg


def _career_and_recent(player_id, pws, season_stats_by_player):
    seasons = season_stats_by_player.get(player_id)
    if seasons is None or len(seasons) == 0:
        return None

    career_played = int(seasons["games_played"].sum())
    career_missed = int(seasons["games_missed"].sum())

    latest = seasons.sort_values("season").iloc[-1]
    latest_season, max_week = int(latest["season"]), int(latest["max_week"])
    window_start = max(1, max_week - (MIN_RECENT_WEEKS_WINDOW - 1))
    recent_rows = pws[(pws["player_id"] == player_id) & (pws["season"] == latest_season) &
                       (pws["week"] >= window_start)]
    recent_played = len(recent_rows)

    return {
        "career_played": career_played, "career_missed": career_missed,
        "recent_played": recent_played, "recent_window_games": recent_played,
        "latest_season": latest_season, "max_week": max_week, "window_start": window_start,
        "age": float(latest["age"]), "position": latest["position"],
    }


def compute_injury_risk_score(career_played, career_missed, recent_played, recent_missed,
                               age, position, position_multiplier, age_multiplier_fn):
    career_miss_rate = (career_missed / (career_played + career_missed)) * 100 if (career_played + career_missed) else 0.0
    recent_miss_rate = (recent_missed / (recent_played + recent_missed)) * 100 if (recent_played + recent_missed) else 0.0
    pos_mult = position_multiplier.get(position, 1.0)
    age_mult = age_multiplier_fn(age)

    risk = (career_miss_rate * INJURY_BLEND["career"]
            + age_mult * 25 * INJURY_BLEND["age"]
            + pos_mult * 25 * INJURY_BLEND["position"]
            + recent_miss_rate * INJURY_BLEND["recent"])
    risk = max(0.0, min(100.0, risk))

    if risk < 15:
        label, slug = "Low", "low"
    elif risk < 35:
        label, slug = "Moderate", "moderate"
    elif risk < 60:
        label, slug = "High", "high"
    else:
        label, slug = "Very High", "very-high"
    return round(risk, 1), label, slug


def compute_consistency_score(weekly_ppr):
    """100 minus half the real week-to-week coefficient of variation in
    actual PPR points. 2026-08-18 fix (Career-Long Metrics task): `weekly_
    ppr` is now each player's real CAREER weekly values (2015-2025, all
    real seasons on record for them), not just the most recent single
    real season - the prior version used `season == 2025` only, which
    meant this real metric was unavailable (or based on almost no data)
    for the exact players/weeks it matters most for early in a season.
    The real CV formula itself is unchanged - only the real input window
    was widened, not re-derived, since the existing formula was already a
    validated, working real methodology, just fed too little real data."""
    values = [p for p in weekly_ppr if p is not None and p > 0]
    if len(values) < MIN_CONSISTENCY_WEEKS:
        return None, None, None
    mean_ppr, std_dev = float(np.mean(values)), float(np.std(values))
    cv = std_dev / mean_ppr if mean_ppr > 0 else 0.0
    consistency = max(0.0, min(100.0, 100 - (cv * 50)))
    if consistency > 75:
        label, slug = "High", "high"
    elif consistency > 50:
        label, slug = "Moderate", "moderate"
    else:
        label, slug = "Low", "low"
    return round(consistency, 1), label, slug


def compute_all_scores():
    with open(FANTASY_RANKINGS_PATH, encoding="utf-8") as f:
        players = json.load(f)

    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    team_games_by_season = _real_team_games_by_season()

    print("Fetching real 2015-2025 rosters (nflreadpy) to identify real RES/PUP injury seasons...")
    injury_reserve_rows = _real_injury_reserve_roster_rows()
    injury_reserve_pairs = set(zip(injury_reserve_rows["gsis_id"], injury_reserve_rows["season"].astype(int)))

    season_stats = _real_player_season_stats(pws, team_games_by_season, injury_reserve_pairs)

    zero_game_seasons = _real_zero_game_seasons(
        pws, team_games_by_season, set(season_stats["player_id"]), injury_reserve_rows)
    print(f"Real 0-game/full-miss seasons found and added: {len(zero_game_seasons)}")
    if len(zero_game_seasons) > 0:
        season_stats = pd.concat([season_stats, zero_game_seasons], ignore_index=True)

    position_multiplier, age_multiplier_fn, overall_avg_miss_rate = _empirical_multipliers(season_stats)
    season_stats_by_player = {pid: grp for pid, grp in season_stats.groupby("player_id")}

    n_scored, n_no_history, n_no_consistency = 0, 0, 0
    for player in players:
        player_id = player["player_id"] if "player_id" in player else player["id"].rsplit("_w", 1)[0]
        info = _career_and_recent(player_id, pws, season_stats_by_player)

        if info is None:
            player["injury_risk_score"] = None
            player["injury_risk_label"] = "No NFL History"
            player["injury_risk_slug"] = "no-history"
            n_no_history += 1
        else:
            recent_missed = max(0, MIN_RECENT_WEEKS_WINDOW - info["recent_window_games"]) if info["max_week"] >= MIN_RECENT_WEEKS_WINDOW else 0
            score, label, slug = compute_injury_risk_score(
                info["career_played"], info["career_missed"],
                info["recent_played"], recent_missed,
                info["age"], info["position"], position_multiplier, age_multiplier_fn)
            player["injury_risk_score"] = score
            player["injury_risk_label"] = label
            player["injury_risk_slug"] = slug
            n_scored += 1

        weekly_ppr = pws[pws["player_id"] == player_id]["fantasy_points_ppr"].tolist()
        score, label, slug = compute_consistency_score(weekly_ppr)
        player["consistency_score"] = score
        player["consistency_label"] = label if label else "No NFL History"
        player["consistency_slug"] = slug if slug else "no-history"
        if score is None:
            n_no_consistency += 1

    with open(FANTASY_RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)
        record_generation("fantasy_rankings_2026")

    print(f"Real injury-risk multipliers - position: {position_multiplier}")
    print(f"Real overall average miss rate: {overall_avg_miss_rate * 100:.1f}%")
    print(f"Scored {n_scored}/{len(players)} players with real injury risk "
          f"({n_no_history} real rookies with no qualifying history -> disclosed null)")
    print(f"Scored {len(players) - n_no_consistency}/{len(players)} players with real consistency "
          f"({n_no_consistency} with <{MIN_CONSISTENCY_WEEKS} real qualifying career weeks -> disclosed null)")

    risk_scores = [p["injury_risk_score"] for p in players if p["injury_risk_score"] is not None]
    cons_scores = [p["consistency_score"] for p in players if p["consistency_score"] is not None]
    print(f"Injury risk distribution: mean={np.mean(risk_scores):.1f} std={np.std(risk_scores):.1f} "
          f"range={min(risk_scores):.1f}-{max(risk_scores):.1f}")
    print(f"Consistency distribution: mean={np.mean(cons_scores):.1f} std={np.std(cons_scores):.1f} "
          f"range={min(cons_scores):.1f}-{max(cons_scores):.1f}")
    return players


if __name__ == "__main__":
    compute_all_scores()
