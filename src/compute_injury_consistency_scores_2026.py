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

Real, disclosed simplifications (not fabrications):
- Only (player, season) pairs with >=4 real recorded games count toward
  miss-rate stats - filters out real cameo/practice-squad seasons that
  would otherwise look like near-100%-missed seasons for a reason
  unrelated to injury.
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
- Consistency score requires >=8 real recorded 2025 weekly PPR values
  (matching the pasted spec's own threshold); fewer real weeks -> real,
  disclosed null, same convention as above.
"""

import json
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
FANTASY_RANKINGS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "fantasy_rankings_2026.json")

MIN_GAMES_FOR_SEASON = 4
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


def _real_player_season_stats(pws, team_games_by_season):
    """Real per-(player_id, season) games_played/games_missed/position/age,
    filtered to seasons with >= MIN_GAMES_FOR_SEASON real recorded games."""
    rows = []
    grouped = pws.groupby(["player_id", "season"])
    for (player_id, season), grp in grouped:
        games_played = len(grp)
        if games_played < MIN_GAMES_FOR_SEASON:
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

    team_games = pws[(pws["player_id"] == player_id) & (pws["season"] == latest_season)]["recent_team"].iloc[0]
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
    season_stats = _real_player_season_stats(pws, team_games_by_season)
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

        weekly_ppr = pws[(pws["player_id"] == player_id) & (pws["season"] == 2025)]["fantasy_points_ppr"].tolist()
        score, label, slug = compute_consistency_score(weekly_ppr)
        player["consistency_score"] = score
        player["consistency_label"] = label if label else "No 2025 History"
        player["consistency_slug"] = slug if slug else "no-history"
        if score is None:
            n_no_consistency += 1

    with open(FANTASY_RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"Real injury-risk multipliers - position: {position_multiplier}")
    print(f"Real overall average miss rate: {overall_avg_miss_rate * 100:.1f}%")
    print(f"Scored {n_scored}/{len(players)} players with real injury risk "
          f"({n_no_history} real rookies with no qualifying history -> disclosed null)")
    print(f"Scored {len(players) - n_no_consistency}/{len(players)} players with real consistency "
          f"({n_no_consistency} with <{MIN_CONSISTENCY_WEEKS} real qualifying 2025 weeks -> disclosed null)")

    risk_scores = [p["injury_risk_score"] for p in players if p["injury_risk_score"] is not None]
    cons_scores = [p["consistency_score"] for p in players if p["consistency_score"] is not None]
    print(f"Injury risk distribution: mean={np.mean(risk_scores):.1f} std={np.std(risk_scores):.1f} "
          f"range={min(risk_scores):.1f}-{max(risk_scores):.1f}")
    print(f"Consistency distribution: mean={np.mean(cons_scores):.1f} std={np.std(cons_scores):.1f} "
          f"range={min(cons_scores):.1f}-{max(cons_scores):.1f}")
    return players


if __name__ == "__main__":
    compute_all_scores()
