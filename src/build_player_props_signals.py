"""Engineers real, leak-free features for player props models (individual
stat projections conditioned on opponent D_Elo).

Real, serious problems found and fixed in the originally pasted spec before
writing this:

1. Assumed `data/processed/team_defense_stats.csv` with columns like
   `{position}_rec_yards_allowed` - doesn't exist anywhere in this project
   (checked). No yards/TDs-allowed-by-position table exists at all; the
   closest real thing (EPA-allowed-per-position from play-by-play, via
   matchup_features.py) is a different signal than what this task asks for.
   This task's own stated foundation is opponent D_Elo (real, from this
   session's O/D Elo split) - used directly as the real defense-quality
   signal, no fabricated yards-allowed table.
2. Assumed `data/processed/elo_by_week_2015_2025.csv` - doesn't exist (same
   fabricated filename already flagged twice this session). Real file is
   `data/processed/team_elo_history_offensive_defensive_2015_2025.csv`, a
   per-GAME (home/away) table, not a per-team-week lookup - reshaped to
   long format here (_real_opponent_od_elo_long) since no existing function
   already exports that exact shape.
3. The spec's per-row `.apply(..., axis=1)` / nested-filter-per-row pattern
   for career averages (re-filtering the whole player history table once
   per row) is real, needless O(n*m) - the same anti-pattern already caught
   and fixed once this session (backtest_offensive_defensive_elo.py).
   Replaced with a vectorized, leak-free `groupby().expanding().mean().
   shift(1)` - excludes the current row by construction, same real
   discipline this project's Elo carryover already uses.
4. Vegas total / fantasy-PPR-projection features were dropped (the spec
   used a `vegas_total = 45.0` placeholder and a projected-PPR feature
   that doesn't exist as a real leak-free signal for 2015-2025 training
   rows - using a stat DERIVED FROM these same target columns as a
   feature would be a real, if subtle, leakage risk, and no real per-game
   Vegas total exists consistently across 2015-2025 to begin with).

Real weather/rest addition (Quick Wins task): the pasted spec for that
task assumed `data/nfl_game_weather_2015_2025.csv` and, when missing,
fell back to a FABRICATED placeholder - literally `temperature=72.0,
wind_speed=5.0, precipitation=0.0` for every single game, presented as if
real. Checked first: data/raw/schedules_2015_2025.csv already has real
`roof`, `temp`, `wind`, `home_rest`, `away_rest` columns natively (real
nflverse data, no collection script needed at all). But `temp`/`wind` are
real for only ~63% of outdoor games (dome/closed/open games are
genuinely null - weather doesn't apply - and 6% of real outdoor games are
missing) AND, more importantly, aren't knowable in advance for a future
game - a 2026 preseason player-props projection can't use a temperature
that doesn't exist yet. Real fix: only `roof` (stadium-fixed, known
months in advance) and each team's own real rest days (schedule-
determined, also knowable in advance) are used as real features here -
both are genuinely available for 2026 scoring, unlike temp/wind, which
were left out of the shipped model rather than faked."""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OD_ELO_HISTORY_PATH = os.path.join(PROCESSED_DIR, "team_elo_history_offensive_defensive_2015_2025.csv")
SCHEDULE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "schedules_2015_2025.csv")

POSITIONS = ["QB", "RB", "WR", "TE"]

# Real per-position raw stat columns whose leak-free career average becomes
# a feature - only stats that actually exist for that position's real role.
CAREER_AVG_SOURCE_COLS = {
    "QB": ["completions", "attempts", "passing_yards", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "rushing_yards"],
    "TE": ["targets", "receptions", "receiving_yards", "rushing_yards"],
}

# Real per-position TD-rate columns (Major Refinements task, TD logistic
# models) - kept separate from CAREER_AVG_SOURCE_COLS so the existing,
# already-validated yardage/reception linear models are untouched. A
# player's real career TD RATE (not just volume) is the natural leak-free
# predictor for "will they score 1+ TD this game" that the original
# yardage-focused feature set didn't include.
TD_CAREER_AVG_COLS = {
    "QB": ["passing_tds", "rushing_tds"],
    "RB": ["rushing_tds"],
    "WR": ["receiving_tds"],
    "TE": ["receiving_tds"],
}

# Real target stat columns per position (this task's requested props).
TARGET_COLS = {
    "QB": ["completions", "passing_yards", "passing_tds", "rushing_tds"],
    "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards"],
    "WR": ["receptions", "receiving_yards", "receiving_tds", "rushing_yards"],
    "TE": ["receptions", "receiving_yards", "receiving_tds", "rushing_yards"],
}


def _real_opponent_od_elo_long():
    """Reshapes the real per-game O/D Elo history into a long (team,
    season, week) -> real pre-game opponent O/D Elo + real home/away flag
    lookup - two rows per game (home perspective, away perspective).
    Deliberately does NOT carry an opponent-team column - the real target
    dataframe (player_weekly_stats.csv) already has its own real
    `opponent_team` column, so this lookup only needs to add opp_o_elo/
    opp_d_elo/is_home without creating a colliding duplicate on merge."""
    games = pd.read_csv(OD_ELO_HISTORY_PATH)
    home_rows = games[["season", "week", "home_team", "away_o_elo_before", "away_d_elo_before"]].rename(
        columns={"home_team": "team", "away_o_elo_before": "opp_o_elo", "away_d_elo_before": "opp_d_elo"})
    home_rows["is_home"] = True
    away_rows = games[["season", "week", "away_team", "home_o_elo_before", "home_d_elo_before"]].rename(
        columns={"away_team": "team", "home_o_elo_before": "opp_o_elo", "home_d_elo_before": "opp_d_elo"})
    away_rows["is_home"] = False
    return pd.concat([home_rows, away_rows], ignore_index=True)


def _real_roof_and_rest_long():
    """Reshapes real schedule rows into a long (team, season, week) ->
    is_dome/own_rest_days lookup. Both are genuinely knowable in advance
    (roof is stadium-fixed; rest days are schedule-determined) - unlike
    temp/wind, which are excluded here (see module docstring)."""
    games = pd.read_csv(SCHEDULE_PATH)
    games = games[games["game_type"] == "REG"]
    games["is_dome"] = games["roof"].isin(["dome", "closed"]).astype(int)
    home_rows = games[["season", "week", "home_team", "is_dome", "home_rest"]].rename(
        columns={"home_team": "team", "home_rest": "own_rest_days"})
    away_rows = games[["season", "week", "away_team", "is_dome", "away_rest"]].rename(
        columns={"away_team": "team", "away_rest": "own_rest_days"})
    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_player_props_signals():
    print("\nBuilding real, leak-free player props features (2015-2025)...\n")
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[(stats["season_type"] == "REG") & (stats["position"].isin(POSITIONS))].copy()
    stats = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    opp_lookup = _real_opponent_od_elo_long()
    stats = stats.merge(
        opp_lookup, left_on=["recent_team", "season", "week"], right_on=["team", "season", "week"], how="inner")

    roof_rest_lookup = _real_roof_and_rest_long()
    stats = stats.merge(
        roof_rest_lookup, left_on=["recent_team", "season", "week"], right_on=["team", "season", "week"],
        how="inner", suffixes=("", "_roofrest"))

    week_min, week_max = stats["week"].min(), stats["week"].max()
    stats["week_norm"] = (stats["week"] - week_min) / (week_max - week_min)

    for position in POSITIONS:
        print(f"Building features for {position}...")
        pos_stats = stats[stats["position"] == position].copy()

        # Real, leak-free, vectorized career averages: expanding mean of
        # each player's own real PRIOR games only (shift(1) excludes the
        # current row - no information from the game being predicted ever
        # enters its own feature).
        for col in CAREER_AVG_SOURCE_COLS[position]:
            pos_stats[f"career_avg_{col}"] = (
                pos_stats.groupby("player_id")[col].transform(lambda s: s.expanding().mean().shift(1))
            )
        for col in TD_CAREER_AVG_COLS[position]:
            pos_stats[f"career_avg_{col}"] = (
                pos_stats.groupby("player_id")[col].transform(lambda s: s.expanding().mean().shift(1))
            )

        feature_cols = [f"career_avg_{c}" for c in CAREER_AVG_SOURCE_COLS[position]]
        td_feature_cols = [f"career_avg_{c}" for c in TD_CAREER_AVG_COLS[position]]
        # A player's first-ever real game has no real prior history to
        # average - dropped (same real, disclosed rookie-exclusion
        # precedent FantasyRankings.js already uses), not filled with an
        # invented rate.
        pos_stats = pos_stats.dropna(subset=feature_cols + td_feature_cols)

        # Real binary "1+ TD" targets for the new logistic models.
        for col in TD_CAREER_AVG_COLS[position]:
            pos_stats[f"actual_{col}_1plus"] = (pos_stats[col] >= 1).astype(int)
        td_target_cols = [f"actual_{c}_1plus" for c in TD_CAREER_AVG_COLS[position]]

        keep_cols = (
            ["player_id", "player_name", "position", "season", "week", "recent_team", "opponent_team",
             "is_home", "week_norm", "opp_o_elo", "opp_d_elo", "is_dome", "own_rest_days"]
            + feature_cols
            + td_feature_cols
            + TARGET_COLS[position]
            + td_target_cols
        )
        out = pos_stats[keep_cols].rename(columns={f"{t}": f"actual_{t}" for t in TARGET_COLS[position]})
        out_path = os.path.join(PROCESSED_DIR, f"player_props_signals_{position}.csv")
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {len(out)} real player-games -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    build_player_props_signals()
