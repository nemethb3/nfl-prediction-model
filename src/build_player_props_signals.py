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
were left out of the shipped model rather than faked.

Real pace/snap/red-zone addition (Player Props Enrichment task): the
pasted spec for this task assumed `data/nfl_schedules_2015_2025.csv`
(wrong path - real file is data/raw/schedules_2015_2025.csv, the same
fabricated-prefix error already flagged for schedules elsewhere this
session) and a `plays` column on player_weekly_stats.csv that doesn't
exist. It also proposed summing PLAYER-level snap/play counts grouped by
team-season as "team pace" - wrong unit: 11 players are on the field for
every real play, so summing player-level rows overcounts a team's real
play volume by roughly 11x and was never a real pace metric. Real fix:
team pace and red-zone-touch rate are computed directly from real
play-by-play (data/raw/pbp_2015_2025.csv - one row per real play,
`posteam`/`play_type`/`yardline_100`/`rusher_player_id`/
`receiver_player_id`), counted once per real offensive snap
(play_type in {pass, run}), not by re-aggregating already-aggregated
player rows.

The spec's own `snap_pct` feature and its playoff-month factor were also
both leaky as written: the spec joined a player's CURRENT-game snap_pct
(and a playoff-month multiplier computed from the SAME season's own
games) into the model that predicts that same current game's stat -
snap_pct is already known to correlate mechanically with the target
(more snaps -> more yards, by construction), and the playoff factor used
future/same-season outcomes to predict earlier games in that season. Real
fix, matching this file's own already-established leak-free convention:
`career_avg_snap_pct` is a leak-free trailing expanding average (shift(1),
same pattern as every other career_avg_* column - excludes the current
row). Team pace/red-zone rate use the real PRIOR SEASON's rate (not the
current, still-in-progress season), the same "static prior-season number"
convention this project already uses for 2026 preseason O/D Elo
(team_elo_offensive_defensive_2026_regressed.json) - this also means the
exact same real prior-season lookup generalizes cleanly to 2026 scoring
(use real 2025 rates) with no separate logic. The spec's leaky
playoff_performance_factor was replaced with a plain `is_late_season`
(week>=14) calendar flag - real, always knowable in advance, no leakage
risk, and lets the linear model estimate any real Dec/Jan effect via its
own coefficient rather than injecting a pre-computed, same-season-derived
multiplier.

Real, checked coverage note: team pace/red-zone rate require a real prior
season in pbp_2015_2025.csv, so real 2015 rows (no 2014 pbp data exists
in this project) are dropped - a real, disclosed one-season gap, not a
fabricated fallback value."""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OD_ELO_HISTORY_PATH = os.path.join(PROCESSED_DIR, "team_elo_history_offensive_defensive_2015_2025.csv")
SCHEDULE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "schedules_2015_2025.csv")
PBP_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pbp_2015_2025.csv")

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


def _real_prior_season_team_pace_and_rz():
    """Real team-season pace (offensive plays/game, normalized to that
    season's real league average) and red-zone touch rate (real plays with
    yardline_100<=20 carried by a real rusher or targeted receiver),
    counted once per real play from play-by-play - not by re-summing
    already-aggregated player rows (see module docstring). Returned as a
    PRIOR-SEASON lookup (season shifted +1 per team) so it's leak-free by
    construction and generalizes directly to 2026 scoring (real 2025
    rates)."""
    pbp = pd.read_csv(PBP_PATH, usecols=[
        "season", "game_id", "posteam", "play_type", "yardline_100", "rusher_player_id", "receiver_player_id"])
    snaps = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]

    team_game_plays = snaps.groupby(["season", "game_id", "posteam"]).size().reset_index(name="plays")
    team_season = team_game_plays.groupby(["season", "posteam"])["plays"].mean().reset_index().rename(
        columns={"posteam": "team", "plays": "plays_per_game"})
    league_avg_by_season = team_season.groupby("season")["plays_per_game"].transform("mean")
    team_season["pace_factor"] = team_season["plays_per_game"] / league_avg_by_season

    rz = snaps[(snaps["yardline_100"] <= 20) & (snaps["rusher_player_id"].notna() | snaps["receiver_player_id"].notna())]
    team_game_rz = rz.groupby(["season", "game_id", "posteam"]).size().reset_index(name="rz_touches")
    rz_season = team_game_rz.groupby(["season", "posteam"])["rz_touches"].mean().reset_index().rename(
        columns={"posteam": "team", "rz_touches": "rz_touches_per_game"})

    team_season = team_season.merge(rz_season, on=["season", "team"], how="left")
    team_season["rz_touches_per_game"] = team_season["rz_touches_per_game"].fillna(0.0)

    lagged = team_season[["season", "team", "pace_factor", "rz_touches_per_game"]].copy()
    lagged["season"] = lagged["season"] + 1
    return lagged.rename(columns={
        "pace_factor": "prior_season_pace_factor", "rz_touches_per_game": "prior_season_rz_rate"})


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

    print("Computing real prior-season team pace/red-zone rates from play-by-play...")
    pace_rz_lookup = _real_prior_season_team_pace_and_rz()
    stats = stats.merge(
        pace_rz_lookup, left_on=["recent_team", "season"], right_on=["team", "season"],
        how="left", suffixes=("", "_pacerz"))
    n_no_prior_season = int(stats["prior_season_pace_factor"].isna().sum())
    print(f"  {n_no_prior_season} real player-games have no real prior-season pace/RZ rate "
          f"(real 2015 rows - no 2014 pbp data exists in this project) - will be dropped below")

    week_min, week_max = stats["week"].min(), stats["week"].max()
    stats["week_norm"] = (stats["week"] - week_min) / (week_max - week_min)
    stats["is_late_season"] = (stats["week"] >= 14).astype(int)

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
        # Real, leak-free trailing snap-share average - the player's own
        # CURRENT-game snap_pct was excluded (see module docstring: using
        # a game's own snap share to predict that same game's stats is
        # leakage by construction).
        pos_stats["career_avg_snap_pct"] = (
            pos_stats.groupby("player_id")["snap_pct"].transform(lambda s: s.expanding().mean().shift(1))
        )

        feature_cols = [f"career_avg_{c}" for c in CAREER_AVG_SOURCE_COLS[position]] + ["career_avg_snap_pct"]
        td_feature_cols = [f"career_avg_{c}" for c in TD_CAREER_AVG_COLS[position]]
        situational_feature_cols = ["prior_season_pace_factor", "prior_season_rz_rate"]
        # A player's first-ever real game has no real prior history to
        # average - dropped (same real, disclosed rookie-exclusion
        # precedent FantasyRankings.js already uses), not filled with an
        # invented rate. Real 2015 rows (no real prior-season pace/RZ rate
        # exists) are dropped for the same real reason.
        pos_stats = pos_stats.dropna(subset=feature_cols + td_feature_cols + situational_feature_cols)

        # Real binary "1+ TD" targets for the new logistic models.
        for col in TD_CAREER_AVG_COLS[position]:
            pos_stats[f"actual_{col}_1plus"] = (pos_stats[col] >= 1).astype(int)
        td_target_cols = [f"actual_{c}_1plus" for c in TD_CAREER_AVG_COLS[position]]

        keep_cols = (
            ["player_id", "player_name", "position", "season", "week", "recent_team", "opponent_team",
             "is_home", "week_norm", "is_late_season", "opp_o_elo", "opp_d_elo", "is_dome", "own_rest_days"]
            + feature_cols
            + situational_feature_cols
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
