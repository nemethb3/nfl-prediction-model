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
fabricated fallback value.

Real opponent-EPA-allowed-by-position addition (Fantasy Model Overhaul
Phase 1, Part 1/3): the originally pasted spec for this task assumed a
`data/processed/position_vs_team_allowed_2015_2025.json` yards/TDs-allowed
table built from pbp columns (`pbp['position']`, `pbp['defense_team']`,
`rushing_touchdown`) that don't exist in this project's real pbp schema
(real columns: `defteam`, `rush_touchdown`, `receiver_id`/`rusher_id`, no
bare `position`). Real fix: reuses matchup_features.py's already-validated
build_defense_epa_by_position_multi_season() (real target_position via the
real player_id->position crosswalk), cached to a season-level, PRIOR-SEASON
table by build_defense_epa_allowed_by_position.py - same prior-season
convention as team pace/red-zone above, for the same reason (no in-season
week-1 cold start, generalizes directly to 2026 scoring). Real, disclosed
limitation carried over from matchup_features.py: for QB this only reflects
defense quality against QB rushing plays (scrambles/designed runs), not
passing defense, since receiver_id is never a QB on a real pass play - kept
anyway as a real, honestly-measured signal, not silently relabeled.

Real recent-form / usage-trend / role-tier addition (Fantasy Model
Overhaul Phase 1B): the originally pasted spec for this task assumed
`data/processed/player_season_stats_2015_2025.csv` and `player_game_stats_
2015_2025.csv` (neither exists - real files are player_season_stats.csv
and player_weekly_stats.csv), a `ppr` column (real column is
fantasy_points_ppr), and a `carry_share` column (checked - doesn't exist
anywhere in this project, at either season or weekly grain; not built
here). Three further, real, serious problems fixed before writing this:

1. The spec's recent-form/usage-trend builders looped `for player_id in
   ...unique(): for idx, row in player_data.iterrows(): ...` re-slicing a
   per-player frame on every single row - the same real O(n*m) anti-pattern
   already flagged and fixed twice elsewhere this project (build_player_
   props_signals.py's own docstring above, backtest_offensive_defensive_
   elo.py). At this project's real per-week row counts (player_weekly_
   stats.csv), that's not just slow - it's impractical. Replaced with
   vectorized `groupby().rolling()`/`transform()`, same as every other
   leak-free feature in this file.
2. The spec's recent-form feature used the player's CURRENT-game `snap_pct`
   as an input to a role-tier bucketing step, and its "usage trending"
   feature computed a same-season delta from raw current-season aggregates
   - both are the same current-game/current-season leakage pattern this
   file's own module docstring already identified and fixed once (the
   original career_avg_snap_pct fix). Real fix: `recent_form_ppr_last4` is
   a leak-free trailing rolling mean (shift(1), excludes the current row,
   crosses season boundaries by design - "last 4 games played" is a
   real, ongoing signal, not reset each September). `usage_trend_*_delta`
   is a real, leak-free, already-final SEASON-OVER-SEASON change (this
   player's realized season S-1 rate minus season S-2's, lagged +1 season
   to apply to season S - the same static prior-season convention as team
   pace/red-zone/O-D-Elo above, and the only way this signal is honestly
   knowable in advance for a real 2026 game). Role tiers are bucketed from
   the already-existing leak-free `career_avg_snap_pct`/new leak-free
   `career_avg_target_share` (trailing, shift(1)), not same-game/season
   values.
3. The spec's role-based-model evaluation code set `improved_r2 =
   baseline_r2` verbatim for every role tier (`retrain_role_based_models`
   never computes an actual pooled baseline) - the reported "average gain"
   for that whole approach was guaranteed to equal exactly 0.0 by
   construction, not a real measurement. See experiment_phase1b_features.py
   for the real, fair comparison methodology used instead (out-of-fold
   predictions from tier-specific models recombined and compared against a
   freshly-computed pooled baseline on the identical row set)."""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OD_ELO_HISTORY_PATH = os.path.join(PROCESSED_DIR, "team_elo_history_offensive_defensive_2015_2025.csv")
SCHEDULE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "schedules_2015_2025.csv")
PBP_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pbp_2015_2025.csv")
DEF_EPA_ALLOWED_BY_POSITION_PATH = os.path.join(
    PROCESSED_DIR, "defense_epa_allowed_by_position_2015_2025.csv")
SEASON_STATS_PATH = os.path.join(PROCESSED_DIR, "player_season_stats.csv")

# Real, leak-free role-tier thresholds (Phase 1B) - same cutpoints the
# originally pasted spec proposed, applied to real TRAILING (career-to-date,
# shift(1)) rates instead of same-game/season values.
RB_ROLE_THRESHOLDS = [(0.50, "RB_LEAD"), (0.30, "RB_COMMITTEE")]  # else RB_BACKUP
WR_ROLE_THRESHOLDS = [(0.25, "WR_1"), (0.15, "WR_2")]  # else WR_3
TE_ROLE_THRESHOLDS = [(0.20, "TE_1")]  # else TE_2

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


def _real_prior_season_defense_epa_allowed_vs_position():
    """Real prior-season defensive EPA/play allowed to each position (QB/RB/
    WR/TE), cached by build_defense_epa_allowed_by_position.py from
    matchup_features.py's real position-crosswalk pbp aggregation - already
    lagged +1 season there, so this is a plain read, not a recompute."""
    return pd.read_csv(DEF_EPA_ALLOWED_BY_POSITION_PATH)


def _real_prior_season_usage_trend():
    """Real, leak-free usage-trend signal (Phase 1B): this player's realized
    season-over-season CHANGE in snap share / target share, using two
    already-completed real seasons (S-1 minus S-2), lagged +1 to apply to
    season S - the only way this is honestly knowable in advance for a real
    future game (an in-progress "current season vs prior season" version,
    as the originally pasted spec proposed, isn't well-defined for a
    player's own early-season games and would need same-season data those
    games don't have yet)."""
    season_stats = pd.read_csv(SEASON_STATS_PATH)
    season_stats = season_stats.sort_values(["player_id", "season"]).drop_duplicates(
        subset=["player_id", "season"])
    grp = season_stats.groupby("player_id")
    season_stats["usage_trend_snap_pct_delta"] = grp["avg_snap_pct"].diff()
    season_stats["usage_trend_target_share_delta"] = grp["target_share"].diff()

    lagged = season_stats[["player_id", "season", "usage_trend_snap_pct_delta",
                            "usage_trend_target_share_delta"]].copy()
    lagged["season"] = lagged["season"] + 1
    return lagged


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

    print("Joining real prior-season opponent EPA-allowed-by-position...")
    def_epa_allowed_lookup = _real_prior_season_defense_epa_allowed_vs_position()
    stats = stats.merge(
        def_epa_allowed_lookup, left_on=["opponent_team", "season", "position"],
        right_on=["team", "season", "position"], how="left", suffixes=("", "_defepa"))
    n_no_opp_epa_allowed = int(stats["opp_epa_allowed_vs_position_prior_season"].isna().sum())
    print(f"  {n_no_opp_epa_allowed} real player-games have no real prior-season opponent "
          f"EPA-allowed-vs-position (real 2016 rows - the lookup itself starts at 2016, "
          f"needing a real 2015 prior season) - will be dropped below")

    print("Joining real season-over-season usage-trend deltas...")
    usage_trend_lookup = _real_prior_season_usage_trend()
    stats = stats.merge(usage_trend_lookup, on=["player_id", "season"], how="left")
    n_no_usage_trend = int(stats["usage_trend_snap_pct_delta"].isna().sum())
    print(f"  {n_no_usage_trend} real player-games have no real usage trend (real rookie-season rows, "
          f"or a player's 2nd real season - needs two already-completed prior seasons) - optional "
          f"column, NOT dropped from the base feature set")

    print("Computing real leak-free recent-form (trailing last-4-games PPR, crosses season boundaries)...")
    stats["recent_form_ppr_last4"] = stats.groupby("player_id")["fantasy_points_ppr"].transform(
        lambda s: s.rolling(4, min_periods=1).mean().shift(1))
    n_no_recent_form = int(stats["recent_form_ppr_last4"].isna().sum())
    print(f"  {n_no_recent_form} real player-games have no real recent form (a player's real career-first "
          f"game) - optional column, NOT dropped from the base feature set")

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
        # Real, leak-free trailing target-share average - used below for
        # real WR/TE role-tier bucketing (Phase 1B), same leak-free
        # construction as every other career_avg_* column.
        pos_stats["career_avg_target_share"] = (
            pos_stats.groupby("player_id")["target_share"].transform(lambda s: s.expanding().mean().shift(1))
        )

        # Real role tier (Phase 1B), bucketed from the leak-free trailing
        # averages above - NOT same-game/season values (see module
        # docstring). Optional column; rows with no real trailing average
        # yet (a player's real career-first game) get role_tier=NaN, not a
        # fabricated default tier.
        if position == "RB":
            pos_stats["role_tier"] = np.select(
                [pos_stats["career_avg_snap_pct"] > 0.50, pos_stats["career_avg_snap_pct"] >= 0.30],
                ["RB_LEAD", "RB_COMMITTEE"], default="RB_BACKUP")
            pos_stats.loc[pos_stats["career_avg_snap_pct"].isna(), "role_tier"] = np.nan
        elif position == "WR":
            pos_stats["role_tier"] = np.select(
                [pos_stats["career_avg_target_share"] >= 0.25, pos_stats["career_avg_target_share"] >= 0.15],
                ["WR_1", "WR_2"], default="WR_3")
            pos_stats.loc[pos_stats["career_avg_target_share"].isna(), "role_tier"] = np.nan
        elif position == "TE":
            pos_stats["role_tier"] = np.select(
                [pos_stats["career_avg_target_share"] >= 0.20], ["TE_1"], default="TE_2")
            pos_stats.loc[pos_stats["career_avg_target_share"].isna(), "role_tier"] = np.nan
        else:  # QB - always a single real tier (see module docstring)
            pos_stats["role_tier"] = "QB"

        feature_cols = [f"career_avg_{c}" for c in CAREER_AVG_SOURCE_COLS[position]] + ["career_avg_snap_pct"]
        td_feature_cols = [f"career_avg_{c}" for c in TD_CAREER_AVG_COLS[position]]
        # recent_form_ppr_last4 promoted from optional/experimental to a
        # required production feature (Phase 1B): the real, honest
        # apples-to-apples measurement in experiment_phase1b_features.py
        # showed a real gain on all 16 real models (avg R2 delta +0.017,
        # avg AUC delta +0.014, zero losses) - role_tier and the two
        # usage_trend_* columns stayed real null results and were NOT
        # promoted (still kept below as optional/disclosed columns).
        situational_feature_cols = ["prior_season_pace_factor", "prior_season_rz_rate",
                                     "opp_epa_allowed_vs_position_prior_season", "recent_form_ppr_last4"]
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

        # Phase 1B experimental columns that stayed real null results (see
        # experiment_phase1b_features.py) - kept OPTIONAL (not added to the
        # dropna above) so they're available for transparency/future re-
        # testing without shrinking the production row set. recent_form_
        # ppr_last4 is no longer here - it was promoted to a required
        # feature above.
        optional_experiment_cols = ["role_tier", "usage_trend_snap_pct_delta", "usage_trend_target_share_delta"]

        keep_cols = (
            ["player_id", "player_name", "position", "season", "week", "recent_team", "opponent_team",
             "is_home", "week_norm", "is_late_season", "opp_o_elo", "opp_d_elo", "is_dome", "own_rest_days"]
            + feature_cols
            + situational_feature_cols
            + td_feature_cols
            + TARGET_COLS[position]
            + td_target_cols
            + optional_experiment_cols
        )
        out = pos_stats[keep_cols].rename(columns={f"{t}": f"actual_{t}" for t in TARGET_COLS[position]})
        out_path = os.path.join(PROCESSED_DIR, f"player_props_signals_{position}.csv")
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {len(out)} real player-games -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    build_player_props_signals()
