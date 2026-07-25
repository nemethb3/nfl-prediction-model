"""Raw data collection and cleaning for the NFL win prediction model.

Downloads play-by-play, snap count, weekly/seasonal player stats,
PFR defensive stats, and player metadata via nfl_data_py, writes them
to data/raw/, then standardizes names/positions and joins them into
analysis-ready tables in data/processed/.

Training window: 2015-2025 seasons (used to project the 2026 season).
"""

import os
import urllib.error

import nfl_data_py as nfl
import numpy as np
import pandas as pd

YEARS = list(range(2015, 2026))  # 2015-2025 seasons (2025 season completed Feb 2026)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _fetch_with_fallback(import_fn, years, label, **kwargs):
    """Try a bulk multi-year fetch; if a year's file isn't published yet upstream
    (404), fall back to fetching year-by-year and skip only the missing ones."""
    try:
        return import_fn(years, **kwargs)
    except urllib.error.HTTPError:
        print(f"[{label}] bulk fetch failed (some year not yet published); "
              f"retrying year-by-year")
        frames = []
        missing = []
        for y in years:
            try:
                frames.append(import_fn([y], **kwargs))
            except urllib.error.HTTPError:
                missing.append(y)
        if missing:
            print(f"[{label}] WARNING: no data available upstream for years {missing}; "
                  f"skipping and continuing with {[y for y in years if y not in missing]}")
        return pd.concat(frames, ignore_index=True)


def download_pbp_data(years=YEARS):
    """Download play-by-play data for the given years.

    Fetches and downcasts one year at a time before concatenating, since
    nfl_data_py's own import_pbp_data concats all years at float64 *then*
    downcasts, which roughly doubles peak memory versus doing it per-year -
    that peak was enough to trigger a MemoryError on this machine.

    Returns a DataFrame with per-play detail (posteam, defteam, epa,
    yards_gained, touchdown, sack, interception, week, season, etc.).
    """
    frames = []
    for year in years:
        yr_df = nfl.import_pbp_data([year], downcast=True)
        frames.append(yr_df)
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[pbp] {df.shape[0]:,} rows x {df.shape[1]} cols | seasons "
          f"{int(df['season'].min())}-{int(df['season'].max())}")
    return df


def download_snap_counts(years=YEARS):
    """Download weekly snap count data (offense/defense/special teams)."""
    df = nfl.import_snap_counts(years)
    print(f"[snap_counts] {df.shape[0]:,} rows x {df.shape[1]} cols | seasons "
          f"{int(df['season'].min())}-{int(df['season'].max())}")
    return df


def download_player_stats(years=YEARS):
    """Download weekly and seasonal offensive player stats (pass/rush/rec + EPA)."""
    weekly = _fetch_with_fallback(nfl.import_weekly_data, years, "weekly_stats")
    seasonal = _fetch_with_fallback(
        nfl.import_seasonal_data, years, "seasonal_stats", s_type="REG"
    )
    print(f"[weekly_stats] {weekly.shape[0]:,} rows x {weekly.shape[1]} cols")
    print(f"[seasonal_stats] {seasonal.shape[0]:,} rows x {seasonal.shape[1]} cols")
    return weekly, seasonal


def download_defensive_stats(years=YEARS):
    """Download PFR advanced defensive stats (sacks, tackles, pressures, age).

    nfl_data_py's PFR seasonal data only goes back to 2018, so earlier
    years are dropped here regardless of what's passed in.
    """
    pfr_years = [y for y in years if y >= 2018]
    if len(pfr_years) < len(years):
        print(f"[defense_pfr] PFR data unavailable before 2018; using {pfr_years}")
    df = _fetch_with_fallback(
        lambda yrs, **kw: nfl.import_seasonal_pfr("def", years=yrs, **kw), pfr_years, "defense_pfr"
    )
    print(f"[defense_pfr] {df.shape[0]:,} rows x {df.shape[1]} cols | seasons "
          f"{int(df['season'].min())}-{int(df['season'].max())}")
    return df


def download_player_metadata():
    """Download player biographical/draft metadata (birth date, draft capital, position)."""
    df = nfl.import_players()
    print(f"[players] {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def download_schedules(years=YEARS):
    """Download schedules/results, which also carry nflverse's Vegas closing lines
    (spread_line, total_line, moneylines) - used again in Task 1.4."""
    df = nfl.import_schedules(years)
    print(f"[schedules] {df.shape[0]:,} rows x {df.shape[1]} cols | seasons "
          f"{int(df['season'].min())}-{int(df['season'].max())}")
    return df


def validate_downloaded_data(pbp, snaps, weekly, seasonal, defense, players, schedules):
    """Run sanity checks on all downloaded raw data and print a summary."""
    print("\n===== VALIDATION SUMMARY =====")

    # PBP
    assert pbp.shape[0] > 400_000, f"PBP row count too low: {pbp.shape[0]}"
    years_present = sorted(pbp["season"].unique())
    print(f"PBP: {pbp.shape[0]:,} rows | seasons present: {years_present}")
    epa_null_pct = pbp["epa"].isna().mean() * 100
    print(f"PBP epa null rate: {epa_null_pct:.1f}%")

    # Snap counts
    print(f"Snap counts: {snaps.shape[0]:,} rows | "
          f"seasons present: {sorted(snaps['season'].unique())}")

    # Weekly / seasonal stats
    print(f"Weekly player stats: {weekly.shape[0]:,} rows | "
          f"seasons present: {sorted(weekly['season'].unique())}")
    print(f"Seasonal player stats: {seasonal.shape[0]:,} rows | "
          f"seasons present: {sorted(seasonal['season'].unique())}")
    # seasonal/weekly stats only carry player_id (== players.gsis_id); join for name/position
    seasonal_named = seasonal.merge(
        players[["gsis_id", "display_name", "position"]],
        left_on="player_id", right_on="gsis_id", how="left",
    )
    latest_season = int(seasonal_named["season"].max())
    top_qb = (
        seasonal_named[(seasonal_named["season"] == latest_season) & (seasonal_named["position"] == "QB")]
        .nlargest(5, "passing_yards")[["display_name", "passing_yards"]]
    )
    print(f"Top 5 QBs by passing yards, {latest_season} season:")
    print(top_qb.to_string(index=False))

    # Defense
    print(f"Defensive (PFR) stats: {defense.shape[0]:,} rows")

    # Players / metadata
    null_birth = players["birth_date"].isna().mean() * 100
    print(f"Player metadata: {players.shape[0]:,} rows | birth_date null rate: {null_birth:.1f}%")

    # Schedules
    completed_games = schedules[schedules["home_score"].notna()]
    print(f"Schedules: {schedules.shape[0]:,} games | {completed_games.shape[0]:,} completed | "
          f"seasons present: {sorted(schedules['season'].unique())}")

    print("===== END VALIDATION =====\n")


def save_raw_data(pbp, snaps, weekly, seasonal, defense, players, schedules):
    """Persist all raw DataFrames to data/raw/ as UTF-8 CSVs."""
    os.makedirs(RAW_DIR, exist_ok=True)
    outputs = {
        "pbp_2015_2025.csv": pbp,
        "snap_counts_2015_2025.csv": snaps,
        "player_weekly_raw_2015_2025.csv": weekly,
        "player_seasonal_raw_2015_2025.csv": seasonal,
        "defense_pfr_2015_2025.csv": defense,
        "player_metadata.csv": players,
        "schedules_2015_2025.csv": schedules,
    }
    for filename, df in outputs.items():
        path = os.path.join(RAW_DIR, filename)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {path} ({df.shape[0]:,} rows)")


def raw_data_present():
    """True if a prior download already populated data/raw/."""
    return os.path.exists(os.path.join(RAW_DIR, "pbp_2015_2025.csv"))


def run_download_pipeline():
    """Download all raw sources, validate, and save to data/raw/."""
    pbp = download_pbp_data()
    snaps = download_snap_counts()
    weekly, seasonal = download_player_stats()
    defense = download_defensive_stats()
    players = download_player_metadata()
    schedules = download_schedules()

    validate_downloaded_data(pbp, snaps, weekly, seasonal, defense, players, schedules)
    save_raw_data(pbp, snaps, weekly, seasonal, defense, players, schedules)


# ---------------------------------------------------------------------------
# Task 1.3: cleaning & standardization
# ---------------------------------------------------------------------------

# Raw sources use inconsistent position vocabularies (nflverse weekly/seasonal
# stats, PFR snap counts, PFR advanced defense, and the players table all
# differ). Map every source down to one canonical set. Combo codes like
# "DE-DT" or "CB/RCB" (a player who logged snaps at both) are handled by
# splitting on "/" or "-" and mapping only the primary (first) position.
POSITION_MAP = {
    "QB": "QB",
    "RB": "RB", "HB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "C": "OL", "G": "OL", "OG": "OL", "OT": "OL", "T": "OL", "OL": "OL",
    "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
    "DE": "EDGE", "OLB": "EDGE", "LDE": "EDGE", "RDE": "EDGE",
    "LOLB": "EDGE", "ROLB": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL", "LDT": "DL", "RDT": "DL",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "LLB": "LB", "RLB": "LB",
    "LILB": "LB", "RILB": "LB",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "S": "S", "FS": "S", "SS": "S", "SAF": "S",
    "DB": "DB",
    "K": "K",
    "P": "P",
    "LS": "LS",
}


def standardize_positions(position_series):
    """Map a raw position column to the canonical position set above.

    Takes the primary position out of combo codes (e.g. "DE-DT" -> "DE"),
    then maps it. Unrecognized codes are left as their primary token and
    reported so they can be triaged rather than silently dropped.
    """
    null_mask = position_series.isna()
    primary = position_series.astype(str).str.upper().str.strip().str.split(r"[/\-]").str[0]
    primary = primary.where(~null_mask)
    mapped = primary.map(POSITION_MAP)
    unmapped_mask = mapped.isna() & ~null_mask
    unmapped = sorted(primary[unmapped_mask].unique())
    if unmapped:
        print(f"[standardize_positions] WARNING: unmapped codes left as-is: {unmapped}")
    return mapped.where(~unmapped_mask, primary).where(~null_mask)


def build_player_crosswalk(players):
    """Build the canonical per-player reference table.

    gsis_id (renamed player_id) is the join key used by nflverse's
    weekly/seasonal stats; pfr_id is the join key used by snap counts and
    PFR defensive stats. Having both on one row lets every other table
    join to a single canonical player_id.
    """
    cw = players[[
        "gsis_id", "pfr_id", "display_name", "position", "position_group",
        "birth_date", "rookie_season", "last_season", "latest_team",
        "college_name", "draft_year", "draft_round", "draft_pick", "draft_team",
    ]].copy()
    cw = cw.rename(columns={"gsis_id": "player_id"})
    cw["position"] = standardize_positions(cw["position"])
    dupe_gsis = cw["player_id"].duplicated().sum()
    if dupe_gsis:
        print(f"[crosswalk] WARNING: {dupe_gsis} duplicate player_id rows in players table")
    return cw


def add_age_and_experience(df, crosswalk, season_col="season"):
    """Attach age-as-of-Sept-1-of-season and years_in_league to any table
    keyed by [player_id, season]."""
    merged = df.merge(crosswalk[["player_id", "birth_date", "rookie_season"]], on="player_id", how="left")
    birth = pd.to_datetime(merged["birth_date"], errors="coerce")
    season_start = pd.to_datetime(merged[season_col].astype(str) + "-09-01")
    merged["age"] = (season_start - birth).dt.days / 365.25
    merged["years_in_league"] = merged[season_col] - merged["rookie_season"] + 1
    return merged.drop(columns=["birth_date", "rookie_season"])


def create_player_weekly_stats(weekly, snaps, crosswalk):
    """Join weekly offensive box scores to snap counts (via the pfr_id
    crosswalk) and add age/experience. Regular season only.

    nflverse's weekly file only covers offensive skill positions with
    counting stats (QB/RB/WR/TE, mainly); there is no per-week box score
    for defensive players upstream, so defense is season-only (see
    create_defense_season_stats).
    """
    reg = weekly[weekly["season_type"] == "REG"].copy()
    reg["position"] = standardize_positions(reg["position"])

    pfr_to_gsis = (
        crosswalk.dropna(subset=["pfr_id"])[["pfr_id", "player_id"]]
        .drop_duplicates("pfr_id")
    )
    snap_slim = snaps.rename(columns={"pfr_player_id": "pfr_id"}).merge(
        pfr_to_gsis, on="pfr_id", how="left"
    )
    dupes = snap_slim.duplicated(subset=["player_id", "season", "week"]).sum()
    if dupes:
        print(f"[weekly_stats] WARNING: {dupes} duplicate player/season/week snap rows; keeping first")
    snap_slim = snap_slim.dropna(subset=["player_id"]).drop_duplicates(
        subset=["player_id", "season", "week"]
    )[["player_id", "season", "week", "offense_snaps", "offense_pct"]]
    snap_slim = snap_slim.rename(columns={"offense_snaps": "snaps", "offense_pct": "snap_pct"})

    merged = reg.merge(snap_slim, on=["player_id", "season", "week"], how="left")
    merged = add_age_and_experience(merged, crosswalk)
    print(f"[weekly_stats] processed {merged.shape[0]:,} rows | "
          f"snap_pct match rate: {merged['snap_pct'].notna().mean() * 100:.1f}%")
    return merged


def create_player_season_stats(seasonal, weekly_stats, crosswalk):
    """Season totals (offense) enriched with position, games played, average
    season snap share, and age/experience. Regular season only."""
    reg = seasonal[seasonal["season_type"] == "REG"].copy()
    reg = reg.merge(crosswalk[["player_id", "display_name", "position"]], on="player_id", how="left")

    season_snaps = weekly_stats.groupby(["player_id", "season"]).agg(
        games_played=("week", "nunique"),
        avg_snap_pct=("snap_pct", "mean"),
    ).reset_index()
    reg = reg.merge(season_snaps, on=["player_id", "season"], how="left")
    reg = add_age_and_experience(reg, crosswalk)
    print(f"[season_stats] processed {reg.shape[0]:,} rows | "
          f"position match rate: {reg['position'].notna().mean() * 100:.1f}%")
    return reg


def create_defense_season_stats(defense_pfr, crosswalk):
    """PFR advanced defensive season stats, standardized and joined to the
    canonical player_id. Keeps PFR's own 'age' column (more precise than the
    Sept-1 estimate) but still adds years_in_league from the crosswalk."""
    df = defense_pfr.rename(columns={"tm": "team", "pos": "position_raw", "g": "games", "gs": "games_started"})
    df["position"] = standardize_positions(df["position_raw"])

    # PFR gives players traded mid-season BOTH a combined "2TM"/"3TM" row
    # (season totals) AND separate per-team stint rows for the same
    # (pfr_id, season) - unlike the offense stats, which are always exactly
    # one row per player-season. Left as-is this multiplies a traded
    # player's row count and, downstream, produced literal duplicate
    # entries in a sacks leaderboard (caught via Matthew Judon appearing
    # three times in a Task 2.4 EDGE projection). Keep only the combined
    # row when one exists, so every player-season is exactly one row here too.
    multi_team_mask = df["team"].str.match(r"^\dTM$", na=False)
    multi_team_keys = pd.MultiIndex.from_frame(df.loc[multi_team_mask, ["pfr_id", "season"]])
    is_redundant_stint_row = df.set_index(["pfr_id", "season"]).index.isin(multi_team_keys) & ~multi_team_mask
    dropped = int(is_redundant_stint_row.sum())
    if dropped:
        print(f"[defense_season_stats] dropping {dropped} per-team stint rows for "
              f"{multi_team_mask.sum()} multi-team player-seasons (keeping the combined row)")
    df = df[~is_redundant_stint_row].reset_index(drop=True)

    pfr_to_gsis = (
        crosswalk.dropna(subset=["pfr_id"])[["pfr_id", "player_id", "rookie_season"]]
        .drop_duplicates("pfr_id")
    )
    df = df.merge(pfr_to_gsis, on="pfr_id", how="left")
    df["years_in_league"] = df["season"] - df["rookie_season"] + 1
    df = df.drop(columns=["rookie_season", "position_raw"])
    print(f"[defense_season_stats] processed {df.shape[0]:,} rows | "
          f"player_id match rate: {df['player_id'].notna().mean() * 100:.1f}%")
    return df


def calculate_league_averages(player_season_stats, defense_season_stats):
    """Per-position, per-season baselines (mean/std/n) for a fixed set of
    counting stats. Long/tidy format (season, position, stat, mean, std, n)
    so it stays generic across offense and defense instead of hardcoding a
    wide table of mismatched columns. Used later as the replacement-level
    baseline for player impact / WAR calculations.
    """
    off_cols = [
        "passing_yards", "passing_tds", "passing_epa",
        "rushing_yards", "rushing_tds", "rushing_epa",
        "receiving_yards", "receiving_tds", "receiving_epa",
        "fantasy_points_ppr",
    ]
    def_cols = ["sk", "comb", "prss", "m_tkl", "int", "hrry"]

    off_long = player_season_stats.melt(
        id_vars=["season", "position"],
        value_vars=[c for c in off_cols if c in player_season_stats.columns],
        var_name="stat", value_name="value",
    )
    def_long = defense_season_stats.melt(
        id_vars=["season", "position"],
        value_vars=[c for c in def_cols if c in defense_season_stats.columns],
        var_name="stat", value_name="value",
    )
    long = pd.concat([off_long, def_long], ignore_index=True).dropna(subset=["position"])
    summary = (
        long.groupby(["season", "position", "stat"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    print(f"[league_averages] {summary.shape[0]:,} rows | "
          f"positions: {sorted(summary['position'].unique())}")
    return summary


def validate_processed_data(crosswalk, weekly_stats, season_stats, defense_stats, league_avgs):
    """Sanity checks on the processed/cleaned tables."""
    print("\n===== PROCESSED DATA VALIDATION =====")
    print(f"Player metadata (crosswalk): {crosswalk.shape[0]:,} rows | "
          f"{crosswalk['position'].nunique()} distinct positions")
    print(f"Player weekly stats: {weekly_stats.shape[0]:,} rows | "
          f"null snap_pct: {weekly_stats['snap_pct'].isna().mean() * 100:.1f}%")
    print(f"Player season stats: {season_stats.shape[0]:,} rows | "
          f"null position: {season_stats['position'].isna().mean() * 100:.1f}%")
    print(f"Defense season stats: {defense_stats.shape[0]:,} rows | "
          f"null player_id: {defense_stats['player_id'].isna().mean() * 100:.1f}%")
    print(f"League averages by position: {league_avgs.shape[0]:,} rows")

    latest = int(season_stats["season"].max())
    top_qb = (
        season_stats[(season_stats["season"] == latest) & (season_stats["position"] == "QB")]
        .nlargest(5, "passing_yards")[["display_name", "passing_yards", "age"]]
    )
    print(f"Top 5 QBs by passing yards, {latest} (with derived age):")
    print(top_qb.to_string(index=False))

    latest_def = int(defense_stats["season"].max())
    top_edge = (
        defense_stats[(defense_stats["season"] == latest_def) & (defense_stats["position"] == "EDGE")]
        .nlargest(5, "sk")[["player", "sk", "age"]]
    )
    print(f"Top 5 EDGE by sacks, {latest_def}:")
    print(top_edge.to_string(index=False))
    print("===== END VALIDATION =====\n")


def save_processed_data(crosswalk, weekly_stats, season_stats, defense_stats, league_avgs):
    """Persist all processed/cleaned tables to data/processed/ as UTF-8 CSVs."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    outputs = {
        "player_metadata.csv": crosswalk,
        "player_weekly_stats.csv": weekly_stats,
        "player_season_stats.csv": season_stats,
        "player_season_defense.csv": defense_stats,
        "league_averages_by_position.csv": league_avgs,
    }
    for filename, df in outputs.items():
        path = os.path.join(PROCESSED_DIR, filename)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {path} ({df.shape[0]:,} rows)")


def run_cleaning_pipeline():
    """Load raw CSVs from data/raw/ (skipping the large play-by-play file,
    which isn't needed for these joins) and build the processed tables."""
    snaps = pd.read_csv(os.path.join(RAW_DIR, "snap_counts_2015_2025.csv"))
    weekly = pd.read_csv(os.path.join(RAW_DIR, "player_weekly_raw_2015_2025.csv"))
    seasonal = pd.read_csv(os.path.join(RAW_DIR, "player_seasonal_raw_2015_2025.csv"))
    defense = pd.read_csv(os.path.join(RAW_DIR, "defense_pfr_2015_2025.csv"))
    players = pd.read_csv(os.path.join(RAW_DIR, "player_metadata.csv"))

    crosswalk = build_player_crosswalk(players)
    weekly_stats = create_player_weekly_stats(weekly, snaps, crosswalk)
    season_stats = create_player_season_stats(seasonal, weekly_stats, crosswalk)
    defense_stats = create_defense_season_stats(defense, crosswalk)
    league_avgs = calculate_league_averages(season_stats, defense_stats)

    validate_processed_data(crosswalk, weekly_stats, season_stats, defense_stats, league_avgs)
    save_processed_data(crosswalk, weekly_stats, season_stats, defense_stats, league_avgs)


# ---------------------------------------------------------------------------
# Prep Task: refresh 2025 offense data via nflreadpy.
#
# nfl_data_py's import_weekly_data/import_seasonal_data both 404 for 2025
# (still true as of this task - checked directly). nflreadpy (the newer,
# actively maintained replacement) has this data via load_player_stats(),
# confirmed by a direct pull before writing any of this. Its schema differs
# from nfl_data_py's (some columns renamed, e.g. interceptions ->
# passing_interceptions; a handful of legacy nfl_data_py-only derived
# columns like dakota/dom/w8dom/yptmpa/tgt_sh aren't present at all - none
# of those are used anywhere in this project's models, confirmed by
# grepping player_models.py/ol_quality.py/team_aggregation.py, so they're
# just left NaN for the new 2025 rows rather than reconstructed).
#
# Strategy: map nflreadpy's 2025 output onto the EXACT column set of the
# existing raw cache files (player_weekly_raw_2015_2025.csv,
# player_seasonal_raw_2015_2025.csv), append, and reuse run_cleaning_pipeline()
# completely unchanged - guarantees identical downstream processing to
# every other season already in this project, rather than a parallel
# processing path that could quietly drift from it. Defense (PFR-sourced)
# and player_metadata.csv already cover 2025 (verified directly - PFR data
# is PBP-native and doesn't share nflverse's seasonal-stats lag), so
# neither needs refreshing here.
# ---------------------------------------------------------------------------

def _pull_2025_via_nflreadpy():
    import nflreadpy as nfl_read

    weekly_raw_cols = pd.read_csv(
        os.path.join(RAW_DIR, "player_weekly_raw_2015_2025.csv"), nrows=0
    ).columns.tolist()
    seasonal_raw_cols = pd.read_csv(
        os.path.join(RAW_DIR, "player_seasonal_raw_2015_2025.csv"), nrows=0
    ).columns.tolist()

    weekly_2025 = nfl_read.load_player_stats(summary_level="week", seasons=[2025]).to_pandas()
    weekly_2025 = weekly_2025.rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
    })
    weekly_2025 = weekly_2025.reindex(columns=weekly_raw_cols)

    seasonal_2025 = nfl_read.load_player_stats(summary_level="reg", seasons=[2025]).to_pandas()
    seasonal_2025 = seasonal_2025.rename(columns={
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
        "wopr": "wopr_x",
    })
    seasonal_2025 = seasonal_2025.reindex(columns=seasonal_raw_cols)

    print(f"[nflreadpy] pulled {len(weekly_2025):,} 2025 weekly rows, {len(seasonal_2025):,} 2025 seasonal rows")
    return weekly_2025, seasonal_2025


def refresh_2025_offense_data():
    weekly_2025, seasonal_2025 = _pull_2025_via_nflreadpy()

    weekly_path = os.path.join(RAW_DIR, "player_weekly_raw_2015_2025.csv")
    seasonal_path = os.path.join(RAW_DIR, "player_seasonal_raw_2015_2025.csv")
    weekly_existing = pd.read_csv(weekly_path)
    seasonal_existing = pd.read_csv(seasonal_path)

    if 2025 in weekly_existing["season"].unique():
        print("[refresh_2025] 2025 already present in weekly raw cache - dropping before re-appending "
              "(idempotent rerun)")
        weekly_existing = weekly_existing[weekly_existing["season"] != 2025]
    if 2025 in seasonal_existing["season"].unique():
        seasonal_existing = seasonal_existing[seasonal_existing["season"] != 2025]

    weekly_combined = pd.concat([weekly_existing, weekly_2025], ignore_index=True)
    seasonal_combined = pd.concat([seasonal_existing, seasonal_2025], ignore_index=True)

    weekly_combined.to_csv(weekly_path, index=False, encoding="utf-8")
    seasonal_combined.to_csv(seasonal_path, index=False, encoding="utf-8")
    print(f"Saved {weekly_path} ({len(weekly_combined):,} rows, seasons "
          f"{int(weekly_combined['season'].min())}-{int(weekly_combined['season'].max())})")
    print(f"Saved {seasonal_path} ({len(seasonal_combined):,} rows, seasons "
          f"{int(seasonal_combined['season'].min())}-{int(seasonal_combined['season'].max())})")

    print("\n[refresh_2025] Re-running run_cleaning_pipeline() on the updated raw caches "
          "(same code path as every other season, no parallel logic)")
    run_cleaning_pipeline()


# ---------------------------------------------------------------------------
# Task 1.4: Vegas lines & game results
# ---------------------------------------------------------------------------
# nflverse's schedules file (already downloaded in Task 1.2) carries closing
# lines and final results together for every game back to 1999, sourced from
# a mix of public betting-market archives - no separate scrape/proxy needed.
# Only *closing* lines are available this way (no opener), so line-movement
# analysis isn't possible, but that's fine for the win-total/edge-detection
# use case, which only needs the closing number to compare against.

VEGAS_LINE_COLS = [
    "game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
    "spread_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds",
]
GAME_RESULT_COLS = [
    "game_id", "season", "week", "game_type", "home_team", "away_team",
    "home_score", "away_score", "result", "total", "overtime",
]


def load_historical_vegas_lines(schedules):
    """Extract the closing Vegas lines subset of the schedules table.

    spread_line convention (confirmed empirically against results): positive
    means the home team was favored by that many points, negative means the
    away team was favored.
    """
    df = schedules[VEGAS_LINE_COLS].copy()
    print(f"[vegas_lines] {df.shape[0]:,} games | seasons {int(df['season'].min())}-{int(df['season'].max())}")
    return df


def load_game_results(schedules):
    """Extract final scores/results, keeping only completed games."""
    df = schedules[GAME_RESULT_COLS].copy()
    df = df[df["home_score"].notna()]
    print(f"[game_results] {df.shape[0]:,} completed games")
    return df


def merge_vegas_and_results(lines, results):
    """Join lines to results and derive backtest-ready accuracy columns:
    ATS cover (with pushes), over/under hit (with pushes), and raw
    spread/total error versus the closing line.
    """
    merged = lines.merge(
        results[["game_id", "home_score", "away_score", "result", "total", "overtime"]],
        on="game_id", how="inner",
    )
    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    ats_diff = merged["home_margin"] - merged["spread_line"]
    merged["home_covered"] = ats_diff > 0
    merged["away_covered"] = ats_diff < 0
    merged["spread_push"] = ats_diff == 0
    merged["spread_error"] = ats_diff

    total_diff = merged["total"] - merged["total_line"]
    merged["over_hit"] = total_diff > 0
    merged["under_hit"] = total_diff < 0
    merged["total_push"] = total_diff == 0
    merged["total_error"] = total_diff

    merged["home_win"] = merged["home_margin"] > 0
    return merged


def validate_vegas_data(lines, results, merged):
    """Sanity checks on the Vegas lines / results / merged backtest data."""
    print("\n===== VEGAS DATA VALIDATION =====")
    print(f"Vegas lines: {lines.shape[0]:,} games | "
          f"spread_line range: [{lines['spread_line'].min()}, {lines['spread_line'].max()}] | "
          f"total_line range: [{lines['total_line'].min()}, {lines['total_line'].max()}]")
    assert lines["spread_line"].between(-28, 28).mean() > 0.99, "spread_line has too many extreme outliers"
    assert lines["total_line"].between(25, 70).mean() > 0.99, "total_line has too many extreme outliers"

    completeness = 1 - lines[["spread_line", "total_line"]].isna().mean().mean()
    print(f"Line completeness: {completeness * 100:.1f}%")
    assert completeness > 0.95, "too much missing line data"

    print(f"Game results: {results.shape[0]:,} completed games | "
          f"seasons {sorted(results['season'].unique())}")
    print(f"Merged backtest rows: {merged.shape[0]:,} "
          f"(expected ~{results.shape[0]:,})")

    reg = merged[merged["game_type"] == "REG"]
    print(f"REG season ATS: home covers {reg['home_covered'].mean() * 100:.1f}%, "
          f"away covers {reg['away_covered'].mean() * 100:.1f}%, "
          f"push {reg['spread_push'].mean() * 100:.1f}% (expect ~50/50, some push)")
    print(f"REG season totals: over {reg['over_hit'].mean() * 100:.1f}%, "
          f"under {reg['under_hit'].mean() * 100:.1f}%, "
          f"push {reg['total_push'].mean() * 100:.1f}%")
    print(f"Mean |spread_error|: {merged['spread_error'].abs().mean():.2f} pts | "
          f"Mean |total_error|: {merged['total_error'].abs().mean():.2f} pts")
    print("===== END VEGAS VALIDATION =====\n")


def run_vegas_pipeline():
    """Build the Vegas lines / game results / merged backtest tables from
    the schedules file downloaded in Task 1.2."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))

    lines = load_historical_vegas_lines(schedules)
    results = load_game_results(schedules)
    merged = merge_vegas_and_results(lines, results)
    validate_vegas_data(lines, results, merged)

    lines.to_csv(os.path.join(RAW_DIR, "vegas_lines_2015_2025.csv"), index=False, encoding="utf-8")
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "backtest"), exist_ok=True)
    results_path = os.path.join(PROJECT_ROOT, "data", "backtest", "game_results_2015_2025.csv")
    merged_path = os.path.join(PROJECT_ROOT, "data", "backtest", "vegas_with_results_2015_2025.csv")
    results.to_csv(results_path, index=False, encoding="utf-8")
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"Saved {os.path.join(RAW_DIR, 'vegas_lines_2015_2025.csv')} ({lines.shape[0]:,} rows)")
    print(f"Saved {results_path} ({results.shape[0]:,} rows)")
    print(f"Saved {merged_path} ({merged.shape[0]:,} rows)")


if __name__ == "__main__":
    if raw_data_present():
        print("Raw data already present in data/raw/; skipping download. "
              "Delete data/raw/*.csv to force a fresh download.")
    else:
        run_download_pipeline()

    run_cleaning_pipeline()
    run_vegas_pipeline()
