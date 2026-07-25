"""Player impact via Wins Above Replacement (WAR).

Phase 2.X-Team (agreed hybrid scope, narrower than the original 8-task WAR
spec): rather than decomposing team defensive EPA down to individual CB/S/LB
using invented allocation weights, this module computes WAR only where PBP
gives real per-play, per-player attribution - pass rush (sacks, QB hits) for
EDGE/DL. Team-level defensive EPA prediction (not individual secondary/LB
WAR) is Task 2 of this phase, in a separate module.

  Task 1: Pass-Rush WAR (EDGE/DL), from real PBP attribution

Phase 2.X-Defense, Task 1 (this module's second addition): leverage EPA
attribution for all defenders, using PBP's defense_players column - a real
semicolon-separated list of the 11 defenders actually on the field for each
play (populated 2016+, ~93% of plays). This is a genuine upgrade over that
phase's own proposed proxy ("credit every player who had snaps that whole
week, weighted by season-long snap share") - credit here goes only to the
players who were verifiably on the field for that specific play. Phase
2.X-Defense's Task 2 (efficiency metrics: lockdown rate, yards/snap
allocation, gap-fit accuracy) was evaluated and not built - those proxies
don't hold up even with real participation data (e.g. yards-after-catch is
not the same thing as separation-at-catch, and no amount of knowing who was
on the field fixes that), so building them would have reintroduced the same
invented-constant problem this whole WAR framework was designed to avoid.

  Task 2 of this phase: Leverage EPA Attribution (all defensive positions,
  with CB/S/LB as the focus - EDGE/DL already have pass-rush WAR)
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

EPA_PER_WIN = 0.13  # standard football-analytics rule of thumb (e.g. nflfastR/Sharpe): ~0.13 net EPA = 1 win. An approximation, not a precise constant - used for scale, not exact conversion.
PBP_COLS = [
    "game_id", "play_id", "season", "week", "epa", "play_type",
    "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
    "qb_hit_1_player_id", "qb_hit_2_player_id",
]


def load_pass_rush_events(pbp_path=None):
    """Long-format table of every sack/half-sack/QB-hit event with the play's
    EPA (from the offense's perspective) and the credited defender's
    player_id. Split-sack plays (half_sack_1 + half_sack_2) get 0.5 weight
    each so their combined credit equals one full play's value, not double -
    every other event type (full sack, QB hit) gets full weight, matching how
    PFR itself scores a full sack as 1.0 and a split sack as 0.5 + 0.5.
    """
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    attribution_cols = ["sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
                         "qb_hit_1_player_id", "qb_hit_2_player_id"]

    # The full file (1.3GB, 532K rows x 398 cols even with usecols applied at
    # the C-parser level) has been enough to trigger MemoryError on this
    # machine before (see Task 1.2) - only ~35K of these 532K rows have any
    # attribution at all, so stream in chunks and drop the rest immediately
    # rather than materializing the whole column subset in memory at once.
    keep_chunks = []
    for chunk in pd.read_csv(path, usecols=PBP_COLS, low_memory=False, chunksize=50_000):
        mask = chunk[attribution_cols].notna().any(axis=1) & chunk["epa"].notna()
        if mask.any():
            keep_chunks.append(chunk.loc[mask])
    pbp = pd.concat(keep_chunks, ignore_index=True)

    frames = []
    for col, event_type, weight in [
        ("sack_player_id", "sack", 1.0),
        ("half_sack_1_player_id", "half_sack", 0.5),
        ("half_sack_2_player_id", "half_sack", 0.5),
        ("qb_hit_1_player_id", "qb_hit", 1.0),
        ("qb_hit_2_player_id", "qb_hit", 1.0),
    ]:
        sub = pbp.loc[pbp[col].notna(), ["game_id", "play_id", "season", "epa", col]].copy()
        sub = sub.rename(columns={col: "player_id"})
        sub["event_type"] = event_type
        sub["weight"] = weight
        frames.append(sub)

    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates(subset=["game_id", "play_id", "player_id", "event_type"])
    events["defensive_epa"] = -events["epa"] * events["weight"]
    print(f"[pass_rush_events] {events.shape[0]:,} attributed events "
          f"({events['event_type'].value_counts().to_dict()})")
    return events


def aggregate_pass_rush_epa(events):
    """Per player-season: total defensive EPA generated on disruptive plays,
    and counts of sacks/half-sacks/QB hits (for sanity-checking against the
    PFR sack totals already in player_season_defense.csv)."""
    events = events.copy()
    events["sack_weight"] = np.where(events["event_type"].isin(["sack", "half_sack"]), events["weight"], 0.0)
    events["is_qb_hit"] = (events["event_type"] == "qb_hit").astype(int)

    agg = events.groupby(["player_id", "season"]).agg(
        defensive_epa_generated=("defensive_epa", "sum"),
        disruptive_plays=("event_type", "count"),
        sacks_from_pbp=("sack_weight", "sum"),
        qb_hits_from_pbp=("is_qb_hit", "sum"),
    ).reset_index()
    return agg


def player_defensive_snaps(snap_counts, crosswalk):
    """Total defensive snaps played per player-season, via the pfr_id
    crosswalk (snap_counts is PFR-keyed, everything else here is gsis-keyed)."""
    pfr_to_gsis = crosswalk.dropna(subset=["pfr_id"])[["pfr_id", "player_id"]].drop_duplicates("pfr_id")
    snaps = snap_counts.rename(columns={"pfr_player_id": "pfr_id"}).merge(pfr_to_gsis, on="pfr_id", how="left")
    out = snaps.dropna(subset=["player_id"]).groupby(["player_id", "season"])["defense_snaps"].sum().reset_index()
    return out.rename(columns={"defense_snaps": "defensive_snaps"})


def calculate_replacement_level(df, position_col="position", value_col="epa_per_snap",
                                 min_snaps=100, percentile=10):
    """Empirically derived replacement level per position: the value_col at
    the given bottom percentile among players with at least min_snaps that
    season (a player getting real snaps but performing at the bottom of the
    pool is a reasonable proxy for "replacement," matching the reasoning a
    human GM uses - rather than asserting a made-up constant)."""
    pool = df[df["defensive_snaps"] >= min_snaps]
    levels = pool.groupby(position_col)[value_col].apply(
        lambda s: np.percentile(s, percentile)
    ).reset_index(name="replacement_epa_per_snap")
    print(f"[replacement_level] derived from bottom {percentile}% of players with >={min_snaps} defensive snaps:")
    print(levels.to_string(index=False))
    return levels


def position_season_scaffold(snap_counts, crosswalk, positions=("EDGE", "DL")):
    """One row per (player_id, season, team) for the given standardized
    positions, spanning every season snap_counts has (2015-2025) - unlike
    player_features_with_history.csv, which is built from PFR's advanced
    defense stats and is capped at 2018-2025. Sourcing the scaffold from
    snap_counts instead is what actually delivers the wider 2015-2025 PBP
    coverage this module is built to take advantage of (a first version of
    this function sourced from features_df and silently inherited the
    2018-2025 cap - caught by the validation print showing "seasons
    2018-2025" instead of the expected 2015-2025)."""
    from data_pipeline import standardize_positions  # local import: avoids a hard dependency at module load time

    sc = snap_counts.copy()
    sc["position"] = standardize_positions(sc["position"])
    sc = sc[sc["position"].isin(positions)]

    pfr_to_gsis = crosswalk.dropna(subset=["pfr_id"])[["pfr_id", "player_id", "display_name"]].drop_duplicates("pfr_id")
    sc = sc.rename(columns={"pfr_player_id": "pfr_id"}).merge(pfr_to_gsis, on="pfr_id", how="left")
    sc = sc.dropna(subset=["player_id"])

    # team = whichever team the player logged the most defensive snaps for
    # that season (handles in-season trades better than "last team played for").
    team_by_snaps = (
        sc.groupby(["player_id", "season", "team"])["defense_snaps"].sum()
        .reset_index()
        .sort_values("defense_snaps", ascending=False)
        .drop_duplicates(subset=["player_id", "season"])
    )
    scaffold = sc[["player_id", "season", "position", "display_name"]].drop_duplicates(subset=["player_id", "season"])
    scaffold = scaffold.merge(team_by_snaps[["player_id", "season", "team"]], on=["player_id", "season"], how="left")
    return scaffold


def calculate_pass_rush_war(crosswalk, defense_stats_df, snap_counts, pbp_path=None):
    """Builds pass-rush WAR for every EDGE/DL player-season with real PBP
    sack/QB-hit attribution (2015-2025 - PBP's full range, wider than PFR's
    2018-2025 advanced-defense coverage used elsewhere in this project)."""
    events = load_pass_rush_events(pbp_path)
    epa_agg = aggregate_pass_rush_epa(events)
    snaps = player_defensive_snaps(snap_counts, crosswalk)
    positions = position_season_scaffold(snap_counts, crosswalk)

    df = positions.merge(epa_agg, on=["player_id", "season"], how="left")
    df = df.merge(snaps, on=["player_id", "season"], how="left")
    df[["defensive_epa_generated", "disruptive_plays", "sacks_from_pbp", "qb_hits_from_pbp"]] = df[[
        "defensive_epa_generated", "disruptive_plays", "sacks_from_pbp", "qb_hits_from_pbp"
    ]].fillna(0)
    df["defensive_snaps"] = df["defensive_snaps"].fillna(0)
    df["epa_per_snap"] = np.where(df["defensive_snaps"] > 0,
                                   df["defensive_epa_generated"] / df["defensive_snaps"], 0.0)

    replacement = calculate_replacement_level(df, min_snaps=100)
    df = df.merge(replacement, on="position", how="left")

    df["surplus_epa_per_snap"] = df["epa_per_snap"] - df["replacement_epa_per_snap"]
    df["surplus_epa_total"] = df["surplus_epa_per_snap"] * df["defensive_snaps"]
    df["war"] = df["surplus_epa_total"] * EPA_PER_WIN

    # cross-check against PFR sacks already in player_season_defense.csv
    pfr_sacks = defense_stats_df[defense_stats_df["position"].isin(["EDGE", "DL"])][
        ["player_id", "season", "sk"]
    ].rename(columns={"sk": "pfr_sacks"})
    df = df.merge(pfr_sacks, on=["player_id", "season"], how="left")

    return df.sort_values(["season", "war"], ascending=[True, False]).reset_index(drop=True)


def validate_pass_rush_war(war_df):
    print("\n===== PASS-RUSH WAR VALIDATION =====")
    print(f"Total player-seasons: {war_df.shape[0]:,} "
          f"(seasons {int(war_df['season'].min())}-{int(war_df['season'].max())})")

    corr = war_df[war_df["defensive_snaps"] >= 100][["war", "pfr_sacks"]].corr().iloc[0, 1]
    print(f"Correlation(WAR, PFR sacks) among players with 100+ def. snaps: {corr:.3f} "
          f"(expect strongly positive - sacks are the dominant WAR driver here by construction, "
          f"this checks the pipeline isn't broken, not that WAR adds new information)")

    sack_check = war_df.groupby("season").apply(
        lambda g: (g["sacks_from_pbp"].sum(), g["pfr_sacks"].sum() if g["pfr_sacks"].notna().any() else np.nan)
    )
    print("\nPBP-derived sack totals vs. PFR sack totals, by season (sanity check - should be close, "
          "not exact, since PFR includes plays PBP attribution occasionally misses and vice versa; "
          "PFR is N/A before 2018, PBP attribution has no such gap):")
    for season, (pbp_total, pfr_total) in sack_check.items():
        pfr_str = f"{pfr_total:.1f}" if pd.notna(pfr_total) else "N/A (pre-2018)"
        print(f"  {season}: PBP={pbp_total:.1f} | PFR={pfr_str}")

    latest = int(war_df["season"].max())
    top10 = war_df[war_df["season"] == latest].nlargest(10, "war")[
        ["display_name", "position", "team", "war", "pfr_sacks", "defensive_snaps"]
    ]
    print(f"\nTop 10 pass-rush WAR, {latest}:")
    print(top10.to_string(index=False))

    garrett = war_df[(war_df["display_name"].str.contains("Garrett", na=False)) & (war_df["season"] == 2024)]
    if not garrett.empty:
        print(f"\nMyles Garrett 2024 spot check: WAR={garrett['war'].iloc[0]:.2f}, "
              f"sacks={garrett['pfr_sacks'].iloc[0]}")

    neg_war = (war_df[war_df["defensive_snaps"] >= 100]["war"] < 0).mean() * 100
    print(f"\nShare of 100+ snap players with negative WAR: {neg_war:.1f}% "
          f"(expect a meaningful minority - most rostered players hover near replacement, "
          f"a large chunk below it is normal for a replacement-level baseline)")
    print("===== END VALIDATION =====\n")


def run_pass_rush_war():
    defense_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_defense.csv"))
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    snap_counts = pd.read_csv(os.path.join(RAW_DIR, "snap_counts_2015_2025.csv"))

    war_df = calculate_pass_rush_war(crosswalk, defense_stats_df, snap_counts)
    validate_pass_rush_war(war_df)

    out_path = os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv")
    war_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({war_df.shape[0]:,} rows)")
    return war_df


# ---------------------------------------------------------------------------
# Phase 2.X-Defense, Task 1: Leverage EPA Attribution (real on-field rosters)
# ---------------------------------------------------------------------------

LEVERAGE_PBP_COLS = ["game_id", "play_id", "season", "week", "epa", "play_type",
                     "season_type", "defense_players"]
CATASTROPHIC_THRESHOLD = 2.0   # offense gained this much EPA or more - defensive failure
CLUTCH_THRESHOLD = -1.5        # offense lost this much EPA or more - defensive success
LEVERAGE_POSITIONS = ("EDGE", "DL", "LB", "CB", "S")  # defense_players covers everyone on
                                                        # the field, not just CB/S/LB - kept
                                                        # general even though this task's focus
                                                        # is the secondary/LB, since EDGE/DL
                                                        # could use this as a supplementary
                                                        # signal alongside pass-rush WAR later.


def identify_leverage_plays(pbp_path=None, catastrophic_threshold=CATASTROPHIC_THRESHOLD,
                             clutch_threshold=CLUTCH_THRESHOLD):
    """Streams PBP (same chunked/memory-safe approach as load_pass_rush_events)
    and keeps only regular-season pass/run plays with a populated defense_players
    list and an EPA beyond the leverage thresholds - catastrophic defensive
    failures or clutch defensive successes."""
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    keep_chunks = []
    for chunk in pd.read_csv(path, usecols=LEVERAGE_PBP_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[
            chunk["play_type"].isin(["pass", "run"])
            & (chunk["season_type"] == "REG")
            & chunk["defense_players"].notna()
            & chunk["epa"].notna()
            & ((chunk["epa"] > catastrophic_threshold) | (chunk["epa"] < clutch_threshold))
        ]
        if len(sub):
            keep_chunks.append(sub)
    plays = pd.concat(keep_chunks, ignore_index=True)
    plays["leverage_category"] = np.where(plays["epa"] > catastrophic_threshold, "catastrophic", "clutch")
    print(f"[leverage_plays] {plays.shape[0]:,} plays "
          f"({plays['leverage_category'].value_counts().to_dict()}), "
          f"{int(plays['season'].min())}-{int(plays['season'].max())}")
    return plays


def expand_defenders_on_field(plays):
    """One row per (play, on-field defender), splitting defense_players'
    semicolon-separated ID list. Each play's EPA is split equally among
    however many defenders are actually listed (normally 11, but a handful
    of plays have fewer due to upstream charting gaps - dividing by the
    actual count rather than assuming 11 keeps the credit summing correctly
    even on those rows) - not weighted by season-long snap share like the
    original Phase 2.X-Defense spec proposed, since we now know exactly who
    was on the field for this specific play."""
    plays = plays.copy()
    plays["defender_list"] = plays["defense_players"].str.split(";")
    exploded = plays.explode("defender_list").rename(columns={"defender_list": "player_id"})
    n_defenders = exploded.groupby(["game_id", "play_id"])["player_id"].transform("count")
    exploded["defensive_epa_share"] = -exploded["epa"] / n_defenders
    return exploded


def aggregate_leverage_epa(exploded):
    """Per player-season: catastrophic/clutch play counts and net leverage
    EPA. Sign convention matches pass-rush WAR's epa_per_snap: HIGHER
    (more positive) leverage_epa is BETTER - a clutch interception (offense
    epa very negative) flips to a large positive defensive_epa_share, while
    a blown coverage (offense epa very positive) flips to a large negative
    share. Net positive = more associated with clutch defensive plays; net
    negative = more associated with catastrophic ones."""
    exploded = exploded.copy()
    exploded["is_catastrophic"] = (exploded["leverage_category"] == "catastrophic").astype(int)
    exploded["is_clutch"] = (exploded["leverage_category"] == "clutch").astype(int)

    agg = exploded.groupby(["player_id", "season"]).agg(
        catastrophic_plays=("is_catastrophic", "sum"),
        clutch_plays=("is_clutch", "sum"),
        leverage_epa=("defensive_epa_share", "sum"),
    ).reset_index()
    return agg


def calculate_leverage_war(crosswalk, snap_counts, pbp_path=None, positions=LEVERAGE_POSITIONS):
    """Builds leverage WAR for every defender-season across the given
    positions, using real on-field rosters (2016-2025 - defense_players has
    no 2015 coverage, one year narrower than pass-rush WAR's 2015-2025)."""
    plays = identify_leverage_plays(pbp_path)
    exploded = expand_defenders_on_field(plays)
    epa_agg = aggregate_leverage_epa(exploded)
    snaps = player_defensive_snaps(snap_counts, crosswalk)
    scaffold = position_season_scaffold(snap_counts, crosswalk, positions=positions)

    df = scaffold.merge(epa_agg, on=["player_id", "season"], how="left")
    df = df.merge(snaps, on=["player_id", "season"], how="left")
    df[["catastrophic_plays", "clutch_plays", "leverage_epa"]] = df[
        ["catastrophic_plays", "clutch_plays", "leverage_epa"]
    ].fillna(0)
    df["defensive_snaps"] = df["defensive_snaps"].fillna(0)
    df["leverage_epa_per_snap"] = np.where(df["defensive_snaps"] > 0,
                                            df["leverage_epa"] / df["defensive_snaps"], 0.0)

    # Empirically-derived replacement level, same approach and same sign
    # convention as pass-rush WAR (bottom 10th percentile among players with
    # a real role - higher leverage_epa_per_snap is better, exactly like
    # pass-rush WAR's epa_per_snap, so this is a direct copy of that pattern).
    #
    # An earlier version of this function used the 90th percentile with the
    # subtraction reversed, on a mistaken belief that lower leverage_epa was
    # better. That inverted every downstream number: a validation check
    # (correlating team-average leverage_war against the real, already-
    # computed team_defense_epa_allowed from Task 2) caught it immediately -
    # the correlation came out +0.84 (the WORST real defenses scoring the
    # HIGHEST leverage_war) instead of the expected strong negative. Fixed
    # here; re-validated after the fix (see validate_leverage_war).
    replacement = calculate_replacement_level(df, value_col="leverage_epa_per_snap",
                                               min_snaps=100, percentile=10)
    replacement = replacement.rename(columns={"replacement_epa_per_snap": "replacement_leverage_per_snap"})
    df = df.merge(replacement, on="position", how="left")

    df["surplus_leverage_per_snap"] = df["leverage_epa_per_snap"] - df["replacement_leverage_per_snap"]
    df["surplus_leverage_total"] = df["surplus_leverage_per_snap"] * df["defensive_snaps"]
    df["leverage_war"] = df["surplus_leverage_total"] * EPA_PER_WIN

    return df.sort_values(["season", "leverage_war"], ascending=[True, False]).reset_index(drop=True)


def validate_leverage_war(war_df):
    print("\n===== LEVERAGE WAR VALIDATION =====")
    print(f"Total player-seasons: {war_df.shape[0]:,} "
          f"(seasons {int(war_df['season'].min())}-{int(war_df['season'].max())})")

    per_team_season = war_df.groupby("season")[["catastrophic_plays", "clutch_plays"]].sum() / 32
    print("\nAvg catastrophic/clutch plays per team-season (spec expected ~100-150 catastrophic, "
          "~50-100 clutch - note our counts are the DEFENDER-credit count, i.e. play_count x 11 "
          "on-field defenders, so compare against play-level counts, not this table, for that check):")
    print(per_team_season.round(1).to_string())

    min_snaps_pool = war_df[war_df["defensive_snaps"] >= 100]
    print(f"\nleverage_war range (100+ snap players): min={min_snaps_pool['leverage_war'].min():.2f}, "
          f"max={min_snaps_pool['leverage_war'].max():.2f}, mean={min_snaps_pool['leverage_war'].mean():.2f}")

    # The load-bearing check: this is what caught the sign-inversion bug
    # (originally +0.84, wrong direction, before the replacement-level fix).
    # Team-average leverage_war should correlate NEGATIVELY with real team
    # defense EPA allowed (Task 2's output) - better real defenses should
    # score higher leverage_war.
    team_def_path = os.path.join(PROCESSED_DIR, "team_defense_epa_2015_2025.csv")
    if os.path.exists(team_def_path):
        team_def = pd.read_csv(team_def_path)
        team_avg = min_snaps_pool.groupby(["team", "season"])["leverage_war"].mean().reset_index()
        merged = team_avg.merge(team_def[["team", "season", "def_epa_allowed"]], on=["team", "season"])
        corr = merged["leverage_war"].corr(merged["def_epa_allowed"])
        print(f"\nCorrelation(team-avg leverage_war, real team def_epa_allowed): {corr:.3f} "
              f"(expect clearly negative - better real defenses should score higher leverage_war; "
              f"a positive value here means the sign is backwards, re-check replacement-level direction)")
        assert corr < -0.3, f"leverage_war correlates {corr:.3f} with team defense EPA - expected clearly negative, sign is likely inverted"
    else:
        print("\n[WARNING] team_defense_epa_2015_2025.csv not found - skipping the team-defense-EPA cross-check")

    for position in sorted(war_df["position"].unique()):
        latest = int(war_df["season"].max())
        pos_latest = war_df[(war_df["position"] == position) & (war_df["season"] == latest)
                             & (war_df["defensive_snaps"] >= 100)]
        print(f"\nTop 5 {position} by leverage WAR, {latest} (100+ snaps):")
        print(pos_latest.nlargest(5, "leverage_war")[
            ["display_name", "team", "leverage_war", "catastrophic_plays", "clutch_plays", "defensive_snaps"]
        ].to_string(index=False))
    print("===== END VALIDATION =====\n")


# ---------------------------------------------------------------------------
# Phase 2 Refinement, Task 2: Tackle-Efficiency component (for CB/S/LB blend)
# ---------------------------------------------------------------------------

TACKLE_EFFICIENCY_POSITIONS = ("CB", "S", "LB")


def calculate_tackle_efficiency(crosswalk, defense_stats_df, snap_counts, positions=TACKLE_EFFICIENCY_POSITIONS):
    """Replacement-level-adjusted tackle rate for CB/S/LB, same construction
    as pass-rush/leverage WAR (empirical bottom-10th-percentile replacement
    among 100+ snap players) but deliberately kept in raw tackle units
    (surplus tackles above replacement), NOT converted through EPA_PER_WIN.

    Combined tackles (PFR's 'comb') aren't EPA - they're a count of a
    different kind of event entirely. Multiplying a tackle-surplus by
    EPA_PER_WIN (a rate calibrated from real EPA-to-win data) wouldn't
    "convert tackles to wins," it would just produce a number in the same
    numeric neighborhood as WAR by coincidence, dressed up as if it were
    calibrated - the same category of problem this project has deliberately
    avoided elsewhere (invented decomposition weights, the leverage WAR sign
    bug). The two components are combined via standardization instead (see
    build_blended_score in player_models.py), not a shared "WAR" unit.

    2018-2025 only (player_season_defense.csv is PFR-sourced, PFR's earliest
    coverage - narrower than snap_counts' 2015-2025, same cap already in
    effect for the existing tackle-count CB/S/LB models)."""
    scaffold = position_season_scaffold(snap_counts, crosswalk, positions=positions)
    snaps = player_defensive_snaps(snap_counts, crosswalk)
    tackles = defense_stats_df[defense_stats_df["position"].isin(positions)][
        ["player_id", "season", "comb"]
    ].rename(columns={"comb": "combined_tackles"})

    df = scaffold.merge(tackles, on=["player_id", "season"], how="inner")  # inner: PFR's 2018-2025 cap
    df = df.merge(snaps, on=["player_id", "season"], how="left")
    df["defensive_snaps"] = df["defensive_snaps"].fillna(0)
    df["tackle_efficiency_per_snap"] = np.where(df["defensive_snaps"] > 0,
                                                 df["combined_tackles"] / df["defensive_snaps"], 0.0)

    replacement = calculate_replacement_level(df, value_col="tackle_efficiency_per_snap",
                                               min_snaps=100, percentile=10)
    replacement = replacement.rename(columns={"replacement_epa_per_snap": "replacement_tackle_per_snap"})
    df = df.merge(replacement, on="position", how="left")

    df["surplus_tackle_efficiency_per_snap"] = df["tackle_efficiency_per_snap"] - df["replacement_tackle_per_snap"]
    df["surplus_tackle_efficiency_total"] = df["surplus_tackle_efficiency_per_snap"] * df["defensive_snaps"]

    print(f"[tackle_efficiency] {df.shape[0]:,} player-seasons, "
          f"{int(df['season'].min())}-{int(df['season'].max())}")
    return df


def run_tackle_efficiency():
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    defense_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_defense.csv"))
    snap_counts = pd.read_csv(os.path.join(RAW_DIR, "snap_counts_2015_2025.csv"))

    df = calculate_tackle_efficiency(crosswalk, defense_stats_df, snap_counts)
    out_path = os.path.join(PROCESSED_DIR, "tackle_efficiency_2018_2025.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({df.shape[0]:,} rows)")
    return df


def run_leverage_war():
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    snap_counts = pd.read_csv(os.path.join(RAW_DIR, "snap_counts_2015_2025.csv"))

    war_df = calculate_leverage_war(crosswalk, snap_counts)
    validate_leverage_war(war_df)

    out_path = os.path.join(PROCESSED_DIR, "leverage_war_2016_2025.csv")
    war_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({war_df.shape[0]:,} rows)")
    return war_df


if __name__ == "__main__":
    run_pass_rush_war()
    run_leverage_war()
    run_tackle_efficiency()
