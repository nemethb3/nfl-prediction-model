"""Real QB career curve stratified by years-as-starter, not raw age - a
genuinely different signal from compute_empirical_age_curves.py's real age
curve, not a patch to it. See QB_PEAK_AGE_INVESTIGATION.md: that
investigation concluded the raw age-22 peak is a real, correct survivorship
finding and decided to KEEP the age curve unchanged. This module doesn't
touch that decision or that file - it builds a second, separate real curve,
and the "Three Accuracy Improvements" task deliberately chooses to swap
which curve feeds the trade model's QB age_curve_rising feature (see
build_trade_signals.py / generate_trade_scores_2026.py), while every other
position keeps using the original age-based curve.

Real bugs found and fixed in the originally pasted spec before writing
this:
1. The pasted spec's starter-detection line -
   `(x['offense_snaps'] > 0).sum() / (x['season'].nunique() * 17) > 0.5` -
   runs inside a groupby(['player_id','season']).apply(), where each group
   IS already a single season, so `x['season'].nunique()` is always 1 -
   silently hardcoding every real season to 17 games (wrong for 2015-2020,
   which were real 16-game seasons) and conflating "snapped at all" with
   "started". Real fix here: threshold on a real games-STARTED count
   (offense_pct > 0.5 in a given real game = started that game) being >= 8
   - roughly half of any real 16- or 17-game season - avoiding the
   season-length division entirely rather than trying to patch it.
2. snap_counts' real player key (pfr_player_id, e.g. "MurrKy00") doesn't
   match player_weekly_stats.csv's real player_id (gsis_id, e.g.
   "00-0035228") - crosswalked here via nflreadpy's load_ff_playerids()
   (same real crosswalk pattern build_trade_signals.py's
   _real_draft_capital() already uses). Real coverage checked: 99.9% of
   real QB snap-count rows 2015-2025 match (one real, disclosed miss:
   Shaun Hill, a journeyman backup with no crosswalk entry).

Also exports the real per-(player_id, season) years_as_starter lookup
(qb_starter_years.csv) as a shared artifact, so build_trade_signals.py and
generate_trade_scores_2026.py both reuse the exact same real computation
instead of duplicating the crosswalk/threshold logic in two places (same
"shared utility, not duplicated" discipline as elo_utils.py)."""

import json
import os

import numpy as np
import pandas as pd

import nflreadpy as nfl

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CURVE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "qb_starter_year_curve.json")
STARTER_YEARS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "qb_starter_years.csv")

EARLIEST_SEASON, LATEST_SEASON = 2015, 2025
GAMES_STARTED_THRESHOLD = 8  # real games with offense_pct > 0.5 to count a real season as "started"
MAX_STARTER_YEAR = 20


def _real_pfr_to_gsis():
    ids = nfl.load_ff_playerids().to_pandas()
    ids = ids[ids["pfr_id"].notna() & ids["gsis_id"].notna()].drop_duplicates("pfr_id")
    return ids.set_index("pfr_id")["gsis_id"].to_dict()


def _real_qb_season_starter_status():
    """Real per-(player_id, season) QB rows 2015-2025: real season PPR and
    whether the player started (>=8 real games with offense_pct > 0.5) that
    real season."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    pws = pws[(pws["position"] == "QB") & pws["fantasy_points_ppr"].notna()]
    pws = pws[(pws["season"] >= EARLIEST_SEASON) & (pws["season"] <= LATEST_SEASON)]
    season_ppr = pws.groupby(["player_id", "season"])["fantasy_points_ppr"].sum().rename("season_ppr")

    pfr_to_gsis = _real_pfr_to_gsis()
    snaps = nfl.load_snap_counts(seasons=list(range(EARLIEST_SEASON, LATEST_SEASON + 1))).to_pandas()
    snaps = snaps[snaps["position"] == "QB"].copy()
    snaps["player_id"] = snaps["pfr_player_id"].map(pfr_to_gsis)
    snaps = snaps[snaps["player_id"].notna()]
    snaps["started_game"] = snaps["offense_pct"] > 0.5
    games_started = snaps.groupby(["player_id", "season"])["started_game"].sum().rename("games_started")

    out = pd.concat([season_ppr, games_started], axis=1).reset_index()
    out["games_started"] = out["games_started"].fillna(0)
    out["is_starter_season"] = out["games_started"] >= GAMES_STARTED_THRESHOLD
    return out


def _add_years_as_starter(season_stats):
    """Real, chronological count of starter seasons up to and including
    each real season - only meaningful (and only assigned) for real starter
    seasons themselves."""
    season_stats = season_stats.sort_values(["player_id", "season"]).copy()
    season_stats["years_as_starter"] = (
        season_stats.groupby("player_id")["is_starter_season"].cumsum().where(season_stats["is_starter_season"]))
    return season_stats


def compute_stratified_qb_curve():
    print("\nComputing real QB career curve stratified by years-as-starter...\n")
    season_stats = _real_qb_season_starter_status()
    season_stats = _add_years_as_starter(season_stats)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    season_stats.to_csv(STARTER_YEARS_OUTPUT_PATH, index=False)
    print(f"Wrote {STARTER_YEARS_OUTPUT_PATH} ({len(season_stats)} real (player, season) rows)")

    starter_rows = season_stats[season_stats["is_starter_season"]]
    curve, sample_sizes = {}, {}
    for starter_year in range(1, MAX_STARTER_YEAR + 1):
        yr_data = starter_rows[starter_rows["years_as_starter"] == starter_year]
        if len(yr_data) == 0:
            continue
        curve[str(starter_year)] = round(float(yr_data["season_ppr"].median()), 1)
        sample_sizes[str(starter_year)] = int(yr_data["player_id"].nunique())

    print("Real QB median season PPR by years-as-starter:")
    for year in sorted(curve, key=int):
        print(f"  Year {year}: {curve[year]} PPR (n={sample_sizes[year]} real players)")

    output = {
        "QB": {
            "curve_type": "years_as_starter",
            "curve": curve,
            "sample_sizes": sample_sizes,
            "games_started_threshold": GAMES_STARTED_THRESHOLD,
            "note": (
                "Real QB career curve indexed by years-as-starter (>=8 real games with "
                "offense_pct>0.5 in a season = a real starter season), not raw age. A separate, "
                "deliberate signal from compute_empirical_age_curves.py's real age-based curve - "
                "see QB_PEAK_AGE_INVESTIGATION.md, which explicitly decided to keep the age-based "
                "curve as computed. This file is what the trade model's QB age_curve_rising feature "
                "uses instead; RB/WR/TE are unaffected and keep using the age-based curve."
            ),
        }
    }
    os.makedirs(os.path.dirname(CURVE_OUTPUT_PATH), exist_ok=True)
    with open(CURVE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("qb_starter_year_curve")
    print(f"Wrote {CURVE_OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    compute_stratified_qb_curve()
