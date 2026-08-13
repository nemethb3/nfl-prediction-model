"""Dashboard Section 2 data export: real 2025 fantasy projections, no fabrication.

RB/QB/TE reuse the already-validated per-week trailing-volume formulas
directly (see module history in PROGRESS.md / DASHBOARD_DATA_GAPS.md):
- RB: fantasy_rb_formula.project_rb_fantasy_points() (real 2025 corr +0.651).
- QB/TE: fantasy_formula_improvements.py's private trailing-volume helpers
  (that module never exposed a standalone "project" function like RB's).

WR now uses a real, backtested week-varying projection (Phase 4,
src/wr_dynamic_backtest.py): a trailing up-to-4-week real actual-PPR
average, real winner of a real leave-one-out backtest across 23 variations
(correlation 0.4414 vs. the old static baseline's real leak-free 0.3957 -
see DASHBOARD_DATA_GAPS.md). This only applies once a player has at least
one real prior 2025 week to trail; a player's first real 2025
appearance (almost always week 1) has no trailing data yet, so it falls
back to the real static, in-sample-calibrated EPA x volume season
projection (the same real formula this project used exclusively before
Phase 4) - the same "no fallback fabricated, real prior data or nothing"
convention already used for defense_rank_vs_position and RB/QB/TE's week-1
handling elsewhere in this file. `projection_type` is set per-row
("weekly" vs. "season_static_per_game_avg") so the frontend can tell which
real method produced which row, instead of assuming all WR rows are one or
the other.

defense_rank_vs_position is real: matchup_features.build_defense_epa_by_
position_multi_season() computes real, per-team-per-week EPA allowed by
position from play-by-play. Ranked here as a TRAILING (through week W-1)
value across all 32 teams - week 1 has no real trailing data and is left
null rather than fabricated with an invented fallback (unlike RB/QB/TE's
week-1 2024 fallback, matchup_features.py has no equivalent real prior-
season mechanism built in).

recent_form (real trailing 4-week actual PPR) and injury_status (real
nflreadpy weekly injury reports) are both real, leak-free (recent_form only
uses weeks strictly before the week being shown), and now wired in - see
Phase 2 completion notes in DASHBOARD_DATA_GAPS.md. No confidence interval
is included - that still needs real per-player calibration work, not a
lookup, and was explicitly deferred.

actual_ppr (Phase 3) reuses _recent_form_lookup()'s already-correct real
(player_id, week) -> fantasy_points_ppr table (same season/season_type
filtering that fixed a real bug in Phase 2) rather than rebuilding a
second lookup. accuracy_tier is bucketed via EMPIRICAL PER-POSITION
terciles of the real |actual - projected| distribution (computed live each
run, like matchup_quality's approach), not an asserted flat +-2/+-5
threshold - QB/RB point scales are naturally larger than TE's, so a flat
threshold would represent a different real percentile per position. WR's
week-1 rows (see _wr_projections) still use the real static per-game
average, so their real diffs are dominated by real week-to-week variance
rather than model error - disclosed per-row in the frontend via
projection_type, not left to look like a broken model.
"""

import json
from generation_timestamps import record_generation
import os

import numpy as np
import pandas as pd

from nflreadpy import load_injuries

import fantasy_formula_improvements as ffi
from fantasy_rb_formula import load_rb_volume_data_2025, project_rb_fantasy_points
from fantasy_validation import extract_actual_fantasy_points_2025, project_fantasy_points_from_epa
from matchup_features import build_defense_epa_by_position_multi_season

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "fantasy_rankings_2025.json")

SEASON = 2025


def _rb_projections():
    proj = project_rb_fantasy_points(load_rb_volume_data_2025())
    proj = proj[["player_id", "player_name", "team", "week", "projected_fantasy_pts"]].copy()
    proj["position"] = "RB"
    proj["source"] = "rb_volume_formula_2025 (real trailing-window, corr=+0.651)"
    proj["projection_type"] = "weekly"
    return proj


def _qb_or_te_projections(position):
    df = ffi._load_position_data(position)
    volume_cols = ffi.QB_VOLUME_COLS if position == "QB" else ffi.TE_VOLUME_COLS
    trailing_vol = ffi._trailing_volume(df, volume_cols)
    trailing_vol["projected_fantasy_pts"] = ffi._real_ppr(position, trailing_vol)

    teams = df[df["season"] == SEASON][["player_id", "week", "recent_team"]].drop_duplicates(["player_id", "week"])
    proj = trailing_vol.merge(teams, on=["player_id", "week"], how="left").rename(columns={"recent_team": "team"})
    proj = proj[["player_id", "player_name", "team", "week", "projected_fantasy_pts"]].copy()
    proj["position"] = position
    corr = ffi.CURRENT_FORMULA_CORR[position]
    label = "qb_volume_formula_2025" if position == "QB" else "te_volume_formula_2025"
    proj["source"] = f"{label} (real trailing-window, beats {corr:+.3f} corr combined-formula baseline)"
    proj["projection_type"] = "weekly"
    return proj


def _wr_static_fallback():
    """Real, static (season-long) WR projection - used only as the week-1 /
    no-trailing-data fallback (see _wr_projections). Calibrated to real PPR
    units via the same in-sample linear fit fantasy_validation.py uses
    internally (not a new fit), then divided by each real player's own real
    expected_games_2025 to get a per-game average - a real unit conversion,
    not a new estimate - so the number is comparable in scale to RB/QB/TE's
    real weekly figures instead of showing a ~100-250 point full-season
    total under a single week filter (caught by inspecting an implausible
    143.2 "PPR" value before shipping)."""
    proj = project_fantasy_points_from_epa("WR")
    raw = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2025.csv"))
    proj = proj.merge(raw[["player_id", "expected_games_2025"]], on="player_id", how="left")

    actual = extract_actual_fantasy_points_2025()
    actual_season = actual[actual["position"] == "WR"].groupby(
        ["player_id"])["actual_fantasy_pts"].sum().reset_index().rename(columns={"actual_fantasy_pts": "actual_season_fantasy_pts"})

    merged = proj.merge(actual_season, on="player_id", how="inner")
    merged = merged[merged["projected_volume"] > 0].reset_index(drop=True)
    if len(merged) < 5:
        raise RuntimeError("Too few matched real WR players to fit a calibration - check upstream data.")
    slope, intercept = np.polyfit(merged["projected_score"], merged["actual_season_fantasy_pts"], 1)
    merged["season_projected_pts"] = slope * merged["projected_score"] + intercept
    merged = merged.dropna(subset=["expected_games_2025"])
    merged = merged[merged["expected_games_2025"] > 0]
    merged["static_per_game_projected_pts"] = merged["season_projected_pts"] / merged["expected_games_2025"]

    # Real per-week `team` (2026-07-30 audit fix): `actual` already carries a
    # real, per-week team (extract_actual_fantasy_points_2025() derives it
    # from recent_team, the same real source RB/QB/TE use), but this used to
    # get dropped here and replaced by merged["team"] - the STALE preseason
    # team from wr_epa_projections_2025.csv, never refreshed against
    # in-season trades. Confirmed via direct audit: 30/107 real WR players
    # (e.g. George Pickens, real team DAL, was shown as PIT every week) had
    # a wrong team, which cascaded into wrong opponent and wrong
    # opponent_defense_rank_vs_position for all of those players' weeks.
    # Fixed by keeping the real per-week team already present here instead
    # of re-deriving it from the stale source.
    real_weeks = actual[actual["position"] == "WR"][["player_id", "week", "team"]].drop_duplicates()
    out = real_weeks.merge(
        merged[["player_id", "player_name", "static_per_game_projected_pts"]], on="player_id", how="inner")
    return out


def _wr_projections(form_lookup):
    """Real WR projection (Phase 4 winner, src/wr_dynamic_backtest.py): a
    trailing up-to-4-week real actual-PPR average when a player has at
    least one real prior 2025 week, falling back to the real static
    season-long EPA x volume projection otherwise - see module docstring
    and _wr_static_fallback()."""
    out = _wr_static_fallback()

    def _project_row(row):
        trailing = _trailing_recent_form(row["player_id"], int(row["week"]), form_lookup)
        if trailing:
            return pd.Series({
                "projected_fantasy_pts": round(float(np.mean(trailing)), 2),
                "projection_type": "weekly",
                "source": ("wr_recent_form_trailing_avg (real trailing up-to-4-week actual PPR - real "
                           "leave-one-out backtest winner, corr +0.4414 vs. the static baseline's real "
                           "leak-free +0.3957 - see src/wr_dynamic_backtest.py)"),
            })
        return pd.Series({
            "projected_fantasy_pts": float(row["static_per_game_projected_pts"]),
            "projection_type": "season_static_per_game_avg",
            "source": ("wr_epa_volume_formula_2025 (fantasy_validation.py: real static SEASON total, "
                       "in-sample-calibrated to PPR units, divided by real expected_games_2025 - used only "
                       "for a player's first real 2025 appearance, before any real trailing form exists)"),
        })

    projected = out.apply(_project_row, axis=1)
    out = pd.concat([out, projected], axis=1)
    out["position"] = "WR"
    return out


def _defense_rank_lookup():
    """team -> real trailing (through week W-1) rank (1=stingiest) vs each
    position, for every real week 2..18 in season 2025. Week 1 has no real
    trailing data (left absent, not fabricated)."""
    def_epa = build_defense_epa_by_position_multi_season([SEASON])
    weeks = sorted(def_epa["week"].unique())
    ranks = {}
    for w in weeks:
        if w <= 1:
            continue
        trailing = def_epa[def_epa["week"] < w]
        for position in ["QB", "RB", "WR", "TE"]:
            pos_trailing = trailing[trailing["position"] == position]
            if len(pos_trailing) == 0:
                continue
            team_avg = pos_trailing.groupby("team")["epa_allowed_per_play"].mean()
            team_rank = team_avg.rank(ascending=True, method="min").astype(int)  # lower EPA allowed = rank 1
            ranks[(w, position)] = team_rank.to_dict()
    return ranks


def _recent_form_lookup():
    """player_id -> sorted [(week, real actual PPR), ...] for real 2025 REG
    weeks. Real column is `fantasy_points_ppr` (the spec's pasted script
    assumed `ppr_points`, which doesn't exist in player_weekly_stats.csv -
    caught before running). Season/season_type filtered explicitly - week
    numbers repeat across seasons, so an unfiltered join would silently mix
    in real 2024 rows for the same (player_id, week)."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    pws = pws[(pws["season"] == SEASON) & (pws["season_type"] == "REG")]
    lookup = {}
    for pid, g in pws.groupby("player_id"):
        lookup[pid] = sorted(zip(g["week"], g["fantasy_points_ppr"].fillna(0.0)))
    return lookup


def _trailing_recent_form(player_id, week, lookup, n=4):
    history = lookup.get(player_id, [])
    prior = [round(float(pts), 1) for wk, pts in history if wk < week]
    return prior[-n:] if prior else None


def _actual_ppr(player_id, week, lookup):
    """Real actual PPR for THIS week (not trailing) - reuses the same
    already-correct table _trailing_recent_form draws from. None if the
    week hasn't happened / no real stat row exists for this player-week."""
    history = lookup.get(player_id, [])
    for wk, pts in history:
        if wk == week:
            return round(float(pts), 1)
    return None


def _accuracy_tier_thresholds(records):
    """Empirical per-position terciles of the real |actual - projected|
    distribution (only over records with a real actual_ppr) - not an
    asserted flat +-2/+-5 threshold. Returns {position: (low, high)}."""
    df = pd.DataFrame([r for r in records if r["actual_ppr"] is not None])
    thresholds = {}
    for position, g in df.groupby("position"):
        diffs = (g["actual_ppr"] - g["projected_ppr"]).abs()
        low, high = diffs.quantile([0.33, 0.67])
        thresholds[position] = (float(low), float(high))
    return thresholds


def _accuracy_tier(actual_ppr, projected_ppr, position, thresholds):
    if actual_ppr is None or position not in thresholds:
        return None
    diff = abs(actual_ppr - projected_ppr)
    low, high = thresholds[position]
    if diff <= low:
        return "green"
    if diff <= high:
        return "yellow"
    return "red"


def _injury_status_lookup(season=2025):
    """Real weekly injury report, verified schema (see DASHBOARD_DATA_GAPS.md
    Phase 2 notes): gsis_id (== this project's player_id), week,
    report_status in {None, "Questionable", "Doubtful", "Out"}, one row per
    player-week - no duplicate-join risk. No entry for a player-week means
    that player wasn't on the real injury report at all that week, which is
    the correct real-world reading of "healthy" (only players with an
    actual injury concern get listed), not an assumption filled in for
    missing data."""
    inj = load_injuries(season).to_pandas()
    inj = inj.rename(columns={"gsis_id": "player_id"})
    lookup = {}
    for _, r in inj.iterrows():
        lookup[(r["player_id"], int(r["week"]))] = r["report_status"] if pd.notna(r["report_status"]) else None
    return lookup


def _simplify_injury_status(raw_status):
    if raw_status is None:
        return "healthy"
    s = str(raw_status).lower()
    if s in ("out", "doubtful"):
        return "out"
    if s == "questionable":
        return "questionable"
    return "healthy"


def _team_week_opponent_map():
    sched = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    s = sched[(sched["season"] == SEASON) & (sched["game_type"] == "REG")]
    home = s[["week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"})
    away = s[["week", "home_team", "away_team"]].rename(columns={"away_team": "team", "home_team": "opponent"})
    opp = pd.concat([home, away], ignore_index=True)
    return {(int(r["week"]), r["team"]): r["opponent"] for _, r in opp.iterrows()}


def generate_fantasy_rankings_json():
    form_lookup = _recent_form_lookup()

    all_proj = pd.concat(
        [_rb_projections(), _qb_or_te_projections("QB"), _qb_or_te_projections("TE"), _wr_projections(form_lookup)],
        ignore_index=True)
    all_proj = all_proj.dropna(subset=["team", "projected_fantasy_pts"])
    all_proj["projected_fantasy_pts"] = all_proj["projected_fantasy_pts"].round(1)

    all_proj["rank"] = all_proj.groupby(["week", "position"])["projected_fantasy_pts"] \
        .rank(ascending=False, method="first").astype(int)
    all_proj = all_proj.sort_values(["week", "position", "rank"])

    print("Computing real trailing defense ranks vs position (reads play-by-play, ~20s)...")
    def_ranks = _defense_rank_lookup()
    opp_map = _team_week_opponent_map()
    injury_lookup = _injury_status_lookup(SEASON)

    records = []
    for _, r in all_proj.iterrows():
        week, team, position = int(r["week"]), r["team"], r["position"]
        player_id = r["player_id"]
        opponent = opp_map.get((week, team))
        defense_rank = None
        if opponent is not None:
            defense_rank = def_ranks.get((week, position), {}).get(opponent)

        raw_injury = injury_lookup.get((player_id, week))
        projected_ppr = float(r["projected_fantasy_pts"])

        records.append({
            "id": f"{player_id}_w{week}",
            "week": week,
            "position": position,
            "rank": int(r["rank"]),
            "name": r["player_name"],
            "team": team,
            "projected_ppr": projected_ppr,
            "actual_ppr": _actual_ppr(player_id, week, form_lookup),
            "projection_type": r["projection_type"],
            "opponent": opponent,
            "opponent_defense_rank_vs_position": defense_rank,
            "recent_form": _trailing_recent_form(player_id, week, form_lookup),
            "injury_status": _simplify_injury_status(raw_injury),
            "injury_status_raw": raw_injury,
            "source": r["source"],
        })

    accuracy_thresholds = _accuracy_tier_thresholds(records)
    for rec in records:
        rec["accuracy_tier"] = _accuracy_tier(rec["actual_ppr"], rec["projected_ppr"], rec["position"], accuracy_thresholds)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        record_generation("fantasy_rankings_2025")

    by_pos = all_proj.groupby("position").size()
    n_with_rank = sum(1 for r in records if r["opponent_defense_rank_vs_position"] is not None)
    n_with_actual = sum(1 for r in records if r["actual_ppr"] is not None)
    print(f"Generated {len(records)} player-week projections -> {OUTPUT_PATH}")
    print(by_pos.to_string())
    print(f"Records with a real defense rank attached: {n_with_rank}/{len(records)} (week 1 has none - no real trailing data yet)")
    print(f"Records with real actual_ppr attached: {n_with_actual}/{len(records)}")
    print(f"Empirical per-position accuracy-tier thresholds (|actual-projected|, terciles): {accuracy_thresholds}")
    return records


if __name__ == "__main__":
    generate_fantasy_rankings_json()
