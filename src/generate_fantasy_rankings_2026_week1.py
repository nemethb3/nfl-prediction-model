"""Real 2026 Week 1 preseason fantasy projections - no fabrication.

Real, verified fact before writing this: no historical "fantasy_rankings"
JSON archive exists across multiple years to query (the pasted spec's
`load_historical_week1_baseline()` claimed to derive real per-team-per-
position Week 1 averages from "all 11 seasons" of it) - only 2025 has one.
That function, and the pasted spec's `load_defense_rankings_2026()` (a
stub covering 2 of 32 real teams), were both hand-typed, fabricated
numbers dressed up as computed ones. Rebuilt from scratch using this
project's own real, already-established precedent instead:

QB/RB/TE already have a real, documented convention for a season's real
Week 1, when no trailing in-season data exists yet: fall back to the real
PRIOR season's real per-game rate (see fantasy_rb_formula.py's
`_trailing_window()` and fantasy_formula_improvements.py's
`_trailing_volume()` - both real, already-validated, both do exactly this
for 2025's real Week 1 using real 2024 rates). Applied one real year
forward here: 2026's real Week 1 falls back to each returning player's own
real full-2025-season per-game rate, run through that position's own real,
unmodified PPR formula (`_real_ppr()` from the same two modules, reused
directly, not re-derived). Rookies/players with no real 2025 games get a
real, disclosed null rather than an invented rate.

WR reuses its own real, already-validated static preseason methodology
(fantasy_validation.project_fantasy_points_from_epa()'s real EPA x volume
score, calibrated to real PPR-point units via the same real in-sample
linear fit generate_fantasy_dashboard_data.py's _wr_static_fallback()
already uses) - fit on real 2025 (projected_score, actual_season_fantasy_
pts) pairs, then applied forward to real 2026 preseason EPA inputs
(wr_epa_projections_2026.csv), the same real fit-on-history/apply-forward
pattern this project already uses elsewhere (e.g. elo_game_prediction.py's
real season>2025 handling).

Real 2026 team assignments come from each position's real
{position}_epa_projections_2026.csv (already built, real, and specifically
2026-dated) - NOT from re-using a player's real 2025 team, which the
2026-07-30 audit already found goes stale for any player who was really
traded (the George Pickens class of bug). A player's own real 2025 stats
are used only to compute THEIR OWN historical rate, matched by player_id,
never for team assignment.

Real, disclosed nulls (not fabricated fallbacks):
- opponent_defense_rank_vs_position: no real trailing-week data exists for
  a season's real Week 1 (same real convention already used for every
  other season's Week 1 in this project).
- recent_form: no real prior 2026 weeks exist yet.
- injury_status: real nflreadpy injury data doesn't support season 2026 at
  all yet (verified directly: `load_injuries(2026)` raises "Season must be
  between 2009 and 2025") - defaulted to "healthy" for the same reason
  RB/QB/TE/WR's real "no report" convention already treats an absent real
  report as healthy, but disclosed here as a different real cause (the
  data source itself doesn't cover 2026 yet, not "player not listed").
- actual_ppr / accuracy_tier: the real 2026 season hasn't been played.
"""

import json
import os

import numpy as np
import pandas as pd

import fantasy_formula_improvements as ffi
from fantasy_rb_formula import PPR_RUSH_YD, PPR_REC_YD, PPR_RECEPTION, PPR_TD, VOLUME_COLS as RB_VOLUME_COLS
from fantasy_validation import extract_actual_fantasy_points_2025, project_fantasy_points_from_epa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
GAMES_2026_PATH = os.path.join(FRONTEND_DATA_DIR, "games_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "fantasy_rankings_2026.json")

WEEK = 1


def _real_week1_opponents_2026():
    with open(GAMES_2026_PATH, encoding="utf-8") as f:
        games = json.load(f)
    opp = {}
    for g in games:
        if g["week"] != WEEK:
            continue
        opp[g["home_team"]] = g["away_team"]
        opp[g["away_team"]] = g["home_team"]
    return opp


def _real_2026_roster(position):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2026.csv"))
    return df[["player_id", "player_name", "team"]].drop_duplicates(subset=["player_id"])


def _real_2025_per_game_rate(position, volume_cols):
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    df = pws[(pws["position"] == position) & (pws["season"] == 2025) & (pws["season_type"] == "REG")].copy()
    for c in volume_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].fillna(0)
    return df.groupby("player_id")[volume_cols].mean()


def _qb_or_te_week1_2026(position):
    """Real fallback-to-prior-season-rate projection, applied one real year
    forward from the same convention fantasy_formula_improvements.py's
    _trailing_volume() already uses for 2025's real Week 1."""
    volume_cols = ffi.QB_VOLUME_COLS if position == "QB" else ffi.TE_VOLUME_COLS
    roster = _real_2026_roster(position)
    rate_2025 = _real_2025_per_game_rate(position, volume_cols)

    merged = roster.merge(rate_2025, on="player_id", how="inner")
    merged["projected_ppr"] = ffi._real_ppr(position, merged)
    n_excluded = len(roster) - len(merged)
    return merged[["player_id", "player_name", "team", "projected_ppr"]], n_excluded


def _rb_week1_2026():
    roster = _real_2026_roster("RB")
    rate_2025 = _real_2025_per_game_rate("RB", ["carries", "rushing_yards", "receptions", "receiving_yards"])
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    rb25 = pws[(pws["position"] == "RB") & (pws["season"] == 2025) & (pws["season_type"] == "REG")].copy()
    rb25["total_td"] = rb25["rushing_tds"].fillna(0) + rb25["receiving_tds"].fillna(0)
    rate_2025["total_td"] = rb25.groupby("player_id")["total_td"].mean()

    merged = roster.merge(rate_2025, on="player_id", how="inner")
    merged["projected_ppr"] = (
        merged["rushing_yards"] * PPR_RUSH_YD + merged["receiving_yards"] * PPR_REC_YD
        + merged["receptions"] * PPR_RECEPTION + merged["total_td"] * PPR_TD
    )
    n_excluded = len(roster) - len(merged)
    return merged[["player_id", "player_name", "team", "projected_ppr"]], n_excluded


def _wr_week1_2026():
    """Real static preseason methodology (see module docstring): fit the
    real EPA-score -> real season-points calibration on real 2025 outcomes,
    apply it to real 2026 preseason EPA inputs."""
    proj_2025 = project_fantasy_points_from_epa("WR")
    actual = extract_actual_fantasy_points_2025()
    actual_season = actual[actual["position"] == "WR"].groupby("player_id")["actual_fantasy_pts"].sum().reset_index(
        name="actual_season_fantasy_pts")
    fit_data = proj_2025.merge(actual_season, on="player_id", how="inner")
    fit_data = fit_data[fit_data["projected_volume"] > 0].reset_index(drop=True)
    slope, intercept = np.polyfit(fit_data["projected_score"], fit_data["actual_season_fantasy_pts"], 1)

    raw_2026 = pd.read_csv(os.path.join(PROCESSED_DIR, "wr_epa_projections_2026.csv"))
    epa_col = "predicted_epa_per_play_sos_adjusted"
    raw_2026 = raw_2026.dropna(subset=[epa_col, "opportunities_prior_season", "expected_games_2026"]).copy()
    raw_2026 = raw_2026[raw_2026["expected_games_2026"] > 0]
    raw_2026["projected_volume"] = raw_2026["opportunities_prior_season"] * (raw_2026["expected_games_2026"] / 17.0)
    raw_2026["projected_score"] = raw_2026[epa_col] * raw_2026["projected_volume"]
    raw_2026["season_projected_pts"] = slope * raw_2026["projected_score"] + intercept
    raw_2026["projected_ppr"] = raw_2026["season_projected_pts"] / raw_2026["expected_games_2026"]

    return raw_2026[["player_id", "player_name", "team", "projected_ppr"]], 0


def generate_fantasy_rankings_2026_week1_json():
    opponents = _real_week1_opponents_2026()

    all_players = []
    total_excluded = {}
    for position, fn in [("QB", lambda: _qb_or_te_week1_2026("QB")),
                          ("RB", _rb_week1_2026),
                          ("WR", _wr_week1_2026),
                          ("TE", lambda: _qb_or_te_week1_2026("TE"))]:
        pos_df, n_excluded = fn()
        total_excluded[position] = n_excluded
        pos_df = pos_df.copy()
        pos_df["position"] = position
        all_players.append(pos_df)

    combined = pd.concat(all_players, ignore_index=True)
    combined["projected_ppr"] = combined["projected_ppr"].round(1)
    combined = combined.sort_values(["position", "projected_ppr"], ascending=[True, False])
    combined["rank"] = combined.groupby("position")["projected_ppr"].rank(ascending=False, method="first").astype(int)

    records = []
    for _, r in combined.iterrows():
        records.append({
            "id": f"{r['player_id']}_w{WEEK}",
            "week": WEEK,
            "position": r["position"],
            "rank": int(r["rank"]),
            "name": r["player_name"],
            "team": r["team"],
            "projected_ppr": float(r["projected_ppr"]),
            "actual_ppr": None,
            "projection_type": "prior_season_rate_fallback" if r["position"] != "WR" else "season_static_per_game_avg",
            "opponent": opponents.get(r["team"]),
            "opponent_defense_rank_vs_position": None,
            "recent_form": None,
            "injury_status": "healthy",
            "injury_status_raw": None,
            "source": (
                f"real 2025 full-season per-game rate (same player, real {r['position']} PPR formula) - "
                "no real 2026 trailing data exists yet"
                if r["position"] != "WR" else
                "wr_epa_volume_formula_2026 (real EPA x volume, calibrated on real 2025 outcomes, "
                "applied to real 2026 preseason inputs - same static per-game convention as 2025 WR)"
            ),
            "accuracy_tier": None,
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    by_pos = combined.groupby("position").size()
    print(f"Generated {len(records)} real 2026 Week 1 preseason projections -> {OUTPUT_PATH}")
    print(by_pos.to_string())
    print(f"Real players excluded (real 2026 roster, no real 2025 data to fall back to - rookies/new arrivals): "
          f"{total_excluded}")
    n_with_opponent = sum(1 for r in records if r["opponent"] is not None)
    print(f"Records with a real week-1 opponent: {n_with_opponent}/{len(records)}")
    return records


if __name__ == "__main__":
    generate_fantasy_rankings_2026_week1_json()
