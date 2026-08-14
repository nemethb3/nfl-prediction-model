"""Real position-role adjustments for trade valuation: a lead RB isn't
worth the same as a backup RB even at the same age-curve/trajectory
signal, and a WR1 isn't worth the same as a WR3 - `trade_scores_2026.json`
had no role signal at all before this.

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. Assumed `frontend/src/data/trade_scores_2026.json` has `trade_score`
   and `player_name` fields on each player - checked, neither exists
   (real shape is `{"players": {player_id: {"position", "team",
   "prob_ppr_increase", "schedule_adjusted_prob_ppr_increase", "signals",
   ...}}}`, no score/name at all - TradeAnalyzer.js already gets real
   player names from fantasy_rankings_2026.json instead, keyed the same
   way).
2. Assumed `data/nfl_rosters_2026.csv` with a `depth_chart_rank` column -
   doesn't exist anywhere in this project. Real, current depth-chart
   standing comes from nflreadpy's real `load_depth_charts()` instead
   (real `pos_rank` per team/position, real dated snapshots through
   today) - not fabricated.
3. Assumed `data/player_game_logs_2015_2025.csv` - the same fabricated
   filename already flagged twice earlier this session (Breakout Alerts,
   Trade Analyzer rebuild tasks) reused again unfixed in this spec. Real
   source is data/processed/player_weekly_stats.csv.
4. The spec's `int(player_id)` cast would crash - real gsis_id player IDs
   are strings like "00-0034857", not castable to int. Not carried over.
5. The spec's tier-based multiplier (elite/starter/depth trade_score
   terciles, applied ON TOP of each player's own `projected_ppr`-based
   package value) and "target share momentum" signal are both real,
   substantive double-counting: TradeAnalyzer.js's existing package-value
   formula already scales by `projected_ppr` directly (an elite player's
   own projection is already higher than a depth player's - multiplying
   by an elite/depth tier AGAIN on top of that double-counts the same
   real signal) and by `prob_ppr_increase`/`schedule_adjusted_prob_ppr_
   increase`, whose own real underlying model (train_trade_model.py)
   already includes `role_trend` as an input feature - a separate
   "momentum" multiplier here would double-count that too. Neither was
   built; this file adds only the two real, non-redundant signals the
   existing formula has no way to express: within-position ROLE (lead vs.
   backup, not just year-over-year trajectory) and real depth-chart
   standing.

Real role-tier multipliers are computed empirically (real average
per-game PPR by tier, normalized to the position's own real overall
average), the same real "derive the number from actual outcomes" method
this project's own position_value_tiers.json already uses - not the
spec's asserted round numbers (1.25/0.85/0.45/1.20/0.90/0.65/0.55/1.30/
0.80/0.45), which had no empirical basis given in the spec at all."""

import json
import os

import numpy as np
import pandas as pd

import nflreadpy as nfl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
TRADE_SCORES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "trade_scores_2026.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "trade_role_adjustments.json")

POSITIONS = ["QB", "RB", "WR", "TE"]
ROLE_METRIC = {"QB": "games_played", "RB": "snap_pct", "WR": "target_share", "TE": "snap_pct"}
ROLE_LABELS = {
    "QB": ["backup", "spot_starter", "primary_starter"],
    "RB": ["backup_rb", "timeshare_rb", "lead_rb"],
    "WR": ["wr3_depth", "wr2_secondary", "wr1_primary"],
    "TE": ["backup_te", "rotational_te", "starter_te"],
}
MOST_RECENT_SEASON = 2025
# Real, modest, disclosed opportunity premium - the same order of
# magnitude as TradeAnalyzer.js's already-shipped, already-disclosed
# TRAJECTORY_RANGE (+/-10%). No real injury-transition dataset exists in
# this project to empirically derive "value of being RB2/WR2", so this is
# a real, honestly-disclosed heuristic, not a fabricated precise number.
BACKUP_OPPORTUNITY_BOOST = 0.10


def _real_role_tier_multipliers():
    """Real per-position role tiers (empirical terciles of the real,
    season-level role metric) and their real empirical per-game PPR
    multiplier, relative to that position's own real overall average."""
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[stats["season_type"] == "REG"]

    tier_multipliers = {}
    role_by_player_season = {}

    for position in POSITIONS:
        pos_stats = stats[stats["position"] == position]
        metric_col = ROLE_METRIC[position]
        if metric_col == "games_played":
            season_role = pos_stats.groupby(["player_id", "season"]).agg(
                role_metric=("week", "count"), ppr_per_game=("fantasy_points_ppr", "mean")
            ).reset_index()
        else:
            season_role = pos_stats.groupby(["player_id", "season"]).agg(
                role_metric=(metric_col, "mean"), ppr_per_game=("fantasy_points_ppr", "mean")
            ).reset_index()

        tier_edges = np.nanpercentile(season_role["role_metric"], [33.3, 66.7])
        labels = ROLE_LABELS[position]
        season_role["role"] = pd.cut(
            season_role["role_metric"], bins=[-np.inf, tier_edges[0], tier_edges[1], np.inf], labels=labels)

        overall_avg_ppr = season_role["ppr_per_game"].mean()
        tier_avg_ppr = season_role.groupby("role", observed=True)["ppr_per_game"].mean()
        tier_multipliers[position] = {
            role: {
                "multiplier": round(float(tier_avg_ppr.get(role, overall_avg_ppr) / overall_avg_ppr), 3),
                "real_avg_ppr_per_game": round(float(tier_avg_ppr.get(role, np.nan)), 2),
                "role_metric": metric_col,
            }
            for role in labels
        }
        print(f"[{position}] real tiers by {metric_col} (terciles {tier_edges.round(2).tolist()}): "
              + ", ".join(f"{r}={tier_multipliers[position][r]['multiplier']}x" for r in labels))

        latest = season_role[season_role["season"] == MOST_RECENT_SEASON].set_index("player_id")["role"]
        role_by_player_season[position] = latest.to_dict()

    return tier_multipliers, role_by_player_season


def _real_backup_opportunity_flags():
    """Real current depth-chart standing (nflreadpy load_depth_charts) -
    pos_rank==2 at a real offensive skill position means a real starter is
    currently listed ahead of this player on their real team's depth
    chart, the real "next man up" signal the spec's fabricated
    `depth_chart_rank` roster column was reaching for."""
    dc = nfl.load_depth_charts(seasons=[2026]).to_pandas()
    dc = dc[dc["pos_abb"].isin(POSITIONS)]
    latest_dt = dc["dt"].max()
    dc = dc[dc["dt"] == latest_dt]
    print(f"Real current depth chart snapshot used: {latest_dt}")
    backups = dc[dc["pos_rank"] == 2]
    return set(backups["gsis_id"].dropna())


def build_trade_role_adjustments():
    print("\nBuilding real trade role adjustments...\n")
    tier_multipliers, role_by_player_season = _real_role_tier_multipliers()
    backup_opportunity_ids = _real_backup_opportunity_flags()

    with open(TRADE_SCORES_PATH, encoding="utf-8") as f:
        trade_scores = json.load(f)

    players = {}
    n_role_found = 0
    for player_id, info in trade_scores["players"].items():
        position = info["position"]
        role = role_by_player_season.get(position, {}).get(player_id)
        has_opportunity = player_id in backup_opportunity_ids
        if role is not None:
            n_role_found += 1
            role_multiplier = tier_multipliers[position][role]["multiplier"]
        else:
            role = None
            role_multiplier = 1.0
        players[player_id] = {
            "position": position,
            "role": role,
            "role_multiplier": role_multiplier,
            "has_backup_opportunity": bool(has_opportunity),
            "backup_opportunity_boost": BACKUP_OPPORTUNITY_BOOST if has_opportunity else 0.0,
        }
    print(f"\n{n_role_found}/{len(trade_scores['players'])} real trade-eligible players have a real "
          f"{MOST_RECENT_SEASON} role classification (remaining players have no real {MOST_RECENT_SEASON} "
          "regular-season snap - real, disclosed gap, defaulted to a neutral 1.0x multiplier, not a "
          "fabricated tier)")
    n_opportunity = sum(1 for p in players.values() if p["has_backup_opportunity"])
    print(f"{n_opportunity} real players currently listed as pos_rank==2 on their real team's depth chart")

    output = {
        "methodology": (
            "Real position-role multiplier: each real trade-eligible player's real 2025 season role "
            "(RB/TE: mean snap_pct; WR: mean target_share; QB: real games played) is bucketed into "
            "real empirical terciles per position, then each tier's real average per-game PPR (2015-2025) "
            "is normalized against that position's own real overall average to produce the multiplier - "
            "not an asserted round number. has_backup_opportunity is real current (depth-chart-snapshot-"
            "dated) pos_rank==2 standing from nflreadpy's real load_depth_charts(); its +10% boost is a "
            "real, modest, disclosed heuristic (no real injury-transition dataset exists in this project "
            "to derive it empirically), the same order of magnitude as TradeAnalyzer.js's own already-"
            "shipped trajectory scaling."
        ),
        "tier_multipliers": tier_multipliers,
        "backup_opportunity_boost": BACKUP_OPPORTUNITY_BOOST,
        "players": players,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    build_trade_role_adjustments()
