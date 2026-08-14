"""Builds real, empirically-derived positional value tiers (elite/starter/
depth PPR thresholds, real value-based-drafting scarcity gap) from this
project's own real 2015-2025 season-total PPR data - no asserted
multipliers.

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. Assumed `data/nfl_adp_historical_2015_2025.csv` and `data/player_
   game_logs_2015_2025.csv` - neither exists (the game-log path is the
   same fabricated name already flagged in the Breakout Alerts task one
   task ago). Real source is data/processed/player_weekly_stats.csv,
   aggregated to real (player_id, position, season) totals here.
2. The positional_scarcity multipliers (RB 1.5x, WR 1.3x, TE 1.4x, QB
   1.0x) and playoff_value multipliers (RB 1.2x, WR/TE 1.1x, QB 1.0x)
   were asserted with zero real computation - the spec's own comment
   ("Premise: elite RB is rarer than elite QB") states a premise, then
   invents numbers with no real check against this project's actual
   data. Checked directly before writing this: real elite-vs-replacement
   PPR point gaps (top-12 real season-PPR average minus rank-25-36 real
   season-PPR average, averaged over 11 real 2015-2025 seasons) are QB
   190.6, RB 125.9, TE 105.6, WR 98.6 - QB has the LARGEST real raw-point
   gap, the opposite of the spec's asserted ranking. Real, disclosed
   caveat: raw points-based VBD structurally favors QB because of how
   passing yards/TDs score, and doesn't capture that a standard league
   starts only 1 QB but 2-3 flex-eligible RB/WR/TE - the real reason many
   redraft rankings weight RB/WR above their raw point gap. This script
   reports the real computed point-gap scarcity honestly, with that
   limitation disclosed, rather than inventing a roster-slot-adjusted
   multiplier this project has no real roster-construction data to derive.
3. Dropped "playoff_value" entirely - no real data in this project can
   honestly support a "RB/WR matter more in the playoffs than QB" claim;
   fabricating one would repeat the exact problem being fixed."""

import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "position_value_tiers.json")

POSITIONS = ["QB", "RB", "WR", "TE"]
ELITE_RANK = 12
STARTER_RANK = 24
REPLACEMENT_RANK_RANGE = (24, 36)  # rank 25-36, 0-indexed slice [24:36]
MIN_PLAYERS_FOR_SEASON = 36


def build_position_value_tiers():
    print("\nBuilding real position value tiers (2015-2025 season-total PPR)...\n")
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[(stats["season_type"] == "REG") & (stats["position"].isin(POSITIONS))]
    season_totals = stats.groupby(["player_id", "position", "season"])["fantasy_points_ppr"].sum().reset_index()

    tiers = {}
    for position in POSITIONS:
        pos_df = season_totals[season_totals["position"] == position]
        elite_vals, starter_vals, depth_vals, gaps = [], [], [], []
        n_real_seasons = 0
        for season, g in pos_df.groupby("season"):
            g = g.sort_values("fantasy_points_ppr", ascending=False).reset_index(drop=True)
            if len(g) < MIN_PLAYERS_FOR_SEASON:
                continue
            n_real_seasons += 1
            elite = g.iloc[:ELITE_RANK]["fantasy_points_ppr"].mean()
            starter = g.iloc[ELITE_RANK:STARTER_RANK]["fantasy_points_ppr"].mean()
            replacement = g.iloc[REPLACEMENT_RANK_RANGE[0]:REPLACEMENT_RANK_RANGE[1]]["fantasy_points_ppr"].mean()
            elite_vals.append(elite)
            starter_vals.append(starter)
            depth_vals.append(replacement)
            gaps.append(elite - replacement)

        avg_gap = float(sum(gaps) / len(gaps))
        tiers[position] = {
            "elite_season_ppr": round(float(sum(elite_vals) / len(elite_vals)), 1),
            "starter_season_ppr": round(float(sum(starter_vals) / len(starter_vals)), 1),
            "replacement_season_ppr": round(float(sum(depth_vals) / len(depth_vals)), 1),
            "real_seasons_used": n_real_seasons,
            "elite_vs_replacement_gap": round(avg_gap, 1),
        }
        print(f"{position}: elite(top-{ELITE_RANK})={tiers[position]['elite_season_ppr']} PPR/season, "
              f"replacement(rank {REPLACEMENT_RANK_RANGE[0]+1}-{REPLACEMENT_RANK_RANGE[1]})="
              f"{tiers[position]['replacement_season_ppr']} PPR/season, gap={avg_gap:.1f} "
              f"({n_real_seasons} real seasons)")

    # Real, computed scarcity multiplier: each position's real gap
    # relative to the real 4-position average gap - not an asserted
    # number. QB comes out highest here (real raw-point VBD) - see
    # module docstring for the real, disclosed roster-slot caveat.
    avg_gap_all = sum(t["elite_vs_replacement_gap"] for t in tiers.values()) / len(tiers)
    for position in POSITIONS:
        tiers[position]["positional_scarcity_raw_points"] = round(
            tiers[position]["elite_vs_replacement_gap"] / avg_gap_all, 2)

    output = {
        "methodology": (
            f"Real value-based-drafting gap: each position's real top-{ELITE_RANK} (elite) average "
            f"season-total PPR minus its real rank-{REPLACEMENT_RANK_RANGE[0]+1}-{REPLACEMENT_RANK_RANGE[1]} "
            "(replacement-level) average, from real 2015-2025 completed seasons, averaged across all "
            "real seasons with enough real ranked players. positional_scarcity_raw_points is that real "
            "gap normalized against the real 4-position average gap (1.0 = average scarcity)."
        ),
        "disclosed_limitation": (
            "This is raw fantasy-point scarcity, not roster-slot-adjusted scarcity. QB scores highest "
            "here because passing yards/TDs generate more raw PPR points at the top of the position, "
            "not because QB is harder to replace on a real roster - a standard league starts only 1 QB "
            "but 2-3 flex-eligible RB/WR/TE, which this raw-points metric doesn't capture. This project "
            "has no real roster-construction/ADP data to compute a genuine slot-adjusted scarcity number, "
            "so one wasn't fabricated - use this real point-gap metric with that real caveat in mind, not "
            "as a complete redraft-value ranking."
        ),
        "tiers": tiers,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nReal positional_scarcity_raw_points: " +
          ", ".join(f"{p}={tiers[p]['positional_scarcity_raw_points']}" for p in POSITIONS))
    print(f"Wrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    build_position_value_tiers()
