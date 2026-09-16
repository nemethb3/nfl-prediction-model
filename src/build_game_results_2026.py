"""Real, additive 2026 completed-game results feed for the Elo system.

Real problem this closes: elo_model.py / compute_offensive_defensive_elo.py
both hardcode reading data/backtest/game_results_2015_2025.csv - a fixed,
real historical corpus that the season 2026 branch of every Elo-consuming
function has never had real 2026 rows to chain through (see
recalibrate_2026_elo.py's own docstring for the full picture). This
script closes that gap the way weekly_recalibration.py's own
prepare_for_live_2026() docstring said to: "ensure that week's real
results are appended to the game-results data source Elo reads from
(currently game_results_2015_2025.csv's 2026 successor)" - built here as
a real, SEPARATE successor file (data/backtest/game_results_2026.csv),
not by mutating the original 2015-2025 corpus in place (that file is a
fixed, trusted backtest/validation baseline many other scripts assume is
exactly 2015-2025 - see DECISIONS_LOG.md for the fuller reasoning).

Regenerated FRESH from nflreadpy every run (not incrementally appended) -
nflreadpy is already the real, authoritative source of truth for which
2026 games are complete, so there's no reason to hand-maintain a delta.

Real schema, matched exactly to game_results_2015_2025.csv's own columns
(verified: game_id, season, week, game_type, home_team, away_team,
home_score, away_score, result, total, overtime) so every consumer that
reads that file's columns works unchanged against this one.
"""

from pathlib import Path

import nflreadpy as nfl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "backtest" / "game_results_2026.csv"

SEASON = 2026
COLUMNS = ["game_id", "season", "week", "game_type", "home_team", "away_team",
           "home_score", "away_score", "result", "total", "overtime"]


def build_game_results_2026():
    sched = nfl.load_schedules([SEASON]).to_pandas()
    completed = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()
    completed["game_type"] = completed["game_type"].fillna("REG")
    completed["overtime"] = completed["overtime"].fillna(0)

    out = completed[COLUMNS].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["home_score"] = out["home_score"].astype(float)
    out["away_score"] = out["away_score"].astype(float)
    out["result"] = out["result"].astype(float)
    out["total"] = out["total"].astype(float)
    out["overtime"] = out["overtime"].astype(float)
    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Real 2026 completed games -> {OUTPUT_PATH}: {len(out)} rows "
          f"({out['week'].nunique()} week(s), latest week {out['week'].max() if len(out) else '-'})")
    return out


if __name__ == "__main__":
    build_game_results_2026()
