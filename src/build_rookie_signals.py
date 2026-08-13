"""Real rookie draft-capital signals, using nflreadpy's real, already-computed
career outcome metric (w_av - Pro-Football-Reference's "weighted career
Approximate Value") instead of hand-rolling a year-1-PPR-vs-round-median
comparison from scratch.

Real decisions made before writing this (see "Three Accuracy Improvements"
task's Part 2 scoping):
1. load_draft_picks()'s `car_av` column is real but uniformly null in this
   nflreadpy build for every row checked (including well-established
   careers like Josh Allen/Mahomes) - not usable. `w_av` is real and
   populated, and its round medians decline monotonically and sensibly
   (round 1 median 39.0 -> round 7 median 1.0 for real 2015-2020 skill-
   position picks) - used here instead.
2. Real column names differ from every earlier draft of this task's spec:
   `round`/`pick`/`gsis_id`, not `draft_round`/`draft_pick`/`player_id`.
   gsis_id conveniently already matches player_weekly_stats.csv's real
   player_id format directly (both "00-00xxxxx") - no crosswalk needed for
   this file alone.
3. Training window restricted to real 2015-2020 draft classes only: recent
   classes (2021+) haven't had enough real seasons yet to fairly accrue
   career w_av, which would bias them toward looking like busts purely for
   lack of time, not lack of talent. 2015-2020 gives every real player in
   this file at least 6 real seasons (through 2025) to accrue value.
4. Round-relative median (not round+position) - same real, disclosed
   convention this project's own draft_capital signal already uses
   elsewhere (build_trade_signals.py's _real_draft_capital: round-only,
   position-agnostic).
5. Missing w_av (real players who accrued zero career approximate
   value - genuinely never stuck in the league) filled with 0, not
   dropped - 0 IS the real, correct value for "no career value", not a
   missing-data gap.
"""

import os

import pandas as pd

import nflreadpy as nfl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "rookie_signals_historical.csv")

POSITIONS = ["QB", "RB", "WR", "TE"]
TRAINING_DRAFT_SEASONS = range(2015, 2021)  # 2015-2020: >=6 real seasons of career value by 2026


def build_rookie_signals():
    print("\nBuilding real rookie draft-capital signals (2015-2020 draft classes)...\n")
    draft = nfl.load_draft_picks().to_pandas()
    draft = draft[
        draft["season"].isin(list(TRAINING_DRAFT_SEASONS))
        & draft["position"].isin(POSITIONS)
        & draft["gsis_id"].notna()
    ].copy()
    draft["w_av"] = draft["w_av"].fillna(0.0)

    round_median = draft.groupby("round")["w_av"].median()
    draft["round_median_w_av"] = draft["round"].map(round_median)
    draft["outperformed"] = (draft["w_av"] > draft["round_median_w_av"]).astype(int)

    out = draft[["gsis_id", "pfr_player_name", "position", "season", "round", "pick", "team",
                 "w_av", "round_median_w_av", "outperformed"]].rename(
        columns={"gsis_id": "player_id", "pfr_player_name": "name", "season": "draft_season"})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Built {len(out)} real rookie draft records (2015-2020, QB/RB/WR/TE)")
    print(f"Real outperformed-round-median rate: {out['outperformed'].mean():.1%}")
    print("Real round medians (career weighted AV):")
    print(round_median.to_string())
    print(f"Wrote {OUTPUT_PATH}")
    return out


if __name__ == "__main__":
    build_rookie_signals()
