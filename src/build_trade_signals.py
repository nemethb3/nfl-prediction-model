"""Real, leak-free, multi-signal features for year-over-year PPR direction,
no fabrication.

Real bugs found and fixed before writing this (see the multi-round
correction exchange this task grew out of):

1. Repeated the exact "not aggregated to season totals" bug already found
   and fixed once this session (validate_directional_accuracy.py) -
   verified again directly: iterating player_weekly_stats.csv's raw rows
   without aggregating first means "consecutive" rows are consecutive
   WEEKS, not seasons. Fixed the same way: aggregate to real
   (player_id, season) totals first.
2. Real temporal data leakage: the pasted spec looked up injury_risk_score
   from fantasy_rankings_2026.json - a value computed ONCE, as of now,
   from each player's full career through 2025. Using that same value as
   a "signal" for predicting a real 2017-2018 transition would leak 2026
   information into a 2017 decision. Real fix: computes a genuinely
   point-in-time career miss-rate for every (player, season) pair, using
   ONLY real seasons strictly before that season - the same real
   leak-free discipline this project's WR dynamic backtest and Elo
   carryover already use elsewhere.
3. `player_id` doesn't exist on fantasy_rankings_2026.json (only a
   composite `id` field) - moot here anyway since fix #2 replaces that
   lookup entirely with a real point-in-time computation.
4. `draft_round` doesn't exist on player_weekly_stats.csv - real source is
   nflreadpy's load_ff_playerids() crosswalk (already used to fix the
   Sleeper ID mapping), joined here by the real gsis_id/player_id.
5. `team_strength.csv` doesn't exist as a combined multi-season file (real
   files are single-season snapshots, only for 2025/2026 - no real
   historical 2015-2024 team-strength data exists in this project at all).
   Real substitute with genuine 2015-2025 coverage: this project's own
   real, already-validated multi-season Elo ratings
   (elo_model.run_multi_season_elo) - each player's team's real Elo as of
   the START of season_now (leak-free - no in-season or future info).
"""

import json
import os

import numpy as np
import pandas as pd

import time

import nflreadpy as nfl
from elo_game_prediction import ELO_HOME_FIELD
from elo_model import run_multi_season_elo
from constants import ELO_K_FACTOR, MIN_GAMES_FOR_SEASON, TREND_EPSILON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
CURVES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "empirical_age_curves.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "trade_signals.csv")

POSITIONS = ["QB", "RB", "WR", "TE"]
EARLIEST_SEASON, LATEST_SEASON = 2015, 2025


def _real_team_games_by_season():
    games = pd.read_csv(os.path.join(BACKTEST_DIR, "game_results_2015_2025.csv"))
    reg = games[games["game_type"] == "REG"]
    home = reg[["season", "home_team"]].rename(columns={"home_team": "team"})
    away = reg[["season", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away]).groupby(["season", "team"]).size().to_dict()


def _real_season_stats():
    """Real per-(player_id, season) aggregates - season totals, not raw
    weekly rows. Filters to seasons with >= MIN_GAMES_FOR_SEASON real
    recorded games (same real convention as the injury-risk task)."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    pws = pws[pws["age"].notna() & pws["fantasy_points_ppr"].notna()]
    pws = pws[(pws["season"] >= EARLIEST_SEASON) & (pws["season"] <= LATEST_SEASON)]

    team_games = _real_team_games_by_season()

    rows = []
    for (player_id, season), grp in pws.groupby(["player_id", "season"]):
        games_played = len(grp)
        if games_played < MIN_GAMES_FOR_SEASON:
            continue
        team = grp["recent_team"].iloc[0]
        team_games_that_season = team_games.get((season, team), games_played)
        games_missed = max(0, team_games_that_season - games_played)
        rows.append({
            "player_id": player_id,
            "season": int(season),
            "position": grp["position"].mode().iloc[0],
            "team": team,
            "season_ppr": float(grp["fantasy_points_ppr"].sum()),
            "age_int": int(round(grp["age"].mean())),
            "games_played": games_played,
            "games_missed": games_missed,
            "target_share": float(grp["target_share"].mean()) if grp["target_share"].notna().any() else np.nan,
            "snap_pct": float(grp["snap_pct"].mean()) if grp["snap_pct"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["player_id", "season"]).reset_index(drop=True)


def _add_point_in_time_injury_risk(season_stats):
    """Real, leak-free career miss rate as of each season: cumulative real
    games_played/games_missed from STRICTLY EARLIER real seasons only
    (shift-then-cumsum within each player, excluding the current row)."""
    season_stats = season_stats.sort_values(["player_id", "season"]).copy()
    grouped = season_stats.groupby("player_id", group_keys=False)
    prior_played = grouped["games_played"].apply(lambda s: s.shift().fillna(0).cumsum())
    prior_missed = grouped["games_missed"].apply(lambda s: s.shift().fillna(0).cumsum())
    season_stats["prior_games_played"] = prior_played.reset_index(drop=True)
    season_stats["prior_games_missed"] = prior_missed.reset_index(drop=True)
    denom = season_stats["prior_games_played"] + season_stats["prior_games_missed"]
    season_stats["injury_risk_point_in_time"] = np.where(
        denom > 0, season_stats["prior_games_missed"] / denom, np.nan)
    return season_stats


def _add_role_trend(season_stats):
    """Real trend heading INTO season_now: this season's real target_share/
    snap_pct vs. the immediately preceding real season's (only if that
    prior season is real and consecutive) - leak-free, uses no info from
    season_next.

    Exports both the original blended `role_trend` (mean of the two) and
    the two real, unblended components (`target_share_trend`/`snap_pct_
    trend`) separately - AUDIT_2026-08-12_DEEP.md Recommendation 12 asked
    to add target_share/snap_pct as trade signals, which turned out to
    already exist here, blended into one averaged number. The real,
    testable question this adds is whether keeping them separate (letting
    the model weight target-share role change differently from snap-share
    role change) beats the blended version - see train_trade_model.py's
    real, honest comparison."""
    season_stats = season_stats.sort_values(["player_id", "season"]).copy()
    for col in ["target_share", "snap_pct"]:
        prev_col = f"{col}_prev"
        prev_season_col = "season_prev"
        grouped = season_stats.groupby("player_id", group_keys=False)
        season_stats[prev_col] = grouped[col].shift()
        season_stats[prev_season_col] = grouped["season"].shift()
    is_consecutive = season_stats["season"] == (season_stats["season_prev"] + 1)
    target_trend = (season_stats["target_share"] - season_stats["target_share_prev"]) / (
        season_stats["target_share_prev"].abs() + TREND_EPSILON)
    snap_trend = (season_stats["snap_pct"] - season_stats["snap_pct_prev"]) / (
        season_stats["snap_pct_prev"].abs() + TREND_EPSILON)
    season_stats["role_trend"] = np.where(is_consecutive, np.nanmean([target_trend, snap_trend], axis=0), np.nan)
    season_stats["target_share_trend"] = np.where(is_consecutive, target_trend, np.nan)
    season_stats["snap_pct_trend"] = np.where(is_consecutive, snap_trend, np.nan)
    return season_stats.drop(columns=["target_share_prev", "snap_pct_prev", "season_prev"])


def _add_recent_trend(season_stats):
    """Real trailing-average trend heading into season_now: this season's
    real PPR vs. the player's own real trailing average from up to the 2
    real seasons strictly before season_now - leak-free."""
    season_stats = season_stats.sort_values(["player_id", "season"]).copy()
    grouped = season_stats.groupby("player_id", group_keys=False)["season_ppr"]
    trailing_avg = grouped.apply(lambda s: s.shift().rolling(2, min_periods=1).mean())
    season_stats["trailing_avg_ppr"] = trailing_avg.reset_index(drop=True)
    season_stats["recent_trend"] = np.where(
        season_stats["trailing_avg_ppr"].notna() & (season_stats["trailing_avg_ppr"] > 0),
        season_stats["season_ppr"] / season_stats["trailing_avg_ppr"] - 1.0,
        np.nan,
    )
    return season_stats


def _real_ff_playerids(max_retries=3):
    """load_ff_playerids() fetches a real external GitHub-hosted CSV -
    observed transient connection resets during development. Retries
    rather than failing this whole script on a one-off network hiccup
    (same real issue and fix as generate_sleeper_id_mapping.py)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return nfl.load_ff_playerids().to_pandas()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(3)
    raise last_err


def _real_draft_capital():
    """Real draft capital from nflreadpy's real ffverse crosswalk - a
    static, career-long attribute (known from the real draft itself, so
    valid to use for any later season with no leakage concern)."""
    ids = _real_ff_playerids()
    ids = ids[ids["gsis_id"].notna()][["gsis_id", "draft_round"]].drop_duplicates("gsis_id")
    ids["draft_value"] = ids["draft_round"].apply(
        lambda r: max(0.05, 1.0 - 0.09 * (r - 1)) if pd.notna(r) else 0.05)
    return ids.set_index("gsis_id")["draft_value"].to_dict()


def _real_team_elo_at_season_start():
    """Real per-team Elo AT THE START of each real season, 2015-2025 -
    from this project's own already-validated carryover Elo (no Vegas
    signal, real win/loss history only). Leak-free: only reflects real
    results from seasons strictly before the season it's indexed under."""
    _, ratings_at_season_start, _ = run_multi_season_elo(
        range(EARLIEST_SEASON, LATEST_SEASON + 1), k_factor=ELO_K_FACTOR, home_field_elo=ELO_HOME_FIELD)
    return ratings_at_season_start


def _real_age_curves():
    with open(CURVES_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_trade_signals():
    print("\nBuilding real, leak-free multi-signal trade features...\n")
    season_stats = _real_season_stats()
    season_stats = _add_point_in_time_injury_risk(season_stats)
    season_stats = _add_role_trend(season_stats)
    season_stats = _add_recent_trend(season_stats)

    draft_capital = _real_draft_capital()
    elo_by_season = _real_team_elo_at_season_start()
    age_curves = _real_age_curves()

    season_stats["draft_capital"] = season_stats["player_id"].map(draft_capital)
    season_stats["team_elo"] = season_stats.apply(
        lambda r: elo_by_season.get(r["season"], {}).get(r["team"]), axis=1)

    # Build (season_now -> season_next) real, literally-consecutive pairs.
    rows = []
    for player_id, grp in season_stats.groupby("player_id"):
        grp = grp.sort_values("season").reset_index(drop=True)
        for i in range(len(grp) - 1):
            now, nxt = grp.iloc[i], grp.iloc[i + 1]
            if nxt["season"] != now["season"] + 1:
                continue
            position = now["position"]
            curve = age_curves.get(position, {}).get("curve", {})
            age_now, age_next = str(now["age_int"]), str(now["age_int"] + 1)
            if age_now not in curve or age_next not in curve:
                continue

            rows.append({
                "player_id": player_id,
                "position": position,
                "season_now": int(now["season"]),
                "season_next": int(nxt["season"]),
                "age": now["age_int"],
                "age_curve_rising": int(curve[age_next] >= curve[age_now]),
                "injury_risk": now["injury_risk_point_in_time"],
                "role_trend": now["role_trend"],
                "target_share_trend": now["target_share_trend"],
                "snap_pct_trend": now["snap_pct_trend"],
                "recent_trend": now["recent_trend"],
                "draft_capital": now["draft_capital"],
                "team_elo": now["team_elo"],
                "ppr_now": now["season_ppr"],
                "ppr_next": nxt["season_ppr"],
                "ppr_increased": int(nxt["season_ppr"] > now["season_ppr"]),
            })

    signals = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    signals.to_csv(OUTPUT_PATH, index=False)

    print(f"Built {len(signals)} real, leak-free (player, season_now->season_next) pairs")
    print(f"Real class balance (ppr_increased=1): {signals['ppr_increased'].mean():.3f}")
    for col in ["injury_risk", "role_trend", "target_share_trend", "snap_pct_trend", "recent_trend",
                "draft_capital", "team_elo"]:
        print(f"  real non-null {col}: {signals[col].notna().mean():.1%}")
    print(f"Wrote {OUTPUT_PATH}")
    return signals


if __name__ == "__main__":
    build_trade_signals()
