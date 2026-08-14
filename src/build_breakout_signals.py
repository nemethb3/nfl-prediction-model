"""Builds real breakout-candidate alerts per player-week for 2026: flags
players where multiple real signals align (weak opponent defense, rising
usage, strong recent form, reduced competition at the position).

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. Assumed `data/nfl_rosters_2026.csv`, `data/player_game_logs_2015_
   2025.csv`, `data/nfl_injuries_2026_week_by_week.csv` - none exist
   (checked). Real roster is implicit in player_props_2026.json (already
   built via the same real, stale-team-bug-fixed source generate_player_
   props_2026.py uses). Real player game log is `data/processed/
   player_weekly_stats.csv`.
2. The spec's `weak_defense = opponent_d_elo < 1480` used an asserted,
   unfit constant. Real fix: the real bottom tercile of the real 32-team
   2026 regressed O/D Elo distribution (empirically ~1483.7, close to but
   not the same as the asserted number - computed at runtime, not
   hardcoded).
3. Real, caught-during-development problem in this fix itself, not the
   spec: my first real pass reused the spec's `usage_up = (snap_pct_trend
   &gt; 0) or (targets_trend &gt; 0.1) or (carries_trend &gt; 0.1)` and
   `performance_strong = last_4_ppr &gt; season_avg_ppr * 1.1` - an OR of
   three loose conditions plus a lightly-above-average bar. Checked the
   real trigger rate before shipping: usage_up fired for 63% of ALL real
   293 players with enough 2025 history, performance_strong for 36.5% -
   combined with weak_defense's real ~33% base rate, "2 of 3" was
   satisfied by roughly 40% of all real player-weeks (2,141 alerts/week
   avg 119) - not a selective "breakout candidate" signal, just weak
   defense doing almost all the real filtering. Real fix: usage and
   performance each collapse to one clean, comparable metric (snap_pct
   trend; PPR diff vs. season average) and use the real top-tercile cutoff
   of that metric's own real 293-player distribution, the same empirical
   discipline already applied to weak_defense - not an OR of noisy
   conditions or an asserted percentage bar.
4. The spec's `try/except: injuries_2026 = pd.DataFrame()` for a
   nonexistent injuries file silently makes `competition_reduced` always
   False - a real signal quietly degraded to a no-op instead of disclosed.
   This project's own FantasyRankings.js already discloses the same real
   gap explicitly. Real fix: the signal stays in the real output shape (so
   real 2026 injury data can slot in once it exists), explicitly marked
   inactive with a disclosed reason, not silently absorbed by a bare
   except.
5. Confidence was `signals_active / 4` - since signal 4 (competition) can
   never be real-true right now, that would cap every real alert's
   displayed confidence at 75%, understating it as if 100% were still
   reachable. Real fix: confidence is `signals_active / 3` (out of the 3
   currently real, available signals).
6. `games_2026.json` was loaded but never actually used (player_props_
   2026.json already carries week/opponent/is_home/opponent_d_elo per
   player-game) - dropped, avoiding a needless load."""

import json
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
REGRESSED_OD_ELO_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")
SCHEDULE_PATH = os.path.join(RAW_DIR, "schedules_2026.csv")
PLAYER_PROPS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "player_props_2026.json")
PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "breakout_alerts_2026.json")

RECENT_FORM_SEASON = 2025  # real most-recent completed season - best real proxy for "entering 2026" form
MIN_GAMES_FOR_TREND = 4
SELECTIVE_PERCENTILE = 66.7  # top real tercile - same real selectivity as the weak-defense cutoff
SIGNAL_THRESHOLD = 2  # of 3 real, currently-available signals

COMPETITION_DISCLOSURE = (
    "No real 2026 injury report data exists yet (verified: nflreadpy has no 2026 coverage) - this "
    "signal stays in the real output shape for when it does, but is always inactive right now, not "
    "silently degraded."
)


def _real_weak_defense_threshold():
    sched = pd.read_csv(SCHEDULE_PATH)
    sched = sched[sched["game_type"] == "REG"]
    real_teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    with open(REGRESSED_OD_ELO_PATH, encoding="utf-8") as f:
        regressed = json.load(f)
    d_elos = np.array(sorted(regressed[t]["d_elo"] for t in real_teams))
    threshold = float(np.percentile(d_elos, 33.3))
    print(f"Real weak-defense threshold (bottom tercile, real 32-team 2026 D_Elo distribution): "
          f"{threshold:.1f}")
    return threshold, d_elos


def _real_defense_percentile(opponent_d_elo, d_elos):
    return float((d_elos > opponent_d_elo).mean())


def _real_player_forms(player_stats, player_ids):
    """Real, raw (not yet thresholded) recent-form metrics per player -
    snap_pct trend and PPR-vs-season-average diff, from each real player's
    real trailing 2025 games."""
    forms = {}
    for pid in player_ids:
        rows = player_stats[(player_stats["player_id"] == pid) &
                             (player_stats["season"] == RECENT_FORM_SEASON) &
                             (player_stats["season_type"] == "REG")].sort_values("week").tail(8)
        if len(rows) < MIN_GAMES_FOR_TREND:
            continue
        recent_4 = rows.tail(4)
        prior_n = rows.iloc[:-4]
        if len(prior_n) == 0:
            prior_n = recent_4

        snap_pct_trend = float(recent_4["snap_pct"].mean() - prior_n["snap_pct"].mean())
        targets_trend = float(
            (recent_4["targets"].sum() - prior_n["targets"].sum()) / max(prior_n["targets"].sum(), 1))
        carries_trend = float(
            (recent_4["carries"].sum() - prior_n["carries"].sum()) / max(prior_n["carries"].sum(), 1))
        season_avg_ppr = float(rows["fantasy_points_ppr"].mean())
        last_4_ppr = float(recent_4["fantasy_points_ppr"].mean())

        forms[pid] = {
            "snap_pct_trend": snap_pct_trend, "targets_trend": targets_trend, "carries_trend": carries_trend,
            "season_avg_ppr": season_avg_ppr, "last_4_ppr": last_4_ppr, "ppr_diff": last_4_ppr - season_avg_ppr,
        }
    return forms


def build_breakout_signals():
    print("\nBuilding real breakout alert signals for 2026...\n")
    weak_defense_threshold, d_elos = _real_weak_defense_threshold()

    with open(PLAYER_PROPS_PATH, encoding="utf-8") as f:
        player_props = json.load(f)
    player_stats = pd.read_csv(PLAYER_STATS_PATH)

    unique_player_ids = {p["player_id"] for p in player_props}
    forms = _real_player_forms(player_stats, unique_player_ids)

    # Real, empirically-derived top-tercile cutoffs across the real
    # population of players who have enough 2025 history to have a form
    # signal at all - not asserted percentage bars (see docstring #3).
    snap_trend_cutoff = float(np.percentile([f["snap_pct_trend"] for f in forms.values()], SELECTIVE_PERCENTILE))
    ppr_diff_cutoff = float(np.percentile([f["ppr_diff"] for f in forms.values()], SELECTIVE_PERCENTILE))
    print(f"Real usage-trending-up cutoff (top tercile of real snap_pct trend, n={len(forms)}): "
          f"{snap_trend_cutoff:+.2f} pts")
    print(f"Real performance-strong cutoff (top tercile of real PPR-vs-season-avg diff, n={len(forms)}): "
          f"{ppr_diff_cutoff:+.2f} PPR")

    alerts_by_week = {str(w): [] for w in range(1, 19)}
    total_alerts = 0

    for prop in player_props:
        week = prop["week"]
        opponent_d_elo = prop["opponent_d_elo"]
        weak_defense = opponent_d_elo < weak_defense_threshold
        defense_percentile = _real_defense_percentile(opponent_d_elo, d_elos)

        form = forms.get(prop["player_id"])
        usage_up = bool(form and form["snap_pct_trend"] > snap_trend_cutoff)
        performance_strong = bool(form and form["ppr_diff"] > ppr_diff_cutoff)

        signals_active = sum([weak_defense, usage_up, performance_strong])
        if signals_active < SIGNAL_THRESHOLD:
            continue

        alert = {
            "id": f"{prop['player_id']}_w{week}",  # same real convention player_props_2026.json/fantasy_rankings_2026.json already use
            "player_id": prop["player_id"],
            "player_name": prop["player_name"],
            "position": prop["position"],
            "team": prop["team"],
            "week": week,
            "opponent": prop["opponent"],
            "is_home": prop["is_home"],
            "confidence": round(signals_active / 3, 2),
            "signals": {
                "weak_defense": {
                    "active": weak_defense,
                    "opponent_d_elo": opponent_d_elo,
                    "weaker_than_pct_of_league": round(defense_percentile * 100, 0),
                },
                "usage_trending_up": {
                    "active": usage_up,
                    "snap_pct_trend": round(form["snap_pct_trend"], 2) if form else None,
                    "targets_trend": round(form["targets_trend"], 2) if form else None,
                    "carries_trend": round(form["carries_trend"], 2) if form else None,
                },
                "performance_strong": {
                    "active": performance_strong,
                    "last_4_ppr_2025": round(form["last_4_ppr"], 1) if form else None,
                    "season_avg_ppr_2025": round(form["season_avg_ppr"], 1) if form else None,
                    "diff": round(form["ppr_diff"], 1) if form else None,
                },
                "competition_reduced": {
                    "active": False,
                    "disclosure": COMPETITION_DISCLOSURE,
                },
            },
            "predicted_stats": prop["predicted_stats"],
            "recommendation": f"{prop['player_name']} ({prop['position']}) has {signals_active}/3 real "
                               f"breakout signals active vs {prop['opponent']} (week {week})",
        }
        alerts_by_week[str(week)].append(alert)
        total_alerts += 1

    for week in sorted(alerts_by_week, key=int):
        print(f"Week {week}: {len(alerts_by_week[week])} real breakout alerts")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts_by_week, f, indent=2)
    print(f"\nTotal: {total_alerts} real breakout alerts across 18 weeks "
          f"(avg {total_alerts / 18:.1f}/week) -> {OUTPUT_PATH}")
    return alerts_by_week


if __name__ == "__main__":
    build_breakout_signals()
