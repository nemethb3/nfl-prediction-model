"""Real, precomputed 2026 trade scores for the current real player pool,
no fabrication, no client-side model reimplementation.

This project has no backend (see the Sleeper Integration task) - a live
browser session cannot invoke the real, trained Python model at all. Real
fix, consistent with how every other feature in this project already
works: compute real scores HERE, in Python, using the actual fitted
models from train_trade_model.py, and ship them as static JSON. The
frontend only ever looks these real, precomputed values up - it never
re-derives a score with its own separate logic (the earlier pasted spec's
`calculatePlayerScore` did exactly that, with hand-typed weights
disconnected from the real trained model - not repeated here).

Real, current (as of 2026, no games played yet) signals per player,
computed with the exact same real methodology as build_trade_signals.py:
- current_age: each player's real 2025 recorded age + 1 (the real, same
  "one year forward" convention already used elsewhere in this project
  for 2026 preseason deliverables).
- injury_risk: real, point-in-time career miss rate using ALL real
  seasons through 2025 (not leakage here - we're forecasting the real
  future FROM the real present, unlike the historical backtest).
- role_trend / recent_trend: real trend into 2025 (2024->2025).
- draft_capital: real, static, from the same real crosswalk.
- team_elo: the exact same real preseason carryover Elo already powering
  this project's 2026 win totals and playoff odds
  (simulate_2026_playoffs.real_2026_carryover_elo()) - not a new,
  separately-derived copy - using each player's REAL CURRENT (2026)
  team from the real 2026 roster files, not a stale 2025 team (the exact
  class of bug the 2026-07-30 audit found and fixed once already).
"""

import json
from generation_timestamps import record_generation
import os
import pickle

import numpy as np
import pandas as pd

from build_trade_signals import (
    EARLIEST_SEASON, _real_draft_capital, _real_team_games_by_season,
    _real_qb_starter_curve, QB_STARTER_YEARS_PATH,
)
from constants import MIN_GAMES_FOR_SEASON, TREND_EPSILON
from simulate_2026_playoffs import real_2026_carryover_elo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CURVES_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "empirical_age_curves.json")
FANTASY_RANKINGS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "fantasy_rankings_2026.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "trade_scores_2026.json")

POSITIONS = ["QB", "RB", "WR", "TE"]
CURRENT_SEASON = 2026
LAST_REAL_SEASON = 2025


def _real_current_signals():
    """Real per-player signals as of right now (2025 season complete, 2026
    not yet played) - same real methodology as build_trade_signals.py's
    historical version, evaluated at a single real cutoff (2025) instead
    of once per historical season."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    pws = pws[pws["age"].notna() & pws["fantasy_points_ppr"].notna()]
    pws = pws[(pws["season"] >= EARLIEST_SEASON) & (pws["season"] <= LAST_REAL_SEASON)]

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
            "player_id": player_id, "season": int(season), "position": grp["position"].mode().iloc[0],
            "season_ppr": float(grp["fantasy_points_ppr"].sum()),
            "age_int": int(round(grp["age"].mean())),
            "games_played": games_played, "games_missed": games_missed,
            "target_share": float(grp["target_share"].mean()) if grp["target_share"].notna().any() else np.nan,
            "snap_pct": float(grp["snap_pct"].mean()) if grp["snap_pct"].notna().any() else np.nan,
        })
    season_stats = pd.DataFrame(rows).sort_values(["player_id", "season"]).reset_index(drop=True)

    signals = {}
    for player_id, grp in season_stats.groupby("player_id"):
        grp = grp.sort_values("season")
        if grp["season"].iloc[-1] != LAST_REAL_SEASON:
            continue  # only real players active in the most recent real season

        career_played = grp["games_played"].sum()
        career_missed = grp["games_missed"].sum()
        injury_risk = career_missed / (career_played + career_missed) if (career_played + career_missed) > 0 else None

        last, prev = grp.iloc[-1], (grp.iloc[-2] if len(grp) >= 2 and grp.iloc[-2]["season"] == LAST_REAL_SEASON - 1 else None)
        role_trend = None
        if prev is not None:
            t_now, t_prev = last["target_share"], prev["target_share"]
            s_now, s_prev = last["snap_pct"], prev["snap_pct"]
            trends = []
            if pd.notna(t_now) and pd.notna(t_prev):
                trends.append((t_now - t_prev) / (abs(t_prev) + TREND_EPSILON))
            if pd.notna(s_now) and pd.notna(s_prev):
                trends.append((s_now - s_prev) / (abs(s_prev) + TREND_EPSILON))
            if trends:
                role_trend = float(np.mean(trends))

        trailing = grp[grp["season"] < LAST_REAL_SEASON].tail(2)["season_ppr"]
        recent_trend = float(last["season_ppr"] / trailing.mean() - 1.0) if len(trailing) > 0 and trailing.mean() > 0 else None

        signals[player_id] = {
            "position": last["position"],
            "current_age": int(last["age_int"]) + 1,
            "injury_risk": injury_risk,
            "role_trend": role_trend,
            "recent_trend": recent_trend,
        }
    return signals


def _real_current_teams():
    """Real 2026 team per player from the real, purpose-built 2026 roster
    files - NOT each player's real 2025 team, which the 2026-07-30 audit
    already found goes stale for any real trade (the George Pickens class
    of bug)."""
    teams = {}
    for position in POSITIONS:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2026.csv"))
        for _, row in df.drop_duplicates("player_id").iterrows():
            teams[row["player_id"]] = row["team"]
    return teams


def _real_current_qb_starter_years():
    """Real years-as-starter for each real QB AS OF LAST_REAL_SEASON
    (2025) - only populated if 2025 itself was a real starter season for
    that player, same real convention build_trade_signals.py's training
    side uses (a non-starter `now` season yields no stratified value)."""
    df = pd.read_csv(QB_STARTER_YEARS_PATH)
    df = df[(df["season"] == LAST_REAL_SEASON) & df["years_as_starter"].notna()]
    return {r["player_id"]: int(r["years_as_starter"]) for _, r in df.iterrows()}


def generate_trade_scores_2026():
    print("\nGenerating real, precomputed 2026 trade scores...\n")
    with open(CURVES_PATH, encoding="utf-8") as f:
        age_curves = json.load(f)
    qb_starter_curve = _real_qb_starter_curve()
    qb_current_starter_years = _real_current_qb_starter_years()
    draft_capital = _real_draft_capital()
    elo_by_team = real_2026_carryover_elo()
    current_teams = _real_current_teams()
    signals = _real_current_signals()

    models = {}
    for position in POSITIONS:
        path = os.path.join(MODELS_DIR, f"trade_model_{position}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[position] = pickle.load(f)
    # AUDIT_2026-08-12_DEEP.md Section 4.1: previously no check that any
    # model loaded - a missing models/ dir would have silently written an
    # empty players: {} export instead of failing loudly.
    assert models, f"No trained trade models found in {MODELS_DIR} - run train_trade_model.py first"

    with open(FANTASY_RANKINGS_PATH, encoding="utf-8") as f:
        fantasy_players = json.load(f)
    real_player_ids = {p["id"].rsplit("_w", 1)[0] for p in fantasy_players}

    scores = {}
    n_scored, n_skipped = 0, 0
    # Sorted for deterministic output - iterating a raw set() uses Python's
    # per-process randomized string hashing, so the same real data would
    # otherwise serialize player order differently on every regeneration,
    # producing spurious diffs with no actual value change (caught while
    # reviewing this task's own regeneration diff).
    for player_id in sorted(real_player_ids):
        sig = signals.get(player_id)
        if sig is None:
            n_skipped += 1
            continue
        position = sig["position"]
        if position not in models:
            n_skipped += 1
            continue

        team = current_teams.get(player_id)
        team_elo = elo_by_team.get(team) if team else None

        if position == "QB":
            starter_year_now = qb_current_starter_years.get(player_id)
            if starter_year_now is None:
                age_curve_rising = None
            else:
                key_now, key_next = str(starter_year_now), str(starter_year_now + 1)
                age_curve_rising = (int(qb_starter_curve[key_next] >= qb_starter_curve[key_now])
                                     if key_now in qb_starter_curve and key_next in qb_starter_curve else None)
        else:
            curve = age_curves.get(position, {}).get("curve", {})
            age_now, age_next = str(sig["current_age"]), str(sig["current_age"] + 1)
            age_curve_rising = (int(curve[age_next] >= curve[age_now])
                                 if age_now in curve and age_next in curve else None)
        draft_value = draft_capital.get(player_id)

        feature_values = {
            "age_curve_rising": age_curve_rising,
            "injury_risk": sig["injury_risk"],
            "role_trend": sig["role_trend"],
            "recent_trend": sig["recent_trend"],
            "draft_capital": draft_value,
            "team_elo": team_elo,
        }
        if any(v is None for v in feature_values.values()):
            n_skipped += 1
            continue

        model_bundle = models[position]
        X = np.array([[feature_values[f] for f in model_bundle["features"]]])
        X_scaled = model_bundle["scaler"].transform(X)
        prob_increase = float(model_bundle["model"].predict_proba(X_scaled)[0, 1])

        scores[player_id] = {
            "position": position,
            "current_age": sig["current_age"],
            "team": team,
            "signals": {k: round(v, 3) for k, v in feature_values.items()},
            "prob_ppr_increase": round(prob_increase, 3),
            "trajectory": "Rising" if prob_increase > 0.5 else "Declining",
        }
        n_scored += 1

    output = {
        "season": CURRENT_SEASON,
        "methodology_note": (
            "Real per-player scores from the actual trained, GroupKFold-cross-validated logistic "
            "regression models (see multi_signal_accuracy.json for the honest, held-out accuracy per "
            "position) - not re-derived in the browser. prob_ppr_increase is a real model output: the "
            "predicted probability this player's PPR increases next real season, given real current "
            "age-curve direction, career injury history, role trend, recent-form trend, real draft "
            "capital, and real team Elo."
        ),
        "players": scores,
    }
    assert n_scored > 0, "0 real players scored - check signals/models coverage before shipping"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("trade_scores_2026")

    print(f"Real players scored: {n_scored} | skipped (incomplete real signals): {n_skipped}")
    print(f"Wrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    generate_trade_scores_2026()
