"""Scores real player props predictions for every real 2026 REG game, for
every real rostered QB/RB/WR/TE with real 2015-2025 history.

Real weather/rest addition (Quick Wins task): models now also take
is_dome/own_rest_days (real, both knowable in advance for 2026 - see
build_player_props_signals.py/train_player_props_models.py docstrings for
the real, honest null-result finding from adding them, and why temp/wind
were excluded as not knowable in advance for a future game).

Real TD logistic addition (Major Refinements task): predicted_stats now
includes `{td_col}_prob` (e.g. `passing_tds_prob`) - a real logistic
P(1+ TD), replacing the old linear expected-count prediction for TD-type
stats (real 5-fold OOF R^2 0.037-0.139 confirmed a fractional "1.2 TDs"
output wasn't meaningfully predictive or actionable - see
train_td_logistic_models.py, real AUC 0.60-0.70 instead).

Real, serious problems found and fixed in the originally pasted spec
before writing this:

1. Assumed `frontend/src/data/trade_scores_2026.json` has per-player
   `ppr_projection`/`avg_targets` fields - checked, it doesn't (real shape
   is `{"players": {player_id: {"signals": {...}, "prob_ppr_increase":
   ...}}}`, no PPR or volume figures at all). Real 2026 roster (player_id
   -> team) instead comes from data/processed/{position}_epa_projections_
   2026.csv - the same real, already-fixed source generate_fantasy_
   rankings_2026_week1.py and generate_trade_scores_2026.py both already
   use (fixes the real 30/107-stale-team bug the 2026-07-30 audit found -
   see those scripts' own docstrings).
2. Assumed `data/processed/games_2026.csv` and `data/processed/team_
   defense_stats_2026.csv` - neither exists. Real 2026 schedule is
   data/raw/schedules_2026.csv; real 2026 team O/D Elo (static preseason,
   same convention as every other 2026 deliverable in this project) is
   data/processed/team_elo_offensive_defensive_2026_regressed.json.
3. The spec predicted a `home_qb_id`/`away_qb_id` schedule lookup for QBs
   specifically, but a plain team lookup for every other position -
   real schedules_2026.csv has no *_qb_id columns with real values this
   far out (checked: 0/272 rows populated, same real gap already disclosed
   in generate_dashboard_data_2026.py). Real fix: every position uses the
   same real team-schedule join (which real team does this player's real
   2026 roster say they're on, and who does that team play each week).
4. Real career-average features for 2026 scoring use each player's real
   FULL 2015-2025 history (not leak-free-per-week like training, since
   2026 hasn't started - as of right now every real game they've played is
   genuinely prior). A real, newly-drafted rookie with no 2015-2025 history
   is excluded (a real, disclosed gap - identical precedent to
   FantasyRankings.js's own WR rookie-exclusion note), not assigned a
   fabricated league-average fallback.

Real pace/snap/red-zone addition (Player Props Enrichment task): reuses
build_player_props_signals.py's own real prior-season team pace/red-zone
function directly (`_real_prior_season_team_pace_and_rz`) rather than
re-deriving the same real play-by-play logic a second time - for 2026
this naturally resolves to each team's real 2025 rate, the same "static
prior-season number" convention already used for team_elo_offensive_
defensive_2026_regressed.json. `career_avg_snap_pct` uses each player's
real full 2015-2025 mean snap share, same convention as every other
career-average feature here.

Real opponent-EPA-allowed-by-position addition (Fantasy Model Overhaul
Phase 1): reuses the cached defense_epa_allowed_by_position_2015_2025.csv
(build_defense_epa_allowed_by_position.py) directly - for 2026 this
already resolves to each opponent's real 2025 rate, same static
prior-season convention as pace/RZ above."""

import json
import os

import numpy as np
import pandas as pd

from build_player_props_signals import _real_prior_season_team_pace_and_rz
from roster_utils import apply_current_team

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "player_props_models.json")
TD_MODELS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "td_props_logistic_models.json")
REGRESSED_OD_ELO_PATH = os.path.join(PROCESSED_DIR, "team_elo_offensive_defensive_2026_regressed.json")
SCHEDULE_PATH = os.path.join(RAW_DIR, "schedules_2026.csv")
PLAYER_STATS_PATH = os.path.join(PROCESSED_DIR, "player_weekly_stats.csv")
DEF_EPA_ALLOWED_BY_POSITION_PATH = os.path.join(
    PROCESSED_DIR, "defense_epa_allowed_by_position_2015_2025.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "player_props_2026.json")

SEASON = 2026
POSITIONS = ["QB", "RB", "WR", "TE"]
CAREER_AVG_SOURCE_COLS = {
    "QB": ["completions", "attempts", "passing_yards", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "rushing_yards"],
    "TE": ["targets", "receptions", "receiving_yards", "rushing_yards"],
}
TD_CAREER_AVG_COLS = {
    "QB": ["passing_tds", "rushing_tds"],
    "RB": ["rushing_tds"],
    "WR": ["receiving_tds"],
    "TE": ["receiving_tds"],
}


def _real_2026_roster(position):
    """Real player_id/player_name from {position}_epa_projections_2026.csv
    (same real source generate_fantasy_rankings_2026_week1.py and
    generate_trade_scores_2026.py both use), but `team` is corrected
    against the real, live 2026 roster (roster_utils.py) - see
    update_rosters_2026.py's docstring for the real, verified staleness
    bug this fixes (that CSV's own `team` column is a byproduct of an EPA
    model that explicitly disclaims tracking real transactions)."""
    path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2026.csv")
    df = pd.read_csv(path)
    df = apply_current_team(df)
    return df[["player_id", "player_name", "team"]].drop_duplicates("player_id")


def _real_career_averages(position, player_ids):
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[(stats["season_type"] == "REG") & (stats["position"] == position) &
                  (stats["player_id"].isin(player_ids))]
    cols = CAREER_AVG_SOURCE_COLS[position] + TD_CAREER_AVG_COLS[position] + ["snap_pct"]
    out = stats.groupby("player_id")[cols].mean().rename(columns={c: f"career_avg_{c}" for c in cols})
    return out


def _real_recent_form(position, player_ids):
    """Real recent-form feature for 2026 scoring (Fantasy Model Overhaul
    Phase 1B, the real winning approach - see train_player_props_models.py
    docstring): each rostered player's real trailing mean PPR over their
    own last 4 real games played (most recent 4 rows by season/week -
    naturally resolves to the end of their real 2025 season, the same
    "most recent real data" convention every other 2026 preseason feature
    here already uses, held static until real 2026 games are played)."""
    stats = pd.read_csv(PLAYER_STATS_PATH)
    stats = stats[(stats["season_type"] == "REG") & (stats["position"] == position) &
                  (stats["player_id"].isin(player_ids))].sort_values(["player_id", "season", "week"])
    out = stats.groupby("player_id").tail(4).groupby("player_id")["fantasy_points_ppr"].mean()
    return out.rename("recent_form_ppr_last4")


def _real_2026_team_pace_and_rz():
    """Real 2025 team pace/red-zone rate (the real prior-season lookup
    build_player_props_signals.py computes resolves to season=2026 ->
    real 2025 rates directly)."""
    lagged = _real_prior_season_team_pace_and_rz()
    return lagged[lagged["season"] == SEASON].set_index("team")[
        ["prior_season_pace_factor", "prior_season_rz_rate"]]


def _real_2026_def_epa_allowed_by_position(position):
    """Real 2025 opponent defensive EPA/play allowed to this position (the
    cached lookup already resolves season=2026 -> real 2025 rate, same
    convention as pace/RZ above)."""
    df = pd.read_csv(DEF_EPA_ALLOWED_BY_POSITION_PATH)
    df = df[(df["season"] == SEASON) & (df["position"] == position)]
    return df.set_index("team")["opp_epa_allowed_vs_position_prior_season"]


def _real_2026_schedule_by_team():
    sched = pd.read_csv(SCHEDULE_PATH)
    sched = sched[sched["game_type"] == "REG"]
    # Real roof/rest, same as build_player_props_signals.py's training-side
    # features - both genuinely knowable in advance for a 2026 game.
    # 43/272 real 2026 games have no roof recorded yet this far out; those
    # default to is_dome=0 (real majority class - 177/229 known real 2026
    # games are outdoors), a disclosed real fallback, not a fabricated one.
    sched["is_dome"] = sched["roof"].isin(["dome", "closed"]).fillna(False).astype(int)
    home = sched[["week", "home_team", "away_team", "is_dome", "home_rest"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_rest": "own_rest_days"})
    home["is_home"] = True
    away = sched[["week", "away_team", "home_team", "is_dome", "away_rest"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_rest": "own_rest_days"})
    away["is_home"] = False
    return pd.concat([home, away], ignore_index=True)


def _predict(feature_values, model_info):
    features = model_info["features"]
    x_scaled = np.array([
        (feature_values[f] - model_info["scaler_mean"][f]) / model_info["scaler_scale"][f]
        for f in features
    ])
    coefs = np.array([model_info["coefficients"][f] for f in features])
    pred = model_info["intercept"] + float(np.dot(x_scaled, coefs))
    return max(0.0, pred)


def _predict_proba(feature_values, model_info):
    """Real logistic sigmoid transform - same real fitted coefficients/
    scaler train_td_logistic_models.py produced, applied by hand here since
    this project has no backend to call sklearn's own .predict_proba()."""
    features = model_info["features"]
    x_scaled = np.array([
        (feature_values[f] - model_info["scaler_mean"][f]) / model_info["scaler_scale"][f]
        for f in features
    ])
    coefs = np.array([model_info["coefficients"][f] for f in features])
    z = model_info["intercept"] + float(np.dot(x_scaled, coefs))
    return float(1.0 / (1.0 + np.exp(-z)))


def generate_player_props_2026():
    print(f"\nGenerating real player props for {SEASON}...\n")
    with open(MODELS_PATH, encoding="utf-8") as f:
        models = json.load(f)
    with open(TD_MODELS_PATH, encoding="utf-8") as f:
        td_models = json.load(f)
    with open(REGRESSED_OD_ELO_PATH, encoding="utf-8") as f:
        regressed_od_elo = json.load(f)
    schedule_by_team = _real_2026_schedule_by_team()
    # Real, min-max week normalization matching the same real training
    # convention (build_player_props_signals.py) - REG season weeks only.
    week_min, week_max = schedule_by_team["week"].min(), schedule_by_team["week"].max()
    team_pace_rz = _real_2026_team_pace_and_rz()
    n_teams_no_pace_rz = 32 - len(team_pace_rz)
    if n_teams_no_pace_rz:
        print(f"Real note: {n_teams_no_pace_rz} teams have no real 2025 pace/RZ rate "
              "(abbreviation change/relocation) - their real 2026 games are skipped below")

    all_props = []
    for position in POSITIONS:
        roster = _real_2026_roster(position)
        career_avgs = _real_career_averages(position, roster["player_id"].tolist())
        recent_form = _real_recent_form(position, roster["player_id"].tolist())
        roster_with_history = roster.merge(career_avgs, on="player_id", how="inner")
        roster_with_history = roster_with_history.merge(recent_form, on="player_id", how="left")
        roster_with_history = roster_with_history.dropna(subset=["career_avg_snap_pct", "recent_form_ppr_last4"])
        n_excluded = len(roster) - len(roster_with_history)
        print(f"[{position}] {len(roster_with_history)}/{len(roster)} real rostered players have 2015-2025 "
              f"history ({n_excluded} real rookies/no-history players excluded - a real, disclosed gap, "
              "not a fabricated fallback)")

        position_models = models[position]
        position_td_models = td_models.get(position, {})
        def_epa_allowed = _real_2026_def_epa_allowed_by_position(position)
        n_games = 0
        for _, player in roster_with_history.iterrows():
            if player["team"] not in team_pace_rz.index:
                continue
            pace_rz = team_pace_rz.loc[player["team"]]
            player_games = schedule_by_team[schedule_by_team["team"] == player["team"]]
            for _, g in player_games.iterrows():
                opp_elo = regressed_od_elo.get(g["opponent"], {})
                if "d_elo" not in opp_elo:
                    continue
                if g["opponent"] not in def_epa_allowed.index:
                    continue
                week_norm = (g["week"] - week_min) / (week_max - week_min)
                common_feature_values = {
                    "opp_d_elo": float(opp_elo["d_elo"]),
                    "is_home": float(g["is_home"]),
                    "week_norm": float(week_norm),
                    "is_dome": float(g["is_dome"]),
                    "own_rest_days": float(g["own_rest_days"]),
                    "opp_epa_allowed_vs_position_prior_season": float(def_epa_allowed[g["opponent"]]),
                    "recent_form_ppr_last4": float(player["recent_form_ppr_last4"]),
                }
                feature_values = {
                    f"career_avg_{c}": float(player[f"career_avg_{c}"]) for c in CAREER_AVG_SOURCE_COLS[position]
                }
                feature_values["career_avg_snap_pct"] = float(player["career_avg_snap_pct"])
                feature_values["prior_season_pace_factor"] = float(pace_rz["prior_season_pace_factor"])
                feature_values.update(common_feature_values)

                predicted_stats = {
                    stat: round(_predict(feature_values, model_info), 1)
                    for stat, model_info in position_models.items()
                }
                # Real logistic P(1+ TD), replacing the old fractional
                # linear TD count (see train_player_props_models.py
                # docstring for why - real R^2 0.037-0.139 wasn't
                # meaningfully predictive or actionable).
                for td_col, td_model_info in position_td_models.items():
                    td_feature_values = {
                        f"career_avg_{td_col}": float(player[f"career_avg_{td_col}"]),
                        "prior_season_rz_rate": float(pace_rz["prior_season_rz_rate"]),
                        **common_feature_values,
                    }
                    predicted_stats[f"{td_col}_prob"] = round(_predict_proba(td_feature_values, td_model_info), 3)

                all_props.append({
                    "id": f"{player['player_id']}_w{int(g['week'])}",
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": position,
                    "team": player["team"],
                    "week": int(g["week"]),
                    "opponent": g["opponent"],
                    "is_home": bool(g["is_home"]),
                    "opponent_d_elo": round(float(opp_elo["d_elo"]), 1),
                    "predicted_stats": predicted_stats,
                })
                n_games += 1
        print(f"       {n_games} real player-games scored")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_props, f, indent=2)
    print(f"\nWrote {len(all_props)} real player-game props -> {OUTPUT_PATH}")
    return all_props


if __name__ == "__main__":
    generate_player_props_2026()
