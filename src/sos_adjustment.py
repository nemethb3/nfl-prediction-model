"""Strength of Schedule adjustment (Phase 2 Task 5.1).

Corrects two problems in the original task spec (see the Task 5.1 completion
report for the full reasoning):

1. The spec proposed using each opponent's PRIOR-season defense EPA as the
   SOS signal. team_strength.py already proved that's a poor proxy: naive
   "assume same as last year" defense-EPA prediction scored an average R2 of
   -0.77 across 6 honest holdout years - worse than just predicting the
   league average (-0.09). This module instead trains two real, honestly
   backtested Ridge models (reusing team_strength.py's exact pipeline,
   generalized to take a target_col) - one for pass_epa_allowed, one for
   rush_epa_allowed - and uses THEIR projections as the SOS input, not a raw
   lag.

2. The spec asserted SOS weights (QB: -0.08x, RB: -0.06x, WR: -0.03x). This
   module instead estimates a real slope per position via the same
   residual-regression technique used for the OL adjustment (Task 4.2):
   regress each position's EPA-model residual against their team's REAL,
   REALIZED average opponent defense quality that season (not a projection -
   this measures the actual historical relationship), then applies that
   slope to the model-PROJECTED 2025 opponent quality (built here) for the
   real forward adjustment, and validates against real 2025 outcomes.
"""

import os

import numpy as np
import pandas as pd

from ol_quality import POSITION_EPA_CONFIG, compute_epa_model_residuals, compute_real_epa_per_play, load_epa_model, load_real_2025_pbp
from team_strength import (
    HOLDOUT_SEASON, TRAIN_START, build_team_defense_features, compute_team_defense_epa, predict_next_season,
    train_defense_component_model,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Which opponent-defense signal matters for which position. QB and WR both
# face the same team pass defense (no separate "pass rush" vs "secondary"
# split is available without a much larger PBP-by-receiver-position build -
# see the pre-Task-5.1 discussion), so both use pass_epa_allowed; RB uses
# rush_epa_allowed.
POSITION_SOS_METRIC = {"QB": "pass_epa_allowed", "WR": "pass_epa_allowed", "RB": "rush_epa_allowed",
                        "TE": "pass_epa_allowed"}


def build_team_opponents(schedules, season):
    """Long table of (team, opponent) for every real regular-season game a
    team played in `season`."""
    sched = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    home = sched[["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"})
    away = sched[["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"})
    return pd.concat([home, away], ignore_index=True)


def compute_team_sos(opponents, defense_metrics, metric_cols):
    """Average opponent defense quality across a team's real schedule.
    `defense_metrics` supplies one row per opponent with metric_cols - pass
    either REAL realized values (for estimating the true historical
    relationship) or PROJECTED values (for the forward 2025 application)."""
    merged = opponents.merge(
        defense_metrics[["team"] + metric_cols].rename(columns={"team": "opponent"}),
        on="opponent", how="left",
    )
    return merged.groupby("team")[metric_cols].mean().reset_index()


def build_component_models():
    """Trains + honestly backtests the pass_epa_allowed and rush_epa_allowed
    Ridge models (reusing team_strength.py's validated pipeline), and
    projects both forward to 2025 (from 2024 data - the season our QB/WR/RB
    EPA projections are actually for)."""
    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    team_epa = compute_team_defense_epa()
    df = build_team_defense_features(team_epa, pass_rush_war_df)

    models = {}
    projections_2025 = {}
    for target_col, out_col in [("pass_epa_allowed", "predicted_pass_epa_allowed"),
                                 ("rush_epa_allowed", "predicted_rush_epa_allowed")]:
        model, _ = train_defense_component_model(df, target_col)
        models[target_col] = model
        proj, ref_season = predict_next_season(df, model, ref_season=HOLDOUT_SEASON,
                                                 target_col=target_col, out_col=out_col)
        projections_2025[target_col] = proj[["team", out_col]]

    proj_2025 = projections_2025["pass_epa_allowed"].merge(projections_2025["rush_epa_allowed"], on="team")
    return df, models, proj_2025


def estimate_sos_weight(position, residuals_df, team_epa_raw, schedules, train_seasons):
    """Regresses residual_S ~ team's REAL, REALIZED average opponent defense
    quality for season S (the true historical relationship - not a
    projection, since here we're measuring whether facing tougher schedules
    actually hurt production, using what really happened)."""
    metric_col = POSITION_SOS_METRIC[position]
    rows = []
    for season in sorted(set(train_seasons) & set(residuals_df["season"].unique())):
        opponents = build_team_opponents(schedules, season)
        sos = compute_team_sos(opponents, team_epa_raw[team_epa_raw["season"] == season], [metric_col])
        sos = sos.rename(columns={metric_col: "opponent_quality"})
        season_resid = residuals_df[residuals_df["season"] == season].merge(sos, on="team", how="inner")
        rows.append(season_resid)

    merged = pd.concat(rows, ignore_index=True).dropna(subset=["opponent_quality", "residual"])
    x = merged["opponent_quality"].to_numpy()
    y = merged["residual"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    print(f"[{position}] SOS weight regression (n={len(merged)}, metric={metric_col}): "
          f"slope={slope:+.4f} EPA/play per unit opponent {metric_col} | corr={corr:+.3f} | R2={corr**2:.4f}")
    return slope, corr, len(merged)


def apply_sos_adjustment(projections_df, sos_2025, slope, metric_col):
    projections_df = projections_df.drop(columns=["opponent_sos", "predicted_epa_per_play_sos_adjusted"], errors="ignore")
    out = projections_df.merge(sos_2025.rename(columns={metric_col: "opponent_sos"}), on="team", how="left")
    out["opponent_sos"] = out["opponent_sos"].fillna(sos_2025[metric_col].mean())
    out["predicted_epa_per_play_sos_adjusted"] = out["predicted_epa_per_play"] + slope * out["opponent_sos"]
    return out


def validate_sos_adjustment(position, adjusted_2025, real_2025):
    merged = adjusted_2025.merge(real_2025[["player_id", "real_2025_epa_per_play"]], on="player_id", how="inner")
    if not len(merged):
        print(f"[{position}] no real-2025 matches - skipping validation")
        return None

    def _score(pred_col):
        actual = merged["real_2025_epa_per_play"].to_numpy()
        pred = merged[pred_col].to_numpy()
        mae = np.mean(np.abs(pred - actual))
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
        return mae, r2

    mae_base, r2_base = _score("predicted_epa_per_play")
    mae_adj, r2_adj = _score("predicted_epa_per_play_sos_adjusted")
    print(f"[{position}] real 2025 (n={len(merged)}): baseline MAE={mae_base:.4f} R2={r2_base:.3f} | "
          f"SOS-adjusted MAE={mae_adj:.4f} R2={r2_adj:.3f}")
    return {"mae_base": mae_base, "mae_adj": mae_adj, "r2_base": r2_base, "r2_adj": r2_adj,
            "helps": mae_adj < mae_base}


def run_sos_adjustment():
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))

    print("=" * 60 + "\nBuilding pass/rush defense-split models\n" + "=" * 60)
    team_epa_raw, models, proj_2025 = build_component_models()

    pbp_2025 = load_real_2025_pbp()
    opponents_2025 = build_team_opponents(schedules, 2025)

    results = {}
    for position in ["QB", "WR", "RB", "TE"]:
        print(f"\n{'=' * 60}\n{position}\n{'=' * 60}")
        metric_col = POSITION_SOS_METRIC[position]
        pred_col = f"predicted_{metric_col}"

        model, prepped = load_epa_model(position, features_df, season_stats_df)
        residuals = compute_epa_model_residuals(model, prepped)

        train_seasons = range(TRAIN_START, HOLDOUT_SEASON)
        slope, corr, n = estimate_sos_weight(position, residuals, team_epa_raw, schedules, train_seasons)

        sos_2025 = compute_team_sos(opponents_2025, proj_2025.rename(columns={pred_col: metric_col}), [metric_col])

        proj_path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2025.csv")
        projections = pd.read_csv(proj_path)
        adjusted_2025 = apply_sos_adjustment(projections, sos_2025, slope, metric_col)

        real_2025 = compute_real_epa_per_play(position, pbp_2025)
        metrics = validate_sos_adjustment(position, adjusted_2025, real_2025)

        adjusted_2025.to_csv(proj_path, index=False, encoding="utf-8")
        print(f"Saved {proj_path} (added opponent_sos + predicted_epa_per_play_sos_adjusted columns)")

        results[position] = {"slope": slope, "corr": corr, "n": n, **(metrics or {})}

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for position, r in results.items():
        verdict = "HELPS" if r.get("helps") else "NO HELP"
        print(f"{position}: slope={r['slope']:+.4f} corr={r['corr']:+.3f} (n={r['n']}) | real-2025 verdict: {verdict}")
    return results



# ---------------------------------------------------------------------------
# Task 5.2: Availability Factor (real games-played data, not hardcoded
# "injury rates")
# ---------------------------------------------------------------------------

from xgboost import XGBRegressor

from utilities import compute_history_features

# Regular season length changed from 16 to 17 games starting 2021 - dividing
# every season by a flat 17 would understate every pre-2021 player's real
# availability (a "16/16" healthy season would read as 94%, not 100%).
SEASON_LENGTH = {**{y: 16 for y in range(2015, 2021)}, **{y: 17 for y in range(2021, 2027)}}

AVAILABILITY_FEATURE_COLS = ["age", "years_in_league",
                             "games_pct_last_year", "games_pct_last_3yr_avg", "games_pct_career_avg"]


def compute_games_pct(stats_df, games_col):
    """Per player-season, games actually played as a fraction of that
    season's real length (16 pre-2021, 17 from 2021 on) - the real,
    already-available signal this task needs, instead of the original
    spec's hardcoded per-position 'injury_rate' table."""
    df = stats_df.copy()
    df["season_length"] = df["season"].map(SEASON_LENGTH)
    games_col_data = df[games_col].fillna(0)
    df["games_pct"] = (games_col_data / df["season_length"]).clip(upper=1.0)
    return df


def add_games_lookback(df):
    feats = compute_history_features(df[["player_id", "season", "games_pct"]], "games_pct")
    return df.join(feats)


def estimate_availability_model(df, train_seasons, holdout_season):
    """Tests whether a lookback-feature XGBoost model can predict next
    season's games_pct better than two naive baselines (last_3yr_avg,
    position-average) - availability/durability is largely injury-luck
    driven, so this project's established discipline (never assume a
    fancier model wins) applies here more than almost anywhere else in the
    pipeline. Returns whichever approach wins on the 2024 holdout."""
    sub = df.dropna(subset=AVAILABILITY_FEATURE_COLS + ["games_pct"])
    train = sub[sub["season"].isin(train_seasons)]
    holdout = sub[sub["season"] == holdout_season]

    naive_pred = holdout["games_pct_last_3yr_avg"].to_numpy()
    pos_avg_pred = np.full(len(holdout), train["games_pct"].mean())
    actual = holdout["games_pct"].to_numpy()

    model = XGBRegressor(random_state=42, n_estimators=40, max_depth=2, learning_rate=0.05)
    model.fit(train[AVAILABILITY_FEATURE_COLS], train["games_pct"])
    xgb_pred = model.predict(holdout[AVAILABILITY_FEATURE_COLS])

    mae_naive = np.mean(np.abs(naive_pred - actual))
    mae_pos_avg = np.mean(np.abs(pos_avg_pred - actual))
    mae_xgb = np.mean(np.abs(xgb_pred - actual))
    print(f"  holdout MAE: last_3yr_avg={mae_naive:.4f} | position_avg={mae_pos_avg:.4f} | xgb={mae_xgb:.4f}")

    best = min([("last_3yr_avg", mae_naive), ("position_avg", mae_pos_avg), ("xgb", mae_xgb)], key=lambda t: t[1])
    print(f"  winner: {best[0]}")
    return {"method": best[0], "model": model if best[0] == "xgb" else None,
            "position_avg": train["games_pct"].mean(), "mae": best[1]}


def predict_availability(df, choice, ref_season):
    """Projects next season's expected games_pct using whichever method won
    the holdout comparison, from ref_season's real state."""
    current = df[df["season"] == ref_season].copy()
    if choice["method"] == "xgb":
        proj = pd.DataFrame({
            "player_id": current["player_id"],
            "age": current["age"] + 1,
            "years_in_league": current["years_in_league"] + 1,
            "games_pct_last_year": current["games_pct"],
            "games_pct_last_3yr_avg": current[["games_pct_last_3yr_avg", "games_pct"]].mean(axis=1),
            "games_pct_career_avg": current[["games_pct_career_avg", "games_pct"]].mean(axis=1),
        })
        predicted = choice["model"].predict(proj[AVAILABILITY_FEATURE_COLS])
    elif choice["method"] == "last_3yr_avg":
        proj = current[["player_id"]].copy()
        predicted = current[["games_pct_last_3yr_avg", "games_pct"]].mean(axis=1).to_numpy()
    else:  # position_avg
        proj = current[["player_id"]].copy()
        predicted = np.full(len(current), choice["position_avg"])

    out = proj[["player_id"]].copy()
    out["availability_factor"] = np.clip(predicted, 0.0, 1.0)
    return out


def build_position_availability(position, source_df, games_col, train_seasons, holdout_season, ref_season):
    pos_df = source_df[source_df["position"] == position].copy()
    pos_df = compute_games_pct(pos_df, games_col)
    pos_df = add_games_lookback(pos_df)

    print(f"[{position}] availability model selection:")
    choice = estimate_availability_model(pos_df, train_seasons, holdout_season)
    availability_2025 = predict_availability(pos_df, choice, ref_season)
    return availability_2025, choice


def compute_real_games_played_offense(pbp_2025):
    """Real 2025 games played per offensive player - distinct weeks with any
    passing/rushing/receiving involvement, from real PBP (2025 already
    happened). Used to validate the availability projections out of sample,
    same discipline as every other Task 4/5 validation."""
    cols = ["week", "passer_id", "rusher_id", "receiver_id"]
    long = pd.concat([
        pbp_2025[["week", "passer_id"]].dropna().rename(columns={"passer_id": "player_id"}),
        pbp_2025[["week", "rusher_id"]].dropna().rename(columns={"rusher_id": "player_id"}),
        pbp_2025[["week", "receiver_id"]].dropna().rename(columns={"receiver_id": "player_id"}),
    ], ignore_index=True)
    games = long.drop_duplicates(subset=["player_id", "week"]).groupby("player_id").size()
    out = games.reset_index(name="real_games_2025")
    out["real_availability_2025"] = (out["real_games_2025"] / SEASON_LENGTH[2025]).clip(upper=1.0)
    return out


def validate_availability(position, availability_2025, real_games, projections_df):
    merged = projections_df.merge(availability_2025, on="player_id", how="left").merge(
        real_games, on="player_id", how="inner")
    if not len(merged):
        print(f"[{position}] no real-2025 matches - skipping validation")
        return None
    actual = merged["real_availability_2025"].to_numpy()
    pred = merged["availability_factor"].to_numpy()
    naive_full = np.ones(len(merged))  # "assume everyone plays all season" - the implicit assumption
                                        # every projection in this pipeline has made until now
    mae_model = np.mean(np.abs(pred - actual))
    mae_naive_full = np.mean(np.abs(naive_full - actual))
    print(f"[{position}] real 2025 (n={len(merged)}): availability-model MAE={mae_model:.4f} | "
          f"'assume full season' MAE={mae_naive_full:.4f}")
    return {"mae_model": mae_model, "mae_naive_full": mae_naive_full, "n": len(merged)}


def run_availability_adjustment():
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    defense_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_defense.csv"))
    pbp_2025 = load_real_2025_pbp()
    real_games_offense = compute_real_games_played_offense(pbp_2025)

    offense_train_seasons = range(TRAIN_START, HOLDOUT_SEASON)
    results = {}
    for position, proj_file in [("QB", "qb_epa_projections_2025.csv"),
                                 ("WR", "wr_epa_projections_2025.csv"),
                                 ("RB", "rb_epa_projections_2025.csv"),
                                 ("TE", "te_epa_projections_2025.csv")]:
        print(f"\n{'=' * 60}\n{position} (offense, projecting 2025)\n{'=' * 60}")
        availability_2025, choice = build_position_availability(
            position, season_stats_df, "games_played", offense_train_seasons, HOLDOUT_SEASON, HOLDOUT_SEASON)

        proj_path = os.path.join(PROCESSED_DIR, proj_file)
        projections = pd.read_csv(proj_path).drop(
            columns=["availability_factor", "expected_games_2025"], errors="ignore")
        metrics = validate_availability(position, availability_2025, real_games_offense, projections)

        merged = projections.merge(availability_2025, on="player_id", how="left")
        merged["availability_factor"] = merged["availability_factor"].fillna(choice["position_avg"])
        merged["expected_games_2025"] = (merged["availability_factor"] * SEASON_LENGTH[2025]).round(1)
        merged.to_csv(proj_path, index=False, encoding="utf-8")
        print(f"Saved {proj_path} (added availability_factor + expected_games_2025 - informational only, "
              f"NOT multiplied into predicted_epa_per_play - see report)")
        results[position] = {"method": choice["method"], **(metrics or {})}

    # Defense: PFR games coverage is 2018-2025, so the training/holdout
    # windows differ from offense's 2016-2024. Projections here are for 2026
    # (genuine future season, not yet played) - no real-outcome check
    # possible, unlike offense; validated via 2024 holdout only.
    defense_train_seasons = range(2019, 2024)
    defense_holdout = 2024
    defense_ref_season = 2025
    for position, proj_file in [("CB", "cb_blended_projections_2026.csv"),
                                 ("S", "s_blended_projections_2026.csv"),
                                 ("LB", "lb_blended_projections_2026.csv")]:
        print(f"\n{'=' * 60}\n{position} (defense, projecting 2026 - not yet played, no real-outcome check)\n{'=' * 60}")
        availability_2026, choice = build_position_availability(
            position, defense_stats_df, "games", defense_train_seasons, defense_holdout, defense_ref_season)

        proj_path = os.path.join(PROCESSED_DIR, proj_file)
        projections = pd.read_csv(proj_path)
        # cb/s/lb_blended_projections files don't carry player_id (see
        # OffenseEpaModel vs BlendedDefenseModel - the latter predates the
        # player_id fix made in Task 4.2) - join on player_name+team instead.
        pos_df = defense_stats_df[defense_stats_df["position"] == position]
        id_lookup = pos_df[pos_df["season"] == defense_ref_season][["player_id", "player", "team"]].rename(
            columns={"player": "player_name"}).drop_duplicates(subset=["player_name", "team"])
        availability_2026 = availability_2026.merge(id_lookup, on="player_id", how="left")

        projections = projections.drop(columns=["availability_factor", "expected_games_2026"], errors="ignore")
        merged = projections.merge(availability_2026[["player_name", "team", "availability_factor"]],
                                    on=["player_name", "team"], how="left")
        merged["availability_factor"] = merged["availability_factor"].fillna(choice["position_avg"])
        merged["expected_games_2026"] = (merged["availability_factor"] * SEASON_LENGTH[2026]).round(1)
        merged.to_csv(proj_path, index=False, encoding="utf-8")
        print(f"Saved {proj_path} (added availability_factor + expected_games_2026 - informational only)")
        results[position] = {"method": choice["method"]}

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for position, r in results.items():
        print(f"{position}: winning method={r['method']}"
              + (f" | real-2025 MAE={r['mae_model']:.4f} (vs. assume-full-season {r['mae_naive_full']:.4f})"
                 if "mae_model" in r else " | (2026 - not yet played, holdout-only validation)"))
    return results



# ---------------------------------------------------------------------------
# Task 5.3: CB/S Pass-Rush Synergy - residual regression + spurious-
# correlation check
# ---------------------------------------------------------------------------

import pickle

from player_models import BlendedDefenseModel, BLEND_TRAIN_SEASONS, build_blended_score_table

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Winning blend ratios from Phase 2 Refinement Task 2's holdout search
# (deterministic given fixed random_state=42 - reused, not re-derived blind).
BLEND_RATIO_BY_POSITION = {"CB": (0.8, 0.2), "S": (0.5, 0.5)}


def fit_dl_sacks_to_war_conversion(pass_rush_war_df):
    """DL's forward projection (dl_predictions_2026.csv) is sacks-based, not
    WAR-based (Task 3 discarded DL's WAR model - it scored worse on
    holdout). To fold DL into a team-level pass-rush-WAR figure for 2026
    without reintroducing the EPA_PER_WIN-on-non-EPA-units bug fixed twice
    already this project, fit a real, data-derived sacks->war conversion
    from DL's own historical rows (which DO have both a real war figure and
    real PFR sacks, from Task 1's PBP attribution) rather than assuming any
    constant."""
    dl = pass_rush_war_df[(pass_rush_war_df["position"] == "DL")].dropna(subset=["war", "pfr_sacks"])
    slope, intercept = np.polyfit(dl["pfr_sacks"], dl["war"], 1)
    corr = np.corrcoef(dl["pfr_sacks"], dl["war"])[0, 1]
    print(f"[DL sacks->war] n={len(dl)}, war = {slope:.4f} * sacks + {intercept:.4f} (corr={corr:.3f})")
    return slope, intercept


def compute_team_pass_rush_war_2026():
    """Projected 2026 team pass-rush WAR: real EDGE WAR projection
    (edge_war_predictions_2026.csv) + DL's projected sacks converted to a
    WAR-equivalent via the real conversion above, summed by team."""
    pass_rush_war_df = pd.read_csv(os.path.join(PROCESSED_DIR, "pass_rush_war_2015_2025.csv"))
    slope, intercept = fit_dl_sacks_to_war_conversion(pass_rush_war_df)

    edge = pd.read_csv(os.path.join(PROCESSED_DIR, "edge_war_predictions_2026.csv"))
    dl = pd.read_csv(os.path.join(PROCESSED_DIR, "dl_predictions_2026.csv"))
    dl["dl_war_estimate"] = slope * dl["predicted_sacks"] + intercept

    edge_team = edge.groupby("team")["predicted_war"].sum().rename("edge_war_total")
    dl_team = dl.groupby("team")["dl_war_estimate"].sum().rename("dl_war_total")
    team_pr = pd.concat([edge_team, dl_team], axis=1).fillna(0)
    team_pr["team_pass_rush_war_2026"] = team_pr["edge_war_total"] + team_pr["dl_war_total"]
    return team_pr.reset_index()[["team", "team_pass_rush_war_2026"]]


def compute_blended_residuals(position, tackle_df, leverage_df, crosswalk, schedules):
    tw, lw = BLEND_RATIO_BY_POSITION[position]
    scored = build_blended_score_table(tackle_df, leverage_df, position, tw, lw)
    model = BlendedDefenseModel(position)
    with open(os.path.join(MODELS_DIR, f"{position.lower()}_blended.pkl"), "rb") as f:
        model.xgb_model = pickle.load(f)
    prepped = model.prepare_data(scored, crosswalk, schedules)
    prepped["predicted_blended_score"] = model.xgb_model.predict(prepped[model.FEATURE_COLS])
    prepped["residual"] = prepped["blended_score"] - prepped["predicted_blended_score"]
    return prepped


def _partial_regression(x1, y, x2):
    """Partial correlation/slope of x1 on y, controlling for x2: residualize
    both x1 and y against x2, then correlate/regress the residuals. This is
    the actual spurious-correlation check - if team pass-rush quality and
    CB/S performance are both just proxies for 'good team' (a common
    defensive-investment/scheme confound), controlling for the team's own
    general defensive quality (def_epa_allowed_last_year) should mostly
    kill the raw correlation. If it survives, that's real evidence of an
    incremental pass-rush-specific effect, not just team quality twice."""
    b_x1 = np.polyfit(x2, x1, 1)
    x1_resid = x1 - np.polyval(b_x1, x2)
    b_y = np.polyfit(x2, y, 1)
    y_resid = y - np.polyval(b_y, x2)
    partial_corr = np.corrcoef(x1_resid, y_resid)[0, 1]
    partial_slope = np.polyfit(x1_resid, y_resid, 1)[0]
    return partial_corr, partial_slope


def estimate_synergy_weight(position, residuals_df, team_defense_df, train_seasons=BLEND_TRAIN_SEASONS):
    """Regresses residual_S ~ team's REAL, REALIZED team_pass_rush_war for
    that same season (the true historical relationship, not a projection -
    same approach as Task 5.1's SOS estimation), then checks whether the
    relationship survives controlling for the team's general defensive
    quality (def_epa_allowed_last_year)."""
    df = residuals_df[residuals_df["season"].isin(train_seasons)].copy()
    merged = df.merge(
        team_defense_df[["team", "season", "team_pass_rush_war", "def_epa_allowed_last_year"]],
        on=["team", "season"], how="inner",
    ).dropna(subset=["team_pass_rush_war", "def_epa_allowed_last_year", "residual"])

    x = merged["team_pass_rush_war"].to_numpy()
    y = merged["residual"].to_numpy()
    x2 = merged["def_epa_allowed_last_year"].to_numpy()

    raw_slope, _ = np.polyfit(x, y, 1)
    raw_corr = np.corrcoef(x, y)[0, 1]
    partial_corr, partial_slope = _partial_regression(x, y, x2)

    print(f"[{position}] synergy regression (n={len(merged)}): "
          f"raw corr={raw_corr:+.3f} (R2={raw_corr**2:.4f}) | "
          f"partial corr (controlling for team def quality)={partial_corr:+.3f} (R2={partial_corr**2:.4f})")
    # The "% retained after controlling for team quality" framing only means
    # something if there was a real correlation to begin with - checked
    # |raw_corr| against a noise floor first, since a near-zero correlation
    # "retaining most of its magnitude" is just near-zero retaining
    # near-zero, not evidence of anything.
    NOISE_FLOOR = 0.05
    if abs(raw_corr) < NOISE_FLOOR:
        verdict = f"no real signal to begin with (|raw corr|={abs(raw_corr):.3f} < {NOISE_FLOOR} noise floor)"
    else:
        survival_pct = abs(partial_corr) / abs(raw_corr) * 100
        verdict = (f"retains {survival_pct:.0f}% of raw magnitude after controlling for team quality - "
                    + ("looks like a real incremental effect" if survival_pct > 50 else "looks mostly like a team-quality confound"))
    print(f"  -> {verdict}")
    return {"raw_corr": raw_corr, "raw_slope": raw_slope, "partial_corr": partial_corr,
            "partial_slope": partial_slope, "n": len(merged)}


def apply_synergy_adjustment(projections_df, team_pr_2026, slope):
    # Drop any columns this function is about to (re-)add, so re-running the
    # pipeline on an already-adjusted file (e.g. after an upstream fix) is
    # safe instead of producing merge-suffix collisions.
    projections_df = projections_df.drop(
        columns=["team_pass_rush_war_2026", "predicted_blended_score_synergy_adjusted"], errors="ignore")
    out = projections_df.merge(team_pr_2026, on="team", how="left")
    out["team_pass_rush_war_2026"] = out["team_pass_rush_war_2026"].fillna(team_pr_2026["team_pass_rush_war_2026"].mean())
    out["predicted_blended_score_synergy_adjusted"] = (
        out["predicted_blended_score"] + slope * out["team_pass_rush_war_2026"]
    )
    return out


def validate_synergy_holdout(position, residuals_df, team_defense_df, slope, holdout_season=2024):
    """2026 hasn't been played (this project's current date is 2026-07-24 -
    the season starts in September), so unlike Task 5.1/5.2's offense
    checks, there's no real-outcome validation available here. Falls back
    to the 2024 holdout - never used to fit the slope - same as every model
    in this project uses before a real-outcome check is possible."""
    hold = residuals_df[residuals_df["season"] == holdout_season].merge(
        team_defense_df[["team", "season", "team_pass_rush_war"]], on=["team", "season"], how="inner")
    if not len(hold):
        print(f"[{position}] no holdout rows - skipping")
        return None
    actual = hold["blended_score"].to_numpy()
    baseline_pred = hold["predicted_blended_score"].to_numpy()
    adjusted_pred = baseline_pred + slope * hold["team_pass_rush_war"].to_numpy()

    mae_base = np.mean(np.abs(baseline_pred - actual))
    mae_adj = np.mean(np.abs(adjusted_pred - actual))
    print(f"[{position}] 2024 holdout (n={len(hold)}): baseline MAE={mae_base:.3f} | "
          f"synergy-adjusted MAE={mae_adj:.3f} | {'HELPS' if mae_adj < mae_base else 'NO HELP'}")
    return {"mae_base": mae_base, "mae_adj": mae_adj, "helps": mae_adj < mae_base}


# ---------------------------------------------------------------------------
# Phase 3 Rebuild Task 1: TE pass-rush synergy - extends the CB/S mechanism
# (Task 5.3) to an offensive position. TE isn't scored with the blended
# tackle+leverage model CB/S use, so this reuses the EPA/play residual
# machinery from the OL/SOS adjustments above instead of CB/S's
# apply_synergy_adjustment. It's also weaker footed causally than CB/S: a
# team's OWN pass rush is a defensive asset, not something that obviously
# helps its OWN TE get open, so unlike CB/S this isn't assumed to work - it's
# tested the same leak-free way (train-seasons-only slope, spurious-
# correlation check, then a genuine real-2025 outcome check since TE
# projects to an already-played season, unlike CB/S's not-yet-played 2026)
# and dropped if it doesn't hold up.
# ---------------------------------------------------------------------------

def estimate_te_synergy_weight(residuals_df, team_defense_df, train_seasons):
    df = residuals_df[residuals_df["season"].isin(train_seasons)].copy()
    merged = df.merge(
        team_defense_df[["team", "season", "team_pass_rush_war", "def_epa_allowed_last_year"]],
        on=["team", "season"], how="inner",
    ).dropna(subset=["team_pass_rush_war", "def_epa_allowed_last_year", "residual"])

    x = merged["team_pass_rush_war"].to_numpy()
    y = merged["residual"].to_numpy()
    x2 = merged["def_epa_allowed_last_year"].to_numpy()

    raw_slope, _ = np.polyfit(x, y, 1)
    raw_corr = np.corrcoef(x, y)[0, 1]
    partial_corr, partial_slope = _partial_regression(x, y, x2)

    print(f"[TE] synergy regression (n={len(merged)}): "
          f"raw corr={raw_corr:+.3f} (R2={raw_corr**2:.4f}) | "
          f"partial corr (controlling for team def quality)={partial_corr:+.3f} (R2={partial_corr**2:.4f})")
    NOISE_FLOOR = 0.05
    if abs(raw_corr) < NOISE_FLOOR:
        verdict = f"no real signal to begin with (|raw corr|={abs(raw_corr):.3f} < {NOISE_FLOOR} noise floor)"
    else:
        survival_pct = abs(partial_corr) / abs(raw_corr) * 100
        verdict = (f"retains {survival_pct:.0f}% of raw magnitude after controlling for team quality - "
                    + ("looks like a real incremental effect" if survival_pct > 50 else "looks mostly like a team-quality confound"))
    print(f"  -> {verdict}")
    return {"raw_corr": raw_corr, "raw_slope": raw_slope, "partial_corr": partial_corr,
            "partial_slope": partial_slope, "n": len(merged)}


def apply_te_synergy_adjustment(projections_df, team_defense_df, slope, ref_season):
    projections_df = projections_df.drop(
        columns=["team_pass_rush_war_ref", "predicted_epa_per_play_synergy_adjusted"], errors="ignore")
    pr_ref = team_defense_df[team_defense_df["season"] == ref_season][["team", "team_pass_rush_war"]].rename(
        columns={"team_pass_rush_war": "team_pass_rush_war_ref"})
    out = projections_df.merge(pr_ref, on="team", how="left")
    out["team_pass_rush_war_ref"] = out["team_pass_rush_war_ref"].fillna(pr_ref["team_pass_rush_war_ref"].mean())
    out["predicted_epa_per_play_synergy_adjusted"] = (
        out["predicted_epa_per_play"] + slope * out["team_pass_rush_war_ref"]
    )
    return out


def validate_te_synergy(adjusted_2025, real_2025):
    merged = adjusted_2025.merge(real_2025[["player_id", "real_2025_epa_per_play"]], on="player_id", how="inner")
    if not len(merged):
        print("[TE] no real-2025 matches - skipping validation")
        return None

    def _score(pred_col):
        actual = merged["real_2025_epa_per_play"].to_numpy()
        pred = merged[pred_col].to_numpy()
        mae = np.mean(np.abs(pred - actual))
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
        return mae, r2

    mae_base, r2_base = _score("predicted_epa_per_play")
    mae_adj, r2_adj = _score("predicted_epa_per_play_synergy_adjusted")
    print(f"[TE] real 2025 (n={len(merged)}): baseline MAE={mae_base:.4f} R2={r2_base:.3f} | "
          f"synergy-adjusted MAE={mae_adj:.4f} R2={r2_adj:.3f}")
    return {"mae_base": mae_base, "mae_adj": mae_adj, "r2_base": r2_base, "r2_adj": r2_adj,
            "helps": mae_adj < mae_base}


def run_te_synergy_adjustment(team_defense_df, pbp_2025):
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))

    print(f"\n{'=' * 60}\nTE (offense pass-rush synergy)\n{'=' * 60}")
    model, prepped = load_epa_model("TE", features_df, season_stats_df)
    residuals = compute_epa_model_residuals(model, prepped)

    train_seasons = range(TRAIN_START, HOLDOUT_SEASON)
    weight_info = estimate_te_synergy_weight(residuals, team_defense_df, train_seasons)
    slope = weight_info["partial_slope"]

    proj_path = os.path.join(PROCESSED_DIR, "te_epa_projections_2025.csv")
    projections = pd.read_csv(proj_path)
    ref_season = int(prepped["season"].max())
    adjusted_2025 = apply_te_synergy_adjustment(projections, team_defense_df, slope, ref_season=ref_season)

    real_2025 = compute_real_epa_per_play("TE", pbp_2025)
    metrics = validate_te_synergy(adjusted_2025, real_2025)

    adjusted_2025.to_csv(proj_path, index=False, encoding="utf-8")
    print(f"Saved {proj_path} (added team_pass_rush_war_ref + predicted_epa_per_play_synergy_adjusted columns)")

    return {"TE": {**weight_info, **(metrics or {})}}


def run_synergy_adjustment():
    crosswalk = pd.read_csv(os.path.join(PROCESSED_DIR, "player_metadata.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    tackle_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tackle_efficiency_2018_2025.csv"))
    leverage_df = pd.read_csv(os.path.join(PROCESSED_DIR, "leverage_war_2016_2025.csv"))
    team_defense_df = pd.read_csv(os.path.join(PROCESSED_DIR, "team_defense_epa_2015_2025.csv"))

    team_pr_2026 = compute_team_pass_rush_war_2026()

    results = {}
    for position, proj_file in [("CB", "cb_blended_projections_2026.csv"), ("S", "s_blended_projections_2026.csv")]:
        print(f"\n{'=' * 60}\n{position}\n{'=' * 60}")
        residuals = compute_blended_residuals(position, tackle_df, leverage_df, crosswalk, schedules)
        weight_info = estimate_synergy_weight(position, residuals, team_defense_df)

        # Use the PARTIAL slope (post spurious-correlation check) for the
        # actual applied adjustment - the raw slope would risk re-applying
        # the "good team" confound the partial check exists to catch.
        slope = weight_info["partial_slope"]
        holdout_metrics = validate_synergy_holdout(position, residuals, team_defense_df, slope)

        proj_path = os.path.join(PROCESSED_DIR, proj_file)
        projections = pd.read_csv(proj_path)
        adjusted = apply_synergy_adjustment(projections, team_pr_2026, slope)
        adjusted.to_csv(proj_path, index=False, encoding="utf-8")
        print(f"Saved {proj_path} (added team_pass_rush_war_2026 + predicted_blended_score_synergy_adjusted)")

        results[position] = {**weight_info, **(holdout_metrics or {})}

    pbp_2025 = load_real_2025_pbp()
    results.update(run_te_synergy_adjustment(team_defense_df, pbp_2025))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for position, r in results.items():
        check_label = "2024 holdout" if position != "TE" else "real 2025"
        print(f"{position}: raw corr={r['raw_corr']:+.3f} | partial corr={r['partial_corr']:+.3f} "
              f"(applied slope, n={r['n']}) | {check_label}: {'HELPS' if r.get('helps') else 'NO HELP'}")
    return results


if __name__ == "__main__":
    run_sos_adjustment()
    run_availability_adjustment()
    run_synergy_adjustment()
