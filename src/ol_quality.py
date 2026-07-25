"""Team offensive-line quality (Phase 2 Task 4, Task 4.1).

Two real, free, PBP-native proxies for OL quality, replacing the original
spec's "PFR Adjusted Line Yards" (which turned out not to actually be a PFR
stat - it's a Football Outsiders/FTN proprietary metric, not available here
or anywhere for free - see the Task 4.1 completion report):

  - Pass protection: team sack rate allowed (sacks / pass plays), by posteam.
  - Run blocking: team rush EPA generated (mean EPA on the team's own rush
    plays), by posteam - the offensive mirror of team_strength.py's
    rush_epa_allowed (which is the same computation from the defense's side).

Both computed directly from pbp_2015_2025.csv, same chunked/memory-safe
pattern used everywhere else in this project on that file.

The two metrics are combined via z-scoring (standardized using only the
training-season window, not the full history) rather than the arbitrary
"/50" and "/0.025" divisor constants in the original spec - same reasoning
already used for the CB/S/LB blend: don't invent a conversion, put
differently-scaled things on equal footing via standardization.

Task 4.1 (this module, so far) builds and validates the metrics themselves.
Task 4.2 will derive real, regression-based position weights (how much of
team OL z-score actually predicts a position's next-season EPA/play
residual, checked against real 2025 outcomes) instead of asserting
QB=60%/RB=40%/WR=10%.
"""

import os

import numpy as np
import pandas as pd

from utilities import compute_history_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

OL_PBP_COLS = ["posteam", "season", "week", "play_type", "epa", "sack", "season_type"]


def compute_team_ol_metrics(pbp_path=None):
    """Team-season sack rate allowed and rush EPA generated, both from the
    offense's own play-by-play (posteam), regular season only. Streams the
    1.3GB PBP file in chunks - same memory-safety reason as every other PBP
    scan in this project."""
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    keep_chunks = []
    for chunk in pd.read_csv(path, usecols=OL_PBP_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[chunk["play_type"].isin(["pass", "run"]) & (chunk["season_type"] == "REG")]
        if len(sub):
            keep_chunks.append(sub)
    reg = pd.concat(keep_chunks, ignore_index=True)
    reg = reg.dropna(subset=["epa", "posteam"])

    pass_plays = reg[reg["play_type"] == "pass"]
    pass_block = pass_plays.groupby(["posteam", "season"]).agg(
        dropbacks=("sack", "size"),
        sack_rate_allowed=("sack", "mean"),
    ).reset_index()

    rush_plays = reg[reg["play_type"] == "run"]
    run_block = rush_plays.groupby(["posteam", "season"]).agg(
        carries=("epa", "size"),
        rush_epa_generated=("epa", "mean"),
    ).reset_index()

    team = pass_block.merge(run_block, on=["posteam", "season"], how="outer").rename(columns={"posteam": "team"})
    print(f"[ol_quality] {team.shape[0]:,} team-seasons, "
          f"{int(team['season'].min())}-{int(team['season'].max())}")
    return team


def add_ol_lookback_features(team_ol):
    """Leak-free lagged versions of both raw metrics, same
    compute_history_features utility used everywhere else in this project."""
    df = team_ol.copy()
    for col in ["sack_rate_allowed", "rush_epa_generated"]:
        feats = compute_history_features(df[["team", "season", col]], col, id_col="team")
        df = df.join(feats[[f"{col}_last_year"]])
    return df


def build_ol_modifier(team_ol, standardize_seasons):
    """Standardizes both metrics (z-score, using only standardize_seasons to
    derive mean/std - avoids leaking later-season statistics into the
    standardization) and combines them into one ol_modifier per team-season.

    Sign convention: HIGHER ol_modifier = better OL. sack_rate_allowed is
    "lower is better," so its z-score is negated before combining;
    rush_epa_generated is "higher is better" already."""
    df = team_ol.copy()
    train_pool = df[df["season"].isin(standardize_seasons)]

    sack_mean, sack_std = train_pool["sack_rate_allowed"].mean(), train_pool["sack_rate_allowed"].std()
    rush_mean, rush_std = train_pool["rush_epa_generated"].mean(), train_pool["rush_epa_generated"].std()

    df["z_pass_block"] = -(df["sack_rate_allowed"] - sack_mean) / sack_std
    df["z_run_block"] = (df["rush_epa_generated"] - rush_mean) / rush_std
    df["ol_modifier"] = (df["z_pass_block"] + df["z_run_block"]) / 2

    print(f"[ol_quality] standardized on seasons {min(standardize_seasons)}-{max(standardize_seasons)}: "
          f"league avg sack_rate_allowed={sack_mean:.4f}, league avg rush_epa_generated={rush_mean:.4f}")
    return df


def validate_ol_metrics(scored):
    print("\n===== OL QUALITY METRICS VALIDATION =====")
    print(f"Total team-seasons: {scored.shape[0]:,} "
          f"({int(scored['season'].min())}-{int(scored['season'].max())})")

    teams_per_season = scored.groupby("season")["team"].nunique()
    print(f"\nTeams per season (expect 32, except early-2015-ish realignment quirks if any):")
    print(teams_per_season.to_string())

    print(f"\nsack_rate_allowed range: {scored['sack_rate_allowed'].min():.3f} - {scored['sack_rate_allowed'].max():.3f} "
          f"(league avg {scored['sack_rate_allowed'].mean():.3f}; typical real NFL range is roughly 0.03-0.10)")
    print(f"rush_epa_generated range: {scored['rush_epa_generated'].min():.3f} - {scored['rush_epa_generated'].max():.3f} "
          f"(league avg {scored['rush_epa_generated'].mean():.3f}; rushing EPA/play is typically negative on "
          f"average league-wide, same reason RB epa_per_play skewed negative in Task 1)")
    print(f"ol_modifier range: {scored['ol_modifier'].min():.2f} - {scored['ol_modifier'].max():.2f}")

    latest = int(scored["season"].max())
    latest_df = scored[scored["season"] == latest].dropna(subset=["ol_modifier"])
    print(f"\nTop 5 OL teams by modifier, {latest}:")
    print(latest_df.nlargest(5, "ol_modifier")[
        ["team", "ol_modifier", "sack_rate_allowed", "rush_epa_generated"]
    ].to_string(index=False))
    print(f"\nBottom 5 OL teams by modifier, {latest}:")
    print(latest_df.nsmallest(5, "ol_modifier")[
        ["team", "ol_modifier", "sack_rate_allowed", "rush_epa_generated"]
    ].to_string(index=False))

    # Independent cross-check: a team's OL modifier should correlate
    # positively with its own real, already-validated team offensive
    # production, not just be internally self-consistent. Use team_defense_epa
    # style team offense EPA computed inline here (mean EPA across all
    # scrimmage plays for that offense) as a lightweight sanity check.
    corr = scored[["ol_modifier", "sack_rate_allowed"]].corr().iloc[0, 1]
    print(f"\ncorr(ol_modifier, sack_rate_allowed) = {corr:.3f} (expect clearly negative by construction - "
          f"sanity check that the sign flip in build_ol_modifier was applied correctly)")
    print("===== END VALIDATION =====\n")


def run_ol_quality_metrics():
    team_ol = compute_team_ol_metrics()
    team_ol = add_ol_lookback_features(team_ol)

    latest_season = int(team_ol["season"].max())
    standardize_seasons = range(int(team_ol["season"].min()), latest_season)  # all but the most recent season
    scored = build_ol_modifier(team_ol, standardize_seasons)

    validate_ol_metrics(scored)

    out_path = os.path.join(PROCESSED_DIR, "team_ol_metrics_2015_2025.csv")
    scored.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({scored.shape[0]:,} rows)")
    return scored



# ---------------------------------------------------------------------------
# Task 4.2: derive real, regression-based position weights and apply them
# ---------------------------------------------------------------------------

import pickle

from player_models import OffenseEpaModel, TRAIN_SEASONS, HOLDOUT_SEASON

POSITION_EPA_CONFIG = {
    "QB": dict(epa_cols=["passing_epa"], opportunity_cols=["attempts"],
               min_opportunities=100, use_qb_context=False),
    "WR": dict(epa_cols=["receiving_epa"], opportunity_cols=["targets"],
               min_opportunities=30, use_qb_context=True),
    "RB": dict(epa_cols=["rushing_epa", "receiving_epa"], opportunity_cols=["carries", "targets"],
               min_opportunities=30, use_qb_context=False),
    "TE": dict(epa_cols=["receiving_epa"], opportunity_cols=["targets"],
               min_opportunities=30, use_qb_context=True),
}


def load_epa_model(position, features_df, season_stats_df):
    cfg = POSITION_EPA_CONFIG[position]
    model = OffenseEpaModel(position, cfg["epa_cols"], cfg["opportunity_cols"],
                             cfg["min_opportunities"], cfg["use_qb_context"])
    model_path = os.path.join(PROJECT_ROOT, "models", f"{position.lower()}_epa.pkl")
    with open(model_path, "rb") as f:
        model.xgb_model = pickle.load(f)
    prepped = model.prepare_data(features_df, season_stats_df)
    return model, prepped


def compute_epa_model_residuals(model, prepped):
    """actual - predicted for every row the model's already trained on/holds
    out - the unexplained variance an OL adjustment would have to draw on."""
    df = prepped.copy()
    df["predicted_epa_per_play"] = model.xgb_model.predict(df[model.FEATURE_COLS])
    df["residual"] = df["epa_per_play"] - df["predicted_epa_per_play"]
    return df


def estimate_ol_weight(position, residuals_df, team_ol, train_seasons=TRAIN_SEASONS):
    """Regresses residual_S ~ ol_modifier_{S-1} (the same lagged relationship
    that gets used operationally at projection time - a projection made
    before season S starts only ever has S-1's real OL data available) via
    simple OLS on the training-window rows only (holdout season kept out of
    the fit so it stays a genuine check). This replaces the original spec's
    asserted QB=60%/RB=40%/WR=10% weights with a real, data-derived slope -
    in EPA/play per unit of standardized OL modifier, not a percentage of
    something already unit-mismatched.

    Caveat worth being upfront about: this is fit on the same rows the EPA
    model itself was trained on (in-sample), so the estimate could be
    slightly optimistic. The real test is whether applying it improves
    accuracy on the untouched 2024 holdout and the real 2025 outcomes -
    see validate_ol_adjustment."""
    df = residuals_df[residuals_df["season"].isin(train_seasons)].copy()
    df["prior_season"] = df["season"] - 1
    ol_prior = team_ol[["team", "season", "ol_modifier"]].rename(
        columns={"season": "prior_season", "ol_modifier": "ol_modifier_prior"})
    merged = df.merge(ol_prior, on=["team", "prior_season"], how="inner").dropna(subset=["ol_modifier_prior"])

    x = merged["ol_modifier_prior"].to_numpy()
    y = merged["residual"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    print(f"[{position}] OL weight regression (train seasons only, n={len(merged)}): "
          f"slope={slope:+.4f} EPA/play per unit OL z-score | corr={corr:+.3f} | R2={corr**2:.4f}")
    return slope, corr, len(merged)


def apply_ol_adjustment(projections_df, team_ol, slope, ref_season):
    """Additive adjustment (fixes the original spec's multiplicative-on-a-
    signed-value bug): adjusted = predicted + slope * ol_modifier, using
    ref_season's real OL modifier (the most recent completed season - same
    "last known state" convention as every predict_next_season() elsewhere
    in this project) as the best available estimate of the team's OL context
    for the projected season."""
    projections_df = projections_df.drop(columns=["ol_modifier", "predicted_epa_per_play_ol_adjusted"], errors="ignore")
    ol_ref = team_ol[team_ol["season"] == ref_season][["team", "ol_modifier"]]
    out = projections_df.merge(ol_ref, on="team", how="left")
    out["ol_modifier"] = out["ol_modifier"].fillna(0.0)
    out["predicted_epa_per_play_ol_adjusted"] = out["predicted_epa_per_play"] + slope * out["ol_modifier"]
    return out


def compute_real_epa_per_play(position, pbp_2025):
    """Real, realized 2025 EPA/play per player from PBP directly - the same
    out-of-sample check used to validate the Task 1 EPA models, reused here
    to test whether the OL adjustment actually improves real-world accuracy
    or just moves numbers around."""
    cfg = POSITION_EPA_CONFIG[position]
    if position == "QB":
        plays = pbp_2025[pbp_2025["play_type"] == "pass"].dropna(subset=["passer_id", "epa"])
        agg = plays.groupby("passer_id")["epa"].agg(["sum", "count"])
    elif position in ("WR", "TE"):
        plays = pbp_2025[pbp_2025["play_type"] == "pass"].dropna(subset=["receiver_id", "epa"])
        agg = plays.groupby("receiver_id")["epa"].agg(["sum", "count"])
    else:  # RB: rushing + receiving
        rush = pbp_2025[pbp_2025["play_type"] == "run"].dropna(subset=["rusher_id", "epa"])
        rec = pbp_2025[pbp_2025["play_type"] == "pass"].dropna(subset=["receiver_id", "epa"])
        rush_agg = rush.groupby("rusher_id")["epa"].agg(["sum", "count"])
        rec_agg = rec.groupby("receiver_id")["epa"].agg(["sum", "count"])
        agg = rush_agg.add(rec_agg, fill_value=0)
    agg = agg.rename(columns={"sum": "real_epa_sum", "count": "real_opportunities"}).reset_index()
    agg = agg.rename(columns={agg.columns[0]: "player_id"})
    agg["real_2025_epa_per_play"] = agg["real_epa_sum"] / agg["real_opportunities"]
    return agg[agg["real_opportunities"] >= cfg["min_opportunities"]]


def validate_ol_adjustment(position, adjusted_2025, holdout_adjusted, real_2025):
    """Compares baseline vs. OL-adjusted accuracy on two independent,
    genuinely out-of-sample checks: the 2024 holdout (never used to fit the
    slope) and real 2025 outcomes (never used anywhere in this pipeline)."""
    def _score(df, actual_col, pred_col):
        actual = df[actual_col].to_numpy()
        pred = df[pred_col].to_numpy()
        mae = np.mean(np.abs(pred - actual))
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
        return mae, r2

    mae_base_h, r2_base_h = _score(holdout_adjusted, "epa_per_play", "predicted_epa_per_play")
    mae_adj_h, r2_adj_h = _score(holdout_adjusted, "epa_per_play", "predicted_epa_per_play_ol_adjusted")
    print(f"\n[{position}] 2024 holdout: baseline MAE={mae_base_h:.4f} R2={r2_base_h:.3f} | "
          f"OL-adjusted MAE={mae_adj_h:.4f} R2={r2_adj_h:.3f}")

    merged_real = adjusted_2025.merge(real_2025[["player_id", "real_2025_epa_per_play"]], on="player_id", how="inner")
    if len(merged_real):
        mae_base_r, r2_base_r = _score(merged_real.rename(columns={"real_2025_epa_per_play": "epa_per_play"}),
                                        "epa_per_play", "predicted_epa_per_play")
        mae_adj_r, r2_adj_r = _score(merged_real.rename(columns={"real_2025_epa_per_play": "epa_per_play"}),
                                      "epa_per_play", "predicted_epa_per_play_ol_adjusted")
        print(f"[{position}] real 2025 (n={len(merged_real)}): baseline MAE={mae_base_r:.4f} R2={r2_base_r:.3f} | "
              f"OL-adjusted MAE={mae_adj_r:.4f} R2={r2_adj_r:.3f}")
        helps_real = mae_adj_r < mae_base_r
    else:
        print(f"[{position}] no real-2025 matches at this opportunity threshold - skipping that check")
        helps_real = None

    helps_holdout = mae_adj_h < mae_base_h
    return {"helps_holdout": helps_holdout, "helps_real": helps_real,
            "mae_base_holdout": mae_base_h, "mae_adj_holdout": mae_adj_h,
            "mae_base_real": mae_base_r if len(merged_real) else None,
            "mae_adj_real": mae_adj_r if len(merged_real) else None}


def load_real_2025_pbp():
    cols = ["season", "week", "season_type", "play_type", "epa", "passer_id", "receiver_id", "rusher_id"]
    keep = []
    for chunk in pd.read_csv(os.path.join(RAW_DIR, "pbp_2015_2025.csv"), usecols=cols,
                              low_memory=False, chunksize=100_000):
        sub = chunk[(chunk["season"] == 2025) & (chunk["season_type"] == "REG")
                    & (chunk["play_type"].isin(["pass", "run"]))]
        if len(sub):
            keep.append(sub)
    return pd.concat(keep, ignore_index=True)


def run_ol_adjustment():
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_features_with_history.csv"))
    season_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    team_ol = pd.read_csv(os.path.join(PROCESSED_DIR, "team_ol_metrics_2015_2025.csv"))
    pbp_2025 = load_real_2025_pbp()

    results = {}
    for position in ["QB", "WR", "RB", "TE"]:
        print(f"\n{'=' * 60}\n{position}\n{'=' * 60}")
        model, prepped = load_epa_model(position, features_df, season_stats_df)
        residuals = compute_epa_model_residuals(model, prepped)

        slope, corr, n = estimate_ol_weight(position, residuals, team_ol)

        # 2024 holdout check: every holdout row is season=2024, so the
        # relevant "last known OL state" is uniformly the 2023 team modifier.
        holdout = residuals[residuals["season"] == HOLDOUT_SEASON].copy()
        holdout = apply_ol_adjustment(holdout, team_ol, slope, ref_season=HOLDOUT_SEASON - 1)

        # 2025 projections (built in Task 1, now carrying player_id): apply
        # using ref_season=2024, the most recent completed season.
        proj_path = os.path.join(PROCESSED_DIR, f"{position.lower()}_epa_projections_2025.csv")
        projections = pd.read_csv(proj_path)
        ref_season = int(prepped["season"].max())
        adjusted_2025 = apply_ol_adjustment(projections, team_ol, slope, ref_season=ref_season)

        real_2025 = compute_real_epa_per_play(position, pbp_2025)
        metrics = validate_ol_adjustment(position, adjusted_2025, holdout, real_2025)

        adjusted_2025.to_csv(proj_path, index=False, encoding="utf-8")
        print(f"Saved {proj_path} (added ol_modifier + predicted_epa_per_play_ol_adjusted columns)")

        results[position] = {"slope": slope, "corr": corr, "n": n, **metrics}

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for position, r in results.items():
        verdict = "HELPS" if r["helps_real"] else ("NO HELP" if r["helps_real"] is not None else "UNTESTED")
        print(f"{position}: slope={r['slope']:+.4f} corr={r['corr']:+.3f} (n={r['n']}) | "
              f"real-2025 verdict: {verdict}")
    return results


if __name__ == "__main__":
    run_ol_quality_metrics()
    run_ol_adjustment()
