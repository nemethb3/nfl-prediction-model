"""Phase A Component 2: EPA-Elo Ensemble.

Corrects 2 issues found in the spec before building:

1. Component 1 produced TWO Elo variants: Vegas-informed (circular - imports
   Vegas's own number as the starting rating, corr=+0.828 MAE=1.71) and
   carryover (real win/loss history only, no Vegas anywhere, corr=+0.316
   MAE=2.74 - the genuine, independent signal). The spec's
   get_elo_season_predictions() doesn't say which to use. Using the
   Vegas-informed variant here would make this "EPA-Elo ensemble" actually
   an "EPA-Vegas ensemble" wearing an Elo label - and Component 3 already
   blends with Vegas explicitly and separately, so that would double-count
   Vegas under two different names. Uses the CARRYOVER variant - the only
   one that's a genuinely different signal from EPA (win/loss history vs.
   play-by-play efficiency), which is the actual point of an "EPA-Elo"
   ensemble.

2. learn_optimal_weights() as specified ("test weights, minimize MAE on
   2025") would fit the blend weight on the exact same 2025 outcomes
   validate_ensemble_backtest() then scores it against - the identical
   leakage class already caught and fixed once this session (ensemble.py's
   combine_stacking_loocv). Fixed the same way: leave-one-team-out CV -
   for each held-out team, the weight is chosen using the other 31 teams
   only, then applied to the held-out team. The reported LOOCV MAE is the
   honest accuracy figure; a plain in-sample refit would be optimistic.

Because EPA and carryover-Elo are genuinely different signals (unlike
Ensemble Part 1's candidates, which all shared the same underlying EPA
data and therefore had correlated errors), combining their variances as
independent in blend_ensemble_predictions() is a more defensible
assumption here than it was in Part 1 - flagged explicitly since Part 1's
CI work already showed how easily this assumption breaks when components
share a data source.
"""

import os
import pickle

import numpy as np
import pandas as pd

from constants import (
    EPA_BASELINE_SEASON_CORR_2025 as EPA_BASELINE_CORR,
    EPA_BASELINE_SEASON_MAE_2025 as EPA_BASELINE_MAE,
    ELO_CARRYOVER_SEASON_CORR_2025 as ELO_CARRYOVER_CORR,
    ELO_CARRYOVER_SEASON_MAE_2025 as ELO_CARRYOVER_MAE,
    VEGAS_BASELINE_SEASON_CORR_2025 as VEGAS_BASELINE_CORR,
    VEGAS_BASELINE_SEASON_MAE_2025 as VEGAS_BASELINE_MAE,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_epa_season_predictions(season=2025):
    """[team, epa_projected_wins, epa_std_wins]. Loads the existing
    win_projections_{season}.csv if it exists (season=2025); otherwise
    builds it on the fly from team_strength_{season}.csv + the already-
    fitted epa_to_wins.pkl model (season=2026 - no played games yet, same
    build-on-demand pattern every other 2026 preseason deliverable in this
    project uses). epa_std_wins is derived from the file's own
    projected_wins_low/high band (a flat 1-std residual, not team-specific -
    that's how epa_to_wins.py's own CI is built)."""
    path = os.path.join(PROCESSED_DIR, f"win_projections_{season}.csv")
    if os.path.exists(path):
        proj = pd.read_csv(path)
    else:
        from epa_to_wins import project_season_wins
        team_strength = pd.read_csv(os.path.join(PROCESSED_DIR, f"team_strength_{season}.csv"))
        with open(os.path.join(MODELS_DIR, "epa_to_wins.pkl"), "rb") as f:
            model = pickle.load(f)
        proj = project_season_wins(team_strength, model, target_season=season)

    proj = proj.copy()
    proj["epa_std_wins"] = (proj["projected_wins_high"] - proj["projected_wins_low"]) / 2.0
    return proj[["team", "projected_wins", "epa_std_wins"]].rename(columns={"projected_wins": "epa_projected_wins"})


def get_elo_season_predictions(season=2025):
    """[team, elo_projected_wins, elo_std_wins] - the CARRYOVER Elo variant
    (real win/loss history, no Vegas signal), per correction #1 above. Loads
    elo_season_wins_{season}.csv if Component 1 already saved it (season=
    2025); otherwise builds it fresh via elo_model's own functions
    (season=2026)."""
    path = os.path.join(PROCESSED_DIR, f"elo_season_wins_{season}.csv")
    if os.path.exists(path):
        elo = pd.read_csv(path)
        elo = elo.copy()
        elo["elo_std_wins"] = (elo["wins_high_90_carryover"] - elo["wins_low_90_carryover"]) / (2 * 1.645)
        return elo[["team", "projected_wins_carryover", "elo_std_wins"]].rename(
            columns={"projected_wins_carryover": "elo_projected_wins"})

    from elo_model import learn_elo_hyperparameters, run_multi_season_elo, project_season_wins_from_elo, TRAIN_SEASONS
    from game_predictions import _load_schedule_for_season

    k_factor, points_per_win, home_field_elo, _ = learn_elo_hyperparameters()
    _, ratings_at_season_start, _ = run_multi_season_elo(
        range(min(TRAIN_SEASONS), season + 1), k_factor=k_factor, home_field_elo=home_field_elo)
    team_elos = pd.DataFrame(list(ratings_at_season_start[season].items()), columns=["team", "elo_rating"])

    schedule = _load_schedule_for_season(season)
    reg_schedule = schedule[schedule["game_type"] == "REG"]
    proj = project_season_wins_from_elo(team_elos, reg_schedule, home_field_elo)
    proj["elo_std_wins"] = (proj["projected_wins_high_90"] - proj["projected_wins_low_90"]) / (2 * 1.645)
    return proj[["team", "projected_wins", "elo_std_wins"]].rename(columns={"projected_wins": "elo_projected_wins"})


def blend_ensemble_predictions(epa_df, elo_df, weights=(0.5, 0.5)):
    """ensemble_wins = w_epa*epa + w_elo*elo. CI combines both components'
    variance assuming independence (see module docstring - a more defensible
    assumption here than in ensemble.py's Part 1, since EPA and carryover-
    Elo are genuinely different signals rather than both derived from the
    same underlying EPA data)."""
    w_epa, w_elo = weights
    merged = epa_df.merge(elo_df, on="team")
    merged["ensemble_wins"] = w_epa * merged["epa_projected_wins"] + w_elo * merged["elo_projected_wins"]

    combined_std = np.sqrt((w_epa * merged["epa_std_wins"]) ** 2 + (w_elo * merged["elo_std_wins"]) ** 2)
    merged["ensemble_wins_low_90"] = np.clip(merged["ensemble_wins"] - 1.645 * combined_std, 0, 17)
    merged["ensemble_wins_high_90"] = np.clip(merged["ensemble_wins"] + 1.645 * combined_std, 0, 17)

    return merged.rename(columns={"epa_projected_wins": "epa_wins", "elo_projected_wins": "elo_wins"})[
        ["team", "epa_wins", "elo_wins", "ensemble_wins", "ensemble_wins_low_90", "ensemble_wins_high_90"]]


def learn_optimal_weights(epa_df, elo_df, actual_df, weight_grid=np.round(np.arange(0.0, 1.01, 0.05), 2)):
    """Leave-one-team-out CV (see module docstring correction #2): for each
    held-out team, the best w_epa is chosen using the OTHER 31 teams' MAE
    only, then applied to the held-out team - the held-out team's own actual
    result never influences the weight used to predict it. Returns
    (w_epa, w_elo, loocv_mae) - loocv_mae is the honest, leakage-free
    accuracy estimate of this weight-learning process itself."""
    merged = epa_df.merge(elo_df, on="team").merge(actual_df[["team", "actual_wins"]], on="team").reset_index(drop=True)
    n = len(merged)

    chosen_weights, loocv_errors = [], []
    for i in range(n):
        train = merged.drop(index=i)
        best_w, best_mae = None, np.inf
        for w in weight_grid:
            pred = w * train["epa_projected_wins"] + (1 - w) * train["elo_projected_wins"]
            mae = np.mean(np.abs(pred - train["actual_wins"]))
            if mae < best_mae:
                best_mae, best_w = mae, w
        chosen_weights.append(best_w)
        held_out = merged.iloc[i]
        test_pred = best_w * held_out["epa_projected_wins"] + (1 - best_w) * held_out["elo_projected_wins"]
        loocv_errors.append(abs(test_pred - held_out["actual_wins"]))

    loocv_mae = float(np.mean(loocv_errors))
    w_epa = float(np.mean(chosen_weights))
    print(f"[learn_optimal_weights] LOOCV: mean chosen w_epa={w_epa:.2f} (w_elo={1 - w_epa:.2f}) "
          f"across {n} leave-one-team-out folds | LOOCV MAE={loocv_mae:.3f} wins")
    return w_epa, 1.0 - w_epa, loocv_mae


def validate_ensemble_backtest(season=2025, weights=None):
    """Real validation against completed 2025. Reports the LOOCV MAE (honest,
    leakage-free) alongside the in-sample blend (for inspection only - NOT
    the headline number, since fitting and validating weights on the same
    2025 outcomes would be circular)."""
    epa_df = get_epa_season_predictions(season)
    elo_df = get_elo_season_predictions(season)
    actual = pd.read_csv(os.path.join(BACKTEST_DIR, "actual_wins_2025.csv")) if season == 2025 else None
    if actual is None:
        raise ValueError(f"No real final-outcome ground truth available for season {season}")

    loocv_mae = None
    if weights is None:
        w_epa, w_elo, loocv_mae = learn_optimal_weights(epa_df, elo_df, actual)
        weights = (w_epa, w_elo)

    ensemble = blend_ensemble_predictions(epa_df, elo_df, weights)
    merged = ensemble.merge(actual[["team", "actual_wins"]], on="team")

    corr = merged["ensemble_wins"].corr(merged["actual_wins"])
    mae = float(np.mean(np.abs(merged["ensemble_wins"] - merged["actual_wins"])))

    epa_corr = merged["epa_wins"].corr(merged["actual_wins"])
    epa_mae = float(np.mean(np.abs(merged["epa_wins"] - merged["actual_wins"])))
    elo_corr = merged["elo_wins"].corr(merged["actual_wins"])
    elo_mae = float(np.mean(np.abs(merged["elo_wins"] - merged["actual_wins"])))

    print(f"\n{'=' * 70}\nENSEMBLE (EPA-Elo) VALIDATION (real {season}, weights: "
          f"w_epa={weights[0]:.2f}, w_elo={weights[1]:.2f})\n{'=' * 70}")
    print(f"EPA alone            : corr={epa_corr:+.3f} MAE={epa_mae:.2f} wins")
    print(f"Carryover Elo alone   : corr={elo_corr:+.3f} MAE={elo_mae:.2f} wins")
    print(f"Ensemble (in-sample, weights fit on this same 2025 data - optimistic, NOT the headline number): "
          f"corr={corr:+.3f} MAE={mae:.2f} wins")
    if loocv_mae is not None:
        print(f"Ensemble (LOOCV, honest/leakage-free - THIS is the real accuracy figure): MAE={loocv_mae:.2f} wins")
    print(f"\nBaselines for context: EPA={EPA_BASELINE_CORR:+.3f}/{EPA_BASELINE_MAE:.2f} | "
          f"Elo carryover={ELO_CARRYOVER_CORR:+.3f}/{ELO_CARRYOVER_MAE:.2f} | "
          f"Vegas={VEGAS_BASELINE_CORR:+.3f}/{VEGAS_BASELINE_MAE:.2f}")

    headline_mae = loocv_mae if loocv_mae is not None else mae
    best_solo_mae = min(epa_mae, elo_mae)
    if headline_mae < best_solo_mae - 1e-6:
        verdict = "BEATS both EPA and Elo alone"
    elif abs(headline_mae - best_solo_mae) <= 1e-6:
        verdict = "TIES the better solo model (does not improve on it)"
    else:
        verdict = "does NOT beat the better solo model"
    print(f"\nEnsemble {verdict} on MAE ({'LOOCV' if loocv_mae is not None else 'in-sample'} figure, "
          f"weights: w_epa={weights[0]:.2f}/w_elo={weights[1]:.2f})")
    if weights[0] == 0.0:
        print("NOTE: LOOCV selected w_epa=0.00 in every fold - the ensemble has collapsed to pure carryover "
              "Elo. EPA contributes nothing when blended with Elo this way; this mirrors ensemble.py Part 1's "
              "finding (no combining rule beat using the single strongest component alone) with Elo now playing "
              "the role Vegas played there.")

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"ensemble_season_wins_{season}.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8")
    weights_path = os.path.join(DIAGNOSTIC_DIR, "ensemble_optimal_weights.csv")
    pd.DataFrame([{"season": season, "w_epa": weights[0], "w_elo": weights[1], "loocv_mae": loocv_mae}]).to_csv(
        weights_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}\nSaved {weights_path}")
    print("=" * 70)

    return {"w_epa": weights[0], "w_elo": weights[1], "corr": corr, "mae": mae, "loocv_mae": loocv_mae,
            "epa_corr": epa_corr, "epa_mae": epa_mae, "elo_corr": elo_corr, "elo_mae": elo_mae}


def generate_ensemble_predictions_2026(weights=None):
    """2026 ensemble season-win predictions (team totals + CI only - NOT
    game spreads, despite the spec briefly mentioning them here: game-spread
    blending is Component 3's job (blend_with_vegas_game_spreads is defined
    there), and Elo alone has no native points/spread output to blend with
    EPA's spread - repeating that prematurely here would just be scope
    creep on Component 3's already-specified work)."""
    if weights is None:
        epa_2025 = get_epa_season_predictions(2025)
        elo_2025 = get_elo_season_predictions(2025)
        actual_2025 = pd.read_csv(os.path.join(BACKTEST_DIR, "actual_wins_2025.csv"))
        weights = learn_optimal_weights(epa_2025, elo_2025, actual_2025)[:2]

    epa_2026 = get_epa_season_predictions(2026)
    elo_2026 = get_elo_season_predictions(2026)
    ensemble_2026 = blend_ensemble_predictions(epa_2026, elo_2026, weights)

    print(f"\n{'=' * 70}\n2026 ENSEMBLE SEASON-WIN PREDICTIONS (weights: w_epa={weights[0]:.2f}, "
          f"w_elo={weights[1]:.2f} - NOT validated, 2026 hasn't been played)\n{'=' * 70}")
    print(ensemble_2026.sort_values("ensemble_wins", ascending=False).head(10).to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, "ensemble_season_wins_2026.csv")
    ensemble_2026.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {out_path}")
    print("=" * 70)
    return ensemble_2026


if __name__ == "__main__":
    validate_ensemble_backtest()
    generate_ensemble_predictions_2026()
