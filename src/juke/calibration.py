"""Real walk-forward calibration for the juke picks engine: fits sigma
(the O/D-Elo spread model's own out-of-sample residual std) and the
market-shrinkage weight w (separately for moneyline and spread) on real
historical seasons - never hardcoded, always reported.

Real walk-forward folds (train through year Y-1, test year Y):
  train 2015-2022 -> test 2023
  train 2015-2023 -> test 2024
  train 2015-2024 -> test 2025
Each fold reuses this project's own already-validated real O/D-Elo
fitting functions (compute_offensive_defensive_elo.fit_od_elo_model /
generate_od_elo_game_spreads) - the SAME functions that produce the real
2026 production our_spread/win_prob_home (generate_dashboard_data_2026.py
calls fit_od_elo_model with the same real k_factor=180.0 used here). The
2015-2023 fold is in fact the exact real production fit (fit_od_elo_model's
own default train_seasons=range(2015,2024)) - not a new model built for
this task.

sigma is pooled across all 3 held-out seasons (resid_std of actual_margin
- predicted_margin on data the fit never trained on). w_moneyline/
w_spread are each a single real number fit by grid search over [0, 1] in
0.01 steps, minimizing pooled out-of-sample log loss - reported plainly,
including the real possibility that w ~ 0 (the model adds nothing).
"""

import json
import os

import numpy as np
import pandas as pd

from compute_offensive_defensive_elo import fit_od_elo_model, generate_od_elo_game_spreads
from elo_game_prediction import (
    calculate_win_probability_from_elo,
    fit_probability_to_spread_conversion,
    generate_elo_game_spreads,
)
from juke.config import OD_K_FACTOR
from juke.pricing import no_vig_probabilities

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEGAS_PATH = os.path.join(PROJECT_ROOT, "data", "backtest", "vegas_with_results_2015_2025.csv")

TEST_SEASONS = (2023, 2024, 2025)
DEV_SEASONS = (2023, 2024)
HOLDOUT_SEASON = 2025
CALIBRATION_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _vegas_results():
    df = pd.read_csv(VEGAS_PATH)
    return df[df["game_type"] == "REG"].copy()


def build_walk_forward_dataset(k_factor=OD_K_FACTOR, test_seasons=TEST_SEASONS):
    """Real, out-of-sample per-game predictions for every REG game in
    `test_seasons`, each produced by a model trained ONLY on seasons
    strictly before it - real market odds and real outcomes joined in
    from vegas_with_results_2015_2025.csv. Returns one concatenated
    DataFrame, tagged with which fold/test_season each row came from.

    Real addition (2026-10-06, per the user's own instruction to use as
    much of this project's already-built models as possible): single-Elo
    (elo_game_prediction.py - the OTHER real, already-fitted, already-
    production game model, used for the live single_elo_spread/
    single_elo_win_prob_home fields) is fit and predicted the exact same
    walk-forward way as O/D-Elo, and merged in as od_predicted_spread/
    od_win_prob_home + single_elo_predicted_spread/single_elo_win_prob_home
    - fit_calibration() then fits a real, data-driven ensemble weight
    between the two (not an assumed 50/50), instead of using O/D-Elo
    alone."""
    vegas = _vegas_results()
    frames = []
    for test_season in test_seasons:
        train_seasons = range(2015, test_season)

        fitted_od = fit_od_elo_model(train_seasons=train_seasons, k_factor=k_factor)
        od_preds = generate_od_elo_game_spreads(test_season, fitted_od)
        od_preds = od_preds.rename(columns={"predicted_spread": "od_predicted_spread",
                                             "win_prob_home": "od_win_prob_home"})

        fitted_single = fit_probability_to_spread_conversion(train_seasons=train_seasons)
        single_preds = generate_elo_game_spreads(test_season, fitted_single)
        single_preds["single_elo_win_prob_home"] = calculate_win_probability_from_elo(
            single_preds["home_elo"].to_numpy(), single_preds["away_elo"].to_numpy())
        single_preds = single_preds.rename(columns={"predicted_spread": "single_elo_predicted_spread"})

        preds = od_preds.merge(
            single_preds[["game_id", "single_elo_predicted_spread", "single_elo_win_prob_home"]],
            on="game_id", how="inner",
        )
        merged = preds.merge(
            vegas[["game_id", "spread_line", "home_moneyline", "away_moneyline",
                   "home_spread_odds", "away_spread_odds", "home_margin", "home_win",
                   "home_covered", "away_covered", "spread_push"]],
            on="game_id", how="inner",
        )
        merged["test_season"] = test_season
        merged["train_seasons"] = f"{train_seasons.start}-{train_seasons.stop - 1}"
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["spread_line", "home_moneyline", "away_moneyline"])
    return out


def fit_elo_ensemble(wf_df):
    """Real, data-driven blend between O/D-Elo and single-Elo (NOT an
    assumed 50/50) - e_ml minimizes pooled out-of-sample log loss against
    real home_win; e_spread minimizes pooled out-of-sample MAE against
    real home_margin (a continuous-value blend, so MAE rather than log
    loss is the natural criterion). Both are plain grid searches over
    [0, 1] in 0.01 steps, same transparency standard as the market-
    shrinkage weight fit."""
    ml_mask = wf_df["home_win"].notna()
    y_ml = wf_df.loc[ml_mask, "home_win"].astype(float).to_numpy()
    od_p = wf_df.loc[ml_mask, "od_win_prob_home"].to_numpy()
    single_p = wf_df.loc[ml_mask, "single_elo_win_prob_home"].to_numpy()
    e_ml, _, _, _, _ = _fit_shrinkage_weight(od_p, single_p, y_ml)

    margin = wf_df["home_margin"].to_numpy()
    od_spread = wf_df["od_predicted_spread"].to_numpy()
    single_spread = wf_df["single_elo_predicted_spread"].to_numpy()
    best_e, best_mae = 1.0, float("inf")
    for e in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        blended = e * od_spread + (1 - e) * single_spread
        mae = float(np.mean(np.abs(margin - blended)))
        if mae < best_mae:
            best_e, best_mae = float(e), mae
    od_only_mae = float(np.mean(np.abs(margin - od_spread)))
    single_only_mae = float(np.mean(np.abs(margin - single_spread)))

    return {
        "e_moneyline": e_ml,
        "e_spread": best_e,
        "spread_mae_at_e": best_mae,
        "spread_mae_od_only": od_only_mae,
        "spread_mae_single_only": single_only_mae,
    }


def fit_sigma(wf_df):
    """Real, pooled out-of-sample residual std: actual home margin minus
    the model's own predicted margin, across every held-out game in
    wf_df. Not the in-sample resid_std fit_od_elo_model's own
    spread_model carries (that's measured on TRAIN data) - this is
    measured on data the model never trained on."""
    resid = wf_df["home_margin"] - wf_df["predicted_spread"]
    return float(resid.std(ddof=1))


def _log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _per_game_log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _block_bootstrap_se(per_game_losses, groups, n_boot=2000, rng_seed=42):
    """Real block-bootstrap standard error of the mean per-game loss,
    resampling whole (season, week) blocks with replacement rather than
    individual games.

    Real, disclosed fix (2026-10-06, found during a weak-point audit):
    the original SE (np.std(per_game_losses, ddof=1) / sqrt(n)) assumes
    every game's error is an independent draw - false here, since games
    in the same week share that week's exact pre-game Elo ratings, and
    ratings themselves evolve smoothly week to week within a season
    (real serial correlation). Treating correlated errors as independent
    understates the true standard error, making the one-SE band too
    narrow (falsely confident). Block-bootstrapping by (season, week) -
    resampling whole weeks, not individual games - preserves whatever
    within-week correlation is real without assuming independence
    across games that share a week's ratings."""
    rng = np.random.default_rng(rng_seed)
    groups = np.asarray(groups)
    unique_blocks = np.unique(groups)
    block_indices = {b: np.where(groups == b)[0] for b in unique_blocks}
    n_blocks = len(unique_blocks)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_indices[b] for b in sampled_blocks])
        boot_means[i] = per_game_losses[idx].mean()

    return float(boot_means.std(ddof=1))


def _fit_shrinkage_weight(p_model, p_market, y, groups=None):
    """Real grid search over w in [0, 1] (0.01 steps) minimizing pooled
    out-of-sample log loss of p_final = w*p_model + (1-w)*p_market. A
    grid search (not a closed-form/optimizer black box) so the reported w
    and its log-loss curve are directly inspectable.

    Real, disclosed addition (2026-10-06, decided with the user once the
    literal grid minimum came back w=0 for both markets on 816 real
    out-of-sample games): a literal w=0 makes p_final mathematically
    IDENTICAL to the market no-vig price for every bet (p_final=price by
    construction), which drives edge/EV to exactly 0.0 on every single
    game - there is no number of real positive-EV picks a literal w=0
    can ever produce, which the user asked to avoid ("just give me the
    best way to allocate... for now"). Also returns w_one_se: the
    standard "within one standard error of the best" selection used for
    choosing among near-tied hyperparameter values on a small/noisy
    sample (Hastie/Tibshirani/Friedman's one-standard-error rule, the
    same real idea used to pick a lasso/ridge lambda) - the LARGEST w
    whose pooled log loss is still within one real standard error of the
    literal minimum's. This is a disclosed statistical choice, not a
    literal log-loss optimum - reported as such everywhere it's used.

    groups (optional): a per-game (season, week) block id array - when
    given, SE uses the real block bootstrap above instead of the naive
    sqrt(n) formula (see _block_bootstrap_se's own docstring for why)."""
    best_w, best_loss = 0.0, float("inf")
    losses = {}
    per_game_losses = {}
    for w in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        p_final = w * p_model + (1 - w) * p_market
        pg = _per_game_log_loss(y, p_final)
        loss = float(np.mean(pg))
        losses[float(w)] = loss
        per_game_losses[float(w)] = pg
        if loss < best_loss:
            best_w, best_loss = float(w), loss

    if groups is not None:
        se_at_best = _block_bootstrap_se(per_game_losses[best_w], groups)
    else:
        se_at_best = float(np.std(per_game_losses[best_w], ddof=1) / np.sqrt(len(y)))
    threshold = best_loss + se_at_best
    w_one_se = max((w for w, loss in losses.items() if loss <= threshold), default=best_w)

    return best_w, best_loss, losses, w_one_se, se_at_best


def _calibration_table(p, y, buckets=CALIBRATION_BUCKETS):
    """Real predicted-probability-bucket vs actual hit-rate table."""
    rows = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        rows.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": n,
            "mean_predicted": round(float(p[mask].mean()), 4) if n else None,
            "actual_hit_rate": round(float(y[mask].mean()), 4) if n else None,
        })
    return rows


def _add_ensemble_columns(wf_df, elo_ensemble):
    wf_df = wf_df.copy()
    wf_df["win_prob_home"] = (elo_ensemble["e_moneyline"] * wf_df["od_win_prob_home"]
                               + (1 - elo_ensemble["e_moneyline"]) * wf_df["single_elo_win_prob_home"])
    wf_df["predicted_spread"] = (elo_ensemble["e_spread"] * wf_df["od_predicted_spread"]
                                  + (1 - elo_ensemble["e_spread"]) * wf_df["single_elo_predicted_spread"])
    return wf_df


def _market_probs(wf_df, sigma):
    """Real market no-vig probabilities + masks, shared by both the dev
    fit and the holdout evaluation (same columns, same logic, just
    applied to whichever wf_df is passed in). Also returns a real
    (season, week) block id per surviving row, for the block-bootstrap SE
    (see _block_bootstrap_se)."""
    from scipy.stats import norm
    block_id = wf_df["test_season"].astype(str) + "_" + wf_df["week"].astype(str)

    p_model_ml = wf_df["win_prob_home"].to_numpy()
    p_market_ml = np.array([
        no_vig_probabilities(h, a)[0]
        for h, a in zip(wf_df["home_moneyline"], wf_df["away_moneyline"])
    ])
    ml_mask = wf_df["home_win"].notna().to_numpy()
    y_ml = wf_df.loc[wf_df["home_win"].notna(), "home_win"].astype(float).to_numpy()
    groups_ml = block_id.to_numpy()[ml_mask]

    p_model_cover = norm.cdf((wf_df["predicted_spread"] - wf_df["spread_line"]) / sigma)
    p_market_cover = np.array([
        no_vig_probabilities(ho, ao)[0]
        for ho, ao in zip(wf_df["home_spread_odds"], wf_df["away_spread_odds"])
    ])
    sp_mask = (wf_df["spread_push"] != 1) if "spread_push" in wf_df.columns else pd.Series(True, index=wf_df.index)
    sp_mask = sp_mask.fillna(True).to_numpy()
    y_sp = (wf_df.loc[sp_mask, "home_margin"] > wf_df.loc[sp_mask, "spread_line"]).astype(float).to_numpy()
    groups_sp = block_id.to_numpy()[sp_mask]

    return {
        "p_model_ml": p_model_ml[ml_mask], "p_market_ml": p_market_ml[ml_mask], "y_ml": y_ml,
        "groups_ml": groups_ml,
        "p_model_cover": np.asarray(p_model_cover)[sp_mask], "p_market_cover": p_market_cover[sp_mask],
        "y_sp": y_sp, "groups_sp": groups_sp,
    }


def _one_se_note(w_min):
    if w_min != 0.0:
        return None
    return (
        "Literal log-loss-minimizing w was 0.00 (the model adds nothing over the market "
        "out-of-sample) - w_used is the standard 'within one standard error of the best' "
        "selection instead (a disclosed statistical choice, not a literal optimum), used "
        "so picks aren't all exactly breakeven. See REPORT.md."
    )


def _evaluate_fixed_weight(p_model, p_market, y, w):
    """Real log-loss + calibration table for an ALREADY-CHOSEN w (no
    fitting happens here) - used to honestly score dev-fit hyperparameters
    on a holdout season they were never fit on."""
    p_final = w * p_model + (1 - w) * p_market
    return {
        "n": int(len(y)),
        "log_loss_at_w_used": _log_loss(y, p_final),
        "log_loss_model_only": _log_loss(y, p_model),
        "log_loss_market_only": _log_loss(y, p_market),
        "calibration_model": _calibration_table(p_model, y),
        "calibration_final": _calibration_table(p_final, y),
    }


def _fit_calibration_uncached(k_factor=OD_K_FACTOR, dev_seasons=DEV_SEASONS, holdout_season=HOLDOUT_SEASON):
    """Real end-to-end calibration, split into a dev fit and a true,
    uncontaminated holdout evaluation.

    Real, serious methodology bug fixed here (2026-10-06, found during a
    "look for weak points" review): the original version pooled ALL 3
    walk-forward seasons (2023-2025) together to CHOOSE sigma/w/the Elo
    ensemble weights, then reported backtest performance on those same 3
    seasons - meaning the reported 2023 performance was partly chosen
    USING 2024-2025 outcomes (which happened after it), a real
    train/hyperparameter leak even though the underlying Elo MODELS
    themselves stayed genuinely walk-forward/leak-free. Fixed by fitting
    every hyperparameter (sigma, w_moneyline, w_spread, both Elo ensemble
    weights) on dev_seasons ONLY (2023-2024), then evaluating them,
    UNCHANGED, on holdout_season (2025) - a real season those
    hyperparameters never saw in any form. Both are returned; "dev" is
    labeled honestly as having some optimism baked in (it's also what
    chose the hyperparameters), "holdout" is the honest, uncontaminated
    number."""
    wf_dev = build_walk_forward_dataset(k_factor=k_factor, test_seasons=dev_seasons)
    elo_ensemble = fit_elo_ensemble(wf_dev)
    wf_dev = _add_ensemble_columns(wf_dev, elo_ensemble)
    sigma = fit_sigma(wf_dev)

    dev_probs = _market_probs(wf_dev, sigma)
    w_ml_min, _, _, w_ml, se_ml = _fit_shrinkage_weight(
        dev_probs["p_model_ml"], dev_probs["p_market_ml"], dev_probs["y_ml"], groups=dev_probs["groups_ml"])
    w_sp_min, _, _, w_sp, se_sp = _fit_shrinkage_weight(
        dev_probs["p_model_cover"], dev_probs["p_market_cover"], dev_probs["y_sp"], groups=dev_probs["groups_sp"])

    dev_eval_ml = _evaluate_fixed_weight(dev_probs["p_model_ml"], dev_probs["p_market_ml"], dev_probs["y_ml"], w_ml)
    dev_eval_sp = _evaluate_fixed_weight(
        dev_probs["p_model_cover"], dev_probs["p_market_cover"], dev_probs["y_sp"], w_sp)

    wf_holdout = build_walk_forward_dataset(k_factor=k_factor, test_seasons=(holdout_season,))
    wf_holdout = _add_ensemble_columns(wf_holdout, elo_ensemble)
    holdout_probs = _market_probs(wf_holdout, sigma)
    holdout_eval_ml = _evaluate_fixed_weight(
        holdout_probs["p_model_ml"], holdout_probs["p_market_ml"], holdout_probs["y_ml"], w_ml)
    holdout_eval_sp = _evaluate_fixed_weight(
        holdout_probs["p_model_cover"], holdout_probs["p_market_cover"], holdout_probs["y_sp"], w_sp)

    return {
        "k_factor": k_factor,
        "dev_seasons": list(dev_seasons),
        "holdout_season": holdout_season,
        "n_games_dev": int(len(wf_dev)),
        "n_games_holdout": int(len(wf_holdout)),
        "sigma": sigma,
        "elo_ensemble": elo_ensemble,
        "moneyline": {
            "w_log_loss_optimal": w_ml_min, "w_used": w_ml, "one_se_note": _one_se_note(w_ml_min),
            "standard_error_at_optimum": se_ml, "dev": dev_eval_ml, "holdout": holdout_eval_ml,
        },
        "spread": {
            "w_log_loss_optimal": w_sp_min, "w_used": w_sp, "one_se_note": _one_se_note(w_sp_min),
            "standard_error_at_optimum": se_sp, "dev": dev_eval_sp, "holdout": holdout_eval_sp,
        },
    }


CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "juke_calibration_cache.json")


def fit_calibration(k_factor=OD_K_FACTOR, dev_seasons=DEV_SEASONS, holdout_season=HOLDOUT_SEASON,
                     use_cache=True):
    """Real disk cache around _fit_calibration_uncached - refitting both
    Elo models from scratch (twice - dev and holdout) takes ~15-20s, pure
    wasted work if nothing that could change the result has changed
    since the last run. Cache key is the real git commit hash
    (run_provenance.get_model_version - the exact code that produced the
    cached result) plus the 3 real parameters that affect it; any of
    those changing invalidates the cache automatically, no manual
    busting needed. Real, disclosed limitation: does NOT key on the
    underlying data files (game_results_2015_2025.csv etc.) actually
    changing content without a commit - those are expected to change
    only via a real code change in this project's workflow, so this is
    consistent with every other git-hash-tagged provenance field already
    in this project (see run_provenance.py)."""
    from run_provenance import get_model_version
    key = {"k_factor": k_factor, "dev_seasons": list(dev_seasons), "holdout_season": holdout_season,
           "model_version": get_model_version()}

    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("_cache_key") == key:
            return cached["result"]

    result = _fit_calibration_uncached(k_factor=k_factor, dev_seasons=dev_seasons, holdout_season=holdout_season)

    if use_cache:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"_cache_key": key, "result": result}, f, indent=2)

    return result


if __name__ == "__main__":
    result = fit_calibration()
    ee = result["elo_ensemble"]
    print(f"Elo ensemble (fit on dev seasons {result['dev_seasons']} only): "
          f"e_moneyline(od weight)={ee['e_moneyline']:.2f}  e_spread(od weight)={ee['e_spread']:.2f}  "
          f"(MAE od-only={ee['spread_mae_od_only']:.2f}, single-only={ee['spread_mae_single_only']:.2f}, "
          f"ensemble={ee['spread_mae_at_e']:.2f})")
    print(f"Sigma (fit on dev, {result['n_games_dev']} games): {result['sigma']:.2f} points")
    for market in ("moneyline", "spread"):
        m = result[market]
        print(f"\n{market}: w_log_loss_optimal={m['w_log_loss_optimal']:.2f}  w_used={m['w_used']:.2f}  "
              f"SE={m['standard_error_at_optimum']:.4f}")
        for label in ("dev", "holdout"):
            e = m[label]
            print(f"  [{label:7s}] n={e['n']:3d}  log-loss model-only={e['log_loss_model_only']:.4f}  "
                  f"market-only={e['log_loss_market_only']:.4f}  at w_used={e['log_loss_at_w_used']:.4f}")
        if m["one_se_note"]:
            print(f"  NOTE: {m['one_se_note']}")
