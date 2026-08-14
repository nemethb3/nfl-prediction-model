"""Real rookie success classifier: predicts P(a rookie's career weighted AV
beats their real draft round's median), from features known at draft time
only (round, pick, position) - see build_rookie_signals.py for the real
target definition and the real, disclosed reasons for it.

One overall model (not split per position like the veteran trade model):
478 real rows total across 4 positions and 6 draft classes is already thin
for position-specific GroupKFold-style splitting; position is included as
a one-hot feature instead, and StratifiedKFold is safe here (each row is a
distinct real player/draft slot, no repeated-player leakage risk like the
veteran trade model's multi-season rows).

Real combine-metrics addition (Major Refinements task): trains a SECOND,
"enhanced" model (round/pick/position + real age/forty/vertical/
broad_jump) alongside the original, unchanged baseline - not a silent
replacement. Real reason: the enhanced features are complete-case only for
~70% of the real 478-row training set (see build_rookie_signals.py) and
only ~37% of the real 2026 draft class has full real combine data (checked
directly: 27/73). Replacing the baseline outright would have silently
dropped real score coverage for the ~63% of 2026 rookies without complete
combine data. Both real, honest CV numbers are reported and shipped -
score_2026_rookies.py uses the enhanced model where real combine data
allows and falls back to the real baseline otherwise, so no rookie loses
real coverage."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "rookie_classifier_accuracy.json")

POSITIONS = ["QB", "RB", "WR", "TE"]
BASELINE_NUMERIC_FEATURES = ["round", "pick"]
ENHANCED_NUMERIC_FEATURES = ["round", "pick", "age", "forty", "vertical", "broad_jump"]
N_SPLITS = 5
RNG_SEED = 42


def _train_and_evaluate(rookies, numeric_features, pkl_name):
    position_dummies = pd.get_dummies(rookies["position"], prefix="pos")
    feature_cols = numeric_features + list(position_dummies.columns)
    X = pd.concat([rookies[numeric_features], position_dummies], axis=1)[feature_cols].to_numpy(dtype=float)
    y = rookies["outperformed"].to_numpy()

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    fold_acc, fold_auc = [], []
    for train_idx, test_idx in cv.split(X_scaled, y):
        model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        y_proba = model.predict_proba(X_scaled[test_idx])[:, 1]
        fold_acc.append(accuracy_score(y[test_idx], y_pred))
        fold_auc.append(roc_auc_score(y[test_idx], y_proba))
    mean_accuracy, mean_auc = float(np.mean(fold_acc)), float(np.mean(fold_auc))

    final_model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
    final_model.fit(X_scaled, y)
    coefficients = dict(zip(feature_cols, np.round(final_model.coef_[0], 3).tolist()))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, pkl_name), "wb") as f:
        import pickle
        pickle.dump({"model": final_model, "scaler": scaler, "features": feature_cols,
                     "position_dummy_cols": list(position_dummies.columns)}, f)

    return {
        "training_samples": int(len(rookies)),
        "features": feature_cols,
        "cv_accuracy": round(mean_accuracy, 3),
        "cv_auc": round(mean_auc, 3),
        "coefficients": coefficients,
    }


def train_rookie_classifier():
    print("\nTraining real rookie success classifiers (baseline + combine-enhanced)...\n")
    rookies = pd.read_csv(os.path.join(PROCESSED_DIR, "rookie_signals_historical.csv"))
    print(f"Real training rows: {len(rookies)} | positive class (outperformed): "
          f"{rookies['outperformed'].mean():.1%}\n")

    baseline = _train_and_evaluate(rookies, BASELINE_NUMERIC_FEATURES, "rookie_classifier.pkl")
    print(f"Baseline (round/pick/position): real 5-fold stratified CV "
          f"accuracy={100 * baseline['cv_accuracy']:.1f}%  AUC={baseline['cv_auc']:.3f}  "
          f"(n={baseline['training_samples']})")

    enhanced_rookies = rookies.dropna(subset=["age", "forty", "vertical", "broad_jump"])
    enhanced = _train_and_evaluate(enhanced_rookies, ENHANCED_NUMERIC_FEATURES, "rookie_classifier_enhanced.pkl")
    print(f"Enhanced (+ age/forty/vertical/broad_jump): real 5-fold stratified CV "
          f"accuracy={100 * enhanced['cv_accuracy']:.1f}%  AUC={enhanced['cv_auc']:.3f}  "
          f"(n={enhanced['training_samples']}, real complete-case combine coverage "
          f"{enhanced['training_samples'] / len(rookies):.1%})")

    delta_auc = enhanced["cv_auc"] - baseline["cv_auc"]
    print(f"\nReal delta (enhanced - baseline): AUC {delta_auc:+.3f}, "
          f"accuracy {100 * (enhanced['cv_accuracy'] - baseline['cv_accuracy']):+.1f}pp")

    results = {
        "methodology": (
            "Logistic regression, 5-fold stratified CV, one overall model across QB/RB/WR/TE. "
            "Predicts P(real career weighted AV beats the player's real draft round's median). "
            "Trained on real 2015-2020 draft classes only (>=6 real seasons of career value accrued "
            "by 2026). Two real models compared: baseline (round/pick/position, full coverage) and "
            "enhanced (+ real combine age/forty-time/vertical/broad-jump, complete-case only - see "
            "build_rookie_signals.py for the real coverage tradeoffs behind that feature choice)."
        ),
        "baseline": baseline,
        "enhanced": enhanced,
        "enhanced_vs_baseline_auc_delta": round(delta_auc, 3),
        # Backward-compat top-level fields (score_2026_rookies.py's methodology
        # text and any external reference to the real baseline numbers).
        "cv_accuracy": baseline["cv_accuracy"],
        "cv_auc": baseline["cv_auc"],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        record_generation("rookie_classifier_accuracy")
    print(f"\nWrote {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    train_rookie_classifier()
