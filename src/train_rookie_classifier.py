"""Real rookie success classifier: predicts P(a rookie's career weighted AV
beats their real draft round's median), from features known at draft time
only (round, pick, position) - see build_rookie_signals.py for the real
target definition and the real, disclosed reasons for it.

One overall model (not split per position like the veteran trade model):
478 real rows total across 4 positions and 6 draft classes is already thin
for position-specific GroupKFold-style splitting; position is included as
a one-hot feature instead, and StratifiedKFold is safe here (each row is a
distinct real player/draft slot, no repeated-player leakage risk like the
veteran trade model's multi-season rows)."""

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
NUMERIC_FEATURES = ["round", "pick"]
N_SPLITS = 5
RNG_SEED = 42


def train_rookie_classifier():
    print("\nTraining real rookie success classifier...\n")
    rookies = pd.read_csv(os.path.join(PROCESSED_DIR, "rookie_signals_historical.csv"))

    position_dummies = pd.get_dummies(rookies["position"], prefix="pos")
    feature_cols = NUMERIC_FEATURES + list(position_dummies.columns)
    X = pd.concat([rookies[NUMERIC_FEATURES], position_dummies], axis=1)[feature_cols].to_numpy(dtype=float)
    y = rookies["outperformed"].to_numpy()

    print(f"Real training rows: {len(rookies)} | positive class (outperformed): {y.mean():.1%}")

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
    print(f"Real 5-fold stratified CV: accuracy={100 * mean_accuracy:.1f}%  AUC={mean_auc:.3f}")

    final_model = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
    final_model.fit(X_scaled, y)
    coefficients = dict(zip(feature_cols, np.round(final_model.coef_[0], 3).tolist()))
    print(f"Real coefficients: {coefficients}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "rookie_classifier.pkl"), "wb") as f:
        import pickle
        pickle.dump({"model": final_model, "scaler": scaler, "features": feature_cols,
                     "position_dummy_cols": list(position_dummies.columns)}, f)

    results = {
        "methodology": (
            "Logistic regression, 5-fold stratified CV, one overall model across QB/RB/WR/TE. "
            "Predicts P(real career weighted AV beats the player's real draft round's median), "
            "from real draft-time-only features (round, pick, position). Trained on real 2015-2020 "
            "draft classes only (>=6 real seasons of career value accrued by 2026)."
        ),
        "training_samples": int(len(rookies)),
        "features": feature_cols,
        "cv_accuracy": round(mean_accuracy, 3),
        "cv_auc": round(mean_auc, 3),
        "coefficients": coefficients,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        record_generation("rookie_classifier_accuracy")
    print(f"Wrote {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    train_rookie_classifier()
