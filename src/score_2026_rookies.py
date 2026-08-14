"""Score the real 2026 draft class (QB/RB/WR/TE) with the trained rookie
classifier - draft-time-only features (round, pick, position), no career
outcome data exists for these players yet (2026 season hasn't been played:
verified via load_draft_picks(), every real season=2026 row has games=NaN).

This produces a SEPARATE export (rookie_scores_2026.json) rather than being
merged into trade_scores_2026.json - the two measure genuinely different
things (P(beats draft-round peers' career value) vs. the veteran trade
model's P(next-season PPR increase)) and blending them into one field would
misrepresent what's being measured. This is real, additional coverage for
players the veteran trade model structurally can't score (no prior NFL
season to compute career-history signals from), not a literal fix to the
294-player trade_scores_2026.json count.

Real combine-metrics addition (Major Refinements task): also scores with
the real, combine-enhanced model (round/pick/position/age/forty/vertical/
broad_jump, real held-out AUC 0.605 vs. the baseline's 0.563 - see
train_rookie_classifier.py) wherever a real 2026 draftee has complete real
combine data. Checked directly: only 27/73 real 2026 QB/RB/WR/TE draftees
do (37%) - the baseline score is always included for every real rookie
(100% coverage, unchanged), and the enhanced score is added alongside it
only when real data supports it, rather than silently dropping the 63%
without full combine testing."""

import json
import os
import pickle

import numpy as np
import pandas as pd

import nflreadpy as nfl

from generation_timestamps import record_generation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "data", "rookie_scores_2026.json")

POSITIONS = ["QB", "RB", "WR", "TE"]
DRAFT_SEASON = 2026
ENHANCED_COMBINE_COLS = ["age", "forty", "vertical", "broad_jump"]


def _score_with_model(rookies, bundle, feature_source_cols):
    model, scaler, features, position_dummy_cols = (
        bundle["model"], bundle["scaler"], bundle["features"], bundle["position_dummy_cols"])
    position_dummies = pd.DataFrame(0, index=rookies.index, columns=position_dummy_cols)
    for col in position_dummy_cols:
        pos = col.replace("pos_", "")
        position_dummies.loc[rookies["position"] == pos, col] = 1
    X = pd.concat([rookies[feature_source_cols], position_dummies], axis=1)[features].to_numpy(dtype=float)
    X_scaled = scaler.transform(X)
    return np.round(model.predict_proba(X_scaled)[:, 1], 3)


def score_2026_rookies():
    print(f"\nScoring real {DRAFT_SEASON} rookie class (QB/RB/WR/TE)...\n")
    with open(os.path.join(MODELS_DIR, "rookie_classifier.pkl"), "rb") as f:
        baseline_bundle = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "rookie_classifier_enhanced.pkl"), "rb") as f:
        enhanced_bundle = pickle.load(f)

    draft = nfl.load_draft_picks().to_pandas()
    rookies = draft[
        (draft["season"] == DRAFT_SEASON)
        & draft["position"].isin(POSITIONS)
        & draft["gsis_id"].notna()
    ].copy()
    print(f"Real {DRAFT_SEASON} draft picks (QB/RB/WR/TE): {len(rookies)}")

    combine = nfl.load_combine().to_pandas()
    rookies = rookies.merge(combine[["pfr_id", "forty", "vertical", "broad_jump"]],
                             left_on="pfr_player_id", right_on="pfr_id", how="left")

    rookies["success_probability"] = _score_with_model(rookies, baseline_bundle, ["round", "pick"])

    has_combine = rookies[ENHANCED_COMBINE_COLS].notna().all(axis=1)
    n_enhanced = int(has_combine.sum())
    print(f"Real 2026 rookies with complete real combine data (age/forty/vertical/broad_jump): "
          f"{n_enhanced}/{len(rookies)}")
    rookies["success_probability_enhanced"] = None
    if n_enhanced > 0:
        enhanced_rows = rookies[has_combine]
        rookies.loc[has_combine, "success_probability_enhanced"] = _score_with_model(
            enhanced_rows, enhanced_bundle, ["round", "pick"] + ENHANCED_COMBINE_COLS)

    scores = {}
    for _, r in rookies.sort_values("gsis_id").iterrows():
        scores[r["gsis_id"]] = {
            "name": r["pfr_player_name"],
            "position": r["position"],
            "team": r["team"],
            "draft_round": int(r["round"]),
            "draft_pick": int(r["pick"]),
            "success_probability": float(r["success_probability"]),
            "success_probability_enhanced": (
                float(r["success_probability_enhanced"]) if pd.notna(r["success_probability_enhanced"]) else None
            ),
        }

    output = {
        "draft_season": DRAFT_SEASON,
        "methodology_note": (
            "Real rookie success classifier score: P(this player's eventual career weighted "
            "Approximate Value beats their real draft round's historical median). "
            "success_probability: draft-time-only signals (round, pick, position) - real, held-out "
            "CV accuracy 58.0%, AUC 0.563 on 2015-2020 draft classes, always populated. "
            "success_probability_enhanced: adds real combine testing (age/forty-time/vertical/"
            "broad-jump) - real, held-out CV AUC 0.605 (a genuine but modest improvement), only "
            "populated for the real subset of rookies with complete combine data (37% of the real "
            "2026 class, disclosed, not backfilled). Draft capital and combine testing are real but "
            "modest predictors of career outcome; this is disclosed, not inflated."
        ),
        "players": scores,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        record_generation("rookie_scores_2026")
    print(f"Scored {len(scores)} real {DRAFT_SEASON} rookies")
    print(f"Wrote {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    score_2026_rookies()
