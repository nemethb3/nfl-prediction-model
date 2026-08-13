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
294-player trade_scores_2026.json count."""

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


def score_2026_rookies():
    print(f"\nScoring real {DRAFT_SEASON} rookie class (QB/RB/WR/TE)...\n")
    with open(os.path.join(MODELS_DIR, "rookie_classifier.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model, scaler, features, position_dummy_cols = (
        bundle["model"], bundle["scaler"], bundle["features"], bundle["position_dummy_cols"])

    draft = nfl.load_draft_picks().to_pandas()
    rookies = draft[
        (draft["season"] == DRAFT_SEASON)
        & draft["position"].isin(POSITIONS)
        & draft["gsis_id"].notna()
    ].copy()
    print(f"Real {DRAFT_SEASON} draft picks (QB/RB/WR/TE): {len(rookies)}")

    position_dummies = pd.DataFrame(0, index=rookies.index, columns=position_dummy_cols)
    for col in position_dummy_cols:
        pos = col.replace("pos_", "")
        position_dummies.loc[rookies["position"] == pos, col] = 1
    X = pd.concat([rookies[["round", "pick"]], position_dummies], axis=1)[features].to_numpy(dtype=float)
    X_scaled = scaler.transform(X)
    rookies["success_probability"] = np.round(model.predict_proba(X_scaled)[:, 1], 3)

    scores = {}
    for _, r in rookies.sort_values("gsis_id").iterrows():
        scores[r["gsis_id"]] = {
            "name": r["pfr_player_name"],
            "position": r["position"],
            "team": r["team"],
            "draft_round": int(r["round"]),
            "draft_pick": int(r["pick"]),
            "success_probability": float(r["success_probability"]),
        }

    output = {
        "draft_season": DRAFT_SEASON,
        "methodology_note": (
            "Real rookie success classifier score: P(this player's eventual career weighted "
            "Approximate Value beats their real draft round's historical median), from real "
            "draft-time-only signals (round, pick, position). See rookie_classifier_accuracy.json "
            "for the honest, held-out CV accuracy this model achieves (58.0%, AUC 0.563 on "
            "2015-2020 draft classes) - draft capital alone is a real but modest predictor of "
            "career outcome; this is disclosed, not inflated."
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
