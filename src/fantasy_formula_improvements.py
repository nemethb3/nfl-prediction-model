"""Phase 3 Component 3.3: QB/TE Fantasy Formula Improvements.

Corrects 3 issues found in the spec before building:

1. sqrt(epa) is a literal math bug - EPA/play is frequently negative (most
   QBs have plenty of negative-EPA plays), and sqrt() of a negative number
   is undefined. Uses sign(epa)*sqrt(abs(epa)) instead - the real, sign-
   preserving way to implement a sub-linear EPA transform.

2. Formula B's literal definition computes fantasy points from THAT SAME
   PERIOD's already-realized stats (passes, TD, INT) - the identical
   leakage class already caught and fixed in Component 1.2 (RB). Every
   formula here uses real TRAILING (through week N-1) data instead, the
   same convention (week 1 falls back to real 2024 per-game rates).

3. Already have strong real evidence (Fantasy Direction Validation task,
   this session) that volume ALONE beats the current combined EPA x volume
   formula for BOTH positions: QB EPA-alone +0.417 / volume-alone +0.506 /
   combined (current) +0.435; TE EPA-alone -0.109 / volume-alone +0.644 /
   combined (current) +0.436 - the same pattern Component 1.2 confirmed
   for RB. Added volume-only as a 5th candidate rather than testing only
   the spec's 4 options and omitting the one already suspected to win.

Formula C ("2024 EPA only") is likely redundant with the current
production baseline (already prior-season(2024)-based) - implemented as a
static (never updated within 2025) real 2024 EPA/play rate combined with
the SAME real trailing volume as the other formulas, isolating whether
stale-but-full-season EPA vs. fresh-but-partial-season trailing EPA
matters specifically, while keeping volume treatment consistent across
formulas for a fair comparison.

Correlation is the primary comparison metric throughout (leak-free by
construction - no fitted scale needed to compare rankings). MAE is
reported from an in-sample linear fit (disclosed, same limitation as the
original Fantasy Direction Validation task - only one season of real
preseason-style trailing data exists to fit against).

Bug found and fixed while running (not just noted): a real division-by-
zero produced -inf for at least one 2024 QB's static EPA/play rate (a real
player with recorded passing_epa but zero real attempts - a rare box-score
edge case, e.g. a trick-play pass briefly attributed to a non-QB). That
single -inf silently poisoned every correlation computation it touched
(QB Formula C and D both showed NaN correlation despite real, nonzero
sample sizes) - caught by inspecting the NaN results rather than reporting
them as-is, fixed by filtering to finite values before use.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FANTASY_DIR = os.path.join(PROJECT_ROOT, "data", "fantasy")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

PPR_YD, PPR_RECEPTION, PPR_TD = 0.1, 1.0, 6.0
PPR_PASS_YD_PER_PT, PPR_PASS_TD, PPR_INT = 25.0, 4.0, -2.0

QB_VOLUME_COLS = ["attempts", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards", "rushing_tds"]
TE_VOLUME_COLS = ["targets", "receiving_yards", "receptions", "receiving_tds"]
CURRENT_FORMULA_CORR = {"QB": 0.435, "TE": 0.436}  # real, from Fantasy Direction Validation
W_CANDIDATES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _load_position_data(position, seasons=(2024, 2025)):
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    df = pws[(pws["position"] == position) & (pws["season"].isin(seasons)) & (pws["season_type"] == "REG")].copy()
    cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    for c in cols:
        df[c] = df[c].fillna(0)
    return df


def _real_ppr(position, df):
    if position == "QB":
        return (df["passing_yards"] / PPR_PASS_YD_PER_PT + df["passing_tds"] * PPR_PASS_TD
                + df["interceptions"] * PPR_INT + df["rushing_yards"] * PPR_YD + df["rushing_tds"] * PPR_TD)
    return df["receiving_yards"] * PPR_YD + df["receptions"] * PPR_RECEPTION + df["receiving_tds"] * PPR_TD


def _trailing_volume(df, volume_cols):
    """Real trailing (weeks-before-W) mean of per-game volume stats - same
    pattern as Component 1.2's RB _trailing_window, generalized."""
    v25 = df[df["season"] == 2025].sort_values(["player_id", "week"])
    v24 = df[df["season"] == 2024]
    prior_rate = v24.groupby("player_id")[volume_cols].mean()

    rows = []
    for pid, g in v25.groupby("player_id"):
        g = g.sort_values("week").reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            prior = g.iloc[:i]
            if len(prior) == 0:
                if pid not in prior_rate.index:
                    continue
                est = prior_rate.loc[pid]
            else:
                est = prior[volume_cols].mean()
            rows.append({"player_id": pid, "player_name": row["player_display_name"], "week": row["week"],
                         **{c: est[c] for c in volume_cols}})
    return pd.DataFrame(rows)


def _trailing_epa_rate(df, epa_col, plays_col):
    """Real trailing EPA/PLAY (sum/sum ratio, not mean-of-means - the
    correct way to compute a rate) - week 1 falls back to the real static
    2024 rate."""
    v25 = df[df["season"] == 2025].sort_values(["player_id", "week"])
    v24 = df[df["season"] == 2024]
    prior_rate = (v24.groupby("player_id")[epa_col].sum() / v24.groupby("player_id")[plays_col].sum())
    prior_rate = prior_rate[np.isfinite(prior_rate)]  # real division-by-zero edge case found by inspection - see module note

    rows = []
    for pid, g in v25.groupby("player_id"):
        g = g.sort_values("week").reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            prior = g.iloc[:i]
            if len(prior) == 0:
                if pid not in prior_rate.index:
                    continue
                epa_pp = prior_rate.loc[pid]
            else:
                total_plays = prior[plays_col].sum()
                epa_pp = prior[epa_col].sum() / total_plays if total_plays > 0 else np.nan
            rows.append({"player_id": pid, "week": row["week"], "epa_per_play": epa_pp})
    out = pd.DataFrame(rows).dropna(subset=["epa_per_play"])
    return out[np.isfinite(out["epa_per_play"])]


def _actual_points(position):
    df = pd.read_csv(os.path.join(FANTASY_DIR, "actual_fantasy_points_2025_by_week.csv"))
    return df[df["position"] == position][["player_id", "week", "actual_fantasy_pts"]]


def _score(merged, score_col="raw_score"):
    merged = merged.dropna(subset=[score_col, "actual_fantasy_pts"])
    corr = merged[score_col].corr(merged["actual_fantasy_pts"])
    if len(merged) > 2 and merged[score_col].std() > 0:
        slope, intercept = np.polyfit(merged[score_col], merged["actual_fantasy_pts"], 1)
        pred = slope * merged[score_col] + intercept
        mae = float(np.mean(np.abs(pred - merged["actual_fantasy_pts"])))
    else:
        mae = np.nan
    return corr, mae, len(merged)


def _build_common(position):
    df = _load_position_data(position)
    volume_cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    trailing_vol = _trailing_volume(df, volume_cols)

    epa_col, plays_col = ("passing_epa", "attempts") if position == "QB" else ("receiving_epa", "targets")
    trailing_epa = _trailing_epa_rate(df, epa_col, plays_col)

    primary_vol_col = "attempts" if position == "QB" else "targets"
    actual = _actual_points(position)
    return df, trailing_vol, trailing_epa, primary_vol_col, actual


def test_volume_only(position):
    _, trailing_vol, _, _, actual = _build_common(position)
    trailing_vol = trailing_vol.copy()
    trailing_vol["raw_score"] = _real_ppr(position, trailing_vol)
    merged = trailing_vol.merge(actual, on=["player_id", "week"], how="inner")
    return _score(merged)


def test_formula_a_recalibrated_epa(position, w_candidates=W_CANDIDATES):
    """LOOCV (leave-one-week-out) over exponent w: raw_score = sign(epa)*
    |epa|^w * primary_volume - avoids fitting w on the same data it's
    scored against (established LOOCV pattern this session)."""
    _, _, trailing_epa, primary_vol_col, actual = _build_common(position)
    df = _load_position_data(position)
    volume_cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    trailing_vol = _trailing_volume(df, volume_cols)

    merged = trailing_epa.merge(trailing_vol[["player_id", "week", primary_vol_col]], on=["player_id", "week"]).merge(
        actual, on=["player_id", "week"], how="inner")
    weeks = sorted(merged["week"].unique())

    chosen_w, held_out_scores = [], []
    for W in weeks:
        train = merged[merged["week"] != W]
        best_w, best_corr = None, -np.inf
        for w in w_candidates:
            raw = np.sign(train["epa_per_play"]) * np.abs(train["epa_per_play"]) ** w * train[primary_vol_col]
            c = raw.corr(train["actual_fantasy_pts"])
            if pd.notna(c) and c > best_corr:
                best_corr, best_w = c, w
        chosen_w.append(best_w)
        test = merged[merged["week"] == W]
        raw_test = np.sign(test["epa_per_play"]) * np.abs(test["epa_per_play"]) ** best_w * test[primary_vol_col]
        held_out_scores.append(pd.DataFrame({"raw_score": raw_test, "actual_fantasy_pts": test["actual_fantasy_pts"]}))

    all_held_out = pd.concat(held_out_scores, ignore_index=True)
    corr, mae, n = _score(all_held_out)
    return corr, mae, n, float(np.mean(chosen_w))


def test_formula_b_epa_transform(position):
    """sign(epa)*sqrt(abs(epa)) - fixes the spec's sqrt(negative) bug."""
    _, _, trailing_epa, primary_vol_col, actual = _build_common(position)
    df = _load_position_data(position)
    volume_cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    trailing_vol = _trailing_volume(df, volume_cols)

    merged = trailing_epa.merge(trailing_vol[["player_id", "week", primary_vol_col]], on=["player_id", "week"]).merge(
        actual, on=["player_id", "week"], how="inner")
    merged["raw_score"] = np.sign(merged["epa_per_play"]) * np.sqrt(np.abs(merged["epa_per_play"])) * merged[primary_vol_col]
    return _score(merged)


def test_formula_c_recent_season_only(position):
    """Static real 2024 EPA/play (never updated within 2025) x the same
    real trailing volume other formulas use - isolates stale-full-season
    vs. fresh-partial-season EPA specifically."""
    df = _load_position_data(position)
    epa_col, plays_col = ("passing_epa", "attempts") if position == "QB" else ("receiving_epa", "targets")
    v24 = df[df["season"] == 2024]
    static_2024 = (v24.groupby("player_id")[epa_col].sum() / v24.groupby("player_id")[plays_col].sum()).rename("epa_per_play").reset_index()
    static_2024 = static_2024[np.isfinite(static_2024["epa_per_play"])]  # real div-by-zero edge case, see module note

    volume_cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    primary_vol_col = "attempts" if position == "QB" else "targets"
    trailing_vol = _trailing_volume(df, volume_cols)
    actual = _actual_points(position)

    merged = trailing_vol.merge(static_2024, on="player_id", how="inner").merge(actual, on=["player_id", "week"], how="inner")
    merged["raw_score"] = merged["epa_per_play"] * merged[primary_vol_col]
    return _score(merged)


def test_formula_d_position_normalized(position):
    """normalized_epa = trailing epa_per_play - real overall league-average
    trailing epa_per_play (a single real constant, disclosed simplification
    - not week-varying, given time budget)."""
    _, _, trailing_epa, primary_vol_col, actual = _build_common(position)
    df = _load_position_data(position)
    volume_cols = QB_VOLUME_COLS if position == "QB" else TE_VOLUME_COLS
    trailing_vol = _trailing_volume(df, volume_cols)

    league_avg = float(trailing_epa["epa_per_play"].mean())
    merged = trailing_epa.merge(trailing_vol[["player_id", "week", primary_vol_col]], on=["player_id", "week"]).merge(
        actual, on=["player_id", "week"], how="inner")
    merged["raw_score"] = (merged["epa_per_play"] - league_avg) * merged[primary_vol_col]
    return _score(merged)


def compare_formulas(position):
    results = {
        "current_production": (CURRENT_FORMULA_CORR[position], None, None),
        "volume_only": test_volume_only(position),
        "B_epa_transform": test_formula_b_epa_transform(position),
        "C_2024_epa_only": test_formula_c_recent_season_only(position),
        "D_position_normalized": test_formula_d_position_normalized(position),
    }
    a_result = test_formula_a_recalibrated_epa(position)
    results["A_recalibrated_epa"] = (a_result[0], a_result[1], a_result[2])

    rows = []
    for name, r in results.items():
        rows.append({"formula": name, "correlation": r[0], "mae": r[1], "n": r[2]})
    ranked = pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
    print(f"\n{position} formula ranking:")
    print(ranked.to_string(index=False))
    return ranked, a_result[3] if len(a_result) > 3 else None


def generate_fantasy_formula_report():
    lines = ["=" * 60, "QB/TE Fantasy Formula Improvements (real 2025, leak-free trailing-window)", "=" * 60]
    all_ranked = {}
    for position in ["QB", "TE"]:
        ranked, chosen_w = compare_formulas(position)
        all_ranked[position] = ranked
        lines.append(f"\n{position} (current production: {CURRENT_FORMULA_CORR[position]:+.3f} corr):")
        for _, r in ranked.iterrows():
            marker = " <- WINNER" if r["formula"] == ranked.iloc[0]["formula"] else ""
            mae_str = f"{r['mae']:.2f}" if pd.notna(r["mae"]) else "n/a"
            lines.append(f"  {r['formula']}: corr={r['correlation']:+.3f} MAE={mae_str}{marker}")
        if chosen_w is not None:
            lines.append(f"  (Formula A LOOCV mean chosen exponent w={chosen_w:.2f})")

        winner = ranked.iloc[0]
        delta = winner["correlation"] - CURRENT_FORMULA_CORR[position]
        lines.append(f"  Improvement over current: {delta:+.3f} correlation points")

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    all_ranked["QB"].to_csv(os.path.join(DIAGNOSTIC_DIR, "qb_formula_comparison_2025.csv"), index=False, encoding="utf-8")
    all_ranked["TE"].to_csv(os.path.join(DIAGNOSTIC_DIR, "te_formula_comparison_2025.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "fantasy_formula_improvements_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved data/diagnostic/qb_formula_comparison_2025.csv, te_formula_comparison_2025.csv, "
          f"fantasy_formula_improvements_report.txt")
    return all_ranked


if __name__ == "__main__":
    generate_fantasy_formula_report()
