"""Phase 1 Component 1.2: RB Fantasy Formula Fix.

Corrects 3 issues found in the spec before building:

1. Formula A's literal definition ("carries * 0.1 + rec_yards * 0.1 + 6*tds")
   has two real bugs: it multiplies CARRIES (a count) by 0.1 instead of
   RUSHING YARDS, and omits the reception bonus (1 pt/reception) entirely -
   a large share of a receiving back's real value. Uses the real, already-
   verified PPR formula instead (0.1/rush yd + 0.1/rec yd + 1/reception +
   6/TD - the same convention fantasy_validation.py already spot-checked
   against real box scores).

2. The more important issue: if "volume" in any formula means THAT SAME
   WEEK's own already-realized carries/yards, comparing the resulting
   "projection" to that same week's real actual fantasy points isn't a
   predictive test - both are functions of the identical already-known box
   score, so they'd correlate almost perfectly by construction. Same
   leakage class already caught repeatedly this project (same-season EPA
   diffs, in-sample weight fits). Fixed: every formula predicts week W
   using only real data from weeks BEFORE W (an expanding trailing window
   within real 2025; week 1, with no real prior-2025 data, falls back to
   real prior-season 2024 per-game rates - the same leak-free convention
   the original Fantasy Validation task used to get the +0.667 figure this
   component is chasing).

3. "LOOCV (leave out each week, fit on others)" as literally read would let
   FUTURE weeks inform a prediction about an EARLIER week - a lookahead
   leak for a fundamentally time-ordered forecasting problem. Every other
   genuinely-predictive component this session (weekly_predictions.py,
   dynamic_tracking.py, weekly_recalibration.py) strictly uses only past
   information; the trailing-window design here does the same instead of
   literal both-directions leave-one-out.

Formula C ("injury-adjusted... if backup RB got significant carries") needed
a real, computable definition, not a hand-wave: operationalized as - if a
player's own team had another RB take >30% of the team's real total RB
carries in a trailing week, shrink this player's own-history-based
projection by a flat 30% for that week's prediction. This 0.7 shrink factor
is an ASSUMED constant, not empirically fit (out of scope for this pass'
time budget) - disclosed, not hidden, same as this project's other assumed
constants (e.g. elo_model.py's 1/3 season-to-season regression).
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FANTASY_DIR = os.path.join(PROJECT_ROOT, "data", "fantasy")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

PPR_RUSH_YD, PPR_REC_YD, PPR_RECEPTION, PPR_TD = 0.1, 0.1, 1.0, 6.0
COMMITTEE_SHARE_THRESHOLD = 0.30
COMMITTEE_SHRINK_FACTOR = 0.7  # assumed, not fit - see module docstring


def load_rb_volume_data_2025():
    """Real weekly RB volume, 2024 (for prior-season fallback) + 2025 REG -
    from the same real player_weekly_stats.csv fantasy_validation.py uses,
    not a rebuild."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    rb = pws[(pws["position"] == "RB") & (pws["season"].isin([2024, 2025])) & (pws["season_type"] == "REG")].copy()
    rb["total_td"] = rb["rushing_tds"].fillna(0) + rb["receiving_tds"].fillna(0)
    for c in ["carries", "rushing_yards", "receptions", "receiving_yards"]:
        rb[c] = rb[c].fillna(0)
    cols = ["player_id", "player_display_name", "recent_team", "season", "week",
            "carries", "rushing_yards", "receptions", "receiving_yards", "total_td"]
    return rb[cols].rename(columns={"player_display_name": "player_name", "recent_team": "team"})


def load_rb_actual_fantasy_2025():
    df = pd.read_csv(os.path.join(FANTASY_DIR, "actual_fantasy_points_2025_by_week.csv"))
    return df[df["position"] == "RB"][["player_id", "week", "actual_fantasy_pts"]]


def _real_ppr(rush_yd, rec_yd, receptions, tds):
    return rush_yd * PPR_RUSH_YD + rec_yd * PPR_REC_YD + receptions * PPR_RECEPTION + tds * PPR_TD


VOLUME_COLS = ["carries", "rushing_yards", "receptions", "receiving_yards", "total_td"]


def _trailing_window(volume_df, weight_recent=False):
    """For every real (player, week) in 2025: the trailing (weeks-strictly-
    before-W) mean of each volume stat. Week 1 (no real prior-2025 data)
    falls back to the real 2024 per-game rate; rookies with neither are
    excluded (disclosed via sample size, not silently dropped)."""
    v25 = volume_df[volume_df["season"] == 2025].sort_values(["player_id", "week"])
    v24 = volume_df[volume_df["season"] == 2024]
    prior_season_rate = v24.groupby("player_id")[VOLUME_COLS].mean()

    rows = []
    for pid, g in v25.groupby("player_id"):
        g = g.sort_values("week").reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            prior = g.iloc[:i]
            if len(prior) == 0:
                if pid not in prior_season_rate.index:
                    continue
                est = prior_season_rate.loc[pid]
            elif weight_recent:
                w = np.where(prior["week"].to_numpy() >= (row["week"] - 4), 2.0, 1.0)
                est = pd.Series({c: np.average(prior[c], weights=w) for c in VOLUME_COLS})
            else:
                est = prior[VOLUME_COLS].mean()
            rows.append({"player_id": pid, "player_name": row["player_name"], "team": row["team"],
                         "week": row["week"], **{c: est[c] for c in VOLUME_COLS}})
    return pd.DataFrame(rows)


def _score_and_merge(trailing_df, actual_df):
    df = trailing_df.copy()
    df["projected_fantasy_pts"] = _real_ppr(df["rushing_yards"], df["receiving_yards"], df["receptions"], df["total_td"])
    merged = df.merge(actual_df, on=["player_id", "week"], how="inner")
    corr = merged["projected_fantasy_pts"].corr(merged["actual_fantasy_pts"])
    mae = float(np.mean(np.abs(merged["projected_fantasy_pts"] - merged["actual_fantasy_pts"])))
    return corr, mae, merged


def test_formula_a_volume_only(volume_df, actual_df):
    trailing = _trailing_window(volume_df, weight_recent=False)
    return _score_and_merge(trailing, actual_df)


def test_formula_b_volume_with_recency(volume_df, actual_df):
    trailing = _trailing_window(volume_df, weight_recent=True)
    return _score_and_merge(trailing, actual_df)


def test_formula_c_injury_adjusted(volume_df, actual_df):
    """Real, computable committee-signal shrink (see module docstring):
    if a teammate took >30% of the team's real RB carries in any trailing
    week, shrink this player's trailing-window projection by 30% for the
    week being predicted."""
    trailing = _trailing_window(volume_df, weight_recent=False)
    v25 = volume_df[volume_df["season"] == 2025]

    adjustments = []
    for _, row in trailing.iterrows():
        team_prior = v25[(v25["team"] == row["team"]) & (v25["week"] < row["week"])]
        team_week_totals = team_prior.groupby("week")["carries"].sum()
        teammate_prior = team_prior[team_prior["player_id"] != row["player_id"]]
        signal = False
        for wk, team_total in team_week_totals.items():
            if team_total <= 0:
                continue
            teammate_carries = teammate_prior[teammate_prior["week"] == wk]["carries"].sum()
            if (teammate_carries / team_total) > COMMITTEE_SHARE_THRESHOLD:
                signal = True
                break
        adjustments.append(COMMITTEE_SHRINK_FACTOR if signal else 1.0)

    trailing = trailing.copy()
    trailing["committee_adjustment"] = adjustments
    for c in VOLUME_COLS:
        trailing[c] = trailing[c] * trailing["committee_adjustment"]
    return _score_and_merge(trailing, actual_df)


def test_formula_d_recent_season(volume_df, actual_df, prior_season_volume=None):
    """Pure real 2024 per-game rate, applied uniformly to every 2025 week
    (never updated within-season, unlike A/B/C) - excludes rookies with no
    real 2024 data (disclosed via sample size)."""
    if prior_season_volume is None:
        v24 = volume_df[volume_df["season"] == 2024]
        prior_season_volume = v24.groupby("player_id")[VOLUME_COLS].mean().reset_index()

    v25_weeks = volume_df[volume_df["season"] == 2025][["player_id", "player_name", "team", "week"]].drop_duplicates()
    proj = v25_weeks.merge(prior_season_volume, on="player_id", how="inner")
    return _score_and_merge(proj, actual_df)


def compare_formulas(results_a, results_b, results_c, results_d):
    rows = []
    for name, (corr, mae, merged) in [("A_volume_only", results_a), ("B_recency_weighted", results_b),
                                        ("C_injury_adjusted", results_c), ("D_recent_season", results_d)]:
        rows.append({"formula": name, "correlation": corr, "mae": mae, "n": len(merged)})
    ranked = pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
    print("\nFormula ranking (by correlation):")
    print(ranked.to_string(index=False))
    return ranked


def generate_rb_formula_report():
    volume_df = load_rb_volume_data_2025()
    actual_df = load_rb_actual_fantasy_2025()

    results_a = test_formula_a_volume_only(volume_df, actual_df)
    results_b = test_formula_b_volume_with_recency(volume_df, actual_df)
    results_c = test_formula_c_injury_adjusted(volume_df, actual_df)
    results_d = test_formula_d_recent_season(volume_df, actual_df)

    labels = {"A": ("Volume only (trailing real weeks, leak-free)", results_a),
              "B": ("Volume + recency (last 4 real weeks weighted 2x)", results_b),
              "C": ("Injury/committee-adjusted (real teammate-share signal)", results_c),
              "D": ("Prior-season (2024) rate only, never updated in-season", results_d)}

    lines = ["=" * 60, "RB Fantasy Formula Comparison (2025, leak-free trailing-window)", "=" * 60]
    for key, (desc, (corr, mae, merged)) in labels.items():
        lines.append(f"\nFormula {key} ({desc}):")
        lines.append(f"  Correlation: {corr:+.3f}")
        lines.append(f"  MAE: {mae:.2f} pts")
        lines.append(f"  n: {len(merged)} player-weeks")

    ranked = compare_formulas(results_a, results_b, results_c, results_d)
    winner = ranked.iloc[0]
    lines.append(f"\nWinner: Formula {winner['formula']} (corr={winner['correlation']:+.3f}, MAE={winner['mae']:.2f})")

    epa_volume_corr = -0.504  # original spec formula (EPA x volume), from Fantasy Direction Validation
    lines.append(f"\nOriginal EPA x volume formula (Fantasy Direction Validation task): corr={epa_volume_corr:+.3f}")
    lines.append(f"Improvement: {winner['correlation'] - epa_volume_corr:+.3f} correlation points")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    ranked.to_csv(os.path.join(DIAGNOSTIC_DIR, "rb_formula_comparison_2025.csv"), index=False, encoding="utf-8")
    with open(os.path.join(DIAGNOSTIC_DIR, "rb_formula_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved data/diagnostic/rb_formula_comparison_2025.csv, rb_formula_report.txt")

    return ranked, labels


def project_rb_fantasy_points(volume_df=None):
    """The winning formula (B: trailing real volume, last 4 weeks weighted
    2x), exported for fantasy_validation.py to call directly. Real 2025
    result: corr=+0.651 vs. Formula A's +0.649 - a noise-level difference,
    not a meaningful win; A (plain trailing average, no recency weighting)
    would be an equally defensible, simpler choice. Both dramatically beat
    the original EPA x volume formula's -0.504."""
    if volume_df is None:
        volume_df = load_rb_volume_data_2025()
    trailing = _trailing_window(volume_df, weight_recent=True)
    trailing = trailing.copy()
    trailing["projected_fantasy_pts"] = _real_ppr(
        trailing["rushing_yards"], trailing["receiving_yards"], trailing["receptions"], trailing["total_td"])
    return trailing


if __name__ == "__main__":
    generate_rb_formula_report()
