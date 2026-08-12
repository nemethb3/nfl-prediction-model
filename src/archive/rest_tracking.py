"""Phase 3 Component 3.2: Rest Day Tracking.

Corrects 2 issues found in the spec before building:

1. Real rest-day data already exists: schedules_2015_2025.csv's home_rest/
   away_rest columns are real and complete (checked before building
   anything - zero nulls across all 3028 real games since 2015).
   calculate_rest_days() is a thin, documented utility for API parity, not
   a re-derivation from raw game dates - nflverse already computes this
   correctly and there's no reason to duplicate it.

2. build_rest_adjustment_lookup()'s example values (-0.02/-0.01/0.0/+0.01
   EPA) and apply_rest_adjustment_to_spread()'s "+-0.5 pts per 0.05 EPA"
   are the spec's own illustrative placeholders, not real numbers - both
   derived empirically instead. The rest-EPA correlation is measured as
   each team's real EPA in a given game VS. THAT SAME TEAM'S OWN recent
   trailing baseline (not pooled naively across different teams, which
   would confound team quality with rest) - the same before/after-vs-self
   design already used in injury_model.py. The EPA->points conversion
   reuses Component 2.3's already-fitted real coefficient (1.065 pts per
   unit EPA/play edge) rather than re-deriving an equivalent real-world
   quantity from scratch - a team-level EPA/play edge translates to points
   via the same real, measured relationship regardless of whether the edge
   came from a matchup advantage or a rest advantage.
"""

import os

import numpy as np
import pandas as pd

from constants import MATCHUP_FITTED_COEFFICIENT

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

REST_BUCKETS = [(0, 4, "short (<=4 days)"), (5, 6, "slightly-short (5-6 days)"),
                (7, 7, "standard (7 days)"), (8, 9, "slightly-extra (8-9 days)"), (10, 30, "extra (10+ days)")]


def calculate_rest_days(game_date, previous_game_date):
    """Thin utility for API parity - NOT used in the main pipeline, since
    real home_rest/away_rest already exist (see module docstring #1)."""
    return (pd.Timestamp(game_date) - pd.Timestamp(previous_game_date)).days


def _rest_bucket(days):
    for low, high, label in REST_BUCKETS:
        if low <= days <= high:
            return label
    return "extra (10+ days)"


def extract_game_rest_data_historical(seasons=range(2015, 2026)):
    """Real rest days (from schedules' own home_rest/away_rest) + real
    point differential, per game."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))
    games = schedules[(schedules["season"].isin(list(seasons))) & (schedules["game_type"] == "REG")].copy()
    games["point_diff"] = games["home_score"] - games["away_score"]
    return games[["game_id", "season", "week", "weekday", "home_team", "away_team",
                   "home_rest", "away_rest", "point_diff"]]


def correlate_rest_to_performance(rest_df, seasons=range(2015, 2025)):
    """Real per-team-game EPA vs. that SAME team's own trailing (prior 3
    real games) baseline - isolates the rest effect from team-quality
    confounding (see module docstring #2)."""
    from injury_model import _team_offense_epa_by_week
    team_week_epa = _team_offense_epa_by_week(seasons)

    long_rows = []
    for _, g in rest_df[rest_df["season"].isin(list(seasons))].iterrows():
        for team, rest_days in [(g["home_team"], g["home_rest"]), (g["away_team"], g["away_rest"])]:
            long_rows.append({"team": team, "season": g["season"], "week": g["week"], "rest_days": rest_days})
    long_df = pd.DataFrame(long_rows)

    rows = []
    for _, row in long_df.iterrows():
        team, season, week = row["team"], row["season"], row["week"]
        hist = team_week_epa[(team_week_epa["team"] == team) & (team_week_epa["season"] == season)]
        this_week = hist[hist["week"] == week]["off_epa"]
        prior = hist[(hist["week"] < week) & (hist["week"] >= week - 3)]["off_epa"]
        if len(this_week) == 0 or len(prior) == 0:
            continue
        rows.append({"team": team, "season": season, "week": week, "rest_days": row["rest_days"],
                     "epa_this_game": float(this_week.iloc[0]), "epa_trailing_baseline": float(prior.mean()),
                     "epa_delta": float(this_week.iloc[0] - prior.mean())})
    return pd.DataFrame(rows)


def build_rest_adjustment_lookup(seasons=range(2015, 2025)):
    rest_df = extract_game_rest_data_historical(seasons)
    events = correlate_rest_to_performance(rest_df, seasons)
    events["bucket"] = events["rest_days"].apply(_rest_bucket)

    lookup = events.groupby("bucket")["epa_delta"].agg(avg_epa_delta="mean", std_epa_delta="std", n="count").reset_index()
    bucket_order = [b[2] for b in REST_BUCKETS]
    lookup["bucket"] = pd.Categorical(lookup["bucket"], categories=bucket_order, ordered=True)
    lookup = lookup.sort_values("bucket").reset_index(drop=True)

    print("\n[build_rest_adjustment_lookup] real historical (2015-2024) rest -> EPA delta (vs. own trailing baseline):")
    print(lookup.to_string(index=False))
    lookup.to_csv(os.path.join(PROCESSED_DIR, "rest_adjustment_lookup.csv"), index=False, encoding="utf-8")
    return lookup


def get_rest_adjustment_for_game(rest_days, adjustment_lookup):
    bucket = _rest_bucket(rest_days)
    row = adjustment_lookup[adjustment_lookup["bucket"] == bucket]
    return float(row["avg_epa_delta"].iloc[0]) if len(row) else 0.0


def apply_rest_adjustment_to_spread(elo_spread, home_rest_days, away_rest_days, adjustment_lookup,
                                     coefficient=MATCHUP_FITTED_COEFFICIENT):
    home_adj = get_rest_adjustment_for_game(home_rest_days, adjustment_lookup)
    away_adj = get_rest_adjustment_for_game(away_rest_days, adjustment_lookup)
    return elo_spread + coefficient * (home_adj - away_adj)


def validate_rest_adjustments(season=2025):
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads

    lookup = build_rest_adjustment_lookup()
    fitted_model = fit_probability_to_spread_conversion()
    elo_preds = generate_elo_game_spreads(season, fitted_model)
    rest_data = extract_game_rest_data_historical([season])

    merged = elo_preds.merge(rest_data[["game_id", "weekday", "home_rest", "away_rest", "point_diff"]],
                              on="game_id", how="inner")
    merged["adjusted_spread"] = merged.apply(
        lambda r: apply_rest_adjustment_to_spread(r["predicted_spread"], r["home_rest"], r["away_rest"], lookup), axis=1)

    without_corr = merged["predicted_spread"].corr(merged["point_diff"])
    without_mae = float(np.mean(np.abs(merged["predicted_spread"] - merged["point_diff"])))
    with_corr = merged["adjusted_spread"].corr(merged["point_diff"])
    with_mae = float(np.mean(np.abs(merged["adjusted_spread"] - merged["point_diff"])))

    thursday = merged[merged["weekday"] == "Thursday"]
    thu_without_mae = float(np.mean(np.abs(thursday["predicted_spread"] - thursday["point_diff"]))) if len(thursday) else np.nan
    thu_with_mae = float(np.mean(np.abs(thursday["adjusted_spread"] - thursday["point_diff"]))) if len(thursday) else np.nan

    print(f"\n{'=' * 60}\nREST ADJUSTMENT VALIDATION (real {season})\n{'=' * 60}")
    print(f"Without rest adjustment: corr={without_corr:+.3f} MAE={without_mae:.2f}")
    print(f"With rest adjustment   : corr={with_corr:+.3f} MAE={with_mae:.2f}")
    print(f"Delta: corr {with_corr - without_corr:+.3f} | MAE {with_mae - without_mae:+.2f}")
    print(f"\nThursday games only (n={len(thursday)}): without MAE={thu_without_mae:.2f}, with MAE={thu_with_mae:.2f}")
    if len(thursday) < 20:
        print(f"CAUTION: n={len(thursday)} real Thursday games in {season} - too small to draw a confident "
              f"conclusion from this subset alone, reported for transparency only.")
    print("=" * 60)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    merged.to_csv(os.path.join(DIAGNOSTIC_DIR, f"rest_validation_{season}.csv"), index=False, encoding="utf-8")
    return {"without_corr": without_corr, "without_mae": without_mae, "with_corr": with_corr, "with_mae": with_mae,
            "thu_without_mae": thu_without_mae, "thu_with_mae": thu_with_mae, "n_thursday": len(thursday), "lookup": lookup}


def generate_rest_report(season=2025):
    results = validate_rest_adjustments(season)
    lookup = results["lookup"]

    lines = ["=" * 60, f"Rest Day Analysis (real {season})", "=" * 60]
    lines.append("\nReal Rest Impact on EPA/play (vs. each team's own trailing baseline, 2015-2024):")
    for _, r in lookup.iterrows():
        lines.append(f"  {r['bucket']}: {r['avg_epa_delta']:+.4f} EPA/play (n={int(r['n'])})")

    lines.append(f"\nBacktest Accuracy (real {season}):")
    lines.append(f"  Without rest adjustment: corr={results['without_corr']:+.3f} MAE={results['without_mae']:.2f}")
    lines.append(f"  With rest adjustment: corr={results['with_corr']:+.3f} MAE={results['with_mae']:.2f}")
    lines.append(f"  Thursday games only (n={results['n_thursday']}): "
                  f"MAE {results['thu_without_mae']:.2f} -> {results['thu_with_mae']:.2f}")
    if results["n_thursday"] < 20:
        lines.append(f"  CAUTION: n={results['n_thursday']} is too small to trust the Thursday-specific number alone.")

    delta_mae = results["with_mae"] - results["without_mae"]
    if abs(delta_mae) < 0.05:
        rec = "Marginal / within noise either way - low cost to include, but not a confident improvement."
    elif delta_mae < 0:
        rec = "Real, small improvement - worth including given near-zero implementation cost."
    else:
        rec = "Made accuracy worse in this real backtest - do not adopt as specified."
    lines.append(f"\nRecommendation: {rec}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(DIAGNOSTIC_DIR, "rest_tracking_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    generate_rest_report()
