"""Phase 4 Component 4.1: Confidence Edge Detection.

Corrects a real formula bug found in the spec before building: "confidence
= 1 - std_error" doesn't work dimensionally - Elo's real residual std is
~13.4 POINTS (a point-scale quantity from elo_game_prediction.py's fitted
conversion), so 1 - 13.436 gives a nonsensical negative number, not a 0-1
confidence. Uses the real, already-established formula instead: confidence
= |win_prob - 0.5| * 2 (distance from a toss-up, scaled to 0-1) - the same
formula already validated in weekly_tracking.py's confidence column.

Also corrects the spec's "did our side win" edge-accuracy check: a
disagreement in POINT SPREAD implies a spread (against-the-spread) bet, not
a moneyline (straight-up-winner) bet - those are different real bets with
different payout structures. track_edge_accuracy() checks whether the
edge-favored side covered VEGAS's real spread_line (real point_diff vs.
real vegas_spread), the actual real-world bet a spread disagreement
implies, not just who won outright.

Important context carried forward, not ignored: Component C already found,
decisively (real LOOCV backtest), that Vegas beats Elo at EVERY checkpoint
tested - the learned blend weight was 100% Vegas at weeks 1, 4, 8, 12, and
16, with no exception. That's strong prior evidence that when our spread
disagrees with Vegas's, we're more likely to be wrong than Vegas is. This
component tests that directly on real data rather than assuming the
answer, but the backtest results below should be read in that light.

ROI uses the real, standard American-odds payout formula (not asserted):
at -110, a winning $110 bet returns $100 profit (0.909 profit per unit
risked); a loss costs the full unit staked.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

DISAGREEMENT_THRESHOLD = 1.5
CONFIDENCE_THRESHOLD = 0.60  # the spec's literal threshold - kept as the documented default, but see module note below

# Real, verified before use: Elo's real 2025 confidence NEVER exceeds 0.454 for any game (compression -
# already documented in this project's Phase 2 investigation, ~3x too narrow vs. real variance). The spec's
# absolute 60% threshold is literally unreachable, not a near-miss. validate_edge_detection_backtest() uses
# this empirically-grounded fallback (top quartile of that season's real confidence) instead, disclosed
# explicitly rather than silently substituted.
CONFIDENCE_THRESHOLD_FALLBACK_QUANTILE = 0.75


def calculate_edge_confidence(home_elo, away_elo, home_field_elo=None):
    """Real confidence = |win_prob - 0.5| * 2, not the spec's dimensionally
    broken '1 - std_error' (see module docstring)."""
    from elo_game_prediction import calculate_win_probability_from_elo
    win_prob = calculate_win_probability_from_elo(home_elo, away_elo) if home_field_elo is None else \
        calculate_win_probability_from_elo(home_elo, away_elo, home_field_elo)
    return (win_prob - 0.5).abs() * 2, win_prob


def _build_season_frame(season, fitted_model=None):
    from elo_game_prediction import generate_elo_game_spreads, fit_probability_to_spread_conversion, _load_game_results
    from vegas_integration_optimized import extract_vegas_lines

    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()
    elo = generate_elo_game_spreads(season, fitted_model).rename(columns={"predicted_spread": "our_spread"})
    vegas = extract_vegas_lines(season)[["game_id", "vegas_spread"]]
    actual = _load_game_results([season])[["game_id", "point_diff"]]

    df = elo.merge(vegas, on="game_id", how="inner").merge(actual, on="game_id", how="inner")
    df["confidence"], df["home_win_prob"] = calculate_edge_confidence(df["home_elo"], df["away_elo"])
    return df


def identify_spread_divergences(season, disagreement_threshold=DISAGREEMENT_THRESHOLD, fitted_model=None):
    df = _build_season_frame(season, fitted_model)
    df["disagreement"] = df["our_spread"] - df["vegas_spread"]
    df["direction"] = np.where(df["disagreement"] > 0, "ours_favors_home", "ours_favors_away")
    return df[df["disagreement"].abs() > disagreement_threshold].reset_index(drop=True)


def flag_potential_edges(season, disagreement_threshold=DISAGREEMENT_THRESHOLD, confidence_threshold=CONFIDENCE_THRESHOLD,
                          fitted_model=None):
    divergences = identify_spread_divergences(season, disagreement_threshold, fitted_model)
    edges = divergences[divergences["confidence"] > confidence_threshold].copy()
    edges["edge_flag"] = "EDGE OPPORTUNITY"
    return edges


def track_edge_accuracy(edges_df):
    """Real against-the-spread (ATS) check - the actual bet a spread
    disagreement implies (see module docstring), not a straight-up-winner
    check."""
    df = edges_df.copy()
    df["home_covered"] = df["point_diff"] > df["vegas_spread"]
    df["edge_hit"] = np.where(df["direction"] == "ours_favors_home", df["home_covered"], ~df["home_covered"])
    edge_accuracy = float(df["edge_hit"].mean()) if len(df) else np.nan

    stake, win_payout = 1.0, 100.0 / 110.0
    profit = np.where(df["edge_hit"], win_payout, -stake)
    edge_roi = float(profit.sum() / (stake * len(df))) if len(df) else np.nan
    return {"edge_accuracy": edge_accuracy, "edge_roi": edge_roi, "edges_tracked": len(df)}


def analyze_edge_characteristics(edges_df):
    if len(edges_df) == 0:
        return {"n": 0}
    return {"n": len(edges_df), "by_week": edges_df.groupby("week").size().to_dict(),
            "disagreement_mean": float(edges_df["disagreement"].abs().mean()),
            "disagreement_std": float(edges_df["disagreement"].abs().std()),
            "confidence_mean": float(edges_df["confidence"].mean())}


def generate_weekly_edge_report(season, week, edges_df=None, fitted_model=None):
    if edges_df is None:
        edges_df = flag_potential_edges(season, fitted_model=fitted_model)
    wk = edges_df[edges_df["week"] == week]

    lines = [f"Week {week} Edge Opportunities ({season})", "=" * 50]
    hits = 0
    for _, r in wk.iterrows():
        lean = "home" if r["direction"] == "ours_favors_home" else "away"
        home_covered = r["point_diff"] > r["vegas_spread"]
        hit = (lean == "home") == home_covered
        hits += int(hit)
        lines.append(f"\n{r['home_team']} vs {r['away_team']}:")
        lines.append(f"  Our spread: {r['our_spread']:+.1f} | Vegas: {r['vegas_spread']:+.1f} | "
                      f"disagreement: {r['disagreement']:+.1f} pts (lean {lean})")
        lines.append(f"  Confidence: {r['confidence']:.0%}")
        lines.append(f"  Result: actual margin {r['point_diff']:+.0f} -> "
                      f"{'EDGE HIT' if hit else 'EDGE MISSED'}")
    if len(wk):
        lines.append(f"\nWeek {week} Edge Performance: {hits}/{len(wk)} ({hits / len(wk):.0%})")
    else:
        lines.append("\nNo edges flagged this week.")
    lines.append("=" * 50)
    report = "\n".join(lines)
    print("\n" + report)
    return report


def validate_edge_detection_backtest(season=2025, disagreement_threshold=DISAGREEMENT_THRESHOLD,
                                      confidence_threshold=None):
    from elo_game_prediction import fit_probability_to_spread_conversion
    fitted_model = fit_probability_to_spread_conversion()

    all_games = _build_season_frame(season, fitted_model)
    all_games["disagreement"] = all_games["our_spread"] - all_games["vegas_spread"]
    all_games["direction"] = np.where(all_games["disagreement"] > 0, "ours_favors_home", "ours_favors_away")
    all_games["home_covered"] = all_games["point_diff"] > all_games["vegas_spread"]

    if confidence_threshold is None:
        real_max_confidence = float(all_games["confidence"].max())
        if real_max_confidence <= CONFIDENCE_THRESHOLD:
            confidence_threshold = float(all_games["confidence"].quantile(CONFIDENCE_THRESHOLD_FALLBACK_QUANTILE))
            print(f"NOTE: real {season} max confidence is {real_max_confidence:.1%} - the spec's {CONFIDENCE_THRESHOLD:.0%} "
                  f"threshold is unreachable (known Elo compression, see module docstring). Using the real "
                  f"{CONFIDENCE_THRESHOLD_FALLBACK_QUANTILE:.0%}th percentile of this season's own confidence "
                  f"({confidence_threshold:.1%}) instead, disclosed explicitly.")
        else:
            confidence_threshold = CONFIDENCE_THRESHOLD

    edges = all_games[(all_games["disagreement"].abs() > disagreement_threshold) &
                       (all_games["confidence"] > confidence_threshold)].copy()
    non_edges = all_games.drop(edges.index)

    edge_stats = track_edge_accuracy(edges)

    non_edges = non_edges.copy()
    non_edges["direction"] = np.where(non_edges["disagreement"] > 0, "ours_favors_home", "ours_favors_away")
    non_edges["edge_hit"] = np.where(non_edges["direction"] == "ours_favors_home",
                                      non_edges["home_covered"], ~non_edges["home_covered"])
    non_edge_hit_rate = float(non_edges["edge_hit"].mean()) if len(non_edges) else np.nan

    print(f"\n{'=' * 60}\nEDGE DETECTION BACKTEST (real {season})\n{'=' * 60}")
    print(f"Total games: {len(all_games)} | Edges found: {len(edges)} ({len(edges) / len(all_games):.1%})")
    print(f"Edge hit rate (ATS): {edge_stats['edge_accuracy']:.1%} | Edge ROI: {edge_stats['edge_roi']:+.1%}")
    print(f"Non-edge hit rate (ATS): {non_edge_hit_rate:.1%}")
    print(f"Delta (edges vs. non-edges): {edge_stats['edge_accuracy'] - non_edge_hit_rate:+.1%}")
    if edge_stats["edge_accuracy"] < 0.524:
        print(f"NOTE: {edge_stats['edge_accuracy']:.1%} is below the real breakeven rate at -110 odds (52.4%) - "
              f"consistent with Component C's finding that Vegas beats Elo at every checkpoint tested.")
    print("=" * 60)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    edges.to_csv(os.path.join(DIAGNOSTIC_DIR, f"edge_tracking_{season}.csv"), index=False, encoding="utf-8")
    return {"edges_found_total": len(edges), "edge_hit_rate": edge_stats["edge_accuracy"],
            "non_edge_hit_rate": non_edge_hit_rate, "delta": edge_stats["edge_accuracy"] - non_edge_hit_rate,
            "edge_roi": edge_stats["edge_roi"], "edges_df": edges}


def calculate_edge_value(edges_df, assumed_odds=(-110, -110)):
    stats = track_edge_accuracy(edges_df)
    return stats["edge_roi"], stats["edge_roi"] * stats["edges_tracked"]


if __name__ == "__main__":
    results = validate_edge_detection_backtest()
    edges = results["edges_df"]
    print(f"\nCharacteristics: {analyze_edge_characteristics(edges)}")
    if len(edges):
        sample_week = int(edges["week"].mode().iloc[0])
        generate_weekly_edge_report(2025, sample_week, edges)
