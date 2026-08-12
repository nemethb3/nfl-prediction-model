"""Phase 1 Component 1.1: Weekly Tracking Infrastructure.

Corrects 2 issues found in the spec before building:

1. predictions_df's expected columns (elo_spread, confidence, predicted_
   win_prob, elo_home, elo_away) don't come directly from any existing
   function - elo_game_prediction.generate_elo_game_spreads() returns
   home_elo/away_elo/predicted_spread/ci_low_90/ci_high_90 but no win
   probability or confidence column at all. build_weekly_predictions_df()
   below derives both from the real Elo ratings it already returns
   (predicted_win_prob via the same calculate_win_probability_from_elo()
   Component A already validated; confidence = |win_prob - 0.5| * 2, the
   same style of derived confidence vegas_comparison.py's identify_edges()
   already uses) rather than treating them as separate inputs to source.

2. "Calibration: of N games with X% predicted win prob, how many actually
   won? (90% CI coverage?)" conflates two different, real metrics: win-
   probability reliability (are 70%-confidence picks right ~70% of the
   time?) and spread CI coverage (does the actual point diff fall inside
   the 90% band?). Computed and reported as two separate, clearly-labeled
   numbers, not one ambiguous "calibration" figure. Win-probability
   reliability is reported as mean absolute calibration error (a real
   binned reliability curve needs far more than the ~13-16 games/week this
   project has - disclosed, not glossed over).

Also: "vs Vegas" is computed from real spread_line data for THAT SPECIFIC
WEEK's games, not by reprinting the whole-season constants from
constants.py (VEGAS_BASELINE_GAME_CORR_2025 etc.) as if they applied to
every individual week - those are real, correct season-aggregate numbers,
not a substitute for a real per-week measurement.

Uses Python's built-in sqlite3 (stdlib) rather than SQLAlchemy, despite
SQLAlchemy being listed in requirements.txt (unused so far, per AUDIT_2026-
07-27.md) - this is a single local file-based DB with no server component,
exactly what sqlite3 alone is for; reaching for an ORM here would be
unneeded abstraction for one file.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "tracking.db")

PREDICTIONS_COLS = ["game_id", "season", "week", "home_team", "away_team", "elo_spread",
                     "elo_home", "elo_away", "predicted_win_prob", "confidence",
                     "ci_low_90", "ci_high_90", "timestamp"]
RESULTS_COLS = ["game_id", "season", "week", "actual_point_diff", "home_team_won",
                 "home_score", "away_score"]


def create_tracking_database(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        game_id TEXT PRIMARY KEY, season INTEGER, week INTEGER,
        home_team TEXT, away_team TEXT,
        elo_spread REAL, elo_home REAL, elo_away REAL,
        predicted_win_prob REAL, confidence REAL,
        ci_low_90 REAL, ci_high_90 REAL, timestamp TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS results (
        game_id TEXT PRIMARY KEY, season INTEGER, week INTEGER,
        actual_point_diff REAL, home_team_won REAL,
        home_score REAL, away_score REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (
        season INTEGER, week INTEGER, metric_type TEXT, value REAL,
        PRIMARY KEY (season, week, metric_type))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS cumulative (
        season INTEGER, through_week INTEGER,
        cumulative_corr REAL, cumulative_mae REAL,
        cumulative_win_prob_cal_error REAL, cumulative_ci_coverage REAL,
        n_games INTEGER, PRIMARY KEY (season, through_week))""")
    conn.commit()
    return conn


def build_weekly_predictions_df(season, week_n, fitted_model=None):
    """Real Elo predictions for week_n's games, in the schema save_weekly_
    predictions() expects - derives predicted_win_prob/confidence from the
    real Elo ratings (see module docstring #1) rather than assuming they
    already exist somewhere."""
    from elo_game_prediction import (generate_elo_game_spreads, calculate_win_probability_from_elo,
                                      fit_probability_to_spread_conversion)
    if fitted_model is None:
        fitted_model = fit_probability_to_spread_conversion()

    all_spreads = generate_elo_game_spreads(season, fitted_model)
    wk = all_spreads[all_spreads["week"] == week_n].copy()
    wk["predicted_win_prob"] = calculate_win_probability_from_elo(wk["home_elo"], wk["away_elo"])
    wk["confidence"] = (wk["predicted_win_prob"] - 0.5).abs() * 2
    wk = wk.rename(columns={"predicted_spread": "elo_spread", "home_elo": "elo_home", "away_elo": "elo_away"})
    return wk[["game_id", "home_team", "away_team", "elo_spread", "elo_home", "elo_away",
               "predicted_win_prob", "confidence", "ci_low_90", "ci_high_90"]].reset_index(drop=True)


def build_weekly_results_df(season, week_n):
    from elo_game_prediction import _load_game_results
    games = _load_game_results([season])
    wk = games[games["week"] == week_n].copy()
    wk["actual_point_diff"] = wk["point_diff"]
    wk["home_team_won"] = np.select([wk["point_diff"] > 0, wk["point_diff"] < 0], [1.0, 0.0], default=0.5)
    return wk[["game_id", "actual_point_diff", "home_team_won", "home_score", "away_score"]].reset_index(drop=True)


def save_weekly_predictions(db_conn, season, week, predictions_df):
    """INSERT OR REPLACE on game_id (primary key) means re-running the same
    week overwrites cleanly instead of erroring or duplicating - the
    mechanism satisfying the spec's 'verify no duplicates' requirement."""
    rows = predictions_df.copy()
    rows["season"], rows["week"] = season, week
    rows["timestamp"] = datetime.now().isoformat()
    cur = db_conn.cursor()
    for _, r in rows.iterrows():
        cur.execute(f"INSERT OR REPLACE INTO predictions VALUES ({','.join(['?'] * len(PREDICTIONS_COLS))})",
                     tuple(r[c] for c in PREDICTIONS_COLS))
    db_conn.commit()
    return len(rows)


def log_weekly_results(db_conn, season, week, results_df):
    existing = pd.read_sql("SELECT game_id FROM predictions WHERE season=? AND week=?",
                            db_conn, params=(season, week))
    missing = set(results_df["game_id"]) - set(existing["game_id"])
    if missing:
        print(f"WARNING: {len(missing)} result(s) with no matching logged prediction: {sorted(missing)[:5]}")

    rows = results_df.copy()
    rows["season"], rows["week"] = season, week
    cur = db_conn.cursor()
    for _, r in rows.iterrows():
        cur.execute(f"INSERT OR REPLACE INTO results VALUES ({','.join(['?'] * len(RESULTS_COLS))})",
                     tuple(r[c] for c in RESULTS_COLS))
    db_conn.commit()
    return len(rows)


def _weekly_vegas_accuracy(season, week):
    """Real per-week Vegas accuracy, computed live from that week's real
    spread_line vs. real point differential - NOT the whole-season
    constants.py figures reprinted as if they applied to one week."""
    from vegas_integration_optimized import extract_vegas_lines
    from elo_game_prediction import _load_game_results
    vegas = extract_vegas_lines(season)
    vegas = vegas[vegas["week"] == week]
    actual = _load_game_results([season])
    actual = actual[actual["week"] == week][["game_id", "point_diff"]]
    m = vegas.merge(actual, on="game_id", how="inner").dropna(subset=["vegas_spread"])
    if len(m) < 2:
        return None, None
    return m["vegas_spread"].corr(m["point_diff"]), float(np.mean(np.abs(m["vegas_spread"] - m["point_diff"])))


def compute_weekly_accuracy(db_conn, season, week):
    df = pd.read_sql("""SELECT p.*, r.actual_point_diff, r.home_team_won FROM predictions p
                         JOIN results r ON p.game_id = r.game_id
                         WHERE p.season=? AND p.week=?""", db_conn, params=(season, week))
    if len(df) == 0:
        print(f"No matched predictions+results for {season} week {week}")
        return None

    corr = df["elo_spread"].corr(df["actual_point_diff"])
    mae = float(np.mean(np.abs(df["elo_spread"] - df["actual_point_diff"])))
    ci_coverage = float(((df["actual_point_diff"] >= df["ci_low_90"]) &
                          (df["actual_point_diff"] <= df["ci_high_90"])).mean())
    win_prob_cal_error = float(np.mean(np.abs(df["predicted_win_prob"] - df["home_team_won"])))

    vegas_corr, vegas_mae = _weekly_vegas_accuracy(season, week)

    metrics = {"correlation": corr, "mae": mae, "ci_coverage_90": ci_coverage,
               "win_prob_cal_error": win_prob_cal_error}
    if vegas_corr is not None:
        metrics["vegas_corr"] = vegas_corr
        metrics["vegas_mae"] = vegas_mae

    cur = db_conn.cursor()
    for k, v in metrics.items():
        cur.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)", (season, week, k, float(v)))
    db_conn.commit()
    metrics["n_games"] = len(df)
    return metrics


def compute_cumulative_accuracy(db_conn, season, through_week):
    df = pd.read_sql("""SELECT p.*, r.actual_point_diff, r.home_team_won FROM predictions p
                         JOIN results r ON p.game_id = r.game_id
                         WHERE p.season=? AND p.week<=?""", db_conn, params=(season, through_week))
    if len(df) == 0:
        return None

    corr = df["elo_spread"].corr(df["actual_point_diff"])
    mae = float(np.mean(np.abs(df["elo_spread"] - df["actual_point_diff"])))
    ci_coverage = float(((df["actual_point_diff"] >= df["ci_low_90"]) &
                          (df["actual_point_diff"] <= df["ci_high_90"])).mean())
    win_prob_cal_error = float(np.mean(np.abs(df["predicted_win_prob"] - df["home_team_won"])))

    db_conn.execute("INSERT OR REPLACE INTO cumulative VALUES (?,?,?,?,?,?,?)",
                     (season, through_week, corr, mae, win_prob_cal_error, ci_coverage, len(df)))
    db_conn.commit()
    return {"cumulative_corr": corr, "cumulative_mae": mae, "cumulative_ci_coverage": ci_coverage,
            "cumulative_win_prob_cal_error": win_prob_cal_error, "n_games": len(df)}


def generate_weekly_report(db_conn, season, week):
    weekly = compute_weekly_accuracy(db_conn, season, week)
    cumulative = compute_cumulative_accuracy(db_conn, season, week)
    if weekly is None:
        return f"No data for {season} week {week}"

    lines = [f"Week {week} Report ({season} Season)", "=" * 40]
    lines.append(f"Games: {weekly['n_games']}")
    lines.append("Our Accuracy:")
    lines.append(f"  Correlation: {weekly['correlation']:+.3f}")
    lines.append(f"  MAE: {weekly['mae']:.2f} pts")
    lines.append(f"  CI coverage (90% target): {weekly['ci_coverage_90']:.0%}")
    lines.append(f"  Win-prob calibration error (mean |pred_prob - actual|): {weekly['win_prob_cal_error']:.3f}")

    if "vegas_corr" in weekly:
        lines.append("\nvs Vegas (this week's real lines):")
        lines.append(f"  Vegas corr: {weekly['vegas_corr']:+.3f} | Vegas MAE: {weekly['vegas_mae']:.2f} pts")
        lines.append(f"  Delta (us - Vegas): corr {weekly['correlation'] - weekly['vegas_corr']:+.3f}, "
                      f"MAE {weekly['mae'] - weekly['vegas_mae']:+.2f}")
    else:
        lines.append("\nvs Vegas: no real lines available for this week")

    if cumulative:
        lines.append(f"\nCumulative (Weeks 1-{week}):")
        lines.append(f"  Overall corr: {cumulative['cumulative_corr']:+.3f}")
        lines.append(f"  Overall MAE: {cumulative['cumulative_mae']:.2f} pts")
        lines.append(f"  N games: {cumulative['n_games']}")
    lines.append("=" * 40)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    with open(os.path.join(DIAGNOSTIC_DIR, f"weekly_accuracy_report_{week}.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def export_weekly_csv(db_conn, season, week, output_dir=PREDICTIONS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    preds = pd.read_sql("SELECT * FROM predictions WHERE season=? AND week=?", db_conn, params=(season, week))
    results = pd.read_sql("SELECT * FROM results WHERE season=? AND week=?", db_conn, params=(season, week))
    pred_path = os.path.join(output_dir, f"week_{week}_predictions.csv")
    result_path = os.path.join(output_dir, f"week_{week}_results.csv")
    preds.to_csv(pred_path, index=False, encoding="utf-8")
    results.to_csv(result_path, index=False, encoding="utf-8")
    return pred_path, result_path


def get_cumulative_performance(db_conn, season, through_week=None):
    if through_week is None:
        return pd.read_sql("SELECT * FROM cumulative WHERE season=? ORDER BY through_week", db_conn, params=(season,))
    return pd.read_sql("SELECT * FROM cumulative WHERE season=? AND through_week<=? ORDER BY through_week",
                        db_conn, params=(season, through_week))


def run_2025_backtest_demo(weeks=(1, 4, 8, 12, 16), db_path=DEFAULT_DB_PATH):
    """End-to-end validation: builds real predictions/results for each
    checkpoint week of the completed 2025 season, logs them, computes
    weekly + cumulative accuracy, generates reports - proves the whole
    pipeline works before it's ever pointed at a live, in-progress 2026."""
    from elo_game_prediction import fit_probability_to_spread_conversion
    season = 2025
    fitted_model = fit_probability_to_spread_conversion()

    if os.path.exists(db_path):
        os.remove(db_path)  # fresh demo DB each run
    conn = create_tracking_database(db_path)

    for week in weeks:
        preds = build_weekly_predictions_df(season, week, fitted_model)
        results = build_weekly_results_df(season, week)
        n_pred = save_weekly_predictions(conn, season, week, preds)
        n_result = log_weekly_results(conn, season, week, results)
        print(f"\nWeek {week}: saved {n_pred} predictions, logged {n_result} results")
        generate_weekly_report(conn, season, week)
        export_weekly_csv(conn, season, week)

    print(f"\n{'=' * 40}\nCumulative trajectory across checkpoints:\n{'=' * 40}")
    cum = get_cumulative_performance(conn, season)
    print(cum.to_string(index=False))

    conn.close()
    return cum


if __name__ == "__main__":
    run_2025_backtest_demo()
