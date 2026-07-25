"""Feature engineering: age curves and player history features.

Builds on the processed tables from Task 1.3 (data/processed/player_season_stats.csv,
player_season_defense.csv, player_weekly_stats.csv) to produce:
  - data/processed/age_curves_by_position.pkl: fitted aging curves (degree-3
    polynomial) for a primary production metric per position.
  - data/processed/age_curves_visualization/*.png: actual-vs-fitted plots.
  - data/processed/player_features_with_history.csv: per player-season history
    features (last_year, last_3yr_avg, career_avg, rookie/team-change/role-change/
    coaching-change flags) used to train the Phase 2 position models.

Only QB/RB/WR/TE (offense) and EDGE/DL/LB/CB/S (defense) get a primary metric -
OL/K/P/LS/DB are skipped since there's no single meaningful production stat for
them in the available data.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VIZ_DIR = os.path.join(PROCESSED_DIR, "age_curves_visualization")

# position -> (source table, metric column). "total_yards" is derived
# (rushing_yards + receiving_yards) rather than a raw column.
PRIMARY_METRIC = {
    "QB": ("offense", "passing_yards"),
    "RB": ("offense", "total_yards"),
    "WR": ("offense", "receiving_yards"),
    "TE": ("offense", "receiving_yards"),
    "EDGE": ("defense", "sk"),
    "DL": ("defense", "sk"),
    "LB": ("defense", "comb"),
    "CB": ("defense", "comb"),
    "S": ("defense", "comb"),
}


def load_processed_tables():
    season_off = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_stats.csv"))
    season_def = pd.read_csv(os.path.join(PROCESSED_DIR, "player_season_defense.csv"))
    weekly_off = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    return season_off, season_def, weekly_off


def season_team_from_weekly(weekly_off):
    """A player's team for a season = the team they played for in their last
    game that season (best single-value proxy for in-season trades)."""
    idx = weekly_off.sort_values("week").groupby(["player_id", "season"])["week"].idxmax()
    last_week = weekly_off.loc[idx, ["player_id", "season", "recent_team"]]
    return last_week.rename(columns={"recent_team": "team"})


def build_history_base(season_off, season_def, weekly_off):
    """Long table of one row per (player_id, season) for every position that
    has a primary metric defined, with a unified metric_value column."""
    season_off = season_off.copy()
    season_off["total_yards"] = season_off["rushing_yards"].fillna(0) + season_off["receiving_yards"].fillna(0)
    season_team = season_team_from_weekly(weekly_off)
    season_off = season_off.merge(season_team, on=["player_id", "season"], how="left")

    off_frames = []
    for position, (source, metric) in PRIMARY_METRIC.items():
        if source != "offense":
            continue
        sub = season_off[season_off["position"] == position].copy()
        sub["metric_value"] = sub[metric]
        off_frames.append(sub[[
            "player_id", "display_name", "position", "season", "team", "age",
            "years_in_league", "games_played", "avg_snap_pct", "metric_value",
        ]])

    season_def = season_def.rename(columns={"player": "display_name", "games": "games_played"})
    def_frames = []
    for position, (source, metric) in PRIMARY_METRIC.items():
        if source != "defense":
            continue
        sub = season_def[season_def["position"] == position].copy()
        sub["metric_value"] = sub[metric]
        sub["avg_snap_pct"] = np.nan  # defensive snap share not wired in yet (see report)
        def_frames.append(sub[[
            "player_id", "display_name", "position", "season", "team", "age",
            "years_in_league", "games_played", "avg_snap_pct", "metric_value",
        ]])

    history = pd.concat(off_frames + def_frames, ignore_index=True)
    history = history.dropna(subset=["player_id"]).sort_values(["player_id", "season"]).reset_index(drop=True)
    print(f"[history_base] {history.shape[0]:,} player-seasons across {sorted(PRIMARY_METRIC)}")
    return history


def build_coach_crosswalk(schedules):
    """Team-season -> head coach of record (mode of the coach listed across
    that team's games that season; head_coach can vary mid-season on a fire/
    hire, so mode picks the coach who ran most of the games)."""
    home = schedules[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"})
    away = schedules[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"})
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["coach"])
    crosswalk = (
        long.groupby(["team", "season"])["coach"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    return crosswalk


def identify_team_changes(history):
    """team_change_flag: True if a player's season team differs from their
    immediately preceding season's team. NaN if there's no consecutive prior
    season (rookie, or gap year out of the league)."""
    history = history.sort_values(["player_id", "season"]).copy()
    grp = history.groupby("player_id")
    prev_season = grp["season"].shift(1)
    prev_team = grp["team"].shift(1)
    consecutive = history["season"] - prev_season == 1
    flag = pd.Series(np.nan, index=history.index, dtype="object")
    flag[consecutive] = history.loc[consecutive, "team"] != prev_team[consecutive]
    history["team_change_flag"] = flag
    return history


def identify_role_changes(history, threshold=0.10):
    """role_change_flag: True if avg_snap_pct moved more than `threshold`
    (10 percentage points by default) versus the immediately preceding
    season. Offense only (avg_snap_pct is NaN for defense - see build_history_base)."""
    history = history.sort_values(["player_id", "season"]).copy()
    grp = history.groupby("player_id")
    prev_season = grp["season"].shift(1)
    prev_snap_pct = grp["avg_snap_pct"].shift(1)
    consecutive = (history["season"] - prev_season == 1) & history["avg_snap_pct"].notna() & prev_snap_pct.notna()
    flag = pd.Series(np.nan, index=history.index, dtype="object")
    flag[consecutive] = (history.loc[consecutive, "avg_snap_pct"] - prev_snap_pct[consecutive]).abs() > threshold
    history["role_change_flag"] = flag
    return history


def identify_coaching_changes(history, coach_crosswalk):
    """head_coach_change_flag: True if a player's team has a different head
    coach of record than that same team had the prior season."""
    cw = coach_crosswalk.sort_values(["team", "season"]).copy()
    cw["prev_coach"] = cw.groupby("team")["coach"].shift(1)
    cw["prev_season"] = cw.groupby("team")["season"].shift(1)
    cw["head_coach_change_flag"] = np.where(
        cw["season"] - cw["prev_season"] == 1, cw["coach"] != cw["prev_coach"], np.nan
    )
    merged = history.merge(
        cw[["team", "season", "head_coach_change_flag"]], on=["team", "season"], how="left"
    )
    return merged


def _lookback_avg(group, value_col, window):
    """For each row, the mean of value_col over prior seasons within
    [season - window, season - 1] - whatever subset of that window actually
    exists for this player, not requiring perfect consecutiveness."""
    seasons = group["season"].to_numpy()
    values = group[value_col].to_numpy(dtype=float)
    out = np.full(len(group), np.nan)
    for i, s in enumerate(seasons):
        mask = (seasons < s) & (seasons >= s - window)
        if mask.any():
            out[i] = values[mask].mean()
    return pd.Series(out, index=group.index)


def compute_history_features(df, value_col, id_col="player_id", season_col="season"):
    """Generic lookback features for any per-player-season column: {value_col}_last_year,
    {value_col}_last_3yr_avg, {value_col}_career_avg - each respecting actual season gaps
    (a player who missed a year doesn't get a neighboring season silently treated as
    "last year"). Returns a DataFrame aligned to df's index (not a copy of df itself).

    Used both for the primary per-position metric (see build_regression_targets) and,
    in Phase 2, for secondary stats a specific position model needs (e.g. TDs, EPA,
    volume) that aren't the position's primary metric.
    """
    df = df.sort_values([id_col, season_col])
    grp = df.groupby(id_col)
    prev_season = grp[season_col].shift(1)
    prev_val = grp[value_col].shift(1)
    last_year = np.where(df[season_col] - prev_season == 1, prev_val, np.nan)
    last_3yr = df.groupby(id_col, group_keys=False).apply(lambda g: _lookback_avg(g, value_col, 3))
    career = df.groupby(id_col, group_keys=False).apply(lambda g: _lookback_avg(g, value_col, 99))
    return pd.DataFrame({
        f"{value_col}_last_year": last_year,
        f"{value_col}_last_3yr_avg": last_3yr,
        f"{value_col}_career_avg": career,
    }, index=df.index)


def build_regression_targets(history):
    """Adds last_year_value, last_3yr_avg, career_avg, and rookie_flag."""
    history = history.sort_values(["player_id", "season"]).copy()
    hist_feats = compute_history_features(history, "metric_value")
    history["last_year_value"] = hist_feats["metric_value_last_year"]
    history["last_3yr_avg"] = hist_feats["metric_value_last_3yr_avg"]
    history["career_avg"] = hist_feats["metric_value_career_avg"]
    history["rookie_flag"] = history["years_in_league"] == 1
    return history


def build_position_age_curves(history, delta_degree=2, min_pair_n=10):
    """Fit each position's aging curve via the delta method rather than a
    naive fit on raw age-bucket means.

    A naive mean-by-age curve is badly survivorship-biased: only the players
    good/durable enough to still have a roster spot stick around at older
    ages, so the *average* of survivors can look flat or even rise late in
    the aging window even though every individual is declining (this showed
    up here as e.g. QB "peaking" at 37 and RB/WR "peaking" at the youngest
    age in the data). The delta method instead looks only at same-player
    year-over-year changes, which cancels that bias: fit a low-degree
    polynomial to (age, mean year-over-year change), then integrate it to
    get a level curve, and calibrate the integration constant against the
    raw mean at one reference age.
    """
    curves = {}
    for position in sorted(history["position"].unique()):
        sub = (
            history[history["position"] == position]
            .dropna(subset=["age", "metric_value"])
            .sort_values(["player_id", "season"])
            .copy()
        )
        grp = sub.groupby("player_id")
        prev_age = grp["age"].shift(1)
        prev_metric = grp["metric_value"].shift(1)
        prev_season = grp["season"].shift(1)
        consecutive = (sub["season"] - prev_season) == 1

        pairs = pd.DataFrame({
            "age_start": prev_age[consecutive].values,
            "delta": sub.loc[consecutive, "metric_value"].values - prev_metric[consecutive].values,
        }).dropna()
        pairs["age_round"] = pairs["age_start"].round()
        by_delta = pairs.groupby("age_round")["delta"].agg(["mean", "count"])
        by_delta = by_delta[by_delta["count"] >= min_pair_n]

        if len(by_delta) < delta_degree + 2:
            print(f"[age_curves] WARNING: skipping {position}, only {len(by_delta)} "
                  f"age-transition buckets with n>={min_pair_n}")
            continue

        delta_coeffs = np.polyfit(by_delta.index.values, by_delta["mean"].values, deg=delta_degree)
        level_coeffs = np.polyint(delta_coeffs)  # constant term = 0 for now

        raw_by_age = sub.assign(age_round=sub["age"].round()).groupby("age_round")["metric_value"].mean()
        anchor_age = raw_by_age.index[np.argmin(np.abs(raw_by_age.index - sub["age"].median()))]
        anchor_level = raw_by_age.loc[anchor_age]
        level_coeffs[-1] += anchor_level - np.polyval(level_coeffs, anchor_age)

        curves[position] = {
            "coeffs": level_coeffs,
            "metric": PRIMARY_METRIC[position][1],
            "age_range": (float(by_delta.index.min()), float(by_delta.index.max()) + 1),
            "n_seasons": int(sub["season"].nunique()),
            "n_transition_pairs": int(pairs.shape[0]),
        }
    return curves


def predict_by_age_curve(position, age, curves):
    """Expected production for a position at a given age, per the fitted
    curve. Clipped at 0 since polynomials can dip negative outside the
    fitted range."""
    if position not in curves:
        raise KeyError(f"No fitted age curve for position '{position}'")
    coeffs = curves[position]["coeffs"]
    return max(0.0, float(np.polyval(coeffs, age)))


def plot_age_curves(history, curves, out_dir=VIZ_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for position, curve in curves.items():
        sub = history[history["position"] == position].dropna(subset=["age", "metric_value"]).copy()
        sub["age_round"] = sub["age"].round()
        by_age = sub.groupby("age_round")["metric_value"].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(by_age.index, by_age.values, label="actual mean", color="steelblue")
        age_grid = np.linspace(curve["age_range"][0], curve["age_range"][1], 100)
        fitted = np.polyval(curve["coeffs"], age_grid)
        ax.plot(age_grid, fitted, color="firebrick", label="fitted curve")
        ax.set_title(f"{position} aging curve ({curve['metric']})")
        ax.set_xlabel("age")
        ax.set_ylabel(curve["metric"])
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{position}_age_curve.png"), dpi=100)
        plt.close(fig)
    print(f"[age_curves] saved {len(curves)} plots to {out_dir}")


def validate_age_curves(curves):
    print("\n===== AGE CURVE VALIDATION =====")
    for position, curve in sorted(curves.items()):
        lo, hi = curve["age_range"]
        age_grid = np.linspace(lo, hi, 200)
        fitted = np.polyval(curve["coeffs"], age_grid)
        peak_age = age_grid[np.argmax(fitted)]
        print(f"{position:5s} | metric={curve['metric']:15s} | age range {lo:.0f}-{hi:.0f} | "
              f"peak age ~{peak_age:.1f} | seasons of data: {curve['n_seasons']}")
    print("===== END AGE CURVE VALIDATION =====\n")


def validate_player_features(history):
    print("\n===== PLAYER FEATURES VALIDATION =====")
    print(f"Total player-seasons: {history.shape[0]:,}")
    print(history.groupby("position").size().rename("n").to_string())
    print(f"\nRookie share: {history['rookie_flag'].mean() * 100:.1f}%")
    non_rookie = history[~history["rookie_flag"]]
    print(f"Non-rookie rows missing last_year_value (gap years): "
          f"{non_rookie['last_year_value'].isna().mean() * 100:.1f}%")
    print(f"team_change_flag known-rate: {history['team_change_flag'].notna().mean() * 100:.1f}%, "
          f"True rate (of known): {history['team_change_flag'].dropna().mean() * 100:.1f}%")
    role_known = history["role_change_flag"].notna()
    print(f"role_change_flag known-rate (offense only): {role_known.mean() * 100:.1f}%, "
          f"True rate (of known): {history.loc[role_known, 'role_change_flag'].mean() * 100:.1f}%")
    print(f"head_coach_change_flag known-rate: {history['head_coach_change_flag'].notna().mean() * 100:.1f}%, "
          f"True rate (of known): {history['head_coach_change_flag'].dropna().mean() * 100:.1f}%")
    print("===== END PLAYER FEATURES VALIDATION =====\n")


def run_feature_engineering():
    season_off, season_def, weekly_off = load_processed_tables()
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules_2015_2025.csv"))

    history = build_history_base(season_off, season_def, weekly_off)
    history = identify_team_changes(history)
    history = identify_role_changes(history)
    coach_crosswalk = build_coach_crosswalk(schedules)
    history = identify_coaching_changes(history, coach_crosswalk)
    history = build_regression_targets(history)

    curves = build_position_age_curves(history)
    validate_age_curves(curves)
    plot_age_curves(history, curves)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "age_curves_by_position.pkl"), "wb") as f:
        pickle.dump(curves, f)
    print(f"Saved {os.path.join(PROCESSED_DIR, 'age_curves_by_position.pkl')} ({len(curves)} positions)")

    validate_player_features(history)
    out_path = os.path.join(PROCESSED_DIR, "player_features_with_history.csv")
    history.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({history.shape[0]:,} rows)")


if __name__ == "__main__":
    run_feature_engineering()
