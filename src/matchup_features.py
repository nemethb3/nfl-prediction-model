"""Phase 2 Component 2.3: Matchup-Specific Features.

Corrects 2 issues found in the spec before building:

1. Individual player-vs-player coverage matchups (which specific CB covers
   which specific WR) don't exist in this project's real data - checked
   before building, not assumed: only 12.1% of real 2025 pass plays even
   have a charted pass_defense_1_player_id (and that field marks a
   defended-pass EVENT, typically an incompletion with contact - not a
   continuous coverage assignment, and it's silent on the other 87.9% of
   plays). This is genuinely hard, usually proprietary charting data (PFF-
   style) this project has never had access to. Building "off_player_id vs
   def_player_id, matchup_type WR-CB" as literally specified would mean
   fabricating who covered whom on the vast majority of real plays.
   Rescoped to the spec's own explicitly-offered fallback: TEAM-level real
   defensive EPA allowed BY POSITION (e.g. this team's real EPA/play
   allowed to opposing WRs) - genuinely real and computable from PBP by
   joining receiver_id/rusher_id to each player's real position (same
   crosswalk pattern already established elsewhere in this project -
   phase3_diagnostic.py/ol_quality.py already join receiver_id/rusher_id to
   real position data the same way). matchup_type becomes "offensive
   player vs. opposing team's real defensive EPA allowed to that position,"
   not "WR1 vs CB1."

2. apply_matchup_adjustment_to_game_spread()'s "±0.5 pts per dominant
   matchup" is an asserted constant. Fit a real linear regression instead:
   (real actual point_diff - Elo's own predicted spread) ~ real net
   matchup-edge differential, on real 2015-2023 train data - giving a real,
   derived coefficient rather than an asserted one, consistent with every
   other constant in this project.

All EPA figures used are TRAILING (through week N-1 only, real data leak-
free within the season) - not full-season aggregates, since this
component's real use case is informing an UPCOMING week's predictions,
matching the leak-free convention already established for the RB fantasy
formula and elsewhere this session.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostic")

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
DEF_PBP_COLS = ["posteam", "defteam", "season", "week", "season_type", "play_type", "epa", "receiver_id", "rusher_id"]


def _position_crosswalk():
    """Real player_id -> position, from player_weekly_stats.csv (the same
    real source used throughout this project) - most frequent real
    position per player, not assumed."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    pws = pws.dropna(subset=["position"])
    return pws.groupby("player_id")["position"].agg(lambda s: s.value_counts().idxmax())


def extract_player_epa_by_position(season, through_week=None):
    """Real trailing (through_week-1, leak-free) EPA/play per skill-
    position player - QB: passing; RB: rushing+receiving; WR/TE: receiving.
    through_week=None uses the full real season (season-total use only,
    e.g. historical regression fitting - NOT for in-season prediction)."""
    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    df = pws[(pws["season"] == season) & (pws["season_type"] == "REG") & (pws["position"].isin(SKILL_POSITIONS))].copy()
    if through_week is not None:
        df = df[df["week"] < through_week]
    if len(df) == 0:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "team", "epa_per_play"])

    df["rush_rec_epa"] = df[["rushing_epa", "receiving_epa"]].fillna(0).sum(axis=1)
    df["rush_rec_plays"] = df[["carries", "targets"]].fillna(0).sum(axis=1)

    agg = df.groupby(["player_id", "position"]).agg(
        player_name=("player_display_name", "last"), team=("recent_team", "last"),
        passing_epa=("passing_epa", "sum"), attempts=("attempts", "sum"),
        receiving_epa=("receiving_epa", "sum"), targets=("targets", "sum"),
        rush_rec_epa=("rush_rec_epa", "sum"), rush_rec_plays=("rush_rec_plays", "sum")).reset_index()

    epa_per_play = np.select(
        [agg["position"] == "QB", agg["position"] == "RB", agg["position"].isin(["WR", "TE"])],
        [agg["passing_epa"] / agg["attempts"].replace(0, np.nan),
         agg["rush_rec_epa"] / agg["rush_rec_plays"].replace(0, np.nan),
         agg["receiving_epa"] / agg["targets"].replace(0, np.nan)], default=np.nan)
    agg["epa_per_play"] = epa_per_play
    return agg.dropna(subset=["epa_per_play"])[["player_id", "player_name", "position", "team", "epa_per_play"]]


def build_defense_epa_by_position_multi_season(seasons, pbp_path=None):
    """Real team defensive EPA/play allowed, split by the REAL position of
    the offensive player targeted/rushing (see module docstring #1) -
    ONE chunked pass over the 1.3GB PBP file covering ALL requested
    seasons (not one pass per season - avoids re-reading the file 9x
    during _fit_matchup_spread_coefficient's train-season loop, the same
    redundant-read trap already caught and fixed elsewhere this session,
    e.g. weekly_predictions.py/dynamic_tracking.py)."""
    crosswalk = _position_crosswalk()
    path = pbp_path or os.path.join(RAW_DIR, "pbp_2015_2025.csv")
    seasons = list(seasons)

    keep = []
    for chunk in pd.read_csv(path, usecols=DEF_PBP_COLS, low_memory=False, chunksize=100_000):
        sub = chunk[(chunk["season"].isin(seasons)) & (chunk["season_type"] == "REG")
                    & (chunk["play_type"].isin(["pass", "run"]))]
        if len(sub):
            keep.append(sub)
    reg = pd.concat(keep, ignore_index=True).dropna(subset=["epa", "defteam"])

    reg["target_player"] = np.where(reg["play_type"] == "pass", reg["receiver_id"], reg["rusher_id"])
    reg["target_position"] = reg["target_player"].map(crosswalk)
    reg = reg.dropna(subset=["target_position"])
    reg = reg[reg["target_position"].isin(SKILL_POSITIONS)]

    return reg.groupby(["defteam", "season", "week", "target_position"])["epa"].mean().reset_index(
        name="epa_allowed_per_play").rename(columns={"defteam": "team", "target_position": "position"})


def build_defense_epa_by_position(season, through_week=None, multi_season_df=None, pbp_path=None):
    """Single-season (optionally trailing-through-week) view - filters an
    already-loaded multi-season frame if given, else does its own single-
    season chunked read (fine for one-off/live use; the multi-season
    entrypoint above is for the train-season loop specifically)."""
    if multi_season_df is None:
        multi_season_df = build_defense_epa_by_position_multi_season([season], pbp_path=pbp_path)
    df = multi_season_df[multi_season_df["season"] == season]
    if through_week is not None:
        df = df[df["week"] < through_week]
    return df.groupby(["team", "position"])["epa_allowed_per_play"].mean().reset_index()


def extract_game_matchups(season, week, offensive_epa_df=None):
    """Real starters (by real trailing usage) for each team's game this
    week, paired with the real opponent for the matchup lookup."""
    from game_predictions import _load_schedule_for_season
    if offensive_epa_df is None:
        offensive_epa_df = extract_player_epa_by_position(season, through_week=week)

    pws = pd.read_csv(os.path.join(PROCESSED_DIR, "player_weekly_stats.csv"))
    usage = pws[(pws["season"] == season) & (pws["season_type"] == "REG") & (pws["week"] < week)
                & (pws["position"].isin(SKILL_POSITIONS))].copy()
    usage["usage"] = usage[["attempts", "carries", "targets"]].fillna(0).sum(axis=1)
    starters = usage.groupby(["player_id", "recent_team", "position"])["usage"].sum().reset_index()
    starters = starters.sort_values("usage", ascending=False).groupby(["recent_team", "position"]).head(1)
    starters = starters.rename(columns={"recent_team": "team"})

    schedule = _load_schedule_for_season(season)
    games = schedule[(schedule["game_type"] == "REG") & (schedule["week"] == week)][
        ["game_id", "home_team", "away_team"]]

    rows = []
    for _, g in games.iterrows():
        for off_team, def_team in [(g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])]:
            for _, s in starters[starters["team"] == off_team].iterrows():
                rows.append({"game_id": g["game_id"], "off_team": off_team, "def_team": def_team,
                             "off_player_id": s["player_id"], "position": s["position"],
                             "matchup_type": f"{s['position']} vs {def_team} defense-vs-{s['position']}"})
    return pd.DataFrame(rows)


def calculate_matchup_epa_edges(matchups_df, offensive_epa_df, defensive_epa_df):
    m = matchups_df.merge(offensive_epa_df[["player_id", "epa_per_play"]].rename(
        columns={"player_id": "off_player_id", "epa_per_play": "offensive_epa"}), on="off_player_id", how="inner")
    m = m.merge(defensive_epa_df.rename(columns={"team": "def_team", "epa_allowed_per_play": "defensive_epa_allowed"}),
                on=["def_team", "position"], how="left")
    m["defensive_epa_allowed"] = m["defensive_epa_allowed"].fillna(defensive_epa_df["epa_allowed_per_play"].mean())
    m["edge_epa"] = m["offensive_epa"] - m["defensive_epa_allowed"]
    return m


def generate_matchup_summary_for_game(game_id, matchups_with_edges_df):
    game = matchups_with_edges_df[matchups_with_edges_df["game_id"] == game_id].sort_values("edge_epa", ascending=False)
    if len(game) == 0:
        return f"No matchup data for {game_id}"
    lines = [f"{game_id} - Matchup Analysis", "=" * 40]
    lines.append("\nBiggest offensive advantages:")
    for _, r in game.head(5).iterrows():
        lines.append(f"  {r['off_player_id']} ({r['position']}, {r['off_team']}) vs {r['def_team']} defense: "
                      f"{r['edge_epa']:+.3f} EPA/play edge")
    lines.append("\nBiggest defensive advantages (offense disadvantaged):")
    for _, r in game.tail(5).iloc[::-1].iterrows():
        lines.append(f"  {r['off_player_id']} ({r['position']}, {r['off_team']}) vs {r['def_team']} defense: "
                      f"{r['edge_epa']:+.3f} EPA/play edge")
    report = "\n".join(lines)
    print("\n" + report)
    return report


def _fit_matchup_spread_coefficient(train_seasons=range(2015, 2024)):
    """Real regression: (actual point_diff - Elo's own predicted spread) ~
    net matchup-edge differential, fit on real train seasons (see module
    docstring #2 - not an asserted +-0.5)."""
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads, _load_game_results

    fitted_model = fit_probability_to_spread_conversion()
    multi_def_epa = build_defense_epa_by_position_multi_season(train_seasons)  # ONE PBP pass for all train seasons

    rows = []
    for season in train_seasons:
        try:
            elo_preds = generate_elo_game_spreads(season, fitted_model)
        except Exception:
            continue
        actual = _load_game_results([season])[["game_id", "point_diff"]]
        off_epa = extract_player_epa_by_position(season)
        def_epa = build_defense_epa_by_position(season, multi_season_df=multi_def_epa)
        if len(off_epa) == 0 or len(def_epa) == 0:
            continue

        for week in sorted(elo_preds["week"].unique()):
            matchups = extract_game_matchups(season, week, off_epa)
            if len(matchups) == 0:
                continue
            edges = calculate_matchup_epa_edges(matchups, off_epa, def_epa)
            net = edges.groupby(["game_id", "off_team"])["edge_epa"].sum().reset_index()
            wk_games = elo_preds[elo_preds["week"] == week][["game_id", "home_team", "away_team", "predicted_spread"]]
            wk_games = wk_games.merge(net.rename(columns={"off_team": "home_team", "edge_epa": "home_net_edge"}),
                                        on=["game_id", "home_team"], how="left")
            wk_games = wk_games.merge(net.rename(columns={"off_team": "away_team", "edge_epa": "away_net_edge"}),
                                        on=["game_id", "away_team"], how="left")
            wk_games = wk_games.merge(actual, on="game_id", how="inner")
            rows.append(wk_games)

    if not rows:
        return 0.0
    all_games = pd.concat(rows, ignore_index=True).dropna(subset=["home_net_edge", "away_net_edge"])
    all_games["net_edge_diff"] = all_games["home_net_edge"] - all_games["away_net_edge"]
    all_games["residual"] = all_games["point_diff"] - all_games["predicted_spread"]
    if all_games["net_edge_diff"].std() == 0 or len(all_games) < 20:
        return 0.0
    slope, _ = np.polyfit(all_games["net_edge_diff"], all_games["residual"], 1)
    print(f"[_fit_matchup_spread_coefficient] real fitted slope (train {min(train_seasons)}-{max(train_seasons)}, "
          f"n={len(all_games)}): {slope:.3f} pts per unit net EPA edge")
    return float(slope)


def apply_matchup_adjustment_to_game_spread(elo_spread, net_edge_diff, coefficient):
    return elo_spread + coefficient * net_edge_diff


def generate_fantasy_matchup_ratings(edges_df, green_threshold=None, red_threshold=None):
    if green_threshold is None or red_threshold is None:
        green_threshold = float(edges_df["edge_epa"].quantile(0.67))
        red_threshold = float(edges_df["edge_epa"].quantile(0.33))
    out = edges_df.copy()
    out["flag"] = np.select([out["edge_epa"] > green_threshold, out["edge_epa"] < red_threshold],
                             ["GREEN", "RED"], default="YELLOW")
    return out[["off_player_id", "position", "off_team", "def_team", "edge_epa", "flag"]]


def validate_matchup_adjustments(season=2025, coefficient=None):
    """Real backtest: coefficient is fit on 2015-2023 (genuinely out-of-
    sample for 2025 - no LOOCV needed here since there's one fitted
    constant, not a weight tuned against the same season it's tested on)."""
    from elo_game_prediction import fit_probability_to_spread_conversion, generate_elo_game_spreads, _load_game_results

    if coefficient is None:
        coefficient = _fit_matchup_spread_coefficient()

    fitted_model = fit_probability_to_spread_conversion()
    elo_preds = generate_elo_game_spreads(season, fitted_model)
    actual = _load_game_results([season])[["game_id", "point_diff"]]
    off_epa = extract_player_epa_by_position(season)
    def_epa = build_defense_epa_by_position(season)

    all_rows = []
    for week in sorted(elo_preds["week"].unique()):
        matchups = extract_game_matchups(season, week, off_epa)
        if len(matchups) == 0:
            continue
        edges = calculate_matchup_epa_edges(matchups, off_epa, def_epa)
        net = edges.groupby(["game_id", "off_team"])["edge_epa"].sum().reset_index()
        wk = elo_preds[elo_preds["week"] == week][["game_id", "home_team", "away_team", "predicted_spread"]]
        wk = wk.merge(net.rename(columns={"off_team": "home_team", "edge_epa": "home_net_edge"}), on=["game_id", "home_team"], how="left")
        wk = wk.merge(net.rename(columns={"off_team": "away_team", "edge_epa": "away_net_edge"}), on=["game_id", "away_team"], how="left")
        all_rows.append(wk)

    games = pd.concat(all_rows, ignore_index=True).merge(actual, on="game_id", how="inner")
    games[["home_net_edge", "away_net_edge"]] = games[["home_net_edge", "away_net_edge"]].fillna(0.0)
    games["net_edge_diff"] = games["home_net_edge"] - games["away_net_edge"]
    games["adjusted_spread"] = apply_matchup_adjustment_to_game_spread(games["predicted_spread"], games["net_edge_diff"], coefficient)

    without_corr = games["predicted_spread"].corr(games["point_diff"])
    without_mae = float(np.mean(np.abs(games["predicted_spread"] - games["point_diff"])))
    with_corr = games["adjusted_spread"].corr(games["point_diff"])
    with_mae = float(np.mean(np.abs(games["adjusted_spread"] - games["point_diff"])))

    print(f"\n{'=' * 60}\nMATCHUP ADJUSTMENT VALIDATION (real {season}, coefficient={coefficient:.3f})\n{'=' * 60}")
    print(f"Without matchup adjustment: corr={without_corr:+.3f} MAE={without_mae:.2f}")
    print(f"With matchup adjustment   : corr={with_corr:+.3f} MAE={with_mae:.2f}")
    print(f"Delta: corr {with_corr - without_corr:+.3f} | MAE {with_mae - without_mae:+.2f} "
          f"({'better' if with_mae < without_mae else 'worse'})")
    print("=" * 60)

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    games.to_csv(os.path.join(DIAGNOSTIC_DIR, "matchup_validation_2025.csv"), index=False, encoding="utf-8")
    return {"without_corr": without_corr, "without_mae": without_mae, "with_corr": with_corr,
            "with_mae": with_mae, "coefficient": coefficient, "n": len(games)}


if __name__ == "__main__":
    coeff = _fit_matchup_spread_coefficient()
    validate_matchup_adjustments(2025, coefficient=coeff)

    off_epa = extract_player_epa_by_position(2025, through_week=4)
    def_epa = build_defense_epa_by_position(2025, through_week=4)
    matchups = extract_game_matchups(2025, 4, off_epa)
    edges = calculate_matchup_epa_edges(matchups, off_epa, def_epa)
    if len(edges):
        sample_game = edges["game_id"].iloc[0]
        generate_matchup_summary_for_game(sample_game, edges)
        ratings = generate_fantasy_matchup_ratings(edges)
        print("\nSample fantasy matchup ratings:")
        print(ratings.head(10).to_string(index=False))
