"""Real walk-forward backtest of the EXACT live staking strategy (STEP 6)
- same forecasting/pricing/valuation/staking modules the live weekly
picks generator uses, run week-by-week across real 2023/2024/2025
out-of-sample games, with a fresh 1000 credits every week (no bankroll
carryover - matches the live competition's own weekly reset).

No lookahead: every game's p_model/win_prob_home already comes from
calibration.build_walk_forward_dataset's real walk-forward predictions
(trained only on seasons strictly before the test season); every market
price is the real historical closing-ish line for that same game (no
future information).
"""

import os

import numpy as np
import pandas as pd

from juke.config import TOTAL_CREDITS
from juke.calibration import build_walk_forward_dataset, fit_calibration
from juke.engine import build_game_options
from juke.pricing import get_market_price
from juke.staking import allocate_stakes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _row_to_event(row):
    return {
        "game_id": row["game_id"], "week": int(row["week"]), "season": int(row["test_season"]),
        "home_team": row["home_team"], "away_team": row["away_team"],
        # Real production fields (od-elo): kept under these real names so
        # forecasting.py's ensemble blend (which reads event["our_spread"]/
        # event["win_prob_home"] as the od-elo leg) works identically on
        # historical rows and live games_2026.json rows.
        "our_spread": row["od_predicted_spread"], "win_prob_home": row["od_win_prob_home"],
        "single_elo_spread": row["single_elo_predicted_spread"],
        "single_elo_win_prob_home": row["single_elo_win_prob_home"],
        "vegas_spread": row["spread_line"],
        "home_moneyline": row["home_moneyline"], "away_moneyline": row["away_moneyline"],
        "home_spread_odds": row["home_spread_odds"], "away_spread_odds": row["away_spread_odds"],
        "used_default_spread_juice": False,
        "vegas_line_source": "nflverse_historical", "vegas_spread_observed_at": None,
        "home_margin": row["home_margin"], "home_win": row["home_win"],
        "spread_push": row.get("spread_push", 0),
    }


def _settle(pick):
    event = pick["event"]
    if pick["market"] == "moneyline":
        if event["home_win"] != event["home_win"]:  # real NaN check (a real, rare tie)
            return "push", 0.0
        home_won = bool(event["home_win"])
        won = home_won if pick["side"] == "home" else not home_won
    else:
        if event.get("spread_push") == 1:
            return "push", 0.0
        home_covered = event["home_margin"] > event["vegas_spread"]
        won = home_covered if pick["side"] == "home" else not home_covered
    if won:
        return "win", pick["credits"] * (pick["decimal_odds"] - 1.0)
    return "loss", -float(pick["credits"])


def _run_strategy_week(events, sigma, weights, elo_ensemble=None):
    options = []
    for event in events:
        options.extend(build_game_options(event, sigma, weights, elo_ensemble))
    picks, warning = allocate_stakes(options)
    for pick in picks:
        result, pnl = _settle(pick)
        pick["result"], pick["pnl"] = result, pnl
    return picks, warning


def _baseline_our_pick_flat(events, sigma, weights, elo_ensemble=None):
    """Real baseline (a): 1000 credits spread equally across our own
    pick (same selection logic, no Kelly weighting) in every game."""
    picks = []
    n = len(events)
    if n == 0:
        return picks
    flat_credits = TOTAL_CREDITS / n
    for event in events:
        best = build_game_options(event, sigma, weights, elo_ensemble)
        if not best:
            continue
        pick = dict(best[0])
        pick["credits"] = flat_credits
        result, pnl = _settle(pick)
        pick["result"], pick["pnl"] = result, pnl
        picks.append(pick)
    return picks


def _baseline_vegas_favorite_flat(events):
    """Real baseline (b): 1000 credits spread equally across the real
    market moneyline favorite in every game."""
    picks = []
    n = len(events)
    if n == 0:
        return picks
    flat_credits = TOTAL_CREDITS / n
    for event in events:
        price_home = get_market_price(event, "moneyline", "home")
        price_away = get_market_price(event, "moneyline", "away")
        if price_home is None or price_away is None:
            continue
        side = "home" if price_home["price"] >= price_away["price"] else "away"
        price_info = price_home if side == "home" else price_away
        pick = {
            "game_id": event["game_id"], "week": event["week"], "market": "moneyline", "side": side,
            "decimal_odds": price_info["decimal_odds"], "credits": flat_credits, "event": event,
        }
        result, pnl = _settle(pick)
        pick["result"], pick["pnl"] = result, pnl
        picks.append(pick)
    return picks


def _summarize(picks, label):
    if not picks:
        return {"label": label, "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_pct": 0.0, "roi_pct": 0.0, "total_pnl": 0.0}
    df = pd.DataFrame(picks)
    decided = df[df["result"] != "push"]
    wins = int((decided["result"] == "win").sum())
    losses = int((decided["result"] == "loss").sum())
    staked = float(decided["credits"].sum())
    pnl = float(decided["pnl"].sum())
    return {
        "label": label, "bets": int(len(df)), "wins": wins, "losses": losses,
        "pushes": int((df["result"] == "push").sum()),
        "win_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        "roi_pct": round(pnl / staked * 100, 2) if staked else 0.0,
        "total_pnl": round(pnl, 2),
    }


def run_backtest(test_seasons=(2023, 2024, 2025)):
    calib = fit_calibration()
    sigma = calib["sigma"]
    weights = {"moneyline": calib["moneyline"]["w_used"], "spread": calib["spread"]["w_used"]}
    elo_ensemble = calib["elo_ensemble"]
    wf_df = build_walk_forward_dataset(test_seasons=test_seasons)

    per_season = {}
    strategy_picks_all, core_filler_all = [], []
    baseline_a_all, baseline_b_all = [], []
    weekly_rows = []

    for season in test_seasons:
        season_df = wf_df[wf_df["test_season"] == season]
        season_strategy, season_a, season_b = [], [], []
        for week, week_df in season_df.groupby("week"):
            events = [_row_to_event(r) for _, r in week_df.iterrows()]
            picks, warning = _run_strategy_week(events, sigma, weights, elo_ensemble)
            picks_a = _baseline_our_pick_flat(events, sigma, weights, elo_ensemble)
            picks_b = _baseline_vegas_favorite_flat(events)

            week_pnl = sum(p["pnl"] for p in picks)
            weekly_rows.append({
                "season": int(season), "week": int(week), "bets": len(picks),
                "credits_staked": round(sum(p["credits"] for p in picks), 1),
                "pnl": round(week_pnl, 2), "warning": warning,
            })
            season_strategy.extend(picks)
            season_a.extend(picks_a)
            season_b.extend(picks_b)

        strategy_picks_all.extend(season_strategy)
        baseline_a_all.extend(season_a)
        baseline_b_all.extend(season_b)
        core_filler_all.extend(season_strategy)

        pnl_series = pd.Series([r["pnl"] for r in weekly_rows if r["season"] == season])
        # Real, disclosed distinction (2026-10-06): sigma/w/elo_ensemble were
        # fit ONLY on calib["dev_seasons"] - a season in that set partially
        # informed its own reported number here (the usual hyperparameter-
        # selection caveat); calib["holdout_season"] never did, and is the
        # one honest, uncontaminated out-of-sample result.
        per_season[int(season)] = {
            "role": "holdout (true out-of-sample)" if season == calib["holdout_season"] else
                    "dev (hyperparameters partly fit on this season too - expect some optimism)",
            "strategy": _summarize(season_strategy, "Our strategy"),
            "baseline_our_pick_flat": _summarize(season_a, "Baseline: our pick, flat stake"),
            "baseline_vegas_favorite_flat": _summarize(season_b, "Baseline: Vegas favorite, flat stake"),
            "worst_week_pnl": round(float(pnl_series.min()), 2) if len(pnl_series) else None,
            "max_drawdown": round(float(_max_drawdown(pnl_series)), 2) if len(pnl_series) else None,
        }

    core_picks = [p for p in strategy_picks_all if p.get("tier") == "core"]
    filler_picks = [p for p in strategy_picks_all if p.get("tier") == "filler"]
    holdout_picks = [p for p in strategy_picks_all if p["event"]["season"] == calib["holdout_season"]]

    return {
        "calibration": calib,
        "per_season": per_season,
        "overall": {
            "strategy": _summarize(strategy_picks_all, "Our strategy"),
            "baseline_our_pick_flat": _summarize(baseline_a_all, "Baseline: our pick, flat stake"),
            "baseline_vegas_favorite_flat": _summarize(baseline_b_all, "Baseline: Vegas favorite, flat stake"),
            "core": _summarize(core_picks, "Core picks only"),
            "filler": _summarize(filler_picks, "Filler picks only"),
        },
        "holdout_only": _summarize(holdout_picks, f"Strategy, {calib['holdout_season']} holdout only (honest number)"),
        "weekly": weekly_rows,
    }


def _max_drawdown(pnl_series):
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    return drawdown.min() if len(drawdown) else 0.0


if __name__ == "__main__":
    result = run_backtest()
    print(f"Sigma: {result['calibration']['sigma']:.2f}  "
          f"w_moneyline={result['calibration']['moneyline']['w_used']:.2f}  "
          f"w_spread={result['calibration']['spread']['w_used']:.2f}\n")
    for season, s in result["per_season"].items():
        print(f"Season {season} [{s['role']}]:")
        for key in ("strategy", "baseline_our_pick_flat", "baseline_vegas_favorite_flat"):
            print(f"  {s[key]['label']:35s} bets={s[key]['bets']:3d} win%={s[key]['win_pct']:5.1f} "
                  f"ROI%={s[key]['roi_pct']:6.2f} PnL={s[key]['total_pnl']:8.2f}")
        print(f"  Worst week PnL: {s['worst_week_pnl']}   Max drawdown: {s['max_drawdown']}")
    print(f"\nHoldout-only (the honest number): {result['holdout_only']}")
    print("\nOverall (dev+holdout pooled, dev seasons carry some optimism):")
    for key in ("strategy", "baseline_our_pick_flat", "baseline_vegas_favorite_flat", "core", "filler"):
        o = result["overall"][key]
        print(f"  {o['label']:35s} bets={o['bets']:3d} win%={o['win_pct']:5.1f} "
              f"ROI%={o['roi_pct']:6.2f} PnL={o['total_pnl']:8.2f}")
