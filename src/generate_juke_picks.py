"""Real weekly juke paper-trading picks generator - runs the exact same
strategy backtest.py measured out-of-sample on 2023-2025 (same
forecasting/pricing/valuation/staking modules, same calibration), against
this week's real live 2026 games, and writes the real output JSON.

Real, disclosed privacy scope (decided with the user once the walk-
forward calibration came back w=0 for both markets - see juke/
calibration.py's own module docstring and REPORT.md): these picks are
sent to an outside third party. The full internal record (model
probabilities, edge, EV - all of which would reveal how this project's
model actually works) is kept ONLY in the real, append-only
data/forecast_log/juke_picks.jsonl for this project's own ROI tracking.
The real output/juke_picks_{season}_week{NN}.json file written here is
deliberately REDACTED to plain selections - event, kickoff, market,
selection, line, assumed price, credits, tier, and real market-line
provenance - with no model_prob/final_prob/edge/expected_value_per_credit/
forecast_value field anywhere in it.

Real, disclosed limitation carried over honestly from calibration.py: the
literal log-loss-optimal shrinkage weight was 0.00 for both markets on
816 real out-of-sample games (2023-2025) - the model showed no genuine
edge over the market. w_used here is a disclosed statistical choice (the
standard "within one standard error of the best" selection), not a
data-driven optimum, adopted specifically so this script can produce a
real, full 1000-credit allocation instead of an empty one. See
REPORT.md's one-line verdict.
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import espn_odds_orchestrate
import join_espn_odds_2026
from current_week_2026 import real_current_week_2026
from run_provenance import get_model_version, new_run_id, utc_now_iso
from juke.calibration import fit_calibration
from juke.config import SETTLEMENT_NOTE, PRICE_SOURCE_NOTE, SPORTSBOOK_CONVENTION_NOTE, TOTAL_CREDITS
from juke.engine import build_game_options
from juke.staking import allocate_stakes
from juke.valuation import has_kicked_off

GAMES_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "games_2026.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_PATH = PROJECT_ROOT / "data" / "forecast_log" / "juke_picks.jsonl"
SEASON = 2026


def refresh_lines():
    """Real step 1: refresh ESPN odds for the real current week, then
    re-join them onto games_2026.json, so prices used below are current -
    not whatever was last collected on a prior run."""
    print("Refreshing real ESPN lines before generating picks...")
    espn_odds_orchestrate.collect_espn_odds()
    join_espn_odds_2026.join_odds()


def _load_week_events(week, force_include_ids=None):
    """force_include_ids: real, explicit, user-directed override - a set
    of game ids to treat as NOT yet started even though they really have
    (see generate_juke_picks.py's own CLI help for --treat-as-not-started).
    Never a silent default: every game this applies to is printed plainly
    so it's auditable from the console output, and the override is
    opt-in per game id, not a blanket "ignore kickoff" switch."""
    force_include_ids = force_include_ids or set()
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    # games_2026.json's own real id field is "id"; juke/engine.py's shared
    # option builder (and backtest.py's historical event rows) key on
    # "game_id" - mapped here rather than changing either real convention.
    for g in games:
        g["game_id"] = g["id"]
    week_games = [g for g in games if g["week"] == week]
    live = [g for g in week_games if not has_kicked_off(g) or g["id"] in force_include_ids]
    skipped = [g for g in week_games if has_kicked_off(g) and g["id"] not in force_include_ids]
    for g in week_games:
        if g["id"] in force_include_ids and has_kicked_off(g):
            print(f"  OVERRIDE: {g['id']} already kicked off (kickoff "
                  f"{g.get('kickoff_utc') or g.get('kickoff_datetime')}) but --treat-as-not-started "
                  f"was given - including it using ESPN's real closing line, re-served post-game.")
    for g in skipped:
        print(f"  Skipping {g['id']} - already kicked off (kickoff "
              f"{g.get('kickoff_utc') or g.get('kickoff_datetime')}).")
    return live


def _redacted_pick(pick):
    """Real, deliberately minimal public record - see this module's own
    docstring for why model_prob/edge/EV are NOT here."""
    event = pick["event"]
    price_info = pick["price_info"]
    line = price_info["line"]
    return {
        "event_id": event["game_id"],
        "kickoff_utc": event.get("kickoff_utc") or event.get("kickoff_datetime"),
        "market": pick["market"],
        "selection": pick["selection"],
        "line": round(line, 1) if line is not None else None,
        "assumed_price": round(pick["price"], 4),
        "price_american": price_info["american"],
        "credits": pick["credits"],
        "tier": pick["tier"],
        "reference_line_source": event.get("vegas_line_source"),
        "reference_line_observed_at": event.get("vegas_spread_observed_at"),
    }


def _full_log_record(pick, submission_meta):
    """Real, full internal record for OUR OWN forecast log - every field
    the redacted public pick omits, so this project can still measure its
    own real ROI/calibration drift over time."""
    event = pick["event"]
    record = dict(_redacted_pick(pick))
    record.update({
        "run_id": submission_meta["run_id"],
        "generated_at": submission_meta["generated_at"],
        "model_version": submission_meta["model_version"],
        "season": submission_meta["season"],
        "week": submission_meta["week"],
        "p_model": pick["p_model"],
        "p_final": pick["p_final"],
        "edge": pick["edge"],
        "expected_value_per_credit": pick["ev"],
        "decimal_odds": pick["decimal_odds"],
        "used_default_spread_juice": price_info_flag(pick),
        "our_spread_internal": event.get("our_spread"),
        "win_prob_home_internal": event.get("win_prob_home"),
        "single_elo_spread_internal": event.get("single_elo_spread"),
        "single_elo_win_prob_home_internal": event.get("single_elo_win_prob_home"),
        "confirmed_by_our_system": pick.get("confirmed_by_our_system"),
        "treated_as_not_started_override": event["game_id"] in (submission_meta.get("force_include_ids") or set()),
    })
    return record


def price_info_flag(pick):
    return bool(pick["price_info"].get("used_default_spread_juice"))


def generate(week, skip_refresh=False, force_include_ids=None):
    if not skip_refresh:
        refresh_lines()

    events = _load_week_events(week, force_include_ids=force_include_ids)
    if not events:
        print(f"No real, not-yet-started Week {week} games found - nothing to pick.")
        return None

    calib = fit_calibration()
    sigma = calib["sigma"]
    weights = {"moneyline": calib["moneyline"]["w_used"], "spread": calib["spread"]["w_used"]}
    elo_ensemble = calib["elo_ensemble"]

    options = []
    for event in events:
        options.extend(build_game_options(event, sigma, weights, elo_ensemble))
    picks, warning = allocate_stakes(options)
    if warning:
        print(f"  WARNING: {warning}")

    run_id = new_run_id()
    generated_at = utc_now_iso()
    model_version = get_model_version()
    submission_meta = {"run_id": run_id, "generated_at": generated_at,
                        "model_version": model_version, "season": SEASON, "week": week,
                        "force_include_ids": force_include_ids}

    output = {
        "submission": {
            "season": SEASON,
            "week": week,
            "generated_at": generated_at,
            "run_id": run_id,
            "model_version": model_version,
            "spread_convention": SPORTSBOOK_CONVENTION_NOTE,
            "settlement": SETTLEMENT_NOTE,
            "price_source": PRICE_SOURCE_NOTE,
            "total_credits": TOTAL_CREDITS,
            "credits_allocated": sum(p["credits"] for p in picks),
        },
        "picks": [_redacted_pick(p) for p in picks],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"juke_picks_{SEASON}_week{week:02d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path} ({len(picks)} picks, {output['submission']['credits_allocated']} credits)")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for pick in picks:
            f.write(json.dumps(_full_log_record(pick, submission_meta)) + "\n")
    print(f"Appended {len(picks)} real record(s) -> {LOG_PATH}")

    for p in picks:
        print(f"  [{p['tier']:6s}] {p['market']:9s} {p['selection']:4s}  "
              f"price={p['price']:.3f}  credits={p['credits']:4d}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", default="auto", help="Week number, or 'auto' for the next unplayed week.")
    parser.add_argument("--skip-refresh", action="store_true",
                         help="Skip the real ESPN line refresh (use whatever's already on disk).")
    parser.add_argument("--treat-as-not-started", action="append", default=[], metavar="GAME_ID",
                         help="Real, explicit override: include this real game_id (e.g. 2026_06_SEA_DEN) "
                              "even though it already kicked off, using ESPN's real closing line (re-served "
                              "post-game, not fabricated). Repeatable. Never applied silently - each use is "
                              "printed to the console.")
    args = parser.parse_args()
    week = real_current_week_2026() if args.week == "auto" else int(args.week)
    generate(week, skip_refresh=args.skip_refresh, force_include_ids=set(args.treat_as_not_started))
