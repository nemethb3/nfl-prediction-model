"""Real weekly settlement tracker for the juke picks competition - reads
every real output/juke_picks_{season}_week{NN}.json file (the canonical,
one-per-week "what we actually picked" record - overwritten in place each
time that week is regenerated, unlike the append-only internal forecast
log), settles each pick against games_2026.json's own real actual_*
result fields, and writes a real, regenerated-fresh-each-run summary of
credits won/lost per week.

Real, deliberate design choice: settlement is computed purely from the
PUBLIC output file's own fields (market/selection/line/assumed_price) -
decimal_odds is re-derived via juke.pricing.price_to_decimal_odds(price)
rather than cross-referencing the internal forecast log, so this script
works even if the internal log were ever unavailable, and so there's a
single, auditable settlement path using only what a third party could
also see.

A week whose game hasn't completed yet (real games_2026.json actual_
winner still null) is reported as "pending", not silently skipped or
guessed - this project's own standing rule against asserting results
that don't exist yet.
"""

import json
from pathlib import Path

from juke.pricing import price_to_decimal_odds
from run_provenance import utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
GAMES_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "games_2026.json"
RESULTS_PATH = OUTPUT_DIR / "juke_weekly_results.json"


def _load_games_by_id():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)
    return {g["id"]: g for g in games}


def _settle_pick(pick, game):
    """Real settlement from the real, already-completed game's own
    actual_home_score/actual_away_score/actual_winner - returns
    (result, pnl_credits) where result is "win"/"loss"/"push"/"pending"."""
    if game.get("actual_winner") is None:
        return "pending", 0.0

    decimal_odds = price_to_decimal_odds(pick["assumed_price"])
    credits = pick["credits"]

    if pick["market"] == "moneyline":
        if game["actual_winner"] == "TIE":
            return "push", 0.0
        won = game["actual_winner"] == pick["selection"]
    else:  # spread
        home_margin = game["actual_home_score"] - game["actual_away_score"]
        side_margin = home_margin if pick["selection"] == game["home_team"] else -home_margin
        # Real sportsbook convention (negative = favorite, see juke/
        # pricing.py's own sign-bug-fix docstring): the selected side
        # covers when its own real margin exceeds -line.
        net = side_margin + pick["line"]
        if net == 0:
            return "push", 0.0
        won = net > 0

    if won:
        return "win", credits * (decimal_odds - 1.0)
    return "loss", -float(credits)


def _week_summary(output_path, games_by_id):
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    submission = data["submission"]

    settled_picks = []
    for pick in data["picks"]:
        game = games_by_id.get(pick["event_id"])
        if game is None:
            result, pnl = "pending", 0.0
        else:
            result, pnl = _settle_pick(pick, game)
        settled_picks.append({**pick, "result": result, "pnl": round(pnl, 2)})

    decided = [p for p in settled_picks if p["result"] in ("win", "loss", "push")]
    scored = [p for p in decided if p["result"] != "push"]
    pending = [p for p in settled_picks if p["result"] == "pending"]

    credits_staked = sum(p["credits"] for p in settled_picks)
    credits_staked_decided = sum(p["credits"] for p in decided)
    net_credits_won = round(sum(p["pnl"] for p in decided), 2)

    return {
        "season": submission["season"],
        "week": submission["week"],
        "run_id": submission["run_id"],
        "generated_at": submission["generated_at"],
        "picks_total": len(settled_picks),
        "picks_decided": len(decided),
        "picks_pending": len(pending),
        "credits_staked": credits_staked,
        "credits_staked_decided": credits_staked_decided,
        "wins": sum(1 for p in scored if p["result"] == "win"),
        "losses": sum(1 for p in scored if p["result"] == "loss"),
        "pushes": sum(1 for p in decided if p["result"] == "push"),
        "net_credits_won": net_credits_won,
        "roi_pct": round(net_credits_won / credits_staked_decided * 100, 2) if credits_staked_decided else None,
        "fully_settled": len(pending) == 0,
        "picks": settled_picks,
    }


def track_juke_results():
    games_by_id = _load_games_by_id()
    week_files = sorted(OUTPUT_DIR.glob("juke_picks_*_week*.json"))
    if not week_files:
        print("No real output/juke_picks_*.json files found yet - nothing to track.")
        return None

    weeks = [_week_summary(p, games_by_id) for p in week_files]
    weeks.sort(key=lambda w: (w["season"], w["week"]))

    settled_weeks = [w for w in weeks if w["fully_settled"]]
    season_credits_staked = sum(w["credits_staked_decided"] for w in settled_weeks)
    season_net = round(sum(w["net_credits_won"] for w in settled_weeks), 2)

    result = {
        "generated_at": utc_now_iso(),
        "weeks": weeks,
        "season_to_date": {
            "weeks_fully_settled": len(settled_weeks),
            "weeks_pending": len(weeks) - len(settled_weeks),
            "credits_staked": season_credits_staked,
            "net_credits_won": season_net,
            "roi_pct": round(season_net / season_credits_staked * 100, 2) if season_credits_staked else None,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")

    for w in weeks:
        status = "FINAL" if w["fully_settled"] else f"PENDING ({w['picks_pending']}/{w['picks_total']} not yet played)"
        print(f"  Week {w['week']:2d} [{status:35s}] net={w['net_credits_won']:+8.2f}  "
              f"record={w['wins']}-{w['losses']}-{w['pushes']}  roi={w['roi_pct']}")
    print(f"\nSeason-to-date (fully settled weeks only): net={season_net:+.2f} credits, "
          f"ROI={result['season_to_date']['roi_pct']}%")

    return result


if __name__ == "__main__":
    track_juke_results()
