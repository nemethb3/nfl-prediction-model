"""Real completed-game results ingester for the 2026 season.

Real context this exists for: as of 2026-09-10 the 2026 season has begun
(first game NE @ SEA, 2026-09-09). This project's 2026 pipeline was built
entirely preseason (see constants/seasons.js) - every generator re-runs
*predictions*; nothing consumed a completed game's real outcome. That gap
is what this script closes.

Real scope (deliberately minimal - see the "Ingest Completed Game
Results" task decision, 2026-09-10):

1. games_2026.json - fill the real, already-existing-but-null result
   fields (actual_home_score / actual_away_score / actual_winner /
   actual_spread_margin / did_we_predict_correctly) for every game
   nflreadpy.load_schedules([2026]) now has a real final score for.
   GameCard.js already renders these (its "Actual Result" / "Betting
   Outcome" sections key off actual_home_score !== null) - no frontend
   change is needed, and this does NOT touch the season-wide
   SEASON_HAS_RESULTS[2026] flag (that flag also gates the Accuracy
   Tracker / Weekly Summary / Betting Analysis tabs, whose 2026 data
   files are still legitimately null - flipping it would crash them).

2. fantasy_rankings_2026.json - fill the real, already-existing-but-null
   actual_ppr field for every ranked player on a team whose game is
   complete, computed with THIS PROJECT'S OWN PPR formulas
   (fantasy_formula_improvements._real_ppr for QB/WR/TE,
   fantasy_rb_formula._real_ppr for RB - imported directly, not
   re-derived) applied to nflreadpy.load_player_stats([2026]) real box
   scores. A ranked skill player on a completed game's roster with no
   real box-score row (inactive, or active but zero production) gets a
   real actual_ppr of 0.0 - either way their real fantasy value that
   week was 0.

3. accuracy_tier is deliberately LEFT NULL. It is defined
   (generate_fantasy_dashboard_data.py) as empirical per-position
   terciles of the real |actual - projected| distribution - a single
   completed game (~14 fantasy-relevant players, 2-4 per position) can't
   produce a meaningful distribution, so asserting a tier now would
   violate this project's "constants derived from a real sample, never
   asserted" rule.

Idempotent: re-running after more games complete only adds the new ones;
re-running with no new completed games is a real no-op (exit 0).

nflreadpy returns polars - converted to pandas explicitly at each load.
"""

import json
import sys
from pathlib import Path

import nflreadpy as nfl

import fantasy_formula_improvements as ffi
from fantasy_rb_formula import _real_ppr as _rb_real_ppr
from generation_timestamps import record_generation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
GAMES_PATH = DATA_DIR / "games_2026.json"
RANKINGS_PATH = DATA_DIR / "fantasy_rankings_2026.json"

SEASON = 2026


def _f(value):
    """Real numeric coercion - nflreadpy nulls arrive as NaN/None."""
    try:
        if value is None:
            return 0.0
        f = float(value)
        return 0.0 if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return 0.0


def _actual_ppr(position, stat_row):
    """This project's own real PPR formula, by position - imported, not
    re-implemented (see module docstring)."""
    py, ptd = _f(stat_row.get("passing_yards")), _f(stat_row.get("passing_tds"))
    pint = _f(stat_row.get("passing_interceptions"))
    ry, rtd = _f(stat_row.get("rushing_yards")), _f(stat_row.get("rushing_tds"))
    recy, rec = _f(stat_row.get("receiving_yards")), _f(stat_row.get("receptions"))
    rectd = _f(stat_row.get("receiving_tds"))

    if position == "QB":
        # ffi._real_ppr("QB", ...) reads passing_yards/passing_tds/interceptions/
        # rushing_yards/rushing_tds - nflreadpy calls the INT column
        # passing_interceptions, so map it here.
        row = {"passing_yards": py, "passing_tds": ptd, "interceptions": pint,
               "rushing_yards": ry, "rushing_tds": rtd}
        return float(ffi._real_ppr("QB", row))
    if position == "RB":
        return float(_rb_real_ppr(ry, recy, rec, rtd + rectd))
    # WR / TE - receiving-only, identical to ffi._real_ppr's non-QB branch
    row = {"receiving_yards": recy, "receptions": rec, "receiving_tds": rectd}
    return float(ffi._real_ppr(position, row))


def _load_completed_games():
    """Real 2026 games that have a real final score in nflreadpy."""
    sched = nfl.load_schedules([SEASON]).to_pandas()
    done = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    out = {}
    for _, g in done.iterrows():
        out[g["game_id"]] = {
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_score": int(g["home_score"]),
            "away_score": int(g["away_score"]),
            "week": int(g["week"]),
        }
    return out


def _load_week1_player_ppr(weeks):
    """Real per-player actual PPR for the given weeks, keyed by gsis id.
    Uses this project's own formula per the stat row's position group."""
    ps = nfl.load_player_stats([SEASON]).to_pandas()
    ps = ps[(ps["season_type"] == "REG") & (ps["week"].isin(list(weeks)))]
    by_gsis = {}
    for _, s in ps.iterrows():
        pid = s.get("player_id")
        pos = s.get("position")
        if not pid or pos not in ("QB", "RB", "WR", "TE"):
            continue
        by_gsis[pid] = {
            "ppr": round(_actual_ppr(pos, s), 1),
            "team": s.get("team"),
            "name": s.get("player_display_name"),
        }
    return by_gsis


def ingest():
    completed = _load_completed_games()
    if not completed:
        print("No completed 2026 games in nflreadpy yet - nothing to ingest.")
        return 0

    games = json.loads(GAMES_PATH.read_text(encoding="utf-8"))
    rankings = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))

    completed_weeks = {g["week"] for g in completed.values()}
    completed_teams = set()
    for g in completed.values():
        completed_teams.add(g["home_team"])
        completed_teams.add(g["away_team"])

    # ---- games_2026.json ----
    games_updated = 0
    for game in games:
        result = completed.get(game["id"])
        if result is None:
            continue
        home, away = game["home_team"], game["away_team"]
        hs, as_ = result["home_score"], result["away_score"]
        if hs > as_:
            winner, margin = home, hs - as_
        elif as_ > hs:
            winner, margin = away, hs - as_  # signed, home perspective (negative = away won)
        else:
            winner, margin = "TIE", 0

        our_spread = game.get("our_spread")
        if winner == "TIE" or our_spread in (None, 0):
            predicted_correct = None
        else:
            predicted_winner = home if our_spread > 0 else away
            predicted_correct = predicted_winner == winner

        before = (game.get("actual_home_score"), game.get("did_we_predict_correctly"))
        game["actual_home_score"] = hs
        game["actual_away_score"] = as_
        game["actual_winner"] = winner
        game["actual_spread_margin"] = margin
        game["did_we_predict_correctly"] = predicted_correct
        if before != (hs, predicted_correct):
            games_updated += 1

        total = hs + as_
        pred_total = game.get("predicted_total_value")
        spread_note = ""
        if our_spread not in (None, 0):
            # model's predicted home margin vs real home margin
            pred_home_margin = our_spread
            spread_note = f", spread err {margin - pred_home_margin:+.1f} (home margin)"
        tot_note = f", total {total} vs pred {pred_total:.1f} ({total - pred_total:+.1f})" if pred_total else ""
        mark = "OK " if predicted_correct else ("-- " if predicted_correct is None else "XX ")
        print(f"  {mark}{away} {as_} @ {home} {hs}  ->  {winner}{spread_note}{tot_note}")

    # ---- fantasy_rankings_2026.json ----
    ppr_by_gsis = _load_week1_player_ppr(completed_weeks)
    matched = zeroed = 0
    movers = []
    for row in rankings:
        gsis = row["id"].split("_w")[0]
        stat = ppr_by_gsis.get(gsis)
        if stat is not None:
            actual = stat["ppr"]
            matched += 1
        elif row.get("team") in completed_teams:
            actual = 0.0
            zeroed += 1
        else:
            continue  # team hasn't played - leave actual_ppr null
        row["actual_ppr"] = actual
        proj = row.get("projected_ppr")
        if proj is not None:
            movers.append((row["name"], row["position"], row["team"], proj, actual, actual - proj))

    print(f"\n  fantasy_rankings: {matched} matched to a real box score, "
          f"{zeroed} zeroed (completed game, no box-score row)")
    movers.sort(key=lambda m: m[5])
    print("\n  Biggest under-performers vs projection:")
    for name, pos, team, proj, act, diff in movers[:5]:
        print(f"    {name:22s} {pos} {team}  proj {proj:5.1f} -> actual {act:5.1f}  ({diff:+.1f})")
    print("  Biggest over-performers vs projection:")
    for name, pos, team, proj, act, diff in list(reversed(movers))[:5]:
        print(f"    {name:22s} {pos} {team}  proj {proj:5.1f} -> actual {act:5.1f}  ({diff:+.1f})")

    GAMES_PATH.write_text(json.dumps(games, indent=2), encoding="utf-8")
    record_generation("games_2026")
    RANKINGS_PATH.write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    record_generation("fantasy_rankings_2026")

    print(f"\nWrote {games_updated} game result(s) -> {GAMES_PATH.name}")
    print(f"Wrote {matched + zeroed} actual_ppr value(s) -> {RANKINGS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(ingest())
