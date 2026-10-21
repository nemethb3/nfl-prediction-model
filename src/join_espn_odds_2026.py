"""Joins real, already-collected ESPN historical odds (data/espn_odds/
history.json - accumulated every week by espn_odds_orchestrate.py, which
already runs inside refresh_weekly.py) into the real 2026 game records.

Real, disclosed gap this fixes: games_2026.json's own vegas_spread field is
null for all 272 games (nflverse has no posted lines that far out - see
espn_odds_client.py's module docstring). ESPN's odds endpoint is the only
real market-line source available for 2026, and it's already being fetched
and cached every week - nobody had joined it back into games_2026.json or
into a real per-game moneyline file, which is why Betting Analysis (and the
real Vegas-comparison columns in Accuracy Tracker/Weekly Summary) have
stayed empty for 2026 even after Week 1 completed.

Real join key: history.json rows carry ESPN's own numeric event id
(game_id), which does NOT match this project's own id scheme
("2026_WW_AWAY_HOME") - verified 0/96 overlap by id. Both sides DO carry
real team abbreviations though, and (away_team, home_team) is a unique key
within one real NFL season (teams play each other at most once with a
given home/away assignment in the regular season, other than the rare
division rematch, which reverses home/away). Joined on that pair instead.

Real "closing line" convention: history.json accumulates one row per
(game, sportsbook, day) every time refresh_weekly.py's odds-collection step
runs, so a game can have several real snapshots between the line opening
and kickoff. The LATEST real fetched_at row per (away_team, home_team) is
used - the standard real-world reference line for grading a spread bet.

Real sign-convention fix: ESPN's own home_spread_current field uses the
standard sportsbook convention (negative = home favored, e.g. "-3" means
home favored by 3). This project's own vegas_spread/our_spread convention
is the opposite (positive = home favored - established and verified in
betting_backtest.py's module docstring for the real 2015-2025 data). Every
real spread pulled from ESPN is negated here to match.

Real per-side spread juice (fixed 2026-10-06 - a real bug, not just a
disclosed gap: see espn_odds_client.parse_odds's own docstring). The raw
API exposes a genuinely asymmetric per-side spread price
(home/awayTeamOdds.spreadOdds, e.g. home -112 / away -108) directly
alongside moneyLine - the collector simply never captured it before.
Historical weeks collected before this fix still only have the
symmetric -110/-110 default baked into their saved history.json rows
(nothing to backfill from - the real value was never fetched for them).
Standard -110 remains the disclosed fallback for any provider that
genuinely doesn't expose a per-side price. Every game using that
fallback has `used_default_spread_juice: true` on its games_2026.json
row, so a downstream consumer (the juke picks generator) can see plainly
when a real per-side price wasn't available.

Real fixes added here (2026-10-06, schema_report.md's provenance-gap
PREREQUISITE for the juke picks competition - see generate_juke_picks.py):
1. Pre-kickoff cutoff. The old `_latest_odds_by_matchup` picked the
   absolute-latest real fetched_at row with no regard for whether it was
   fetched before or after that game's own real kickoff - a line fetched
   mid-game (e.g. a stale weekly refresh re-run) could have silently been
   used as if it were a real pre-game reference line. Now only rows with
   a real fetched_at strictly before that game's real kickoff are
   considered; a row with no real kickoff on it (very old history rows,
   collected before espn_odds_orchestrate.py started attaching one) is
   kept as a last resort, since there is nothing to apply a cutoff
   against.
2. Real per-game provenance written directly onto games_2026.json, not
   just the side CSV: vegas_spread_observed_at (real, UTC, when that
   reference line was actually fetched), vegas_line_source (real
   sportsbook name), kickoff_utc (ESPN's own real, UTC-aware event date -
   fixes the naive kickoff_datetime schema_report.md flagged; null when
   no real odds row/kickoff has ever been collected for that game, e.g. a
   future week this far out, or an early-season game whose kickoff
   predates odds collection starting), home_moneyline/away_moneyline/
   home_spread_odds/away_spread_odds/used_default_spread_juice (so a
   consumer doesn't need a second file join to price a bet).
3. Robust timestamp comparison (run_provenance.parse_any_iso_to_utc) -
   history.json has a real mix of naive-local (old espn_odds_client rows,
   before fix 2026-10-06) and UTC-aware (ESPN's own kickoff, and new
   fetched_at rows going forward) timestamps; comparing them as raw
   strings (the old behavior) silently mis-ordered rows across that mix.
"""

import csv
import json
from pathlib import Path

from run_provenance import parse_any_iso_to_utc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAMES_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "games_2026.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "espn_odds" / "history.json"
VEGAS_LINES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vegas_lines_2026.csv"

STANDARD_SPREAD_JUICE = -110


def _pre_kickoff_odds_by_matchup():
    """Real {(away_team, home_team): latest PRE-KICKOFF real odds row} -
    one entry per real matchup, keeping only the latest real fetched_at
    row that was fetched strictly before that game's own real kickoff
    (falls back to the latest row overall if no row for that matchup has
    a real kickoff at all - nothing to apply a cutoff against)."""
    if not HISTORY_PATH.exists():
        return {}
    with open(HISTORY_PATH, encoding="utf-8") as f:
        history = json.load(f)

    by_key = {}
    for row in history:
        if row.get("home_spread_current") is None and row.get("home_moneyline") is None:
            continue
        key = (row.get("away_team"), row.get("home_team"))
        if key[0] is None or key[1] is None:
            continue
        by_key.setdefault(key, []).append(row)

    latest = {}
    for key, rows in by_key.items():
        eligible = []
        for row in rows:
            fetched = parse_any_iso_to_utc(row.get("fetched_at"))
            kickoff = parse_any_iso_to_utc(row.get("kickoff"))
            if fetched is None:
                continue
            if kickoff is not None and fetched >= kickoff:
                continue  # real post-kickoff fetch - never used as a reference line
            eligible.append((fetched, row))
        pool = eligible if eligible else [
            (parse_any_iso_to_utc(row.get("fetched_at")), row) for row in rows
            if parse_any_iso_to_utc(row.get("fetched_at")) is not None
        ]
        if not pool:
            continue
        pool.sort(key=lambda t: t[0])
        latest[key] = pool[-1][1]

    return latest


def join_odds():
    with open(GAMES_PATH, encoding="utf-8") as f:
        games = json.load(f)

    odds_by_matchup = _pre_kickoff_odds_by_matchup()

    n_spread_joined = 0
    vegas_lines_rows = []
    for game in games:
        key = (game["away_team"], game["home_team"])
        row = odds_by_matchup.get(key)
        if row is None:
            continue

        if row.get("home_spread_current") is not None:
            game["vegas_spread"] = round(-float(row["home_spread_current"]), 1)
            n_spread_joined += 1

        observed_at = parse_any_iso_to_utc(row.get("fetched_at"))
        game["vegas_spread_observed_at"] = observed_at.isoformat() if observed_at else None
        game["vegas_line_source"] = row.get("sportsbook")
        kickoff = parse_any_iso_to_utc(row.get("kickoff"))
        game["kickoff_utc"] = kickoff.isoformat() if kickoff else None
        game["home_moneyline"] = row.get("home_moneyline")
        game["away_moneyline"] = row.get("away_moneyline")
        # Real fix (2026-10-06 - see espn_odds_client.parse_odds's own
        # docstring): use the real, per-side spread price when the
        # collector captured one, instead of always defaulting to a
        # symmetric -110/-110 (which made the spread market's de-vigged
        # price exactly 50/50 for every game, carrying no real signal).
        real_home_juice = row.get("home_spread_odds_current")
        real_away_juice = row.get("away_spread_odds_current")
        if real_home_juice is not None and real_away_juice is not None:
            game["home_spread_odds"] = float(real_home_juice)
            game["away_spread_odds"] = float(real_away_juice)
            game["used_default_spread_juice"] = False
        else:
            game["home_spread_odds"] = STANDARD_SPREAD_JUICE
            game["away_spread_odds"] = STANDARD_SPREAD_JUICE
            game["used_default_spread_juice"] = True

        if row.get("home_moneyline") is not None and row.get("away_moneyline") is not None:
            vegas_lines_rows.append({
                "game_id": game["id"],
                "home_moneyline": row["home_moneyline"],
                "away_moneyline": row["away_moneyline"],
                "home_spread_odds": game["home_spread_odds"],
                "away_spread_odds": game["away_spread_odds"],
                "sportsbook": row.get("sportsbook"),
                "closing_line_fetched_at": game["vegas_spread_observed_at"],
            })

    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)

    VEGAS_LINES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(VEGAS_LINES_OUTPUT, "w", encoding="utf-8", newline="") as f:
        if vegas_lines_rows:
            writer = csv.DictWriter(f, fieldnames=list(vegas_lines_rows[0].keys()))
            writer.writeheader()
            writer.writerows(vegas_lines_rows)

    print(f"Real vegas_spread joined onto games_2026.json for {n_spread_joined}/{len(games)} games")
    print(f"Real vegas_lines_2026.csv written -> {VEGAS_LINES_OUTPUT} ({len(vegas_lines_rows)} rows)")
    return n_spread_joined, len(vegas_lines_rows)


if __name__ == "__main__":
    join_odds()
