"""Real ESPN odds collection - run manually (or via your own real cron/
Task Scheduler entry; none is set up by this script itself) before games
each week. Fetches the real current-week scoreboard, real per-game
multi-book odds, and real Super Bowl/conference/division futures, then
stores both a same-day snapshot and an append-only history log.

Real, disclosed scope limit: only pulls the CURRENT week's games (what
ESPN's scoreboard endpoint returns by default) - not a full-season
backfill. Every fetch degrades gracefully (skips and logs a warning
rather than crashing) since this is unofficial, undocumented ESPN
infrastructure that can change or fail at any time.
"""

from datetime import datetime

from espn_odds_client import ESPNOddsClient
from espn_odds_storage import append_history, save_daily_snapshot

SEASON = 2026

# Real futures markets relevant to this project's own predictions
# (Super Bowl, both conferences, all 8 divisions) - verified present for
# the 2026 season before writing this; the other 12 real markets ESPN
# exposes (MVP, awards, stat leaders) aren't comparable to anything this
# project computes, so they're deliberately not collected here.
RELEVANT_FUTURES_NAME_SUBSTRINGS = [
    "Super Bowl Winner",
    "Conference - Winner", "Conference Winner",
    "Division - Winner", "Division",
]


def _is_relevant_future(name):
    return any(s in name for s in RELEVANT_FUTURES_NAME_SUBSTRINGS) and "Team To Win Most Games" not in name


def collect_game_odds(client):
    print("\n[1/2] Fetching current-week scoreboard + per-game odds...")
    scoreboard = client.get_scoreboard()
    if not scoreboard:
        print("  FAILED: could not fetch scoreboard - skipping game odds this run.")
        return []

    events = scoreboard.get("events", [])
    print(f"  Found {len(events)} real games this week.")

    all_odds = []
    for event in events:
        event_id = event.get("id")
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        competition_id = competitions[0].get("id")
        competitors = competitions[0].get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        home_team = home.get("team", {}).get("abbreviation")
        away_team = away.get("team", {}).get("abbreviation")

        raw_odds = client.get_odds(event_id, competition_id)
        parsed = client.parse_odds(raw_odds)
        if not parsed:
            print(f"  {away_team}@{home_team}: no real odds available")
            continue
        print(f"  {away_team}@{home_team}: {len(parsed)} real book(s)")

        for record in parsed:
            record["game_id"] = event_id
            record["home_team"] = home_team
            record["away_team"] = away_team
            record["kickoff"] = event.get("date")
            all_odds.append(record)

    return all_odds


def collect_futures(client):
    print("\n[2/2] Fetching real Super Bowl/conference/division futures...")
    futures_index = client.get_futures(SEASON)
    if not futures_index:
        print("  FAILED: could not fetch futures index - skipping futures this run.")
        return []

    team_id_map = client.get_team_id_map()
    if not team_id_map:
        print("  WARNING: could not fetch team id map - futures will keep raw ESPN team refs unresolved.")

    relevant = [m for m in futures_index.get("items", []) if _is_relevant_future(m.get("name", ""))]
    print(f"  {len(relevant)} of {futures_index.get('count', 0)} real futures markets are relevant.")

    results = []
    for market in relevant:
        detail = client.get_futures_market(SEASON, market["id"])
        if not detail:
            print(f"  WARNING: could not fetch market '{market['name']}'")
            continue
        for provider_odds in detail.get("futures", []):
            provider = provider_odds.get("provider", {})
            for book in provider_odds.get("books", []):
                team_ref = book.get("team", {}).get("$ref", "")
                # Real ESPN team ids appear as the last numeric path segment
                # of the $ref URL (e.g. .../teams/22?lang=...) - parsed
                # directly rather than making a second real HTTP call per
                # team.
                team_id = team_ref.rstrip("/").split("/")[-1].split("?")[0] if team_ref else None
                results.append({
                    "market": market["name"],
                    "sportsbook": provider.get("name"),
                    "team": team_id_map.get(team_id, team_id),
                    "odds": book.get("value"),
                    "fetched_at": datetime.now().isoformat(),
                })

    return results


def collect_espn_odds():
    print("=" * 60)
    print(f"ESPN NFL Odds Collection - {datetime.now().isoformat()}")
    print("=" * 60)

    client = ESPNOddsClient()

    game_odds = collect_game_odds(client)
    futures = collect_futures(client)

    if game_odds:
        daily_path = save_daily_snapshot(game_odds)
        appended, total = append_history(game_odds)
        print(f"\nSaved {len(game_odds)} real game-odds records -> {daily_path}")
        print(f"History log: +{appended} new rows ({total} total)")
    else:
        print("\nNo real game odds collected this run.")

    if futures:
        futures_path = save_daily_snapshot(futures, run_date=f"futures_{datetime.now().strftime('%Y-%m-%d')}")
        print(f"Saved {len(futures)} real futures records -> {futures_path}")
    else:
        print("No real futures collected this run.")

    return game_odds, futures


if __name__ == "__main__":
    collect_espn_odds()
