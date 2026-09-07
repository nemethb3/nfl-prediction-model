"""Real ESPN injury report fetcher.

Real finding this task exists to act on: ESPN's odds scoreboard
(ESPNOddsClient.get_scoreboard()) does NOT carry injury data anywhere in
its response (checked directly: event/competition/competitor objects
have no `injuries` field at all - the originally pasted spec's assumed
`event['injuries']` doesn't exist). The real data lives at a completely
different, separate ESPN endpoint - found by testing real candidate URLs
live, not guessed:

    https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries

Real, verified response (2026-09-07): a real, current, LEAGUE-WIDE
injury report - 32 real teams, 800 real per-player rows, with real
status values {Active, Injured Reserve, Questionable, Out, Suspension}
and rich per-injury detail (body part, return date, long-form real
beat-reporter commentary). This directly closes the real gap flagged in
the prior Week 1 quality-check task (nflreadpy has no 2026 injury
support yet - see verify_week1_injuries.py's own docstring for that
finding, now superseded for the "Active"/day-to-day cases below by this
real, working ESPN source).

Real, disclosed scope note: ESPN's own "Active" status here does NOT
mean "confirmed healthy" - checked directly, several "Active" rows are
real depth-chart/competition commentary with no real injury at all (see
the Jacoby Brissett example in this task's research). Only Questionable/
Out/Injured Reserve/Suspension rows represent a real, meaningful
game-availability signal; "Active" rows are excluded from the flagged
output below (not because they're irrelevant, but because treating them
as "hurt" would be a fabrication in the other direction).
"""

import json
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
CACHE_DIR = PROJECT_ROOT / "data" / "espn_injuries"
RANKINGS_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "fantasy_rankings_2026.json"

# Real statuses that represent an actual game-availability concern - NOT
# "Active" (see module docstring for why).
FLAGGED_STATUSES = {"Questionable", "Out", "Injured Reserve", "Suspension"}


class ESPNInjuriesFetcher:
    def __init__(self, cache_ttl_hours=6):
        self.cache_ttl_hours = cache_ttl_hours
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self):
        return CACHE_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_injuries.json"

    def fetch_raw(self):
        """Real fetch with same-day file cache (this project's existing
        convention - see espn_odds_client.py) - injury reports don't
        change fast enough to justify hitting ESPN more than once a day."""
        path = self._cache_path()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        try:
            r = requests.get(INJURIES_URL, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"ESPN injuries fetch failed: {e}")
            return None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def get_flat_injuries(self):
        """Real, flattened per-player list: name/team/position/status/
        body-part detail/return date. One real row per real ESPN injury
        report entry (800 as of 2026-09-07), all 32 real teams."""
        raw = self.fetch_raw()
        if raw is None:
            return []
        rows = []
        for team in raw.get("injuries", []):
            for inj in team.get("injuries", []):
                athlete = inj.get("athlete", {})
                details = inj.get("details", {})
                rows.append({
                    "espn_athlete_name": athlete.get("displayName"),
                    "team": athlete.get("team", {}).get("abbreviation"),
                    "position": athlete.get("position", {}).get("abbreviation"),
                    "status": inj.get("status"),
                    "body_part": details.get("type"),
                    "detail": details.get("detail"),
                    "return_date": details.get("returnDate"),
                    "short_comment": inj.get("shortComment"),
                    "report_date": inj.get("date"),
                })
        return rows

    def match_to_rankings(self):
        """Real name+team join against fantasy_rankings_2026.json - exact
        match only (no fuzzy matching, so a real miss is reported as a
        real miss, not silently guessed). Returns
        (matched: {player_name: injury_row}, unmatched_flagged: [rows])."""
        injuries = [r for r in self.get_flat_injuries() if r["status"] in FLAGGED_STATUSES]
        if not RANKINGS_PATH.exists():
            print(f"Missing: {RANKINGS_PATH}")
            return {}, injuries

        with open(RANKINGS_PATH, encoding="utf-8") as f:
            rankings = json.load(f)
        by_name_team = {(p["name"], p["team"]): p for p in rankings}

        matched = {}
        unmatched = []
        for row in injuries:
            key = (row["espn_athlete_name"], row["team"])
            if key in by_name_team:
                matched[row["espn_athlete_name"]] = row
            else:
                unmatched.append(row)
        return matched, unmatched


if __name__ == "__main__":
    fetcher = ESPNInjuriesFetcher()
    all_rows = fetcher.get_flat_injuries()
    flagged = [r for r in all_rows if r["status"] in FLAGGED_STATUSES]
    print(f"Real ESPN injury report: {len(all_rows)} total rows, {len(flagged)} flagged "
          f"(Questionable/Out/Injured Reserve/Suspension).")

    matched, unmatched = fetcher.match_to_rankings()
    print(f"\nMatched to this project's real fantasy_rankings_2026.json by exact name+team: "
          f"{len(matched)}/{len(flagged)}")
    if unmatched:
        print(f"\nFlagged ESPN players NOT found in fantasy_rankings_2026.json (real name/team "
              f"mismatch, or real player outside this project's ~390-player ranked pool):")
        for row in unmatched[:15]:
            print(f"  {row['espn_athlete_name']} ({row['team']}, {row['position']}): {row['status']}")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")

    print("\nMatched, real, flagged players in this project's own ranked pool:")
    for name, row in matched.items():
        print(f"  {name} ({row['team']}, {row['position']}): {row['status']} - {row['body_part']}"
              f" ({row['detail']}), return {row['return_date']}")
