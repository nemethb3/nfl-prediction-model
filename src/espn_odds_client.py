"""ESPN's unofficial NFL odds API client - real, fragile, undocumented
infrastructure (ESPN shut down its official developer program in 2018;
every endpoint here is reverse-engineered, no auth, no published rate
limit, and can change without notice).

Real schema verified directly against the live API before writing this
(NOT guessed from the pasted spec, which assumed a `displayOdds`/`type`
list structure that doesn't exist): each odds item is a flat object with
`spread`/`overUnder`/`overOdds`/`underOdds` plus `awayTeamOdds`/
`homeTeamOdds` sub-objects, each carrying `moneyLine` and `open`/`current`
snapshots of `pointSpread`/`spread`/`moneyLine`. Also verified: a single
real game can have as few as ONE provider in the response (DraftKings
only, in the test game used to verify this), not necessarily all 6 books
named in the original spec - real per-game book coverage varies and is
never assumed to be complete.

Real, disclosed reason this is worth having despite the fragility: this
project's real 2026 game predictions (games_2026.json) currently have
vegas_spread=null for all 272 games - nflverse has no real posted lines
that far out. ESPN's odds endpoint is currently the only real market-line
source available for the season this project is actively predicting, not
merely "extra book depth" on top of an existing Vegas line.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE_API = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "espn_odds" / "cache"

REQUEST_TIMEOUT_S = 15
CACHE_TTL_HOURS = 6
MAX_RETRIES_ON_429 = 1
RETRY_BACKOFF_S = 60


class ESPNOddsClient:
    """Fetches real ESPN odds/futures data with file-based caching and
    graceful degradation (every fetch method returns None on failure
    instead of raising, so a broken/changed ESPN endpoint can't crash a
    caller that's willing to handle a missing value)."""

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, cache_ttl_hours=CACHE_TTL_HOURS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours

    def _cache_path(self, cache_key):
        today = datetime.now().strftime("%Y-%m-%d")
        safe_key = cache_key.replace("/", "_").replace(":", "").replace("?", "_")
        return self.cache_dir / f"{today}_{safe_key}.json"

    def _load_cache(self, cache_key):
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            cached_at = datetime.fromisoformat(record["cached_at"])
            if cached_at > datetime.now() - timedelta(hours=self.cache_ttl_hours):
                return record["response"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None
        return None

    def _save_cache(self, cache_key, data):
        path = self._cache_path(cache_key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cached_at": datetime.now().isoformat(), "response": data}, f)
        except OSError as e:
            print(f"  WARNING: ESPN cache write failed for {cache_key}: {e}")

    def _get(self, url, cache_key, use_cache=True, _retries_left=MAX_RETRIES_ON_429):
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as e:
            print(f"  WARNING: ESPN request failed for {cache_key}: {e}")
            return None

        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"  WARNING: ESPN returned non-JSON for {cache_key}: {e}")
                return None
            self._save_cache(cache_key, data)
            return data

        if response.status_code == 429 and _retries_left > 0:
            print(f"  WARNING: ESPN rate-limited ({cache_key}) - waiting {RETRY_BACKOFF_S}s and retrying once")
            time.sleep(RETRY_BACKOFF_S)
            return self._get(url, cache_key, use_cache=False, _retries_left=_retries_left - 1)

        print(f"  WARNING: ESPN returned {response.status_code} for {cache_key}")
        return None

    def get_scoreboard(self, week=None, season=None, seasontype=2):
        """Real NFL scoreboard - includes a real, single highest-priority
        book's odds inline per game (e.g. DraftKings), useful as a quick
        check but NOT the full multi-book breakdown (use get_odds() for
        that).

        Real, serious bug found and fixed here (2026-10-06): calling this
        with no params does NOT reliably return the real CURRENT week -
        verified live the week this was found, ESPN's own default
        scoreboard still reported week=4 as current on 2026-10-06, a full
        day after week 4 finished and 2 days before week 5's first game,
        silently returning the WRONG week's games/odds to any caller that
        trusted the default. Pass explicit week/season/seasontype (as
        espn_odds_orchestrate.py now does, using this project's own real
        current_week_2026.real_current_week_2026()) to get the real
        intended week deterministically."""
        url = f"{SITE_API}/scoreboard"
        if week is None:
            return self._get(url, "scoreboard")
        params = {"week": week, "seasontype": seasontype, "year": season}
        cache_key = f"scoreboard_wk{week}_{season}_{seasontype}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as e:
            print(f"  WARNING: ESPN request failed for {cache_key}: {e}")
            return None
        if response.status_code != 200:
            print(f"  WARNING: ESPN returned {response.status_code} for {cache_key}")
            return None
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"  WARNING: ESPN returned non-JSON for {cache_key}: {e}")
            return None
        self._save_cache(cache_key, data)
        return data

    def get_odds(self, event_id, competition_id):
        """Real per-game odds, one item per provider that has a real
        posted line for this game right now - real, verified count for
        the game this module was built against: 1 (not the 6 books named
        in the original spec; real per-game coverage varies)."""
        url = f"{CORE_API}/events/{event_id}/competitions/{competition_id}/odds"
        return self._get(url, f"odds_{event_id}")

    def get_line_movement(self, event_id, competition_id, provider_id):
        """Real full intraday line-movement timeline for one provider.
        Only needed for the full timeline - get_odds()'s own real
        open/current snapshot per provider already answers "opening vs.
        current" without this extra call."""
        url = f"{CORE_API}/events/{event_id}/competitions/{competition_id}/odds/{provider_id}/history/0/movement"
        return self._get(url, f"movement_{event_id}_{provider_id}")

    def get_futures(self, season):
        """Real season futures index (Super Bowl winner, all 8 division
        winners, both conference winners, MVP/awards, stat leaders - 23
        real markets verified present for the 2026 season). Returns the
        real market INDEX (id + name per market), not odds themselves -
        each market's own real book/team odds live behind its own $ref,
        fetched separately by get_futures_market()."""
        url = f"{CORE_API}/seasons/{season}/futures"
        return self._get(url, f"futures_{season}")

    def get_futures_market(self, season, market_id):
        """Real odds for one specific futures market (e.g. market 1561,
        real 'NFL - Super Bowl Winner', verified present for 2026)."""
        url = f"{CORE_API}/seasons/{season}/futures/{market_id}"
        return self._get(url, f"futures_{season}_{market_id}")

    def get_team_id_map(self):
        """Real {espn_team_id: team_abbreviation} map (verified: ESPN's
        numeric team ids, e.g. '22' for ARI, are NOT this project's own
        2/3-letter codes - futures markets reference teams by numeric id
        via a $ref, not by abbreviation, so this map is needed to join
        futures odds back to this project's own team-keyed data)."""
        url = f"{SITE_API}/teams?limit=40"
        data = self._get(url, "teams")
        if not data:
            return {}
        try:
            teams = data["sports"][0]["leagues"][0]["teams"]
        except (KeyError, IndexError):
            return {}
        return {t["team"]["id"]: t["team"]["abbreviation"] for t in teams}

    @staticmethod
    def parse_odds(raw_odds):
        """Real, verified parser - matches the actual live schema (flat
        spread/overUnder/overOdds/underOdds + awayTeamOdds/homeTeamOdds
        each with a real open/current snapshot), NOT the pasted spec's
        guessed `displayOdds`/`type` structure, which doesn't exist.
        Returns one dict per real provider present in the response - do
        not assume any fixed set of books.

        Real bug found and fixed here (2026-10-06): `awayTeamOdds`/
        `homeTeamOdds` each carry a real, per-side CURRENT spread price
        (`spreadOdds`, e.g. away -108 / home -112 - genuinely asymmetric,
        verified live) directly alongside `moneyLine` - this parser never
        captured it, so every real spread bet built on top of this data
        (join_espn_odds_2026.py, betting_backtest_2026.py, and the juke
        picks engine) silently used a hardcoded symmetric -110/-110
        instead of the real price that was sitting in the same response.
        That made the spread market's de-vigged "price" exactly 50/50 by
        construction for every single spread bet, regardless of the real
        line - not a market signal at all. Captured here as
        home_spread_odds_current/away_spread_odds_current; a provider
        that genuinely doesn't expose it still falls back to the
        disclosed -110 default downstream (join_espn_odds_2026.py)."""
        parsed = []
        if not raw_odds or not raw_odds.get("items"):
            return parsed

        for book in raw_odds["items"]:
            provider = book.get("provider", {})
            home = book.get("homeTeamOdds", {})
            away = book.get("awayTeamOdds", {})

            def _american(side, snapshot, field):
                return side.get(snapshot, {}).get(field, {}).get("american")

            parsed.append({
                "sportsbook": provider.get("name"),
                "provider_id": provider.get("id"),
                "total": book.get("overUnder"),
                "over_odds": book.get("overOdds"),
                "under_odds": book.get("underOdds"),
                "home_moneyline": home.get("moneyLine"),
                "away_moneyline": away.get("moneyLine"),
                "home_spread_current": _american(home, "current", "pointSpread"),
                "away_spread_current": _american(away, "current", "pointSpread"),
                "home_spread_open": _american(home, "open", "pointSpread"),
                "away_spread_open": _american(away, "open", "pointSpread"),
                "home_spread_odds_current": home.get("spreadOdds"),
                "away_spread_odds_current": away.get("spreadOdds"),
                "home_moneyline_open": _american(home, "open", "moneyLine"),
                "away_moneyline_open": _american(away, "open", "moneyLine"),
                "home_favorite": home.get("favorite"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        return parsed
