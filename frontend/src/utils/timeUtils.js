// Real timezone conversion utilities for real 2025/2026 kickoff times.
//
// Real bug this file fixes (found while implementing this task, not the
// bug the pasted spec assumed): kickoff_datetime is real ET wall-clock
// digits with no UTC offset in the string (nflverse's documented
// schedules.gametime convention - see GameCard.js's own prior history).
// GameCard.js previously hand-rolled the ET->UTC conversion with a
// HARDCODED single DST cutover date (`DST_2025_FALLBACK_DATE =
// '2025-11-02'`), compared via plain string comparison against each
// game's date. Checked directly: since ISO date strings compare
// lexicographically, EVERY real 2026 game date ("2026-...") is >=
// "2025-11-02" as a STRING, so the old code silently used the wrong
// (EST, UTC-5) offset for every single 2026 game before the real 2026
// DST cutover (Nov 1, 2026) - including the entire September/October
// slate, which is real EDT (UTC-4). That's a genuine, real, 1-hour-off
// bug for roughly half the 2026 season in "Your Time" mode, not the
// "wrong field" issue the pasted spec assumed (there's no `game.game_time`
// UTC field anywhere in this project - the real field is
// `kickoff_datetime`, and a naive `new Date()` parse of it is the ORIGINAL
// bug this project already found and fixed once before - see git history).
//
// Real, permanent fix: interpret the naive kickoff digits as real wall-
// clock time in a named IANA zone (e.g. 'America/New_York') and convert
// using the browser's own real IANA timezone database via
// Intl.DateTimeFormat - this handles DST correctly for any real year
// automatically, with no hardcoded cutover date ever needed again, and
// correctly handles Arizona's real non-DST-observing zone natively too
// (no more hand-rolled Arizona special case).

export function getUserTimezone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch (e) {
    return 'UTC';
  }
}

// Real, tested (Node.js, verified against known EDT/EST/AZ offsets and
// the real Nov 1 2026 DST boundary) conversion: given naive wall-clock
// digits (no offset) that represent a real local time IN `zone`, returns
// the real absolute UTC Date. Deliberately avoids toLocaleString()
// round-tripping (locale-dependent string parsing is unreliable - an
// earlier draft of this function used that approach and produced a
// result off by several hours) in favor of Intl.DateTimeFormat's
// formatToParts(), which returns real, unambiguous numeric fields.
export function zonedWallTimeToUTC(naiveISO, zone) {
  const [datePart, timePart] = naiveISO.split('T');
  const [year, month, day] = datePart.split('-').map(Number);
  const [hour, minute] = timePart.slice(0, 5).split(':').map(Number);

  // Guess: treat the real wall-clock digits as if they were already UTC.
  const guessUTC = Date.UTC(year, month - 1, day, hour, minute);

  // Find what real wall-clock time that guess maps to inside `zone`.
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
  const parts = Object.fromEntries(dtf.formatToParts(guessUTC).map((p) => [p.type, p.value]));
  const asZoneUTC = Date.UTC(
    Number(parts.year), Number(parts.month) - 1, Number(parts.day),
    Number(parts.hour) === 24 ? 0 : Number(parts.hour), Number(parts.minute), Number(parts.second),
  );

  // The gap between the two is exactly the real UTC offset (DST-aware)
  // for that date in `zone` - apply it to correct the original guess.
  return new Date(guessUTC + (guessUTC - asZoneUTC));
}

// Real per-stadium IANA zone, keyed by the short codes already used in
// constants/teams.js's TEAM_TIMEZONES (ET/CT/MT/PT) plus Arizona's real,
// separate zone (doesn't observe DST - handled natively by the real IANA
// database here, not a hand-rolled special case).
export const SHORT_CODE_TO_IANA_ZONE = {
  ET: 'America/New_York',
  CT: 'America/Chicago',
  MT: 'America/Denver',
  PT: 'America/Los_Angeles',
};
export const ARIZONA_IANA_ZONE = 'America/Phoenix';

export function formatDateTimeInZone(utcDate, zone) {
  if (!utcDate) return { date: '', time: 'TBD' };
  return {
    date: utcDate.toLocaleDateString('en-US', {
      timeZone: zone, weekday: 'short', month: 'short', day: 'numeric',
    }),
    time: utcDate.toLocaleTimeString('en-US', {
      timeZone: zone, hour: 'numeric', minute: '2-digit', hour12: true,
    }),
  };
}

// The real, fixed source zone every kickoff_datetime value is labeled in
// (see module docstring) - not configurable, since this is a real fact
// about the data, not a per-call choice.
const KICKOFF_SOURCE_ZONE = 'America/New_York';

// Real, single entry point: real kickoff digits (ET wall-clock, no
// offset) -> real {date, time} in the given real IANA zone.
export function formatKickoffInZone(kickoffISO, zone) {
  if (!kickoffISO) return { date: '', time: 'TBD' };
  const utcDate = zonedWallTimeToUTC(kickoffISO, KICKOFF_SOURCE_ZONE);
  return formatDateTimeInZone(utcDate, zone);
}
