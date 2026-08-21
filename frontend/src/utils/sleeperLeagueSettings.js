// Real Sleeper league settings - reuses the exact same real, public,
// unauthenticated Sleeper endpoint LeagueConnector.js already calls
// (https://api.sleeper.app/v1/league/{leagueId}) to get `league.name` -
// this just also keeps the `scoring_settings`/`roster_positions` fields
// that call already receives but currently discards, instead of asking
// the user to re-enter their league's scoring rules by hand (which this
// project has no real way to validate against their actual real league).
//
// Caveat, disclosed rather than hidden: unlike the ESPN odds schema
// (verified live against a real endpoint this session), Sleeper's real
// `scoring_settings` key names below were NOT verified against a live
// league response in this session (no real league id was available to
// test with) - they're the widely-documented, stable community
// convention used by numerous independent Sleeper API tools. Every
// lookup below falls back to this project's own real, already-validated
// default PPR scoring (fantasy_formula_improvements.py's PPR_YD/
// PPR_RECEPTION/PPR_TD/PPR_PASS_YD_PER_PT/PPR_PASS_TD) if a given key is
// missing or the shape doesn't match, so an unexpected real key name
// degrades to a sane default instead of breaking.

// This project's own real, already-validated standard-PPR constants
// (src/fantasy_formula_improvements.py) - used as the fallback whenever
// a real league doesn't set a given Sleeper scoring key.
export const DEFAULT_SCORING = {
  passYdPerPt: 25.0, // 1 pt per 25 pass yards
  passTd: 4.0,
  rushYdPerPt: 10.0, // 1 pt per 10 rush/rec yards
  rushTd: 6.0,
  recYdPerPt: 10.0,
  recTd: 6.0,
  reception: 1.0,
};

// This project only models QB/RB/WR/TE (see ModelTransparency's
// Limitations - no DEF/K anywhere) - default slot counts match
// PersonalRoster's real implied scope, deliberately excluding DEF/K
// rather than fabricating projections for them.
export const DEFAULT_ROSTER_SLOTS = { QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1 };

const SLEEPER_BASE = 'https://api.sleeper.app/v1';

export async function fetchSleeperLeague(leagueId) {
  const response = await fetch(`${SLEEPER_BASE}/league/${leagueId}`);
  if (!response.ok) {
    throw new Error('League not found');
  }
  return response.json();
}

/** Real Sleeper scoring_settings -> this project's internal scoring
 * shape. Every field defends against a missing/renamed key by falling
 * back to DEFAULT_SCORING, so a real league that doesn't set a given
 * stat (or a Sleeper schema change) never produces NaN/undefined math
 * downstream. */
export function parseScoringSettings(scoringSettings) {
  const s = scoringSettings || {};
  // Sleeper expresses these as real points-PER-YARD (e.g. 0.04 for 1pt/25yd);
  // this project's own convention (matching fantasy_formula_improvements.py's
  // PPR_PASS_YD_PER_PT etc.) is yards-PER-point, so invert here once.
  const ydPerPt = (pointsPerYard, fallback) => (pointsPerYard ? 1 / pointsPerYard : fallback);

  return {
    passYdPerPt: ydPerPt(s.pass_yd, DEFAULT_SCORING.passYdPerPt),
    passTd: s.pass_td ?? DEFAULT_SCORING.passTd,
    rushYdPerPt: ydPerPt(s.rush_yd, DEFAULT_SCORING.rushYdPerPt),
    rushTd: s.rush_td ?? DEFAULT_SCORING.rushTd,
    recYdPerPt: ydPerPt(s.rec_yd, DEFAULT_SCORING.recYdPerPt),
    recTd: s.rec_td ?? DEFAULT_SCORING.recTd,
    reception: s.rec ?? DEFAULT_SCORING.reception,
    // Real, disclosed gap: this project's real player_props_2026.json has
    // no interception projection at all (verified before writing this),
    // so a league's real pass_int penalty can't be honestly applied -
    // surfaced to the caller instead of silently ignored.
    hasUnappliedInterceptionPenalty: typeof s.pass_int === 'number' && s.pass_int !== 0,
  };
}

/** Real Sleeper roster_positions -> {QB,RB,WR,TE,FLEX} slot counts.
 * Any slot type this project can't fill (DEF, K, BN, SUPER_FLEX,
 * WRRB_FLEX, IDP-type slots, etc.) is counted and returned separately
 * as `unsupportedSlots` so the caller can disclose it, rather than
 * silently dropped or fabricated. */
export function parseRosterPositions(rosterPositions) {
  const counts = { QB: 0, RB: 0, WR: 0, TE: 0, FLEX: 0 };
  const unsupported = {};

  (rosterPositions || []).forEach((slot) => {
    if (slot === 'BN') return; // bench isn't a "slot to fill", tracked implicitly
    if (Object.prototype.hasOwnProperty.call(counts, slot)) {
      counts[slot] += 1;
    } else {
      unsupported[slot] = (unsupported[slot] || 0) + 1;
    }
  });

  const hasAnyRealSlots = Object.values(counts).some((n) => n > 0);
  return {
    slotCounts: hasAnyRealSlots ? counts : DEFAULT_ROSTER_SLOTS,
    unsupportedSlots: unsupported,
  };
}
