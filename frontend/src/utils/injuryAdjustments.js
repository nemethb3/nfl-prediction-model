// Real, shared helpers for the ESPN-sourced injury overlay
// (injury_adjustments_2026.json - see generate_injury_adjustments_2026.py).
// Pulled out into one shared module rather than re-implemented per
// component (as happened the first time this was built, in
// FantasyRankings.js and GamePredictions.js independently) so every real
// consumer - Fantasy Rankings, Game Predictions, Suggested Lineups,
// Trade Analyzer, Personal Roster - treats "confirmed out" identically.
//
// Real, deliberate scope boundary, unchanged from when this overlay was
// first built: only confirmed Out/Injured Reserve players are zeroed.
// Questionable players are flagged (via getInjuryAdjustmentMap) but never
// numerically adjusted anywhere - no real, fit model exists in this
// project for how much to discount a game-status designation. Game-level
// predictions (GameCard.js's spread/win probability) are also
// deliberately NOT touched by this overlay for the same reason: no real
// model exists for the point value of one player at the team-Elo level.

// Real, verified gap (checked directly against the actual data): this
// project's two real name sources for the same player don't always agree
// on suffixes - e.g. sleeper_id_mapping.json has "Dont'e Thornton" while
// fantasy_rankings_2026.json/injury_adjustments_2026.json have "Dont'e
// Thornton Jr." (a real, independent-crosswalk inconsistency, not a typo
// introduced here). An exact-string join would silently miss that real
// player in any component keyed off sleeperIdMapping (PersonalRoster.js,
// SuggestedLineups.js). Normalizing away common suffixes before matching
// closes that real gap generally, not just for this one case.
function normalizeKey(name, team) {
  const stripped = (name || '').replace(/\s+(Jr\.?|Sr\.?|II|III|IV)$/i, '').trim().toLowerCase();
  return `${stripped}|${team}`;
}

/** Real Set of normalized "name|team" keys for confirmed Out/Injured
 * Reserve players, gated on the injury data's own real `week` field so a
 * future multi-week season doesn't apply stale Week 1 news to a later
 * week. */
export function getConfirmedOutSet(injuryAdjustments, week) {
  if (!injuryAdjustments || injuryAdjustments.week !== week) return new Set();
  return new Set(
    injuryAdjustments.players.filter((p) => p.confirmed_out).map((p) => normalizeKey(p.name, p.team)));
}

/** Real per-player injury detail map (all flagged players, not just
 * confirmed-out), keyed the same normalized way, same week-gating as
 * getConfirmedOutSet. */
export function getInjuryAdjustmentMap(injuryAdjustments, week) {
  if (!injuryAdjustments || injuryAdjustments.week !== week) return new Map();
  return new Map(injuryAdjustments.players.map((p) => [normalizeKey(p.name, p.team), p]));
}

export function isConfirmedOut(name, team, outSet) {
  return outSet.has(normalizeKey(name, team));
}

/** Real lookup into the map from getInjuryAdjustmentMap() - always use
 * this (not a raw `.get(`${name}|${team}`)`) so every consumer benefits
 * from the same real suffix normalization. */
export function getInjuryAdjustment(name, team, injuryMap) {
  return injuryMap.get(normalizeKey(name, team)) || null;
}
