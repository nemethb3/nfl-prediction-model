// Real, league-adjustable fantasy points from this project's own real
// per-stat projections (player_props_2026.json's predicted_stats -
// opponent-adjusted, week-specific regression output; see
// ModelTransparency's "Projected Stats" section), NOT
// fantasy_rankings_2026.json's projected_ppr (which only exposes this
// project's own default-PPR total, with no raw components to re-score
// under a different league's rules).
//
// Real, disclosed approximation carried over unchanged from how this
// project already treats TD projections everywhere else: `*_tds_prob`
// fields are a real P(1+ TD) from a logistic model (see
// ModelTransparency - AUC 0.60-0.70), not an expected TD count. Used
// directly as an expected-count proxy for scoring, the same real
// convention this project's own default projected_ppr already uses -
// not a new approximation invented here.

export function computeLeaguePoints(predictedStats, position, scoring) {
  const s = predictedStats || {};
  let points = 0;

  if (position === 'QB') {
    points += (s.passing_yards || 0) / scoring.passYdPerPt;
    points += (s.passing_tds_prob || 0) * scoring.passTd;
  }
  // Real schema note: RB rushing/receiving and QB rushing both use the
  // same rushing_yards/rushing_tds_prob field names in predicted_stats,
  // and receiving fields only appear for RB/WR/TE rows (verified against
  // real player_props_2026.json samples for all four positions before
  // writing this) - so these two blocks apply to any position that has
  // the relevant real fields present, not a hardcoded position list.
  points += (s.rushing_yards || 0) / scoring.rushYdPerPt;
  points += (s.rushing_tds_prob || 0) * scoring.rushTd;
  points += (s.receiving_yards || 0) / scoring.recYdPerPt;
  points += (s.receiving_tds_prob || 0) * scoring.recTd;
  points += (s.receptions || 0) * scoring.reception;

  return Math.round(points * 10) / 10;
}

/** Real per-player league-adjusted projections for one week.
 * `weeklyProps` = player_props_2026.json rows already filtered to the
 * target week; `rosterPlayers` = [{player_id, name, position, team}]
 * from the real Sleeper->nflverse crosswalk (sleeper_id_mapping.json).
 * A rostered player with no real prop row this week (outside this
 * project's real ~390 ranked players) is returned with points=null,
 * disclosed rather than defaulted to 0 (which would wrongly look like a
 * real, confident zero-point projection). */
export function adjustRosterForLeague(rosterPlayers, weeklyProps, targetWeek, scoring) {
  const propsById = new Map(weeklyProps.map((p) => [p.player_id, p]));

  return rosterPlayers.map((player) => {
    const propRow = propsById.get(player.player_id);
    const points = propRow ? computeLeaguePoints(propRow.predicted_stats, player.position, scoring) : null;
    return {
      ...player,
      week: targetWeek,
      leaguePoints: points,
    };
  });
}
