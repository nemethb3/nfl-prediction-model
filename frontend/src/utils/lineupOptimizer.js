// Real, greedy lineup optimizer for a single-FLEX roster (QB/RB/WR/TE/
// FLEX only - this project has no DEF/K model, so those slots are never
// requested here; see sleeperLeagueSettings.js).
//
// Correctness note: for a roster with exactly one FLEX slot (RB/WR/TE
// eligible) and disjoint per-position minimums, "top-N by position rank
// for each required slot, then single best remaining RB/WR/TE for FLEX"
// IS the real, provably optimal assignment - not just a reasonable
// heuristic. Reasoning: the required per-position slots must be filled
// from that position's own pool regardless of order, so the top-N by
// that position's own points is always correct to lock in; the single
// FLEX slot then goes to whichever remaining RB/WR/TE (from any
// position) has the highest points, which greedy naturally finds by
// processing FLEX last against the leftover pool. A league with more
// than one true FLEX-type slot (e.g. SUPERFLEX) is NOT covered by this
// proof and isn't attempted - sleeperLeagueSettings.js already reports
// those as `unsupportedSlots` rather than silently mishandling them.

const FLEX_ELIGIBLE = ['RB', 'WR', 'TE'];

export function optimizeLineup(players, slotCounts) {
  const sorted = [...players]
    .filter((p) => typeof p.leaguePoints === 'number')
    .sort((a, b) => b.leaguePoints - a.leaguePoints);

  const used = new Set();
  const starting = [];
  const unfilled = [];

  const fillSlot = (slotLabel, isEligible, count) => {
    let filled = 0;
    for (const player of sorted) {
      if (filled >= count) break;
      if (used.has(player.player_id)) continue;
      if (!isEligible(player)) continue;
      starting.push({ slot: count > 1 ? `${slotLabel}${filled + 1}` : slotLabel, player });
      used.add(player.player_id);
      filled += 1;
    }
    if (filled < count) unfilled.push({ slot: slotLabel, missing: count - filled });
  };

  fillSlot('QB', (p) => p.position === 'QB', slotCounts.QB || 0);
  fillSlot('RB', (p) => p.position === 'RB', slotCounts.RB || 0);
  fillSlot('WR', (p) => p.position === 'WR', slotCounts.WR || 0);
  fillSlot('TE', (p) => p.position === 'TE', slotCounts.TE || 0);
  fillSlot('FLEX', (p) => FLEX_ELIGIBLE.includes(p.position), slotCounts.FLEX || 0);

  const bench = sorted.filter((p) => !used.has(p.player_id));
  const totalPoints = starting.reduce((sum, s) => sum + s.player.leaguePoints, 0);

  return { starting, bench, unfilled, totalPoints: Math.round(totalPoints * 10) / 10 };
}
