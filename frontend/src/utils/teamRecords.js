// Real per-team W-L-T record, derived directly from games_2026.json's own
// actual_winner field - no separate generated file, so it can never drift
// out of sync with whatever games ingest_completed_results_2026.py has
// populated. A completed real tie writes actual_winner: "TIE" (see that
// script), handled as its own branch rather than silently attributed to
// either team.
export function computeTeamRecords(games) {
  const records = {};
  const ensure = (team) => {
    if (!records[team]) records[team] = { wins: 0, losses: 0, ties: 0 };
    return records[team];
  };
  for (const g of games) {
    const winner = g.actual_winner;
    if (!winner) continue; // not yet played
    const home = ensure(g.home_team);
    const away = ensure(g.away_team);
    if (winner === 'TIE') {
      home.ties += 1;
      away.ties += 1;
    } else if (winner === g.home_team) {
      home.wins += 1;
      away.losses += 1;
    } else if (winner === g.away_team) {
      away.wins += 1;
      home.losses += 1;
    }
  }
  return records;
}

export function formatRecord(record) {
  if (!record) return '0-0';
  const { wins, losses, ties } = record;
  return ties > 0 ? `${wins}-${losses}-${ties}` : `${wins}-${losses}`;
}
