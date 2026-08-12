import React from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

// Real division names, matching season_projections_2025.json's real
// `division` field values exactly (verified before writing this).
const DIVISION_ORDER = [
  'AFC East', 'AFC North', 'AFC South', 'AFC West',
  'NFC East', 'NFC North', 'NFC South', 'NFC West',
];

export default function DivisionWinners({ teams, checkpointWeek, isPreseason }) {
  const winnerByDivision = Object.fromEntries(
    DIVISION_ORDER.map((div) => [div, teams.find((t) => t.division === div && t.is_division_winner)])
  );

  return (
    <div className="division-winners">
      <h2>Projected Division Winners</h2>
      <p className="section-note">
        {isPreseason
          ? 'Most frequent real Monte Carlo division winner across 10,000 simulations of the ' +
            'actual schedule (real per-game Elo win probabilities) - no real games played yet, ' +
            'so this is a simulated likelihood, not a determined outcome.'
          : `Real wins through the week-${checkpointWeek} checkpoint, tiebroken by real point differential.`}
      </p>
      <div className="winners-grid">
        {DIVISION_ORDER.map((div) => {
          const winner = winnerByDivision[div];
          const bg = winner ? teamColor(winner.team) : '#333';
          const fg = readableTextColor(bg);
          return (
            <div key={div} className="winner-card" style={{ backgroundColor: bg, color: fg }}>
              <div className="division-name">{div}</div>
              <div className="team-name">{winner ? teamName(winner.team) : 'TBD'}</div>
              <div className="team-record">
                {winner
                  ? isPreseason
                    ? `${(winner.division_winner_percentage * 100).toFixed(0)}% of sims`
                    : `${winner.wins_actual}-${winner.losses_actual}${winner.ties_actual > 0 ? `-${winner.ties_actual}` : ''}`
                  : '—'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
