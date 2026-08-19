import React, { useState } from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

// Real division names, matching season_projections_2025.json's real
// `division` field values exactly (verified before writing this).
const DIVISION_ORDER = [
  'AFC East', 'AFC North', 'AFC South', 'AFC West',
  'NFC East', 'NFC North', 'NFC South', 'NFC West',
];

export default function DivisionWinners({ teams, checkpointWeek, isPreseason }) {
  const [expandedDivision, setExpandedDivision] = useState(null);

  const teamsByDivision = Object.fromEntries(
    DIVISION_ORDER.map((div) => [
      div,
      teams
        .filter((t) => t.division === div)
        .sort((a, b) => (b.division_winner_percentage || 0) - (a.division_winner_percentage || 0)),
    ])
  );

  const toggleDivision = (div) => {
    setExpandedDivision((current) => (current === div ? null : div));
  };

  return (
    <div className="division-winners">
      <h2>Projected Division Winners</h2>
      <p className="section-note">
        {isPreseason
          ? 'Most frequent real Monte Carlo division winner across 10,000 simulations of the ' +
            'actual schedule (real per-game Elo win probabilities) - no real games played yet, ' +
            'so this is a simulated likelihood, not a determined outcome. Click a division to ' +
            "see every team's real odds."
          : `Real wins through the week-${checkpointWeek} checkpoint, tiebroken by real point ` +
            "differential. Click a division to see every team's real record and odds."}
      </p>
      <div className="division-groups">
        {DIVISION_ORDER.map((div) => {
          const divTeams = teamsByDivision[div];
          const winner = divTeams.find((t) => t.is_division_winner) || divTeams[0];
          const isExpanded = expandedDivision === div;
          const bg = winner ? teamColor(winner.team) : '#333';
          const fg = readableTextColor(bg);

          return (
            <div key={div} className="division-group">
              <button
                type="button"
                className={`division-header ${isExpanded ? 'expanded' : ''}`}
                onClick={() => toggleDivision(div)}
                aria-expanded={isExpanded}
              >
                <span className="division-name">{div}</span>
                <span className="division-header-right">
                  {winner && (
                    <span className="winner-badge" style={{ backgroundColor: bg, color: fg }}>
                      {winner.team}{' '}
                      {isPreseason
                        ? `${((winner.division_winner_percentage || 0) * 100).toFixed(0)}%`
                        : `${winner.wins_actual}-${winner.losses_actual}${winner.ties_actual > 0 ? `-${winner.ties_actual}` : ''}`}
                    </span>
                  )}
                  <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                </span>
              </button>

              {isExpanded && (
                <div className="division-expanded">
                  <table className="division-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>{isPreseason ? 'Div Win %' : 'Record'}</th>
                        <th>Playoff %</th>
                        <th>{isPreseason ? 'Remaining Str. (Elo)' : 'Seed'}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {divTeams.map((t) => (
                        <tr key={t.team} className={t.is_division_winner ? 'division-winner' : ''}>
                          <td className="team-cell">
                            <span
                              className="team-badge"
                              style={{ backgroundColor: teamColor(t.team), color: readableTextColor(teamColor(t.team)) }}
                            >
                              {t.team}
                            </span>
                            <span className="team-name">{teamName(t.team)}</span>
                          </td>
                          <td>
                            {isPreseason
                              ? `${((t.division_winner_percentage || 0) * 100).toFixed(1)}%`
                              : `${t.wins_actual}-${t.losses_actual}${t.ties_actual > 0 ? `-${t.ties_actual}` : ''}`}
                          </td>
                          <td>{t.playoff_percentage !== null ? `${(t.playoff_percentage * 100).toFixed(1)}%` : '--'}</td>
                          <td>
                            {isPreseason
                              ? (t.remaining_schedule_strength ?? '--')
                              : (t.is_playoff_team ? `#${t.playoff_seed}` : '--')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
