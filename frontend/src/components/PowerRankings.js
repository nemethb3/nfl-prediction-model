import React, { useState } from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

const SORT_OPTIONS = [
  { key: 'single_elo', label: 'Single Elo' },
  { key: 'o_elo', label: 'Offensive Elo' },
  { key: 'd_elo', label: 'Defensive Elo' },
];

export default function PowerRankings({ data }) {
  const [sortBy, setSortBy] = useState('single_elo');

  const teams = [...data.teams].sort((a, b) => b[sortBy] - a[sortBy]);
  const maxVal = Math.max(...teams.map((t) => t[sortBy]));
  const minVal = Math.min(...teams.map((t) => t[sortBy]));
  const range = maxVal - minVal || 1;

  return (
    <div className="power-rankings">
      <h2>Power Rankings</h2>
      <p className="section-note">
        Real preseason carryover Elo (the same rating powering this project&apos;s 2026 win totals,
        playoff simulation, and Super Bowl odds) - not a heuristic power ranking. Split into
        offensive/defensive Elo using the same real source the player-props model&apos;s opponent
        signal already uses.
      </p>

      <div className="sort-buttons">
        {SORT_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            className={sortBy === opt.key ? 'active' : ''}
            onClick={() => setSortBy(opt.key)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <div className="rankings-list">
        {teams.map((team, idx) => {
          const bg = teamColor(team.team);
          const fg = readableTextColor(bg);
          const value = team[sortBy];
          const fillPct = ((value - minVal) / range) * 100;
          return (
            <div key={team.team} className="rankings-row">
              <span className="rank">{idx + 1}.</span>
              <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
                {team.team}
              </span>
              <span className="team-name">{teamName(team.team)}</span>
              <div className="rankings-bar">
                <div className="rankings-fill" style={{ width: `${Math.max(fillPct, 2)}%` }} />
              </div>
              <span className="rankings-value">{value.toFixed(0)}</span>
              <span className="rankings-playoff-pct">
                {(team.playoff_percentage * 100).toFixed(0)}% playoffs
              </span>
            </div>
          );
        })}
      </div>

      <div className="rankings-note">
        <p>{data.methodology_note}</p>
      </div>
    </div>
  );
}
