import React from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

export default function PlayoffBracket({ teams, checkpointWeek, isPreseason }) {
  const getConferenceSeeds = (conference) =>
    teams
      .filter((t) => t.conference === conference && t.is_playoff_team)
      .sort((a, b) => a.playoff_seed - b.playoff_seed);

  const afcSeeds = getConferenceSeeds('AFC');
  const nfcSeeds = getConferenceSeeds('NFC');

  const renderSeed = (team) => {
    const bg = teamColor(team.team);
    const fg = readableTextColor(bg);
    return (
      <div key={team.team} className="seed-row">
        <span className="seed-num">#{team.playoff_seed}</span>
        <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
          {team.team}
        </span>
        <span className="team-name">{teamName(team.team)}</span>
        <span className="record">
          {isPreseason
            ? `${(team.playoff_percentage * 100).toFixed(0)}% of sims`
            : `${team.wins_actual}-${team.losses_actual}${team.ties_actual > 0 ? `-${team.ties_actual}` : ''}`}
        </span>
        {team.playoff_seed === 1 && <span className="bye-tag">BYE</span>}
      </div>
    );
  };

  return (
    <div className="playoff-bracket">
      <h2>Projected Playoff Picture</h2>
      <p className="section-note">
        {isPreseason
          ? 'Top 7 per conference by real Monte Carlo playoff odds (10,000 simulations of the ' +
            'actual schedule). Seeds 1-4 are each division’s most frequent real simulated ' +
            'winner, ranked by real playoff odds; seeds 5-7 are the next-best real wildcard odds - ' +
            'not a determined outcome, no real games played yet.'
          : `Real seeds 1-7 per conference, week-${checkpointWeek} checkpoint. Division winners (seeds ` +
            '1-4) shown above wildcards (5-7) regardless of record, per real NFL seeding rules.'}
      </p>
      <div className="bracket-container">
        <div className="conference-bracket">
          <h3>AFC</h3>
          <div className="seeds-list">{afcSeeds.map(renderSeed)}</div>
        </div>
        <div className="conference-bracket">
          <h3>NFC</h3>
          <div className="seeds-list">{nfcSeeds.map(renderSeed)}</div>
        </div>
      </div>
    </div>
  );
}
