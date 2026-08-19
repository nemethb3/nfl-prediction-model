import React, { useState } from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

const SEED_NUMBERS = [1, 2, 3, 4, 5, 6, 7];

export default function PlayoffBracket({ teams, checkpointWeek, isPreseason }) {
  const [gridConference, setGridConference] = useState('AFC');

  const getConferenceSeeds = (conference) =>
    teams
      .filter((t) => t.conference === conference && t.is_playoff_team)
      .sort((a, b) => a.playoff_seed - b.playoff_seed);

  const afcSeeds = getConferenceSeeds('AFC');
  const nfcSeeds = getConferenceSeeds('NFC');

  const hasSeedDistribution = isPreseason && teams.some((t) => t.seed_distribution);
  const gridTeams = teams
    .filter((t) => t.conference === gridConference && t.seed_distribution)
    .sort((a, b) => b.playoff_percentage - a.playoff_percentage);

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

      {hasSeedDistribution && (
        <div className="seed-distribution">
          <h3>Seed Probability Grid</h3>
          <p className="section-note">
            Real probability of landing at EACH seed (1-7) across all 10,000 Monte Carlo trials,
            for every team in the conference - not just the single most-likely seed shown above.
            &quot;Miss&quot; is the real chance of missing the playoffs entirely.
          </p>
          <div className="conference-selector">
            <button
              className={gridConference === 'AFC' ? 'active' : ''}
              onClick={() => setGridConference('AFC')}
            >
              AFC
            </button>
            <button
              className={gridConference === 'NFC' ? 'active' : ''}
              onClick={() => setGridConference('NFC')}
            >
              NFC
            </button>
          </div>
          <div className="seed-distribution-table-container">
            <table className="seed-distribution-table">
              <thead>
                <tr>
                  <th className="team-col">Team</th>
                  {SEED_NUMBERS.map((s) => (
                    <th key={s}>{s}</th>
                  ))}
                  <th>Miss</th>
                </tr>
              </thead>
              <tbody>
                {gridTeams.map((team) => {
                  const bg = teamColor(team.team);
                  const fg = readableTextColor(bg);
                  const missPct = Math.max(0, 1 - team.playoff_percentage);
                  return (
                    <tr key={team.team}>
                      <td className="team-col">
                        <span className="team-col-inner">
                          <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
                            {team.team}
                          </span>
                          <span className="team-name">{teamName(team.team)}</span>
                        </span>
                      </td>
                      {SEED_NUMBERS.map((s) => {
                        const pct = team.seed_distribution[String(s)] || 0;
                        return (
                          <td key={s} className="seed-pct-cell" style={{ opacity: pct > 0 ? 0.35 + pct * 1.5 : 0.15 }}>
                            {pct > 0 ? `${(pct * 100).toFixed(0)}%` : '–'}
                          </td>
                        );
                      })}
                      <td className="miss-pct-cell">{(missPct * 100).toFixed(0)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
