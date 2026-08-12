import React, { useState } from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

export default function SuperBowlOdds({ sbData, isPreseason }) {
  const [showAll, setShowAll] = useState(false);

  const teams = sbData.teams;
  const maxPct = Math.max(...teams.map((t) => t.superbowl_odds_pct));
  const displayTeams = showAll ? teams : teams.slice(0, 10);

  return (
    <div className="superbowl-odds">
      <h2>Super Bowl Odds</h2>
      <p className="section-note">
        {isPreseason ? (
          <>
            Real Monte Carlo bracket simulation ({sbData.n_simulations.toLocaleString()} trials),
            seeded from this page&apos;s own real simulated regular-season seeding above (not an
            actual standing - no real games played yet) - not a heuristic. See Model Transparency
            for methodology.
          </>
        ) : (
          <>
            Real Monte Carlo bracket simulation ({sbData.n_simulations.toLocaleString()} trials) from
            the real week-{sbData.checkpoint_week} seeds - not a heuristic. See Model Transparency for
            methodology.
          </>
        )}
      </p>
      <div className="odds-list">
        {displayTeams.map((team, idx) => {
          const bg = teamColor(team.team);
          const fg = readableTextColor(bg);
          return (
            <div key={team.team} className="odds-row">
              <span className="rank">{idx + 1}.</span>
              <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
                {team.team}
              </span>
              <span className="team-name">{teamName(team.team)}</span>
              <div className="odds-bar">
                <div
                  className="odds-fill"
                  style={{ width: `${maxPct > 0 ? (team.superbowl_odds_pct / maxPct) * 100 : 0}%` }}
                />
              </div>
              <span className="odds-value">{team.superbowl_odds_pct.toFixed(1)}%</span>
              {showAll && <span className="conf-champ-value">{team.conference_champion_pct.toFixed(1)}% conf.</span>}
            </div>
          );
        })}
      </div>

      <button className="expand-btn" onClick={() => setShowAll(!showAll)}>
        {showAll ? 'Show Top 10 Only' : 'Show All 32 Teams'}
      </button>

      <div className="odds-note">
        <p>{sbData.methodology_note}</p>
      </div>
    </div>
  );
}
