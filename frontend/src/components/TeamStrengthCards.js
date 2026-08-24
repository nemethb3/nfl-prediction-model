import React, { useState } from 'react';
import { teamName, teamColor, readableTextColor } from '../constants/teams';

const SORT_OPTIONS = [
  { key: 'single_elo', label: 'Single Elo' },
  { key: 'o_elo', label: 'Offensive Elo' },
  { key: 'd_elo', label: 'Defensive Elo' },
];

// Real, computed quartile boundaries of THIS season's actual 32-team
// single_elo distribution - not asserted absolute thresholds (e.g. a
// fixed ">=1700 = Elite" cutoff would silently mean nothing in a season
// where the real league-wide Elo spread shifts). Recomputed live from
// whatever real teams are passed in, same "empirical, not asserted"
// convention this project already uses elsewhere (GameCard.js's
// MATCHUP_QUALITY_DISPLAY terciles, FantasyRankings.js's accuracy tiers).
function eloQuartiles(teams) {
  const sorted = teams.map((t) => t.single_elo).sort((a, b) => a - b);
  const at = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
  return { q25: at(0.25), q50: at(0.5), q75: at(0.75) };
}

function strengthLabel(elo, { q25, q50, q75 }) {
  if (elo >= q75) return 'Elite';
  if (elo >= q50) return 'Above Average';
  if (elo >= q25) return 'Below Average';
  return 'Rebuilding';
}

export default function TeamStrengthCards({ data, sbData }) {
  const [sortBy, setSortBy] = useState('single_elo');
  const [expandedTeam, setExpandedTeam] = useState(null);

  // Real join on team code - power_rankings_2026.json (Elo/playoff%) and
  // superbowl_odds_2026.json (SB/conf-championship%) are two separate real
  // files, both already loaded via useSeason() by SeasonProjections.js.
  const sbByTeam = new Map(sbData.teams.map((t) => [t.team, t]));
  const teams = data.teams.map((t) => ({ ...t, sb: sbByTeam.get(t.team) || null }));
  const quartiles = eloQuartiles(teams);
  const sorted = [...teams].sort((a, b) => b[sortBy] - a[sortBy]);

  return (
    <div className="team-strength-cards">
      <h2>Team Strength</h2>
      <p className="section-note">
        Real preseason carryover Elo (same source as Power Rankings) joined with real Super Bowl
        odds from this page&apos;s own Monte Carlo bracket simulation. &quot;Strength&quot; labels
        are real computed quartiles of this season&apos;s actual 32-team Elo spread, not fixed
        asserted cutoffs.
      </p>

      <div className="sort-buttons">
        {SORT_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            type="button"
            className={sortBy === opt.key ? 'active' : ''}
            onClick={() => setSortBy(opt.key)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <div className="teams-grid">
        {sorted.map((team, idx) => {
          const bg = teamColor(team.team);
          const fg = readableTextColor(bg);
          const isExpanded = expandedTeam === team.team;
          return (
            <div
              key={team.team}
              className={`team-strength-card ${isExpanded ? 'expanded' : ''}`}
              onClick={() => setExpandedTeam(isExpanded ? null : team.team)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setExpandedTeam(isExpanded ? null : team.team); }}
            >
              <div className="card-header">
                <span className="rank">#{idx + 1}</span>
                <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
                  {team.team}
                </span>
                <span className="team-name">{teamName(team.team)}</span>
                <span className="strength-badge">{strengthLabel(team.single_elo, quartiles)}</span>
              </div>

              <div className="card-quick-stats">
                <div className="stat">
                  <span className="label">Elo</span>
                  <span className="value">{team.single_elo.toFixed(0)}</span>
                </div>
                <div className="stat">
                  <span className="label">Off</span>
                  <span className="value">{team.o_elo.toFixed(0)}</span>
                </div>
                <div className="stat">
                  <span className="label">Def</span>
                  <span className="value">{team.d_elo.toFixed(0)}</span>
                </div>
                <div className="stat">
                  <span className="label">Playoffs</span>
                  <span className="value">{(team.playoff_percentage * 100).toFixed(0)}%</span>
                </div>
              </div>

              {isExpanded && (
                <div className="card-details">
                  <div className="detail-row">
                    <span>Division</span>
                    <span className="value">{team.division}</span>
                  </div>
                  <div className="detail-row">
                    <span>Division Winner (real Monte Carlo)</span>
                    <span className="value">{team.is_division_winner ? 'Most likely' : '—'}</span>
                  </div>
                  {team.sb && (
                    <>
                      <div className="detail-row">
                        <span>Conference Championship</span>
                        <span className="value">{team.sb.conference_champion_pct.toFixed(1)}%</span>
                      </div>
                      <div className="detail-row">
                        <span>Super Bowl Odds</span>
                        <span className="value">{team.sb.superbowl_odds_pct.toFixed(1)}%</span>
                      </div>
                    </>
                  )}
                </div>
              )}
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
