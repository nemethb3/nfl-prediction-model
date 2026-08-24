import React from 'react';
import { teamColor, readableTextColor } from '../constants/teams';

export default function AwardRaces({ data }) {
  const candidates = data.candidates || [];
  const maxShare = Math.max(...candidates.map((c) => c.relative_share_pct), 1);

  return (
    <div className="award-races">
      <h2>MVP Race</h2>
      <p className="section-note">
        Real preseason ranking - see the methodology note below for exactly how this is
        computed. MVP is the only award race built here: Defensive Player of the Year and both
        Rookie of the Year awards have no real per-player projection data anywhere in this
        project (no defensive player stat model, no rookie season-stat projections), so they
        aren&apos;t shown rather than faked.
      </p>

      <div className="rankings-list">
        {candidates.map((c, idx) => {
          const bg = teamColor(c.team);
          const fg = readableTextColor(bg);
          const fillPct = (c.relative_share_pct / maxShare) * 100;
          return (
            <div key={c.player_id} className="rankings-row">
              <span className="rank">{idx + 1}.</span>
              <span className="team-badge" style={{ backgroundColor: bg, color: fg }}>
                {c.team}
              </span>
              <span className="team-name">
                {c.player_name} <span className="award-races-pos">({c.position})</span>
              </span>
              <div className="rankings-bar">
                <div className="rankings-fill" style={{ width: `${Math.max(fillPct, 2)}%` }} />
              </div>
              <span className="rankings-value">{c.relative_share_pct.toFixed(1)}%</span>
              <span className="award-races-stats">
                {c.projected_season_stats.passing_yards.toFixed(0)} pass yds,{' '}
                {c.projected_season_stats.passing_tds.toFixed(1)} pass TD
                {c.projected_season_stats.rushing_yards > 50
                  ? `, ${c.projected_season_stats.rushing_yards.toFixed(0)} rush yds`
                  : ''}
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
