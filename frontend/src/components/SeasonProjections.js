import React, { useState } from 'react';
import '../styles/SeasonProjections.css';
import '../styles/DivisionWinners.css';
import '../styles/PlayoffBracket.css';
import '../styles/SuperBowlOdds.css';
import { useSeason } from '../context/SeasonContext';
import { teamColor, readableTextColor } from '../constants/teams';
import DivisionWinners from './DivisionWinners';
import PlayoffBracket from './PlayoffBracket';
import SuperBowlOdds from './SuperBowlOdds';

const CHECKPOINT_WEEK = 16; // real 2025 checkpoint only - 2026 has no checkpoint concept (0 real games played)

function SortIcon({ active, order }) {
  if (!active) return <span className="sort-icon">⇅</span>;
  return <span className="sort-icon">{order === 'asc' ? '▲' : '▼'}</span>;
}

export default function SeasonProjections() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const projections = seasonData.seasonProjections;
  const superbowlData = seasonData.superbowlOdds;
  const isPreseason = !hasResults;
  // Real preseason Monte Carlo playoff simulation exists for 2026 (see
  // simulate_2026_playoffs.py) even though hasResults is correctly still
  // false (no real games played) - same "don't gate a section that now has
  // real data behind the season-wide hasResults flag" fix already applied
  // to Fantasy Rankings. Super Bowl odds remain out of scope (still null).
  const hasPlayoffSim = projections.some((t) => t.playoff_percentage !== null);
  const hasWinRangeCI = projections.some((t) => t.projected_wins_low_90 !== null && t.projected_wins_low_90 !== undefined);

  const [sortBy, setSortBy] = useState('projected_wins');
  const [sortOrder, setSortOrder] = useState('desc');

  const sorted = [...projections].sort((a, b) => {
    let aVal = a[sortBy];
    let bVal = b[sortBy];
    if (aVal === null || aVal === undefined) aVal = sortOrder === 'asc' ? Infinity : -Infinity;
    if (bVal === null || bVal === undefined) bVal = sortOrder === 'asc' ? Infinity : -Infinity;
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }
    if (aVal === bVal) return 0;
    const cmp = aVal > bVal ? 1 : -1;
    return sortOrder === 'asc' ? cmp : -cmp;
  });

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  const columns = [
    { key: 'team_name', label: 'Team' },
    { key: 'division', label: 'Division' },
    { key: 'wins_actual', label: 'Record' },
    { key: 'projected_wins', label: 'Proj. Wins' },
    ...(hasWinRangeCI ? [{ key: 'projected_wins_low_90', label: 'Win Range (90%)' }] : []),
    { key: 'playoff_percentage', label: 'Playoff %' },
    { key: 'playoff_seed', label: 'Seed' },
    { key: 'remaining_schedule_strength', label: 'Remaining Str.' },
  ];

  return (
    <div className="season-projections">
      <div className="header">
        <h1>Season Projections & Playoffs</h1>
        <p className="subtitle">
          {hasResults ? (
            <>
              Real {selectedSeason} regular season, snapshot as of the week-{CHECKPOINT_WEEK}
              checkpoint (the latest real playoff-odds checkpoint this project computed - this
              dataset is a completed historical season, not a live feed, so this is a fixed
              point-in-time view, not &quot;today&apos;s&quot; standings).
            </>
          ) : (
            <>
              Real {selectedSeason} preseason projections - the season hasn&apos;t been played
              (real 0-0-0 record for every team), so &quot;Proj. Wins&quot; and its 90% range come
              from this project&apos;s real preseason ensemble (Elo + EPA blend, with a real,
              exact confidence interval from per-game Bernoulli variance), and Playoff %/Seed/
              Division Winners come from a real 10,000-trial Monte Carlo simulation of the actual
              {' '}{selectedSeason} schedule - not a real in-season checkpoint.
            </>
          )}
        </p>
      </div>

      {(hasResults || hasPlayoffSim) ? (
        <>
          <DivisionWinners teams={projections} checkpointWeek={CHECKPOINT_WEEK} isPreseason={isPreseason} />
          <PlayoffBracket teams={projections} checkpointWeek={CHECKPOINT_WEEK} isPreseason={isPreseason} />
          {superbowlData ? (
            <SuperBowlOdds sbData={superbowlData} isPreseason={isPreseason} />
          ) : (
            <div className="preseason-panels-note">
              <p>
                Super Bowl odds aren&apos;t shown for {selectedSeason} - a real Super Bowl bracket
                simulation for this season doesn&apos;t exist yet.
              </p>
            </div>
          )}
        </>
      ) : (
        <div className="preseason-panels-note">
          <p>
            Division Winners, Playoff Picture, and Super Bowl Odds aren&apos;t shown for{' '}
            {selectedSeason} - real seeding and bracket simulation both require actual games
            to have been played, and none have yet. The sortable table below still shows real
            preseason projected wins per team.
          </p>
        </div>
      )}

      <div className="table-container">
        <table className="projections-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key} className="sortable" onClick={() => handleSort(col.key)}>
                  {col.label} <SortIcon active={sortBy === col.key} order={sortOrder} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((team) => (
              <tr
                key={team.team}
                className={`${team.is_division_winner ? 'division-winner' : ''} ${team.is_playoff_team ? 'playoff-team' : ''}`}
              >
                <td className="team-cell">
                  <span
                    className="team-badge"
                    style={{
                      backgroundColor: teamColor(team.team),
                      color: readableTextColor(teamColor(team.team)),
                    }}
                  >
                    {team.team}
                  </span>
                  <span className="team-name">{team.team_name}</span>
                </td>
                <td>{team.division}</td>
                <td className="record">
                  <strong>
                    {team.wins_actual}-{team.losses_actual}
                    {team.ties_actual > 0 ? `-${team.ties_actual}` : ''}
                  </strong>
                </td>
                <td className="projected">{team.projected_wins !== null ? team.projected_wins.toFixed(1) : '--'}</td>
                {hasWinRangeCI && (
                  <td className="win-range">
                    {team.projected_wins_low_90 != null
                      ? `${team.projected_wins_low_90.toFixed(1)}–${team.projected_wins_high_90.toFixed(1)}`
                      : '--'}
                  </td>
                )}
                <td className="percentage">
                  {team.playoff_percentage !== null ? `${(team.playoff_percentage * 100).toFixed(1)}%` : '--'}
                </td>
                <td className="seed">
                  {team.is_playoff_team ? <span className="seed-badge">#{team.playoff_seed}</span> : <span className="no-seed">--</span>}
                </td>
                <td className="strength">
                  {team.remaining_schedule_strength}
                  <span
                    className="elo-hint"
                    title={
                      hasResults
                        ? `Real average Elo of remaining opponents, as of the week-${CHECKPOINT_WEEK} checkpoint`
                        : 'Real average preseason Elo of every real scheduled opponent (the whole season is remaining)'
                    }
                  >
                    {' '}
                    (Elo)
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(hasResults || hasPlayoffSim) && (
        <div className="legend">
          <div className="legend-item">
            <span className="legend-badge division-winner">●</span> Division Leader (
            {hasResults ? `real wins through week ${CHECKPOINT_WEEK}` : 'most likely real Monte Carlo division winner'})
          </div>
          <div className="legend-item">
            <span className="legend-badge playoff-team">●</span> Playoff Team (
            {hasResults ? 'real Monte Carlo odds > 50%' : 'top 7 per conference by real Monte Carlo odds'})
          </div>
        </div>
      )}

      <div className="disclaimer">
        <p>
          {hasResults ? (
            <>
              Real data throughout: playoff odds and projected wins come from this project's real
              Monte Carlo simulation (playoff_probability.py), snapshotted at the real week-{CHECKPOINT_WEEK}
              checkpoint. Playoff seeding (division leaders 1-4, wildcards 5-7) is derived from real
              wins, tiebroken by real point differential - a simplified stand-in for the NFL's full
              multi-step tiebreaker procedure. Remaining schedule strength is each opponent's real
              Elo rating as of this checkpoint, not a projection of their future rating.
              Super Bowl odds (below) come from a real Monte Carlo bracket simulation seeded at this
              same real week-{CHECKPOINT_WEEK} checkpoint - not a heuristic, and not re-simulating
              regular-season uncertainty on top of it. See Model Transparency for the real bracket
              rules modeled.
            </>
          ) : (
            <>
              Real data throughout: projected wins and their 90% range come from this project's
              real preseason ensemble (a blend of Elo-based and EPA-based projections, both rolled
              forward from real 2015-2025 results) - the range is a real, exact 90% confidence
              interval from per-game Bernoulli variance across the real {selectedSeason} schedule,
              not a Monte Carlo approximation. Playoff %/Division %/Seed/Division Winner come from
              a genuinely separate real 10,000-trial Monte Carlo simulation of the actual{' '}
              {selectedSeason} schedule (same real, already-fit Elo win-probability formula used
              for Super Bowl odds elsewhere in this project) - each trial draws every game's
              outcome independently, then applies the same division-leaders-then-wildcards seeding
              rule real completed seasons use, substituting each team's real, static preseason Elo
              rating for real point differential as the tiebreak (no simulated game scores exist to
              tiebreak with). Because no real games have been played, every seed/division-winner
              badge reflects simulation odds, not a determined outcome - real Super Bowl odds
              aren't shown yet (see the note above).
            </>
          )}
        </p>
      </div>
    </div>
  );
}
