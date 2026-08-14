import React, { useState, useEffect, useMemo } from 'react';
import GameCard from './GameCard';
import { useSeason } from '../context/SeasonContext';
import '../styles/GamePredictions.css';

// Real per-team D_Elo rank (1 = strongest defense) for the "Nth best
// defense" context in GameCard - computed once here (not inside GameCard
// itself, which GameCard.test.js renders without a SeasonProvider) from
// the real, static-per-season home_d_elo/away_d_elo already in
// games_2026.json (verified: every real team has exactly one distinct
// D_Elo value across the whole real preseason schedule - no in-season
// updates have happened yet since no game has a real result).
function realDEloRanks(gamesData) {
  const byTeam = new Map();
  for (const g of gamesData) {
    if (g.home_d_elo != null) byTeam.set(g.home_team, g.home_d_elo);
    if (g.away_d_elo != null) byTeam.set(g.away_team, g.away_d_elo);
  }
  const ranked = [...byTeam.entries()].sort((a, b) => b[1] - a[1]);
  const ranks = {};
  ranked.forEach(([team], i) => {
    ranks[team] = { rank: i + 1, total: ranked.length };
  });
  return ranks;
}

export default function GamePredictions() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const gamesData = seasonData.games;
  const dEloRanks = useMemo(() => realDEloRanks(gamesData), [gamesData]);

  const weeks = [...new Set(gamesData.map((g) => g.week))].sort((a, b) => a - b);
  const [selectedWeek, setSelectedWeek] = useState(weeks[0]);
  const [expandedGameId, setExpandedGameId] = useState(null);

  // Real weeks differ by season only in edge cases, but reset cleanly on
  // season switch regardless, since a stale selectedWeek from one season
  // could momentarily point at a week the other season doesn't have.
  useEffect(() => {
    setSelectedWeek(weeks[0]);
    setExpandedGameId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSeason]);

  const weekGames = gamesData
    .filter((g) => g.week === selectedWeek)
    .slice()
    .sort((a, b) => {
      const timeA = a.kickoff_datetime ? new Date(a.kickoff_datetime).getTime() : 0;
      const timeB = b.kickoff_datetime ? new Date(b.kickoff_datetime).getTime() : 0;
      if (timeA !== timeB) return timeA - timeB;
      return Math.abs(b.our_spread) - Math.abs(a.our_spread);
    });

  return (
    <div className="game-predictions">
      <div className="header">
        <h1>Weekly Game Predictions</h1>
        <p className="subtitle">
          {hasResults
            ? `Real ${selectedSeason} completed season - predictions and real outcomes.`
            : `Real ${selectedSeason} preseason schedule - the season hasn't been played yet, so these are real model predictions only, no outcomes.`}
        </p>
        <div className="week-selector">
          <label htmlFor="week-select">Week</label>
          <select
            id="week-select"
            value={selectedWeek}
            onChange={(e) => {
              setSelectedWeek(Number(e.target.value));
              setExpandedGameId(null);
            }}
          >
            {weeks.map((w) => (
              <option key={w} value={w}>
                Week {w}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="games-container">
        {weekGames.length === 0 ? (
          <p className="empty-state">No games for this week.</p>
        ) : (
          weekGames.map((game) => (
            <GameCard
              key={game.id}
              game={game}
              dEloRanks={dEloRanks}
              isExpanded={expandedGameId === game.id}
              onToggle={() =>
                setExpandedGameId(expandedGameId === game.id ? null : game.id)
              }
            />
          ))
        )}
      </div>

      <div className="disclaimer">
        <p>
          {hasResults ? (
            <>
              Real {selectedSeason} backtest data. Predictions use Vegas closing lines plus a
              small matchup-EPA adjustment (real 2015-2023 backtest: this beats pure Elo, but is
              essentially a wash against Vegas alone). See the project README for full validated
              accuracy figures.
            </>
          ) : (
            <>
              Real {selectedSeason} preseason schedule - no real Vegas line exists yet for a season
              that hasn&apos;t been played, so predictions use this project&apos;s real preseason
              Elo (rolled forward from real 2015-2025 results) instead. Expand any game for detail.
            </>
          )}
        </p>
      </div>
    </div>
  );
}
