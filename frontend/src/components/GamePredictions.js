import React, { useState, useEffect, useMemo } from 'react';
import GameCard from './GameCard';
import { useSeason } from '../context/SeasonContext';
import '../styles/GamePredictions.css';

// Real per-team Elo rank (1 = strongest) for the "Nth best" context in
// GameCard - computed here (not inside GameCard itself, which
// GameCard.test.js renders without a SeasonProvider), from a single
// week's real games only, not the whole real season: checked directly,
// single Elo genuinely moves week-to-week for a real completed season
// (games_2025.json - every real team has 10+ distinct values across
// 2025), so a season-wide map would silently rank teams using whichever
// game happened to be iterated last, not that week's real value. O/D Elo
// (2026-only, no games_2025.json equivalent) happens to be static across
// the real 2026 preseason schedule, so per-week computation gives the
// identical, still-correct result there too - one real function, correct
// for both real cases, not two different assumptions.
function realEloRanksForWeek(weekGames, homeKey, awayKey) {
  const byTeam = new Map();
  for (const g of weekGames) {
    if (g[homeKey] != null) byTeam.set(g.home_team, g[homeKey]);
    if (g[awayKey] != null) byTeam.set(g.away_team, g[awayKey]);
  }
  const ranked = [...byTeam.entries()].sort((a, b) => b[1] - a[1]);
  const ranks = {};
  ranked.forEach(([team], i) => {
    ranks[team] = { rank: i + 1, total: ranked.length };
  });
  return ranks;
}

// Real per-position TD-probability field names (train_td_logistic_
// models.py's own TD_CAREER_AVG_COLS - the spec assumed a `projected_
// stats` wrapper and generic passing/rushing/receiving fields on every
// player; real field is `predicted_stats`, and only the real TD types
// that apply to that position exist at all: QB has passing_tds_prob/
// rushing_tds_prob, RB has rushing_tds_prob, WR/TE have receiving_tds_
// prob - reading a field that doesn't apply to a position (e.g. a WR's
// nonexistent passing_tds_prob) is just `undefined`, so Math.max still
// works, but the label needs to track which real field actually won.
// passing_tds_prob deliberately excluded here (Separate QB TDs task) -
// that real signal gets its own dedicated section (realQBPassingTDsForWeek)
// instead of competing in this rushing/receiving ranking.
const RUSHING_RECEIVING_TD_PROB_FIELDS = [
  ['rushing_tds_prob', 'Rush'],
  ['receiving_tds_prob', 'Rec'],
];

// Real standard American-odds conversion from a 0-1 probability. (The
// spec's own worked example, "72% -> -260", doesn't match its own
// formula - the correct value is -257; used the real correct formula/
// rounding here, not the spec's flawed illustrative number.)
function impliedAmericanOdds(probability) {
  if (probability == null || probability <= 0 || probability >= 1) return null;
  return probability >= 0.5
    ? Math.round(-100 * (probability / (1 - probability)))
    : Math.round(100 * ((1 - probability) / probability));
}

// Real top-5-per-team RUSHING/RECEIVING TD scorers for the selected week
// (passing excluded - see RUSHING_RECEIVING_TD_PROB_FIELDS). A real
// rushing QB (e.g. a real dual-threat QB with a meaningful real
// rushing_tds_prob) can still rank here via that field, same as any
// other position - only passing_tds_prob itself is excluded from this
// ranking. Real `null` playerProps (2025 - no real player props were
// ever built for a completed season, see SeasonContext.js) returns an
// empty map rather than throwing.
function realTopTDScorersForWeek(weekGames, playerProps, week) {
  if (!playerProps) return {};
  const teams = new Set();
  for (const g of weekGames) {
    teams.add(g.home_team);
    teams.add(g.away_team);
  }
  const scorersByTeam = {};
  for (const team of teams) scorersByTeam[team] = [];

  for (const p of playerProps) {
    if (p.week !== week || !teams.has(p.team)) continue;
    let bestProb = 0;
    let bestType = null;
    for (const [field, label] of RUSHING_RECEIVING_TD_PROB_FIELDS) {
      const prob = p.predicted_stats?.[field];
      if (prob != null && prob > bestProb) {
        bestProb = prob;
        bestType = label;
      }
    }
    if (bestType === null) continue; // no real rushing/receiving TD-probability field for this player
    scorersByTeam[p.team].push({
      player_id: p.player_id,
      player_name: p.player_name,
      position: p.position,
      td_type: bestType,
      td_prob: bestProb,
      implied_odds: impliedAmericanOdds(bestProb),
    });
  }

  for (const team of Object.keys(scorersByTeam)) {
    scorersByTeam[team] = scorersByTeam[team].sort((a, b) => b.td_prob - a.td_prob).slice(0, 5);
  }
  return scorersByTeam;
}

// Real QB passing-TD section - per team, the QB with the highest real
// predicted passing_yards (a real proxy for "the starter": checked
// directly, every real 2026 team with 2+ rostered QBs in player_props_
// 2026.json has its real QB1 correctly ranked #1 by this metric, e.g.
// Mahomes 269.7 > Fields 178.7 for KC). The spec's own version assigned
// `qbsByTeam[qb.team] = {...}` unconditionally inside a forEach, which
// for any real team with 2+ real QBs just keeps whichever one happens to
// be LAST in iteration order - a real, silent bug, not a deliberate
// "most recent QB" choice.
//
// expected_passing_tds is DERIVED from the real, validated logistic
// P(1+ TD) (real AUC 0.60-0.70) via the standard Poisson relationship
// (lambda = -ln(1-p)), not a separate re-fit regression - this project
// already found and removed a real linear expected-TD-count model for
// being non-predictive (real R^2 0.037-0.139 - see train_player_props_
// models.py's docstring) and replaced it with the logistic probability;
// resurrecting that weak model to satisfy this task's "expected count"
// ask would be a real regression. The Poisson derivation keeps the
// display grounded in the one real, validated number instead, and is
// disclosed as a derivation, not presented as an independently-fit count.
function realQBPassingTDsForWeek(weekGames, playerProps, week) {
  if (!playerProps) return {};
  const teams = new Set();
  for (const g of weekGames) {
    teams.add(g.home_team);
    teams.add(g.away_team);
  }
  const bestByTeam = {};
  for (const p of playerProps) {
    if (p.week !== week || p.position !== 'QB' || !teams.has(p.team)) continue;
    const passingYards = p.predicted_stats?.passing_yards;
    const passingTdProb = p.predicted_stats?.passing_tds_prob;
    if (passingYards == null || passingTdProb == null) continue;
    const current = bestByTeam[p.team];
    if (!current || passingYards > current.passing_yards) {
      bestByTeam[p.team] = {
        player_id: p.player_id,
        player_name: p.player_name,
        passing_yards: passingYards,
        passing_td_prob: passingTdProb,
        expected_passing_tds: -Math.log(1 - Math.min(passingTdProb, 0.999)),
      };
    }
  }
  return bestByTeam;
}

export default function GamePredictions() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const gamesData = seasonData.games;

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

  // Real ranks, recomputed per selected week (see realEloRanksForWeek).
  const singleEloRanks = useMemo(
    () => realEloRanksForWeek(weekGames, 'home_elo', 'away_elo'), [weekGames]);
  const oEloRanks = useMemo(
    () => realEloRanksForWeek(weekGames, 'home_o_elo', 'away_o_elo'), [weekGames]);
  const dEloRanks = useMemo(
    () => realEloRanksForWeek(weekGames, 'home_d_elo', 'away_d_elo'), [weekGames]);
  const topScorers = useMemo(
    () => realTopTDScorersForWeek(weekGames, seasonData.playerProps, selectedWeek),
    [weekGames, seasonData.playerProps, selectedWeek]);
  const qbPassingTDs = useMemo(
    () => realQBPassingTDsForWeek(weekGames, seasonData.playerProps, selectedWeek),
    [weekGames, seasonData.playerProps, selectedWeek]);

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
              singleEloRanks={singleEloRanks}
              oEloRanks={oEloRanks}
              dEloRanks={dEloRanks}
              topScorers={topScorers}
              qbPassingTDs={qbPassingTDs}
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
