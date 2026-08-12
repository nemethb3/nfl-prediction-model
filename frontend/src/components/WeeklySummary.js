import React, { useState, useEffect } from 'react';
import '../styles/WeeklySummary.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import SeasonDataUnavailable from './SeasonDataUnavailable';

const MAX_WEEK = 18;

export default function WeeklySummary() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const summaryData = seasonData.weeklySummary;

  const [selectedWeek, setSelectedWeek] = useState(summaryData?.current_week);

  useEffect(() => {
    setSelectedWeek(summaryData?.current_week);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSeason]);

  if (!hasResults) {
    return (
      <div className="weekly-summary">
        <SeasonDataUnavailable season={selectedSeason} sectionName="Weekly Summary" />
      </div>
    );
  }

  const weekData = summaryData.weeks.find((w) => w.week === selectedWeek);

  if (!weekData) {
    return <div className="weekly-summary">No data for week {selectedWeek}</div>;
  }

  return (
    <div className="weekly-summary">
      <div className="header">
        <h1>Weekly Summary</h1>
        <p className="subtitle">Week {selectedWeek} Recap &amp; Preview</p>
      </div>

      <div className="week-selector">
        {summaryData.weeks.map((w) => (
          <button
            key={w.week}
            className={`week-btn ${selectedWeek === w.week ? 'active' : ''}`}
            onClick={() => setSelectedWeek(w.week)}
          >
            W{w.week}
          </button>
        ))}
      </div>

      {weekData.this_week && (
        <section className="section">
          <h2>This Week Recap</h2>

          <div className="metrics">
            <div className="metric">
              <span className="label">Accuracy</span>
              <span className="value">{weekData.this_week.accuracy_pct}%</span>
              <span className="detail">
                {weekData.this_week.correct_predictions}/{weekData.this_week.total_games}
              </span>
            </div>
            <div className="metric">
              <span className="label">Spread Error</span>
              <span className="value">{weekData.this_week.mean_spread_error}</span>
              <span className="detail">Points MAE</span>
            </div>
          </div>

          <div className="games-list">
            <h3>Games</h3>
            {weekData.this_week.games.map((game, i) => (
              <div key={i} className={`game-row ${game.prediction_correct ? 'correct' : 'incorrect'}`}>
                <div className="teams">
                  <span className="team">
                    {game.away_team} @ {game.home_team}
                  </span>
                </div>
                <div className="score">
                  <span className="final">
                    {game.away_score}-{game.home_score}
                  </span>
                </div>
                <div className="prediction">
                  <span className={`badge ${game.prediction_correct ? 'hit' : 'miss'}`}>
                    {game.prediction_correct ? '✅' : '❌'} {game.predicted_winner}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {weekData.this_week.biggest_surprises.length > 0 && (
            <div className="surprises">
              <h3>Biggest Surprises</h3>
              {weekData.this_week.biggest_surprises.map((surprise, i) => (
                <div key={i} className="surprise-card">
                  <div className="surprise-matchup">
                    {surprise.away_team} @ {surprise.home_team}
                  </div>
                  <div className="surprise-score">
                    {surprise.away_score}-{surprise.home_score}
                  </div>
                  <div className="surprise-details">
                    <span className="predicted">Predicted: {surprise.prediction_confidence}</span>
                    <span className="actual">
                      Actual: {surprise.actual_winner} won
                    </span>
                    <span className="magnitude">Swing: {surprise.surprise_score} pts</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {weekData.top_performers && (
        <section className="section">
          <h2>Fantasy Performances</h2>

          <div className="performers-grid">
            <div className="performers-column best">
              <h3>🟢 Best</h3>
              {weekData.top_performers.best.map((player, i) => (
                <div key={i} className="player-card">
                  <div className="player-name">{player.player_name}</div>
                  <div className="player-position">
                    {player.position} · {player.team}
                  </div>
                  <div className="player-stats">
                    <div className="stat">
                      <span className="label">Proj:</span>
                      <span className="value">{player.projected_ppr}</span>
                    </div>
                    <div className="stat">
                      <span className="label">Actual:</span>
                      <span className="value highlight">{player.actual_ppr}</span>
                    </div>
                    <div className="stat">
                      <span className="label">Diff:</span>
                      <span className="value positive">+{player.difference}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="performers-column worst">
              <h3>🔴 Worst</h3>
              {weekData.top_performers.worst.map((player, i) => (
                <div key={i} className="player-card">
                  <div className="player-name">{player.player_name}</div>
                  <div className="player-position">
                    {player.position} · {player.team}
                  </div>
                  <div className="player-stats">
                    <div className="stat">
                      <span className="label">Proj:</span>
                      <span className="value">{player.projected_ppr}</span>
                    </div>
                    <div className="stat">
                      <span className="label">Actual:</span>
                      <span className="value highlight">{player.actual_ppr}</span>
                    </div>
                    <div className="stat">
                      <span className="label">Diff:</span>
                      <span className="value negative">{player.difference}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {weekData.next_week ? (
        <section className="section">
          <h2>Week {weekData.next_week.week} Preview</h2>

          <div className="matchups">
            <h3>Matchups &amp; Predictions</h3>
            {weekData.next_week.games.map((game, i) => (
              <div key={i} className="matchup-card">
                <div className="matchup-header">
                  <span className="teams">
                    {game.away_team} @ {game.home_team}
                  </span>
                  {game.win_prob_home !== null && (
                    <span className={`prediction ${game.win_prob_home > 0.5 ? 'home-favored' : 'away-favored'}`}>
                      {game.predicted_winner} ({Math.round(Math.max(game.win_prob_home, game.win_prob_away) * 100)}%)
                    </span>
                  )}
                </div>
                <div className="key-players">
                  <div className="team-players">
                    <span className="label">{game.away_team}:</span>
                    <span className="players">{game.key_players.away.join(', ') || 'n/a'}</span>
                  </div>
                  <div className="team-players">
                    <span className="label">{game.home_team}:</span>
                    <span className="players">{game.key_players.home.join(', ') || 'n/a'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="top-plays">
            <h3>Top Fantasy Plays</h3>
            {weekData.next_week.top_fantasy_plays.map((player, i) => (
              <div key={i} className={`play-card matchup-${player.matchup_quality}`}>
                <div className="play-header">
                  <span className="player-name">{player.player_name}</span>
                  <span className="position-team">
                    {player.position} · {player.team}
                  </span>
                </div>
                <div className="play-details">
                  <span className="opponent">vs {player.opponent}</span>
                  <span className="projected">{player.projected_ppr} PPR</span>
                  <span className={`matchup matchup-${player.matchup_quality}`}>
                    {player.matchup_quality === 'good' ? '✅' : player.matchup_quality === 'neutral' ? '→' : '⚠️'} {player.matchup_quality}
                  </span>
                  {player.injury_status !== 'healthy' && <span className="injury">{player.injury_status}</span>}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        selectedWeek === MAX_WEEK && (
          <section className="section">
            <h2>Next Week Preview</h2>
            <p className="tab-note">
              No week {MAX_WEEK + 1} preview - this real 2025 season ends at week {MAX_WEEK}.
            </p>
          </section>
        )
      )}

      {weekData.season_context && (
        <section className="section">
          <h2>Season Context</h2>

          <div className="context-grid">
            <div className="context-card">
              <h3>Division Leaders</h3>
              <div className="leaders-list">
                {weekData.season_context.division_leaders.map((div, i) => (
                  <div key={i} className="leader-row">
                    <span className="division">{div.division}</span>
                    <span className="team-record">
                      <strong>{div.leader}</strong> ({div.wins}-{div.losses})
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="context-card">
              <h3>Playoff Race</h3>
              <div className="race-stats">
                <div className="stat">
                  <span className="label">Leading for Playoff</span>
                  <span className="value">{weekData.season_context.playoff_race.leading_for_playoff}</span>
                </div>
                <div className="stat">
                  <span className="label">Leading Division</span>
                  <span className="value">{weekData.season_context.playoff_race.leading_for_division}</span>
                </div>
                <div className="stat">
                  <span className="label">Wild Card Contenders</span>
                  <span className="value">{weekData.season_context.playoff_race.wild_card_contenders}</span>
                </div>
              </div>
              {weekData.season_context.playoff_race_note && (
                <div className="race-note">{weekData.season_context.playoff_race_note}</div>
              )}
            </div>
          </div>
        </section>
      )}

      <div className="disclaimer">
        <p>
          Real 2025 data throughout - no fabricated narratives. "This Week" recaps use real
          completed-game results; "Next Week" preview uses this project's real predictions for
          that week (which, since this is a completed historical season, already happened - shown
          as a preview for demo purposes, not a live forecast). Week {MAX_WEEK} (the default view)
          has no next-week preview since the real season ends there. Season Context reuses
          Section 3's real week-16 checkpoint snapshot for every week shown here (not
          re-computed per week) - "leading for playoff/division" reflects that fixed snapshot,
          not true week-by-week elimination math.
        </p>
      </div>
    </div>
  );
}
