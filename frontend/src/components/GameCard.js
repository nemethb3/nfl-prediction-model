import React, { useState } from 'react';
import { teamName, teamColor, teamSecondaryColor, readableTextColor, TEAM_TIMEZONES, TIMEZONE_LABELS } from '../constants/teams';
import { useKeyboardToggle } from '../hooks/useKeyboardToggle';
import { getUserTimezone, formatKickoffInZone, SHORT_CODE_TO_IANA_ZONE, ARIZONA_IANA_ZONE } from '../utils/timeUtils';

// Sign convention (verified against real 2025 moneylines, matches
// data_pipeline.py's documented convention): positive spread = home team
// favored, negative = away team favored.
function favoriteTeam(spread, homeTeam, awayTeam) {
  if (spread > 0) return homeTeam;
  if (spread < 0) return awayTeam;
  return null;
}

function formatSpread(spread, homeTeam, awayTeam) {
  if (spread === 0) return 'Pick ’em';
  const favorite = favoriteTeam(spread, homeTeam, awayTeam);
  return `${favorite} -${Math.abs(spread).toFixed(1)}`;
}

function ordinal(n) {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}

// Rank maps (singleEloRanks/oEloRanks/dEloRanks) are optional props,
// computed once per week in GamePredictions.js (real Elo moves week-to-
// week for a completed season, so this can't be derived from a single
// game prop) - NOT computed via useSeason() here since GameCard.test.js
// renders this component directly, without a SeasonProvider.
function rankLabel(team, ranks) {
  const r = ranks?.[team];
  if (!r) return '';
  return ` (${ordinal(r.rank)} of ${r.total})`;
}

// Labels match generate_dashboard_data.py's empirical-tercile buckets
// (real quantiles of the season's own net_edge_diff distribution, not an
// asserted threshold).
const MATCHUP_QUALITY_DISPLAY = {
  favorable_home: { emoji: '🟢', label: 'Favorable matchup edge for the home team' },
  favorable_away: { emoji: '🔴', label: 'Favorable matchup edge for the away team' },
  neutral: { emoji: '🟡', label: 'Neutral matchup edge' },
};

// Informational only - NOT a betting recommendation. This project's own
// real backtest (edge_detection.py) found that acting on disagreements
// between our model and Vegas produced -36% ROI (actively harmful), so
// this is shown purely as "here's where the two differ," with that real
// finding attached, never phrased as advice.
function spreadDisagreement(ourSpread, vegasSpread, home, away) {
  if (vegasSpread === null || vegasSpread === undefined) return null;
  const ourFavorite = favoriteTeam(ourSpread, home, away);
  const vegasFavorite = favoriteTeam(vegasSpread, home, away);
  if (ourFavorite === vegasFavorite) return null;
  return { ourFavorite, vegasFavorite };
}

// Real, objective historical fact for completed games only (not a
// prediction) - did the home team's real result beat the real Vegas
// closing line?
function atsResult(actualSpreadMargin, vegasSpread) {
  if (actualSpreadMargin === null || actualSpreadMargin === undefined ||
      vegasSpread === null || vegasSpread === undefined) return null;
  if (actualSpreadMargin === vegasSpread) return 'push';
  return actualSpreadMargin > vegasSpread ? 'home' : 'away';
}

// Real fact, verified against this project's own 2025/2026 schedule data:
// kickoff_datetime's time-of-day is stored in Eastern Time (nflverse's
// documented schedules.gametime convention), not UTC and not each game's
// own stadium-local time - e.g. real LV (Pacific) home games show
// gametime 16:05/16:25, the real ET value for the real 1:05/1:25 PM
// Pacific "late window" kickoff, not an evening LV-local time.
//
// Real, previously-shipped bug found and fixed by this task: an earlier
// version of this file hand-rolled the ET->UTC conversion with a
// hardcoded single DST cutover date. Since ISO date strings compare
// lexicographically, that logic silently used the wrong (EST) offset for
// EVERY real 2026 game before the real 2026 DST cutover (Nov 1, 2026) -
// a genuine ~1-hour-off bug for roughly half the 2026 season in "Your
// Time" mode. Real, permanent fix now lives in utils/timeUtils.js: real
// IANA-timezone-database-driven conversion (Intl.DateTimeFormat), which
// handles DST correctly for any real year with no hardcoded date, and
// handles Arizona's real non-DST-observing zone natively (no more
// hand-rolled Arizona special case).
const USER_TIMEZONE = getUserTimezone();

function stadiumIanaZone(homeTeam) {
  if (homeTeam === 'ARI') return ARIZONA_IANA_ZONE;
  return SHORT_CODE_TO_IANA_ZONE[TEAM_TIMEZONES[homeTeam]] || 'America/New_York';
}

function formatKickoff(kickoffISO, homeTeam, showUserTime) {
  const zone = showUserTime ? USER_TIMEZONE : stadiumIanaZone(homeTeam);
  return formatKickoffInZone(kickoffISO, zone);
}

export default function GameCard({
  game, singleEloRanks, oEloRanks, dEloRanks, topScorers, qbPassingTDs, isExpanded, onToggle,
}) {
  const [showUserTime, setShowUserTime] = useState(true);
  const handleKeyDown = useKeyboardToggle(onToggle);
  const {
    home_team: home,
    away_team: away,
    home_qb_name: homeQb,
    away_qb_name: awayQb,
    home_elo: homeElo,
    away_elo: awayElo,
    our_spread: ourSpread,
    vegas_spread: vegasSpread,
    win_prob_home: winProbHome,
    win_prob_away: winProbAway,
    net_edge_diff: netEdgeDiff,
    matchup_quality: matchupQuality,
    home_recent_form: homeForm,
    away_recent_form: awayForm,
    head_to_head: headToHead,
    base_source: baseSource,
    actual_home_score: homeScore,
    actual_away_score: awayScore,
    actual_winner: actualWinner,
    actual_spread_margin: actualSpreadMargin,
    did_we_predict_correctly: correct,
    predicted_total_value: predictedTotal,
    vegas_total: vegasTotal,
    predicted_total_diff: predictedTotalDiff,
    predicted_total_direction: predictedTotalDirection,
    home_o_elo: homeOElo,
    home_d_elo: homeDElo,
    away_o_elo: awayOElo,
    away_d_elo: awayDElo,
    single_elo_spread: singleEloSpread,
    single_elo_win_prob_home: singleEloWinProbHome,
  } = game;

  const hasResult = homeScore !== null && homeScore !== undefined;
  const favorite = favoriteTeam(ourSpread, home, away);
  const borderColor = favorite ? teamColor(favorite) : '#555';
  const disagreement = spreadDisagreement(ourSpread, vegasSpread, home, away);
  const ats = atsResult(actualSpreadMargin, vegasSpread);
  const winProbFavorite = winProbHome !== null && winProbHome !== undefined
    ? (winProbHome > 0.5 ? home : away) : null;
  const winProbHit = hasResult && winProbFavorite && actualWinner !== 'TIE'
    ? actualWinner === winProbFavorite : null;
  const kickoff = formatKickoff(game.kickoff_datetime, home, showUserTime);

  return (
    <div
      className={`game-card ${isExpanded ? 'game-card-open' : ''}`}
      style={{ borderColor }}
      onClick={onToggle}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-expanded={isExpanded}
    >
      <div className="game-card-collapsed">
        <div className="game-card-teams">
          <span
            className="team-box"
            style={{
              backgroundColor: teamSecondaryColor(away),
              color: readableTextColor(teamSecondaryColor(away)),
            }}
          >
            {away}
          </span>
          <span className="at-symbol">@</span>
          <span
            className="team-box"
            style={{
              backgroundColor: teamSecondaryColor(home),
              color: readableTextColor(teamSecondaryColor(home)),
            }}
          >
            {home}
          </span>
          <span className="kickoff-time">
            {kickoff.date} · {kickoff.time}
            {game.kickoff_datetime && (
              <span className="timezone-label">
                {' '}
                ({showUserTime ? 'Your Time' : `${TIMEZONE_LABELS[TEAM_TIMEZONES[home]] || TEAM_TIMEZONES[home]} Time`})
              </span>
            )}
          </span>
          {game.kickoff_datetime && (
            <button
              type="button"
              className="timezone-toggle"
              onClick={(e) => {
                e.stopPropagation();
                setShowUserTime(!showUserTime);
              }}
              title={showUserTime ? 'Switch to Game Time' : 'Switch to Your Time'}
            >
              {showUserTime ? '🌐 Game Time' : '🌐 Your Time'}
            </button>
          )}
          {homeElo !== null && homeElo !== undefined && awayElo !== null && awayElo !== undefined && (
            <span className="matchup-elo">
              Elo: {Math.round(awayElo)} ({away}) vs {Math.round(homeElo)} ({home})
            </span>
          )}
        </div>

        <div className="game-card-spreads">
          {singleEloSpread !== null && singleEloSpread !== undefined && (
            <div className="single-elo-spread">
              <span className="label">Single-Elo</span> {formatSpread(singleEloSpread, home, away)}
            </div>
          )}
          <div className="our-spread">
            <span className="label">O/D Elo</span> {formatSpread(ourSpread, home, away)}
          </div>
          {vegasSpread !== null && vegasSpread !== undefined && (
            <div className="vegas-spread">
              <span className="label">Vegas</span> {formatSpread(vegasSpread, home, away)}
            </div>
          )}
          {predictedTotal !== null && predictedTotal !== undefined && (
            <div className="predicted-total" title="Weak, informational projection only - see expanded view">
              <span className="label">Total</span> {predictedTotal.toFixed(1)}
            </div>
          )}
        </div>

        {matchupQuality && MATCHUP_QUALITY_DISPLAY[matchupQuality] && (
          <div className="matchup-badge" title={MATCHUP_QUALITY_DISPLAY[matchupQuality].label}>
            {MATCHUP_QUALITY_DISPLAY[matchupQuality].emoji}
          </div>
        )}

        {hasResult && correct !== null && (
          <div className="accuracy" title={correct ? 'Predicted winner correctly' : 'Missed the winner'}>
            {correct ? '✅' : '❌'}
          </div>
        )}

        <div className="expand-hint">{isExpanded ? '▲' : '▼'}</div>
      </div>

      {isExpanded && (
        <div className="game-card-expanded" onClick={(e) => e.stopPropagation()}>
          <h3>
            {teamName(away)} @ {teamName(home)}
          </h3>

          <div className="section">
            <div className="section-title">Prediction</div>
            {singleEloSpread !== null && singleEloSpread !== undefined && (
              <div>
                Single-Elo spread: {formatSpread(singleEloSpread, home, away)}
                {singleEloWinProbHome !== null && singleEloWinProbHome !== undefined && (
                  <> ({Math.round((favoriteTeam(singleEloSpread, home, away) === home
                    ? singleEloWinProbHome : 1 - singleEloWinProbHome) * 100)}% implied)</>
                )}
              </div>
            )}
            <div>O/D Elo spread: {formatSpread(ourSpread, home, away)}</div>
            {vegasSpread !== null && vegasSpread !== undefined && (
              <div>Vegas closing line: {formatSpread(vegasSpread, home, away)}</div>
            )}
            <div>Source: {baseSource === 'vegas' ? 'Vegas line + matchup adjustment' : 'Elo fallback (no posted line)'}</div>
            {netEdgeDiff !== null && netEdgeDiff !== undefined && Math.abs(netEdgeDiff) > 0.001 && (
              <div>Matchup EPA edge differential: {netEdgeDiff > 0 ? '+' : ''}{netEdgeDiff.toFixed(2)} (home perspective)</div>
            )}
            {matchupQuality && MATCHUP_QUALITY_DISPLAY[matchupQuality] && (
              <div>{MATCHUP_QUALITY_DISPLAY[matchupQuality].emoji} {MATCHUP_QUALITY_DISPLAY[matchupQuality].label}</div>
            )}
          </div>

          {predictedTotal !== null && predictedTotal !== undefined && (
            <div className="section">
              <div className="section-title">Projected Total</div>
              <div>Our projection: {predictedTotal.toFixed(1)} combined points</div>
              {vegasTotal !== null && vegasTotal !== undefined && (
                <>
                  <div>Vegas total: {vegasTotal.toFixed(1)}</div>
                  <div>
                    Diff: {predictedTotalDiff > 0 ? '+' : ''}{predictedTotalDiff.toFixed(1)} ({predictedTotalDirection})
                  </div>
                </>
              )}
              <div className="small-text">Weak signal (real backtest R² 0.005) - informational only.</div>
            </div>
          )}

          {winProbFavorite && (
            <div className="section">
              <div className="section-title">Win Probability</div>
              <div>
                Our model favors <strong>{winProbFavorite}</strong> ({Math.round(
                  (winProbFavorite === home ? winProbHome : winProbAway) * 100
                )}% implied)
              </div>
              <div className="small-text">Not a betting recommendation.</div>
            </div>
          )}

          {disagreement && (
            <div className="section disagreement-note">
              <div className="section-title">Model vs. Vegas</div>
              <div>Our model favors {disagreement.ourFavorite}; Vegas favors {disagreement.vegasFavorite}.</div>
              <div className="small-text">Not a recommendation - real backtest ROI on disagreements: -36%.</div>
            </div>
          )}

          {homeElo !== null && homeElo !== undefined && awayElo !== null && awayElo !== undefined && (
            <div className="section">
              <div className="section-title">Team Elo Ratings</div>
              <div className="elo-metrics">
                <div className="team-elo-block">
                  <div className="team-elo-name">{home}</div>
                  <div className="metric">
                    <span className="label">Single Elo</span>
                    <span className="value">{Math.round(homeElo)}{rankLabel(home, singleEloRanks)}</span>
                  </div>
                  {homeOElo !== null && homeOElo !== undefined && (
                    <div className="metric">
                      <span className="label">O_Elo</span>
                      <span className="value">{Math.round(homeOElo)}{rankLabel(home, oEloRanks)}</span>
                    </div>
                  )}
                  {homeDElo !== null && homeDElo !== undefined && (
                    <div className="metric">
                      <span className="label">D_Elo</span>
                      <span className="value">{Math.round(homeDElo)}{rankLabel(home, dEloRanks)}</span>
                    </div>
                  )}
                </div>
                <div className="team-elo-block">
                  <div className="team-elo-name">{away}</div>
                  <div className="metric">
                    <span className="label">Single Elo</span>
                    <span className="value">{Math.round(awayElo)}{rankLabel(away, singleEloRanks)}</span>
                  </div>
                  {awayOElo !== null && awayOElo !== undefined && (
                    <div className="metric">
                      <span className="label">O_Elo</span>
                      <span className="value">{Math.round(awayOElo)}{rankLabel(away, oEloRanks)}</span>
                    </div>
                  )}
                  {awayDElo !== null && awayDElo !== undefined && (
                    <div className="metric">
                      <span className="label">D_Elo</span>
                      <span className="value">{Math.round(awayDElo)}{rankLabel(away, dEloRanks)}</span>
                    </div>
                  )}
                </div>
              </div>
              {homeOElo !== null && homeOElo !== undefined && awayOElo !== null && awayOElo !== undefined && (
                <div className="small-text">
                  {home} off vs {away} def: {(homeOElo - awayDElo) > 0 ? '+' : ''}{Math.round(homeOElo - awayDElo)}.{' '}
                  {away} off vs {home} def: {(awayOElo - homeDElo) > 0 ? '+' : ''}{Math.round(awayOElo - homeDElo)}.
                </div>
              )}
            </div>
          )}

          {qbPassingTDs && (qbPassingTDs[home] || qbPassingTDs[away]) && (
            <div className="section qb-passing-tds">
              <div className="section-title">Passing Touchdowns</div>
              <div className="qb-grid">
                <QBPassingTDCard team={away} qb={qbPassingTDs[away]} />
                <QBPassingTDCard team={home} qb={qbPassingTDs[home]} />
              </div>
              <div className="small-text">
                Real P(1+ pass TD) per real starting QB (logistic regression, real AUC 0.60-0.70 -
                see td_props_logistic_models.json). Expected TD count is DERIVED from that real
                probability (-ln(1-p), standard Poisson relationship), not a separate model - this
                project already found and removed a real linear expected-count model for being
                non-predictive (real R² 0.037-0.139). Not a betting recommendation.
              </div>
            </div>
          )}

          {topScorers && (topScorers[home]?.length > 0 || topScorers[away]?.length > 0) && (
            <div className="section top-scorers">
              <div className="section-title">Top Rushing/Receiving TD Scorers</div>
              <div className="scorers-grid">
                <TeamScorers team={away} scorers={topScorers[away]} />
                <TeamScorers team={home} scorers={topScorers[home]} />
              </div>
              <div className="small-text">
                Real P(1+ TD) per player this game, rushing/receiving only (passing TDs are shown
                separately above) - logistic regression, real AUC 0.60-0.70 by position/TD-type
                (see td_props_logistic_models.json), converted to standard American odds. A real
                rushing QB can still appear here via their own real rushing_tds_prob. Not a
                betting recommendation.
              </div>
            </div>
          )}

          {(homeQb || awayQb) && (
            <div className="section">
              <div className="section-title">Starting QBs</div>
              <div>{away}: {awayQb || 'unknown'}</div>
              <div>{home}: {homeQb || 'unknown'}</div>
            </div>
          )}

          {(homeForm?.length > 0 || awayForm?.length > 0) && (
            <div className="section">
              <div className="section-title">Recent Form (last {Math.max(homeForm?.length || 0, awayForm?.length || 0)} real games)</div>
              <div>{away}: {awayForm?.length ? awayForm.join('-') : 'no prior real games'}</div>
              <div>{home}: {homeForm?.length ? homeForm.join('-') : 'no prior real games'}</div>
            </div>
          )}

          {headToHead && headToHead.meetings_considered > 0 && (
            <div className="section">
              <div className="section-title">Head-to-Head (last {headToHead.meetings_considered} real meetings)</div>
              <div>
                {home} {headToHead.home_team_wins}-{headToHead.away_team_wins}
                {headToHead.ties > 0 ? `-${headToHead.ties}` : ''} vs {away}
              </div>
            </div>
          )}

          {hasResult && (
            <div className="section">
              <div className="section-title">Actual Result</div>
              <div>
                {away} {awayScore} @ {home} {homeScore}
              </div>
              <div>Winner: {actualWinner === 'TIE' ? 'Tie' : teamName(actualWinner)}</div>
              {correct !== null && (
                <div>Prediction: {correct ? '✅ Correct (straight-up winner)' : '❌ Incorrect'}</div>
              )}
              {correct === null && <div>Prediction: push / tie - not scoreable</div>}
            </div>
          )}

          {hasResult && (ats || winProbFavorite) && (
            <div className="section">
              <div className="section-title">Betting Outcome</div>
              <div className="small-text" style={{ marginBottom: '8px' }}>
                Real historical facts for this completed game, not predictions or advice.
              </div>

              {ats && (
                <div className="betting-outcome">
                  <div className="outcome-row">
                    <span className="outcome-label">Spread</span>
                    <span className="outcome-value">
                      Vegas {vegasSpread > 0 ? '+' : ''}{vegasSpread.toFixed(1)} ({home})
                    </span>
                  </div>
                  <div className="outcome-result">
                    {ats === 'push' ? (
                      'Push - exact number, no cover either way'
                    ) : (
                      <>
                        <strong>{ats === 'home' ? home : away}</strong> covered
                        <span className="result-score">
                          {home} {homeScore} - {away} {awayScore}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}

              {winProbFavorite && winProbHit !== null && (
                <div className="betting-outcome">
                  <div className="outcome-row">
                    <span className="outcome-label">Moneyline</span>
                    <span className="outcome-value">
                      Model favored {winProbFavorite} ({Math.round(
                        (winProbFavorite === home ? winProbHome : winProbAway) * 100
                      )}%)
                    </span>
                  </div>
                  <div className="outcome-result">
                    {winProbHit ? (
                      <span className="result-hit">✅ Correct ({teamName(actualWinner)} won)</span>
                    ) : (
                      <span className="result-miss">❌ Incorrect ({teamName(actualWinner)} won)</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function QBPassingTDCard({ team, qb }) {
  return (
    <div className="qb-card">
      <h4>{team}</h4>
      {qb ? (
        <div className="qb-info">
          <span className="qb-name">{qb.player_name}</span>
          <div className="qb-stats">
            <span className="qb-pass-count">{qb.expected_passing_tds.toFixed(1)} pass TDs</span>
            <span className="qb-pass-prob">({Math.round(qb.passing_td_prob * 100)}% to score 1+)</span>
          </div>
        </div>
      ) : (
        <div className="no-data-note">No real QB props for this team yet.</div>
      )}
    </div>
  );
}

function TeamScorers({ team, scorers }) {
  return (
    <div className="team-scorers">
      <h4>{team}</h4>
      {scorers && scorers.length > 0 ? (
        <div className="scorers-list">
          {scorers.map((s) => (
            <div key={s.player_id} className="scorer-row">
              <span className="scorer-info">
                <span className="scorer-name">{s.player_name}</span>
                <span className="scorer-position">{s.position} · {s.td_type}</span>
              </span>
              <span className="scorer-odds">
                <span className="scorer-prob">{Math.round(s.td_prob * 100)}%</span>
                {s.implied_odds != null && (
                  <span className="scorer-implied-odds">
                    {s.implied_odds > 0 ? '+' : ''}{s.implied_odds}
                  </span>
                )}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-data-note">No real player props for this team yet.</div>
      )}
    </div>
  );
}
