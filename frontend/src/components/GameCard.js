import React, { useState } from 'react';
import { teamName, teamColor, teamSecondaryColor, readableTextColor, TEAM_TIMEZONES, TIMEZONE_LABELS } from '../constants/teams';
import { useKeyboardToggle } from '../hooks/useKeyboardToggle';

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

// Real fact, verified against this project's own 2025 schedule data before
// building this: kickoff_datetime's time-of-day is stored in Eastern Time
// (nflverse's documented schedules.gametime convention), not UTC and not
// each game's own stadium-local time - e.g. real LV (Pacific) home games
// show gametime 16:05/16:25, the real ET value for the real 1:05/1:25 PM
// Pacific "late window" kickoff, not an evening LV-local time.
//
// This matters because `new Date("2025-09-04T20:20:00")` (no UTC offset in
// the string) parses those digits as the VIEWER's OWN local time, not ET -
// so a bare `new Date(kickoff_datetime)` (what this component did before)
// silently just echoes the raw ET digits back to every viewer as if they
// were already correct, rather than converting anything. Fixed by
// attaching the real ET UTC offset before parsing, so downstream
// conversions to any real viewer timezone are actually correct.
const DST_2025_FALLBACK_DATE = '2025-11-02'; // real 2025 US DST end date

function etUtcOffsetHours(kickoffISO) {
  const gameDate = kickoffISO.slice(0, 10);
  return gameDate >= DST_2025_FALLBACK_DATE ? 5 : 4; // EST (-5) on/after, EDT (-4) before
}

const STADIUM_HOURS_BEHIND_ET = { ET: 0, CT: 1, MT: 2, PT: 3 };

function stadiumOffsetHours(homeTeam, kickoffISO) {
  if (homeTeam === 'ARI') {
    // Real fact: Arizona doesn't observe DST, so its real offset from ET
    // isn't constant like the other zones - it matches Pacific (3h behind
    // ET) during the real EDT months, then Mountain (2h behind ET) after
    // the real Nov 2 fall-back, even though TEAM_TIMEZONES labels it "MT"
    // year-round per convention.
    return etUtcOffsetHours(kickoffISO) === 4 ? 3 : 2;
  }
  return STADIUM_HOURS_BEHIND_ET[TEAM_TIMEZONES[homeTeam]] ?? 0;
}

function toAbsoluteKickoffDate(kickoffISO) {
  const offsetHours = etUtcOffsetHours(kickoffISO);
  const d = new Date(`${kickoffISO}-0${offsetHours}:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatKickoffDate(kickoffISO) {
  if (!kickoffISO) return '';
  const [y, m, day] = kickoffISO.slice(0, 10).split('-').map(Number);
  const d = new Date(y, m - 1, day); // calendar date only - no timezone/instant involved
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function formatKickoffTime(kickoffISO, homeTeam, showUserTime) {
  if (!kickoffISO) return 'TBD';
  const timePart = kickoffISO.slice(11, 16);
  if (!timePart) return 'TBD';

  if (!showUserTime) {
    // Game Time: real stadium-local clock, derived directly from the real
    // ET source value + the real fixed hour offset for that stadium's zone
    // (all real US mainland zones shift DST together, so CT/MT/PT stay a
    // constant 1/2/3 hours behind ET all season - only Arizona varies, see
    // stadiumOffsetHours).
    const [etHour, etMin] = timePart.split(':').map(Number);
    const offset = stadiumOffsetHours(homeTeam, kickoffISO);
    const localHour = (etHour - offset + 24) % 24;
    const meridiem = localHour >= 12 ? 'PM' : 'AM';
    const displayHour = localHour % 12 || 12;
    return `${displayHour}:${etMin.toString().padStart(2, '0')} ${meridiem}`;
  }

  const absolute = toAbsoluteKickoffDate(kickoffISO);
  if (!absolute) return 'TBD';
  let userTimezone = 'UTC';
  try {
    userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch (e) {
    userTimezone = 'UTC';
  }
  return absolute.toLocaleString('en-US', {
    timeZone: userTimezone,
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

export default function GameCard({ game, isExpanded, onToggle }) {
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
    ci_low_90: ciLow90,
    ci_high_90: ciHigh90,
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
            {formatKickoffDate(game.kickoff_datetime)} · {formatKickoffTime(game.kickoff_datetime, home, showUserTime)}
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
          <div className="our-spread">
            <span className="label">Our</span> {formatSpread(ourSpread, home, away)}
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
            <div>Our spread: {formatSpread(ourSpread, home, away)}</div>
            {ciLow90 !== null && ciLow90 !== undefined && ciHigh90 !== null && ciHigh90 !== undefined && (
              <div className="game-card__confidence-interval">
                90% CI: {ciLow90.toFixed(1)} to {ciHigh90.toFixed(1)} (home-team spread points, from the
                real fitted model's own residual std - not a Vegas-derived number)
              </div>
            )}
            {vegasSpread !== null && vegasSpread !== undefined && (
              <div>Vegas closing line: {formatSpread(vegasSpread, home, away)}</div>
            )}
            {singleEloSpread !== null && singleEloSpread !== undefined && (
              <div className="small-text">
                For comparison, single-Elo alone (team strength only, no offense/defense split)
                would have predicted {formatSpread(singleEloSpread, home, away)}
                {singleEloWinProbHome !== null && singleEloWinProbHome !== undefined && (
                  <> ({Math.round((favoriteTeam(singleEloSpread, home, away) === home
                    ? singleEloWinProbHome : 1 - singleEloWinProbHome) * 100)}% implied)</>
                )} - real, from the same fitted single-Elo model this project used before the O/D
                swap, not a raw Elo-rating difference.
              </div>
            )}
            <div>Source: {baseSource === 'vegas' ? 'Vegas line + matchup adjustment' : 'Elo fallback (no posted line)'}</div>
            {netEdgeDiff !== null && netEdgeDiff !== undefined && Math.abs(netEdgeDiff) > 0.001 && (
              <div>Matchup EPA edge differential: {netEdgeDiff > 0 ? '+' : ''}{netEdgeDiff.toFixed(2)} (home perspective)</div>
            )}
            {matchupQuality && MATCHUP_QUALITY_DISPLAY[matchupQuality] && (
              <div>
                {MATCHUP_QUALITY_DISPLAY[matchupQuality].emoji} {MATCHUP_QUALITY_DISPLAY[matchupQuality].label}
                <span className="small-text"> (empirical tercile of this season's real matchup-edge distribution)</span>
              </div>
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
              <div className="small-text">
                Real, honest finding from this project&apos;s own backtest: Elo/week/season carry
                essentially no real signal for total points (R² 0.005, directional accuracy ~50% at
                every edge size tested, 0.5-5 points, on real held-out data). Shown as weak,
                informational context only - not a betting recommendation, and no confidence/alert
                threshold is implied.
              </div>
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
              <div className="small-text">
                {baseSource === 'vegas' ? (
                  <>
                    From a real backtested model (logistic regression fit on real 2024 Vegas
                    spreads vs. real outcomes; real 2025 weeks 13-17 holdout Brier score 0.2512 -
                    beat both an Elo-based model and a simple heuristic; see backtesting_results.md).
                  </>
                ) : (
                  <>
                    No real Vegas line exists for this game (real preseason matchup, before that
                    model&apos;s own input data exists) - this uses this project&apos;s real
                    offensive/defensive Elo split instead (real held-out 2024 Brier score 0.2182,
                    beating the prior single-Elo model&apos;s 0.2272 on the same real test - see
                    od_elo_production_validation.json).
                  </>
                )}{' '}
                Not a betting recommendation.
              </div>
            </div>
          )}

          {disagreement && (
            <div className="section disagreement-note">
              <div className="section-title">Model vs. Vegas</div>
              <div>Our model favors {disagreement.ourFavorite}; Vegas favors {disagreement.vegasFavorite}.</div>
              <div className="small-text">
                Shown for transparency only, not a recommendation - this project's own real
                backtest (edge_detection.py) found that betting on disagreements like this one
                produced -36% ROI (actively harmful), not a real edge.
              </div>
            </div>
          )}

          {homeElo !== null && homeElo !== undefined && awayElo !== null && awayElo !== undefined && (
            <div className="section">
              <div className="section-title">Matchup Strength (Elo)</div>
              <div>{home}: {Math.round(homeElo)}</div>
              <div>{away}: {Math.round(awayElo)}</div>
              <div className="small-text">
                Difference: {Math.round(Math.abs(homeElo - awayElo))} points
                ({homeElo > awayElo ? home : away} favored) - real, from elo_ratings_2025.csv,
                lagged one real week (entering-this-week rating, leak-free)
              </div>
            </div>
          )}

          {homeOElo !== null && homeOElo !== undefined && awayOElo !== null && awayOElo !== undefined && (
            <div className="section">
              <div className="section-title">Offense/Defense Elo Split</div>
              <div>{home}: {Math.round(homeOElo)} off / {Math.round(homeDElo)} def</div>
              <div>{away}: {Math.round(awayOElo)} off / {Math.round(awayDElo)} def</div>
              <div className="small-text">
                {home}&apos;s offense vs. {away}&apos;s defense: {(homeOElo - awayDElo) > 0 ? '+' : ''}
                {Math.round(homeOElo - awayDElo)}. {away}&apos;s offense vs. {home}&apos;s defense:{' '}
                {(awayOElo - homeDElo) > 0 ? '+' : ''}{Math.round(awayOElo - homeDElo)}. The gap between
                these two real terms is what separates this prediction from single-Elo above, which only
                sees each team&apos;s one combined rating and can&apos;t tell an elite offense facing a
                weak defense apart from an average matchup at the same net strength.
              </div>
              <div className="small-text">
                This split now generates the spread, win probability, and confidence interval above
                for real 2026 preseason games (real, held-out 2024 validation found it beats the
                prior single-Elo model on win/loss accuracy, spread MAE, and Brier score - see
                od_elo_production_validation.json for the full comparison). The single Elo rating
                shown in Matchup Strength above is real but no longer drives the prediction - shown
                as additional context only. One real, disclosed trade-off accepted with this swap:
                this split&apos;s 90% confidence bands were only 86.4% covered on the 2024 holdout
                (vs. the prior model&apos;s 89.3%, closer to target) - bands are a bit tighter than
                ideal.
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
