import React, { useState, useEffect, useMemo } from 'react';
import '../styles/FantasyRankings.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import { teamName, teamColor, teamSecondaryColor, readableTextColor } from '../constants/teams';
import SeasonDataUnavailable from './SeasonDataUnavailable';
import { useKeyboardToggle } from '../hooks/useKeyboardToggle';

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

// Real per-team Defensive Elo + rank for a single selected week, same real
// per-week (not per-season) methodology as GamePredictions.js's
// realEloRanksForWeek - checked directly there, real single/O/D Elo can
// move week-to-week across a real completed season, so a season-wide map
// would silently rank a team using whichever week's game happened to be
// iterated last. Not imported from GamePredictions.js (not exported, and
// this only needs the D_Elo half) - same real fields (home_d_elo/
// away_d_elo/home_team/away_team), independently computed here.
function realDEloRanksForWeek(weekGames) {
  const byTeam = new Map();
  for (const g of weekGames) {
    if (g.home_d_elo != null) byTeam.set(g.home_team, g.home_d_elo);
    if (g.away_d_elo != null) byTeam.set(g.away_team, g.away_d_elo);
  }
  const ranked = [...byTeam.entries()].sort((a, b) => b[1] - a[1]);
  const ranks = {};
  ranked.forEach(([team, d_elo], i) => {
    ranks[team] = { rank: i + 1, total: ranked.length, d_elo };
  });
  return ranks;
}

// RB/QB/TE: real per-week trailing projections. WR: real trailing up-to-
// 4-week actual-PPR average (Phase 4 backtest winner, corr +0.4416 vs. the
// old static baseline's real leak-free +0.3957 - see
// DASHBOARD_DATA_GAPS.md) once a player has a real prior 2025 week; a
// player's first real 2025 appearance falls back to the real static
// season-long projection - see PlayerCard's per-row `projection_type`
// handling and generate_fantasy_dashboard_data.py.
const POSITIONS = ['RB', 'WR', 'QB', 'TE'];

// injury_status is a simplified 3-tier bucket of the real nflreadpy
// report_status ("Doubtful" folded into "out" - see generate_fantasy_
// dashboard_data.py._simplify_injury_status). injury_status_raw (shown in
// the expanded view) preserves the exact original string.
const INJURY_EMOJI = { healthy: '🟢', questionable: '🟡', out: '🔴' };

// accuracy_tier comes from the backend, bucketed via EMPIRICAL PER-POSITION
// terciles of the real |actual - projected| distribution (see
// generate_fantasy_dashboard_data.py._accuracy_tier_thresholds) - not a
// flat asserted +-2/+-5 threshold, since QB/RB point scales differ from TE's.
const ACCURACY_DISPLAY = {
  green: { emoji: '🟢', label: 'Close projection', className: 'accuracy-green' },
  yellow: { emoji: '🟡', label: 'Moderate projection error', className: 'accuracy-yellow' },
  red: { emoji: '🔴', label: 'Large projection error', className: 'accuracy-red' },
};

// Real per-position tercile boundaries (|actual-projected|, PPR), a point-
// in-time snapshot of generate_fantasy_dashboard_data.py's live-computed
// _accuracy_tier_thresholds() output for this project's real 2025 data -
// verified to match by re-running that function directly before adding
// this. Shown for context only; each player's own accuracy_tier below
// still comes straight from the backend, not recomputed against these
// numbers, since the real thresholds are recomputed fresh every pipeline
// run and could drift slightly from this snapshot.
// Re-verified after Phase 4 (WR's real error distribution tightened once
// its projection became dynamic - real WR boundary moved from ±3.2/±6.6 to
// ±2.7/±6.4).
const ACCURACY_TERCILE_RANGES = {
  QB: '±3.3 / ±8.6 PPR',
  RB: '±1.4 / ±4.7 PPR',
  WR: '±2.7 / ±6.4 PPR',
  TE: '±1.4 / ±4.1 PPR',
};

// Real per-position prop stat display order/labels, matching the real
// per-position targets train_player_props_models.py actually trained
// (see that script - a QB has no real receiving props, a WR/TE's real
// rushing_yards prop is a near-zero real signal shown for completeness,
// not because it's a meaningful part of their real role).
// Real per-position prop stat display order/labels, matching the real
// per-position targets train_player_props_models.py/train_td_logistic_
// models.py actually trained. TD-type stats use `_prob` keys (real
// logistic P(1+ TD), Major Refinements task) rendered as a percentage,
// not a fractional expected count - real 5-fold OOF R^2 0.037-0.139 on
// the old linear approach confirmed a "1.2 TDs projected" number wasn't
// meaningfully predictive or actionable (see train_player_props_models.py/
// train_td_logistic_models.py docstrings for the real before/after).
const PROP_STAT_LABELS = {
  QB: [
    ['completions', 'Comp'],
    ['passing_yards', 'Pass Yds'],
    ['passing_tds_prob', '1+ Pass TD'],
    ['rushing_tds_prob', '1+ Rush TD'],
  ],
  RB: [
    ['rushing_yards', 'Rush Yds'],
    ['rushing_tds_prob', '1+ Rush TD'],
    ['receptions', 'Rec'],
    ['receiving_yards', 'Rec Yds'],
  ],
  WR: [
    ['receptions', 'Rec'],
    ['receiving_yards', 'Rec Yds'],
    ['receiving_tds_prob', '1+ Rec TD'],
    ['rushing_yards', 'Rush Yds'],
  ],
  TE: [
    ['receptions', 'Rec'],
    ['receiving_yards', 'Rec Yds'],
    ['receiving_tds_prob', '1+ Rec TD'],
    ['rushing_yards', 'Rush Yds'],
  ],
};

// Real note: 2026 rookie-class scores (rookie_scores_2026.json) use a
// real, different player-ID scheme (nflreadpy draft-pick IDs like
// "HUR541377") than the veteran fantasy_rankings_2026.json roster (real
// gsis_ids like "00-0034857") - checked directly, zero overlap between
// the two real ID sets, because a true rookie has no real 2015-2025 EPA
// history to build a veteran roster row from at all (see generate_
// player_props_2026.py's own disclosed rookie-exclusion gap). A per-row
// "rookie badge" on the existing veteran list is therefore not real -
// there's nothing there to badge. This is shown as its own real,
// standalone section instead, the same real "separate export, don't
// force it into an unrelated real number" precedent score_2026_rookies.py
// itself already documents for why this stays out of trade_scores_2026.json.
function RookieSection({ rookieScores }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState('ALL');

  const rookies = Object.entries(rookieScores.players)
    .map(([id, r]) => ({ id, ...r }))
    .filter((r) => position === 'ALL' || r.position === position)
    .sort((a, b) => b.success_probability - a.success_probability);

  return (
    <div className="rookie-section">
      <button type="button" className="rookie-section__toggle" onClick={() => setOpen(!open)}>
        {open ? '▲' : '▼'} 2026 Rookie Class ({Object.keys(rookieScores.players).length} real QB/RB/WR/TE draftees)
      </button>
      {open && (
        <div className="rookie-section__body">
          <div className="selector">
            <label htmlFor="rookie-position-select">Position</label>
            <select
              id="rookie-position-select"
              value={position}
              onChange={(e) => setPosition(e.target.value)}
            >
              <option value="ALL">All</option>
              {POSITIONS.map((pos) => (
                <option key={pos} value={pos}>{pos}</option>
              ))}
            </select>
          </div>
          <div className="rookie-list">
            {rookies.map((r) => (
              <div key={r.id} className="rookie-row">
                <span className="rookie-name">{r.name}</span>
                <span
                  className="team-box"
                  style={{ backgroundColor: teamSecondaryColor(r.team), color: readableTextColor(teamSecondaryColor(r.team)) }}
                >
                  {r.team}
                </span>
                <span className="rookie-pos">{r.position}</span>
                <span className="rookie-draft">Round {r.draft_round}, Pick {r.draft_pick}</span>
                <span className="rookie-prob" title="Real P(this player's career value beats their real draft round's historical median) - draft-time-only signals, real held-out AUC 0.563">
                  {Math.round(r.success_probability * 100)}%
                </span>
                {r.success_probability_enhanced != null && (
                  <span className="rookie-prob rookie-prob--enhanced" title="Real, combine-data-enhanced version of the same estimate - held-out AUC 0.605, only available for real rookies with complete real combine testing (37% of the 2026 class)">
                    {Math.round(r.success_probability_enhanced * 100)}% (combine-adjusted)
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function FantasyRankings() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const fantasyData = seasonData.fantasy;
  // Real per-player-game stat projections (Player Props Model), keyed by
  // the same real `{player_id}_w{week}` id format fantasy_rankings_*.json
  // already uses - null for seasons this wasn't built for (see
  // SeasonContext.js).
  const propsById = seasonData.playerProps
    ? new Map(seasonData.playerProps.map((p) => [p.id, p]))
    : null;
  // Real data exists for 2026 (Week 1 preseason projections), unlike the
  // other sections gated by the blanket `hasResults` flag - checking for
  // real fantasy data specifically, not the season-wide flag, is what
  // actually determines whether this section has anything to show.
  const isPreseason = !hasResults;

  const weeks = fantasyData ? [...new Set(fantasyData.map((p) => p.week))].sort((a, b) => a - b) : [];
  const [selectedWeek, setSelectedWeek] = useState(weeks[0]);
  const [selectedPosition, setSelectedPosition] = useState('RB');
  const [expandedPlayerId, setExpandedPlayerId] = useState(null);

  // Real breakout alerts are keyed by week (string) at the top level, not
  // a flat list like playerProps - only the selected week's alerts are
  // relevant here, so the per-id lookup is rebuilt from that week's slice.
  const weekBreakoutAlerts = seasonData.breakoutAlerts ? seasonData.breakoutAlerts[String(selectedWeek)] : null;
  const breakoutById = weekBreakoutAlerts
    ? new Map(weekBreakoutAlerts.map((a) => [a.id, a]))
    : null;

  // Real opponent D_Elo/rank for the selected week - each player's real
  // `opponent` field already comes precomputed from the backend (see
  // generate_fantasy_rankings_2026_week1.py's real
  // _real_week1_opponents_2026()), so this only needs a real team ->
  // D_Elo/rank lookup, not the game-matching logic a from-scratch version
  // would otherwise need.
  const dEloRanks = useMemo(
    () => realDEloRanksForWeek(seasonData.games.filter((g) => g.week === selectedWeek)),
    [seasonData.games, selectedWeek]);

  useEffect(() => {
    setSelectedWeek(weeks[0]);
    setExpandedPlayerId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSeason]);

  if (!fantasyData) {
    return (
      <div className="fantasy-rankings">
        <SeasonDataUnavailable season={selectedSeason} sectionName="Fantasy Rankings" />
      </div>
    );
  }

  const players = fantasyData
    .filter((p) => p.week === selectedWeek && p.position === selectedPosition)
    .slice()
    .sort((a, b) => a.rank - b.rank);

  return (
    <div className="fantasy-rankings">
      <div className="header">
        <h1>Fantasy Rankings</h1>
        {isPreseason && (
          <div className="preseason-banner">
            <span className="preseason-banner-title">📋 Week 1 Preseason Projections</span>
            <span className="preseason-banner-note">
              No real {selectedSeason} games played yet - see How This Model Works for methodology.
            </span>
          </div>
        )}
        <div className="controls">
          <div className="selector">
            <label htmlFor="fantasy-week-select">Week</label>
            <select
              id="fantasy-week-select"
              value={selectedWeek}
              onChange={(e) => {
                setSelectedWeek(Number(e.target.value));
                setExpandedPlayerId(null);
              }}
            >
              {weeks.map((w) => (
                <option key={w} value={w}>
                  Week {w}
                </option>
              ))}
            </select>
          </div>
          <div className="selector">
            <label htmlFor="fantasy-position-select">Position</label>
            <select
              id="fantasy-position-select"
              value={selectedPosition}
              onChange={(e) => {
                setSelectedPosition(e.target.value);
                setExpandedPlayerId(null);
              }}
            >
              {POSITIONS.map((pos) => (
                <option key={pos} value={pos}>
                  {pos}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="players-container">
        {players.length === 0 ? (
          <p className="empty-state">No real projection data for this week/position.</p>
        ) : (
          players.map((player) => (
            <PlayerCard
              key={player.id}
              player={player}
              isExpanded={expandedPlayerId === player.id}
              onToggle={() => setExpandedPlayerId(expandedPlayerId === player.id ? null : player.id)}
              props={propsById ? propsById.get(player.id) : null}
              breakoutAlert={breakoutById ? breakoutById.get(player.id) : null}
              opponentDElo={player.opponent ? dEloRanks[player.opponent] : null}
            />
          ))
        )}
      </div>

      {seasonData.rookieScores && <RookieSection rookieScores={seasonData.rookieScores} />}

      <div className="disclaimer">
        <p>
          {isPreseason
            ? `Real ${selectedSeason} Week 1 preseason projections - the season hasn't started yet.`
            : `Real ${selectedSeason} backtest data.`}{' '}
          See How This Model Works for full methodology.
        </p>
      </div>
    </div>
  );
}

function PlayerCard({ player, isExpanded, onToggle, props, breakoutAlert, opponentDElo }) {
  const borderColor = teamColor(player.team);
  const isStatic = player.projection_type === 'season_static_per_game_avg';
  const handleKeyDown = useKeyboardToggle(onToggle);

  return (
    <div
      className={`player-card ${isExpanded ? 'player-card-open' : ''}`}
      style={{ borderColor }}
      onClick={onToggle}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-expanded={isExpanded}
    >
      <div className="player-card-collapsed">
        <div className="rank-name">
          <span className="rank">#{player.rank}</span>
          <span className="name">{player.name}</span>
          <span
            className="team-box"
            style={{
              backgroundColor: teamSecondaryColor(player.team),
              color: readableTextColor(teamSecondaryColor(player.team)),
            }}
          >
            {player.team}
          </span>
          {breakoutAlert && (
            <span className="breakout-badge" title={breakoutAlert.recommendation}>
              <span className="breakout-icon">⚡</span>
              <span className="breakout-text">Breakout</span>
              <span className="confidence">{Math.round(breakoutAlert.confidence * 100)}%</span>
            </span>
          )}
          {player.confidence_tier === 'lower' && (
            <span
              className="confidence-tier-badge"
              title="Real 2025 opportunities below this position's validated projection threshold - included, but a real, meaningfully weaker signal than the rest of the list (see Fantasy Predictions in How This Model Works)."
            >
              Lower confidence
            </span>
          )}
        </div>

        <div className="ppr">
          {player.projected_ppr != null ? player.projected_ppr.toFixed(1) : '--'} PPR{isStatic ? ' (season avg)' : ''}
        </div>

        {player.opponent_defense_rank_vs_position !== null && player.opponent_defense_rank_vs_position !== undefined && (
          <div className="def-rank" title={`Opponent (${player.opponent}) real trailing defense rank vs ${player.position}`}>
            vs #{player.opponent_defense_rank_vs_position}
          </div>
        )}

        {opponentDElo && (
          <div
            className="opponent-d-elo"
            title={`Real ${player.opponent} Defensive Elo this week: ${Math.round(opponentDElo.d_elo)} (${ordinal(opponentDElo.rank)} of ${opponentDElo.total} real teams, 1 = strongest real defense)`}
          >
            vs {player.opponent} ({Math.round(opponentDElo.d_elo)} D_Elo, {ordinal(opponentDElo.rank)} best)
          </div>
        )}

        {player.injury_status && player.injury_status !== 'healthy' && (
          <div className="injury-badge" title={player.injury_status_raw || player.injury_status}>
            {INJURY_EMOJI[player.injury_status] || '⚪'}
          </div>
        )}

        <div className="expand-hint">{isExpanded ? '▲' : '▼'}</div>
      </div>

      {isExpanded && (
        <div className="player-card-expanded" onClick={(e) => e.stopPropagation()}>
          <h3>
            {player.name} ({teamName(player.team)})
          </h3>

          <div className="section">
            <div className="section-title">Projection</div>
            <div>
              {player.projected_ppr != null ? player.projected_ppr.toFixed(1) : '--'} projected PPR points
              {isStatic ? ' (season avg)' : ''}
            </div>
            {player.confidence_tier === 'lower' && (
              <div className="small-text">
                Lower confidence: this player&apos;s real 2025 opportunities (targets/carries/attempts) were
                below this position&apos;s validated projection threshold - included for real roster
                coverage, but a real, meaningfully weaker signal than the rest of the list.
              </div>
            )}
          </div>

          {player.actual_ppr !== null && player.actual_ppr !== undefined && (
            <div className="section">
              <div className="section-title">Game Result vs. Projection</div>
              <div className="accuracy-comparison">
                <div className="accuracy-row">
                  <span className="label">Projected PPR</span>
                  <span className="value projected">{player.projected_ppr != null ? player.projected_ppr.toFixed(1) : '--'}</span>
                </div>
                <div className="accuracy-row">
                  <span className="label">Actual PPR</span>
                  <span className="value actual">{player.actual_ppr.toFixed(1)}</span>
                </div>
                <div className="accuracy-row">
                  <span className="label">Difference</span>
                  <span className={`value difference ${player.accuracy_tier ? ACCURACY_DISPLAY[player.accuracy_tier].className : ''}`}>
                    {player.actual_ppr - player.projected_ppr > 0 ? '+' : ''}
                    {(player.actual_ppr - player.projected_ppr).toFixed(1)}
                  </span>
                </div>
              </div>
              {player.accuracy_tier && (
                <div className="accuracy-indicator">
                  {ACCURACY_DISPLAY[player.accuracy_tier].emoji} {ACCURACY_DISPLAY[player.accuracy_tier].label}
                </div>
              )}
              {ACCURACY_TERCILE_RANGES[player.position] && (
                <div className="accuracy-ranges">
                  <span className="label">Real {player.position} green/yellow boundary this season:</span>
                  <span className="ranges">{ACCURACY_TERCILE_RANGES[player.position]}</span>
                </div>
              )}
            </div>
          )}

          {player.opponent && (
            <div className="section">
              <div className="section-title">Matchup</div>
              <div>vs {teamName(player.opponent)}</div>
              {player.opponent_defense_rank_vs_position !== null && player.opponent_defense_rank_vs_position !== undefined && (
                <div className="small-text">
                  Real trailing defense rank vs {player.position}: #{player.opponent_defense_rank_vs_position} of 32
                </div>
              )}
            </div>
          )}

          {props && PROP_STAT_LABELS[player.position] && (
            <div className="section">
              <div className="section-title">Projected Stats</div>
              <div className="props-grid">
                {PROP_STAT_LABELS[player.position].map(([key, label]) => {
                  const isProb = key.endsWith('_prob');
                  const value = props.predicted_stats[key];
                  return (
                    <div key={key} className="prop-stat">
                      <span className="prop-label">{label}</span>
                      <span className="prop-value">
                        {value == null ? '--' : isProb ? `${Math.round(value * 100)}%` : value}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {breakoutAlert && (
            <div className="section breakout-details">
              <div className="section-title">Breakout Alert Signals</div>
              <div className="signals-breakdown">
                {breakoutAlert.signals.weak_defense.active && (
                  <div className="signal weak-defense">
                    <span className="signal-icon">🛡️</span>
                    <span className="signal-label">Weak Defense</span>
                    <span className="signal-value">
                      {breakoutAlert.signals.weak_defense.opponent_d_elo} D_Elo - weaker than{' '}
                      {breakoutAlert.signals.weak_defense.weaker_than_pct_of_league}% of the real
                      32-team 2026 league
                    </span>
                  </div>
                )}
                {breakoutAlert.signals.usage_trending_up.active && (
                  <div className="signal usage-up">
                    <span className="signal-icon">📈</span>
                    <span className="signal-label">Usage Trending Up</span>
                    <span className="signal-value">
                      Real 2025 trailing snap-share trend: {breakoutAlert.signals.usage_trending_up.snap_pct_trend > 0 ? '+' : ''}
                      {breakoutAlert.signals.usage_trending_up.snap_pct_trend} pts (top real tercile of the league)
                    </span>
                  </div>
                )}
                {breakoutAlert.signals.performance_strong.active && (
                  <div className="signal performance">
                    <span className="signal-icon">⭐</span>
                    <span className="signal-label">Strong Recent Form</span>
                    <span className="signal-value">
                      Real last-4-game 2025 PPR {breakoutAlert.signals.performance_strong.last_4_ppr_2025} vs.
                      real season average {breakoutAlert.signals.performance_strong.season_avg_ppr_2025} ({breakoutAlert.signals.performance_strong.diff > 0 ? '+' : ''}
                      {breakoutAlert.signals.performance_strong.diff})
                    </span>
                  </div>
                )}
              </div>
              <div className="small-text">{breakoutAlert.recommendation}</div>
            </div>
          )}

          {player.injury_status && (
            <div className="section">
              <div className="section-title">Injury Status</div>
              <div>
                {INJURY_EMOJI[player.injury_status] || '⚪'} {player.injury_status}
                {player.injury_status_raw ? ` (${player.injury_status_raw})` : ''}
              </div>
            </div>
          )}

          <div className="section">
            <div className="section-title">Injury Risk &amp; Consistency</div>
            {player.injury_risk_score !== null && player.injury_risk_score !== undefined ? (
              <div className="stat-row injury-risk">
                <span className="label">Career Injury Risk</span>
                <div className="risk-bar">
                  <div className={`risk-fill risk-${player.injury_risk_slug}`} style={{ width: `${player.injury_risk_score}%` }} />
                </div>
                <span className="risk-value">{player.injury_risk_label}</span>
              </div>
            ) : (
              <div className="small-text">
                Career Injury Risk: no real 2015-2025 NFL history to compute a career miss rate from.
              </div>
            )}
            {player.consistency_score !== null && player.consistency_score !== undefined ? (
              <div className="stat-row consistency">
                <span className="label">Consistency</span>
                <div className="consistency-bar">
                  <div className={`consistency-fill consistency-${player.consistency_slug}`} style={{ width: `${player.consistency_score}%` }} />
                </div>
                <span className="consistency-value">{player.consistency_label}</span>
              </div>
            ) : (
              <div className="small-text">
                Consistency: fewer than 8 real career weekly PPR values to compute week-to-week variance from.
              </div>
            )}
          </div>

          {player.recent_form && player.recent_form.length > 0 && (
            <div className="section">
              <div className="section-title">Recent Form (last {player.recent_form.length} real weeks)</div>
              <div className="trend">
                {player.recent_form.map((ppr, i) => (
                  <span key={i} className="week-ppr">{ppr.toFixed(1)}</span>
                ))}
                {player.recent_form.length > 1 && (
                  <span className="trend-indicator">
                    {player.recent_form[player.recent_form.length - 1] > player.recent_form[0]
                      ? '↑ trending up'
                      : player.recent_form[player.recent_form.length - 1] < player.recent_form[0]
                      ? '↓ trending down'
                      : '→ flat'}
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="collapse-hint">Click to collapse</div>
        </div>
      )}
    </div>
  );
}
