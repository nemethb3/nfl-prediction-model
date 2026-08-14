import React, { useState, useEffect } from 'react';
import '../styles/FantasyRankings.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import { teamName, teamColor, teamSecondaryColor, readableTextColor } from '../constants/teams';
import SeasonDataUnavailable from './SeasonDataUnavailable';
import { useKeyboardToggle } from '../hooks/useKeyboardToggle';

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
          <p className="small-text">{rookieScores.methodology_note}</p>
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
              Real {selectedSeason} preseason data only - each returning player&apos;s own real
              full-2025-season per-game rate (WR: real EPA-based static projection), no real 2026
              games played yet. Real results appear once the season starts.
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

      {selectedPosition === 'WR' && (
        <p className="wr-static-note">
          {isPreseason ? (
            <>
              WR projections use a real static EPA-based season-long figure for every {selectedSeason}
              player shown - there&apos;s no real prior {selectedSeason} week yet to switch to a
              trailing average, unlike a season already in progress.
            </>
          ) : (
            <>
              WR projections use a real trailing up-to-4-week actual-PPR average once a player has a
              real prior {selectedSeason} week - a player&apos;s first real {selectedSeason} appearance
              (usually week 1) falls back to a static season-long figure instead. Each card&apos;s
              &quot;Methodology&quot; section shows which real method produced that specific projection.
            </>
          )}
        </p>
      )}

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
              isPreseason={isPreseason}
              props={propsById ? propsById.get(player.id) : null}
              breakoutAlert={breakoutById ? breakoutById.get(player.id) : null}
            />
          ))
        )}
      </div>

      {seasonData.rookieScores && <RookieSection rookieScores={seasonData.rookieScores} />}

      <div className="disclaimer">
        <p>
          {isPreseason ? (
            <>
              Real {selectedSeason} Week 1 preseason projections, no fabrication. RB/QB/TE:
              each returning player&apos;s own real full-2025-season per-game rate, run through
              this project&apos;s real, unmodified PPR formula - the same real convention already
              used for a completed season&apos;s own Week 1 (fall back to the real prior season&apos;s
              rate), just applied one real year forward. WR: the same real static EPA-based
              formula as a completed season, calibrated on real 2025 outcomes and applied to real{' '}
              {selectedSeason} preseason inputs. Real 2026 team assignments come from this
              project&apos;s own real, {selectedSeason}-specific roster files, not a stale prior-year
              team (the exact class of bug the 2026-07-30 audit caught and fixed). Opponent defense
              rank, recent form, injury status, actual PPR, and accuracy tier are all real nulls -
              none of them have any real data to compute from before a season starts. Players with
              real 2025 data are all wired in as this project&apos;s real 2026 preseason roster
              files; a real, newly-drafted rookie with no prior real season would be excluded here
              (a real, disclosed gap) rather than assigned an invented rate.
            </>
          ) : (
            <>
              Real {selectedSeason} backtest data. RB/QB/TE use validated volume-only
              trailing-window formulas, updated every real week (RB: real corr
              +0.651, QB/TE: beat their combined-formula baselines - see
              PROGRESS.md). WR uses a real trailing up-to-4-week actual-PPR
              average (real corr +0.4416, winner of a real leave-one-out
              backtest across 23 variations - see src/wr_dynamic_backtest.py)
              once a player has a real prior {selectedSeason} week; a player&apos;s first real
              {selectedSeason} appearance falls back to the original validated static
              EPA+volume season projection instead - each card&apos;s own
              &quot;Methodology&quot; section shows which real method produced that
              specific row. Opponent defense rank vs. position and injury
              status (real
              nflreadpy weekly reports) are real, but absent for week 1 where no
              real trailing data or report exists yet - not fabricated with a
              fallback. Recent form is the real trailing 4-week actual PPR (also
              absent in week 1). Actual PPR and projection accuracy (for
              completed games) use real results, with accuracy color tiers set
              from empirical per-position terciles, not a fixed threshold. No
              confidence interval is shown - that still needs real per-player
              calibration work, not a lookup. See the project&apos;s Data Gaps
              Report for detail.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function PlayerCard({ player, isExpanded, onToggle, isPreseason, props, breakoutAlert }) {
  const borderColor = teamColor(player.team);
  const isStatic = player.projection_type === 'season_static_per_game_avg';
  const isPriorSeasonFallback = player.projection_type === 'prior_season_rate_fallback';
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
        </div>

        <div className="ppr">
          {player.projected_ppr != null ? player.projected_ppr.toFixed(1) : '--'} PPR{isStatic ? ' (season avg)' : ''}
        </div>

        {player.opponent_defense_rank_vs_position !== null && player.opponent_defense_rank_vs_position !== undefined && (
          <div className="def-rank" title={`Opponent (${player.opponent}) real trailing defense rank vs ${player.position}`}>
            vs #{player.opponent_defense_rank_vs_position}
          </div>
        )}

        {player.injury_status && player.injury_status !== 'healthy' && (
          <div className="injury-badge" title={player.injury_status_raw || player.injury_status}>
            {INJURY_EMOJI[player.injury_status] || '⚪'}
          </div>
        )}

        {breakoutAlert && (
          <div className="breakout-badge" title={breakoutAlert.recommendation}>
            <span className="breakout-icon">⚡</span>
            <span className="breakout-text">Breakout</span>
            <span className="confidence">{Math.round(breakoutAlert.confidence * 100)}%</span>
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
              {isStatic
                ? ' (static season-long projection, per-game average)'
                : isPriorSeasonFallback
                ? " (this player's own real prior-season per-game rate, no real trailing data yet this season)"
                : ' (this week, real trailing data)'}
            </div>
            <div className="small-text">Source: {player.source}</div>
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
                  <span className="small-text">
                    {' '}(empirical tercile of real {player.position} projection errors this season, not a fixed threshold)
                  </span>
                </div>
              )}
              {ACCURACY_TERCILE_RANGES[player.position] && (
                <div className="accuracy-ranges">
                  <span className="label">Real {player.position} green/yellow boundary this season:</span>
                  <span className="ranges">{ACCURACY_TERCILE_RANGES[player.position]}</span>
                </div>
              )}
              {isStatic && (
                <div className="small-text">
                  {isPreseason
                    ? "No real trailing form exists yet this early in the season, so this row uses the static season-long average instead of the usual trailing actual-PPR projection - large differences here often reflect real game-to-game variance, not model error."
                    : "This is the player's first real appearance this season, so no trailing form exists yet - this row uses the static season-long average instead of the usual trailing actual-PPR projection, so large differences here often reflect real game-to-game variance, not model error."}
                </div>
              )}
            </div>
          )}

          {player.opponent && (
            <div className="section">
              <div className="section-title">Matchup</div>
              <div>vs {teamName(player.opponent)}</div>
              {player.opponent_defense_rank_vs_position !== null && player.opponent_defense_rank_vs_position !== undefined ? (
                <div className="small-text">
                  Real trailing defense rank vs {player.position}: #{player.opponent_defense_rank_vs_position} of 32
                  (1 = stingiest, through the prior real week's play-by-play)
                </div>
              ) : (
                <div className="small-text">No real trailing defense data yet (week 1 - not fabricated with a fallback)</div>
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
              <div className="small-text">
                Real, position-specific linear regression (5-fold cross-validated), conditioned on
                this player&apos;s own real 2015-2025 career per-game average and the real opponent
                Defensive Elo ({props.opponent_d_elo}) from this project&apos;s O/D Elo split - not a
                fixed per-position adjustment. Volume/yardage props are meaningfully predictive (real
                out-of-fold R² 0.16-0.31). TD props show a real probability of 1+ TD instead of an
                expected count (logistic regression, real AUC 0.60-0.70, GroupKFold by player) - the
                old linear expected-count approach was replaced after a real R² 0.037-0.139 confirmed
                it wasn&apos;t meaningfully predictive or actionable for a binary, rare event. See
                player_props_models.json / td_props_logistic_models.json for the full real validation.
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
              <div className="small-text">
                {breakoutAlert.recommendation}. Confidence is out of 3 real, currently-available
                signals (weak defense, usage trend, recent form) - a 4th real signal slot
                (competition/injury at the position) exists in this data but is always inactive right
                now: {breakoutAlert.signals.competition_reduced.disclosure} Real, empirically-derived
                thresholds throughout (top/bottom tercile of each signal&apos;s own real league-wide
                distribution), not asserted cutoffs - about 26% of all real player-weeks meet the 2-of-3
                bar, the honest combinatorial result of that design, not a rare/exclusive signal.
              </div>
            </div>
          )}

          {player.injury_status && (
            <div className="section">
              <div className="section-title">Injury Status</div>
              <div>
                {INJURY_EMOJI[player.injury_status] || '⚪'} {player.injury_status}
                {player.injury_status_raw ? ` (${player.injury_status_raw})` : ''}
              </div>
              <div className="small-text">
                {isPreseason
                  ? "Defaulted to healthy - real nflreadpy injury data doesn't cover this season at all yet (verified directly), not because this specific player was checked and cleared."
                  : "From real nflreadpy weekly injury reports - no report entry means the player wasn't listed that week (real absence, not an assumption)."}
              </div>
            </div>
          )}

          <div className="section">
            <div className="section-title">Injury Risk &amp; Consistency</div>
            {player.injury_risk_score !== null && player.injury_risk_score !== undefined ? (
              <div className="stat-row injury-risk">
                <span className="label">Injury Risk</span>
                <div className="risk-bar">
                  <div className={`risk-fill risk-${player.injury_risk_slug}`} style={{ width: `${player.injury_risk_score}%` }} />
                </div>
                <span className="risk-value">{player.injury_risk_label}</span>
              </div>
            ) : (
              <div className="small-text">
                Injury risk: no real 2015-2025 NFL history to compute a career miss rate from.
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
                Consistency: fewer than 8 real 2025 weekly PPR values to compute week-to-week variance from.
              </div>
            )}
            <div className="small-text">
              Injury Risk: real career miss rate (2015-2025, &gt;=4-game seasons only) blended with
              real, empirically-computed position/age risk multipliers and real recent (last 8
              real weeks) miss rate - not asserted constants. Consistency: 100 minus real
              week-to-week coefficient of variation in 2025 actual PPR (higher = more predictable).
            </div>
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
              <div className="small-text">Real actual PPR, most recent real weeks strictly before this one.</div>
            </div>
          )}

          <div className="section">
            <div className="section-title">Methodology</div>
            <div className="small-text">
              {isPriorSeasonFallback
                ? "No real trailing 2026 data exists yet - this real formula uses this player's own real full-2025-season per-game rate (rushing/receiving/passing touches, yards, TDs) run through the same real formula used once the season is underway, the same real fallback convention already used for a completed season's own real Week 1."
                : isStatic && player.position === 'WR' && isPreseason
                ? 'Static, real season-long EPA x volume projection, calibrated on real 2025 outcomes and applied to real 2026 preseason inputs - every WR row uses this real static method before the season starts, since no real trailing weeks exist yet.'
                : isStatic && player.position === 'WR'
                ? "Static, real season-long EPA x volume projection (the original validated WR formula), calibrated to real PPR units and divided by real expected games played - used because this is this player's first real appearance this season, before any real trailing form exists."
                : isStatic
                ? 'Static, real season-long EPA x volume projection, calibrated to real PPR units and divided by real expected games played - does not update within the season.'
                : player.position === 'WR'
                ? 'Real trailing up-to-4-week actual-PPR average - the real winner of a leave-one-out backtest across 23 variations (corr +0.4416 vs. a leak-free static baseline\'s +0.3957 - see src/wr_dynamic_backtest.py), not a volume formula like RB/QB/TE.'
                : "Volume-based formula (rushing/receiving/passing touches, yards, TDs) using each player's own real trailing weeks of this season's data (week 1 falls back to the real prior season's per-game rates)."}
              {' '}No confidence interval is factored in or shown - real per-player calibration for that doesn't exist yet.
            </div>
          </div>

          <div className="collapse-hint">Click to collapse</div>
        </div>
      )}
    </div>
  );
}
