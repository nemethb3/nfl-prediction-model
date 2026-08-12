import React, { useState, useEffect } from 'react';
import '../styles/FantasyRankings.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import { teamName, teamColor, teamSecondaryColor, readableTextColor } from '../constants/teams';
import SeasonDataUnavailable from './SeasonDataUnavailable';

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

export default function FantasyRankings() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const fantasyData = seasonData.fantasy;
  // Real data exists for 2026 (Week 1 preseason projections), unlike the
  // other sections gated by the blanket `hasResults` flag - checking for
  // real fantasy data specifically, not the season-wide flag, is what
  // actually determines whether this section has anything to show.
  const isPreseason = !hasResults;

  const weeks = fantasyData ? [...new Set(fantasyData.map((p) => p.week))].sort((a, b) => a - b) : [];
  const [selectedWeek, setSelectedWeek] = useState(weeks[0]);
  const [selectedPosition, setSelectedPosition] = useState('RB');
  const [expandedPlayerId, setExpandedPlayerId] = useState(null);

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
            />
          ))
        )}
      </div>

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

function PlayerCard({ player, isExpanded, onToggle, isPreseason }) {
  const borderColor = teamColor(player.team);
  const isStatic = player.projection_type === 'season_static_per_game_avg';
  const isPriorSeasonFallback = player.projection_type === 'prior_season_rate_fallback';

  return (
    <div
      className={`player-card ${isExpanded ? 'player-card-open' : ''}`}
      style={{ borderColor }}
      onClick={onToggle}
      role="button"
      tabIndex={0}
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
          {player.projected_ppr.toFixed(1)} PPR{isStatic ? ' (season avg)' : ''}
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
              {player.projected_ppr.toFixed(1)} projected PPR points
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
                  <span className="value projected">{player.projected_ppr.toFixed(1)}</span>
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
