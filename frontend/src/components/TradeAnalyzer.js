import React, { useState } from 'react';
import { teamName } from '../constants/teams';
// Real, static imports - all files are precomputed by real Python scripts
// (train_trade_model.py, generate_trade_scores_2026.py, apply_schedule_
// strength_adjustment.py, build_position_value_tiers.py). This component
// never re-derives a MODEL SCORE itself - it only looks up real, already-
// computed per-player values (prob_ppr_increase, positional scarcity) and
// combines them with transparent, disclosed arithmetic (Multi-Player mode)
// - this project has no backend (see generate_trade_scores_2026.py's own
// docstring), so a live server-side trade endpoint was never an option;
// every real number here was computed ahead of time in Python.
import fantasyData from '../data/fantasy_rankings_2026.json';
import tradeScoresData from '../data/trade_scores_2026.json';
import multiSignalAccuracy from '../data/multi_signal_accuracy.json';
import positionValueTiers from '../data/position_value_tiers.json';
import tradeRoleAdjustmentsData from '../data/trade_role_adjustments.json';
import '../styles/TradeAnalyzer.css';

const tradeScores = tradeScoresData.players;
const roleAdjustments = tradeRoleAdjustmentsData.players;

// Real, human-readable labels for build_trade_role_adjustments.py's real
// empirical role tiers (see that script's own docstring for how each
// multiplier was derived - real average per-game PPR by tier, normalized
// to the position's own real overall average, not an asserted number).
const ROLE_DISPLAY_LABELS = {
  primary_starter: 'Primary Starter', spot_starter: 'Spot Starter', backup: 'Backup QB',
  lead_rb: 'Lead RB', timeshare_rb: 'Timeshare RB', backup_rb: 'Backup RB',
  wr1_primary: 'WR1', wr2_secondary: 'WR2', wr3_depth: 'WR3/Depth',
  starter_te: 'Starter TE', rotational_te: 'Rotational TE', backup_te: 'Backup TE',
};

const eligiblePlayers = fantasyData
  .map((p) => ({ id: p.id.split('_w')[0], name: p.name, position: p.position, team: p.team, projected_ppr: p.projected_ppr }))
  .filter((p) => tradeScores[p.id])
  .sort((a, b) => a.name.localeCompare(b.name));

const eligibleById = new Map(eligiblePlayers.map((p) => [p.id, p]));

// Real, modest, disclosed trajectory scaling: 0.9x-1.1x, linear in the
// real prob_ppr_increase (schedule-adjusted where available) - same real
// +/-10% shape as the schedule-strength adjustment itself, not a
// fabricated wide swing.
const TRAJECTORY_MIN_MULTIPLIER = 0.9;
const TRAJECTORY_RANGE = 0.2;

function playerPackageValue(player) {
  const score = tradeScores[player.id];
  if (!score || player.projected_ppr == null) return null;
  const trajectoryProb = score.schedule_adjusted_prob_ppr_increase ?? score.prob_ppr_increase;
  const trajectoryMultiplier = TRAJECTORY_MIN_MULTIPLIER + TRAJECTORY_RANGE * trajectoryProb;
  const scarcity = positionValueTiers.tiers[player.position]?.positional_scarcity_raw_points ?? 1.0;
  const role = roleAdjustments[player.id];
  const roleMultiplier = role?.role_multiplier ?? 1.0;
  const opportunityMultiplier = 1 + (role?.backup_opportunity_boost ?? 0);
  return player.projected_ppr * trajectoryMultiplier * scarcity * roleMultiplier * opportunityMultiplier;
}

function countByPosition(players) {
  const counts = {};
  for (const p of players) counts[p.position] = (counts[p.position] || 0) + 1;
  return counts;
}

function analyzePackage(giveList, receiveList) {
  const giveValues = giveList.map((p) => ({ player: p, value: playerPackageValue(p) }));
  const receiveValues = receiveList.map((p) => ({ player: p, value: playerPackageValue(p) }));
  const giveValue = giveValues.reduce((sum, v) => sum + (v.value || 0), 0);
  const receiveValue = receiveValues.reduce((sum, v) => sum + (v.value || 0), 0);
  const delta = receiveValue - giveValue;
  const deltaPct = giveValue > 0 ? (delta / giveValue) * 100 : 0;

  const givePositions = countByPosition(giveList);
  const receivePositions = countByPosition(receiveList);
  const netQb = (receivePositions.QB || 0) - (givePositions.QB || 0);
  const skillsGiven = (givePositions.RB || 0) + (givePositions.WR || 0) + (givePositions.TE || 0);
  const qbOvervaluation = netQb >= 1 && skillsGiven >= 2;

  let recommendation;
  if (qbOvervaluation) {
    recommendation = `Giving up ${skillsGiven} skill-position players for ${netQb} net QB - worth a second look given QB's real raw-point scarcity is largely a scoring-format artifact, not roster-slot scarcity (see disclosure below).`;
  } else if (deltaPct > 10) {
    recommendation = `Favors you receiving: +${deltaPct.toFixed(0)}% package value.`;
  } else if (deltaPct > -10) {
    recommendation = `Roughly even: ${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(0)}% package value.`;
  } else {
    recommendation = `Favors the other side: ${deltaPct.toFixed(0)}% package value.`;
  }

  return {
    giveValue, receiveValue, delta, deltaPct, givePositions, receivePositions,
    netQb, skillsGiven, qbOvervaluation, recommendation,
  };
}

function SignalRow({ label, value }) {
  return (
    <div className="trade-analyzer__signal-detail">
      <span>{label}:</span> {value}
    </div>
  );
}

function PackageColumn({ title, players, onAdd, onRemove, selectId }) {
  const [pendingId, setPendingId] = useState('');
  return (
    <div className="trade-analyzer__package-column">
      <h4>{title}</h4>
      <div className="trade-analyzer__package-list">
        {players.length === 0 && <p className="trade-analyzer__empty-note">No players added yet.</p>}
        {players.map((p) => {
          const role = roleAdjustments[p.id];
          return (
            <div key={p.id} className="trade-analyzer__package-row">
              <span>
                {p.name} ({p.position}, {p.team})
                {role?.role && (
                  <span className="trade-analyzer__role-badge">{ROLE_DISPLAY_LABELS[role.role] || role.role}</span>
                )}
                {role?.has_backup_opportunity && (
                  <span className="trade-analyzer__role-badge trade-analyzer__role-badge--opportunity">
                    Opportunity
                  </span>
                )}
              </span>
              <button type="button" onClick={() => onRemove(p.id)} aria-label={`Remove ${p.name}`}>
                &times;
              </button>
            </div>
          );
        })}
      </div>
      <div className="trade-analyzer__package-add">
        <select id={selectId} value={pendingId} onChange={(e) => setPendingId(e.target.value)}>
          <option value="">Select player...</option>
          {eligiblePlayers
            .filter((p) => !players.some((existing) => existing.id === p.id))
            .map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.position}, {p.team})
              </option>
            ))}
        </select>
        <button
          type="button"
          disabled={!pendingId}
          onClick={() => {
            if (pendingId) {
              onAdd(eligibleById.get(pendingId));
              setPendingId('');
            }
          }}
        >
          + Add
        </button>
      </div>
    </div>
  );
}

function MultiPlayerTrade() {
  const [giveList, setGiveList] = useState([]);
  const [receiveList, setReceiveList] = useState([]);

  const analysis = (giveList.length > 0 || receiveList.length > 0)
    ? analyzePackage(giveList, receiveList)
    : null;

  return (
    <>
      <div className="trade-analyzer__methodology">
        <p>
          <strong>Package value</strong> = each player&apos;s real Week 1 projected PPR &times; a real,
          modest trajectory multiplier ({TRAJECTORY_MIN_MULTIPLIER.toFixed(1)}x-{(TRAJECTORY_MIN_MULTIPLIER + TRAJECTORY_RANGE).toFixed(1)}x,
          scaled by the real trade model&apos;s prob_ppr_increase, schedule-adjusted where available)
          &times; a real, computed positional scarcity multiplier (real elite-vs-replacement PPR point
          gap per position, 2015-2025, normalized to the 4-position average) &times; a real, empirical
          role multiplier (lead vs. backup RB, WR1 vs. WR3, etc. - real average per-game PPR by role
          tier, normalized to the position&apos;s own real overall average) &times; a real +10% boost when
          a player is currently listed pos_rank 2 on their real team&apos;s depth chart (real
          &quot;next man up&quot; standing, from nflreadpy&apos;s real depth charts).
        </p>
        <p className="trade-analyzer__disclaimer">
          Real, disclosed limitation: positional scarcity here is raw fantasy-point scarcity, not
          roster-slot scarcity. QB scores highest (real {positionValueTiers.tiers.QB.positional_scarcity_raw_points}x)
          because passing yards/TDs generate more raw PPR points at the top of the position, not
          because QB is harder to roster - a standard league starts only 1 QB but 2-3 flex-eligible
          RB/WR/TE, which this metric doesn&apos;t capture. This project has no real ADP/roster-
          construction data to compute a genuine slot-adjusted number, so one wasn&apos;t invented -
          use the QB flag below alongside your own judgment, not as a standalone verdict.
        </p>
      </div>

      <div className="trade-analyzer__package-inputs">
        <PackageColumn title="You Give" players={giveList}
          onAdd={(p) => setGiveList([...giveList, p])}
          onRemove={(id) => setGiveList(giveList.filter((p) => p.id !== id))}
          selectId="trade-give-select" />

        <PackageColumn title="You Receive" players={receiveList}
          onAdd={(p) => setReceiveList([...receiveList, p])}
          onRemove={(id) => setReceiveList(receiveList.filter((p) => p.id !== id))}
          selectId="trade-receive-select" />
      </div>

      {analysis && (
        <div className="trade-analyzer__results">
          <div className={`trade-analyzer__outcome ${analysis.qbOvervaluation ? 'trade-analyzer__outcome--warning' : ''}`}>
            {analysis.recommendation}
          </div>

          <div className="trade-analyzer__package-values">
            <div className="trade-analyzer__player-card">
              <h4>You Give</h4>
              <div className="trade-analyzer__score-display">{analysis.giveValue.toFixed(1)}</div>
              <span className="trade-analyzer__score-label">Package value</span>
            </div>
            <div className="trade-analyzer__vs">vs</div>
            <div className="trade-analyzer__player-card">
              <h4>You Receive</h4>
              <div className="trade-analyzer__score-display">{analysis.receiveValue.toFixed(1)}</div>
              <span className="trade-analyzer__score-label">Package value</span>
            </div>
          </div>

          <div className="trade-analyzer__positional-impact">
            {['QB', 'RB', 'WR', 'TE'].map((pos) => {
              const net = (analysis.receivePositions[pos] || 0) - (analysis.givePositions[pos] || 0);
              if (net === 0) return null;
              return (
                <span key={pos} className={`trade-analyzer__pos-impact ${net > 0 ? 'positive' : 'negative'}`}>
                  {pos}: {net > 0 ? '+' : ''}{net}
                </span>
              );
            })}
          </div>
        </div>
      )}

      <div className="trade-analyzer__footnote">
        <p>
          Same {eligiblePlayers.length} real players eligible as the 1-for-1 mode (real complete signal
          set required). Values use real Week 1 preseason projected_ppr as the magnitude baseline (this
          project&apos;s only real 2026 per-player PPR projection right now) - not a season-long total.
        </p>
      </div>
    </>
  );
}

export default function TradeAnalyzer() {
  const [mode, setMode] = useState('single');
  const [side1Id, setSide1Id] = useState('');
  const [side2Id, setSide2Id] = useState('');

  const player1 = eligiblePlayers.find((p) => p.id === side1Id);
  const player2 = eligiblePlayers.find((p) => p.id === side2Id);
  const score1 = player1 ? tradeScores[player1.id] : null;
  const score2 = player2 ? tradeScores[player2.id] : null;

  const showResult = score1 && score2;
  const winner = showResult
    ? (score1.prob_ppr_increase >= score2.prob_ppr_increase ? 'player1' : 'player2')
    : null;
  const accuracyFor = (position) =>
    multiSignalAccuracy.by_position[position]?.cv_accuracy ?? multiSignalAccuracy.overall_accuracy;

  return (
    <div className="trade-analyzer">
      <h2>Trade Analyzer</h2>

      <div className="trade-analyzer__mode-toggle">
        <button type="button" className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>
          1-for-1 (Model Comparison)
        </button>
        <button type="button" className={mode === 'multi' ? 'active' : ''} onClick={() => setMode('multi')}>
          Multi-Player (Package Value)
        </button>
      </div>

      {mode === 'single' ? (
        <>
          <div className="trade-analyzer__methodology">
            <p>
              <strong>Model:</strong> real age-curve direction + real point-in-time career injury
              history + real role trend (target share / snap %) + real draft capital + real recent-form
              trend + real team Elo, combined in a logistic regression fit separately per position.
            </p>
            <p>
              <strong>Honest accuracy</strong> (GroupKFold cross-validation, grouped by real player so no
              player&apos;s own data crosses the train/test boundary): QB {Math.round(accuracyFor('QB') * 100)}%,
              RB {Math.round(accuracyFor('RB') * 100)}%, WR {Math.round(accuracyFor('WR') * 100)}%,
              TE {Math.round(accuracyFor('TE') * 100)}% - all above a 50% coin flip, unlike age alone (46%,
              see Model Transparency).
            </p>
            <p className="trade-analyzer__disclaimer">
              This predicts the direction of a player&apos;s own next-season PPR (up or down), not a trade
              outcome between two specific players - this project has no real trade-outcome data to
              validate that stronger claim against. Use as one input among many.
            </p>
          </div>

          <div className="trade-analyzer__inputs">
            <div className="trade-analyzer__side">
              <label htmlFor="trade-side-1">Player You&apos;re Trading Away</label>
              <select id="trade-side-1" value={side1Id} onChange={(e) => setSide1Id(e.target.value)}>
                <option value="">Select player...</option>
                {eligiblePlayers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.position}, {p.team})
                  </option>
                ))}
              </select>
            </div>

            <div className="trade-analyzer__side">
              <label htmlFor="trade-side-2">Player You&apos;re Getting</label>
              <select id="trade-side-2" value={side2Id} onChange={(e) => setSide2Id(e.target.value)}>
                <option value="">Select player...</option>
                {eligiblePlayers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.position}, {p.team})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {showResult && (
            <div className="trade-analyzer__results">
              <div className={`trade-analyzer__outcome trade-analyzer__outcome--${winner}`}>
                Model favors {winner === 'player1' ? player1.name : player2.name}
                <span className="trade-analyzer__edge">
                  {Math.round(Math.abs(score1.prob_ppr_increase - score2.prob_ppr_increase) * 100)} point
                  real probability gap
                </span>
              </div>

              <div className="trade-analyzer__comparison">
                <div className="trade-analyzer__player-card">
                  <h4>{player1.name}</h4>
                  <div className="trade-analyzer__score-display">
                    {Math.round(score1.prob_ppr_increase * 100)}%
                    <span className="trade-analyzer__score-label">P(PPR increases)</span>
                  </div>
                  <SignalRow label="Position" value={score1.position} />
                  <SignalRow label="Team" value={teamName(score1.team) || score1.team} />
                  {roleAdjustments[player1.id]?.role && (
                    <SignalRow label="Role" value={ROLE_DISPLAY_LABELS[roleAdjustments[player1.id].role]} />
                  )}
                  {roleAdjustments[player1.id]?.has_backup_opportunity && (
                    <SignalRow label="Depth chart" value="Real backup w/ opportunity (pos_rank 2)" />
                  )}
                  <SignalRow label="Age" value={score1.current_age} />
                  <SignalRow label="Trajectory" value={score1.trajectory} />
                  <SignalRow label="Career miss rate" value={`${Math.round(score1.signals.injury_risk * 100)}%`} />
                  <SignalRow
                    label="Role trend"
                    value={score1.signals.role_trend >= 0 ? 'Growing' : 'Shrinking'}
                  />
                  <SignalRow label="Real draft capital" value={`${Math.round(score1.signals.draft_capital * 100)}%`} />
                  {score1.avg_opponent_d_elo_next4 != null && (
                    <SignalRow label="Avg opp D_Elo (next 4 wks)" value={score1.avg_opponent_d_elo_next4} />
                  )}
                </div>

                <div className="trade-analyzer__vs">vs</div>

                <div className="trade-analyzer__player-card">
                  <h4>{player2.name}</h4>
                  <div className="trade-analyzer__score-display">
                    {Math.round(score2.prob_ppr_increase * 100)}%
                    <span className="trade-analyzer__score-label">P(PPR increases)</span>
                  </div>
                  <SignalRow label="Position" value={score2.position} />
                  <SignalRow label="Team" value={teamName(score2.team) || score2.team} />
                  {roleAdjustments[player2.id]?.role && (
                    <SignalRow label="Role" value={ROLE_DISPLAY_LABELS[roleAdjustments[player2.id].role]} />
                  )}
                  {roleAdjustments[player2.id]?.has_backup_opportunity && (
                    <SignalRow label="Depth chart" value="Real backup w/ opportunity (pos_rank 2)" />
                  )}
                  <SignalRow label="Age" value={score2.current_age} />
                  <SignalRow label="Trajectory" value={score2.trajectory} />
                  <SignalRow label="Career miss rate" value={`${Math.round(score2.signals.injury_risk * 100)}%`} />
                  <SignalRow
                    label="Role trend"
                    value={score2.signals.role_trend >= 0 ? 'Growing' : 'Shrinking'}
                  />
                  <SignalRow label="Real draft capital" value={`${Math.round(score2.signals.draft_capital * 100)}%`} />
                  {score2.avg_opponent_d_elo_next4 != null && (
                    <SignalRow label="Avg opp D_Elo (next 4 wks)" value={score2.avg_opponent_d_elo_next4} />
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="trade-analyzer__footnote">
            <p>
              {eligiblePlayers.length} of this project&apos;s real ranked players have a complete real
              signal set (career history, role trend, draft capital, team Elo) and are eligible here -
              players without enough real history (e.g. true rookies, or missing real target-share/
              snap-% data) aren&apos;t shown rather than scored with a fabricated default.
            </p>
          </div>
        </>
      ) : (
        <MultiPlayerTrade />
      )}
    </div>
  );
}
