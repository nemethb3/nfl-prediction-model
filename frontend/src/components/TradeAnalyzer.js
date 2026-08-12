import React, { useState } from 'react';
import { teamName } from '../constants/teams';
// Real, static imports - both files are precomputed by real Python scripts
// (train_trade_model.py, generate_trade_scores_2026.py). This component
// never re-derives a score itself - it only looks up real, already-
// computed values, so the accuracy badge shown always matches the model
// that actually produced the displayed numbers.
import fantasyData from '../data/fantasy_rankings_2026.json';
import tradeScoresData from '../data/trade_scores_2026.json';
import multiSignalAccuracy from '../data/multi_signal_accuracy.json';
import '../styles/TradeAnalyzer.css';

const tradeScores = tradeScoresData.players;

const eligiblePlayers = fantasyData
  .map((p) => ({ id: p.id.split('_w')[0], name: p.name, position: p.position, team: p.team }))
  .filter((p) => tradeScores[p.id])
  .sort((a, b) => a.name.localeCompare(b.name));

function SignalRow({ label, value }) {
  return (
    <div className="trade-analyzer__signal-detail">
      <span>{label}:</span> {value}
    </div>
  );
}

export default function TradeAnalyzer() {
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
              <SignalRow label="Age" value={score1.current_age} />
              <SignalRow label="Trajectory" value={score1.trajectory} />
              <SignalRow label="Career miss rate" value={`${Math.round(score1.signals.injury_risk * 100)}%`} />
              <SignalRow
                label="Role trend"
                value={score1.signals.role_trend >= 0 ? 'Growing' : 'Shrinking'}
              />
              <SignalRow label="Real draft capital" value={`${Math.round(score1.signals.draft_capital * 100)}%`} />
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
              <SignalRow label="Age" value={score2.current_age} />
              <SignalRow label="Trajectory" value={score2.trajectory} />
              <SignalRow label="Career miss rate" value={`${Math.round(score2.signals.injury_risk * 100)}%`} />
              <SignalRow
                label="Role trend"
                value={score2.signals.role_trend >= 0 ? 'Growing' : 'Shrinking'}
              />
              <SignalRow label="Real draft capital" value={`${Math.round(score2.signals.draft_capital * 100)}%`} />
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
    </div>
  );
}
